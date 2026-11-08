import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fno_adapter
import mgn_adapter
import run_review
from test_fno_adapter import build as build_fno
from test_mgn_adapter import build as build_mgn


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        metrics, samples = build_fno(self.dir / "fno")
        mgn = build_mgn(self.dir / "mgn")
        self.argv = [
            "--fno-metrics",
            str(metrics),
            "--fno-samples",
            str(samples),
            "--mgn-input",
            str(mgn),
            "--output-root",
            str(self.dir / "review"),
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def go(self, *extra):
        args = run_review.parser().parse_args(self.argv + list(extra))
        return run_review.run(args, args.output_root)

    def combos(self, s):
        return {(c["model_family"], c["eval_mode"]): c for c in s["combinations"]}

    def test_four_combinations_all_hold_with_files(self):
        out, s = self.go()
        self.assertEqual(
            {p.name for p in out.iterdir()}, {"report.html", "summary.json", "fno", "mgn"}
        )
        self.assertEqual(
            [(c["model_family"], c["eval_mode"]) for c in s["combinations"]],
            [("FNO", "onestep"), ("FNO", "rollout"), ("MGN", "onestep"), ("MGN", "rollout")],
        )
        for c in s["combinations"]:
            self.assertEqual(c["status"], "ok")
            self.assertEqual(c["split"], "validation")
            self.assertEqual(c["decision"]["action"], "HOLD")
            self.assertFalse(c["decision"]["training_started"])
            self.assertFalse(c["decision"]["active_model_changed"])
            self.assertTrue(c["decision"]["remaining_blockers_even_if_approved"])
        saved = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["all_decisions"], s["all_decisions"])

    def test_adapters_are_reused_not_reimplemented(self):
        with (
            mock.patch.object(fno_adapter, "run", wraps=fno_adapter.run) as f,
            mock.patch.object(mgn_adapter, "run", wraps=mgn_adapter.run) as m,
        ):
            _, s = self.go()
        self.assertEqual((f.call_count, m.call_count), (1, 1))
        # values come straight from the adapter payloads
        out_dir = Path(s["models"]["FNO"]["adapter_output_dir"])
        payload = json.loads((out_dir / "harness_input_rollout.json").read_text(encoding="utf-8"))
        self.assertEqual(self.combos(s)[("FNO", "rollout")]["metrics"], payload["metrics"])

    def test_report_shows_required_fields(self):
        out, s = self.go()
        page = (out / "report.html").read_text(encoding="utf-8")
        for text in [
            "모델 버전",
            "레시피 값 집계",
            "Chamfer 정의",
            "임시 상한",
            "상한 초과 레시피 수",
            "보류 이유",
            "[미측정]",
            "미확인 사항",
            "fno-minseo-metrics5-onestep-stepmean-v1",
            "mgn-jinju-nov9-rollout-laststep-pointchamfer-v1",
            run_review.NO_RANKING,
        ]:
            self.assertIn(text, page)
        self.assertNotIn("<script", page)

    def test_per_recipe_exceed_ids_normalized_for_both_models(self):
        _, s = self.go()
        for key in [("FNO", "onestep"), ("MGN", "onestep")]:
            rows = {r["metric"]: r for r in self.combos(s)[key]["provisional"]["rows"]}
            ch = rows["chamfer_um_recipe_mean"]
            self.assertIsInstance(ch["recipes_over"], int)
            self.assertEqual(len(ch["recipes_over_ids"]), ch["recipes_over"])
            self.assertEqual(rows["invalid_geometry_count"]["result"], "미측정")

    def test_no_ranking_between_models(self):
        out, s = self.go()
        body = json.dumps(s, ensure_ascii=False).lower()
        for word in ["rank", "best", "winner", "우세", "순위:"]:
            self.assertNotIn(word, body)
        self.assertNotEqual(s["chamfer_definitions"]["FNO"], s["chamfer_definitions"]["MGN"])

    def test_order_is_fixed_regardless_of_scores(self):
        def inflate(rows):  # FNO scores far above MGN; order must not follow scores
            for row in rows:
                if row["방식"] == fno_adapter.GROUP:
                    row["Chamfer(um)"] = float(row["Chamfer(um)"]) * 1000

        metrics, samples = build_fno(self.dir / "fno_big", mutate_val=inflate)
        _, s = self.go("--fno-metrics", str(metrics), "--fno-samples", str(samples))
        self.assertEqual(
            [(c["model_family"], c["eval_mode"]) for c in s["combinations"]],
            [("FNO", "onestep"), ("FNO", "rollout"), ("MGN", "onestep"), ("MGN", "rollout")],
        )
        self.assertGreater(
            self.combos(s)[("FNO", "onestep")]["metrics"]["chamfer_um_recipe_mean"],
            self.combos(s)[("MGN", "rollout")]["metrics"]["chamfer_um_recipe_mean"],
        )

    def test_mgn_test_is_reference_only_and_separate(self):
        out, s = self.go()
        self.assertTrue(all(c["split"] == "validation" for c in s["combinations"]))
        ref = s["models"]["MGN"]["test_reference"]
        self.assertEqual(set(ref), {"test/onestep", "test/rollout"})
        page = (out / "report.html").read_text(encoding="utf-8")
        self.assertIn("참고 정보: MGN test (validation과 분리)", page)
        self.assertGreater(page.index("참고 정보: MGN test"), page.index("[최종 판정]"))

    def test_self_intersection_notice_shown(self):
        out, _ = self.go()
        self.assertIn(
            "교차(자기교차) 있는 레시피가 남아 있음",
            (out / "report.html").read_text(encoding="utf-8"),
        )

    def test_missing_input_marks_failure_without_past_results(self):
        out_ok, _ = self.go()  # a previous successful run exists in the same root
        out, s = self.go("--fno-metrics", str(self.dir / "nope.csv"))
        self.assertNotEqual(out_ok, out)
        self.assertEqual(s["models"]["FNO"]["status"], "failed")
        self.assertIn("입력 파일/폴더 없음", s["models"]["FNO"]["reason"])
        for mode in ("onestep", "rollout"):
            c = self.combos(s)[("FNO", mode)]
            self.assertEqual(c["status"], "failed")
            self.assertNotIn("metrics", c)
            self.assertNotIn("decision", c)
        self.assertEqual(self.combos(s)[("MGN", "rollout")]["decision"]["action"], "HOLD")
        self.assertFalse((out / "fno").exists())
        page = (out / "report.html").read_text(encoding="utf-8")
        self.assertIn("실행 실패", page)
        self.assertIn("과거 결과를", page)

    def test_adapter_exception_is_reported(self):
        with mock.patch.object(mgn_adapter, "run", side_effect=RuntimeError("broken <b>csv</b>")):
            out, s = self.go()
        self.assertEqual(s["models"]["MGN"]["status"], "failed")
        self.assertIn("RuntimeError: broken <b>csv</b>", s["models"]["MGN"]["reason"])
        page = (out / "report.html").read_text(encoding="utf-8")
        self.assertIn("broken &lt;b&gt;csv&lt;/b&gt;", page)  # escaped
        self.assertNotIn("broken <b>csv</b>", page)
        self.assertEqual(self.combos(s)[("FNO", "onestep")]["status"], "ok")

    def test_main_exits_nonzero_when_a_model_fails(self):
        argv = ["run_review.py", *self.argv, "--mgn-input", str(self.dir / "missing")]
        with mock.patch("sys.argv", argv), mock.patch("builtins.print"):
            with self.assertRaises(SystemExit) as ctx:
                run_review.main()
        self.assertEqual(ctx.exception.code, 1)

    def test_data_problem_keeps_data_review(self):
        # duplicate FNO row -> adapter routes to DATA_REVIEW; review must show it, not HOLD
        metrics, samples = build_fno(
            self.dir / "fno_bad", mutate_val=lambda rows: rows.append(dict(rows[0]))
        )
        _, s = self.go("--fno-metrics", str(metrics), "--fno-samples", str(samples))
        self.assertEqual(self.combos(s)[("FNO", "onestep")]["decision"]["action"], "DATA_REVIEW")
        self.assertFalse(self.combos(s)[("FNO", "onestep")]["data_check"]["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
