import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from controller import decide
from harness import ROOT, judge, load_thresholds, read_json
from mgn_adapter import AGGREGATION, DATA_ORIGIN, STEP_COLUMNS, SUMMARY_COLUMNS, run

PROVISIONAL = ROOT / "thresholds_mgn_provisional.json"
DEMO = ROOT / "thresholds_demo.json"
README = {
    "author": "tester",
    "date": "2026-11-09",
    "model": "mgn_test.pt",
    "model_sha256_16": "abcdef0123456789",
    "splits": {"val": "r0300-r0302 (tuned)", "test": "r0303-r0305"},
    "unit": "um",
    "chamfer_definition": "point-based sum of two one-way means",
    "limitations": ["교차 있는 레시피 남아 있음, Chamfer만으로 합격 판단 금지"],
}


def series(i, n):
    # one-step: 0.01*step*(i+1); rollout grows with step
    return [
        dict(step=s, onestep=0.01 * s * (i + 1), rollout=0.1 * s + 0.01 * i)
        for s in range(1, n + 1)
    ]


def build(folder, mutate_steps=None, mutate_summary=None, readme=None, test_scale=1.0):
    folder = Path(folder)
    folder.mkdir(parents=True)
    steps, summary = [], []
    for i in range(6):
        rid, split = f"r{300 + i:04d}", "val" if i < 3 else "test"
        scale = test_scale if split == "test" else 1.0
        rows = series(i, 3 + i % 2)
        for r in rows:
            steps.append(
                dict(
                    rid=rid,
                    split=split,
                    step=r["step"],
                    onestep_chamfer=r["onestep"] * scale,
                    rollout_chamfer=r["rollout"] * scale,
                )
            )
        summary.append(
            dict(
                rid=rid,
                split=split,
                last_step=len(rows),
                onestep_chamfer=rows[-1]["onestep"] * scale,
                rollout_chamfer=rows[-1]["rollout"] * scale,
                n_pred_points=100,
                n_gt_points=90,
            )
        )
    if mutate_steps:
        mutate_steps(steps)
    if mutate_summary:
        mutate_summary(summary)
    write(folder / "metrics_by_step.csv", STEP_COLUMNS, steps)
    write(folder / "summary_last_step.csv", SUMMARY_COLUMNS, summary)
    (folder / "README.json").write_text(
        json.dumps(readme or README, ensure_ascii=False), encoding="utf-8"
    )
    (folder / "r0300_onestep.npz").write_bytes(b"npz")
    return folder


def write(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


class MgnAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.out = self.dir / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def go(self, name="in", **kw):
        return run(build(self.dir / name, **kw), PROVISIONAL, self.out)

    def kinds(self, r):
        return {p["kind"] for p in r["inspection"]["problems"]}

    def test_clean_input_splits_validation_and_test(self):
        _, _, r = self.go()
        self.assertEqual(r["inspection"]["problems"], [])
        self.assertEqual(
            set(r["summary"]),
            {"validation/onestep", "validation/rollout", "test/onestep", "test/rollout"},
        )
        self.assertEqual(set(r["inputs"]), {"onestep", "rollout"})
        for p in r["inputs"].values():
            self.assertEqual(p["split"], "validation")
            self.assertEqual(p["recipe_count"], 3)

    def test_onestep_is_step_mean_and_rollout_is_last_step(self):
        _, _, r = self.go()
        one = r["recipes"][("validation", "onestep")]["r0300"]  # steps 1..3 -> 0.01,0.02,0.03
        self.assertAlmostEqual(one["chamfer_um"], 0.02)
        self.assertAlmostEqual(
            one["reported_last_step_onestep_chamfer_um"], 0.03
        )  # summary value != score
        roll = r["recipes"][("validation", "rollout")]["r0300"]
        self.assertAlmostEqual(roll["chamfer_um"], 0.3)
        ids = {m: p["evaluation_contract_id"] for m, p in r["inputs"].items()}
        self.assertEqual(ids, {m: AGGREGATION[m]["contract_id"] for m in ids})
        self.assertNotEqual(ids["onestep"], ids["rollout"])

    def test_test_scores_do_not_affect_comparison_or_decision(self):
        _, _, base = self.go("a")
        _, _, big = self.go("b", test_scale=100.0)
        for mode in ("onestep", "rollout"):
            self.assertEqual(base["inputs"][mode]["metrics"], big["inputs"][mode]["metrics"])
            self.assertEqual(base["comparisons"][mode]["rows"], big["comparisons"][mode]["rows"])
        self.assertGreater(
            big["summary"]["test/rollout"]["chamfer_um"]["mean"],
            base["summary"]["test/rollout"]["chamfer_um"]["mean"],
        )
        self.assertIn("not used to tune", big["summary"]["test/onestep"]["use"])

    def test_test_split_payload_is_rejected_by_controller(self):
        _, _, r = self.go()
        p = dict(r["inputs"]["onestep"], split="test", model_id="MGN/x")
        approved = dict(load_thresholds(PROVISIONAL)[("MGN", "onestep")], status="approved")
        d = decide(p, approved)
        self.assertEqual(d["action"], "HOLD")
        self.assertIn("test", d["reasons"][0])

    def test_missing_metrics_are_not_zero_filled(self):
        out, _, r = self.go()
        for p in r["inputs"].values():
            self.assertEqual(set(p["metrics"]), {"chamfer_um_recipe_mean"})
            self.assertTrue(
                {"depth_error", "iou", "invalid_geometry_count", "self_intersection"}
                <= set(p["not_provided"])
            )
            self.assertEqual(p["data_origin"], DATA_ORIGIN)
            self.assertIs(p["data_check"]["reference_verified"], False)
        rows = {x["metric"]: x["result"] for x in r["comparisons"]["onestep"]["rows"]}
        self.assertEqual(rows["invalid_geometry_count"], "미측정")

    def test_self_intersection_notice_in_report(self):
        out, text, r = self.go()
        notice = [line for line in text.splitlines() if line.startswith("!! 제공 설명")]
        self.assertEqual(len(notice), 1)
        self.assertIn("자기교차", notice[0])
        self.assertIn("Chamfer만으로 합격 판단 금지", notice[0])
        self.assertLess(text.index(notice[0]), text.index("## 점수 요약"))  # shown before any score
        self.assertIn(
            "교차 있는 레시피 남아 있음, Chamfer만으로 합격 판단 금지",
            r["inputs"]["rollout"]["known_issues"],
        )

    def test_decision_and_unverified_are_separate(self):
        out, text, r = self.go()
        for d in r["decisions"].values():
            self.assertEqual(d["action"], "HOLD")
            self.assertFalse(d["training_started"])
            self.assertFalse(d["active_model_changed"])
        self.assertTrue(r["unverified"]["items"])
        self.assertTrue(
            (out / "unverified.json").is_file() and (out / "official_decisions.json").is_file()
        )
        self.assertLess(text.index("[임시 비교]"), text.index("[최종 판정]"))
        self.assertLess(text.index("[최종 판정]"), text.index("[미확인 사항]"))
        decisions = read_json(out / "official_decisions.json")
        body = json.dumps(decisions, ensure_ascii=False)
        self.assertFalse(any(item in body for item in r["unverified"]["items"]))

    def test_duplicate_rows_detected(self):
        _, _, r = self.go(mutate_steps=lambda rows: rows.append(dict(rows[0])))
        self.assertIn("duplicate_row", self.kinds(r))
        self.assertEqual({d["action"] for d in r["decisions"].values()}, {"DATA_REVIEW"})
        _, _, r2 = self.go("b", mutate_summary=lambda rows: rows.append(dict(rows[0])))
        self.assertIn("duplicate_row", self.kinds(r2))

    def test_missing_column_detected(self):
        folder = build(self.dir / "in")
        with (folder / "metrics_by_step.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        write(
            folder / "metrics_by_step.csv",
            [c for c in STEP_COLUMNS if c != "rollout_chamfer"],
            rows,
        )
        _, _, r = run(folder, PROVISIONAL, self.out)
        self.assertIn("missing_columns", self.kinds(r))
        self.assertFalse(r["inspection"]["data_check"]["passed"])

    def test_numeric_errors_detected(self):
        for i, (col, bad) in enumerate(
            [
                ("onestep_chamfer", "nan"),
                ("onestep_chamfer", ""),
                ("rollout_chamfer", "x"),
                ("rollout_chamfer", -0.1),
                ("step", 1.5),
            ]
        ):
            with self.subTest(col=col, bad=bad):
                _, _, r = self.go(
                    f"n{i}", mutate_steps=lambda rows, c=col, b=bad: rows[1].update({c: b})
                )
                self.assertTrue(self.kinds(r) & {"bad_number", "out_of_range"})
                self.assertFalse(r["inspection"]["data_check"]["passed"])

    def test_split_label_and_range_checks(self):
        _, _, r = self.go("a", mutate_summary=lambda rows: rows[0].update(split="test"))
        self.assertTrue({"split_mismatch", "split_range_mismatch"} <= self.kinds(r))
        _, _, r2 = self.go("b", mutate_steps=lambda rows: rows[0].update(split="train"))
        self.assertIn("bad_split", self.kinds(r2))

    def test_summary_must_match_last_step(self):
        _, _, r = self.go(mutate_summary=lambda rows: rows[0].update(rollout_chamfer=9.9))
        self.assertIn("summary_mismatch", self.kinds(r))

    def test_step_gap_and_id_mismatch(self):
        _, _, r = self.go("a", mutate_steps=lambda rows: rows[0].update(step=7))
        self.assertIn("step_gap_or_duplicate", self.kinds(r))
        _, _, r2 = self.go("b", mutate_summary=lambda rows: rows.pop())
        self.assertIn("id_mismatch", self.kinds(r2))

    def test_demo_and_fno_policies_cannot_judge_mgn_scores(self):
        _, _, r = self.go()
        for mode, p in r["inputs"].items():
            _, d = judge(p, load_thresholds(DEMO))
            self.assertEqual(d["action"], "HOLD")
            approved_demo = dict(load_thresholds(DEMO)[("MGN", mode)], status="approved")
            self.assertEqual(decide(dict(p, model_id="MGN/x"), approved_demo)["action"], "HOLD")
            approved_fno = dict(
                load_thresholds(ROOT / "thresholds_fno_provisional.json")[("FNO", mode)],
                status="approved",
            )
            self.assertEqual(decide(dict(p, model_id="MGN/x"), approved_fno)["action"], "HOLD")

    def test_provisional_file_marked_and_contracts_match(self):
        data = read_json(PROVISIONAL)
        self.assertIn("임시 비교용", data["_notice"])
        for p in data["policies"]:
            self.assertEqual((p["model_family"], p["status"]), ("MGN", "provisional"))
            self.assertEqual(
                p["evaluation_contract_id"], AGGREGATION[p["eval_mode"]]["contract_id"]
            )

    def test_outputs_new_folder_sources_unchanged(self):
        folder = build(self.dir / "in")
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
        out1, _, _ = run(folder, PROVISIONAL, self.out)
        out2, _, _ = run(folder, PROVISIONAL, self.out)
        self.assertNotEqual(out1, out2)
        self.assertEqual(
            before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
        )
        src = read_json(out1 / "source_files.json")["files"]
        self.assertEqual({Path(k).name: v["sha256"] for k, v in src.items()}, before)
        self.assertEqual(
            {p.name for p in out1.iterdir()},
            {
                "source_files.json",
                "readme_as_provided.json",
                "selected_rows.json",
                "inspection.json",
                "score_summary.json",
                "test_scores_record.json",
                "not_provided.json",
                "harness_input_validation_onestep.json",
                "harness_input_validation_rollout.json",
                "provisional_comparison.json",
                "official_decisions.json",
                "unverified.json",
                "report.txt",
            },
        )

    def test_saved_input_accepted_by_harness(self):
        out, _, _ = self.go()
        policy, d = judge(
            read_json(out / "harness_input_validation_rollout.json"), load_thresholds(PROVISIONAL)
        )
        self.assertEqual(policy["version"], "mgn-provisional-rollout-v1")
        self.assertEqual(d["action"], "HOLD")


if __name__ == "__main__":
    unittest.main(verbosity=2)
