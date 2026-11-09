import copy
import csv
import hashlib
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from controller import decide
from fno_adapter import (
    AGGREGATION,
    DATA_ORIGIN,
    GROUP,
    STEP_COLUMNS,
    VAL_COLUMNS,
    provisional_comparison,
    run,
)
from harness import ROOT, judge, load_thresholds, read_json

PROVISIONAL = ROOT / "thresholds_fno_provisional.json"
DEMO = ROOT / "thresholds_demo.json"
LAST_COLUMNS = [
    "레시피",
    "설명",
    "스텝수",
    "Chamfer(um)",
    "정답깊이(um)",
    "예측깊이(um)",
    "깊이오차(%)",
    "CD_위오차(%)",
    "CD_중간오차(%)",
    "CD_바닥오차(%)",
    "IoU",
]


def step_series(i, n):
    """Per-step synthetic scores; depth error alternates sign so |mean| != mean|.|."""
    rows = []
    for s in range(1, n + 1):
        true = 1.0 * s
        err = (2.0 if s % 2 else -1.0) * (1 + i)
        rows.append(
            {
                "Chamfer(um)": 0.01 * s + 0.001 * i,
                "정답깊이(um)": true,
                "예측깊이(um)": true * (1 + err / 100),
                "깊이오차(%)": err,
                "CD_위오차(%)": -0.5 * s,
                "CD_중간오차(%)": 0.25 * s,
                "CD_바닥오차(%)": "" if s == 1 else 1.0 * s,
                "IoU": 0.999 - 0.001 * s,
            }
        )
    return rows


def aggregate(series, mode):
    out = {}
    for k in [
        "Chamfer(um)",
        "정답깊이(um)",
        "깊이오차(%)",
        "CD_위오차(%)",
        "CD_중간오차(%)",
        "CD_바닥오차(%)",
        "IoU",
    ]:
        vals = [r[k] for r in series if r[k] != ""]
        out[k] = sum(vals) / len(vals) if mode == "onestep" else series[-1][k]
    return out


def build(folder, recipes=3, steps=4, mutate_val=None, mutate_step=None):
    folder = Path(folder)
    samples = folder / "samples_main_s1"
    samples.mkdir(parents=True)
    val, step_rows, last_rows = [], [], []
    for i in range(recipes):
        rid = f"r{300 + i:04d}"
        series = step_series(i, steps)
        for kind, mode in [("1스텝", "onestep"), ("롤아웃", "rollout")]:
            agg = aggregate(series, mode)
            for group, scale in [(GROUP, 1.0), ("50런(spike4_50)", 3.0), ("베끼기", 9.0)]:
                val.append(
                    {
                        "방식": group,
                        "종류": kind,
                        "런": rid,
                        "스텝": steps,
                        "Chamfer(um)": agg["Chamfer(um)"] * scale,
                        "정답깊이(um)": agg["정답깊이(um)"],
                        "깊이오차(%)": agg["깊이오차(%)"] * scale,
                        "CD_위(um)": 7.0,
                        "CD_위 오차(%)": agg["CD_위오차(%)"],
                        "CD_중간(um)": 6.0,
                        "CD_중간 오차(%)": agg["CD_중간오차(%)"],
                        "CD_바닥(um)": 5.0,
                        "CD_바닥 오차(%)": agg["CD_바닥오차(%)"],
                        "IoU": agg["IoU"],
                    }
                )
            if i < 2:  # two sample recipes
                for s, r in enumerate(series, start=1):
                    step_rows.append(
                        {
                            "레시피": rid,
                            "방식": "one-step" if mode == "onestep" else "rollout",
                            "스텝": f"t{s:03d}",
                            **r,
                        }
                    )
        if i < 2:
            last = series[-1]
            last_rows.append({"레시피": rid, "설명": "x", "스텝수": steps, **last})
    if mutate_val:
        mutate_val(val)
    if mutate_step:
        mutate_step(step_rows)
    write(folder / "metrics5_val.csv", VAL_COLUMNS, val)
    write(
        samples / "metrics_by_step.csv",
        STEP_COLUMNS[:3]
        + [
            "Chamfer(um)",
            "정답깊이(um)",
            "예측깊이(um)",
            "깊이오차(%)",
            "CD_위오차(%)",
            "CD_중간오차(%)",
            "CD_바닥오차(%)",
            "IoU",
        ],
        step_rows,
    )
    write(samples / "summary_last_step.csv", LAST_COLUMNS, last_rows)
    (samples / "r0300_onestep.npz").write_bytes(b"binary")
    return folder / "metrics5_val.csv", samples


def write(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


class FnoAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.out = self.dir / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def go(self, **kw):
        metrics, samples = build(self.dir / "in", **kw)
        return run(metrics, samples, PROVISIONAL, self.out)

    def kinds(self, result):
        return {p["kind"] for p in result["inspection"]["problems"]}

    def test_clean_input_selects_group_and_separates_modes(self):
        out, text, r = self.go()
        self.assertEqual(r["inspection"]["problems"], [])
        self.assertEqual(set(r["inputs"]), {"onestep", "rollout"})
        for mode in ("onestep", "rollout"):
            self.assertEqual(r["inputs"][mode]["recipe_count"], 3)
            self.assertEqual(r["inputs"][mode]["eval_mode"], mode)
        raw = json.loads((out / "selected_rows.json").read_text(encoding="utf-8"))["rows"]
        self.assertTrue(all(row["방식"] == GROUP for rows in raw.values() for row in rows))
        self.assertEqual(len(raw["onestep"]), 3)
        # Other groups have 3x / 9x scores; the selected mean must not include them.
        self.assertLess(r["inputs"]["rollout"]["metrics"]["chamfer_um_recipe_mean"], 0.06)

    def test_raw_rows_preserved_as_original_strings(self):
        metrics, samples = build(self.dir / "in")
        with metrics.open(encoding="utf-8-sig", newline="") as f:
            original = [
                row for row in csv.DictReader(f) if row["방식"] == GROUP and row["종류"] == "1스텝"
            ]
        out, _, _ = run(metrics, samples, PROVISIONAL, self.out)
        saved = json.loads((out / "selected_rows.json").read_text(encoding="utf-8"))["rows"][
            "onestep"
        ]
        self.assertEqual([{k: v for k, v in row.items() if k != "line"} for row in saved], original)

    def test_aggregation_contracts_differ_and_are_verified(self):
        _, _, r = self.go()
        ids = {m: r["inputs"][m]["evaluation_contract_id"] for m in r["inputs"]}
        self.assertNotEqual(ids["onestep"], ids["rollout"])
        self.assertTrue(all(not i.startswith("demo-") for i in ids.values()))
        ev = r["inspection"]["aggregation_evidence"]
        self.assertTrue(ev and all(e["match"] for e in ev))
        self.assertEqual({e["rule"] for e in ev}, {"step_mean (non-blank steps)", "last_step"})

    def test_wrong_aggregation_is_detected(self):
        def mutate(rows):  # rollout sample now ends with a different last-step value
            last = [r for r in rows if r["방식"] == "rollout" and r["레시피"] == "r0300"][-1]
            last["Chamfer(um)"] = 9.9

        _, _, r = self.go(mutate_step=mutate)
        self.assertIn("aggregation_mismatch", self.kinds(r))
        self.assertEqual(r["decisions"]["rollout"]["action"], "DATA_REVIEW")

    def test_signed_depth_preserved_and_abs_used_for_magnitude(self):
        def mutate(rows):
            for row in rows:
                if row["방식"] == GROUP and row["종류"] == "1스텝":
                    row["깊이오차(%)"] = -2.0 if row["런"] == "r0300" else 2.0

        metrics, samples = build(self.dir / "in", mutate_val=mutate)
        _, _, r = run(metrics, samples, PROVISIONAL, self.out)
        s = r["summary"]["onestep"]
        self.assertAlmostEqual(s["depth_error_pct_signed"]["mean"], 2 / 3)
        self.assertAlmostEqual(s["abs_depth_error_pct_recipe_values"]["mean"], 2.0)
        self.assertAlmostEqual(
            r["inputs"]["onestep"]["metrics"]["abs_depth_error_pct_recipe_mean"], 2.0
        )
        self.assertAlmostEqual(
            r["inputs"]["onestep"]["reported_not_judged"]["signed_depth_error_pct_recipe_mean"],
            2 / 3,
        )

    def test_recipe_abs_mean_and_all_step_mae_are_kept_apart(self):
        _, _, r = self.go()
        ref = r["summary"]["onestep"]["sample_step_level_reference"]["recipes"]["r0300"]
        # steps: +2,-1,+2,-1 -> provided recipe value 0.5 (signed mean); all-step |%| mean 1.5
        self.assertAlmostEqual(ref["abs_of_provided_recipe_value_pct"], 0.5)
        self.assertAlmostEqual(ref["all_step_abs_depth_error_pct_mean"], 1.5)
        self.assertNotIn("depth_mae_um", r["inputs"]["onestep"]["metrics"])
        self.assertIn("unmeasured", r["summary"]["onestep"]["all_step_depth_mae"])

    def test_unmeasured_and_real_origin_are_not_faked(self):
        _, _, r = self.go()
        for mode, p in r["inputs"].items():
            self.assertNotIn("invalid_geometry_count", p["metrics"])
            self.assertIn("invalid_geometry_count", p["unmeasured"])
            self.assertEqual(p["data_origin"], DATA_ORIGIN)
            self.assertNotEqual(p["data_origin"], "synthetic_demo")
            self.assertIs(p["data_check"]["reference_verified"], False)
            self.assertIs(p["training_data_ready"], False)

    def test_blank_step_cells_not_filled_with_zero(self):
        _, _, r = self.go()
        self.assertEqual(r["inspection"]["step_file_blank_cells"], {"CD_바닥오차(%)": 4})
        ev = [
            e
            for e in r["inspection"]["aggregation_evidence"]
            if e["column"] == "CD_바닥 오차(%)" and e["mode"] == "onestep"
        ]
        self.assertTrue(all(e["blank_steps"] == 1 and e["match"] for e in ev))

    def test_report_counts_come_from_data(self):
        _, text, _ = self.go()  # synthetic: 3 recipes, 2 sample recipes
        self.assertIn("3개 레시피 전체는 미측정", text)
        self.assertIn("샘플 2개 참고값", text)
        self.assertNotIn("40개", text)

    def test_duplicate_row_routes_to_data_review(self):
        _, _, r = self.go(mutate_val=lambda rows: rows.append(dict(rows[0])))
        self.assertIn("duplicate_row", self.kinds(r))
        self.assertEqual({d["action"] for d in r["decisions"].values()}, {"DATA_REVIEW"})

    def test_missing_required_column_detected(self):
        def mutate(rows):
            for row in rows:
                row.pop("IoU")

        metrics, samples = build(self.dir / "in")
        with metrics.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        mutate(rows)
        write(metrics, [c for c in VAL_COLUMNS if c != "IoU"], rows)
        _, _, r = run(metrics, samples, PROVISIONAL, self.out)
        self.assertIn("missing_columns", self.kinds(r))
        self.assertFalse(r["inspection"]["data_check"]["passed"])

    def test_numeric_errors_detected(self):
        cases = [
            ("Chamfer(um)", "nan"),
            ("Chamfer(um)", ""),
            ("Chamfer(um)", "abc"),
            ("Chamfer(um)", -0.1),
            ("IoU", 1.5),
            ("정답깊이(um)", 0),
            ("스텝", 2.5),
        ]
        for col, bad in cases:
            with self.subTest(col=col, bad=bad):

                def mutate(rows, col=col, bad=bad):
                    next(row for row in rows if row["방식"] == GROUP)[col] = bad

                metrics, samples = build(
                    self.dir / f"in_{cases.index((col, bad))}", mutate_val=mutate
                )
                _, _, r = run(metrics, samples, PROVISIONAL, self.out)
                self.assertTrue(
                    self.kinds(r) & {"bad_number", "out_of_range", "aggregation_mismatch"}
                )
                self.assertFalse(r["inspection"]["data_check"]["passed"])

    def test_signed_negative_depth_is_not_an_error(self):
        _, _, r = self.go()  # synthetic data contains negative per-step depth errors
        self.assertTrue(r["inspection"]["data_check"]["passed"])

    def test_unknown_mode_and_mode_mismatch_detected(self):
        def mutate(rows):
            next(row for row in rows if row["방식"] == GROUP and row["종류"] == "롤아웃")[
                "종류"
            ] = "2스텝"

        _, _, r = self.go(mutate_val=mutate)
        self.assertTrue({"unknown_mode", "mode_id_mismatch"} <= self.kinds(r))

    def test_official_decision_is_hold_and_separate_from_comparison(self):
        _, text, r = self.go()
        for mode in ("onestep", "rollout"):
            self.assertEqual(r["decisions"][mode]["action"], "HOLD")
            self.assertFalse(r["decisions"][mode]["training_started"])
            self.assertFalse(r["decisions"][mode]["active_model_changed"])
            self.assertEqual(r["comparisons"][mode]["label"], "임시 비교 (공식 판정 아님)")
            blockers = r["decisions"][mode]["remaining_blockers_even_if_approved"]
            self.assertTrue(any("reference_verified" in b for b in blockers))
            self.assertTrue(any("invalid_geometry_count" in b for b in blockers))
        self.assertLess(text.index("[임시 비교]"), text.index("[공식 판정]"))

    def test_demo_policy_cannot_judge_real_scores(self):
        _, _, r = self.go()
        demo = load_thresholds(DEMO)
        for mode, p in r["inputs"].items():
            _, d = judge(p, demo)
            self.assertEqual(d["action"], "HOLD")
            # even an "approved" copy of the demo policy rejects the different metric contract
            approved = dict(demo[("FNO", mode)], status="approved")
            self.assertEqual(decide(dict(p, model_id="FNO/main_s1"), approved)["action"], "HOLD")

    def test_provisional_comparison_marks_exceed_and_unmeasured(self):
        policy = copy.deepcopy(load_thresholds(PROVISIONAL)[("FNO", "rollout")])
        evaluation = dict(
            metrics=dict(chamfer_um_recipe_mean=0.6, abs_depth_error_pct_recipe_mean=1.0)
        )
        rows = {
            x["metric"]: x["result"] for x in provisional_comparison(evaluation, policy)["rows"]
        }
        self.assertEqual(
            rows,
            {
                "chamfer_um_recipe_mean": "초과",
                "abs_depth_error_pct_recipe_mean": "이내",
                "invalid_geometry_count": "미측정",
            },
        )

    def test_provisional_thresholds_file_is_marked_and_matches_contracts(self):
        data = read_json(PROVISIONAL)
        self.assertIn("임시 비교용", data["_notice"])
        for p in data["policies"]:
            self.assertEqual(p["model_family"], "FNO")
            self.assertEqual(p["status"], "provisional")
            self.assertEqual(
                p["evaluation_contract_id"], AGGREGATION[p["eval_mode"]]["contract_id"]
            )

    def test_outputs_written_and_sources_unchanged(self):
        metrics, samples = build(self.dir / "in")
        files = [metrics, *samples.iterdir()]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        out1, _, _ = run(metrics, samples, PROVISIONAL, self.out)
        out2, _, _ = run(metrics, samples, PROVISIONAL, self.out)
        self.assertNotEqual(out1, out2)
        self.assertEqual(
            {p.name for p in out1.iterdir()},
            {
                "source_files.json",
                "selected_rows.json",
                "inspection.json",
                "score_summary.json",
                "unmeasured.json",
                "harness_input_onestep.json",
                "harness_input_rollout.json",
                "provisional_comparison.json",
                "official_decisions.json",
                "provider_definition.json",
                "report.txt",
            },
        )
        src = json.loads((out1 / "source_files.json").read_text(encoding="utf-8"))["files"]
        self.assertEqual(
            {Path(k).name: v["sha256"] for k, v in src.items()},
            {p.name: h for p, h in before.items()},
        )
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})

    def test_local_paths_file_sets_inputs_else_fallback(self):
        import fno_adapter

        cfg = self.dir / "local_paths.json"
        with unittest.mock.patch.object(fno_adapter, "LOCAL_PATHS", cfg):
            self.assertEqual(
                fno_adapter.local_input("fno_metrics", "m.csv"),
                fno_adapter.ROOT / "inputs" / "m.csv",
            )
            cfg.write_text(json.dumps({"fno_metrics": str(self.dir / "x.csv")}), encoding="utf-8")
            self.assertEqual(fno_adapter.local_input("fno_metrics", "m.csv"), self.dir / "x.csv")
            self.assertEqual(
                fno_adapter.local_input("mgn_input", "g"), fno_adapter.ROOT / "inputs" / "g"
            )
            cfg.write_text("{broken", encoding="utf-8")
            self.assertEqual(
                fno_adapter.local_input("fno_metrics", "m.csv"),
                fno_adapter.ROOT / "inputs" / "m.csv",
            )

    def test_saved_harness_input_is_accepted_by_harness(self):
        out, _, _ = self.go()
        p = read_json(out / "harness_input_rollout.json")
        policy, d = judge(p, load_thresholds(PROVISIONAL))
        self.assertEqual(policy["version"], "fno-provisional-rollout-v1")
        self.assertEqual(d["action"], "HOLD")


if __name__ == "__main__":
    unittest.main(verbosity=2)
