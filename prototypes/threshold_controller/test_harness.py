import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from controller import decide
from harness import (
    ROOT,
    ThresholdFileError,
    judge,
    load_thresholds,
    read_json,
    run_cases,
    to_payload,
)

THRESHOLDS = ROOT / "thresholds_demo.json"
EVAL = ROOT / "harness_inputs" / "evaluations"


def ev(name):
    return read_json(EVAL / name)


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.t = load_thresholds(THRESHOLDS)
        self.ok = ev("mgn_onestep_v1_ok.json")

    def test_demo_scenarios_give_expected_decisions(self):
        spec = read_json(ROOT / "harness_inputs" / "scenarios.json")
        self.assertEqual(
            [c["expected"] for c in spec["scenarios"]],
            ["KEEP_CURRENT", "RETRAIN_REQUEST", "DATA_REVIEW", "HOLD", "APPROVE_CANDIDATE"],
        )
        for c in spec["scenarios"]:
            with self.subTest(case=c["id"]):
                cand = ev(Path(c["candidate"]).name) if c.get("candidate") else None
                policy, d = judge(ev(Path(c["current"]).name), self.t, cand)
                self.assertEqual(d["action"], c["expected"])
                self.assertFalse(d["training_started"])
                self.assertFalse(d["active_model_changed"])

    def test_policy_selected_by_family_and_mode(self):
        policy, _ = judge(ev("fno_rollout_v1_missing_metric.json"), self.t)
        self.assertEqual((policy["model_family"], policy["eval_mode"]), ("FNO", "rollout"))

    def test_missing_metric_reason_names_the_metric(self):
        _, d = judge(ev("fno_rollout_v1_missing_metric.json"), self.t)
        self.assertEqual(d["action"], "HOLD")
        self.assertIn("depth_mae_um", d["reasons"][0])

    def test_mgn_result_against_fno_policy_is_held(self):
        d = decide(to_payload(self.ok), self.t[("FNO", "onestep")])
        self.assertEqual(d["action"], "HOLD")
        self.assertIn("model_family", d["reasons"][0])

    def test_onestep_result_against_rollout_policy_is_held(self):
        d = decide(to_payload(self.ok), self.t[("MGN", "rollout")])
        self.assertEqual(d["action"], "HOLD")
        self.assertIn("eval_mode", d["reasons"][0])

    def test_candidate_with_different_mode_or_family_is_held(self):
        for field, value in [("eval_mode", "rollout"), ("model_family", "FNO")]:
            with self.subTest(field=field):
                cand = ev("mgn_onestep_v2_candidate.json")
                cand[field] = value
                _, d = judge(self.ok, self.t, cand)
                self.assertEqual(d["action"], "HOLD")

    def test_metric_unit_mismatch_is_held(self):
        e = copy.deepcopy(self.ok)
        e["metric_units"]["depth_mae_um"] = "nm"
        _, d = judge(e, self.t)
        self.assertEqual(d["action"], "HOLD")
        self.assertIn("unit", d["reasons"][0])

    def test_demo_thresholds_cannot_judge_real_results(self):
        e = dict(self.ok, data_origin="measured")
        _, d = judge(e, self.t)
        self.assertEqual(d["action"], "HOLD")

    def test_unknown_family_mode_or_version_is_held(self):
        for field, value in [
            ("model_family", "GNN"),
            ("eval_mode", "multistep"),
            ("model_version", ""),
        ]:
            with self.subTest(field=field):
                _, d = judge(dict(self.ok, **{field: value}), self.t)
                self.assertEqual(d["action"], "HOLD")

    def test_missing_threshold_entry_is_held(self):
        t = {k: v for k, v in self.t.items() if k != ("MGN", "onestep")}
        policy, d = judge(self.ok, t)
        self.assertIsNone(policy)
        self.assertEqual(d["action"], "HOLD")

    def test_edited_threshold_changes_decision(self):
        t = copy.deepcopy(self.t)
        t[("MGN", "onestep")]["max_metrics"]["mean_chamfer_um"] = 0.05
        _, d = judge(dict(self.ok, training_data_ready=True), t)
        self.assertEqual(d["action"], "RETRAIN_REQUEST")

    def test_broken_threshold_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = read_json(THRESHOLDS)
            data["policies"].append(copy.deepcopy(data["policies"][0]))
            p = Path(tmp) / "dup.json"
            p.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ThresholdFileError):
                load_thresholds(p)

    def test_all_demo_inputs_and_thresholds_are_marked(self):
        t = read_json(THRESHOLDS)
        self.assertIn("시연용", t["_notice"])
        self.assertTrue(all(p["status"] == "demo" for p in t["policies"]))
        for path in EVAL.glob("*.json"):
            e = read_json(path)
            self.assertEqual(e["data_origin"], "synthetic_demo", path.name)
            self.assertIn("가상", e["_notice"], path.name)
            if e["model_family"] == "FNO":
                self.assertIn("FNO", e["_notice"], path.name)

    def test_run_writes_new_folder_and_leaves_inputs_unchanged(self):
        files = sorted(EVAL.glob("*.json")) + [THRESHOLDS]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        cases = [
            dict(id="keep", label="keep", current=EVAL / "mgn_onestep_v1_ok.json"),
            dict(
                id="candidate",
                label="candidate",
                current=EVAL / "mgn_onestep_v1_ok.json",
                candidate=EVAL / "mgn_onestep_v2_candidate.json",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out1, results, texts = run_cases(cases, THRESHOLDS, tmp)
            out2, _, _ = run_cases(cases, THRESHOLDS, tmp)
            self.assertNotEqual(out1, out2)
            self.assertEqual(
                {p.name for p in out1.iterdir()},
                {"keep.json", "candidate.json", "report.txt", "summary.json"},
            )
            self.assertEqual([r["action"] for r in results], ["KEEP_CURRENT", "APPROVE_CANDIDATE"])
            rec = json.loads((out1 / "keep.json").read_text(encoding="utf-8"))
            self.assertEqual(rec["decision"]["action"], "KEEP_CURRENT")
            self.assertEqual(rec["policy_used"]["version"], "demo-MGN-onestep-v1")
            for word in ["MGN", "onestep", "0.1", "≤ 0.2", "KEEP_CURRENT", "이유:"]:
                self.assertIn(word, texts[0])
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


if __name__ == "__main__":
    unittest.main(verbosity=2)
