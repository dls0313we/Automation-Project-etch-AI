import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mock_retrain_loop as loop
from harness import ROOT, read_json

INPUTS = ROOT / "mock_loop_inputs"
SCENARIOS = {s["id"]: s for s in read_json(INPUTS / "scenarios.json")["scenarios"]}


class CountingTrainer(loop.MockTrainer):
    pass


class MockLoopTests(unittest.TestCase):
    def run_one(self, sid, **override):
        trainer = CountingTrainer(dict(SCENARIOS[sid]["mock_training"]), INPUTS)
        result = loop.run_scenario(dict(SCENARIOS[sid], **override), INPUTS, trainer)
        return result, trainer

    def test_every_scenario_reaches_expected_status(self):
        for sid, s in SCENARIOS.items():
            with self.subTest(sid=sid):
                self.assertEqual(self.run_one(sid)[0]["loop_status"], s["expected"])

    def test_trainer_not_called_unless_retrain_requested(self):
        for sid in ("1_meets_criteria", "5a_provisional_thresholds", "5b_reference_unverified"):
            with self.subTest(sid=sid):
                r, t = self.run_one(sid)
                self.assertEqual(t.calls, 0)
                self.assertFalse(r["mock_training_executed"])
                self.assertEqual(
                    [s["step"] for s in r["steps"]], ["evaluate_current", "retrain_request"]
                )

    def test_trainer_called_once_when_retrain_requested(self):
        for sid in (
            "2_retrain_improves",
            "3_candidate_worse",
            "4a_candidate_metric_missing",
            "4b_mock_training_failed",
        ):
            with self.subTest(sid=sid):
                r, t = self.run_one(sid)
                self.assertEqual(t.calls, 1)
                self.assertTrue(r["mock_training_executed"])
                mock_steps = [s for s in r["steps"] if s["mock_executed"]]
                self.assertEqual([s["step"] for s in mock_steps], ["mock_training"])

    def test_real_execution_is_never_recorded(self):
        for sid in SCENARIOS:
            r, _ = self.run_one(sid)
            self.assertFalse(r["real_training_executed"])
            self.assertFalse(r["real_model_replaced"])
            self.assertFalse(any(s["real_executed"] for s in r["steps"]))
            for s in r["steps"]:
                if "decision" in s:
                    self.assertFalse(s["decision"]["training_started"])
                    self.assertFalse(s["decision"]["active_model_changed"])

    def test_recommendation_is_not_execution(self):
        r, _ = self.run_one("2_retrain_improves")
        rep = r["steps"][-1]
        self.assertEqual(rep["step"], "replacement")
        self.assertTrue(rep["recommended"])
        self.assertFalse(rep["executed"])
        self.assertEqual(r["candidate_model"], "mgn-mock-v2")

    def test_stop_reasons_are_recorded(self):
        r, _ = self.run_one("4a_candidate_metric_missing")
        self.assertIn("depth_mae_um", r["final_reason"][0])
        r, _ = self.run_one("4b_mock_training_failed")
        self.assertIn("모의 실패", r["final_reason"][0])
        self.assertEqual(r["steps"][-1]["outcome"], "failed")

    def test_mock_runner_refuses_real_or_unrequested_runs(self):
        t = loop.MockTrainer(
            dict(result="candidate", candidate="evaluations/candidate_better.json"), INPUTS
        )
        for request in [
            dict(decision="KEEP_CURRENT", data_origin="synthetic_demo"),
            dict(decision="RETRAIN_REQUEST", data_origin="mgn_scores_provided_by_jinju"),
        ]:
            with self.subTest(request=request), self.assertRaises(loop.MockTrainingFailed):
                t.run(request)

    def test_real_like_current_data_never_starts_mock_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "in"
            shutil.copytree(INPUTS, base)
            cur = read_json(base / "evaluations/current_over_limit.json")
            cur["data_origin"] = "mgn_scores_provided_by_jinju"
            (base / "evaluations/current_over_limit.json").write_text(
                json.dumps(cur), encoding="utf-8"
            )
            r = loop.run_scenario(SCENARIOS["2_retrain_improves"], base)
        self.assertEqual(r["loop_status"], "NOT_STARTED_HOLD")
        self.assertFalse(r["mock_training_executed"])

    def test_mock_inputs_are_marked_and_separate(self):
        for path in (INPUTS / "evaluations").glob("*.json"):
            e = read_json(path)
            self.assertEqual(e["data_origin"], "synthetic_demo", path.name)
            self.assertIn("가상", e["_notice"])
        for name in ("thresholds_mock.json", "thresholds_mock_provisional.json"):
            t = read_json(INPUTS / name)
            self.assertIn("모의 재학습 루프", t["_notice"])
            self.assertTrue(all(p["version"].startswith("mock-loop-") for p in t["policies"]))

    def test_run_writes_new_folder_and_leaves_existing_files_unchanged(self):
        guarded = [
            ROOT / n
            for n in (
                "controller.py",
                "harness.py",
                "thresholds_demo.json",
                "thresholds_fno_provisional.json",
                "thresholds_mgn_provisional.json",
            )
        ]
        guarded += sorted(INPUTS.rglob("*.json"))
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in guarded}
        with tempfile.TemporaryDirectory() as tmp:
            out1, results, text = loop.run(INPUTS / "scenarios.json", Path(tmp))
            out2, _, _ = loop.run(INPUTS / "scenarios.json", Path(tmp))
            self.assertNotEqual(out1, out2)
            self.assertEqual(
                {p.name for p in out1.iterdir()}, {"summary.json", "report.txt", "scenarios"}
            )
            self.assertEqual(len(list((out1 / "scenarios").iterdir())), len(SCENARIOS))
            s = json.loads((out1 / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(s["real_training_executed"])
            self.assertFalse(s["real_model_replaced"])
            self.assertIn("실제 학습 실행 False", text)
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in guarded})

    def test_main_exits_nonzero_on_unexpected_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "in"
            shutil.copytree(INPUTS, base)
            spec = read_json(base / "scenarios.json")
            spec["scenarios"][0]["expected"] = "RECOMMEND_REPLACEMENT"
            (base / "scenarios.json").write_text(json.dumps(spec), encoding="utf-8")
            argv = [
                "mock_retrain_loop.py",
                "--scenarios",
                str(base / "scenarios.json"),
                "--output-root",
                str(Path(tmp) / "out"),
            ]
            with (
                mock.patch("sys.argv", argv),
                mock.patch("builtins.print"),
                self.assertRaises(SystemExit) as ctx,
            ):
                loop.main()
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
