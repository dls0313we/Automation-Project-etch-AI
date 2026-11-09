"""Counterexamples for the bounded mock retry controller, before implementation."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from harness import load_thresholds, read_json
from mock_retrain_loop import MockTrainingFailed
from retry_retrain_loop import ATTEMPT_CONFIGS, run, run_case

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "mock_loop_inputs"


class ScriptedRunner:
    def __init__(self, results):
        self.results = results
        self.requests = []

    def run(self, request, plan, base):
        self.requests.append(copy.deepcopy(request))
        result = self.results[len(self.requests) - 1]
        if isinstance(result, Exception):
            raise result
        return dict(
            candidate=copy.deepcopy(result), candidate_file="in-memory synthetic evaluation"
        )


class RetryLoopTests(unittest.TestCase):
    def setUp(self):
        self.current = read_json(OLD / "evaluations/current_over_limit.json")
        self.good = read_json(OLD / "evaluations/candidate_better.json")
        self.bad = read_json(OLD / "evaluations/candidate_worse.json")
        self.thresholds = load_thresholds(OLD / "thresholds_mock.json")
        self.plans = [dict(result="candidate")] * 5

    def go(self, results, current=None, thresholds=None, **kwargs):
        runner = ScriptedRunner(results)
        record = run_case(
            self.current if current is None else current,
            self.thresholds if thresholds is None else thresholds,
            self.plans,
            base=OLD,
            runner=runner,
            **kwargs,
        )
        return record, runner

    def test_pass_on_first_attempt_never_calls_second(self):
        r, runner = self.go([self.good, AssertionError("must not execute")])
        self.assertEqual(r["status"], "RECOMMEND_REPLACEMENT")
        self.assertEqual(r["passed_attempt"], 1)
        self.assertEqual(len(runner.requests), 1)
        self.assertEqual(r["attempts"][0]["mock_gate_result"], "PASS")

    def test_first_quality_block_then_second_pass_stops(self):
        r, runner = self.go([self.bad, self.good, AssertionError("must not execute")])
        self.assertEqual(r["passed_attempt"], 2)
        self.assertEqual(len(runner.requests), 2)
        self.assertEqual([a["mock_gate_result"] for a in r["attempts"]], ["BLOCK", "PASS"])

    def test_pass_on_fourth_is_allowed(self):
        r, runner = self.go([self.bad] * 3 + [self.good])
        self.assertEqual(r["passed_attempt"], 4)
        self.assertEqual(len(runner.requests), 4)

    def test_four_blocks_exhaust_and_never_execute_fifth_plan(self):
        r, runner = self.go([self.bad] * 4 + [self.good])
        self.assertEqual(r["status"], "EXHAUSTED_KEEP_CURRENT")
        self.assertEqual(len(runner.requests), 4)
        self.assertEqual(len(r["attempts"]), 4)
        self.assertIsNone(r["recommended_model"])

    def test_attempt_limit_is_a_hard_ceiling(self):
        for limit in [0, -1, 5, True, 1.5]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.go([], max_attempts=limit)
        r, runner = self.go([self.bad] * 4, max_attempts=2)
        self.assertEqual(len(runner.requests), 2)
        self.assertEqual(r["status"], "EXHAUSTED_KEEP_CURRENT")

    def test_escalation_plan_and_baseline_do_not_change_with_failed_candidates(self):
        r, runner = self.go([self.bad] * 4)
        self.assertEqual(
            [q["config"]["train_recipe_count"] for q in runner.requests], [40, 100, 100, 100]
        )
        self.assertEqual([q["config"]["epochs"] for q in runner.requests], [5, 5, 12, 12])
        self.assertEqual(
            [q["config"]["additional_recipes_planned"] for q in runner.requests], [0, 60, 0, 0]
        )
        self.assertEqual(runner.requests[-1]["config"]["learning_rate"], 0.0005)
        self.assertTrue(all(q["starting_model"] == "mgn-mock-v1" for q in runner.requests))
        self.assertEqual([a["config"] for a in r["attempts"]], ATTEMPT_CONFIGS)
        self.assertEqual(len({a["candidate_model"] for a in r["attempts"]}), 4)
        self.assertTrue(
            all(a["decision"]["evidence"].get("candidate_failed_checks") for a in r["attempts"])
        )

    def test_provisional_policy_never_starts_training(self):
        thresholds = load_thresholds(OLD / "thresholds_mock_provisional.json")
        r, runner = self.go([], thresholds=thresholds)
        self.assertEqual(r["status"], "NOT_STARTED_HOLD")
        self.assertFalse(runner.requests)

    def test_unverified_reference_never_starts_training(self):
        current = copy.deepcopy(self.current)
        current["data_check"]["reference_verified"] = False
        r, runner = self.go([], current=current)
        self.assertEqual(r["status"], "NOT_STARTED_HOLD")
        self.assertFalse(runner.requests)

    def test_bad_input_data_stops_before_training(self):
        current = copy.deepcopy(self.current)
        current["data_check"].update(passed=False, invalid_recipe_count=1)
        r, runner = self.go([], current=current)
        self.assertEqual(r["status"], "NOT_STARTED_DATA_REVIEW")
        self.assertFalse(runner.requests)

    def test_missing_training_data_waits(self):
        current = copy.deepcopy(self.current)
        current["training_data_ready"] = False
        r, runner = self.go([], current=current)
        self.assertEqual(r["status"], "NOT_STARTED_WAIT_FOR_TRAINING_DATA")
        self.assertFalse(runner.requests)

    def test_current_model_within_limits_does_not_start(self):
        r, runner = self.go([], current=read_json(OLD / "evaluations/current_ok.json"))
        self.assertEqual(r["status"], "NO_RETRAIN_NEEDED")
        self.assertFalse(runner.requests)

    def test_missing_candidate_score_stops_instead_of_retrying(self):
        candidate = copy.deepcopy(self.good)
        del candidate["metrics"]["depth_mae_um"]
        r, runner = self.go([candidate, self.good])
        self.assertEqual(r["status"], "STOPPED_CANDIDATE_EVALUATION")
        self.assertEqual(len(runner.requests), 1)
        self.assertEqual(r["attempts"][0]["mock_gate_result"], "STOP")

    def test_candidate_data_failure_is_not_a_quality_retry(self):
        candidate = copy.deepcopy(self.good)
        candidate["data_check"].update(passed=False, invalid_recipe_count=1)
        r, runner = self.go([candidate, self.good])
        self.assertEqual(r["status"], "STOPPED_DATA_REVIEW")
        self.assertEqual(len(runner.requests), 1)

    def test_evaluation_dataset_mismatch_stops(self):
        candidate = copy.deepcopy(self.good)
        candidate["evaluation_dataset_id"] = "a-different-validation-set"
        r, runner = self.go([candidate, self.good])
        self.assertEqual(r["status"], "STOPPED_CANDIDATE_EVALUATION")
        self.assertEqual(len(runner.requests), 1)

    def test_runner_failure_is_not_a_quality_retry(self):
        r, runner = self.go([MockTrainingFailed("scripted failure"), self.good])
        self.assertEqual(r["status"], "STOPPED_TRAINING_FAILED")
        self.assertEqual(len(runner.requests), 1)
        self.assertIn("scripted failure", r["reasons"][0])

    def test_real_current_and_real_candidate_are_refused(self):
        current = copy.deepcopy(self.current)
        current["data_origin"] = "real_mgn_results"
        r, runner = self.go([], current=current)
        self.assertEqual(r["status"], "REJECTED_NON_DEMO_INPUT")
        self.assertFalse(runner.requests)
        candidate = copy.deepcopy(self.good)
        candidate["data_origin"] = "real_mgn_results"
        r, runner = self.go([candidate, self.good])
        self.assertEqual(r["status"], "STOPPED_NON_DEMO_CANDIDATE")
        self.assertEqual(len(runner.requests), 1)

    def test_original_inputs_and_actual_execution_flags_remain_unchanged(self):
        before = copy.deepcopy([self.current, self.good, self.bad, self.thresholds])
        r, _ = self.go([self.bad, self.good])
        self.assertEqual([self.current, self.good, self.bad, self.thresholds], before)
        self.assertFalse(r["real_training_executed"])
        self.assertFalse(r["real_model_replaced"])
        self.assertFalse(r["real_data_generation_executed"])
        self.assertTrue(
            all(
                a["mock_training_executed"] and not a["real_training_executed"]
                for a in r["attempts"]
            )
        )
        self.assertTrue(all(not a["decision"]["active_model_changed"] for a in r["attempts"]))

    def test_demo_writes_new_reports_with_all_expected_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, summary = run(output_root=Path(tmp))
            next_out, _ = run(output_root=Path(tmp))
            self.assertNotEqual(out, next_out)
            self.assertEqual(summary["scenario_count"], 6)
            self.assertEqual(summary["expected_matched_count"], 6)
            self.assertTrue((out / "report.html").is_file())
            saved = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["real_model_replaced"])
            for r in saved["results"]:
                self.assertLessEqual(len(r["attempts"]), 4)
                self.assertTrue((out / "cases" / (r["id"] + ".json")).is_file())


if __name__ == "__main__":
    unittest.main()
