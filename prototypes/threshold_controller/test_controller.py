import copy
import unittest

from controller import decide

POLICY = dict(
    version="synthetic-demo-v1",
    status="demo",
    evaluation_contract_id="demo-metric-v1",
    primary_metric="mean_chamfer_um",
    max_metrics=dict(mean_chamfer_um=0.2, depth_mae_um=0.5, invalid_geometry_count=0),
    max_relative_regression=0.0,
    min_primary_improvement_relative=0.05,
    denominator_floor=1e-9,
)
METRICS = dict(
    model_id="demo-old",
    evaluation_dataset_id="synthetic-validation-001",
    evaluation_contract_id="demo-metric-v1",
    split="validation",
    unit="um",
    recipe_count=10,
    data_origin="synthetic_demo",
    training_data_ready=False,
    data_check=dict(passed=True, invalid_recipe_count=0, reference_verified=True),
    metrics=dict(mean_chamfer_um=0.1, depth_mae_um=0.2, invalid_geometry_count=0),
)


class ControllerTests(unittest.TestCase):
    def test_good_current_is_kept(self):
        self.assertEqual(decide(METRICS, POLICY)["action"], "KEEP_CURRENT")

    def test_invalid_data_has_priority(self):
        m = copy.deepcopy(METRICS)
        m["data_check"].update(passed=False, invalid_recipe_count=5)
        self.assertEqual(decide(m, dict(status="draft"))["action"], "DATA_REVIEW")

    def test_draft_thresholds_cannot_judge(self):
        self.assertEqual(decide(METRICS, dict(POLICY, status="draft"))["action"], "HOLD")

    def test_failure_waits_for_approved_labels(self):
        m = copy.deepcopy(METRICS)
        m["metrics"]["mean_chamfer_um"] = 0.3
        self.assertEqual(decide(m, POLICY)["action"], "WAIT_FOR_TRAINING_DATA")
        m["training_data_ready"] = True
        self.assertEqual(decide(m, POLICY)["action"], "RETRAIN_REQUEST")

    def test_missing_measurement_is_not_zero(self):
        m = copy.deepcopy(METRICS)
        del m["metrics"]["invalid_geometry_count"]
        self.assertEqual(decide(m, POLICY)["action"], "HOLD")

    def test_bad_numbers_are_held(self):
        for bad in [float("nan"), float("inf"), -1, True, None]:
            m = copy.deepcopy(METRICS)
            m["metrics"]["mean_chamfer_um"] = bad
            self.assertEqual(decide(m, POLICY)["action"], "HOLD")

    def test_demo_cannot_judge_real_data(self):
        m = dict(METRICS, data_origin="measured")
        self.assertEqual(decide(m, POLICY)["action"], "HOLD")

    def test_different_units_or_protocol_held(self):
        self.assertEqual(decide(dict(METRICS, unit="nm"), POLICY)["action"], "HOLD")
        self.assertEqual(
            decide(dict(METRICS, evaluation_contract_id="different"), POLICY)["action"], "HOLD"
        )

    def test_test_set_not_used_to_design_trigger(self):
        self.assertEqual(decide(dict(METRICS, split="test"), POLICY)["action"], "HOLD")

    def test_candidate_compared_on_same_dataset(self):
        c = copy.deepcopy(METRICS)
        c.update(model_id="demo-new", evaluation_dataset_id="different")
        self.assertEqual(decide(METRICS, POLICY, c)["action"], "HOLD")

    def test_better_candidate_recommended_without_deployment(self):
        c = copy.deepcopy(METRICS)
        c["model_id"] = "demo-new"
        c["metrics"]["mean_chamfer_um"] = 0.08
        result = decide(METRICS, POLICY, c)
        self.assertEqual(result["action"], "APPROVE_CANDIDATE")
        self.assertFalse(result["training_started"])
        self.assertFalse(result["active_model_changed"])

    def test_candidate_regression_rejected_even_if_primary_improves(self):
        c = copy.deepcopy(METRICS)
        c["model_id"] = "demo-new"
        c["metrics"].update(mean_chamfer_um=0.08, depth_mae_um=0.3)
        self.assertEqual(decide(METRICS, POLICY, c)["action"], "KEEP_CURRENT")

    def test_candidate_must_pass_absolute_gate(self):
        m, c = copy.deepcopy(METRICS), copy.deepcopy(METRICS)
        m["metrics"]["mean_chamfer_um"] = 0.5
        c["metrics"]["mean_chamfer_um"] = 0.3
        c["model_id"] = "demo-new"
        self.assertEqual(decide(m, POLICY, c)["action"], "KEEP_CURRENT")

    def test_fractional_count_rejected(self):
        m = dict(METRICS, recipe_count=2.5)
        self.assertEqual(decide(m, POLICY)["action"], "HOLD")

    def test_malformed_nested_payload_held(self):
        for name in ["data_check", "metrics"]:
            self.assertEqual(decide(dict(METRICS, **{name: None}), POLICY)["action"], "HOLD")


if __name__ == "__main__":
    unittest.main(verbosity=2)
