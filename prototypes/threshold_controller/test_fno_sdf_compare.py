"""Analytic CD counterexamples and SDF-reference alignment/integration checks."""

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from fno_gt_overlay import PARAM_KEYS, FnoOverlayError, load_sdf
from fno_sdf_compare import (
    comparison,
    cross_section,
    load_truth,
    percent_error,
    read_params,
    run,
    validate_pair,
)
from test_mgn_geometry import npy_bytes


def unicode_npy(text):
    header = repr(dict(descr=f"<U{len(text)}", fortran_order=False, shape=())).encode("latin1")
    padding = (64 - (10 + len(header) + 1) % 64) % 64
    header += b" " * padding + b"\n"
    return (
        b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + text.encode("utf-32-le")
    )


PARAMS = {k: i + 1.0 for i, k in enumerate(PARAM_KEYS)}
X, Y = [-2.0, -0.5, 0.5, 2.0], [-2.0, -0.5, 1.0]


def write_arrays(path, *, truth, times=None, params=None, x=None):
    x = X if x is None else x
    times = ([0, 1, 2] if truth else [1, 2]) if times is None else times
    frame = [min(1 - abs(xx), yy + 1) for yy in Y for xx in x]
    arrays = dict(
        phi=npy_bytes(frame * len(times), "<f4", (len(times), len(Y), len(x))),
        x=npy_bytes(x, "<f8", (len(x),)),
        y=npy_bytes(Y, "<f8", (len(Y),)),
    )
    arrays["t" if truth else "step"] = npy_bytes(times, "<i8", (len(times),))
    if truth:
        arrays["params"] = unicode_npy(json.dumps(PARAMS if params is None else params))
    with zipfile.ZipFile(path, "w") as z:
        for name, data in arrays.items():
            z.writestr(name + ".npy", data)


class WidthTests(unittest.TestCase):
    def test_linear_roots_with_center_between_grid_points(self):
        field = [-1, 0.5, 0.5, -1] * 2
        v = cross_section(field, X, [-1, 1], 0)
        self.assertAlmostEqual(v["width_um"], 2)
        self.assertAlmostEqual(v["left_x_um"], -1)

    def test_interpolates_y_instead_of_using_nearest_row(self):
        v = cross_section([-2, 1, -2, -1, 2, -1], [-2, 0, 2], [-2, 0], -1)
        self.assertAlmostEqual(v["width_um"], 2)

    def test_uses_center_air_interval_not_other_cavities(self):
        x = list(range(-4, 5))
        row = [-1, 1, -1, -1, 1, -1, -1, 1, -1]
        self.assertAlmostEqual(cross_section(row * 2, x, [-1, 1], 0)["width_um"], 1)

    def test_solid_center_means_zero_predicted_width_and_minus_100_percent(self):
        v = cross_section([-1, -1, -1] * 2, [-1, 0, 1], [-1, 1], 0)
        self.assertEqual(v["width_um"], 0)
        self.assertEqual(percent_error(v["width_um"], 2), -100)
        self.assertIsNone(percent_error(0, 0))

    def test_outside_grid_or_open_side_is_unmeasured_not_domain_width(self):
        self.assertIsNone(cross_section([1] * 6, [-1, 0, 1], [-1, 1], 0)["width_um"])
        self.assertIsNone(cross_section([1] * 6, [-1, 0, 1], [-1, 1], 2)["width_um"])

    def test_signed_error_and_missing_are_preserved(self):
        self.assertEqual(percent_error(1, 2), -50)
        self.assertEqual(comparison("CD", None, "", "percentage_points")["status"], "both_missing")
        self.assertIsNone(comparison("CD", None, "", "percentage_points")["provided"])
        with self.assertRaises(FnoOverlayError):
            comparison("CD", 0, "nan", "percentage_points")


class SdfComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples = self.root / "samples"
        self.truth = self.root / "outer" / "sdf"
        self.samples.mkdir()
        self.truth.mkdir(parents=True)
        self.gt = self.truth / "r0301.npz"
        write_arrays(self.gt, truth=True)
        for mode in ["onestep", "rollout"]:
            write_arrays(self.samples / f"r0301_{mode}.npz", truth=False)
        (self.samples / "README.json").write_text(
            json.dumps(
                {
                    "데이터 구분": "validation",
                    "레시피": {"r0301": {"레시피 값": PARAMS}},
                }
            ),
            encoding="utf-8",
        )
        self.columns = [
            "레시피",
            "방식",
            "스텝",
            "정답깊이(um)",
            "예측깊이(um)",
            "깊이오차(%)",
            "CD_위오차(%)",
            "CD_중간오차(%)",
            "CD_바닥오차(%)",
        ]
        self.rows = [
            dict(zip(self.columns, ["r0301", mode, f"t{step:03d}", 1, 1, 0, 0, 0, 0]))
            for mode in ["one-step", "rollout"]
            for step in [1, 2]
        ]
        self.write_csv()

    def write_csv(self):
        with (self.samples / "metrics_by_step.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.rows)

    def go(self):
        return run(self.samples, self.truth.parent, self.root / "results")

    def test_synthetic_equal_prediction_matches_and_never_approves_or_changes_sources(self):
        before = {str(p): p.read_bytes() for p in [self.gt, *self.samples.iterdir()]}
        out, s = self.go()
        self.assertEqual(s["status"], "ok")
        self.assertEqual(s["frames_checked"], 4)
        self.assertEqual(s["comparison_counts"], {"match": 24})
        self.assertEqual(len(s["cases"]), 2)
        self.assertFalse(s["reference_verified"])
        self.assertFalse(s["training_started"])
        self.assertFalse(s["active_model_changed"])
        self.assertTrue((out / "comparisons.csv").is_file())
        self.assertTrue((out / "report.html").is_file())
        self.assertEqual(before, {p: Path(p).read_bytes() for p in before})

    def test_step_zero_cannot_be_silently_shifted_to_match_predictions(self):
        write_arrays(self.gt, truth=True, times=[1, 2, 3])
        with self.assertRaises(FnoOverlayError):
            load_truth(self.gt)

    def test_recipe_parameter_mismatch_stops_comparison(self):
        write_arrays(self.gt, truth=True, params=dict(PARAMS, ion_flux=100))
        _, s = self.go()
        self.assertEqual(s["status"], "incomplete")
        self.assertEqual(s["frames_checked"], 0)
        self.assertTrue(s["errors"])

    def test_coordinate_grid_mismatch_stops_comparison(self):
        write_arrays(self.samples / "r0301_rollout.npz", truth=False, x=[-3, -0.5, 0.5, 3])
        _, s = self.go()
        self.assertEqual(s["status"], "incomplete")
        self.assertEqual(s["frames_checked"], 2)

    def test_missing_reference_cannot_reuse_previous_success(self):
        out, _ = self.go()
        different = self.root / "missing"
        different.mkdir()
        write_arrays(different / "r0302.npz", truth=True)
        next_out, s = run(self.samples, different, self.root / "results")
        self.assertNotEqual(out, next_out)
        self.assertEqual(s["frames_checked"], 0)
        self.assertEqual(s["status"], "incomplete")

    def test_numeric_disagreement_is_reported_instead_of_changing_original_csv(self):
        self.rows[0]["CD_바닥오차(%)"] = -20
        self.write_csv()
        _, s = self.go()
        self.assertEqual(s["status"], "differences")
        self.assertEqual(s["comparison_counts"].get("difference"), 1)

    def test_unicode_params_only_and_no_pickle(self):
        self.assertEqual(read_params(unicode_npy(json.dumps(PARAMS))), PARAMS)
        with self.assertRaises(FnoOverlayError):
            read_params(npy_bytes([1], "<f8", (1,)))
        with self.assertRaises(FnoOverlayError):
            read_params(unicode_npy("not JSON"))

    def test_pair_validation_uses_timestamps_and_sample_parameters(self):
        truth = load_truth(self.gt)
        pred = load_sdf(self.samples / "r0301_onestep.npz")
        validate_pair(truth, pred, PARAMS)
        pred["steps"] = [0, 1]
        with self.assertRaises(FnoOverlayError):
            validate_pair(truth, pred, PARAMS)


if __name__ == "__main__":
    unittest.main()
