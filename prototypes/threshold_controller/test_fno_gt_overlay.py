"""Counterexamples for scalar-grid contour extraction and reference matching."""

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from fno_gt_overlay import FnoOverlayError, load_sdf, marching_squares, run
from test_mgn_geometry import npy_bytes


def write_sdf(path, *, steps=(1, 2), x=(-1.0, 0.0, 1.0), y=(-1.0, 1.0), bad_shape=False):
    fields = [v for depth in (0.6, 0.8) for yy in y for _xx in x for v in [yy + depth]]
    values = {
        "phi": npy_bytes(fields, "<f4", (2, len(y), len(x)) if not bad_shape else (2, 3, 2)),
        "step": npy_bytes(list(steps), "<i8", (len(steps),)),
        "x": npy_bytes(list(x), "<f8", (len(x),)),
        "y": npy_bytes(list(y), "<f8", (len(y),)),
    }
    with zipfile.ZipFile(path, "w") as z:
        for key, value in values.items():
            z.writestr(key + ".npy", value)


class ContourTests(unittest.TestCase):
    def test_all_sixteen_corner_patterns_cross_each_sign_change_edge_once(self):
        corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
        for mask in range(16):
            with self.subTest(mask=mask):
                vals = [-1 if mask & (1 << k) else 1 for k in range(4)]
                field = [vals[0], vals[1], vals[3], vals[2]]
                expected = set()
                for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                    if vals[a] * vals[b] < 0:
                        expected.add(tuple((u + v) / 2 for u, v in zip(corners[a], corners[b])))
                graph = marching_squares(field, [0, 1], [0, 1])
                self.assertEqual(set(graph["points"]), expected)
                self.assertEqual(2 * len(graph["lines"]), len(expected))

    def test_horizontal_level_uses_physical_y_not_row_index(self):
        g = marching_squares([-3, -3, 5, 5], [-2, 1], [-3, 5])
        self.assertEqual(len(g["lines"]), 1)
        self.assertTrue(all(abs(y) < 1e-12 for x, y in g["points"]))
        self.assertEqual(sorted(x for x, y in g["points"]), [-2, 1])

    def test_vertical_level_uses_x_and_y_in_correct_order(self):
        g = marching_squares([-2, 3, -2, 3], [-2, 3], [0, 4])
        self.assertTrue(all(abs(x) < 1e-12 for x, y in g["points"]))
        self.assertEqual(sorted(y for x, y in g["points"]), [0, 4])

    def test_solid_island_has_closed_boundary(self):
        g = marching_squares([1, 1, 1, 1, -1, 1, 1, 1, 1], [0, 1, 2], [0, 1, 2])
        self.assertEqual(g["diagnostics"]["components"], 1)
        self.assertEqual(g["diagnostics"]["open_endpoints"], 0)
        self.assertEqual(len(g["lines"]), 4)

    def test_saddle_uses_declared_low_connected_policy_without_crossing(self):
        g = marching_squares([-1, 1, 1, -1], [0, 1], [0, 1])
        edges = {frozenset(g["points"][i] for i in line) for line in g["lines"]}
        self.assertEqual(edges, {frozenset([(0.5, 0), (1, 0.5)]), frozenset([(0.5, 1), (0, 0.5)])})
        self.assertEqual(g["diagnostics"]["ambiguous_cells"], 1)

    def test_opposite_saddle_is_also_low_connected(self):
        g = marching_squares([1, -1, -1, 1], [0, 1], [0, 1])
        edges = {frozenset(g["points"][i] for i in line) for line in g["lines"]}
        self.assertEqual(edges, {frozenset([(0.5, 0), (0, 0.5)]), frozenset([(1, 0.5), (0.5, 1)])})

    def test_exact_zero_vertices_do_not_create_zero_length_segments(self):
        g = marching_squares([0, -1, 1, 0], [0, 1], [0, 1])
        self.assertEqual(len(g["lines"]), 1)
        self.assertEqual(set(g["points"]), {(0, 0), (1, 1)})

    def test_zero_plateau_is_reported_not_assumed_valid(self):
        g = marching_squares([0, 0, 0, 0], [0, 1], [0, 1])
        self.assertEqual(g["diagnostics"]["zero_plateau_cells"], 1)
        self.assertEqual(g["points"], [])

    def test_nonfinite_field_is_rejected(self):
        with self.assertRaises(FnoOverlayError):
            marching_squares([0, 1, float("nan"), -1], [0, 1], [0, 1])

    def test_unordered_axes_and_shape_mismatch_are_rejected(self):
        for field, x, y in [([-1, 1, -1, 1], [1, 0], [0, 1]), ([-1, 1], [0, 1], [0, 1])]:
            with self.assertRaises(FnoOverlayError):
                marching_squares(field, x, y)

    def test_numeric_3d_sdf_archive_is_loaded_without_numpy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sample.npz"
            write_sdf(p)
            g = load_sdf(p)
            self.assertEqual(g["shape"], (2, 2, 3))
            self.assertEqual(g["steps"], [1, 2])

    def test_archive_rejects_wrong_shape_and_missing_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sample.npz"
            for kw in [dict(bad_shape=True), dict(steps=(1, 3))]:
                write_sdf(p, **kw)
                with self.assertRaises(FnoOverlayError):
                    load_sdf(p)


PARAMS = dict(
    ion_flux=1,
    etchant_flux=2,
    oxygen_flux=3,
    ion_mean_energy=100,
    ion_exponent=100,
    mask_width=6,
    mask_height=6,
)


class FakeGT:
    def __init__(self, path):
        pass

    def close(self):
        pass

    def snapshot_numbers(self, rid):
        return [0, 1, 2]

    def meta(self, rid):
        return dict(actual_k=2, params=PARAMS)

    def frame(self, rid, k):
        d = [0, 0.5, 0.75][k]
        return dict(points=[(-1, -d), (1, -d)], lines=[(0, 1)], max_abs_z=0.0)


class FnoOverlayIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.samples = self.root / "samples"
        self.samples.mkdir()
        for mode in ["onestep", "rollout"]:
            write_sdf(self.samples / f"r0301_{mode}.npz")
        (self.samples / "README.json").write_text(
            json.dumps(
                {
                    "모델": "main_s1",
                    "데이터 구분": "validation",
                    "레시피": {"r0301": {"레시피 값": PARAMS}},
                }
            ),
            encoding="utf-8",
        )
        self.archive = self.root / "truth.zip"
        self.archive.write_bytes(b"fake for mocked loader")
        with (self.samples / "metrics_by_step.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["레시피", "방식", "스텝", "정답깊이(um)", "예측깊이(um)"]
            )
            w.writeheader()
            for mode in ["one-step", "rollout"]:
                for step, d in [(1, 0.5), (2, 0.75)]:
                    w.writerow(
                        {
                            "레시피": "r0301",
                            "방식": mode,
                            "스텝": f"t{step:03d}",
                            "정답깊이(um)": d,
                            "예측깊이(um)": 0.6 if step == 1 else 0.8,
                        }
                    )

    def go(self):
        with mock.patch("fno_gt_overlay.GroundTruth", FakeGT):
            return run(self.samples, self.archive, self.root / "out")

    def test_real_run_shape_and_scope_guards(self):
        out, s = self.go()
        self.assertEqual(s["status"], "ok")
        self.assertEqual(s["sample_recipe_count"], 1)
        self.assertEqual(s["prediction_frames_checked"], 4)
        self.assertEqual(len(s["cases"]), 2)
        self.assertFalse(s["reference_verified"])
        self.assertFalse(s["training_started"])
        self.assertNotIn("invalid_geometry_count", s)
        self.assertTrue((out / "report.html").is_file())
        self.assertTrue((out / "per_step_diagnostics.csv").is_file())
        case = s["cases"][0]
        self.assertAlmostEqual(case["recomputed_pred_depth_um"], 0.8, places=6)
        self.assertAlmostEqual(case["gt_depth_um"], 0.75)

    def test_recipe_parameters_must_match_ground_truth(self):
        p = self.samples / "README.json"
        v = json.loads(p.read_text())
        v["레시피"]["r0301"]["레시피 값"]["ion_flux"] = 99
        p.write_text(json.dumps(v), encoding="utf-8")
        out, s = self.go()
        self.assertEqual(s["status"], "failed")
        self.assertTrue(s["errors"])
        self.assertFalse(s["cases"])

    def test_missing_prediction_is_failure_not_old_or_zero_result(self):
        (self.samples / "r0301_rollout.npz").unlink()
        out, s = self.go()
        self.assertEqual(s["status"], "partial")
        self.assertEqual(len(s["cases"]), 1)
        self.assertTrue(s["errors"])

    def test_missing_reference_snapshot_is_not_a_successful_alignment(self):
        with mock.patch.object(FakeGT, "snapshot_numbers", return_value=[0, 1]):
            out, s = self.go()
        self.assertEqual(s["status"], "failed")
        self.assertFalse(s["cases"])

    def test_inputs_are_unchanged_and_runs_are_new(self):
        before = {p.name: p.read_bytes() for p in self.samples.iterdir()}
        out1, _ = self.go()
        out2, _ = self.go()
        self.assertNotEqual(out1, out2)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.samples.iterdir()})


if __name__ == "__main__":
    unittest.main()
