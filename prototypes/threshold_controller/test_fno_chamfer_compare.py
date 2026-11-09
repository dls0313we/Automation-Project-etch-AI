"""Analytic Chamfer checks and integration checks for the independent FNO Chamfer diagnostic."""

import csv
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from fno_chamfer_compare import (
    METHOD,
    SPACING_UM,
    PointIndex,
    SegmentIndex,
    chamfer_terms,
    point_segment_distance,
    polylines,
    resample,
    run,
)
from fno_gt_overlay import marching_squares
from test_fno_sdf_compare import PARAMS, write_arrays


def line(x0, y0, x1, y1):
    return [dict(vertices=[(x0, y0), (x1, y1)], closed=False)]


class ResampleTests(unittest.TestCase):
    def test_spacing_and_endpoint_rule(self):
        pts = resample([(0, 0), (1, 0)], False, SPACING_UM)
        self.assertEqual(len(pts), 11)  # 0, 0.1, ..., 1.0 (end coincides with last sample)
        self.assertAlmostEqual(pts[-1][0], 1.0)
        pts = resample([(0, 0), (1.05, 0)], False, SPACING_UM)
        self.assertEqual(len(pts), 12)  # 11 regular samples + the endpoint
        self.assertAlmostEqual(pts[-2][0], 1.0)
        self.assertAlmostEqual(pts[-1][0], 1.05)

    def test_samples_follow_arc_length_across_vertices(self):
        pts = resample([(0, 0), (0.25, 0), (0.25, 0.25)], False, SPACING_UM)
        self.assertEqual(len(pts), 6)  # length 0.5
        self.assertAlmostEqual(pts[3][0], 0.25)
        self.assertAlmostEqual(pts[3][1], 0.05)

    def test_closed_loop_does_not_repeat_start(self):
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        pts = resample(square, True, SPACING_UM)
        self.assertEqual(len(pts), 40)  # perimeter 4
        self.assertNotAlmostEqual(math.dist(pts[0], pts[-1]), 0)


class DistanceTests(unittest.TestCase):
    def test_point_segment_distance(self):
        self.assertAlmostEqual(point_segment_distance((0.5, 1), (0, 0), (1, 0)), 1)
        self.assertAlmostEqual(point_segment_distance((2, 0), (0, 0), (1, 0)), 1)
        self.assertAlmostEqual(point_segment_distance((3, 4), (0, 0), (0, 0)), 5)

    def test_spatial_indexes_equal_brute_force(self):
        rng = random.Random(7)
        segs = []
        for _ in range(300):
            a = (rng.uniform(-10, 10), rng.uniform(-10, 10))
            segs.append((a, (a[0] + rng.uniform(-0.7, 0.7), a[1] + rng.uniform(-0.7, 0.7))))
        pts = [s[0] for s in segs]
        si, pi = SegmentIndex(segs), PointIndex(pts)
        for _ in range(300):
            q = (rng.uniform(-14, 14), rng.uniform(-14, 14))
            brute = min(point_segment_distance(q, a, b) for a, b in segs)
            self.assertAlmostEqual(si.nearest(q), brute, places=12)
            self.assertAlmostEqual(pi.nearest(q), min(math.dist(q, p) for p in pts), places=12)


class AnalyticChamferTests(unittest.TestCase):
    def test_identical_contours_are_zero(self):
        t = chamfer_terms(line(-1, 0, 1, 0), line(-1, 0, 1, 0))
        self.assertAlmostEqual(t["primary_um"], 0, places=12)

    def test_parallel_lines_distinguish_pooled_mean_from_team_sum(self):
        t = chamfer_terms(line(0, 0, 1, 0), line(0, -0.3, 1, -0.3))
        self.assertAlmostEqual(t["primary_um"], 0.3)
        self.assertAlmostEqual(t["alternatives"]["symmetric_mean_point_to_polyline"], 0.3)
        self.assertAlmostEqual(t["alternatives"]["pooled_point_to_point"], 0.3)
        self.assertAlmostEqual(t["alternatives"]["team_sum_point_to_point"], 0.6)

    def test_unequal_lengths_pooled_vs_symmetric_closed_form(self):
        # truth: y=0, x in [0,2] (21 samples); prediction: y=1, x in [0,1] (11 samples)
        t = chamfer_terms(line(0, 0, 2, 0), line(0, 1, 1, 1))
        d_truth = [1 if 0.1 * k <= 1 else math.hypot(0.1 * k - 1, 1) for k in range(21)]
        self.assertEqual((t["n_truth_points"], t["n_pred_points"]), (21, 11))
        self.assertAlmostEqual(t["mean_truth_to_pred_um"], sum(d_truth) / 21)
        self.assertAlmostEqual(t["mean_pred_to_truth_um"], 1)
        self.assertAlmostEqual(t["primary_um"], (sum(d_truth) + 11) / 32)
        self.assertAlmostEqual(
            t["alternatives"]["symmetric_mean_point_to_polyline"], (sum(d_truth) / 21 + 1) / 2
        )
        self.assertNotAlmostEqual(
            t["primary_um"], t["alternatives"]["symmetric_mean_point_to_polyline"], places=4
        )

    def test_point_to_polyline_is_not_point_to_point(self):
        # offset samples by half spacing: distance to the polyline is 0, to samples is 0.05
        t = chamfer_terms(line(0, 0, 1, 0), line(0.05, 0, 0.95, 0))
        self.assertAlmostEqual(t["mean_pred_to_truth_um"], 0)
        # truth samples 0, 0.1, ..., 1.0: only the two ends are off the shorter polyline (0.05)
        self.assertAlmostEqual(t["mean_truth_to_pred_um"], 0.1 / 11)
        self.assertGreater(t["alternatives"]["pooled_point_to_point"], 0.04)

    def test_concentric_circle_sdf_contours_through_extraction(self):
        h = 0.05
        g = [-3 + h * i for i in range(121)]

        def field(r):
            return [math.hypot(xx, yy) - r for yy in g for xx in g]

        a, b = (polylines(marching_squares(field(r), g, g)) for r in (1.0, 1.5))
        self.assertEqual([p["closed"] for p in a + b], [True, True])
        t = chamfer_terms(a, b)
        self.assertAlmostEqual(t["primary_um"], 0.5, delta=2e-3)
        self.assertAlmostEqual(t["n_truth_points"], 2 * math.pi / SPACING_UM, delta=2)


class ChainingTests(unittest.TestCase):
    def test_shuffled_segments_rebuild_one_open_polyline(self):
        pts = [(0.1 * k, (0.1 * k) ** 2) for k in range(30)]
        lines = [(k, k + 1) for k in range(29)]
        random.Random(1).shuffle(lines)
        lines = [(b, a) if i % 2 else (a, b) for i, (a, b) in enumerate(lines)]
        polys = polylines(dict(points=pts, lines=lines))
        self.assertEqual(len(polys), 1)
        self.assertFalse(polys[0]["closed"])
        self.assertEqual(sorted(polys[0]["vertices"]), sorted(pts))

    def test_junction_is_split_and_cycles_are_closed(self):
        pts = [(0, 0), (1, 0), (2, 0), (1, 1), (5, 5), (6, 5), (6, 6)]
        lines = [(0, 1), (1, 2), (1, 3), (4, 5), (5, 6), (6, 4)]
        polys = polylines(dict(points=pts, lines=lines))
        self.assertEqual(sorted(p["closed"] for p in polys), [False, False, False, True])
        self.assertEqual(sum(len(p["vertices"]) - 1 for p in polys if not p["closed"]), 3)


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples, self.truth = self.root / "samples", self.root / "sdf"
        self.samples.mkdir()
        self.truth.mkdir()
        write_arrays(self.truth / "r0301.npz", truth=True)
        for mode in ["onestep", "rollout"]:
            write_arrays(self.samples / f"r0301_{mode}.npz", truth=False)
        (self.samples / "README.json").write_text(
            json.dumps({"데이터 구분": "validation", "레시피": {"r0301": {"레시피 값": PARAMS}}}),
            encoding="utf-8",
        )
        self.rows = [
            {"레시피": "r0301", "방식": m, "스텝": f"t{s:03d}", "Chamfer(um)": 0, "IoU": 0.5}
            for m in ["one-step", "rollout"]
            for s in [1, 2]
        ]
        self.write_csv()

    def write_csv(self):
        with (self.samples / "metrics_by_step.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)

    def go(self):
        return run(self.samples, self.truth, self.root / "results")

    def test_identical_synthetic_prediction_matches_and_keeps_sources(self):
        before = {str(p): p.read_bytes() for p in [*self.truth.iterdir(), *self.samples.iterdir()]}
        out, s = self.go()
        self.assertEqual(s["status"], "ok")
        self.assertEqual(s["frames_checked"], 4)
        self.assertEqual(s["comparison_counts"], {"match": 4})
        self.assertFalse(s["reference_verified"])
        self.assertFalse(s["exact_provider_reproduction"])
        self.assertFalse(s["training_started"] or s["active_model_changed"])
        self.assertIn("IoU", s["excluded"])
        self.assertEqual(s["method"], METHOD)
        self.assertTrue((out / "comparisons.csv").is_file())
        self.assertTrue((out / "report.html").is_file())
        with (out / "comparisons.csv").open(encoding="utf-8-sig") as f:
            self.assertNotIn("IoU", next(csv.reader(f)))
        self.assertEqual(before, {p: Path(p).read_bytes() for p in before})

    def test_disagreement_is_reported_not_fitted(self):
        self.rows[0]["Chamfer(um)"] = 0.25
        self.write_csv()
        _, s = self.go()
        self.assertEqual(s["status"], "differences")
        self.assertEqual(s["comparison_counts"], {"difference": 1, "match": 3})
        self.assertAlmostEqual(s["stats"]["primary"]["max_abs_diff_um"], 0.25)

    def test_missing_provided_value_is_not_zero(self):
        self.rows[1]["Chamfer(um)"] = ""
        self.write_csv()
        _, s = self.go()
        self.assertEqual(s["comparison_counts"].get("provided_missing"), 1)
        self.assertEqual(s["status"], "differences")
        self.rows[1]["Chamfer(um)"] = "nan"
        self.write_csv()
        _, s = self.go()
        self.assertEqual(s["status"], "incomplete")
        self.assertTrue(any("Nonnumeric" in e for e in s["errors"]))

    def test_bad_reference_stops_instead_of_reusing_results(self):
        write_arrays(self.truth / "r0301.npz", truth=True, params=dict(PARAMS, ion_flux=9))
        _, s = self.go()
        self.assertEqual(s["status"], "incomplete")
        self.assertEqual(s["frames_checked"], 0)


if __name__ == "__main__":
    unittest.main()
