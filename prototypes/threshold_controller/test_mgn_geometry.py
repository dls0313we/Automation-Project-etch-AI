"""Geometric counterexamples and numeric-NPY fixtures; no NumPy dependency."""

import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from mgn_geometry import GeometryFormatError, check_frame, read_npy, scan_dataset


def npy_bytes(rows, dtype, shape, *, version=1, fortran=False):
    header = repr(dict(descr=dtype, fortran_order=fortran, shape=shape)).encode("latin1")
    prefix = b"\x93NUMPY" + bytes([version, 0])
    header += b" " * ((-(len(prefix) + (2 if version == 1 else 4) + len(header) + 1)) % 64) + b"\n"
    length = struct.pack("<H" if version == 1 else "<I", len(header))
    codes = {"f4": "f", "f8": "d", "i8": "q", "i4": "i", "i1": "b"}
    endian = ">" if dtype[0] == ">" else "<"
    data = struct.pack(endian + codes[dtype[1:]] * len(rows), *rows)
    return prefix + length + header + data


def write_npz(path, frames):
    """frames = {step: (points, edges)}; material codes are valid zeroes."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for step, (points, edges) in frames.items():
            for key, data in {
                "pred": npy_bytes([v for p in points for v in p], "<f4", (len(points), 2)),
                "lines": npy_bytes([v for e in edges for v in e], "<i8", (len(edges), 2)),
                "mat": npy_bytes([0] * len(points), "|i1", (len(points),)),
            }.items():
                z.writestr(f"{key}_{step:03d}.npy", data)


SQUARE = ([(0, 0), (1, 0), (1, 1), (0, 1)], [(0, 1), (1, 2), (2, 3), (3, 0)])
BOW = ([(0, 0), (2, 2), (0, 2), (2, 0)], [(0, 1), (1, 2), (2, 3)])


class GeometryTests(unittest.TestCase):
    def frame(self, points, edges, eps=1e-8):
        return check_frame(points, edges, [0] * len(points), epsilon_um=eps)

    def test_closed_square_shared_endpoints_are_valid(self):
        r = self.frame(*SQUARE)
        self.assertFalse(r["invalid"])
        self.assertEqual(r["proper_crossing_pair_count"], 0)
        self.assertEqual(r["nonadjacent_touch_pair_count"], 0)

    def test_bow_tie_detects_non_adjacent_crossing(self):
        r = self.frame(*BOW)
        self.assertTrue(r["invalid"])
        self.assertEqual(r["proper_crossing_pair_count"], 1)
        self.assertEqual(r["pair_examples"][0]["edge_ids"], [0, 2])
        self.assertEqual(r["pair_examples"][0]["segments"], [[[0, 0], [2, 2]], [[0, 2], [2, 0]]])

    def test_adjacent_collinear_overlap_is_not_ignored(self):
        r = self.frame([(0, 0), (2, 0), (1, 0)], [(0, 1), (1, 2)])
        self.assertEqual(r["overlap_pair_count"], 1)
        self.assertTrue(r["invalid"])

    def test_collinear_continuation_is_valid(self):
        self.assertFalse(self.frame([(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 2)])["invalid"])

    def test_parallel_and_disjoint_segments_are_valid(self):
        r = self.frame([(0, 0), (2, 0), (0, 1), (2, 1)], [(0, 1), (2, 3)])
        self.assertFalse(r["invalid"])

    def test_nonadjacent_endpoint_touch_is_separate_diagnostic(self):
        r = self.frame([(0, 0), (1, 0), (1, 0), (1, 1)], [(0, 1), (2, 3)])
        self.assertEqual(r["nonadjacent_touch_pair_count"], 1)
        self.assertFalse(r["invalid"])

    def test_t_junction_is_recorded_as_touch(self):
        r = self.frame([(0, 0), (2, 0), (1, 0), (1, 1)], [(0, 1), (2, 3)])
        self.assertEqual(r["nonadjacent_touch_pair_count"], 1)
        self.assertFalse(r["invalid"])

    def test_duplicate_reversed_edge_is_invalid(self):
        r = self.frame([(0, 0), (1, 0)], [(0, 1), (1, 0)])
        self.assertTrue(r["invalid"])
        self.assertEqual(r["structural_issue_counts"]["duplicate_edge"], 1)
        self.assertEqual(r["overlap_pair_count"], 0)
        self.assertEqual(r["duplicate_edge_pair_count"], 1)

    def test_three_copies_count_extra_edges_and_duplicate_pairs_separately(self):
        r = self.frame([(0, 0), (1, 0)], [(0, 1), (1, 0), (0, 1)])
        self.assertEqual(r["duplicate_edge_count"], 2)
        self.assertEqual(r["duplicate_edge_pair_count"], 3)
        self.assertEqual(r["overlap_pair_count"], 0)

    def test_same_coordinates_with_different_node_ids_are_geometric_overlap(self):
        r = self.frame([(0, 0), (2, 0), (0, 0), (2, 0)], [(0, 1), (2, 3)])
        self.assertEqual(r["overlap_pair_count"], 1)
        self.assertEqual(r["duplicate_edge_count"], 0)
        self.assertTrue(r["invalid"])

    def test_frame_union_does_not_add_crossing_and_duplicate_categories(self):
        duplicate = ([(0, 0), (1, 0)], [(0, 1), (1, 0)])
        both = (BOW[0], BOW[1] + [(1, 0)])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for mode in ["onestep", "rollout"]:
                write_npz(p / f"r0300_{mode}.npz", {1: SQUARE, 2: BOW, 3: duplicate, 4: both})
            result = scan_dataset(p, [dict(rid="r0300", split="validation", steps=[1, 2, 3, 4])])
            for g in result["groups"].values():
                self.assertEqual(g["invalid_geometry_count"], 3)
                self.assertEqual(g["self_intersection_frame_count"], 2)
                self.assertEqual(g["duplicate_edge_frame_count"], 2)
                self.assertEqual(g["overlap_frame_count"], 0)

    def test_zero_length_edge_is_invalid(self):
        r = self.frame([(0, 0), (0, 0)], [(0, 1)])
        self.assertTrue(r["invalid"])
        self.assertEqual(r["structural_issue_counts"]["zero_length_edge"], 1)

    def test_bad_index_is_invalid_without_indexing_crash(self):
        r = self.frame([(0, 0), (1, 0)], [(0, 5)])
        self.assertTrue(r["invalid"])
        self.assertEqual(r["structural_issue_counts"]["invalid_edge_index"], 1)

    def test_nan_coordinate_is_invalid(self):
        r = self.frame([(float("nan"), 0), (1, 0)], [(0, 1)])
        self.assertTrue(r["invalid"])

    def test_pair_example_limit_does_not_truncate_counts(self):
        points = [(0, i) for i in range(12)] + [(1, 11 - i) for i in range(12)]
        r = check_frame(points, [(i, 12 + i) for i in range(12)], [0] * 24, max_examples=2)
        self.assertEqual(r["proper_crossing_pair_count"], 66)
        self.assertEqual(len(r["pair_examples"]), 2)
        self.assertTrue(r["examples_truncated"])

    def test_narrow_true_crossing_is_detected(self):
        r = self.frame([(0, 0), (1, 1e-5), (0, 1e-5), (1, 0)], [(0, 1), (2, 3)])
        self.assertEqual(r["proper_crossing_pair_count"], 1)

    def test_npy_reads_versions_and_endianness(self):
        for version in [1, 2, 3]:
            for dtype in ["<f4", ">f4", "<f8", ">f8"]:
                with self.subTest(version=version, dtype=dtype):
                    shape, data, descr = read_npy(
                        npy_bytes([1, 2, 3, 4], dtype, (2, 2), version=version)
                    )
                    self.assertEqual(shape, (2, 2))
                    self.assertEqual(data, [1, 2, 3, 4])
                    self.assertEqual(descr, dtype)

    def test_npy_fortran_is_rejected_explicitly(self):
        with self.assertRaises(GeometryFormatError):
            read_npy(npy_bytes([1, 2], "<f4", (1, 2), fortran=True))

    def test_npy_object_data_is_never_unpickled(self):
        header = b"{'descr':'|O','fortran_order':False,'shape':(1,)}\n"
        value = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"not a pickle"
        with self.assertRaises(GeometryFormatError):
            read_npy(value)

    def test_npy_truncated_or_extra_data_is_rejected(self):
        value = npy_bytes([1, 2], "<f4", (1, 2))
        for bad in [value[:-1], value + b"x", b"bad"]:
            with self.subTest(length=len(bad)), self.assertRaises(GeometryFormatError):
                read_npy(bad)

    def test_scan_counts_frames_not_crossing_pairs_and_splits_stay_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            expected = [
                dict(rid="r0300", split="validation", steps=[1, 2]),
                dict(rid="r0340", split="test", steps=[1, 2]),
            ]
            for rid in ["r0300", "r0340"]:
                for mode in ["onestep", "rollout"]:
                    write_npz(
                        p / f"{rid}_{mode}.npz", {1: SQUARE, 2: BOW if rid == "r0300" else SQUARE}
                    )
            result = scan_dataset(p, expected)
            val = result["groups"]["validation/rollout"]
            self.assertEqual(val["invalid_geometry_count"], 1)
            self.assertEqual(val["affected_recipe_ids"], ["r0300"])
            self.assertEqual(result["groups"]["test/rollout"]["invalid_geometry_count"], 0)
            self.assertEqual(val["frames_checked"], 2)
            self.assertTrue(result["source_files_unchanged"])

    def test_missing_prediction_file_keeps_count_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = scan_dataset(tmp, [dict(rid="r0300", split="validation", steps=[1])])
            for mode in ["onestep", "rollout"]:
                self.assertIsNone(result["groups"][f"validation/{mode}"]["invalid_geometry_count"])
                self.assertFalse(result["groups"][f"validation/{mode}"]["complete"])

    def test_missing_frame_or_member_keeps_count_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            write_npz(p / "r0300_onestep.npz", {1: SQUARE})
            result = scan_dataset(p, [dict(rid="r0300", split="validation", steps=[1, 2])])
            group = result["groups"]["validation/onestep"]
            self.assertIsNone(group["invalid_geometry_count"])
            self.assertEqual(group["frames_checked"], 1)

    def test_test_read_failure_does_not_erase_validation_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for mode in ["onestep", "rollout"]:
                write_npz(p / f"r0300_{mode}.npz", {1: BOW})
            result = scan_dataset(
                p,
                [
                    dict(rid="r0300", split="validation", steps=[1]),
                    dict(rid="r0340", split="test", steps=[1]),
                ],
            )
            self.assertEqual(result["groups"]["validation/onestep"]["invalid_geometry_count"], 1)
            self.assertIsNone(result["groups"]["test/onestep"]["invalid_geometry_count"])


class GeometryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        from test_mgn_adapter import build

        self.folder = build(self.root / "mgn")
        from fno_adapter import Problems
        from mgn_adapter import load_steps

        self.series = load_steps(self.folder, Problems())
        for rid, items in self.series.items():
            for mode in ["onestep", "rollout"]:
                # A single validation crossing, independently of the test data.
                frames = {
                    r["step"]: BOW if rid == "r0300" and r["step"] == 1 else SQUARE for r in items
                }
                write_npz(self.folder / f"{rid}_{mode}.npz", frames)

    def run_adapter(self):
        from mgn_adapter import DEFAULT_THRESHOLDS, run

        return run(self.folder, DEFAULT_THRESHOLDS, self.root / "out", check_geometry=True)

    def test_complete_scan_populates_metric_and_retains_hold_reference_guard(self):
        out, text, result = self.run_adapter()
        for mode in ["onestep", "rollout"]:
            p = result["inputs"][mode]
            self.assertEqual(p["metrics"]["invalid_geometry_count"], 1)
            self.assertNotIn("invalid_geometry_count", p["not_provided"])
            self.assertNotIn("self_intersection", p["not_provided"])
            self.assertIs(p["data_check"]["reference_verified"], False)
            self.assertEqual(result["decisions"][mode]["action"], "HOLD")
            self.assertNotIn(
                "is unmeasured",
                " ".join(result["decisions"][mode]["remaining_blockers_even_if_approved"]),
            )
        self.assertTrue((out / "geometry_frames.csv").is_file())
        self.assertTrue((out / "geometry_diagnostics.json").is_file())
        self.assertIn("선분 형상 검사", text)

    def test_test_geometry_changes_do_not_change_validation_metrics(self):
        _, _, before = self.run_adapter()
        for rid, items in self.series.items():
            if items[0]["split"] == "test":
                for mode in ["onestep", "rollout"]:
                    write_npz(self.folder / f"{rid}_{mode}.npz", {r["step"]: BOW for r in items})
        _, _, after = self.run_adapter()
        for mode in ["onestep", "rollout"]:
            self.assertEqual(before["inputs"][mode]["metrics"], after["inputs"][mode]["metrics"])
            self.assertGreater(
                after["summary"][f"test/{mode}"]["geometry"]["invalid_geometry_count"], 0
            )

    def test_corrupt_frame_stays_unmeasured_without_hiding_known_anomaly(self):
        write_npz(self.folder / "r0301_rollout.npz", {1: SQUARE})
        _, _, result = self.run_adapter()
        self.assertNotIn("invalid_geometry_count", result["inputs"]["rollout"]["metrics"])
        g = result["geometry"]["groups"]["validation/rollout"]
        self.assertIsNone(g["invalid_geometry_count"])
        self.assertEqual(g["known_invalid_frame_count"], 1)
        self.assertFalse(g["complete"])

    def test_review_carries_measured_count_and_ids_and_fno_stays_unmeasured(self):
        import run_review
        from test_fno_adapter import build

        metrics, samples = build(self.root / "fno")
        args = run_review.parser().parse_args(
            [
                "--fno-metrics",
                str(metrics),
                "--fno-samples",
                str(samples),
                "--mgn-input",
                str(self.folder),
                "--check-mgn-geometry",
            ]
        )
        out, summary = run_review.run(args, self.root / "review")
        for c in summary["combinations"]:
            if c["model_family"] == "MGN":
                self.assertEqual(c["metrics"]["invalid_geometry_count"], 1)
                row = next(
                    r for r in c["provisional"]["rows"] if r["metric"] == "invalid_geometry_count"
                )
                self.assertEqual(row["recipes_over_ids"], ["r0300"])
                self.assertEqual(row["result"], "초과")
            else:
                self.assertNotIn("invalid_geometry_count", c["metrics"])
        page = (out / "report.html").read_text(encoding="utf-8")
        self.assertIn("MGN 선분 형상 검사", page)
        self.assertNotIn("자기교차 건수는 제공되지 않아 미측정입니다.", page)
        self.assertFalse(summary["training_started"])
        self.assertFalse(summary["active_model_changed"])

    def test_review_cli_returns_failure_when_requested_geometry_is_incomplete(self):
        import contextlib
        import io
        from unittest import mock

        import run_review
        from test_fno_adapter import build

        metrics, samples = build(self.root / "fno")
        (self.folder / "r0300_rollout.npz").unlink()
        args = run_review.parser().parse_args(
            [
                "--fno-metrics",
                str(metrics),
                "--fno-samples",
                str(samples),
                "--mgn-input",
                str(self.folder),
                "--check-mgn-geometry",
                "--output-root",
                str(self.root / "review"),
            ]
        )
        with mock.patch.object(run_review, "parser") as parser:
            parser.return_value.parse_args.return_value = args
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
                run_review.main()
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(len(list((self.root / "review").glob("*/report.html"))), 1)


if __name__ == "__main__":
    unittest.main()
