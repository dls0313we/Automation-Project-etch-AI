"""Synthetic VTP/NPZ fixtures for the ground-truth overlay; no NumPy, no network."""

import base64
import csv
import hashlib
import json
import re
import struct
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

import mgn_geometry as geo
from mgn_gt_overlay import (
    GroundTruth,
    OverlayError,
    point_chamfer,
    point_materials,
    read_vtp,
    run,
    select_cases,
)
from test_mgn_geometry import write_npz


def vtp_bytes(
    points, lines, materials, *, compressor="vtkZLibDataCompressor", polys=0, corrupt=False
):
    """Minimal VTK XML PolyData with base64 zlib appended arrays (like the real snapshots)."""
    arrays = [
        ("CellData", "MaterialIds", "Float32", 1, struct.pack(f"<{len(materials)}f", *materials)),
        (
            "Points",
            "Points",
            "Float32",
            3,
            struct.pack(f"<{3 * len(points)}f", *[v for x, y in points for v in (x, y, 0.0)]),
        ),
        (
            "Lines",
            "connectivity",
            "Int64",
            1,
            struct.pack(f"<{2 * len(lines)}q", *[v for e in lines for v in e]),
        ),
        (
            "Lines",
            "offsets",
            "Int64",
            1,
            struct.pack(f"<{len(lines)}q", *range(2, 2 * len(lines) + 1, 2)),
        ),
    ]
    payload, tags, offset = "", {}, 0
    for section, name, typ, comp, raw in arrays:
        c = zlib.compress(raw)
        chunk = (
            base64.b64encode(struct.pack("<4I", 1, len(raw), len(raw), len(c))).decode()
            + base64.b64encode(c).decode()
        )
        tags.setdefault(section, []).append(
            f'<DataArray type="{typ}" Name="{name}" NumberOfComponents="{comp}" '
            f'format="appended" offset="{offset}"/>'
        )
        payload += chunk
        offset += len(chunk)
    if corrupt:
        payload = payload[:40] + "!!!!" + payload[44:]
    comp_attr = f' compressor="{compressor}"' if compressor else ""
    xml = (
        f'<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian" '
        f'header_type="UInt32"{comp_attr}>\n<PolyData>\n'
        f'<Piece NumberOfPoints="{len(points)}" NumberOfVerts="0" NumberOfLines="{len(lines)}" '
        f'NumberOfStrips="0" NumberOfPolys="{polys}">\n<PointData>\n</PointData>\n'
        f"<CellData>{''.join(tags['CellData'])}</CellData>\n<Points>{''.join(tags['Points'])}</Points>\n"
        f"<Lines>{''.join(tags['Lines'])}</Lines>\n</Piece>\n</PolyData>\n"
        f'<AppendedData encoding="base64">\n   _{payload}\n  </AppendedData>\n</VTKFile>\n'
    )
    return xml.encode()


def shape(k):
    """Open trench outline whose depth grows with the snapshot number."""
    d = 0.5 + k
    return [(0, 0), (1, 0), (1, -d), (2, -d), (2, 0), (3, 0)], [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ]


def with_crossings(points, lines, n):
    """Add n short vertical segments crossing the first (top-left) segment."""
    pts, lns = list(points), list(lines)
    for i in range(n):
        x = 0.2 + 0.15 * i
        pts += [(x, 0.3), (x, -0.3)]
        lns.append((len(pts) - 2, len(pts) - 1))
    return pts, lns


RECIPES = {"r0300": ("val", 2), "r0301": ("val", 2), "r0340": ("test", 2)}


def build(folder, *, gt_crossing=False, bad_n_gt=False, drop_snapshot=False, reverse_zip=True):
    folder = Path(folder)
    mgn = folder / "mgn"
    mgn.mkdir(parents=True)
    pred = {}  # (rid, mode, step) -> (points, lines)
    for rid, (_split, n) in RECIPES.items():
        for mode in ("onestep", "rollout"):
            for k in range(1, n + 1):
                pred[(rid, mode, k)] = shape(k)
    pred[("r0300", "onestep", 1)] = with_crossings(*shape(1), 1)
    pred[("r0300", "rollout", 2)] = with_crossings(*shape(2), 2)
    p, ln = shape(1)
    pred[("r0301", "onestep", 1)] = (p, ln + [ln[0]])  # duplicate edge only
    pred[("r0340", "onestep", 1)] = with_crossings(
        *shape(1), 3
    )  # test: more crossings, must be ignored
    gt_frames = {(rid, k): shape(k) for rid, (_s, n) in RECIPES.items() for k in range(0, n + 1)}
    if gt_crossing:
        gt_frames[("r0300", 1)] = with_crossings(*shape(1), 1)
    steps_rows, summary_rows = [], []
    for rid, (split, n) in RECIPES.items():
        for mode in ("onestep", "rollout"):
            write_npz(mgn / f"{rid}_{mode}.npz", {k: pred[(rid, mode, k)] for k in range(1, n + 1)})
        for k in range(1, n + 1):
            steps_rows.append(
                dict(
                    rid=rid,
                    split=split,
                    step=k,
                    onestep_chamfer=point_chamfer(
                        pred[(rid, "onestep", k)][0], gt_frames[(rid, k)][0]
                    ),
                    rollout_chamfer=point_chamfer(
                        pred[(rid, "rollout", k)][0], gt_frames[(rid, k)][0]
                    ),
                )
            )
        n_gt = len(gt_frames[(rid, n)][0]) + (1 if bad_n_gt and rid == "r0300" else 0)
        summary_rows.append(
            dict(
                rid=rid,
                split=split,
                last_step=n,
                onestep_chamfer=0,
                rollout_chamfer=0,
                n_pred_points=len(pred[(rid, "onestep", n)][0]),
                n_gt_points=n_gt,
            )
        )
    for name, cols, rows in [
        (
            "metrics_by_step.csv",
            ["rid", "split", "step", "onestep_chamfer", "rollout_chamfer"],
            steps_rows,
        ),
        (
            "summary_last_step.csv",
            [
                "rid",
                "split",
                "last_step",
                "onestep_chamfer",
                "rollout_chamfer",
                "n_pred_points",
                "n_gt_points",
            ],
            summary_rows,
        ),
    ]:
        with (mgn / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    members = []
    for rid, (_s, n) in RECIPES.items():
        members.append(
            (f"{rid}/meta.json", json.dumps(dict(actual_k=n, final_depth_um=0.5 + n)).encode())
        )
        for k in range(0, n + 1):
            if drop_snapshot and rid == "r0300" and k == 2:
                continue
            pts, lns = gt_frames[(rid, k)]
            members.append(
                (
                    f"{rid}/snapshots/t{k:03d}.vtp",
                    vtp_bytes(pts, lns, [10.0 if i else 0.0 for i in range(len(lns))]),
                )
            )
    gt_zip = folder / "etch-raw-n400.zip"
    with zipfile.ZipFile(gt_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in reversed(members) if reverse_zip else members:
            z.writestr(name, data)
    return mgn, gt_zip


class VtpTests(unittest.TestCase):
    def test_roundtrip_points_lines_materials(self):
        pts, lns = shape(1)
        v = read_vtp(vtp_bytes(pts, lns, [0, 10, 10, 10, 0]))
        self.assertEqual(v["lines"], lns)
        self.assertEqual(v["cell_materials"], [0, 10, 10, 10, 0])
        self.assertEqual(
            [tuple(round(c, 5) for c in p) for p in v["points"]],
            [tuple(map(float, p)) for p in pts],
        )
        self.assertEqual(v["max_abs_z"], 0.0)

    def test_unsupported_or_corrupt_vtp_rejected(self):
        pts, lns = shape(1)
        for kwargs in [dict(compressor=None), dict(polys=1), dict(corrupt=True)]:
            with self.subTest(**kwargs), self.assertRaises(OverlayError):
                read_vtp(vtp_bytes(pts, lns, [0] * len(lns), **kwargs))

    def test_material_mapping_and_unknown_codes_flagged(self):
        mats, mixed = point_materials(3, [(0, 1), (1, 2)], [0, 10])
        self.assertEqual(mats, [0, 0, 1])
        self.assertEqual(mixed, 1)
        mats, _ = point_materials(3, [(0, 1)], [7])
        self.assertEqual(mats, [7, 7, -1])
        r = geo.check_frame([(0, 0), (1, 0), (2, 0)], [(0, 1)], mats)
        self.assertIn("material_code", r["structural_issue_counts"])


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def go(self, **kw):
        mgn, gt_zip = build(self.dir / "in", **kw)
        return mgn, gt_zip, *run(mgn, gt_zip, self.dir / "out")

    def test_snapshots_found_by_number_not_zip_order(self):
        _, gt_zip = build(self.dir / "in", reverse_zip=True)
        gt = GroundTruth(gt_zip)
        try:
            names = [i.filename for i in gt.zip.infolist()]
            self.assertLess(
                names.index("r0300/snapshots/t002.vtp"), names.index("r0300/snapshots/t000.vtp")
            )
            self.assertEqual(gt.snapshot_numbers("r0300"), [0, 1, 2])
            self.assertEqual(round(-min(y for _x, y in gt.frame("r0300", 1)["points"]), 5), 1.5)
        finally:
            gt.close()

    def test_full_run_selects_validation_cases_and_checks_both(self):
        mgn, gt_zip, out, s = self.go()
        self.assertEqual(s["status"], "ok")
        self.assertEqual(s["mapping"]["recipes_ok"], 2)
        self.assertEqual(
            set(s["cases"]),
            {"crossing_onestep", "crossing_rollout", "duplicate_only", "clean_reference"},
        )
        self.assertTrue(
            all(c["rid"] != "r0340" for c in s["cases"].values())
        )  # test never selected
        c = s["cases"]
        self.assertEqual(
            (c["crossing_onestep"]["rid"], c["crossing_onestep"]["step"]), ("r0300", 1)
        )
        self.assertEqual(c["crossing_rollout"]["prediction_check"]["proper_crossing_pair_count"], 2)
        self.assertEqual(c["duplicate_only"]["prediction_check"]["duplicate_edge_count"], 1)
        self.assertEqual(c["duplicate_only"]["prediction_check"]["proper_crossing_pair_count"], 0)
        self.assertFalse(c["clean_reference"]["prediction_check"]["invalid"])
        for case in c.values():
            self.assertFalse(case["ground_truth_check"]["invalid"])
            self.assertEqual(case["alignment"]["best_snapshot"], f"t{case['step']:03d}")
        self.assertIs(s["reference_verified"], False)
        self.assertTrue(s["sources_unchanged"])
        self.assertEqual(s["gt_validation_sweep"]["frames"], 4)

    def test_outputs_and_report(self):
        mgn, gt_zip, out, s = self.go()
        files = {p.name for p in (out / "cases").iterdir()}
        cid = s["cases"]["crossing_onestep"]["id"]
        self.assertTrue({f"{cid}.json", f"{cid}_overview.svg", f"{cid}_zoom.svg"} <= files)
        case = json.loads((out / "cases" / f"{cid}.json").read_text(encoding="utf-8"))
        self.assertEqual(case["prediction"]["archive_keys"], ["pred_001", "lines_001", "mat_001"])
        self.assertEqual(case["ground_truth"]["zip_member"], "r0300/snapshots/t001.vtp")
        pair = case["prediction"]["problem_pairs"][0]
        self.assertEqual(pair["kind"], "crossing")
        self.assertEqual(len(pair["segments"]), 2)
        page = (out / "report.html").read_text(encoding="utf-8")
        self.assertNotIn("<script", page)
        for text in [
            "정답",
            "예측",
            "재학습 필요성을 자동으로 판단하지 않습니다",
            "reference_verified=False",
            "사례 선택 규칙 (validation만)",
            "r0300/snapshots/t001.vtp",
        ]:
            self.assertIn(text, page)

    def test_ground_truth_finding_is_reported(self):
        _, _, out, s = self.go(gt_crossing=True)
        same = s["cases"]["crossing_onestep"]["ground_truth_has_same_kind_of_finding"]
        self.assertTrue(same["crossing"])
        self.assertIn(
            "정답에도 교차 1쌍이 있습니다.", (out / "report.html").read_text(encoding="utf-8")
        )
        self.assertEqual(s["gt_validation_sweep"]["recipes_with_crossing"], ["r0300"])

    def test_unclear_mapping_stops_without_cases(self):
        for kw in [dict(bad_n_gt=True), dict(drop_snapshot=True)]:
            with self.subTest(**kw):
                tmp = tempfile.TemporaryDirectory()
                mgn, gt_zip = build(Path(tmp.name) / "in", **kw)
                out, s = run(mgn, gt_zip, Path(tmp.name) / "out")
                self.assertTrue(s["status"].startswith("stopped"))
                self.assertIn("r0300", s["mapping"]["failed"])
                self.assertEqual(s["cases"], {})
                self.assertFalse((out / "cases").exists())
                self.assertIn("중단", (out / "report.html").read_text(encoding="utf-8"))
                tmp.cleanup()

    def test_sources_unchanged(self):
        mgn, gt_zip = build(self.dir / "in")
        files = [*mgn.iterdir(), gt_zip]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        run(mgn, gt_zip, self.dir / "out")
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})

    def test_selection_prefers_rules_and_ignores_test(self):
        base = dict(
            invalid=True,
            proper_crossing_pair_count=0,
            overlap_pair_count=0,
            duplicate_edge_count=0,
            nonadjacent_touch_pair_count=0,
            structural_issue_counts={},
        )
        frames = [
            dict(
                base,
                rid="r0399",
                split="test",
                mode="onestep",
                step=1,
                proper_crossing_pair_count=99,
            ),
            dict(
                base,
                rid="r0302",
                split="validation",
                mode="onestep",
                step=3,
                proper_crossing_pair_count=2,
            ),
            dict(
                base,
                rid="r0301",
                split="validation",
                mode="onestep",
                step=1,
                proper_crossing_pair_count=2,
            ),
            dict(
                base,
                rid="r0301",
                split="validation",
                mode="rollout",
                step=2,
                duplicate_edge_count=1,
                proper_crossing_pair_count=1,
                structural_issue_counts={"duplicate_edge": 1},
            ),
        ]
        picks = select_cases(frames)
        self.assertEqual(
            (picks["crossing_onestep"]["rid"], picks["crossing_onestep"]["step"]), ("r0301", 1)
        )
        self.assertIsNone(
            picks["duplicate_only"]
        )  # duplicate together with a crossing is not "only"
        self.assertIsNone(picks["clean_reference"])

    def test_marker_radius_scales_with_zoom(self):
        mgn, gt_zip, out, s = self.go()
        z = (out / "cases" / f"{s['cases']['crossing_onestep']['id']}_zoom.svg").read_text(
            encoding="utf-8"
        )
        width = float(re.search(r'viewBox="\S+ \S+ (\S+) ', z).group(1))
        radii = [float(r) for r in re.findall(r'<circle[^>]* r="([\d.]+)"', z)]
        self.assertTrue(radii and all(r < width / 10 for r in radii))


if __name__ == "__main__":
    unittest.main(verbosity=2)
