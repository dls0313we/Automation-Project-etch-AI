"""MGN prediction vs ground-truth overlay for selected validation frames.

  python mgn_gt_overlay.py

Reads the MGN prediction folder (local_paths.json: mgn_input) and the ground-truth ZIP
(local_paths.json: mgn_gt_zip, Hugging Face code5312/etch-ai-automation-data etch-raw-n400.zip)
directly from their archives. Nothing is extracted or modified.

Order of work:
  1. Step correspondence is checked by snapshot NUMBER (t{k:03d}.vtp), never by ZIP or list order.
     A recipe passes only if t000..tN exist for its N prediction steps, meta.json actual_k == N,
     the last snapshot's point count equals summary_last_step.csv n_gt_points, and t000 is the
     shallowest snapshot. Selected frames additionally recompute a point Chamfer against
     t(k-1), t(k), t(k+1) as alignment evidence. If any check fails, no overlay is produced.
  2. Cases are chosen from validation frames only (test frames are never read for selection).
  3. Ground truth and prediction are checked with the same mgn_geometry.check_frame and tolerance.
  4. A static report.html with SVG overlays is written to a new folder.

The report describes what is detected. It does not decide the cause of a prediction error or
whether retraining is needed, and it never sets reference_verified to true.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import re
import struct
import sys
import zipfile
import zlib
from datetime import datetime
from pathlib import Path

import mgn_geometry as geo
from fno_adapter import local_input, sha256, write_json

ROOT = Path(__file__).resolve().parent
CONTRACT = "mgn-gt-overlay-v1"
GT_SOURCE = dict(
    repository="code5312/etch-ai-automation-data",
    repo_type="dataset",
    file="etch-raw-n400.zip",
    url="https://huggingface.co/datasets/code5312/etch-ai-automation-data",
)
GT_MATERIAL_MAP = {0: 0, 10: 1}  # VTP cell MaterialIds (0=mask, 10=Si) -> NPZ point codes (0, 1)
MAX_VTP_BYTES = 16 * 1024 * 1024
MAX_EXAMPLES = 500
VTK_TYPES = {
    "Float32": ("f", 4),
    "Float64": ("d", 8),
    "Int64": ("q", 8),
    "Int32": ("i", 4),
    "UInt64": ("Q", 8),
    "UInt32": ("I", 4),
    "Int8": ("b", 1),
    "UInt8": ("B", 1),
}
CATEGORIES = [
    ("crossing_onestep", "자기교차 (one-step)"),
    ("crossing_rollout", "자기교차 (rollout)"),
    ("duplicate_only", "중복 연결만 있음"),
    ("clean_reference", "검사상 문제 없음 (참고)"),
]
SELECTION_RULES = {
    "crossing_onestep": "validation one-step frames with proper crossings; most crossing pairs, "
    "then smallest recipe ID and step",
    "crossing_rollout": "validation rollout frames with proper crossings; most crossing pairs, "
    "then smallest recipe ID and step",
    "duplicate_only": "validation frames whose only finding is duplicate edges "
    "(no crossing, overlap, other structural issue or endpoint contact); "
    "most duplicates, one-step first, "
    "then smallest recipe ID and step",
    "clean_reference": "validation one-step frames with no finding and no endpoint contact; "
    "largest step, then smallest recipe ID",
}


class OverlayError(ValueError):
    """Unsupported file layout or an unclear prediction/ground-truth correspondence."""


# ---------- VTK XML PolyData (appended, base64, zlib) ----------


def _attrs(raw):
    return {k.decode(): v.decode() for k, v in re.findall(rb'(\w+)="([^"]*)"', raw)}


def _b64_len(n_bytes):
    return 4 * math.ceil(n_bytes / 3)


def read_vtp(blob):
    """Points (x, y), 2-node line cells and integer cell MaterialIds from one VTP file."""
    if len(blob) > MAX_VTP_BYTES:
        raise OverlayError("VTP file exceeds size limit")
    marker = blob.find(b"<AppendedData")
    if marker < 0:
        raise OverlayError("Only appended VTP data is supported")
    head = blob[:marker]
    vtk = re.search(rb"<VTKFile([^>]*)>", head)
    if not vtk:
        raise OverlayError("Missing VTKFile element")
    v = _attrs(vtk.group(1))
    if v.get("type") != "PolyData" or v.get("byte_order") != "LittleEndian":
        raise OverlayError(
            f"Unsupported VTKFile type/byte order: {v.get('type')}/{v.get('byte_order')}"
        )
    if v.get("header_type", "UInt32") not in ("UInt32", "UInt64"):
        raise OverlayError("Unsupported header_type")
    if v.get("compressor") != "vtkZLibDataCompressor":
        raise OverlayError(f"Unsupported compressor: {v.get('compressor')}")
    hcode, hsize = ("I", 4) if v.get("header_type", "UInt32") == "UInt32" else ("Q", 8)
    piece = re.search(rb"<Piece([^>]*)>", head)
    if not piece:
        raise OverlayError("Missing Piece element")
    p = {k: int(val) for k, val in _attrs(piece.group(1)).items()}
    if p.get("NumberOfVerts", 0) or p.get("NumberOfStrips", 0) or p.get("NumberOfPolys", 0):
        raise OverlayError("Only line cells are supported (verts/strips/polys must be empty)")
    appended = re.search(rb"<AppendedData([^>]*)>\s*_", blob)
    if not appended or _attrs(appended.group(1)).get("encoding") != "base64":
        raise OverlayError("Only base64 appended data is supported")
    end = blob.find(b"</AppendedData>", appended.end())
    payload = blob[appended.end() : end if end >= 0 else len(blob)]

    def decode(offset, vtk_type, count):
        code, width = VTK_TYPES[vtk_type]
        try:
            first = base64.b64decode(payload[offset : offset + _b64_len(hsize)], validate=True)
            nblocks = struct.unpack("<" + hcode, first[:hsize])[0]
            if not 0 <= nblocks <= 4096:
                raise OverlayError("Unreasonable block count")
            hdr_len = _b64_len(hsize * (3 + nblocks))
            hdr = base64.b64decode(payload[offset : offset + hdr_len], validate=True)
            vals = struct.unpack("<" + hcode * (3 + nblocks), hdr[: hsize * (3 + nblocks)])
            block, last, csizes = vals[1], vals[2], vals[3:]
            body = base64.b64decode(
                payload[offset + hdr_len : offset + hdr_len + _b64_len(sum(csizes))], validate=True
            )
        except (ValueError, struct.error) as exc:
            raise OverlayError(f"Bad appended array at offset {offset}: {exc}") from exc
        raw, pos = bytearray(), 0
        for i, size in enumerate(csizes):
            limit = (last or block) if i == nblocks - 1 else block
            d = zlib.decompressobj()
            chunk = d.decompress(body[pos : pos + size], limit + 1)
            if len(chunk) != limit:
                raise OverlayError("Decompressed block size mismatch")
            raw += chunk
            pos += size
        if len(raw) != count * width:
            raise OverlayError(f"Array length {len(raw)} != {count} x {width} bytes")
        return list(struct.unpack("<" + code * count, raw))

    section, arrays = None, {}
    for m in re.finditer(
        rb"<(/?)(CellData|PointData|Points|Verts|Lines|Strips|Polys|DataArray)\b([^>]*)>", head
    ):
        closing, name, raw = m.group(1), m.group(2).decode(), m.group(3)
        if name != "DataArray":
            section = None if closing else name
            continue
        a = _attrs(raw)
        if a.get("format") != "appended":
            raise OverlayError("Only appended DataArrays are supported")
        if a.get("type") not in VTK_TYPES:
            raise OverlayError(f"Unsupported DataArray type {a.get('type')}")
        arrays[(section, a.get("Name"))] = (
            int(a["offset"]),
            a["type"],
            int(a.get("NumberOfComponents", 1)),
        )
    n_points, n_lines = p.get("NumberOfPoints", 0), p.get("NumberOfLines", 0)
    need = [
        ("Points", "Points"),
        ("Lines", "connectivity"),
        ("Lines", "offsets"),
        ("CellData", "MaterialIds"),
    ]
    missing = [f"{s}/{n}" for s, n in need if (s, n) not in arrays]
    if missing:
        raise OverlayError(f"Missing VTP arrays: {missing}")
    off, typ, comp = arrays[("Points", "Points")]
    if comp != 3:
        raise OverlayError("Points must have 3 components")
    flat = decode(off, typ, n_points * 3)
    pts3 = [tuple(flat[i : i + 3]) for i in range(0, len(flat), 3)]
    offsets = decode(arrays[("Lines", "offsets")][0], arrays[("Lines", "offsets")][1], n_lines)
    total = offsets[-1] if offsets else 0
    conn = decode(arrays[("Lines", "connectivity")][0], arrays[("Lines", "connectivity")][1], total)
    mat_off, mat_type, _ = arrays[("CellData", "MaterialIds")]
    mats = decode(mat_off, mat_type, n_lines)
    lines, start = [], 0
    for end_ in offsets:
        cell = conn[start:end_]
        if len(cell) != 2:
            raise OverlayError(f"Line cell with {len(cell)} nodes (only 2-node lines supported)")
        lines.append((int(cell[0]), int(cell[1])))
        start = end_
    if any(not math.isfinite(m) or m != int(m) for m in mats):
        raise OverlayError("MaterialIds must be integral")
    return dict(
        points=[(x, y) for x, y, _z in pts3],
        max_abs_z=max((abs(z) for *_xy, z in pts3), default=0.0),
        lines=lines,
        cell_materials=[int(m) for m in mats],
        number_of_points=n_points,
        number_of_lines=n_lines,
    )


def point_materials(n_points, lines, cell_materials):
    """Per-point codes from incident cells via GT_MATERIAL_MAP (first incident cell wins).

    Points without a cell get -1 and points whose incident cells disagree are counted, so
    check_frame's material validation flags anything unusual instead of hiding it.
    """
    out, mixed = [None] * n_points, 0
    for (u, v), m in zip(lines, cell_materials, strict=True):
        code = GT_MATERIAL_MAP.get(m, m)
        for node in (u, v):
            if 0 <= node < n_points:
                if out[node] is None:
                    out[node] = code
                elif out[node] != code:
                    mixed += 1
    return [-1 if m is None else m for m in out], mixed


# ---------- archives ----------


class GroundTruth:
    """Read-only access to etch-raw-n400.zip by recipe ID and snapshot number."""

    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(self.path)
        self.names = {i.filename for i in self.zip.infolist()}

    def snapshot_numbers(self, rid):
        pat = re.compile(rf"^{re.escape(rid)}/snapshots/t(\d{{3}})\.vtp$")
        return sorted(int(m.group(1)) for n in self.names if (m := pat.match(n)))

    def name(self, rid, k):
        return f"{rid}/snapshots/t{k:03d}.vtp"

    def meta(self, rid):
        name = f"{rid}/meta.json"
        return json.loads(self.zip.read(name).decode("utf-8")) if name in self.names else None

    def frame(self, rid, k):
        name = self.name(rid, k)
        if name not in self.names:
            raise OverlayError(f"Missing ground-truth snapshot {name}")
        return read_vtp(self.zip.read(name))

    def close(self):
        self.zip.close()


def read_prediction(npz_path, step):
    """Arrays pred/lines/mat for one step, validated like mgn_geometry._archive."""
    with zipfile.ZipFile(npz_path) as z:
        arrays = {}
        for key in ("pred", "lines", "mat"):
            name = f"{key}_{step:03d}.npy"
            if z.getinfo(name).file_size > geo.MAX_MEMBER_BYTES:
                raise OverlayError("Array member exceeds byte limit")
            shape, flat, dtype = geo.read_npy(z.read(name))
            if key in ("pred", "lines"):
                if len(shape) != 2 or shape[1] != 2:
                    raise OverlayError(f"{key} must have shape (N, 2)")
                if (key == "pred" and dtype[1] != "f") or (key == "lines" and dtype[1] not in "iu"):
                    raise OverlayError(f"{key} has wrong numeric dtype")
                arrays[key] = list(zip(flat[::2], flat[1::2], strict=True))
            else:
                if len(shape) != 1 or dtype[1] not in "iu":
                    raise OverlayError("mat must be a one-dimensional integer array")
                arrays[key] = flat
        has_zero = "pred_000.npy" in z.namelist()
    return dict(
        points=arrays["pred"],
        lines=[tuple(map(int, e)) for e in arrays["lines"]],
        materials=arrays["mat"],
        keys=[f"{k}_{step:03d}" for k in ("pred", "lines", "mat")],
        has_step_zero=has_zero,
    )


# ---------- MGN tables ----------


def read_mgn_tables(folder):
    folder = Path(folder)
    with (folder / "metrics_by_step.csv").open(encoding="utf-8-sig", newline="") as f:
        steps = {}
        for r in csv.DictReader(f):
            steps.setdefault(r["rid"], dict(split=r["split"], rows=[]))["rows"].append(
                dict(
                    step=int(r["step"]),
                    onestep_chamfer=float(r["onestep_chamfer"]),
                    rollout_chamfer=float(r["rollout_chamfer"]),
                )
            )
    for v in steps.values():
        v["rows"].sort(key=lambda x: x["step"])
    with (folder / "summary_last_step.csv").open(encoding="utf-8-sig", newline="") as f:
        summary = {
            r["rid"]: dict(
                split=r["split"],
                last_step=int(float(r["last_step"])),
                n_gt_points=int(float(r["n_gt_points"])),
                n_pred_points=int(float(r["n_pred_points"])),
            )
            for r in csv.DictReader(f)
        }
    return steps, summary


# ---------- correspondence ----------


def depth(points):
    return -min(y for _x, y in points) if points else None


def verify_mapping(gt, rid, steps, summary_row):
    """Number-based correspondence checks for one recipe; returns dict(ok, checks, reasons)."""
    reasons, checks = [], {}
    n = len(steps)
    nums = gt.snapshot_numbers(rid)
    checks["snapshot_numbers"] = (
        f"t{nums[0]:03d}..t{nums[-1]:03d} ({len(nums)})" if nums else "none"
    )
    if nums != list(range(0, n + 1)):
        reasons.append(f"snapshot numbers {nums[:3]}..{nums[-3:]} != t000..t{n:03d}")
    if [s["step"] for s in steps] != list(range(1, n + 1)):
        reasons.append("prediction steps are not 1..N")
    meta = gt.meta(rid)
    checks["meta_actual_k"] = meta and meta.get("actual_k")
    if not meta or meta.get("actual_k") != n:
        reasons.append(f"meta.json actual_k={checks['meta_actual_k']} != prediction steps {n}")
    if reasons:
        return dict(ok=False, checks=checks, reasons=reasons)
    last = gt.frame(rid, n)
    checks["last_gt_points"] = last["number_of_points"]
    checks["summary_n_gt_points"] = summary_row and summary_row["n_gt_points"]
    if not summary_row or last["number_of_points"] != summary_row["n_gt_points"]:
        reasons.append(
            f"t{n:03d}.vtp has {last['number_of_points']} points, "
            f"summary n_gt_points={checks['summary_n_gt_points']}"
        )
    depths = [depth(gt.frame(rid, k)["points"]) for k in range(0, n + 1)]
    checks["depth_t000_um"] = depths[0]
    checks["depth_last_um"] = depths[-1]
    checks["meta_final_depth_um"] = meta.get("final_depth_um")
    if depths[0] != min(depths) or depths[0] >= depths[1]:
        reasons.append("t000 is not the shallowest snapshot (initial shape not confirmed)")
    return dict(ok=not reasons, checks=checks, reasons=reasons)


def point_chamfer(a, b):
    """Sum of the two one-way mean nearest-point distances (README definition, point based)."""

    def one_way(src, dst):
        return sum(min(math.dist(p, q) for q in dst) for p in src) / len(src)

    return one_way(a, b) + one_way(b, a)


def alignment_evidence(gt, rid, n_steps, pred_points, step, provided):
    rows = {}
    for k in (step - 1, step, step + 1):
        if 0 <= k <= n_steps:
            rows[k] = point_chamfer(pred_points, gt.frame(rid, k)["points"])
    best = min(rows, key=rows.get)
    return dict(
        chamfer_by_snapshot={f"t{k:03d}": v for k, v in rows.items()},
        best_snapshot=f"t{best:03d}",
        provided_chamfer=provided,
        recomputed_at_step=rows[step],
        relative_difference=abs(rows[step] - provided) / max(abs(provided), 1e-12),
        aligned=best == step,
    )


# ---------- selection ----------


def only_duplicates(f):
    return (
        f["duplicate_edge_count"] > 0
        and f["proper_crossing_pair_count"] == 0
        and f["overlap_pair_count"] == 0
        and f["nonadjacent_touch_pair_count"] == 0
        and set(f["structural_issue_counts"]) == {"duplicate_edge"}
    )


def select_cases(frames):
    """Deterministic picks from validation frames only (see SELECTION_RULES)."""
    val = [f for f in frames if f["split"] == "validation"]
    picks = {}
    for mode in ("onestep", "rollout"):
        c = [f for f in val if f["mode"] == mode and f["proper_crossing_pair_count"] > 0]
        picks[f"crossing_{mode}"] = min(
            c, key=lambda f: (-f["proper_crossing_pair_count"], f["rid"], f["step"]), default=None
        )
    c = [f for f in val if only_duplicates(f)]
    picks["duplicate_only"] = min(
        c,
        key=lambda f: (-f["duplicate_edge_count"], f["mode"] != "onestep", f["rid"], f["step"]),
        default=None,
    )
    c = [
        f
        for f in val
        if f["mode"] == "onestep" and not f["invalid"] and f["nonadjacent_touch_pair_count"] == 0
    ]
    picks["clean_reference"] = min(c, key=lambda f: (-f["step"], f["rid"]), default=None)
    return picks


# ---------- per-case analysis ----------


def intersection(s1, s2):
    (x1, y1), (x2, y2) = s1
    (x3, y3), (x4, y4) = s2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def seg_point_distance(p, a, b):
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def problem_pairs(check):
    out = []
    for ex in check["pair_examples"]:
        point = intersection(*ex["segments"]) if ex["kind"] == "crossing" else None
        out.append(
            dict(
                kind=ex["kind"],
                edge_ids=ex["edge_ids"],
                segments=ex["segments"],
                crossing_point=list(point) if point else None,
            )
        )
    return out


def analyse_case(category, frame, gt, mgn_folder, steps_table, n_steps):
    rid, mode, step = frame["rid"], frame["mode"], frame["step"]
    npz = Path(mgn_folder) / f"{rid}_{mode}.npz"
    pred = read_prediction(npz, step)
    pred_check = geo.check_frame(
        pred["points"], pred["lines"], pred["materials"], max_examples=MAX_EXAMPLES
    )
    for key in (
        "proper_crossing_pair_count",
        "overlap_pair_count",
        "duplicate_edge_count",
        "nonadjacent_touch_pair_count",
    ):
        if pred_check[key] != frame[key]:
            raise OverlayError(f"{rid} {mode} t{step}: re-check {key} differs from geometry scan")
    g = gt.frame(rid, step)
    gt_mats, mixed = point_materials(g["number_of_points"], g["lines"], g["cell_materials"])
    gt_check = geo.check_frame(g["points"], g["lines"], gt_mats, max_examples=MAX_EXAMPLES)
    provided = next(r for r in steps_table if r["step"] == step)[f"{mode}_chamfer"]
    align = alignment_evidence(gt, rid, n_steps, pred["points"], step, provided)
    p_pairs, g_pairs = problem_pairs(pred_check), problem_pairs(gt_check)
    gt_cross = [q["crossing_point"] for q in g_pairs if q["crossing_point"]]
    for q in p_pairs:
        if q["crossing_point"]:
            q["nearest_gt_crossing_um"] = (
                min(math.dist(q["crossing_point"], c) for c in gt_cross) if gt_cross else None
            )
            q["nearest_gt_boundary_um"] = min(
                seg_point_distance(q["crossing_point"], g["points"][u], g["points"][v])
                for u, v in g["lines"]
            )

    def summary(check):
        return {
            k: check[k]
            for k in (
                "invalid",
                "point_count",
                "line_count",
                "proper_crossing_pair_count",
                "overlap_pair_count",
                "duplicate_edge_count",
                "duplicate_edge_pair_count",
                "nonadjacent_touch_pair_count",
                "structural_issue_counts",
                "examples_truncated",
            )
        }

    return dict(
        category=category,
        rid=rid,
        split="validation",
        mode=mode,
        step=step,
        prediction=dict(
            file=npz.name,
            archive_keys=pred["keys"],
            check=summary(pred_check),
            problem_pairs=p_pairs,
            structural_examples=pred_check["structural_examples"],
            touch_examples=pred_check["touch_examples"],
        ),
        ground_truth=dict(
            zip_member=gt.name(rid, step),
            arrays=["Points", "Lines/connectivity", "Lines/offsets", "CellData/MaterialIds"],
            max_abs_z=g["max_abs_z"],
            material_map=GT_MATERIAL_MAP,
            points_with_mixed_cell_material=mixed,
            check=summary(gt_check),
            problem_pairs=g_pairs,
            structural_examples=gt_check["structural_examples"],
            touch_examples=gt_check["touch_examples"],
        ),
        ground_truth_has_same_kind_of_finding=dict(
            crossing=gt_check["proper_crossing_pair_count"] > 0,
            overlap=gt_check["overlap_pair_count"] > 0,
            duplicate_edge=gt_check["duplicate_edge_count"] > 0,
        ),
        alignment=align,
        _geometry=dict(pred=pred, gt=g),
    )


def gt_validation_sweep(gt, recipes):
    """Same check on every ground-truth snapshot t001..tN of validation recipes (context only)."""
    totals = dict(
        frames=0,
        crossing_frames=0,
        overlap_frames=0,
        duplicate_frames=0,
        touch_frames=0,
        recipes_with_crossing=[],
    )
    for rid, n in recipes:
        crossed = False
        for k in range(1, n + 1):
            g = gt.frame(rid, k)
            mats, _ = point_materials(g["number_of_points"], g["lines"], g["cell_materials"])
            c = geo.check_frame(g["points"], g["lines"], mats, max_examples=0)
            totals["frames"] += 1
            totals["crossing_frames"] += c["proper_crossing_pair_count"] > 0
            totals["overlap_frames"] += c["overlap_pair_count"] > 0
            totals["duplicate_frames"] += c["duplicate_edge_count"] > 0
            totals["touch_frames"] += c["nonadjacent_touch_pair_count"] > 0
            crossed |= c["proper_crossing_pair_count"] > 0
        if crossed:
            totals["recipes_with_crossing"].append(rid)
    return totals


# ---------- SVG / HTML ----------


def _fmt(v, digits=4):
    return "-" if v is None else f"{v:.{digits}g}" if isinstance(v, float) else html.escape(str(v))


def svg(case, zoom=False):
    """Inline SVG: ground truth gray, prediction blue, flagged segments highlighted.

    The y axis is flipped so that deeper (more negative) y is drawn lower.
    """
    geom = case["_geometry"]
    pred, gt = geom["pred"], geom["gt"]
    p_flag = {i for q in case["prediction"]["problem_pairs"] for i in q["edge_ids"]}
    g_flag = {i for q in case["ground_truth"]["problem_pairs"] for i in q["edge_ids"]}
    for ex in case["prediction"]["structural_examples"]:
        if ex["kind"] == "duplicate_edge" and ex["index"] is not None:
            p_flag.add(ex["index"])
    if zoom:
        pts = []
        for q in (
            case["prediction"]["problem_pairs"][:1] or case["ground_truth"]["problem_pairs"][:1]
        ):
            pts += [tuple(p) for s in q["segments"] for p in s]
        if not pts:
            dup = [pred["lines"][i] for i in sorted(p_flag)][:1]
            pts = [pred["points"][n] for e in dup for n in e]
        if not pts:
            return ""
    else:
        pts = list(pred["points"]) + list(gt["points"])
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    pad = max(w, h) * (0.6 if zoom else 0.03) + (0.05 if zoom else 0.1)
    x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    r = 0.015 * max(x1 - x0, y1 - y0)  # marker radius relative to the view, not fixed in um

    def path(points, lines, idx):
        return " ".join(
            f"M{points[lines[i][0]][0]:.5f} {-points[lines[i][0]][1]:.5f}"
            f"L{points[lines[i][1]][0]:.5f} {-points[lines[i][1]][1]:.5f}"
            for i in idx
        )

    g_all = path(gt["points"], gt["lines"], range(len(gt["lines"])))
    p_all = path(pred["points"], pred["lines"], range(len(pred["lines"])))
    g_hi = path(gt["points"], gt["lines"], sorted(g_flag))
    p_hi = path(pred["points"], pred["lines"], sorted(p_flag))

    def circles(cls, pairs):
        return "".join(
            f'<circle class="{cls}" cx="{q["crossing_point"][0]:.5f}" '
            f'cy="{-q["crossing_point"][1]:.5f}" r="{r:.5f}"/>'
            for q in pairs
            if q["crossing_point"]
        )

    marks = circles("x", case["prediction"]["problem_pairs"])
    gmarks = circles("gx", case["ground_truth"]["problem_pairs"])
    title = f"{case['rid']} {case['mode']} t{case['step']:03d} {'확대' if zoom else '전체'}"
    return (
        f'<svg class="plot" viewBox="{x0:.5f} {-y1:.5f} {x1 - x0:.5f} {y1 - y0:.5f}" '
        f'preserveAspectRatio="xMidYMid meet" role="img" aria-label="{html.escape(title)}">'
        f'<rect x="{x0:.5f}" y="{-y1:.5f}" width="{x1 - x0:.5f}" '
        f'height="{y1 - y0:.5f}" class="bg"/>'
        f'<path class="gt" d="{g_all}"/><path class="pred" d="{p_all}"/>'
        f'<path class="gthi" d="{g_hi}"/><path class="predhi" d="{p_hi}"/>{gmarks}{marks}</svg>'
    )


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#5f5f5a;--line:#d9d8d2;--card:#fff;--plot:#ffffff;
--gt:#6b6b66;--pred:#2563eb;--hi:#dc2626;--gthi:#9333ea;--warn-bg:#fff4d6;--warn-fg:#6b4a00}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a6a59d;--line:#3a3935;
--card:#1f1f1d;--plot:#111110;--gt:#b5b4ad;--pred:#60a5fa;--hi:#f87171;--gthi:#c084fc;
--warn-bg:#3a2f12;--warn-fg:#f3d48a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:1.5rem;margin:0 0 4px}
h2{font-size:1.15rem;margin:32px 0 8px}h3{font-size:1rem;margin:12px 0 6px}
.muted{color:var(--muted)}
.banner{padding:10px 14px;border-radius:8px;margin:12px 0;
background:var(--warn-bg);color:var(--warn-fg)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;
overflow-x:auto;min-width:0;margin:8px 0}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(320px,100%),1fr));gap:12px}
.two>*{min-width:0}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{border-bottom:1px solid var(--line);padding:5px 7px;text-align:left;vertical-align:top;
overflow-wrap:anywhere}th{color:var(--muted);font-weight:600}td.n{text-align:right}
.counts th:not(:first-child){text-align:right}
.plot{width:100%;height:auto;max-height:520px;display:block}
.plot .bg{fill:var(--plot)}
.plot path{fill:none;vector-effect:non-scaling-stroke;stroke-linecap:round}
.plot .gt{stroke:var(--gt);stroke-width:1.6}.plot .pred{stroke:var(--pred);stroke-width:1.2}
.plot .gthi{stroke:var(--gthi);stroke-width:4;stroke-dasharray:6 4}
.plot .predhi{stroke:var(--hi);stroke-width:3.5}
.plot circle{vector-effect:non-scaling-stroke;fill:none;stroke-width:2}
.plot .x{stroke:var(--hi)}.plot .gx{stroke:var(--gthi)}
.legend span{display:inline-block;margin-right:14px}
.sw{display:inline-block;width:22px;height:0;border-top:3px solid;vertical-align:middle;
margin-right:5px}
code{font-size:.85em}
"""


def coord(p):
    return f"({p[0]:.4f}, {p[1]:.4f})"


def pair_rows(pairs, with_distance):
    rows = []
    for q in pairs:
        segs = " / ".join(f"{coord(s[0])}–{coord(s[1])}" for s in q["segments"])
        extra = ""
        if with_distance:
            extra = (
                f'<td class="n">{_fmt(q.get("nearest_gt_crossing_um"))}</td>'
                f'<td class="n">{_fmt(q.get("nearest_gt_boundary_um"))}</td>'
            )
        rows.append(
            f"<tr><td>{html.escape(q['kind'])}</td><td>{html.escape(str(q['edge_ids']))}</td>"
            f"<td>{segs}</td><td>{coord(q['crossing_point']) if q['crossing_point'] else '-'}</td>"
            f"{extra}</tr>"
        )
    return "".join(rows)


NO_ZOOM = '<p class="muted">검출된 문제가 없어 확대 그림이 없습니다.</p>'


def case_html(label, case):
    pc, gc = case["prediction"]["check"], case["ground_truth"]["check"]
    same = case["ground_truth_has_same_kind_of_finding"]
    gt_note = []
    if gc["proper_crossing_pair_count"]:
        gt_note.append(f"정답에도 교차 {gc['proper_crossing_pair_count']}쌍이 있습니다.")
    if gc["overlap_pair_count"]:
        gt_note.append(f"정답에도 서로 다른 선분의 중첩 {gc['overlap_pair_count']}쌍이 있습니다.")
    if gc["duplicate_edge_count"]:
        gt_note.append(f"정답에도 중복 연결 {gc['duplicate_edge_count']}개가 있습니다.")
    if gc["structural_issue_counts"] and set(gc["structural_issue_counts"]) - {"duplicate_edge"}:
        gt_note.append(
            "정답에 기타 구조 이상: " + html.escape(json.dumps(gc["structural_issue_counts"]))
        )
    gt_text = (
        " ".join(gt_note) or "같은 정의로 검사한 정답 프레임에서는 문제가 검출되지 않았습니다."
    )
    a = case["alignment"]
    align = ", ".join(f"{k}={v:.4g}" for k, v in a["chamfer_by_snapshot"].items())
    zoom = svg(case, zoom=True)
    p_rows = pair_rows(case["prediction"]["problem_pairs"], True)
    g_rows = pair_rows(case["ground_truth"]["problem_pairs"], False)
    dup = [ex for ex in case["prediction"]["structural_examples"] if ex["kind"] == "duplicate_edge"]
    dup_text = ""
    if dup:
        lines = case["_geometry"]["pred"]["lines"]
        pts = case["_geometry"]["pred"]["points"]
        dup_text = (
            "<p>예측 중복 연결(선분 번호, 노드, 좌표): "
            + "; ".join(
                f"#{ex['index']} {lines[ex['index']]} {coord(pts[lines[ex['index']][0]])}–"
                f"{coord(pts[lines[ex['index']][1]])}"
                for ex in dup
            )
            + "</p>"
        )

    def count_row(c):
        keys = (
            "point_count",
            "line_count",
            "proper_crossing_pair_count",
            "overlap_pair_count",
            "duplicate_edge_count",
            "nonadjacent_touch_pair_count",
        )
        return "".join(f'<td class="n">{c[k]}</td>' for k in keys)

    return f"""<h2>{html.escape(label)}: {html.escape(case["rid"])} · {html.escape(case["mode"])} ·
t{case["step"]:03d}</h2>
<div class="card"><table>
<tr><th>예측 파일 / 배열 키</th><td><code>{html.escape(case["prediction"]["file"])}</code> ·
<code>{html.escape(", ".join(case["prediction"]["archive_keys"]))}</code></td></tr>
<tr><th>정답 파일 / 배열</th><td><code>{html.escape(case["ground_truth"]["zip_member"])}</code> ·
<code>{html.escape(", ".join(case["ground_truth"]["arrays"]))}</code></td></tr>
<tr><th>단계 대응 근거</th><td>파일명 번호 t{case["step"]:03d}. 점 Chamfer 재계산 {align};
가장 가까운 정답 {a["best_snapshot"]};
제공값 {a["provided_chamfer"]:.6g}, 재계산 {a["recomputed_at_step"]:.6g}</td></tr>
</table></div>
<div class="card"><table class="counts"><tr><th></th><th>점</th><th>선분</th>
<th>교차 쌍</th><th>중첩 쌍</th><th>중복 연결</th><th>끝점 접촉</th></tr>
<tr><th>예측</th>{count_row(pc)}</tr><tr><th>정답</th>{count_row(gc)}</tr></table>
<p><b>정답 비교:</b> {gt_text}</p>
<p class="muted">같은 종류 검출 여부 — 교차 {same["crossing"]}, 중첩 {same["overlap"]},
중복 연결 {same["duplicate_edge"]}.
이 비교만으로 예측 오류의 원인이나 재학습 필요성을 판단하지 않습니다.</p></div>
<div class="two"><div class="card"><h3>전체 형상</h3>{svg(case)}</div>
<div class="card"><h3>문제 부분 확대</h3>{zoom or NO_ZOOM}</div></div>
<div class="card"><h3>예측 문제 선분</h3>
<table><tr><th>종류</th><th>선분 번호</th><th>좌표 (µm)</th><th>교차점</th>
<th>가장 가까운 정답 교차점 거리</th>
<th>정답 경계까지 거리</th></tr>{p_rows or '<tr><td colspan="6">없음</td></tr>'}</table>{dup_text}
<h3>정답 문제 선분</h3><table><tr><th>종류</th><th>선분 번호</th><th>좌표 (µm)</th>
<th>교차점</th></tr>
{g_rows or '<tr><td colspan="4">없음</td></tr>'}</table></div>"""


def render_html(s, cases):
    m = s["mapping"]
    picks = "".join(
        f"<tr><td>{html.escape(lbl)}</td><td>{html.escape(SELECTION_RULES[key])}</td>"
        f"<td>{html.escape(s['cases'].get(key, {}).get('id', '후보 없음'))}</td></tr>"
        for key, lbl in CATEGORIES
    )
    sweep = s["gt_validation_sweep"]
    body = "".join(case_html(lbl, cases[key]) for key, lbl in CATEGORIES if key in cases)
    prov = s["ground_truth_source"]
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MGN 정답 형상 비교</title><style>{CSS}</style></head><body><main>
<h1>MGN 예측 · 정답 형상 비교</h1>
<p class="muted">생성 {html.escape(s["generated_at"])} · 계약 {CONTRACT} · 형상 검사 {geo.CONTRACT}
(허용치 {geo.EPSILON_UM} µm) · validation 사례만 · reference_verified={s["reference_verified"]}</p>
<div class="banner">그림은 검출 위치를 보여 주는 진단 자료입니다.
정답 자료와의 일치나 물리적 정확도를 검증했다는 뜻이 아니며,
예측 오류의 원인이나 재학습 필요성을 자동으로 판단하지 않습니다.</div>
<p class="legend"><span><i class="sw" style="border-color:var(--gt)"></i>정답</span>
<span><i class="sw" style="border-color:var(--pred)"></i>예측</span>
<span><i class="sw" style="border-color:var(--hi)"></i>예측 문제 선분 · 교차점(원)</span>
<span><i class="sw" style="border-color:var(--gthi);border-top-style:dashed"></i>
정답 문제 선분</span></p>
<h2>단계 대응 확인</h2><div class="card"><p>validation {m["recipes_checked"]}개 레시피 중
{m["recipes_ok"]}개 통과. 예측 단계 k ↔ 정답 <code>t&lt;k&gt;.vtp</code>
(파일명 번호, t000=초기 형상, 예측 없음).
마지막 정답 점 수 = <code>n_gt_points</code>:
{m["n_gt_points_match"]}/{m["recipes_checked"]} 일치.</p></div>
<h2>사례 선택 규칙 (validation만)</h2><div class="card"><table>
<tr><th>사례</th><th>규칙</th><th>선택</th></tr>
{picks}</table></div>
<h2>정답 전체 검사 요약 (validation t001–tN, 참고)</h2>
<div class="card"><p>프레임 {sweep["frames"]}개 중 교차 {sweep["crossing_frames"]},
서로 다른 선분 중첩 {sweep["overlap_frames"]}, 중복 연결 {sweep["duplicate_frames"]},
끝점 접촉 {sweep["touch_frames"]}.
교차가 있는 레시피 {len(sweep["recipes_with_crossing"])}개.</p></div>
{body}
<h2>정답 출처</h2><div class="card"><table>
<tr><th>저장소</th><td>{html.escape(prov["repository"])}
({html.escape(prov["repo_type"])})</td></tr>
<tr><th>파일 / 버전</th><td>{html.escape(prov["file"])} ·
revision <code>{html.escape(str(prov.get("revision")))}</code></td></tr>
<tr><th>SHA256</th><td><code>{html.escape(str(prov.get("sha256_actual")))}</code>
(기록값 일치: {prov.get("sha256_matches_record")})</td></tr>
</table></div></main></body></html>
"""


# ---------- run ----------


def run(mgn_folder, gt_zip, output_root, provenance=None):
    mgn_folder, gt_zip = Path(mgn_folder), Path(gt_zip)
    started = datetime.now()
    sources_before = {p.name: sha256(p) for p in sorted(mgn_folder.iterdir()) if p.is_file()}
    zip_sha = sha256(gt_zip)
    prov = dict(GT_SOURCE, **(provenance or {}))
    prov["sha256_actual"] = zip_sha
    prov["sha256_matches_record"] = (prov.get("sha256") == zip_sha) if prov.get("sha256") else None
    steps_table, summary = read_mgn_tables(mgn_folder)
    val = sorted((rid, v) for rid, v in steps_table.items() if v["split"] == "val")
    gt = GroundTruth(gt_zip)
    try:
        mapping = {rid: verify_mapping(gt, rid, v["rows"], summary.get(rid)) for rid, v in val}
        failed = {rid: m["reasons"] for rid, m in mapping.items() if not m["ok"]}
        mapping_summary = dict(
            recipes_checked=len(mapping),
            recipes_ok=len(mapping) - len(failed),
            failed=failed,
            n_gt_points_match=sum(
                1
                for m in mapping.values()
                if m["checks"].get("last_gt_points") == m["checks"].get("summary_n_gt_points")
                and m["checks"].get("last_gt_points") is not None
            ),
            rule=(
                "prediction step k <-> {rid}/snapshots/t{k:03d}.vtp by file-name number; "
                "t000 = initial"
            ),
        )
        out = Path(output_root) / started.strftime("%Y%m%d_%H%M%S_%f")
        out.mkdir(parents=True, exist_ok=False)
        s = dict(
            contract=CONTRACT,
            geometry_contract=geo.CONTRACT,
            epsilon_um=geo.EPSILON_UM,
            generated_at=started.isoformat(timespec="seconds"),
            mgn_input=str(mgn_folder),
            ground_truth_zip=str(gt_zip),
            ground_truth_source=prov,
            reference_verified=False,
            mapping=mapping_summary,
            mapping_detail=mapping,
            selection_rules=SELECTION_RULES,
            cases={},
            training_started=False,
            active_model_changed=False,
        )
        if failed:
            s["status"] = "stopped: step correspondence unclear"
            write_json(out / "summary.json", s)
            (out / "report.html").write_text(
                '<!doctype html><meta charset="utf-8"><title>MGN 정답 형상 비교</title>'
                "<h1>단계 대응이 불명확하여 중단했습니다</h1><pre>"
                + html.escape(json.dumps(failed, ensure_ascii=False, indent=2))
                + "</pre>",
                encoding="utf-8",
            )
            return out, s
        expected = [
            dict(rid=rid, split="validation", steps=[r["step"] for r in v["rows"]])
            for rid, v in val
        ]
        scan = geo.scan_dataset(mgn_folder, expected)
        if scan["errors"]:
            raise OverlayError(f"Geometry scan read errors: {scan['errors'][:3]}")
        picks = select_cases(scan["frames"])
        cases = {}
        for key, frame in picks.items():
            if frame is None:
                continue
            n = len(steps_table[frame["rid"]]["rows"])
            case = analyse_case(key, frame, gt, mgn_folder, steps_table[frame["rid"]]["rows"], n)
            if not case["alignment"]["aligned"]:
                raise OverlayError(
                    f"{frame['rid']} t{frame['step']:03d}: nearest ground-truth snapshot is "
                    f"{case['alignment']['best_snapshot']}, not the same step"
                )
            cases[key] = case
        sweep = gt_validation_sweep(gt, [(rid, len(v["rows"])) for rid, v in val])
    finally:
        gt.close()
    case_dir = out / "cases"
    case_dir.mkdir()
    for key, case in cases.items():
        cid = f"{key}__{case['rid']}_{case['mode']}_t{case['step']:03d}"
        s["cases"][key] = dict(
            id=cid,
            rid=case["rid"],
            mode=case["mode"],
            step=case["step"],
            prediction_check=case["prediction"]["check"],
            ground_truth_check=case["ground_truth"]["check"],
            ground_truth_has_same_kind_of_finding=case["ground_truth_has_same_kind_of_finding"],
            alignment=case["alignment"],
        )
        write_json(case_dir / f"{cid}.json", {k: v for k, v in case.items() if k != "_geometry"})
        for name, zoom in (("overview", False), ("zoom", True)):
            body = svg(case, zoom=zoom)
            if body:
                (case_dir / f"{cid}_{name}.svg").write_text(
                    f'<?xml version="1.0" encoding="UTF-8"?>\n<!-- {CONTRACT} -->\n'
                    + body.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1).replace(
                        "</svg>", f"<style>{SVG_FILE_CSS}</style></svg>"
                    ),
                    encoding="utf-8",
                )
    s["gt_validation_sweep"] = sweep
    s["status"] = "ok"
    if {
        p.name: sha256(p) for p in sorted(mgn_folder.iterdir()) if p.is_file()
    } != sources_before or sha256(gt_zip) != zip_sha:
        raise OverlayError("A source file changed during the run")
    s["sources_unchanged"] = True
    write_json(out / "summary.json", s)
    (out / "report.html").write_text(render_html(s, cases), encoding="utf-8")
    return out, s


SVG_FILE_CSS = (
    ".bg{fill:#fff}path{fill:none;vector-effect:non-scaling-stroke}"
    ".gt{stroke:#6b6b66;stroke-width:1.6}"
    ".pred{stroke:#2563eb;stroke-width:1.2}"
    ".gthi{stroke:#9333ea;stroke-width:4;stroke-dasharray:6 4}"
    ".predhi{stroke:#dc2626;stroke-width:3.5}circle{fill:none;vector-effect:non-scaling-stroke;"
    "stroke-width:2}.x{stroke:#dc2626}.gx{stroke:#9333ea}"
)


def main():
    cfg = {}
    try:
        cfg = json.loads((ROOT / "local_paths.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mgn-input", default=str(local_input("mgn_input", "mgn_main_s1_nov9")))
    p.add_argument("--gt-zip", default=str(local_input("mgn_gt_zip", "etch-raw-n400.zip")))
    p.add_argument("--output-root", default=str(ROOT / "gt_overlay_results"))
    args = p.parse_args()
    for path in (args.mgn_input, args.gt_zip):
        if not Path(path).exists():
            print("입력 없음:", path)
            raise SystemExit(1)
    out, s = run(args.mgn_input, args.gt_zip, args.output_root, cfg.get("mgn_gt_provenance"))
    m = s["mapping"]
    print(f"단계 대응: validation {m['recipes_ok']}/{m['recipes_checked']} 통과")
    for key, c in s["cases"].items():
        pc, gc = c["prediction_check"], c["ground_truth_check"]
        print(
            f"{key}: {c['rid']} {c['mode']} t{c['step']:03d} | "
            f"예측 교차 {pc['proper_crossing_pair_count']}, "
            f"중첩 {pc['overlap_pair_count']}, 중복 {pc['duplicate_edge_count']} | "
            f"정답 교차 {gc['proper_crossing_pair_count']}, "
            f"중첩 {gc['overlap_pair_count']}, 중복 {gc['duplicate_edge_count']}"
        )
    print("보고서:", out / "report.html")
    if s["status"] != "ok":
        print("중단:", s["status"])
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())
