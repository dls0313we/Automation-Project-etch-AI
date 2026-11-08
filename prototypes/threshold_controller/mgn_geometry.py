"""Inspect numeric MGN NPZ segment graphs with Python's standard library.

No pickle loading, model inference, reference loading, remeshing or training.
NPY layout: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
This is a versioned diagnostic definition, not an approved physical-validity gate.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import struct
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

CONTRACT = "mgn-segment-geometry-v2"
EPSILON_UM = 1e-8
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
DEFINITION = {
    "contract_id": CONTRACT,
    "epsilon_um": EPSILON_UM,
    "invalid_geometry_count": "Unique prediction frames with proper crossings, positive-length "
    "collinear overlaps, nonfinite coordinates, invalid graph indices/materials, duplicate "
    "edges or zero-length edges; count each frame once across all its prediction steps.",
    "self_intersection_frame_count": "Unique frames with at least one proper interior crossing.",
    "overlap_pair_count": "Positive-length collinear overlap between different unordered "
    "node pairs; "
    "repeated connections between the same two node IDs are excluded and reported as duplicates.",
    "duplicate_edge_count": "Extra copies of an unordered node-pair connection "
    "beyond its first copy. "
    "A connectivity/export issue, not proof of coordinate self-intersection. Currently retained in "
    "the provisional problem-frame union; physical-gate treatment requires producer confirmation.",
    "endpoint_contacts": "Normal shared-node endpoints excluded; nonadjacent contacts and "
    "T-junctions are reported separately and do not alone mark a frame invalid.",
    "scope": "2D predicted segment graph only; "
    "no reference, physics, depth or shape-accuracy validation.",
    "missing": "An incomplete split/mode retains a null count, never an assumed zero.",
}


class GeometryFormatError(ValueError):
    """Unsupported or inconsistent numeric-array layout."""


def read_npy(blob):
    """Read bounded C-order numeric NPY 1/2/3; reject object/structured arrays."""
    if len(blob) < 10 or blob[:6] != b"\x93NUMPY":
        raise GeometryFormatError("Invalid NPY magic/header")
    version = tuple(blob[6:8])
    if version not in [(1, 0), (2, 0), (3, 0)]:
        raise GeometryFormatError(f"Unsupported NPY version {version}")
    size = 2 if version[0] == 1 else 4
    if len(blob) < 8 + size:
        raise GeometryFormatError("Truncated NPY header length")
    n = int.from_bytes(blob[8 : 8 + size], "little")
    if not 0 < n <= 16384 or len(blob) < 8 + size + n:
        raise GeometryFormatError("Invalid/truncated/oversized NPY header")
    try:
        header = ast.literal_eval(
            blob[8 + size : 8 + size + n].decode("utf-8" if version[0] == 3 else "latin1")
        )
    except (ValueError, SyntaxError, UnicodeError, RecursionError) as exc:
        raise GeometryFormatError("Invalid NPY dictionary") from exc
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        raise GeometryFormatError("Unexpected NPY header keys")
    descr, shape = header["descr"], header["shape"]
    if header["fortran_order"] is not False:
        raise GeometryFormatError("Fortran-order arrays are not supported")
    if not isinstance(descr, str) or not re.fullmatch(r"[<>|=][fiu][1248]", descr):
        raise GeometryFormatError("Only primitive numeric arrays are supported; no object/pickle")
    codes = {
        "f4": "f",
        "f8": "d",
        "i1": "b",
        "i2": "h",
        "i4": "i",
        "i8": "q",
        "u1": "B",
        "u2": "H",
        "u4": "I",
        "u8": "Q",
    }
    if descr[1:] not in codes or (descr[0] == "|" and descr[2:] != "1"):
        raise GeometryFormatError(f"Unsupported numeric dtype {descr}")
    if (
        not isinstance(shape, tuple)
        or len(shape) not in (1, 2)
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in shape)
    ):
        raise GeometryFormatError("Expected a one/two-dimensional numeric array")
    count = math.prod(shape)
    width = int(descr[2:])
    data = blob[8 + size + n :]
    if count * width > MAX_MEMBER_BYTES or len(data) != count * width:
        raise GeometryFormatError("Array byte count differs from shape/dtype or exceeds limit")
    endian = descr[0] if descr[0] in "<>" else ("<" if sys.byteorder == "little" else ">")
    values = [v[0] for v in struct.iter_unpack(endian + codes[descr[1:]], data)]
    return shape, values, descr


def _relation(a, b, c, d, epsilon):
    """Crossing / overlap / contact / disjoint for two nonzero segments."""
    ux, uy, vx, vy = b[0] - a[0], b[1] - a[1], d[0] - c[0], d[1] - c[1]
    lu, lv = math.hypot(ux, uy), math.hypot(vx, vy)

    def side(p, q, r, length):
        area = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        tol = epsilon * length
        return 1 if area > tol else -1 if area < -tol else 0

    s1, s2 = side(a, b, c, lu), side(a, b, d, lu)
    s3, s4 = side(c, d, a, lv), side(c, d, b, lv)
    if s1 * s2 < 0 and s3 * s4 < 0:
        return "crossing"
    if s1 == s2 == s3 == s4 == 0:
        t1 = ((c[0] - a[0]) * ux + (c[1] - a[1]) * uy) / lu
        t2 = ((d[0] - a[0]) * ux + (d[1] - a[1]) * uy) / lu
        common = min(lu, max(t1, t2)) - max(0.0, min(t1, t2))
        return "overlap" if common > epsilon else "touch" if common >= -epsilon else None

    def inside(p, q, r):
        return (
            min(p[0], q[0]) - epsilon <= r[0] <= max(p[0], q[0]) + epsilon
            and min(p[1], q[1]) - epsilon <= r[1] <= max(p[1], q[1]) + epsilon
        )

    if (
        (s1 == 0 and inside(a, b, c))
        or (s2 == 0 and inside(a, b, d))
        or (s3 == 0 and inside(c, d, a))
        or (s4 == 0 and inside(c, d, b))
    ):
        return "touch"
    return None


def check_frame(points, lines, materials, *, epsilon_um=EPSILON_UM, max_examples=20):
    """Count all anomalous pairs; bound saved examples, not the actual counts."""
    if not math.isfinite(epsilon_um) or epsilon_um <= 0:
        raise ValueError("epsilon_um must be finite and positive")
    issues, structural = [], Counter()

    def issue(kind, index):
        structural[kind] += 1
        if len(issues) < max_examples:
            issues.append(dict(kind=kind, index=index))

    bad_points = set()
    if not points or not lines:
        issue("empty_graph", None)
    for i, p in enumerate(points):
        if len(p) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p):
            bad_points.add(i)
            issue("nonfinite_or_bad_point", i)
    if len(materials) != len(points):
        issue("material_length", None)
    for i, m in enumerate(materials):
        if isinstance(m, bool) or not isinstance(m, int) or m not in (0, 1):
            issue("material_code", i)
    segments, seen = [], set()
    for i, line in enumerate(lines):
        if len(line) != 2 or any(
            isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < len(points) for v in line
        ):
            issue("invalid_edge_index", i)
            continue
        u, v = line
        key = tuple(sorted(line))
        if key in seen:
            issue("duplicate_edge", i)
        seen.add(key)
        if u in bad_points or v in bad_points:
            continue
        a, b = points[u], points[v]
        if math.dist(a, b) <= epsilon_um:
            issue("zero_length_edge", i)
            continue
        segments.append(
            (min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1]), i, u, v, a, b)
        )
    segments.sort(key=lambda s: (s[0], s[4]))
    active, pairs, touches = [], [], []
    counts = Counter()
    for seg in segments:
        xmin, _xmax, ymin, ymax, i, u, v, a, b = seg
        active = [s for s in active if s[1] >= xmin - epsilon_um]
        for prev in active:
            if prev[3] < ymin - epsilon_um or prev[2] > ymax + epsilon_um:
                continue
            duplicate = {u, v} == {prev[5], prev[6]}
            relation = (
                "duplicate_edge" if duplicate else _relation(a, b, prev[7], prev[8], epsilon_um)
            )
            if relation is None:
                continue
            shared = u in (prev[5], prev[6]) or v in (prev[5], prev[6])
            if relation == "touch" and shared:
                continue
            counts[relation] += 1
            example = dict(
                kind=relation,
                edge_ids=sorted([i, prev[4]]),
                segments=(
                    [[list(a), list(b)], [list(prev[7]), list(prev[8])]]
                    if i < prev[4]
                    else [[list(prev[7]), list(prev[8])], [list(a), list(b)]]
                ),
            )
            if relation == "touch":
                if len(touches) < min(5, max_examples):
                    touches.append(example)
            elif len(pairs) < max_examples:
                pairs.append(example)
        active.append(seg)
    invalid = bool(structural or counts["crossing"] or counts["overlap"])
    return dict(
        invalid=invalid,
        point_count=len(points),
        line_count=len(lines),
        proper_crossing_pair_count=counts["crossing"],
        overlap_pair_count=counts["overlap"],
        duplicate_edge_count=structural["duplicate_edge"],
        duplicate_edge_pair_count=counts["duplicate_edge"],
        nonadjacent_touch_pair_count=counts["touch"],
        structural_issue_counts=dict(structural),
        structural_examples=issues,
        pair_examples=pairs,
        touch_examples=touches,
        examples_truncated=(
            counts["crossing"] + counts["overlap"] + counts["duplicate_edge"] > len(pairs)
        ),
    )


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(65536), b""):
            h.update(part)
    return h.hexdigest()


def _archive(path, steps):
    """Yield (step, checked-frame or None, error or None) without extracting files."""
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            names = [v.filename for v in infos]
            if len(names) != len(set(names)) or sum(v.file_size for v in infos) > MAX_ARCHIVE_BYTES:
                raise GeometryFormatError(
                    "Duplicate archive members or uncompressed archive too large"
                )
            expected = {f"{k}_{s:03d}.npy" for s in steps for k in ["pred", "lines", "mat"]}
            extras = sorted(set(names) - expected)
            if extras:
                yield None, None, "Unexpected archive members/steps: " + repr(extras[:5])
            for step in steps:
                try:
                    arrays = {}
                    for key in ["pred", "lines", "mat"]:
                        name = f"{key}_{step:03d}.npy"
                        info = z.getinfo(name)
                        if info.file_size > MAX_MEMBER_BYTES:
                            raise GeometryFormatError("Array member exceeds byte limit")
                        shape, flat, dtype = read_npy(z.read(name))
                        if key in ("pred", "lines"):
                            if len(shape) != 2 or shape[1] != 2:
                                raise GeometryFormatError(f"{key} must have shape (N, 2)")
                            if (key == "pred" and dtype[1] != "f") or (
                                key == "lines" and dtype[1] not in "iu"
                            ):
                                raise GeometryFormatError(f"{key} has wrong numeric dtype")
                            arrays[key] = list(zip(flat[::2], flat[1::2], strict=True))
                        else:
                            if len(shape) != 1 or dtype[1] not in "iu":
                                raise GeometryFormatError(
                                    "mat must be a one-dimensional integer array"
                                )
                            arrays[key] = flat
                    yield step, check_frame(arrays["pred"], arrays["lines"], arrays["mat"]), None
                except (KeyError, ValueError, struct.error, zipfile.BadZipFile, OSError) as exc:
                    yield step, None, f"{type(exc).__name__}: {exc}"
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        yield None, None, f"{type(exc).__name__}: {exc}"


def scan_dataset(folder, expected, *, progress=None):
    """Inspect all requested frames, with independent completeness per split/mode."""
    folder = Path(folder)
    groups, frames, errors, sources = {}, [], [], {}
    for recipe in expected:
        rid, split, steps = recipe["rid"], recipe["split"], recipe["steps"]
        if (
            not re.fullmatch(r"r\d{4}", rid)
            or split not in ("validation", "test")
            or not steps
            or steps != list(range(1, len(steps) + 1))
        ):
            raise GeometryFormatError("Invalid expected recipe/split/step sequence")
        for mode in ["onestep", "rollout"]:
            key = f"{split}/{mode}"
            group = groups.setdefault(
                key,
                dict(
                    frames_expected=0,
                    frames_checked=0,
                    read_error_count=0,
                    known_invalid_frame_count=0,
                    self_intersection_frame_count=0,
                    overlap_frame_count=0,
                    duplicate_edge_frame_count=0,
                    duplicate_edge_count=0,
                    structural_invalid_frame_count=0,
                    touch_only_frame_count=0,
                    proper_crossing_pair_count=0,
                    overlap_pair_count=0,
                    affected_recipe_ids=[],
                    affected_frames=[],
                ),
            )
            group["frames_expected"] += len(steps)
            path = folder / f"{rid}_{mode}.npz"
            if path.is_file():
                sources[path.name] = _sha(path)
            for step, result, error in _archive(path, steps):
                if error:
                    group["read_error_count"] += 1
                    errors.append(
                        dict(
                            rid=rid,
                            split=split,
                            mode=mode,
                            step=step,
                            source=path.name,
                            error=error,
                        )
                    )
                    continue
                group["frames_checked"] += 1
                row = dict(rid=rid, split=split, mode=mode, step=step, source=path.name, **result)
                frames.append(row)
                group["proper_crossing_pair_count"] += result["proper_crossing_pair_count"]
                group["overlap_pair_count"] += result["overlap_pair_count"]
                group["duplicate_edge_count"] += result["duplicate_edge_count"]
                for count, present in [
                    ("self_intersection_frame_count", result["proper_crossing_pair_count"] > 0),
                    ("overlap_frame_count", result["overlap_pair_count"] > 0),
                    ("duplicate_edge_frame_count", result["duplicate_edge_count"] > 0),
                    ("structural_invalid_frame_count", bool(result["structural_issue_counts"])),
                    (
                        "touch_only_frame_count",
                        not result["invalid"] and result["nonadjacent_touch_pair_count"] > 0,
                    ),
                ]:
                    group[count] += int(present)
                if result["invalid"]:
                    group["known_invalid_frame_count"] += 1
                    group["affected_recipe_ids"].append(rid)
                    group["affected_frames"].append(
                        dict(
                            rid=rid,
                            step=step,
                            proper_crossing_pairs=result["proper_crossing_pair_count"],
                            overlap_pairs=result["overlap_pair_count"],
                            duplicate_edges=result["duplicate_edge_count"],
                            structural_issue_counts=result["structural_issue_counts"],
                        )
                    )
            if progress:
                progress(rid, mode, group["frames_checked"])
    for group in groups.values():
        group["complete"] = (
            group["read_error_count"] == 0 and group["frames_checked"] == group["frames_expected"]
        )
        group["invalid_geometry_count"] = (
            group["known_invalid_frame_count"] if group["complete"] else None
        )
        group["affected_recipe_ids"] = sorted(set(group["affected_recipe_ids"]))
        group["affected_recipe_count"] = len(group["affected_recipe_ids"])
    unchanged = all(
        (folder / name).is_file() and _sha(folder / name) == sha for name, sha in sources.items()
    )
    if not unchanged:
        raise RuntimeError("An inspected prediction file changed during geometry inspection")
    return dict(
        definition=DEFINITION,
        groups=groups,
        frames=frames,
        errors=errors,
        files_sha256=sources,
        source_files_unchanged=unchanged,
    )


def write_outputs(out, result):
    """All frames retained; example pairs capped per frame as documented."""
    out = Path(out)
    (out / "geometry_diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    columns = [
        "rid",
        "split",
        "mode",
        "step",
        "source",
        "invalid",
        "point_count",
        "line_count",
        "proper_crossing_pair_count",
        "overlap_pair_count",
        "duplicate_edge_count",
        "duplicate_edge_pair_count",
        "nonadjacent_touch_pair_count",
    ]
    with (out / "geometry_frames.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(result["frames"])
    lines = [
        "MGN 예측 선분 형상 검사",
        CONTRACT,
        "이상 건수는 교차 선분 쌍 수가 아니라 문제가 있는 예측 프레임 수입니다.",
        "끝점 접촉은 별도 진단이며 단독으로 이상 건수에 포함하지 않습니다.",
        "동일한 두 노드의 중복 연결은 출력 구조 문제로 따로 표시하고 "
        "기하학적 중첩에서는 제외합니다.",
    ]
    for key, group in result["groups"].items():
        lines.append(
            f"{key}: 검사 {group['frames_checked']}/{group['frames_expected']}, "
            f"이상 프레임 {group['invalid_geometry_count']}, "
            f"교차 프레임 {group['self_intersection_frame_count']}, "
            f"서로 다른 선분의 중첩 프레임 {group['overlap_frame_count']}, "
            f"중복 연결 프레임 {group['duplicate_edge_frame_count']}, "
            f"이상 레시피 {group['affected_recipe_count']}개 {group['affected_recipe_ids']}"
        )
    lines.append(
        f"읽기 오류 {len(result['errors'])}건. 읽기 오류가 있으면 해당 집계 건수는 미측정입니다."
    )
    (out / "geometry_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    from fno_adapter import local_input

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(local_input("mgn_input", "mgn_main_s1_nov9")))
    p.add_argument(
        "--output-root", default=str(Path(__file__).resolve().parent / "mgn_geometry_results")
    )
    args = p.parse_args()
    with (Path(args.input) / "metrics_by_step.csv").open(encoding="utf-8-sig", newline="") as f:
        series = {}
        for r in csv.DictReader(f):
            item = series.setdefault(
                r["rid"],
                dict(
                    rid=r["rid"], split={"val": "validation", "test": "test"}[r["split"]], steps=[]
                ),
            )
            item["steps"].append(int(r["step"]))
    for item in series.values():
        item["steps"].sort()
    result = scan_dataset(args.input, list(series.values()))
    out = Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    write_outputs(out, result)
    print((out / "geometry_report.txt").read_text(encoding="utf-8"))
    print("결과:", out)
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
