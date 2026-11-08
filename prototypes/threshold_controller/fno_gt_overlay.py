"""FNO sample SDF -> zero contours -> same-step VTP overlays. Standard library only.

Does not replace provided scores, approve a model, train, or set reference_verified.
Marching squares uses linear interpolation and an explicitly low-connected saddle policy.
Reference: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.find_contours
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import zipfile
from datetime import datetime
from pathlib import Path

from fno_adapter import local_input, sha256, write_json
from mgn_geometry import MAX_ARCHIVE_BYTES, MAX_MEMBER_BYTES, read_npy
from mgn_gt_overlay import CSS, GroundTruth

ROOT = Path(__file__).resolve().parent
CONTRACT = "fno-sdf-gt-overlay-v1"
EXTRACTION = dict(
    method="marching squares; level 0; linear edge interpolation",
    array_axes="[step, y, x]; coordinates from explicit x/y arrays in um",
    negative_phase="solid",
    saddle_policy="negative (low) phase connected across a diagonal",
    caveat="Extraction policy can prevent segment crossings by construction. No crossing count "
    "or formal geometry pass/fail is inferred from these contours.",
)
PAIRS = {
    1: [(3, 0)],
    2: [(0, 1)],
    3: [(3, 1)],
    4: [(1, 2)],
    5: [(0, 1), (2, 3)],
    6: [(0, 2)],
    7: [(2, 3)],
    8: [(2, 3)],
    9: [(0, 2)],
    10: [(0, 3), (1, 2)],
    11: [(1, 2)],
    12: [(1, 3)],
    13: [(0, 1)],
    14: [(0, 3)],
}
PARAM_KEYS = (
    "ion_flux",
    "etchant_flux",
    "oxygen_flux",
    "ion_mean_energy",
    "ion_exponent",
    "mask_width",
    "mask_height",
)


class FnoOverlayError(ValueError):
    pass


def _axis(values):
    return (
        len(values) >= 2
        and all(math.isfinite(v) for v in values)
        and all(b > a for a, b in zip(values, values[1:]))
    )


def load_sdf(path):
    """Read only primitive numeric arrays; preserve grid orientation and numbered steps."""
    try:
        with zipfile.ZipFile(path) as z:
            info = z.infolist()
            if (
                len({i.filename for i in info}) != len(info)
                or sum(i.file_size for i in info) > MAX_ARCHIVE_BYTES
            ):
                raise FnoOverlayError("Duplicate archive members or oversized SDF archive")
            arrays = {}
            for key in ["phi", "step", "x", "y"]:
                member = z.getinfo(key + ".npy")
                if member.file_size > MAX_MEMBER_BYTES:
                    raise FnoOverlayError("SDF array exceeds size limit")
                arrays[key] = read_npy(z.read(key + ".npy"), allowed_ndim=(1, 2, 3))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise FnoOverlayError(f"{Path(path).name}: {exc}") from exc
    shape, field, dtype = arrays["phi"]
    steps = arrays["step"][1]
    x = arrays["x"][1]
    y = arrays["y"][1]
    if len(shape) != 3 or shape != (len(steps), len(y), len(x)) or dtype[1] != "f":
        raise FnoOverlayError("phi must have shape (steps, len(y), len(x)) and float dtype")
    for key in ["step", "x", "y"]:
        if len(arrays[key][0]) != 1:
            raise FnoOverlayError(f"{key} must be one dimensional")
    if not steps or arrays["step"][2][1] not in "iu" or steps != list(range(1, len(steps) + 1)):
        raise FnoOverlayError("step must be integers t001..tN, with no missing/duplicate steps")
    if not _axis(x) or not _axis(y) or not all(math.isfinite(v) for v in field):
        raise FnoOverlayError("Axes must increase and all SDF/coordinate values must be finite")
    return dict(shape=shape, field=field, steps=steps, x=x, y=y)


def marching_squares(field, x, y):
    """Extract shared-edge vertices; do not close contours at the domain boundary."""
    nx, ny = len(x), len(y)
    if (
        not _axis(x)
        or not _axis(y)
        or len(field) != nx * ny
        or not all(math.isfinite(v) for v in field)
    ):
        raise FnoOverlayError("Invalid scalar field shape, axes, or nonfinite values")
    points, lines, vertex_ids, seen = [], [], {}, set()
    ambiguous = plateaus = duplicates = zero_segments = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            vals = [field[yy * nx + xx] for xx, yy in corners]
            mask = sum((1 << k) for k, v in enumerate(vals) if v < 0)
            if all(v == 0 for v in vals):
                plateaus += 1
            if mask not in PAIRS:
                continue
            ambiguous += int(mask in (5, 10))

            def vertex(edge):
                ca, cb = [(0, 1), (1, 2), (2, 3), (3, 0)][edge]
                a, b = corners[ca], corners[cb]
                va, vb = vals[ca], vals[cb]
                if va == 0:
                    key = ("vertex", *a)
                    t = 0.0
                elif vb == 0:
                    key = ("vertex", *b)
                    t = 1.0
                else:
                    key = ("edge", *sorted((a, b)))
                    t = va / (va - vb)
                if key not in vertex_ids:
                    vertex_ids[key] = len(points)
                    points.append(
                        (x[a[0]] + t * (x[b[0]] - x[a[0]]), y[a[1]] + t * (y[b[1]] - y[a[1]]))
                    )
                return vertex_ids[key]

            for a, b in PAIRS[mask]:
                u, v = vertex(a), vertex(b)
                if u == v:
                    zero_segments += 1
                    continue
                edge = tuple(sorted((u, v)))
                if edge in seen:
                    duplicates += 1
                    continue
                seen.add(edge)
                lines.append((u, v))
    adjacency = {}
    for a, b in lines:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    unseen = set(adjacency)
    components = 0
    while unseen:
        stack = [unseen.pop()]
        components += 1
        while stack:
            for n in adjacency[stack.pop()]:
                if n in unseen:
                    unseen.remove(n)
                    stack.append(n)
    diag = dict(
        components=components,
        open_endpoints=sum(len(v) == 1 for v in adjacency.values()),
        junction_vertices=sum(len(v) > 2 for v in adjacency.values()),
        ambiguous_cells=ambiguous,
        zero_plateau_cells=plateaus,
        duplicate_extracted_segments=duplicates,
        zero_length_extracted_segments=zero_segments,
    )
    return dict(points=points, lines=lines, diagnostics=diag)


def svg_overlay(pred, gt, *, zoom=False):
    """Physical coordinate overlay; bottom zoom uses both predicted and reference minima."""
    if zoom:
        pp = min(pred["points"], key=lambda p: p[1])
        gg = min(gt["points"], key=lambda p: p[1])
        y0, y1 = min(pp[1], gg[1]) - 1.5, max(pp[1], gg[1]) + 5
        bottom_points = [p for p in pred["points"] + gt["points"] if p[1] <= y1]
        x0, x1 = min(p[0] for p in bottom_points) - 1, max(p[0] for p in bottom_points) + 1
    else:
        allpoints = pred["points"] + gt["points"]
        x0, x1 = min(p[0] for p in allpoints) - 1, max(p[0] for p in allpoints) + 1
        y0, y1 = min(p[1] for p in allpoints) - 1, max(p[1] for p in allpoints) + 1

    def path(g):
        return " ".join(
            f"M{g['points'][a][0]:.7f} {-g['points'][a][1]:.7f}"
            f"L{g['points'][b][0]:.7f} {-g['points'][b][1]:.7f}"
            for a, b in g["lines"]
        )

    return (
        f'<svg class="plot" viewBox="{x0} {-y1} {x1 - x0} {y1 - y0}" '
        'preserveAspectRatio="xMidYMid meet" role="img" aria-label="FNO 정답 예측 비교">'
        f'<rect class="bg" x="{x0}" y="{-y1}" width="{x1 - x0}" height="{y1 - y0}"/>'
        f'<path class="gt" d="{path(gt)}"/><path class="pred" d="{path(pred)}"/></svg>'
    )


def _provided_rows(samples):
    path = samples / "metrics_by_step.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = {}
        for r in csv.DictReader(f):
            mode = {"one-step": "onestep", "rollout": "rollout"}.get(r.get("방식"))
            if mode is None or not re.fullmatch(r"t\d{3}", r.get("스텝", "")):
                raise FnoOverlayError("Unrecognized provided score mode/step")
            key = (r["레시피"], mode, int(r["스텝"][1:]))
            if key in rows:
                raise FnoOverlayError("Duplicate provided per-step score row")
            rows[key] = r
    return rows


def _number(value):
    try:
        v = float(value)
    except (ValueError, TypeError):
        return None
    return v if math.isfinite(v) else None


def _verify_mapping(gt, rid, data, declared):
    n = len(data["steps"])
    meta = gt.meta(rid)
    if gt.snapshot_numbers(rid) != list(range(n + 1)) or not meta or meta.get("actual_k") != n:
        raise FnoOverlayError("Snapshot numbering/actual_k does not match prediction steps")
    if not isinstance(declared, dict) or any(k not in declared for k in PARAM_KEYS):
        raise FnoOverlayError("The seven declared recipe parameters are required")
    for key in PARAM_KEYS:
        a, b = _number(declared[key]), _number(meta.get("params", {}).get(key))
        if a is None or b is None or not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10):
            raise FnoOverlayError(f"Recipe parameter differs from ground truth: {key}")
    return dict(
        ok=True,
        snapshot_numbers=f"t000..t{n:03d}",
        parameters_matched=len(PARAM_KEYS),
        rule="phi[index] at step[index]=k -> snapshots/t{k:03d}.vtp",
    )


def render_html(summary, plots):
    def e(v):
        return html.escape(str(v))

    def fmt(v):
        return "미측정" if v is None else f"{v:.6g}"

    body = []
    for c in summary["cases"]:
        cid = c["id"]
        p = plots[cid]
        body.append(
            f'<h2 id="{e(cid)}">{e(c["rid"])} · {e(c["mode"])} · t{c["step"]:03d}</h2>'
            '<div class="card"><table><tr><th>항목</th><th>제공값</th><th>이번 진단 계산</th></tr>'
            f"<tr><td>정답 깊이 (µm)</td><td>{fmt(c['provided_gt_depth_um'])}</td>"
            f"<td>{fmt(c['gt_depth_um'])}</td></tr>"
            f"<tr><td>예측 깊이 (µm)</td><td>{fmt(c['provided_pred_depth_um'])}</td>"
            f"<td>{fmt(c['recomputed_pred_depth_um'])}</td></tr>"
            f"<tr><td>깊이 오차 (%)</td><td>{fmt(c['provided_depth_error_pct'])}</td>"
            f"<td>{fmt(c['recomputed_depth_error_pct'])}</td></tr>"
            f"<tr><td>바닥 CD 오차 (%)</td><td>{fmt(c['provided_bottom_cd_error_pct'])}</td>"
            "<td>재계산 안 함 — 측정 높이 기준 미확정</td></tr></table>"
            f"<p>추출 경계 {c['contour_diagnostics']['components']}묶음 · "
            f"연결 선택이 필요한 격자 {c['contour_diagnostics']['ambiguous_cells']}개</p>"
            "<details><summary>경계 추출 진단 보기</summary>"
            f"<pre>{e(json.dumps(c['contour_diagnostics'], ensure_ascii=False, indent=2))}"
            "</pre></details>"
            f'<p class="muted">예측 {e(c["prediction_file"])} / phi[{c["array_index"]}], '
            f"정답 {e(c['ground_truth_member'])}</p></div>"
            f'<div class="two"><div class="card"><h3>전체 형상</h3>{p["full"]}</div>'
            f'<div class="card"><h3>바닥 확대</h3>{p["bottom"]}</div></div>'
        )
    errors = "".join(f"<li>{e(x)}</li>" for x in summary["errors"])
    links = " · ".join(
        f'<a href="#{e(c["id"])}">{e(c["rid"])} {e(c["mode"])}</a>' for c in summary["cases"]
    )
    return (
        '<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>FNO 샘플 정답 비교</title><style>{CSS} a{{color:var(--pred)}}</style><main>"
        "<h1>FNO 샘플 · 정답 형상 비교</h1>"
        f"<p>샘플 {summary['sample_recipe_count']}개 조건 · "
        f"예측 {summary['prediction_frames_checked']}개 프레임 · 상태 {e(summary['status'])}</p>"
        '<div class="banner">제공된 샘플만 확인합니다. '
        "전체 validation의 형상 합격 판정이 아닙니다. "
        "경계 추출은 선의 교차를 피하도록 연결할 수 있으므로 "
        "자기교차가 없다는 이유로 모델 정상 판정을 하지 않습니다. "
        "reference_verified=False, 형상 이상 건수는 계속 미측정입니다.</div>"
        "<p>회색: 정답 VTP 원본 선분 · 파랑: FNO φ=0 경계 · 좌표 단위 µm · 깊은 곳은 아래쪽</p>"
        "<p>전체 그림은 VTP의 재료 내부 경계도 함께 표시할 수 있습니다. "
        "모든 회색 선분이 φ=0 경계에 대응한다는 뜻은 아닙니다. "
        "바닥 확대와 깊이 수치도 함께 확인하세요.</p>"
        "<p>원본 VTP에서 계산한 정답 깊이와 제공 평가표의 정답 깊이는 "
        "소폭 다를 수 있어 함께 표시합니다. "
        "차이의 원인은 아직 확인하지 않았으며 제공 점수를 수정하지 않습니다.</p>"
        "<p>모든 샘플 단계의 진단은 per_step_diagnostics.csv에 저장합니다. "
        "아래는 각 방식의 마지막 단계입니다.</p>"
        + f'<nav class="card">{links}</nav>'
        + ('<div class="banner"><ul>' + errors + "</ul></div>" if errors else "")
        + "".join(body)
        + "</main></html>"
    )


def run(samples, gt_zip, output_root, provenance=None):
    samples, gt_zip = Path(samples), Path(gt_zip)
    source_paths = sorted(p for p in samples.iterdir() if p.is_file())
    before = {str(p): sha256(p) for p in source_paths}
    gt_hash = sha256(gt_zip)
    declared_hash = (provenance or {}).get("sha256")
    if declared_hash and declared_hash != gt_hash:
        raise FnoOverlayError("Ground-truth ZIP hash mismatch")
    readme = json.loads((samples / "README.json").read_text(encoding="utf-8-sig"))
    if "validation" not in str(readme.get("데이터 구분", "")).lower():
        raise FnoOverlayError("Sample split must explicitly be validation")
    recipes = readme.get("레시피", {})
    if not recipes:
        raise FnoOverlayError("No sample recipe metadata")
    provided = _provided_rows(samples)
    s = dict(
        contract=CONTRACT,
        extraction=EXTRACTION,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        model_as_provided=readme.get("모델"),
        sample_recipe_count=len(recipes),
        prediction_frames_checked=0,
        reference_verified=False,
        training_started=False,
        active_model_changed=False,
        mapping={},
        cases=[],
        errors=[],
        ground_truth_sha256=gt_hash,
        ground_truth_source=provenance or {},
    )
    rows, plots = [], {}
    gt = GroundTruth(gt_zip)
    try:
        for rid, meta in sorted(recipes.items()):
            if not re.fullmatch(r"r\d{4}", rid) or not 300 <= int(rid[1:]) <= 339:
                s["errors"].append(f"{rid}: expected validation recipe r0300..r0339")
                continue
            for mode in ["onestep", "rollout"]:
                try:
                    path = samples / f"{rid}_{mode}.npz"
                    data = load_sdf(path)
                    mapping = _verify_mapping(gt, rid, data, meta.get("레시피 값"))
                    s["mapping"][f"{rid}/{mode}"] = mapping
                    nx, ny = len(data["x"]), len(data["y"])
                    for index, step in enumerate(data["steps"]):
                        field = data["field"][index * nx * ny : (index + 1) * nx * ny]
                        contour = marching_squares(field, data["x"], data["y"])
                        if not contour["lines"]:
                            raise FnoOverlayError(f"t{step:03d}: no zero contour")
                        ref = gt.frame(rid, step)
                        if ref["max_abs_z"] > 1e-7:
                            raise FnoOverlayError("Reference is not a 2D surface")
                        pred_depth = -min(p[1] for p in contour["points"])
                        truth_depth = -min(p[1] for p in ref["points"])
                        reported = provided.get((rid, mode, step))
                        if reported is None:
                            raise FnoOverlayError(f"t{step:03d}: no provided per-step score")
                        row = dict(
                            rid=rid,
                            mode=mode,
                            step=step,
                            array_index=index,
                            prediction_file=path.name,
                            ground_truth_member=f"{rid}/snapshots/t{step:03d}.vtp",
                            recomputed_pred_depth_um=pred_depth,
                            gt_depth_um=truth_depth,
                            recomputed_depth_error_um=pred_depth - truth_depth,
                            recomputed_depth_error_pct=(
                                100 * (pred_depth - truth_depth) / truth_depth
                                if truth_depth > 0
                                else None
                            ),
                            provided_pred_depth_um=_number(reported.get("예측깊이(um)")),
                            provided_gt_depth_um=_number(reported.get("정답깊이(um)")),
                            provided_depth_error_pct=_number(reported.get("깊이오차(%)")),
                            provided_bottom_cd_error_pct=_number(reported.get("CD_바닥오차(%)")),
                            contour_diagnostics=contour["diagnostics"],
                        )
                        rows.append(row)
                        s["prediction_frames_checked"] += 1
                        if index == len(data["steps"]) - 1:
                            cid = f"{rid}_{mode}_t{step:03d}"
                            s["cases"].append(dict(id=cid, **row))
                            plots[cid] = dict(
                                full=svg_overlay(contour, ref),
                                bottom=svg_overlay(contour, ref, zoom=True),
                                points=contour["points"],
                                lines=contour["lines"],
                            )
                except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
                    s["errors"].append(f"{rid}/{mode}: {type(exc).__name__}: {exc}")
    finally:
        gt.close()
    if before != {str(p): sha256(p) for p in source_paths} or sha256(gt_zip) != gt_hash:
        raise FnoOverlayError("A source file changed during inspection")
    s["sources_unchanged"] = True
    s["status"] = "ok" if not s["errors"] else "partial" if s["cases"] else "failed"
    out = Path(output_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "summary.json", s)
    write_json(out / "per_step_diagnostics.json", dict(rows=rows))
    write_json(out / "source_files.json", dict(files_sha256=before, gt_sha256=gt_hash))
    if rows:
        with (out / "per_step_diagnostics.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "contour_diagnostics"])
            w.writeheader()
            w.writerows({k: v for k, v in r.items() if k != "contour_diagnostics"} for r in rows)
    cases_dir = out / "cases"
    cases_dir.mkdir()
    for c in s["cases"]:
        cid = c["id"]
        write_json(
            cases_dir / (cid + ".json"),
            dict(case=c, predicted_contour={k: plots[cid][k] for k in ["points", "lines"]}),
        )
        for view in ["full", "bottom"]:
            svg = plots[cid][view].replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
            (cases_dir / f"{cid}_{view}.svg").write_text(
                svg.replace("</svg>", f"<style>{CSS}</style></svg>"), encoding="utf-8"
            )
    (out / "report.html").write_text(render_html(s, plots), encoding="utf-8")
    return out, s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", default=str(local_input("fno_samples", "samples_main_s1")))
    p.add_argument("--gt-zip", default=str(local_input("mgn_gt_zip", "etch-raw-n400.zip")))
    p.add_argument("--output-root", default=str(ROOT / "fno_gt_overlay_results"))
    args = p.parse_args()
    try:
        cfg = json.loads((ROOT / "local_paths.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    out, s = run(args.samples, args.gt_zip, args.output_root, cfg.get("mgn_gt_provenance"))
    print(
        "샘플 조건:",
        s["sample_recipe_count"],
        "검사 프레임:",
        s["prediction_frames_checked"],
        "상태:",
        s["status"],
    )
    print("보고서:", out / "report.html")
    if s["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
