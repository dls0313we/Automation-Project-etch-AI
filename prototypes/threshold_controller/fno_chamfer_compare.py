"""Independent FNO Chamfer recalculation on the provided samples; standard library only.

Recomputes Chamfer between the reference SDF and the predicted SDF (phi=0 contours) for the
sample recipes and compares it with the provided metrics_by_step.csv. Separate diagnostic:
it does not modify sources, scores, thresholds or harness decisions, and it is NOT a
reproduction of the provider's code (that code was not available). IoU is out of scope.

  python fno_chamfer_compare.py
  python fno_chamfer_compare.py --samples <samples_main_s1> --truth <sdf folder>
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from fno_adapter import PROVIDER_DEFINITION, local_input, sha256, write_json
from fno_gt_overlay import FnoOverlayError, _number, _provided_rows, load_sdf, marching_squares
from fno_sdf_compare import load_truth, resolve_folder, validate_pair
from mgn_gt_overlay import CSS

ROOT = Path(__file__).resolve().parent
CONTRACT = "fno-sdf-chamfer-diagnostic-v1"
SPACING_UM = 0.1
# Numerical agreement check only; NOT a model quality/acceptance limit.
MATCH_TOLERANCE_UM = 1e-6
CELL_UM = 0.25  # spatial-index bucket size; affects speed only, not results
PROVIDED_TEXT = (
    "양쪽 윤곽선을 0.1 µm 간격으로 찍고, 각 점에서 상대 윤곽선까지 최단거리의 전체 평균 "
    "(samples_main_s1/README.json)"
)
# Fixed before comparing with the provided scores. Do not tune to match them.
METHOD = dict(
    boundary=(
        "phi=0 contours of the reference SDF (t=k) and the predicted SDF (step=k) on the same "
        "grid, extracted with fno_gt_overlay.marching_squares (linear edge interpolation, "
        "negative/solid phase connected at saddle cells). skimage.find_contours is not used."
    ),
    chaining=(
        "Extracted segments are chained into polylines, split at junction vertices; "
        "closed loops stay closed. Contours are not closed at the grid boundary."
    ),
    resampling=(
        "Per polyline, points at arc length 0, 0.1, 0.2, ... um, plus the final vertex when the "
        "remaining length exceeds 1e-9 um. Closed loops do not repeat their start point."
    ),
    distance=(
        "Euclidean distance from each resampled point to the nearest point of the other "
        "contour's piecewise-linear polyline (point-to-segment), not to its resampled points."
    ),
    aggregation=(
        "Pooled mean over all resampled points of both contours: "
        "(sum d(truth->pred) + sum d(pred->truth)) / (N_truth + N_pred)."
    ),
)
# Interpretations declared together with METHOD; reported for sensitivity only, never used
# for the match status or to select a formula after seeing the provided values.
ALTERNATIVES = dict(
    symmetric_mean_point_to_polyline="(mean d(truth->pred) + mean d(pred->truth)) / 2",
    pooled_point_to_point="pooled mean, distance to the other contour's resampled points",
    team_sum_point_to_point=(
        "mean d(truth->pred) + mean d(pred->truth), point-to-point "
        "(form of src/etchpipe/eval/metrics.py chamfer_distance)"
    ),
)


def point_segment_distance(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = dx * dx + dy * dy
    t = 0.0 if n == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / n))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


class SegmentIndex:
    """Exact nearest point-to-segment distance with a uniform bucket grid."""

    def __init__(self, segments, cell=CELL_UM):
        if not segments:
            raise FnoOverlayError("Empty contour")
        self.cell, self.segments, self.grid = cell, list(segments), defaultdict(list)
        for k, (a, b) in enumerate(self.segments):
            i0, i1 = sorted((math.floor(a[0] / cell), math.floor(b[0] / cell)))
            j0, j1 = sorted((math.floor(a[1] / cell), math.floor(b[1] / cell)))
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    self.grid[i, j].append(k)
        keys = list(self.grid)
        self.bounds = (
            min(k[0] for k in keys),
            max(k[0] for k in keys),
            min(k[1] for k in keys),
            max(k[1] for k in keys),
        )

    def nearest(self, p):
        ci, cj = math.floor(p[0] / self.cell), math.floor(p[1] / self.cell)
        i0, i1, j0, j1 = self.bounds
        r_max = max(abs(ci - i0), abs(ci - i1), abs(cj - j0), abs(cj - j1))
        best, seen, r = math.inf, set(), 0
        while True:
            for i in range(ci - r, ci + r + 1):
                for j in range(cj - r, cj + r + 1):
                    if max(abs(i - ci), abs(j - cj)) != r:
                        continue
                    for k in self.grid.get((i, j), ()):
                        if k not in seen:
                            seen.add(k)
                            a, b = self.segments[k]
                            best = min(best, point_segment_distance(p, a, b))
            # every unvisited cell is at least r*cell away from p
            if best <= r * self.cell or r >= r_max:
                return best
            r += 1


class PointIndex(SegmentIndex):
    def __init__(self, points, cell=CELL_UM):
        super().__init__([(p, p) for p in points], cell)


def polylines(contour):
    """Chain extracted segments; split at junctions (degree != 2); keep cycles closed."""
    pts, lines = contour["points"], contour["lines"]
    adj = defaultdict(list)
    for k, (a, b) in enumerate(lines):
        adj[a].append((b, k))
        adj[b].append((a, k))
    used, out = set(), []
    for start in sorted(v for v in adj if len(adj[v]) != 2):
        for nxt, k in adj[start]:
            if k in used:
                continue
            used.add(k)
            chain, cur = [start], nxt
            while True:
                chain.append(cur)
                if len(adj[cur]) != 2:
                    break
                step = [(m, kk) for m, kk in adj[cur] if kk not in used]
                if not step:
                    break
                cur, kk = step[0]
                used.add(kk)
            out.append(dict(vertices=[pts[i] for i in chain], closed=False))
    for k, (a, b) in enumerate(lines):
        if k in used:
            continue
        used.add(k)
        chain, cur = [a], b
        while cur != a:
            chain.append(cur)
            cur, kk = next((m, kk) for m, kk in adj[cur] if kk not in used)
            used.add(kk)
        out.append(dict(vertices=[pts[i] for i in chain], closed=True))
    return out


def resample(vertices, closed, spacing=SPACING_UM):
    verts = list(vertices) + ([vertices[0]] if closed else [])
    lengths = [math.dist(a, b) for a, b in zip(verts, verts[1:])]
    total = sum(lengths)
    if total == 0:
        return [verts[0]]
    n = math.floor(total / spacing + 1e-9)
    targets = [k * spacing for k in range(n + 1)]
    if closed:
        targets = [s for s in targets if s < total - 1e-9]
    elif total - targets[-1] > 1e-9:
        targets.append(total)
    out, seg, start = [], 0, 0.0
    for s in targets:
        while seg < len(lengths) - 1 and start + lengths[seg] < s:
            start += lengths[seg]
            seg += 1
        a, b = verts[seg], verts[seg + 1]
        t = 0.0 if lengths[seg] == 0 else min(1.0, max(0.0, (s - start) / lengths[seg]))
        out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def _segments(polys):
    segs = []
    for p in polys:
        v = p["vertices"] + ([p["vertices"][0]] if p["closed"] else [])
        segs += list(zip(v, v[1:]))
    return segs


def chamfer_terms(truth_polys, pred_polys, spacing=SPACING_UM):
    tp = [q for p in truth_polys for q in resample(p["vertices"], p["closed"], spacing)]
    pp = [q for p in pred_polys for q in resample(p["vertices"], p["closed"], spacing)]
    if not tp or not pp:
        raise FnoOverlayError("Empty contour")
    ts, ps = SegmentIndex(_segments(truth_polys)), SegmentIndex(_segments(pred_polys))
    tpi, ppi = PointIndex(tp), PointIndex(pp)
    d_tp = [ps.nearest(q) for q in tp]  # truth points -> prediction polyline
    d_pt = [ts.nearest(q) for q in pp]
    e_tp = [ppi.nearest(q) for q in tp]  # truth points -> prediction points
    e_pt = [tpi.nearest(q) for q in pp]
    m_tp, m_pt = sum(d_tp) / len(tp), sum(d_pt) / len(pp)
    return dict(
        n_truth_points=len(tp),
        n_pred_points=len(pp),
        n_truth_polylines=len(truth_polys),
        n_pred_polylines=len(pred_polys),
        mean_truth_to_pred_um=m_tp,
        mean_pred_to_truth_um=m_pt,
        primary_um=(sum(d_tp) + sum(d_pt)) / (len(tp) + len(pp)),
        alternatives=dict(
            symmetric_mean_point_to_polyline=(m_tp + m_pt) / 2,
            pooled_point_to_point=(sum(e_tp) + sum(e_pt)) / (len(tp) + len(pp)),
            team_sum_point_to_point=sum(e_tp) / len(tp) + sum(e_pt) / len(pp),
        ),
    )


def _status(calculated, supplied):
    provided = _number(supplied)
    if provided is None and supplied is not None and str(supplied).strip():
        raise FnoOverlayError("Nonnumeric provided Chamfer score")
    if provided is None:
        return provided, "provided_missing"
    return provided, "match" if abs(calculated - provided) <= MATCH_TOLERANCE_UM else "difference"


def _stats(rows, key):
    diffs = [
        (r[key] - r["provided_um"], r["provided_um"]) for r in rows if r["provided_um"] is not None
    ]
    if not diffs:
        return None
    absd = [abs(d) for d, _ in diffs]
    rel = [abs(d) / p for d, p in diffs if p > 0]
    return dict(
        compared=len(diffs),
        max_abs_diff_um=max(absd),
        median_abs_diff_um=statistics.median(absd),
        mean_signed_diff_um=statistics.fmean(d for d, _ in diffs),
        max_rel_diff=max(rel) if rel else None,
        median_rel_diff=statistics.median(rel) if rel else None,
        within_match_tolerance=sum(a <= MATCH_TOLERANCE_UM for a in absd),
        calculated_larger=sum(d > 0 for d, _ in diffs),
    )


def render_report(s, rows):
    def e(v):
        return html.escape(str(v))

    def f(v, spec=".6g"):
        return "미측정" if v is None else format(v, spec)

    st = s["stats"]["primary"] or {}
    groups = "".join(
        f"<tr><td>{e(g)}</td><td>{v['compared']}</td><td>{f(v['median_abs_diff_um'])}</td>"
        f"<td>{f(v['max_abs_diff_um'])}</td><td>{f(v['median_rel_diff'], '.2%')}</td>"
        f"<td>{f(v['max_rel_diff'], '.2%')}</td></tr>"
        for g, v in s["stats_by_group"].items()
        if v
    )
    alts = "".join(
        f"<tr><td>{e(k)}</td><td>{e(ALTERNATIVES[k])}</td><td>{f(v['median_abs_diff_um'])}</td>"
        f"<td>{f(v['max_abs_diff_um'])}</td><td>{f(v['median_rel_diff'], '.2%')}</td></tr>"
        for k, v in s["stats"]["alternatives"].items()
        if v
    )
    table = "".join(
        f"<tr><td>{e(r['rid'])}</td><td>{e(r['mode'])}</td><td>t{r['step']:03d}</td>"
        f"<td>{f(r['provided_um'])}</td><td>{f(r['calculated_um'])}</td>"
        f"<td>{f(r['diff_um'])}</td><td>{e(r['status'])}</td></tr>"
        for r in rows
    )
    method = "".join(f"<li><b>{e(k)}</b>: {e(v)}</li>" for k, v in METHOD.items())
    errors = "".join(f"<li>{e(v)}</li>" for v in s["errors"])
    return (
        '<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>FNO Chamfer 재계산</title><style>{CSS}</style><main>"
        "<h1>FNO · Chamfer 독립 재계산과 제공 점수 비교</h1>"
        f"<p>샘플 {s['sample_recipe_count']}개 조건 · {s['frames_checked']}개 프레임 · "
        f"상태 {e(s['status'])} · {e(json.dumps(s['comparison_counts'], ensure_ascii=False))}</p>"
        '<div class="banner">별도 진단입니다. 제공자 코드가 없어 설명만 보고 구현했으며 '
        "민서 코드와 동일한 재현이 아닙니다. 제공 점수·임계치·판정(HOLD)은 바꾸지 않습니다. "
        "IoU 제외. reference_verified=False, 실제 학습·교체 없음.</div>"
        f"<p>제공 설명: {e(PROVIDED_TEXT)}</p><h2>사용한 계산 방식 (비교 전에 고정)</h2>"
        f"<ul>{method}</ul>"
        f"<p>수치 일치 허용차 {MATCH_TOLERANCE_UM:g} µm (모델 기준 아님). 주 계산: 최대 차이 "
        f"{f(st.get('max_abs_diff_um'))} µm, 중앙값 {f(st.get('median_abs_diff_um'))} µm, "
        f"상대 차이 중앙값 {f(st.get('median_rel_diff'), '.2%')}.</p>"
        "<h2>조건·방식별 차이 (주 계산)</h2><div class='card'><table>"
        "<tr><th>구분</th><th>비교</th><th>|차이| 중앙값 µm</th>"
        "<th>|차이| 최대 µm</th><th>상대 중앙값</th><th>상대 최대</th></tr>"
        f"{groups}</table></div>"
        "<h2>해석 민감도 (참고, 판정·일치 여부에 사용 안 함)</h2><div class='card'><table>"
        "<tr><th>해석</th><th>정의</th><th>|차이| 중앙값 µm</th><th>|차이| 최대 µm</th>"
        f"<th>상대 중앙값</th></tr>{alts}</table></div>"
        + (f'<div class="banner"><ul>{errors}</ul></div>' if errors else "")
        + "<h2>프레임별</h2><div class='card'><table><tr><th>조건</th><th>방식</th><th>단계</th>"
        "<th>제공 µm</th><th>재계산 µm</th><th>차이 µm</th><th>수치 일치</th></tr>"
        f"{table}</table></div></main></html>"
    )


def run(samples, truth_folder, output_root):
    samples, truth_folder = Path(samples), resolve_folder(truth_folder)
    readme = json.loads((samples / "README.json").read_text(encoding="utf-8-sig"))
    if "validation" not in str(readme.get("데이터 구분", "")).lower():
        raise FnoOverlayError("Samples must explicitly declare validation")
    recipes = readme.get("레시피", {})
    if not recipes:
        raise FnoOverlayError("No sample recipes declared")
    supplied = _provided_rows(samples)
    sources = [p for p in samples.iterdir() if p.is_file()]
    sources += [truth_folder / f"{rid}.npz" for rid in recipes if re.fullmatch(r"r\d{4}", rid)]
    before = {str(p): sha256(p) for p in sources if p.is_file()}
    s = dict(
        contract=CONTRACT,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        provided_definition_text=PROVIDED_TEXT,
        provider_definition=PROVIDER_DEFINITION,
        provider_code_available=False,
        exact_provider_reproduction=False,
        method=METHOD,
        spacing_um=SPACING_UM,
        alternatives_for_sensitivity_only=ALTERNATIVES,
        match_tolerance_um=MATCH_TOLERANCE_UM,
        excluded=["IoU", "depth/CD (see fno_sdf_compare.py)", "full validation set"],
        sample_recipe_count=len(recipes),
        frames_checked=0,
        reference_verified=False,
        training_started=False,
        active_model_changed=False,
        errors=[],
    )
    rows = []
    for rid, meta in sorted(recipes.items()):
        try:
            if not re.fullmatch(r"r\d{4}", rid) or not 300 <= int(rid[1:]) < 340:
                raise FnoOverlayError("Only declared validation recipes r0300..r0339 are accepted")
            truth = load_truth(truth_folder / f"{rid}.npz")
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            s["errors"].append(f"{rid}: {type(exc).__name__}: {exc}")
            continue
        for mode in ["onestep", "rollout"]:
            try:
                pred = load_sdf(samples / f"{rid}_{mode}.npz")
                validate_pair(truth, pred, meta.get("레시피 값"))
                x, y = truth["x"], truth["y"]
                size = len(x) * len(y)
                for index, step in enumerate(pred["steps"]):
                    report = supplied.get((rid, mode, step))
                    if report is None:
                        raise FnoOverlayError(f"t{step:03d}: missing provided CSV row")
                    g = truth["field"][step * size : (step + 1) * size]
                    p = pred["field"][index * size : (index + 1) * size]
                    tpoly, ppoly = (polylines(marching_squares(v, x, y)) for v in (g, p))
                    if not tpoly or not ppoly:
                        raise FnoOverlayError(f"t{step:03d}: missing zero contour")
                    t = chamfer_terms(tpoly, ppoly)
                    provided, status = _status(t["primary_um"], report.get("Chamfer(um)"))
                    row = dict(
                        rid=rid,
                        mode=mode,
                        step=step,
                        provided_um=provided,
                        calculated_um=t["primary_um"],
                        diff_um=None if provided is None else t["primary_um"] - provided,
                        rel_diff=(
                            None if not provided else (t["primary_um"] - provided) / provided
                        ),
                        status=status,
                        n_truth_points=t["n_truth_points"],
                        n_pred_points=t["n_pred_points"],
                        n_truth_polylines=t["n_truth_polylines"],
                        n_pred_polylines=t["n_pred_polylines"],
                        mean_truth_to_pred_um=t["mean_truth_to_pred_um"],
                        mean_pred_to_truth_um=t["mean_pred_to_truth_um"],
                        **{f"alt_{k}_um": v for k, v in t["alternatives"].items()},
                    )
                    rows.append(row)
                    s["frames_checked"] += 1
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                s["errors"].append(f"{rid}/{mode}: {type(exc).__name__}: {exc}")
    after = {str(p): sha256(p) for p in sources if p.is_file()}
    if before != after:
        raise FnoOverlayError("Source files changed during comparison")
    s["sources_unchanged"] = True
    s["comparison_counts"] = dict(sorted(Counter(r["status"] for r in rows).items()))
    s["stats"] = dict(
        primary=_stats(rows, "calculated_um"),
        alternatives={k: _stats(rows, f"alt_{k}_um") for k in ALTERNATIVES},
    )
    s["stats_by_group"] = {
        f"{rid}/{mode}": _stats(
            [r for r in rows if (r["rid"], r["mode"]) == (rid, mode)], "calculated_um"
        )
        for rid, mode in sorted({(r["rid"], r["mode"]) for r in rows})
    }
    s["status"] = (
        "incomplete"
        if s["errors"]
        else "ok"
        if rows and all(r["status"] == "match" for r in rows)
        else "differences"
    )
    out = Path(output_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "summary.json", s)
    write_json(out / "source_files.json", dict(files_sha256=before))
    if rows:
        with (out / "comparisons.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "report.html").write_text(render_report(s, rows), encoding="utf-8")
    return out, s


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--samples", default=str(local_input("fno_samples", "samples_main_s1")))
    p.add_argument("--truth", default=str(local_input("fno_gt_sdf", "sdf")))
    p.add_argument("--output-root", default=str(ROOT / "fno_chamfer_compare_results"))
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    out, s = run(args.samples, args.truth, args.output_root)
    st = s["stats"]["primary"] or {}
    print("비교:", s["frames_checked"], "frames;", s["comparison_counts"], "status", s["status"])
    print("최대 |차이| um:", st.get("max_abs_diff_um"), "/ 상대 중앙값:", st.get("median_rel_diff"))
    print("보고서:", out / "report.html")
    if s["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
