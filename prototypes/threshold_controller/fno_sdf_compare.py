"""Read-only FNO prediction/reference SDF comparison; standard library only.

Checks depth and signed CD against the provided per-step CSV on available samples.
It does not reproduce Chamfer/IoU, approve thresholds or change harness decisions.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import csv
import html
import json
import math
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from fno_adapter import PROVIDER_DEFINITION, local_input, sha256, write_json
from fno_gt_overlay import (
    PARAM_KEYS,
    FnoOverlayError,
    _axis,
    _number,
    _provided_rows,
    load_sdf,
    marching_squares,
    svg_overlay,
)
from mgn_geometry import MAX_ARCHIVE_BYTES, MAX_MEMBER_BYTES, read_npy
from mgn_gt_overlay import CSS

ROOT = Path(__file__).resolve().parent
CONTRACT = "fno-sdf-depth-cd-diagnostic-v1"
# Numerical agreement checks only; these are NOT model quality/acceptance limits.
TOLERANCES = {"um": 1e-6, "percentage_points": 1e-4}
CD_COLUMNS = [("위", 0.1), ("중간", 0.5), ("바닥", 0.9)]


def resolve_folder(path):
    p = Path(path)
    if not list(p.glob("*.npz")) and (p / "sdf").is_dir():
        p = p / "sdf"
    if not p.is_dir() or not list(p.glob("*.npz")):
        raise FnoOverlayError("No reference NPZ files in the selected folder or its sdf subfolder")
    return p


def read_params(blob):
    """Decode one bounded scalar Unicode NPY containing JSON, never pickle/object data."""
    if (
        len(blob) > 65536
        or blob[:6] != b"\x93NUMPY"
        or tuple(blob[6:8]) not in ((1, 0), (2, 0), (3, 0))
    ):
        raise FnoOverlayError("Invalid scalar parameter NPY")
    size = 2 if blob[6] == 1 else 4
    n = int.from_bytes(blob[8 : 8 + size], "little")
    if not 0 < n <= 16384 or len(blob) < 8 + size + n:
        raise FnoOverlayError("Invalid parameter NPY header length")
    try:
        h = ast.literal_eval(blob[8 + size : 8 + size + n].decode("latin1"))
        if not isinstance(h, dict) or set(h) != {"descr", "shape", "fortran_order"}:
            raise ValueError("Unexpected parameter NPY header")
        match = re.fullmatch(r"([<>=])U([0-9]+)", str(h["descr"]))
        if not match or h["shape"] != () or h["fortran_order"] is not False:
            raise ValueError("Parameters must be a scalar Unicode JSON string")
        data = blob[8 + size + n :]
        if len(data) != 4 * int(match[2]):
            raise ValueError("Parameter string size mismatch")
        endian = match[1] if match[1] != "=" else ("<" if sys.byteorder == "little" else ">")
        result = json.loads(data.decode("utf-32-le" if endian == "<" else "utf-32-be").rstrip("\0"))
        if not isinstance(result, dict):
            raise ValueError("Parameters must be a JSON dictionary")
        return result
    except (ValueError, SyntaxError, KeyError, UnicodeError, RecursionError) as exc:
        raise FnoOverlayError(f"Invalid reference parameters: {exc}") from exc


def load_truth(path):
    with zipfile.ZipFile(path) as z:
        info = z.infolist()
        if len({v.filename for v in info}) != len(info):
            raise FnoOverlayError("Duplicate reference archive members")
        if sum(v.file_size for v in info) > MAX_ARCHIVE_BYTES:
            raise FnoOverlayError("Reference archive exceeds size limit")
        if any(
            z.getinfo(k + ".npy").file_size > MAX_MEMBER_BYTES
            for k in ["phi", "t", "x", "y", "params"]
        ):
            raise FnoOverlayError("Reference member exceeds size limit")
        arrays = {
            k: read_npy(z.read(k + ".npy"), allowed_ndim=(1, 3)) for k in ["phi", "t", "x", "y"]
        }
        params = read_params(z.read("params.npy"))
    shape, values, dtype = arrays["phi"]
    t, x, y = (arrays[k][1] for k in ["t", "x", "y"])
    if any(len(arrays[k][0]) != 1 for k in ["t", "x", "y"]):
        raise FnoOverlayError("Reference t/x/y must be one dimensional")
    if shape != (len(t), len(y), len(x)) or dtype[1] != "f":
        raise FnoOverlayError("Reference phi must be float [t,y,x]")
    if not t or arrays["t"][2][1] not in "iu" or t != list(range(len(t))):
        raise FnoOverlayError("Reference timestamps must explicitly be t000..tN")
    if not _axis(x) or not _axis(y) or not all(math.isfinite(v) for v in values):
        raise FnoOverlayError("Invalid reference axes or nonfinite SDF")
    return dict(shape=shape, field=values, steps=t, x=x, y=y, params=params)


def validate_pair(truth, pred, params):
    if truth["steps"] != [0, *pred["steps"]]:
        raise FnoOverlayError("Reference t000..tN and prediction t001..tN do not align")
    for key in ["x", "y"]:
        a, b = truth[key], pred[key]
        if len(a) != len(b) or any(abs(u - v) > 1e-10 for u, v in zip(a, b)):
            raise FnoOverlayError(f"Prediction/reference {key} grids differ")
    if not isinstance(params, dict):
        raise FnoOverlayError("Missing sample recipe parameters")
    for k in PARAM_KEYS:
        a, b = _number(truth["params"].get(k)), _number(params.get(k))
        if a is None or b is None or not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10):
            raise FnoOverlayError(f"Reference/sample recipe parameter differs: {k}")
    if not truth["x"][0] < 0 < truth["x"][-1]:
        raise FnoOverlayError("The x grid must contain x=0 in its interior")


def cross_section(field, x, y, height):
    """Linear interpolation in y, then exact linear zero locations along that row.

    Return the positive interval containing x=0. Domain-truncated intervals stay
    unmeasured; never silently substitute the width of the simulation box.
    """
    if not _axis(x) or not _axis(y) or len(field) != len(x) * len(y):
        raise FnoOverlayError("Invalid cross-section grid")
    if not math.isfinite(height) or not y[0] <= height <= y[-1]:
        return dict(width_um=None, reason="height_outside_grid")
    j = max(0, min(len(y) - 2, bisect.bisect_right(y, height) - 1))
    w = (height - y[j]) / (y[j + 1] - y[j])
    nx = len(x)
    row = [(1 - w) * field[j * nx + i] + w * field[(j + 1) * nx + i] for i in range(nx)]
    if not all(math.isfinite(v) for v in row) or not x[0] < 0 < x[-1]:
        raise FnoOverlayError("Invalid scalar row or missing center coordinate")
    i = max(0, min(nx - 2, bisect.bisect_right(x, 0) - 1))
    center = row[i] + (row[i + 1] - row[i]) * (-x[i]) / (x[i + 1] - x[i])
    if center <= 0:
        return dict(width_um=0.0, reason="center_not_air", center_phi=center)
    roots = set()
    for i, (a, b) in enumerate(zip(row, row[1:])):
        if a == 0:
            roots.add(x[i])
        if b == 0:
            roots.add(x[i + 1])
        if a * b < 0:
            roots.add(x[i] - a * (x[i + 1] - x[i]) / (b - a))
    left, right = [r for r in roots if r < 0], [r for r in roots if r > 0]
    if not left or not right:
        return dict(width_um=None, reason="air_interval_reaches_grid_boundary", center_phi=center)
    a, b = max(left), min(right)
    return dict(width_um=b - a, left_x_um=a, right_x_um=b, center_phi=center, reason="measured")


def percent_error(pred, truth):
    return None if pred is None or truth is None or truth <= 0 else 100 * (pred - truth) / truth


def comparison(column, calculated, supplied, unit):
    provided = _number(supplied)
    if provided is None and supplied is not None and str(supplied).strip():
        raise FnoOverlayError(f"Nonnumeric provided score in {column}")
    if provided is None:
        status = "both_missing" if calculated is None else "provided_missing"
    elif calculated is None:
        status = "calculation_missing"
    else:
        status = "match" if abs(calculated - provided) <= TOLERANCES[unit] else "difference"
    return dict(
        column=column,
        calculated=calculated,
        provided=provided,
        unit=unit,
        difference=None if calculated is None or provided is None else calculated - provided,
        status=status,
    )


def inventory(folder):
    counts = Counter()
    files = sorted(folder.glob("*.npz"))
    for f in files:
        if re.fullmatch(r"r\d{4}\.npz", f.name) and 0 <= int(f.stem[1:]) < 380:
            n = int(f.stem[1:])
            counts["train" if n < 300 else "validation" if n < 340 else "test"] += 1
        elif re.fullmatch(r"ood\d{4}\.npz", f.name):
            counts["ood"] += 1
        else:
            counts["other"] += 1
    return dict(
        file_count=len(files),
        bytes=sum(f.stat().st_size for f in files),
        counts_by_filename=dict(counts),
        scope="File inventory only. Full arrays are parsed only for the sample recipes.",
    )


def render_report(s, plots):
    def e(v):
        return html.escape(str(v))

    def fmt(v):
        return "미측정" if v is None else f"{v:.8g}"

    cards = []
    for case in s["cases"]:
        rows = "".join(
            f"<tr><td>{e(c['column'])}</td><td>{fmt(c['provided'])}</td>"
            f"<td>{fmt(c['calculated'])}</td><td>{fmt(c['difference'])}</td>"
            f"<td>{e(c['status'])}</td></tr>"
            for c in case["comparisons"]
        )
        cards.append(
            f"<h2>{e(case['rid'])} · {e(case['mode'])} · t{case['step']:03d}</h2>"
            '<div class="card"><table><tr><th>항목</th><th>제공값</th><th>재계산</th>'
            f"<th>차이</th><th>수치 일치</th></tr>{rows}</table></div>"
            '<div class="two"><div class="card"><h3>전체 형상</h3>'
            f'{plots[case["id"]]["full"]}</div><div class="card"><h3>바닥 확대</h3>'
            f"{plots[case['id']]['bottom']}</div></div>"
        )
    errors = "".join(f"<li>{e(v)}</li>" for v in s["errors"])
    return (
        '<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>FNO 정답 SDF 비교</title><style>{CSS}</style><main>"
        "<h1>FNO · 정답 SDF와 예측 비교</h1>"
        f"<p>정답 파일 {s['inventory']['file_count']}개 중 예측 샘플 "
        f"{s['sample_recipe_count']}개 조건 · {s['frames_checked']}개 프레임 비교 · "
        f"상태 {e(s['status'])}</p>"
        f"<p>수치 일치 현황: {e(json.dumps(s['comparison_counts'], ensure_ascii=False))}</p>"
        '<div class="banner">제공 점수와의 수치 일치 검사입니다. 모델 합격 판정이 아닙니다. '
        "전체 40개 validation 예측 검증, Chamfer·IoU 재계산, 정답의 물리 검증은 포함하지 않습니다. "
        "reference_verified=False, 실제 학습·교체 없음.</div>"
        "<p>회색: 정답 SDF φ=0 · 파랑: 예측 SDF φ=0 · 단위 µm. "
        "두 경계를 같은 격자·레시피·명시된 단계 번호로 비교합니다.</p>"
        "<p>깊이는 선형 보간 경계 꼭짓점의 −min(y)입니다. 제공자의 0.1 µm 재표본화 "
        "코드와 동일하다고 가정하지 않습니다. CD는 정답 깊이의 10/50/90% 높이에서 "
        "x=0을 포함하는 공기(φ&gt;0) 구간 폭입니다. 음수 오차는 예측이 더 좁다는 뜻입니다.</p>"
        "<p>수치 일치 허용차: 깊이 0.000001 µm, 오차율 0.0001 퍼센트포인트. "
        "이 값은 모델의 허용 오차나 재학습 임계치가 아닙니다. "
        "정답 폭 0 등으로 계산이 정의되지 않으면 미측정으로 남깁니다.</p>"
        "<p>모든 단계의 수치는 comparisons.csv, 대응 근거는 summary.json에 저장합니다. "
        "아래는 각 방식의 마지막 단계입니다.</p>"
        + (f'<div class="banner"><ul>{errors}</ul></div>' if errors else "")
        + "".join(cards)
        + "</main></html>"
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
        inventory=inventory(truth_folder),
        provider_definition=PROVIDER_DEFINITION,
        sample_recipe_count=len(recipes),
        frames_checked=0,
        reference_verified=False,
        training_started=False,
        active_model_changed=False,
        numerical_tolerances=TOLERANCES,
        not_reproduced=["Chamfer", "IoU", "provider's exact 0.1 um resampling code"],
        mapping={},
        cases=[],
        errors=[],
    )
    comparisons, diagnostics, plots = [], [], {}
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
                s["mapping"][f"{rid}/{mode}"] = dict(
                    matched_parameters=len(PARAM_KEYS),
                    matched_axes=True,
                    rule="prediction phi[i] at step=k matches reference t=k; "
                    "reference includes t000",
                    reference_steps=truth["steps"],
                    prediction_steps=pred["steps"],
                )
                x, y = truth["x"], truth["y"]
                size = len(x) * len(y)
                for index, step in enumerate(pred["steps"]):
                    g = truth["field"][step * size : (step + 1) * size]
                    p = pred["field"][index * size : (index + 1) * size]
                    gt, pp = (marching_squares(v, x, y) for v in [g, p])
                    if not gt["lines"] or not pp["lines"]:
                        raise FnoOverlayError(f"t{step:03d}: missing zero contour")
                    gd, pd = (-min(v[1] for v in c["points"]) for c in [gt, pp])
                    if gd <= 0:
                        raise FnoOverlayError(f"t{step:03d}: nonpositive reference depth")
                    report = supplied.get((rid, mode, step))
                    if report is None:
                        raise FnoOverlayError(f"t{step:03d}: missing provided CSV row")
                    row = [
                        ("정답깊이(um)", gd, "um"),
                        ("예측깊이(um)", pd, "um"),
                        ("깊이오차(%)", percent_error(pd, gd), "percentage_points"),
                    ]
                    cds = []
                    for label, fraction in CD_COLUMNS:
                        a, b = (cross_section(v, x, y, -fraction * gd) for v in [g, p])
                        value = percent_error(b["width_um"], a["width_um"])
                        column = f"CD_{label}오차(%)"
                        row.append((column, value, "percentage_points"))
                        cds.append(
                            dict(label=label, height_um=-fraction * gd, truth=a, prediction=b)
                        )
                    entries = [
                        dict(
                            rid=rid,
                            mode=mode,
                            step=step,
                            **comparison(col, val, report.get(col), unit),
                        )
                        for col, val, unit in row
                    ]
                    comparisons.extend(entries)
                    diagnostics.append(dict(rid=rid, mode=mode, step=step, cd=cds))
                    s["frames_checked"] += 1
                    if index == len(pred["steps"]) - 1:
                        cid = f"{rid}_{mode}_t{step:03d}"
                        s["cases"].append(
                            dict(id=cid, rid=rid, mode=mode, step=step, comparisons=entries, cd=cds)
                        )
                        plots[cid] = dict(
                            full=svg_overlay(pp, gt), bottom=svg_overlay(pp, gt, zoom=True)
                        )
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                s["errors"].append(f"{rid}/{mode}: {type(exc).__name__}: {exc}")
    after = {str(p): sha256(p) for p in sources if p.is_file()}
    if before != after:
        raise FnoOverlayError("Source files changed during comparison")
    s["sources_unchanged"] = True
    s["comparison_counts"] = dict(Counter(r["status"] for r in comparisons))
    s["max_abs_difference"] = {
        unit: max(
            (
                abs(r["difference"])
                for r in comparisons
                if r["unit"] == unit and r["difference"] is not None
            ),
            default=None,
        )
        for unit in TOLERANCES
    }
    s["status"] = (
        "incomplete"
        if s["errors"]
        else "differences"
        if any(c["status"] in ["difference", "calculation_missing"] for c in comparisons)
        else "ok"
    )
    out = Path(output_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "summary.json", s)
    write_json(out / "source_files.json", dict(files_sha256=before))
    write_json(out / "per_step_diagnostics.json", dict(rows=diagnostics))
    if comparisons:
        with (out / "comparisons.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(comparisons[0]))
            w.writeheader()
            w.writerows(comparisons)
    cases_dir = out / "cases"
    cases_dir.mkdir()
    for cid, views in plots.items():
        for name, svg in views.items():
            svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
            (cases_dir / f"{cid}_{name}.svg").write_text(
                svg.replace("</svg>", f"<style>{CSS}</style></svg>"), encoding="utf-8"
            )
    (out / "report.html").write_text(render_report(s, plots), encoding="utf-8")
    return out, s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", default=str(local_input("fno_samples", "samples_main_s1")))
    p.add_argument("--truth", default=str(local_input("fno_gt_sdf", "sdf")))
    p.add_argument("--output-root", default=str(ROOT / "fno_sdf_compare_results"))
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    out, s = run(args.samples, args.truth, args.output_root)
    print("비교:", s["frames_checked"], "frames;", s["comparison_counts"], "status", s["status"])
    print("보고서:", out / "report.html")
    if s["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
