"""FNO evaluation-score adapter.

Minseo's provided scores -> harness inputs, provisional comparison, official decision.

Reads metrics5_val.csv (all comparison groups) and the samples_main_s1 folder. Read-only on sources.
Does not compute new metrics from predictions, train, replace models or call external services.
Standard library only.

  python fno_adapter.py
  python fno_adapter.py --metrics <metrics5_val.csv> --samples <samples_main_s1>
                        [--thresholds thresholds_fno_provisional.json]

Separation kept on purpose:
  - Only the selected group (default '300런(main_s1)') is used;
    other groups are counted, never mixed in.
  - one-step and rollout are separate payloads with separate evaluation_contract_id values,
    because their per-recipe aggregation differs (step mean vs. last step).
  - The provisional comparison table is NOT a decision.
    The official decision is controller.decide via harness.judge.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from harness import judge, load_thresholds

ROOT = Path(__file__).resolve().parent
LOCAL_PATHS = ROOT / "local_paths.json"  # machine-specific input locations; not committed


def local_input(key, fallback_name):
    """Input path from local_paths.json if set, else inputs/<fallback_name> next to this file."""
    try:
        value = json.loads(LOCAL_PATHS.read_text(encoding="utf-8")).get(key)
    except (OSError, ValueError, AttributeError):
        value = None
    return (
        Path(value) if isinstance(value, str) and value.strip() else ROOT / "inputs" / fallback_name
    )


DEFAULT_METRICS = local_input("fno_metrics", "metrics5_val.csv")
DEFAULT_SAMPLES = local_input("fno_samples", "samples_main_s1")
DEFAULT_THRESHOLDS = ROOT / "thresholds_fno_provisional.json"
GROUP = "300런(main_s1)"
MODEL_VERSION = "main_s1"
MODE_MAP = {"1스텝": "onestep", "롤아웃": "rollout"}
SAMPLE_MODE_MAP = {"one-step": "onestep", "rollout": "rollout"}
DATA_ORIGIN = "fno_validation_scores_provided_by_minseo"

# metrics5_val.csv column -> (internal name, unit, value rule)
VAL_KEYS = ["방식", "종류", "런", "스텝"]
VAL_METRICS = {
    "Chamfer(um)": ("chamfer_um", "um", "nonneg"),
    "정답깊이(um)": ("true_depth_um", "um", "positive"),
    "깊이오차(%)": ("depth_error_pct_signed", "%", "signed"),
    "CD_위(um)": ("cd_top_um", "um", "nonneg"),
    "CD_위 오차(%)": ("cd_top_error_pct_signed", "%", "signed"),
    "CD_중간(um)": ("cd_mid_um", "um", "nonneg"),
    "CD_중간 오차(%)": ("cd_mid_error_pct_signed", "%", "signed"),
    "CD_바닥(um)": ("cd_bottom_um", "um", "nonneg"),
    "CD_바닥 오차(%)": ("cd_bottom_error_pct_signed", "%", "signed"),
    "IoU": ("iou", "ratio", "unit_interval"),
}
VAL_COLUMNS = VAL_KEYS + list(VAL_METRICS)
# metrics_by_step.csv / summary_last_step.csv column -> metrics5 column
STEP_TO_VAL = {
    "Chamfer(um)": "Chamfer(um)",
    "정답깊이(um)": "정답깊이(um)",
    "깊이오차(%)": "깊이오차(%)",
    "CD_위오차(%)": "CD_위 오차(%)",
    "CD_중간오차(%)": "CD_중간 오차(%)",
    "CD_바닥오차(%)": "CD_바닥 오차(%)",
    "IoU": "IoU",
}
STEP_COLUMNS = ["레시피", "방식", "스텝", "예측깊이(um)", *STEP_TO_VAL]
LAST_COLUMNS = ["레시피", "스텝수", "예측깊이(um)", *STEP_TO_VAL]

OVER_RECIPES = "equal-weight mean over recipes (computed here from the provided per-recipe values)"
AGGREGATION = {
    "onestep": dict(
        contract_id="fno-minseo-metrics5-onestep-stepmean-v1",
        per_recipe=(
            "mean over prediction steps t001..tN of the per-step score (provided by the evaluator)"
        ),
        over_recipes=OVER_RECIPES,
    ),
    "rollout": dict(
        contract_id="fno-minseo-metrics5-rollout-laststep-v1",
        per_recipe="score at the last rollout step tN only (provided by the evaluator)",
        over_recipes=OVER_RECIPES,
    ),
}
CHAMFER_DEFINITION = (
    "Both contours sampled every 0.1 um; mean over all points of the shortest distance to the "
    "other contour (from samples_main_s1/README.json). Differs from the local provisional "
    "definition (sum of the two one-way means), so a separate contract id is used."
)
UNMEASURED = {
    "invalid_geometry_count": (
        "No geometry-validity/self-intersection check result was provided. Left missing, not 0."
    ),
    "depth_mae_um": "No um depth MAE was provided. Not derived from averaged % x mean depth.",
    "all_step_abs_depth_error_pct": (
        "Per-step scores exist only for the sample recipes in samples_main_s1; "
        "not available for all recipes."
    ),
    "reference_verified": "Ground-truth data was not independently verified by this adapter.",
    "prediction_arrays": ".npz/.png files are hashed only; not parsed (standard library only).",
    "cd_height_basis": (
        'CD 10/50/90% height basis is marked "평가 담당 확인 필요" in the sample README.'
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def to_float(text):
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def rule_ok(v, rule):
    return {"nonneg": v >= 0, "positive": v > 0, "signed": True, "unit_interval": 0 <= v <= 1}[rule]


def header_unit(column):
    m = re.search(r"\(([^)]*)\)\s*$", column)
    return m.group(1) if m else None


class Problems(list):
    def add(self, kind, file, detail, rid=None):
        self.append(dict(kind=kind, file=file, rid=rid, detail=detail))


def source_files(metrics, samples):
    files = [Path(metrics)] + (
        sorted(p for p in Path(samples).iterdir() if p.is_file()) if Path(samples).is_dir() else []
    )
    out = {}
    for p in files:
        parsed = p.suffix.lower() in (".csv", ".json")
        out[str(p)] = dict(
            size_bytes=p.stat().st_size,
            sha256=sha256(p),
            parsed=parsed,
            note=None if parsed else "hashed only, not parsed",
        )
    return out


def load_val(path, group, problems):
    name = Path(path).name
    header, rows = read_csv(path)
    missing = [c for c in VAL_COLUMNS if c not in header]
    if missing:
        problems.add("missing_columns", name, f"missing required columns: {missing}")
        return {}, {}, Counter(r.get("방식") for r in rows)
    for col, (_, unit, _) in VAL_METRICS.items():
        if col != "IoU" and header_unit(col) != unit:
            problems.add("unit_mismatch", name, f"{col}: expected unit {unit}")
    groups = Counter(r["방식"] for r in rows)
    keys = Counter((r["방식"], r["종류"], r["런"]) for r in rows)
    for (g, kind, rid), n in keys.items():
        if n > 1:
            problems.add("duplicate_row", name, f"({g}, {kind}, {rid}) appears {n} times", rid)
    by_mode, raw = defaultdict(dict), defaultdict(list)
    for line, r in enumerate(rows, start=2):
        if r["방식"] != group:
            continue
        mode = MODE_MAP.get(r["종류"])
        rid = (r["런"] or "").strip()
        if mode is None:
            problems.add("unknown_mode", name, f"line {line}: 종류={r['종류']!r}", rid or None)
            continue
        if not rid:
            problems.add("missing_id", name, f"line {line}: empty 런")
            continue
        raw[mode].append(dict(line=line, **r))
        steps = to_float(r["스텝"])
        if steps is None or steps < 1 or int(steps) != steps:
            problems.add("bad_number", name, f"line {line}: 스텝={r['스텝']!r}", rid)
            steps = None
        values = {}
        for col, (key, _, rule) in VAL_METRICS.items():
            v = to_float(r[col])
            if v is None:
                problems.add("bad_number", name, f"line {line}: {col}={r[col]!r}", rid)
            elif not rule_ok(v, rule):
                problems.add("out_of_range", name, f"line {line}: {col}={v} violates {rule}", rid)
                v = None
            values[key] = v
        by_mode[mode].setdefault(
            rid, dict(rid=rid, steps=int(steps) if steps else None, line=line, **values)
        )
    for mode in MODE_MAP.values():
        if not by_mode[mode]:
            problems.add("missing_mode", name, f"no {mode} rows for group {group!r}")
    ids = {m: set(v) for m, v in by_mode.items()}
    if len(ids) == 2:
        for rid in sorted(ids["onestep"] ^ ids["rollout"]):
            problems.add(
                "mode_id_mismatch", name, "recipe present in only one of onestep/rollout", rid
            )
        for rid in sorted(ids["onestep"] & ids["rollout"]):
            a, b = by_mode["onestep"][rid]["steps"], by_mode["rollout"][rid]["steps"]
            if a is not None and b is not None and a != b:
                problems.add(
                    "step_count_mismatch", name, f"onestep 스텝={a}, rollout 스텝={b}", rid
                )
    return dict(by_mode), dict(raw), groups


def check_samples(samples, by_mode, problems):
    """Verify the stated aggregation on sample recipes; collect step-level reference values."""
    samples = Path(samples)
    evidence, steps_out = [], {}
    step_path, last_path = samples / "metrics_by_step.csv", samples / "summary_last_step.csv"
    if not step_path.is_file():
        problems.add("missing_file", step_path.name, "not found")
        return evidence, steps_out
    header, rows = read_csv(step_path)
    missing = [c for c in STEP_COLUMNS if c not in header]
    if missing:
        problems.add("missing_columns", step_path.name, f"missing required columns: {missing}")
        return evidence, steps_out
    blanks = Counter()
    grouped = defaultdict(list)
    for line, r in enumerate(rows, start=2):
        mode = SAMPLE_MODE_MAP.get(r["방식"])
        m = re.fullmatch(r"t(\d+)", r["스텝"] or "")
        if mode is None or not m:
            problems.add(
                "bad_step_row",
                step_path.name,
                f"line {line}: 방식={r['방식']!r}, 스텝={r['스텝']!r}",
                r["레시피"],
            )
            continue
        vals = {}
        for col in ["예측깊이(um)", *STEP_TO_VAL]:
            if not (r[col] or "").strip():
                blanks[col] += 1  # recorded as missing; never filled with 0
                vals[col] = None
                continue
            v = to_float(r[col])
            if v is None:
                problems.add(
                    "bad_number", step_path.name, f"line {line}: {col}={r[col]!r}", r["레시피"]
                )
            vals[col] = v
        grouped[(r["레시피"], mode)].append((int(m.group(1)), vals))
    for (rid, mode), items in grouped.items():
        nums = [s for s, _ in items]
        if sorted(nums) != list(range(1, len(nums) + 1)) or len(set(nums)) != len(nums):
            problems.add(
                "step_gap_or_duplicate", step_path.name, f"{mode} steps {sorted(nums)}", rid
            )
        items.sort()
        steps_out[(rid, mode)] = [v for _, v in items]
        target = by_mode.get(mode, {}).get(rid)
        if target is None:
            problems.add(
                "sample_not_in_metrics",
                step_path.name,
                f"{mode} sample not found in selected group",
                rid,
            )
            continue
        if target["steps"] is not None and target["steps"] != len(items):
            problems.add(
                "step_count_mismatch",
                step_path.name,
                f"{len(items)} steps vs metrics5 스텝={target['steps']}",
                rid,
            )
        for scol, vcol in STEP_TO_VAL.items():
            series = [v[scol] for _, v in items]
            present = [x for x in series if x is not None]
            want = target[VAL_METRICS[vcol][0]]
            if want is None or not present:
                continue
            got = statistics.fmean(present) if mode == "onestep" else series[-1]
            ok = got is not None and math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12)
            evidence.append(
                dict(
                    rid=rid,
                    mode=mode,
                    column=vcol,
                    rule="step_mean (non-blank steps)" if mode == "onestep" else "last_step",
                    provided=want,
                    recomputed_from_steps=got,
                    blank_steps=len(series) - len(present),
                    match=ok,
                )
            )
            if not ok:
                problems.add(
                    "aggregation_mismatch",
                    step_path.name,
                    f"{mode} {vcol}: provided {want} vs "
                    f"{'mean' if mode == 'onestep' else 'last'} {got}",
                    rid,
                )
    if last_path.is_file():
        header, rows = read_csv(last_path)
        miss = [c for c in LAST_COLUMNS if c not in header]
        if miss:
            problems.add("missing_columns", last_path.name, f"missing required columns: {miss}")
        else:
            for r in rows:
                target = by_mode.get("rollout", {}).get(r["레시피"])
                if target is None:
                    problems.add(
                        "sample_not_in_metrics",
                        last_path.name,
                        "not in selected rollout rows",
                        r["레시피"],
                    )
                    continue
                for scol, vcol in STEP_TO_VAL.items():
                    v, want = to_float(r[scol]), target[VAL_METRICS[vcol][0]]
                    if (
                        v is not None
                        and want is not None
                        and not math.isclose(v, want, rel_tol=1e-9, abs_tol=1e-12)
                    ):
                        problems.add(
                            "summary_mismatch",
                            last_path.name,
                            f"{vcol}: {v} vs metrics5 {want}",
                            r["레시피"],
                        )
    return evidence, dict(blank_cells_by_column=dict(blanks), series=steps_out)


def stats(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return dict(n=0, mean=None, median=None, min=None, max=None)
    return dict(
        n=len(vals),
        mean=statistics.fmean(vals),
        median=statistics.median(vals),
        min=min(vals),
        max=max(vals),
    )


def summarize(by_mode, step_info):
    out = {}
    for mode, recipes in by_mode.items():
        rows = list(recipes.values())

        def worst(key, f=lambda x: x, rows=rows):
            return max(
                (r for r in rows if r[key] is not None), key=lambda r: f(r[key]), default=None
            )

        wc, wd = worst("chamfer_um"), worst("depth_error_pct_signed", abs)
        summary = dict(
            recipe_count=len(rows),
            aggregation=AGGREGATION[mode],
            steps_per_recipe=dict(
                sorted(
                    Counter(r["steps"] for r in rows).items(),
                    key=lambda kv: (kv[0] is None, kv[0] or 0),
                )
            ),
            chamfer_um=dict(stats(r["chamfer_um"] for r in rows), worst_rid=wc and wc["rid"]),
            depth_error_pct_signed=dict(
                stats(r["depth_error_pct_signed"] for r in rows),
                note="signed recipe values as provided; + means over-etch",
            ),
            abs_depth_error_pct_recipe_values=dict(
                stats(
                    abs(r["depth_error_pct_signed"])
                    for r in rows
                    if r["depth_error_pct_signed"] is not None
                ),
                worst_rid=wd and wd["rid"],
                note=(
                    "mean of |provided per-recipe depth error %|. "
                    "For onestep each recipe value is already a "
                    "signed step mean, so this is NOT the all-step MAE."
                ),
            ),
            iou=stats(r["iou"] for r in rows),
            abs_cd_error_pct={
                k: stats(abs(r[k]) for r in rows if r[k] is not None)
                for k in [
                    "cd_top_error_pct_signed",
                    "cd_mid_error_pct_signed",
                    "cd_bottom_error_pct_signed",
                ]
            },
            all_step_depth_mae="unmeasured for the full recipe set (no per-step scores provided)",
        )
        # Step-level reference on the sample recipes only, shown next to the recipe-level value.
        sample = {}
        for (rid, m), series in (step_info.get("series") or {}).items():
            if m != mode or rid not in recipes:
                continue
            pct = [abs(s["깊이오차(%)"]) for s in series if s["깊이오차(%)"] is not None]
            um = [
                abs(s["예측깊이(um)"] - s["정답깊이(um)"])
                for s in series
                if s["예측깊이(um)"] is not None and s["정답깊이(um)"] is not None
            ]
            sample[rid] = dict(
                steps=len(series),
                all_step_abs_depth_error_pct_mean=statistics.fmean(pct) if pct else None,
                all_step_depth_mae_um_from_step_depths=statistics.fmean(um) if um else None,
                abs_of_provided_recipe_value_pct=abs(recipes[rid]["depth_error_pct_signed"]),
            )
        summary["sample_step_level_reference"] = dict(
            recipes=sample,
            note=(
                "Sample recipes only. Step MAE uses per-step 예측깊이/정답깊이 "
                "provided in metrics_by_step.csv; "
                "reference only, not used for any comparison or decision."
            ),
        )
        out[mode] = summary
    return out


def payload(mode, summary, data_check, dataset_id):
    """Harness evaluation input. Only measured metrics are included; missing ones stay absent."""
    return dict(
        _notice=(
            "Real FNO validation scores provided by Minseo (group 300런(main_s1)). "
            "Not synthetic data."
        ),
        model_family="FNO",
        model_version=MODEL_VERSION,
        eval_mode=mode,
        evaluation_dataset_id=dataset_id,
        split="validation",
        evaluation_contract_id=AGGREGATION[mode]["contract_id"],
        unit="um",
        metric_units=dict(
            chamfer_um_recipe_mean="um",
            abs_depth_error_pct_recipe_mean="%",
            invalid_geometry_count="count",
        ),
        metric_aggregation=dict(
            per_recipe=AGGREGATION[mode]["per_recipe"],
            over_recipes=AGGREGATION[mode]["over_recipes"],
            chamfer_definition=CHAMFER_DEFINITION,
            depth_error=(
                "|signed per-recipe %| averaged over recipes; signed mean kept "
                "in signed_depth_error_pct_recipe_mean"
            ),
        ),
        recipe_count=summary["recipe_count"],
        data_origin=DATA_ORIGIN,
        training_data_ready=False,
        data_check=data_check,
        metrics=dict(
            chamfer_um_recipe_mean=summary["chamfer_um"]["mean"],
            abs_depth_error_pct_recipe_mean=summary["abs_depth_error_pct_recipe_values"]["mean"],
        ),
        reported_not_judged=dict(
            signed_depth_error_pct_recipe_mean=summary["depth_error_pct_signed"]["mean"],
            iou_recipe_mean=summary["iou"]["mean"],
        ),
        unmeasured=sorted(UNMEASURED),
    )


def provisional_comparison(evaluation, policy):
    """Compare scores to the provisional limits. Not a decision; missing scores stay '미측정'."""
    rows = []
    for metric, limit in (policy.get("max_metrics") or {}).items():
        observed = (evaluation.get("metrics") or {}).get(metric)
        if observed is None:
            verdict = "미측정"
        elif not isinstance(limit, (int, float)) or isinstance(limit, bool):
            verdict = "기준 없음"
        else:
            verdict = "이내" if observed <= limit else "초과"
        rows.append(
            dict(
                metric=metric,
                unit=(policy.get("metric_units") or {}).get(metric),
                observed=observed,
                provisional_max=limit,
                result=verdict,
            )
        )
    return dict(
        label="임시 비교 (공식 판정 아님)",
        policy_version=policy.get("version"),
        policy_status=policy.get("status"),
        rows=rows,
    )


def remaining_blockers(evaluation, policy):
    """Reasons an approved policy would still not judge this input.

    Read-only listing; no decision.
    """
    out = []
    if (evaluation.get("data_check") or {}).get("reference_verified") is not True:
        out.append("reference_verified is false: ground truth not independently verified")
    for metric in policy.get("max_metrics") or {}:
        if (evaluation.get("metrics") or {}).get(metric) is None:
            out.append(f"{metric} is unmeasured (missing, not 0)")
    return out


def per_recipe_exceed(by_mode, policy, mode):
    lim = policy.get("max_metrics") or {}
    out = {}
    for metric, key, f in [
        ("chamfer_um_recipe_mean", "chamfer_um", lambda x: x),
        ("abs_depth_error_pct_recipe_mean", "depth_error_pct_signed", abs),
    ]:
        if isinstance(lim.get(metric), (int, float)):
            over = sorted(
                r["rid"]
                for r in by_mode[mode].values()
                if r[key] is not None and f(r[key]) > lim[metric]
            )
            out[metric] = dict(recipes_over_provisional_max=len(over), rids=over)
    return out


def fmt(v, digits=4):
    if v is None:
        return "미측정"
    return f"{v:.{digits}g}" if isinstance(v, float) else str(v)


def report(sources, group, groups, inspection, summary, comparisons, decisions):
    L = ["# FNO 평가 점수 변환 결과 (민서 제공 점수, 새 지표 계산 없음)", ""]
    L.append(
        f"선택 그룹: {group} / 제외 그룹: "
        + ", ".join(f"{g}({n}행)" for g, n in groups.items() if g != group)
    )
    L.append(f"원본 파일 {len(sources)}개 (SHA256은 source_files.json)")
    dc = inspection["data_check"]
    L.append(
        f"데이터 검사: passed={dc['passed']}, invalid_recipe_count={dc['invalid_recipe_count']}, "
        f"reference_verified={dc['reference_verified']}, 문제 {len(inspection['problems'])}건 "
        f"{inspection['problem_counts'] or ''}"
    )
    ev = inspection["aggregation_evidence"]
    L.append(
        f"집계 확인(샘플): {sum(e['match'] for e in ev)}/{len(ev)} 일치 - "
        "onestep=단계 평균, rollout=마지막 단계"
    )
    for mode in ("onestep", "rollout"):
        s = summary.get(mode)
        if not s:
            continue
        ch, ad = s["chamfer_um"], s["abs_depth_error_pct_recipe_values"]
        L += [
            "",
            f"## {mode} ({s['recipe_count']}개 레시피, 계약 {s['aggregation']['contract_id']})",
            f"  레시피 값: {s['aggregation']['per_recipe']}",
            f"  Chamfer(um) 레시피 평균 {fmt(ch['mean'])}, "
            f"최대 {fmt(ch['max'])} ({ch['worst_rid']})",
            f"  깊이오차(%) 부호 있는 평균 "
            f"{fmt(s['depth_error_pct_signed']['mean'])} (원래 값 보존)",
            f"  |깊이오차(%)| 레시피 제공값의 절댓값 평균 {fmt(ad['mean'])}, "
            f"최대 {fmt(ad['max'])} ({ad['worst_rid']})",
            f"  전체 단계 깊이 MAE: {s['recipe_count']}개 레시피 전체는 미측정. "
            f"샘플 {len(s['sample_step_level_reference']['recipes'])}개 참고값:",
        ]
        for rid, x in s["sample_step_level_reference"]["recipes"].items():
            L.append(
                f"    {rid}: 레시피 제공값 |%| {fmt(x['abs_of_provided_recipe_value_pct'])} "
                f"vs 전체 단계 |%| 평균 {fmt(x['all_step_abs_depth_error_pct_mean'])} "
                f"/ 단계 깊이 MAE {fmt(x['all_step_depth_mae_um_from_step_depths'])} um"
            )
        L.append(f"  IoU 평균 {fmt(s['iou']['mean'])}, 최소 {fmt(s['iou']['min'])}")
    L += ["", "## 미측정 항목"] + [f"  - {k}: {v}" for k, v in UNMEASURED.items()]
    L += ["", "## [임시 비교] 실제 점수 vs FNO 임시 기준 - 공식 판정 아님"]
    for mode, c in comparisons.items():
        L.append(f"  {mode} (기준 {c['policy_version']}, status={c['policy_status']})")
        # Korean labels are double-width in a terminal.
        L.append(f"    {'지표':<32}{'관측':>8}{'임시 상한':>8}  결과")
        for r in c["rows"]:
            limit = "≤ " + fmt(r["provisional_max"])
            L.append(
                f"    {r['metric']:<34}{fmt(r['observed']):>10}{limit:>12}  "
                f"{r['result']} ({r['unit']})"
            )
        for m, e in c.get("per_recipe", {}).items():
            L.append(f"    참고: 레시피별 {m} 임시 상한 초과 {e['recipes_over_provisional_max']}건")
    L += ["", "## [공식 판정] controller.decide"]
    for mode, d in decisions.items():
        L.append(f"  {mode}: {d['action']} - " + " / ".join(d["reasons"]))
        for b in d.get("remaining_blockers_even_if_approved", []):
            L.append(f"    기준을 승인해도 남는 보류 사유: {b}")
        L.append(f"    학습 시작 {d['training_started']} / 모델 교체 {d['active_model_changed']}")
    return "\n".join(L) + "\n"


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run(metrics, samples, thresholds, output_root, group=GROUP):
    sources = source_files(metrics, samples)
    problems = Problems()
    by_mode, raw, groups = load_val(metrics, group, problems)
    evidence, step_info = check_samples(samples, by_mode, problems) if by_mode else ([], {})
    bad_rids = sorted({p["rid"] for p in problems if p["rid"]})
    data_check = dict(
        passed=not problems, invalid_recipe_count=len(bad_rids), reference_verified=False
    )
    inspection = dict(
        problems=list(problems),
        problem_counts=dict(Counter(p["kind"] for p in problems)),
        invalid_recipe_ids=bad_rids,
        data_check=data_check,
        aggregation_evidence=evidence,
        step_file_blank_cells=step_info.get("blank_cells_by_column") if step_info else None,
    )
    summary = summarize(by_mode, step_info or {})
    dataset_id = f"fno_metrics5_val_{sha256(metrics)[:12]}"
    policies = load_thresholds(thresholds)
    inputs, comparisons, decisions = {}, {}, {}
    for mode in ("onestep", "rollout"):
        if mode not in summary:
            continue
        inputs[mode] = payload(mode, summary[mode], data_check, dataset_id)
        policy, decision = judge(inputs[mode], policies)
        if decision["training_started"] or decision["active_model_changed"]:
            raise AssertionError("The adapter must only decide and record.")
        decisions[mode] = dict(
            decision,
            remaining_blockers_even_if_approved=(
                remaining_blockers(inputs[mode], policy) if policy is not None else []
            ),
        )
        if policy is not None:
            comparisons[mode] = dict(
                provisional_comparison(inputs[mode], policy),
                per_recipe=per_recipe_exceed(by_mode, policy, mode),
            )
    if not inputs:  # nothing usable: still record a decision
        from controller import outcome

        decisions["none"] = outcome(
            "DATA_REVIEW", ["No usable rows for the selected group."], {}, {}
        )
    after = {p: sha256(p) for p in sources}
    if after != {p: v["sha256"] for p, v in sources.items()}:
        raise AssertionError("A source file changed during conversion.")

    out = Path(output_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "source_files.json",
        dict(files=sources, selected_group=group, all_groups=dict(groups)),
    )
    write_json(
        out / "selected_rows.json",
        dict(group=group, note="original CSV strings, unchanged", rows=raw),
    )
    write_json(out / "inspection.json", inspection)
    write_json(out / "score_summary.json", summary)
    write_json(out / "unmeasured.json", UNMEASURED)
    for mode, p in inputs.items():
        write_json(out / f"harness_input_{mode}.json", p)
    write_json(
        out / "provisional_comparison.json",
        dict(notice="임시 비교용. 공식 판정 아님.", by_mode=comparisons),
    )
    write_json(out / "official_decisions.json", dict(thresholds=str(thresholds), by_mode=decisions))
    text = report(sources, group, groups, inspection, summary, comparisons, decisions)
    (out / "report.txt").write_text(text, encoding="utf-8")
    return (
        out,
        text,
        dict(
            inspection=inspection,
            summary=summary,
            inputs=inputs,
            comparisons=comparisons,
            decisions=decisions,
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument("--group", default=GROUP)
    parser.add_argument("--output-root", default=str(ROOT / "fno_adapter_results"))
    args = parser.parse_args()
    out, text, _ = run(args.metrics, args.samples, args.thresholds, args.output_root, args.group)
    print(text)
    print("결과 폴더:", out)


if __name__ == "__main__":
    main()
