"""MGN evaluation-score adapter.

Jinju's mgn_main_s1_nov9 folder -> harness inputs, provisional comparison, decision.
Reads README.json, metrics_by_step.csv and summary_last_step.csv. Read-only on sources;
.npz/.png are normally hashed only. --check-geometry additionally inspects NPZ segment
graphs without changing the provided Chamfer scores. No training or model replacement.
Standard library only.

  python mgn_adapter.py
  python mgn_adapter.py --input <mgn folder> [--thresholds thresholds_mgn_provisional.json]

Separation kept on purpose:
  - validation (val) and test are separate. Only validation is compared with the provisional
    limits and passed to the decision; test scores are recorded for reference and never used
    to set or tune thresholds.
  - one-step = mean over steps of the provided per-step one-step Chamfer;
    rollout = provided Chamfer at the last step.
    summary_last_step.csv's onestep_chamfer is the LAST-STEP one-step value and is only used
    as a consistency check.
  - Depth error, IoU, CD and geometry-anomaly counts are not provided; they stay missing (never 0).
Reuses the generic helpers of fno_adapter.py without changing it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import mgn_geometry
from fno_adapter import (
    Problems,
    fmt,
    local_input,
    provisional_comparison,
    read_csv,
    remaining_blockers,
    sha256,
    stats,
    to_float,
    write_json,
)
from harness import judge, load_thresholds

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = local_input("mgn_input", "mgn_main_s1_nov9")
DEFAULT_THRESHOLDS = ROOT / "thresholds_mgn_provisional.json"
SPLIT_MAP = {"val": "validation", "test": "test"}
DATA_ORIGIN = "mgn_scores_provided_by_jinju"
STEP_COLUMNS = ["rid", "split", "step", "onestep_chamfer", "rollout_chamfer"]
SUMMARY_COLUMNS = [
    "rid",
    "split",
    "last_step",
    "onestep_chamfer",
    "rollout_chamfer",
    "n_pred_points",
    "n_gt_points",
]
README_KEYS = ["model", "model_sha256_16", "splits", "unit", "chamfer_definition", "limitations"]
AGGREGATION = {
    "onestep": dict(
        contract_id="mgn-jinju-nov9-onestep-stepmean-pointchamfer-v1",
        per_recipe=(
            "mean over steps 1..N of the provided per-step one-step Chamfer "
            "(input: ground-truth snapshot k-1, compared with ground truth k)"
        ),
        over_recipes="equal-weight mean over recipes of one split",
    ),
    "rollout": dict(
        contract_id="mgn-jinju-nov9-rollout-laststep-pointchamfer-v1",
        per_recipe="provided rollout Chamfer at the last step N (rollout from the initial shape)",
        over_recipes="equal-weight mean over recipes of one split",
    ),
}
NOT_PROVIDED = {
    "depth_error": "MGN folder provides no depth error; not derived from point clouds here.",
    "iou": "No IoU provided.",
    "cd_error": "No CD error provided.",
    "invalid_geometry_count": (
        "No geometry-anomaly count provided. README says crossings remain; count unknown, not 0."
    ),
    "self_intersection": (
        'README: "교차 있는 레시피 남아 있음, Chamfer만으로 합격 판단 금지". Not counted here.'
    ),
}


def load_readme(folder, problems):
    path = Path(folder) / "README.json"
    if not path.is_file():
        problems.add("missing_file", "README.json", "not found")
        return {}
    try:
        readme = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        problems.add("bad_json", "README.json", str(exc))
        return {}
    missing = [k for k in README_KEYS if k not in readme]
    if missing:
        problems.add("missing_keys", "README.json", f"missing keys: {missing}")
    return readme


def declared_ranges(readme):
    """{'val': (300, 339), 'test': (340, 379)} parsed from README splits text, if present."""
    out = {}
    for split, text in (readme.get("splits") or {}).items():
        m = re.search(r"r(\d{4})\s*-\s*r(\d{4})", str(text))
        if m:
            out[split] = (int(m.group(1)), int(m.group(2)))
    return out


def rid_number(rid):
    m = re.fullmatch(r"r(\d{4})", rid or "")
    return int(m.group(1)) if m else None


def load_steps(folder, problems):
    name = "metrics_by_step.csv"
    path = Path(folder) / name
    if not path.is_file():
        problems.add("missing_file", name, "not found")
        return {}
    header, rows = read_csv(path)
    missing = [c for c in STEP_COLUMNS if c not in header]
    if missing:
        problems.add("missing_columns", name, f"missing required columns: {missing}")
        return {}
    for (rid, step), n in Counter((r["rid"], r["step"]) for r in rows).items():
        if n > 1:
            problems.add("duplicate_row", name, f"(rid={rid}, step={step}) appears {n} times", rid)
    series = defaultdict(list)
    for line, r in enumerate(rows, start=2):
        rid = (r["rid"] or "").strip()
        if not rid:
            problems.add("missing_id", name, f"line {line}: empty rid")
            continue
        if r["split"] not in SPLIT_MAP:
            problems.add("bad_split", name, f"line {line}: split={r['split']!r}", rid)
        step = to_float(r["step"])
        if step is None or step < 1 or int(step) != step:
            problems.add("bad_number", name, f"line {line}: step={r['step']!r}", rid)
            continue
        vals = {}
        for col in ("onestep_chamfer", "rollout_chamfer"):
            v = to_float(r[col])
            if v is None:
                problems.add("bad_number", name, f"line {line}: {col}={r[col]!r}", rid)
            elif v < 0:
                problems.add("out_of_range", name, f"line {line}: {col}={v} < 0", rid)
                v = None
            vals[col] = v
        series[rid].append(dict(line=line, split=r["split"], step=int(step), **vals))
    for rid, items in series.items():
        items.sort(key=lambda x: x["step"])
        nums = [x["step"] for x in items]
        if nums != list(range(1, len(nums) + 1)):
            problems.add("step_gap_or_duplicate", name, f"steps {nums}", rid)
        if len({x["split"] for x in items}) > 1:
            problems.add("split_inconsistent", name, "rid has rows in more than one split", rid)
    return dict(series)


def load_summary(folder, problems):
    name = "summary_last_step.csv"
    path = Path(folder) / name
    if not path.is_file():
        problems.add("missing_file", name, "not found")
        return {}, []
    header, rows = read_csv(path)
    missing = [c for c in SUMMARY_COLUMNS if c not in header]
    if missing:
        problems.add("missing_columns", name, f"missing required columns: {missing}")
        return {}, rows
    for rid, n in Counter(r["rid"] for r in rows).items():
        if n > 1:
            problems.add("duplicate_row", name, f"rid={rid} appears {n} times", rid)
    out = {}
    for line, r in enumerate(rows, start=2):
        rid = (r["rid"] or "").strip()
        if not rid:
            problems.add("missing_id", name, f"line {line}: empty rid")
            continue
        if r["split"] not in SPLIT_MAP:
            problems.add("bad_split", name, f"line {line}: split={r['split']!r}", rid)
        vals = {}
        for col in SUMMARY_COLUMNS[2:]:
            v = to_float(r[col])
            integer = col in ("last_step", "n_pred_points", "n_gt_points")
            if v is None or v < 0 or (integer and (int(v) != v or v < 1)):
                problems.add("bad_number", name, f"line {line}: {col}={r[col]!r}", rid)
                v = None
            vals[col] = v
        out.setdefault(rid, dict(line=line, split=r["split"], **vals))
    return out, rows


def cross_check(series, summary, ranges, problems):
    """IDs, splits and README ranges agree; summary last-step values equal the per-step file."""
    evidence = Counter()
    for rid in sorted(set(series) ^ set(summary)):
        problems.add(
            "id_mismatch",
            "metrics_by_step.csv/summary_last_step.csv",
            "rid present in only one file",
            rid,
        )
    for rid in sorted(set(series) & set(summary)):
        items, s = series[rid], summary[rid]
        if items[0]["split"] != s["split"]:
            problems.add(
                "split_mismatch",
                "summary_last_step.csv",
                f"{s['split']} vs step file {items[0]['split']}",
                rid,
            )
        num, rng = rid_number(rid), ranges.get(s["split"])
        if rng and (num is None or not rng[0] <= num <= rng[1]):
            problems.add(
                "split_range_mismatch",
                "summary_last_step.csv",
                f"{rid} labelled {s['split']} but README range is r{rng[0]:04d}-r{rng[1]:04d}",
                rid,
            )
        if s["last_step"] is not None and s["last_step"] != len(items):
            problems.add(
                "step_count_mismatch",
                "summary_last_step.csv",
                f"last_step={s['last_step']}, steps={len(items)}",
                rid,
            )
        last = items[-1]
        for col in ("onestep_chamfer", "rollout_chamfer"):
            want, got = s[col], last[col]
            if want is None or got is None:
                continue
            ok = math.isclose(want, got, rel_tol=1e-9, abs_tol=1e-12)
            evidence[f"summary.{col} == last step ({'match' if ok else 'MISMATCH'})"] += 1
            if not ok:
                problems.add(
                    "summary_mismatch",
                    "summary_last_step.csv",
                    f"{col}={want} vs last step {got}",
                    rid,
                )
    return dict(evidence)


def per_recipe(series, summary):
    """Recipe values per (split, mode). One-step = step mean; rollout = last step."""
    out = defaultdict(dict)
    for rid, items in series.items():
        split = SPLIT_MAP.get(items[0]["split"])
        if split is None:
            continue
        one = [x["onestep_chamfer"] for x in items]
        out[(split, "onestep")][rid] = dict(
            rid=rid,
            steps=len(items),
            chamfer_um=statistics.fmean(one) if None not in one else None,
            reported_last_step_onestep_chamfer_um=summary.get(rid, {}).get("onestep_chamfer"),
        )
        out[(split, "rollout")][rid] = dict(
            rid=rid, steps=len(items), chamfer_um=items[-1]["rollout_chamfer"]
        )
    return dict(out)


def summarize(recipes):
    out = {}
    order = {"validation": 0, "test": 1, "onestep": 0, "rollout": 1}
    for (split, mode), rows in sorted(
        recipes.items(), key=lambda kv: (order[kv[0][0]], order[kv[0][1]])
    ):
        vals = [r for r in rows.values() if r["chamfer_um"] is not None]
        worst = max(vals, key=lambda r: r["chamfer_um"], default=None)
        entry = dict(
            split=split,
            mode=mode,
            recipe_count=len(rows),
            aggregation=AGGREGATION[mode],
            steps_per_recipe=dict(sorted(Counter(r["steps"] for r in rows.values()).items())),
            chamfer_um=dict(
                stats(r["chamfer_um"] for r in rows.values()), worst_rid=worst and worst["rid"]
            ),
            not_provided=sorted(NOT_PROVIDED),
        )
        if mode == "onestep":
            entry["reported_last_step_onestep_chamfer_um"] = dict(
                stats(r["reported_last_step_onestep_chamfer_um"] for r in rows.values()),
                note=(
                    "summary_last_step.csv value; last step only, "
                    "NOT the step mean used as the one-step score"
                ),
            )
        if split == "test":
            entry["use"] = (
                "reference record only; not compared with provisional limits "
                "and not used to tune them"
            )
        out[f"{split}/{mode}"] = entry
    return out


def payload(mode, summary_entry, data_check, readme, dataset_id):
    return dict(
        _notice=(
            "Real MGN validation scores provided by Jinju (mgn_main_s1_nov9). Not synthetic data. "
            "Validation split was used to choose rollout spacing, "
            "so it is not an independent evaluation."
        ),
        model_family="MGN",
        model_version=(
            f"{Path(str(readme.get('model', 'unknown'))).stem}"
            f"@{readme.get('model_sha256_16', 'unknown')}"
        ),
        eval_mode=mode,
        evaluation_dataset_id=dataset_id,
        split="validation",
        evaluation_contract_id=AGGREGATION[mode]["contract_id"],
        unit="um",
        metric_units=dict(chamfer_um_recipe_mean="um", invalid_geometry_count="count"),
        metric_aggregation=dict(
            per_recipe=AGGREGATION[mode]["per_recipe"],
            over_recipes=AGGREGATION[mode]["over_recipes"],
            chamfer_definition=readme.get("chamfer_definition"),
        ),
        recipe_count=summary_entry["recipe_count"],
        data_origin=DATA_ORIGIN,
        training_data_ready=False,
        data_check=data_check,
        metrics=dict(chamfer_um_recipe_mean=summary_entry["chamfer_um"]["mean"]),
        known_issues=list(readme.get("limitations") or []),
        not_provided=sorted(NOT_PROVIDED),
    )


def unverified(readme, geometry=None):
    """Open questions, kept apart from the decision."""
    items = [
        "Validation split r0300-r0339 was used to pick the rollout spacing "
        "(README: 독립 평가 아님).",
        "Self-intersections remain in some recipes (README); "
        "count and affected recipes are not provided.",
        "Chamfer is point-based (sum of two one-way mean nearest distances); "
        "may differ from 수민님 point-segment method.",
        "Ground truth (etch-raw-n400 snapshots) is not in this folder; "
        "reference not verified here.",
        "Unit is um per README, but README says the unit document itself was not seen directly.",
        "One seed, 40 validation recipes, no OOD recipes.",
        "Prediction point count/order differ from ground truth; "
        ".npz/.png were hashed only, not parsed.",
        "Provisional limits are arbitrary starting values, not team-approved.",
    ]
    if geometry is not None:
        items = [
            x
            for x in items
            if not x.startswith(
                ("Self-intersections remain", "Prediction point count/order differ")
            )
        ]
        items += [
            f"Prediction NPZ segment graphs were inspected under {mgn_geometry.CONTRACT}; "
            "PNG images remain unparsed. Numeric counts do not verify physical validity.",
            f"Geometry scan read errors: {len(geometry['errors'])}; "
            "an incomplete split/mode retains an unknown count.",
            "Duplicate node-pair connections are an output-connectivity issue, reported separately "
            "from geometric overlaps. Their physical-gate treatment needs Jinju's confirmation; "
            "the provisional problem-frame union currently still includes them.",
        ]
    return dict(
        items=items,
        readme_limitations=list(readme.get("limitations") or []),
        readme_unit=readme.get("unit"),
        readme_chamfer_definition=readme.get("chamfer_definition"),
    )


def recipes_over(rows, limit):
    if not isinstance(limit, (int, float)) or isinstance(limit, bool):
        return None
    over = sorted(
        r["rid"] for r in rows.values() if r["chamfer_um"] is not None and r["chamfer_um"] > limit
    )
    return dict(recipes_over_provisional_max=len(over), rids=over)


def report(
    readme, sources_n, inspection, summary, comparisons, decisions, open_items, geometry=None
):
    L = [
        "# MGN 평가 점수 변환 결과 (제공 Chamfer 유지, 선택형 선분 형상 검사)",
        "",
        f"모델: {readme.get('model')} (sha256_16 {readme.get('model_sha256_16')}), "
        f"작성 {readme.get('author')} {readme.get('date')}",
        f"원본 파일 {sources_n}개 (SHA256은 source_files.json)",
        "",
        "!! 제공 설명: 교차(자기교차) 있는 레시피가 남아 있음 — "
        "Chamfer만으로 합격 판단 금지 (README limitations)",
        (
            "   이번 실행의 선분 형상 검사 결과는 아래에 별도 기록."
            if geometry is not None
            else "   자기교차 건수·레시피는 제공되지 않아 미측정이며 0으로 채우지 않음."
        ),
        "",
    ]
    dc = inspection["data_check"]
    L.append(
        f"데이터 검사: passed={dc['passed']}, invalid_recipe_count={dc['invalid_recipe_count']}, "
        f"reference_verified={dc['reference_verified']}, "
        f"문제 {len(inspection['problems'])}건 {inspection['problem_counts'] or ''}"
    )
    for k, n in inspection["consistency_evidence"].items():
        L.append(f"  {k}: {n}건")
    if geometry is not None:
        L += ["", "## 선분 형상 검사 (예측 프레임 수, 교차 쌍 수와 구분)"]
        for key, g in geometry["groups"].items():
            L.append(
                f"  {key}: 검사 {g['frames_checked']}/{g['frames_expected']}, "
                f"이상 프레임 {fmt(g['invalid_geometry_count'])}, "
                f"교차 프레임 {g['self_intersection_frame_count']}, "
                f"중첩 프레임 {g['overlap_frame_count']}, "
                f"중복 연결 프레임 {g['duplicate_edge_frame_count']}, "
                f"이상 레시피 {g['affected_recipe_count']}개 {g['affected_recipe_ids']}"
            )
        L.append(
            "  세부: geometry_diagnostics.json / geometry_frames.csv. "
            "끝점 접촉은 별도 진단이며 단독으로 이상 건수에 포함하지 않음."
        )
        L.append(f"  읽기 오류 {len(geometry['errors'])}건; 미완료 집계 건수는 미측정.")
    L += ["", "## 점수 요약 (Chamfer, um)"]
    for key, s in summary.items():
        c = s["chamfer_um"]
        tag = (
            " [test: 참고 기록만, 임시 기준 비교·조정에 사용 안 함]" if s["split"] == "test" else ""
        )
        L.append(
            f"  {key}: {s['recipe_count']}개 레시피, 평균 {fmt(c['mean'])}, "
            f"중앙값 {fmt(c['median'])}, "
            f"최대 {fmt(c['max'])} ({c['worst_rid']}){tag}"
        )
        L.append(f"    레시피 값: {s['aggregation']['per_recipe']}")
        if "reported_last_step_onestep_chamfer_um" in s:
            L.append(
                f"    참고: summary의 onestep_chamfer(마지막 단계만) 평균 "
                f"{fmt(s['reported_last_step_onestep_chamfer_um']['mean'])} — "
                "단계 평균과 다른 값, 점수로 쓰지 않음"
            )
    missing = dict(NOT_PROVIDED)
    if geometry is not None:
        missing["invalid_geometry_count"] = (
            "See split/mode geometry scan; incomplete scans remain unknown."
        )
        missing["self_intersection"] = "See proper crossing counts in geometry_diagnostics.json."
    L += ["", "## 제공되지 않은 지표 / 별도 검사 상태"] + [
        f"  - {k}: {v}" for k, v in missing.items()
    ]
    L += ["", "## [임시 비교] validation 실제 점수 vs MGN 임시 기준 — 공식 판정 아님"]
    for mode, c in comparisons.items():
        L.append(f"  {mode} (기준 {c['policy_version']}, status={c['policy_status']})")
        L.append(f"    {'지표':<32}{'관측':>8}{'임시 상한':>8}  결과")
        for r in c["rows"]:
            limit = "≤ " + fmt(r["provisional_max"])
            L.append(
                f"    {r['metric']:<34}{fmt(r['observed']):>10}{limit:>12}  "
                f"{r['result']} ({r['unit']})"
            )
        for k in ("depth_error", "iou"):
            L.append(f"    {k:<34}{'미측정':>9}{'-':>12}  제공되지 않음")
        if c.get("per_recipe"):
            over = c["per_recipe"]
            more = " ..." if len(over["rids"]) > 10 else ""
            L.append(
                f"    참고: 레시피별 임시 상한 초과 {over['recipes_over_provisional_max']}건 "
                f"{over['rids'][:10]}{more}"
            )
    L += ["", "## [최종 판정] controller.decide (validation만)"]
    for mode, d in decisions.items():
        L.append(f"  {mode}: {d['action']} - " + " / ".join(d["reasons"]))
        for b in d.get("remaining_blockers_even_if_approved", []):
            L.append(f"    기준을 승인해도 남는 보류 사유: {b}")
        L.append(f"    학습 시작 {d['training_started']} / 모델 교체 {d['active_model_changed']}")
    L += ["", "## [미확인 사항] 판정과 별개"] + [f"  - {x}" for x in open_items["items"]]
    return "\n".join(L) + "\n"


def run(folder, thresholds, output_root, *, check_geometry=False):
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []
    sources = {
        str(p): dict(
            size_bytes=p.stat().st_size,
            sha256=sha256(p),
            parsed=p.suffix.lower() in (".csv", ".json"),
            note=None if p.suffix.lower() in (".csv", ".json") else "hashed only, not parsed",
        )
        for p in files
    }
    problems = Problems()
    readme = load_readme(folder, problems)
    series = load_steps(folder, problems)
    summary_rows, raw_summary = load_summary(folder, problems)
    evidence = (
        cross_check(series, summary_rows, declared_ranges(readme), problems)
        if series and summary_rows
        else {}
    )
    bad = sorted({p["rid"] for p in problems if p["rid"]})
    data_check = dict(passed=not problems, invalid_recipe_count=len(bad), reference_verified=False)
    inspection = dict(
        problems=list(problems),
        problem_counts=dict(Counter(p["kind"] for p in problems)),
        invalid_recipe_ids=bad,
        data_check=data_check,
        consistency_evidence=evidence,
        declared_split_ranges={
            k: f"r{a:04d}-r{b:04d}" for k, (a, b) in declared_ranges(readme).items()
        },
    )
    recipes = per_recipe(series, summary_rows)
    summary = summarize(recipes)
    geometry = None
    if check_geometry and series:
        expected = [
            dict(rid=rid, split=SPLIT_MAP[items[0]["split"]], steps=[x["step"] for x in items])
            for rid, items in sorted(series.items())
            if items[0]["split"] in SPLIT_MAP
            and [x["step"] for x in items] == list(range(1, len(items) + 1))
        ]
        geometry = mgn_geometry.scan_dataset(folder, expected)
        for key, group in geometry["groups"].items():
            if key in summary:
                summary[key]["geometry"] = group
                if group["complete"]:
                    summary[key]["not_provided"] = [
                        k
                        for k in summary[key]["not_provided"]
                        if k not in ("invalid_geometry_count", "self_intersection")
                    ]
        for name, meta in sources.items():
            if Path(name).name in geometry["files_sha256"]:
                meta["parsed"] = not any(e["source"] == Path(name).name for e in geometry["errors"])
                meta["note"] = (
                    "Numeric NPZ geometry scan; errors/completeness in geometry_diagnostics.json"
                )
    dataset_id = (
        f"mgn_main_s1_nov9_val_{sha256(folder / 'metrics_by_step.csv')[:12]}"
        if series
        else "unknown"
    )
    policies = load_thresholds(thresholds)
    inputs, comparisons, decisions = {}, {}, {}
    for mode in ("onestep", "rollout"):
        entry = summary.get(f"validation/{mode}")
        if entry is None:
            continue
        inputs[mode] = payload(mode, entry, data_check, readme, dataset_id)
        group = (geometry or {}).get("groups", {}).get(f"validation/{mode}")
        if group is not None:
            inputs[mode]["geometry"] = group
            inputs[mode]["geometry_contract"] = mgn_geometry.DEFINITION
            if group["complete"]:
                inputs[mode]["metrics"]["invalid_geometry_count"] = group["invalid_geometry_count"]
                inputs[mode]["metric_aggregation"]["invalid_geometry_count"] = (
                    f"Unique problem prediction frames across all steps; {mgn_geometry.CONTRACT}; "
                    "includes output-connectivity duplicates as a provisional diagnostic"
                )
                inputs[mode]["not_provided"] = [
                    k
                    for k in inputs[mode]["not_provided"]
                    if k not in ("invalid_geometry_count", "self_intersection")
                ]
        policy, decision = judge(inputs[mode], policies)
        if decision["training_started"] or decision["active_model_changed"]:
            raise AssertionError("The adapter must only decide and record.")
        decisions[mode] = dict(
            decision,
            remaining_blockers_even_if_approved=(
                remaining_blockers(inputs[mode], policy) if policy else []
            ),
        )
        if group is not None and group["complete"] and group["invalid_geometry_count"] > 0:
            decisions[mode]["remaining_blockers_even_if_approved"].append(
                f"Geometry scan found {group['invalid_geometry_count']} invalid prediction frames; "
                "quality limits would still fail even after threshold approval."
            )
        if policy is not None:
            comparisons[mode] = dict(
                provisional_comparison(inputs[mode], policy),
                per_recipe=recipes_over(
                    recipes[("validation", mode)],
                    (policy.get("max_metrics") or {}).get("chamfer_um_recipe_mean"),
                ),
            )
            if group is not None:
                comparisons[mode]["geometry"] = group
    if not inputs:
        from controller import outcome

        decisions["none"] = outcome("DATA_REVIEW", ["No usable validation rows."], {}, {})
    open_items = unverified(readme, geometry)
    test_record = {k: v for k, v in summary.items() if k.startswith("test/")}
    if {p: sha256(p) for p in sources} != {p: v["sha256"] for p, v in sources.items()}:
        raise AssertionError("A source file changed during conversion.")

    out = Path(output_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "source_files.json", dict(folder=str(folder), files=sources))
    write_json(out / "readme_as_provided.json", readme)
    write_json(
        out / "selected_rows.json",
        dict(
            note=(
                "parsed per-step rows (line numbers refer to metrics_by_step.csv) "
                "and summary rows as original strings"
            ),
            per_step=series,
            summary_rows=raw_summary,
        ),
    )
    write_json(out / "inspection.json", inspection)
    write_json(out / "score_summary.json", summary)
    write_json(
        out / "test_scores_record.json",
        dict(
            notice="test 점수: 참고 기록만. 임시 기준 비교·조정과 판정에 사용하지 않음.",
            by_mode=test_record,
        ),
    )
    missing = dict(NOT_PROVIDED)
    if geometry is not None:
        for name in ("invalid_geometry_count", "self_intersection"):
            missing[name] = (
                "Not provided by the model team; locally inspected in this run. "
                "See geometry_diagnostics.json for completeness and measured counts."
            )
    write_json(out / "not_provided.json", missing)
    if geometry is not None:
        mgn_geometry.write_outputs(out, geometry)
    for mode, p in inputs.items():
        write_json(out / f"harness_input_validation_{mode}.json", p)
    write_json(
        out / "provisional_comparison.json",
        dict(notice="임시 비교용, validation만. 공식 판정 아님.", by_mode=comparisons),
    )
    write_json(
        out / "official_decisions.json",
        dict(thresholds=str(thresholds), split="validation", by_mode=decisions),
    )
    write_json(out / "unverified.json", open_items)
    text = report(
        readme, len(sources), inspection, summary, comparisons, decisions, open_items, geometry
    )
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
            unverified=open_items,
            recipes=recipes,
            geometry=geometry,
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument("--output-root", default=str(ROOT / "mgn_adapter_results"))
    parser.add_argument(
        "--check-geometry", action="store_true", help="inspect numeric NPZ segment geometry"
    )
    args = parser.parse_args()
    out, text, result = run(
        args.input, args.thresholds, args.output_root, check_geometry=args.check_geometry
    )
    print(text)
    print("결과 폴더:", out)
    if args.check_geometry and (result["geometry"] is None or result["geometry"]["errors"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
