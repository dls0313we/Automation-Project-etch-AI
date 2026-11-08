"""Inspect existing 6-epoch MGN evaluation CSVs and pass the data check to controller.decide.

Read-only on the source CSVs. Standard library only. Does not train or replace any model.
The policy is a draft, so a clean inspection ends in HOLD, never in a pass or retrain approval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from controller import decide

ROOT = Path(__file__).resolve().parent
SPLITS = ("test", "ood")
FILES = {
    "recipe_conditions.csv": [
        "rid",
        "split",
        "step_min",
        "ion_flux",
        "etchant_flux",
        "oxygen_flux",
        "ion_mean_energy",
        "ion_exponent",
        "mask_width",
        "mask_height",
    ],
    "per_recipe.csv": ["split", "rid", "model_um", "baseline_um", "depth_mae_um"],
    "per_step.csv": ["split", "rid", "step", "model_um", "baseline_um", "depth_mae_um"],
    "summary.csv": ["split", "recipes", "model_um", "baseline_um", "depth_mae_um", "gain_pct"],
}
CONDITION_NUMERIC = FILES["recipe_conditions.csv"][2:]
METRIC_COLUMNS = ["model_um", "baseline_um", "depth_mae_um"]
UNMEASURED = {
    "self_intersection": "Not measurable from these CSVs (no mesh/geometry output).",
    "rollout": (
        "Not measurable from these CSVs; per_step values are not identified as a free rollout."
    ),
}
# Draft policy: no approved thresholds. Mirrors examples/real.json.
DRAFT_POLICY = dict(
    version="team_policy_TBD",
    status="draft",
    evaluation_contract_id="TBD_define_metric_units_aggregation_and_horizon",
    primary_metric="mean_chamfer_um",
    max_metrics=dict(mean_chamfer_um=None, depth_mae_um=None, invalid_geometry_count=None),
    max_relative_regression=None,
    min_primary_improvement_relative=None,
    denominator_floor=None,
    note="No approved thresholds. Define validation protocol and permitted errors with the team.",
)


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
    """Return a finite float, or None when the cell is missing/unparseable/non-finite."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class Report:
    def __init__(self):
        self.problems = []

    def add(self, kind, file, detail, rid=None):
        self.problems.append(dict(kind=kind, file=file, rid=rid, detail=detail))


def check_columns(report, name, header):
    missing = [c for c in FILES[name] if c not in header]
    if missing:
        report.add("missing_columns", name, f"missing columns: {missing}")
    return not missing


def check_rows(report, name, rows, key_fields):
    """Duplicate keys and empty IDs. Returns {key: row} of first occurrences."""
    seen, counts = {}, Counter(tuple(r.get(k) for k in key_fields) for r in rows)
    for key, n in counts.items():
        if n > 1:
            report.add(
                "duplicate",
                name,
                f"{dict(zip(key_fields, key))} appears {n} times",
                key[key_fields.index("rid")],
            )
    for i, r in enumerate(rows, start=2):  # line 1 is the header
        if not (r.get("rid") or "").strip():
            report.add("missing_id", name, f"line {i}: empty rid")
            continue
        if r.get("split") not in SPLITS:
            report.add("bad_split", name, f"line {i}: split={r.get('split')!r}", r.get("rid"))
        seen.setdefault(tuple(r.get(k) for k in key_fields), r)
    return seen


def check_numbers(report, name, rows, columns, nonnegative):
    values = {}
    for i, r in enumerate(rows, start=2):
        for c in columns:
            v = to_float(r.get(c))
            if v is None:
                report.add(
                    "non_finite_or_missing_number",
                    name,
                    f"line {i}: {c}={r.get(c)!r}",
                    r.get("rid"),
                )
            elif c in nonnegative and v < 0:
                report.add("negative_value", name, f"line {i}: {c}={v}", r.get("rid"))
            values[(i, c)] = v
    return values


def inspect(folder):
    folder = Path(folder)
    report = Report()
    files, tables = {}, {}
    for name in FILES:
        path = folder / name
        if not path.is_file():
            report.add("missing_file", name, "file not found")
            continue
        header, rows = read_csv(path)
        files[name] = dict(path=str(path), sha256=sha256(path), rows=len(rows), columns=header)
        if check_columns(report, name, header):
            tables[name] = rows

    result = dict(
        source_folder=str(folder),
        files=files,
        recipe_conditions=None,
        consistency={},
        unmeasured=UNMEASURED,
        reported_metrics_not_judged=None,
    )

    # recipe_conditions.csv: counts and negative ion mean energy, computed from the data.
    cond = tables.get("recipe_conditions.csv")
    cond_ids = {}
    if cond is not None:
        cond_ids = {
            k[0]: r for k, r in check_rows(report, "recipe_conditions.csv", cond, ["rid"]).items()
        }
        check_numbers(report, "recipe_conditions.csv", cond, CONDITION_NUMERIC, nonnegative=())
        negative = []
        for r in cond:
            v = to_float(r.get("ion_mean_energy"))
            if v is not None and v < 0:
                negative.append(dict(rid=r["rid"], split=r.get("split"), ion_mean_energy=v))
                report.add(
                    "negative_ion_mean_energy",
                    "recipe_conditions.csv",
                    f"ion_mean_energy={v}",
                    r["rid"],
                )
        for c in CONDITION_NUMERIC:
            if c == "ion_mean_energy":
                continue
            for i, r in enumerate(cond, start=2):
                v = to_float(r.get(c))
                if v is not None and v < 0:
                    report.add(
                        "negative_value",
                        "recipe_conditions.csv",
                        f"line {i}: {c}={v}",
                        r.get("rid"),
                    )
        split_counts = Counter(r.get("split") for r in cond)
        result["recipe_conditions"] = dict(
            recipe_count=len(cond_ids),
            row_count=len(cond),
            split_counts={s: split_counts.get(s, 0) for s in SPLITS},
            negative_ion_mean_energy_count=len(negative),
            negative_ion_mean_energy=negative,
        )

    # per_recipe.csv vs recipe_conditions.csv
    per_recipe = tables.get("per_recipe.csv")
    pr_ids = {}
    if per_recipe is not None:
        pr_ids = {
            k[0]: r for k, r in check_rows(report, "per_recipe.csv", per_recipe, ["rid"]).items()
        }
        check_numbers(
            report, "per_recipe.csv", per_recipe, METRIC_COLUMNS, nonnegative=METRIC_COLUMNS
        )

    # per_step.csv: (rid, step) unique, steps contiguous from 0, metrics finite and nonnegative
    per_step = tables.get("per_step.csv")
    steps = defaultdict(list)
    if per_step is not None:
        check_rows(report, "per_step.csv", per_step, ["rid", "step"])
        check_numbers(report, "per_step.csv", per_step, METRIC_COLUMNS, nonnegative=METRIC_COLUMNS)
        for i, r in enumerate(per_step, start=2):
            s = to_float(r.get("step"))
            if s is None or s < 0 or int(s) != s:
                report.add(
                    "bad_step", "per_step.csv", f"line {i}: step={r.get('step')!r}", r.get("rid")
                )
                continue
            steps[r.get("rid")].append((int(s), r))
        for rid, items in steps.items():
            got = sorted({s for s, _ in items})
            if got != list(range(len(got))):
                report.add("step_gap", "per_step.csv", f"steps not contiguous from 0: {got}", rid)
        result["consistency"]["per_step_steps_per_recipe"] = dict(
            sorted(Counter(len({s for s, _ in v}) for v in steps.values()).items())
        )

    # ID agreement and split agreement across files
    id_sets = {
        "recipe_conditions.csv": set(cond_ids),
        "per_recipe.csv": set(pr_ids),
        "per_step.csv": set(steps),
    }
    id_sets = {k: v for k, v in id_sets.items() if tables.get(k) is not None}
    if "recipe_conditions.csv" in id_sets:
        ref = id_sets["recipe_conditions.csv"]
        for name, ids in id_sets.items():
            if name == "recipe_conditions.csv":
                continue
            for rid in sorted(ref - ids):
                report.add(
                    "id_missing", name, "rid present in recipe_conditions.csv but missing here", rid
                )
            for rid in sorted(ids - ref):
                report.add("id_unknown", name, "rid not present in recipe_conditions.csv", rid)
        for rid, r in pr_ids.items():
            if rid in cond_ids and r.get("split") != cond_ids[rid].get("split"):
                report.add(
                    "split_mismatch",
                    "per_recipe.csv",
                    f"{r.get('split')} vs {cond_ids[rid].get('split')}",
                    rid,
                )
        for rid, items in steps.items():
            if rid in cond_ids and any(
                row.get("split") != cond_ids[rid].get("split") for _, row in items
            ):
                report.add(
                    "split_mismatch",
                    "per_step.csv",
                    "split differs from recipe_conditions.csv",
                    rid,
                )
    result["consistency"]["id_counts"] = {k: len(v) for k, v in id_sets.items()}

    # Aggregation: per_recipe == mean(per_step); summary == mean(per_recipe) and recipe counts
    agg_mismatch = 0
    for rid, r in pr_ids.items():
        if rid not in steps:
            continue
        for c in METRIC_COLUMNS:
            want = to_float(r.get(c))
            vals = [to_float(row.get(c)) for _, row in steps[rid]]
            if want is None or any(v is None for v in vals):
                continue
            if not math.isclose(statistics.fmean(vals), want, rel_tol=1e-9, abs_tol=1e-12):
                agg_mismatch += 1
                report.add(
                    "aggregate_mismatch",
                    "per_recipe.csv",
                    f"{c} differs from mean of per_step",
                    rid,
                )
    summary = tables.get("summary.csv")
    reported = {}
    if summary is not None:
        seen_split = Counter(r.get("split") for r in summary)
        for s, n in seen_split.items():
            if n > 1:
                report.add("duplicate", "summary.csv", f"split {s!r} appears {n} times")
        for i, r in enumerate(summary, start=2):
            s = r.get("split")
            nums = {c: to_float(r.get(c)) for c in FILES["summary.csv"][1:]}
            for c, v in nums.items():
                if v is None:
                    report.add(
                        "non_finite_or_missing_number", "summary.csv", f"line {i}: {c}={r.get(c)!r}"
                    )
            reported[s] = nums
            members = [x for x in pr_ids.values() if x.get("split") == s]
            if nums["recipes"] is not None and nums["recipes"] != len(members):
                report.add(
                    "count_mismatch",
                    "summary.csv",
                    f"{s}: recipes={nums['recipes']} but per_recipe has {len(members)}",
                )
            for c in METRIC_COLUMNS:
                vals = [to_float(x.get(c)) for x in members]
                if nums[c] is None or not vals or any(v is None for v in vals):
                    continue
                if not math.isclose(statistics.fmean(vals), nums[c], rel_tol=1e-9, abs_tol=1e-12):
                    agg_mismatch += 1
                    report.add(
                        "aggregate_mismatch",
                        "summary.csv",
                        f"{s}: {c} differs from mean of per_recipe",
                    )
        for s in SPLITS:
            if s not in seen_split and pr_ids:
                report.add("missing_split", "summary.csv", f"no row for split {s!r}")
    result["consistency"]["aggregate_mismatch_count"] = agg_mismatch
    result["reported_metrics_not_judged"] = dict(
        values=reported,
        note="Copied from summary.csv for the record only. "
        "Metric definitions/units are unconfirmed and "
        "the policy is draft, so these values are not judged.",
    )

    invalid_rids = sorted({p["rid"] for p in report.problems if p["rid"]})
    result["problems"] = report.problems
    result["problem_counts"] = dict(sorted(Counter(p["kind"] for p in report.problems).items()))
    result["invalid_recipe_ids"] = invalid_rids
    result["data_check"] = dict(
        passed=not report.problems,
        invalid_recipe_count=len(invalid_rids),
        # The CSVs carry no independent reference to verify against.
        reference_verified=False,
    )
    return result


def controller_input(inspection, model_id):
    folder_digest = hashlib.sha256(
        "".join(f"{n}:{f['sha256']}" for n, f in sorted(inspection["files"].items())).encode()
    ).hexdigest()[:12]
    counts = inspection["recipe_conditions"] or {}
    current = dict(
        model_id=model_id,
        evaluation_dataset_id=f"mgn_eval_results_{folder_digest}",
        evaluation_contract_id="original_one_step_diagnostic_only",
        split="test",
        unit="um",
        recipe_count=counts.get("recipe_count", 0),
        data_origin="existing_6epoch_mgn_evaluation",
        training_data_ready=False,
        data_check=inspection["data_check"],
        metrics={},  # metric definitions unconfirmed; nothing is judged against draft limits
    )
    return dict(current=current, policy=DRAFT_POLICY, candidate=None)


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run(input_folder, output_root, model_id):
    inspection = inspect(input_folder)
    before = {n: f["sha256"] for n, f in inspection["files"].items()}
    ctrl = controller_input(inspection, model_id)
    decision = decide(ctrl["current"], ctrl["policy"], ctrl["candidate"])
    if decision["training_started"] or decision["active_model_changed"]:
        raise AssertionError("Inspection must only decide and record results.")
    after = {n: sha256(Path(f["path"])) for n, f in inspection["files"].items()}
    if before != after:
        raise AssertionError("Source CSV changed during inspection.")

    output = Path(output_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "inspection.json", inspection)
    write_json(
        output / "source_sha256.json",
        {n: dict(path=f["path"], sha256=f["sha256"]) for n, f in inspection["files"].items()},
    )
    write_json(output / "controller_input.json", ctrl)
    write_json(output / "decision.json", decision)
    return output, inspection, decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="folder with the four evaluation CSVs")
    parser.add_argument("--output-root", default=str(ROOT / "inspection_results"))
    parser.add_argument("--model-id", default="MGN_6epoch_existing_eval")
    args = parser.parse_args()
    output, inspection, decision = run(args.input, args.output_root, args.model_id)
    rc = inspection["recipe_conditions"] or {}
    print("자료: 기존 6epoch MGN 평가 결과 (데이터 점검만, 학습·모델 교체 없음)")
    print("파일별 행 수:", {n: f["rows"] for n, f in inspection["files"].items()})
    print("레시피 수:", rc.get("recipe_count"), "/ split:", rc.get("split_counts"))
    print("평균 이온 에너지 음수:", rc.get("negative_ion_mean_energy_count"), "건")
    for item in rc.get("negative_ion_mean_energy", []):
        print("  ", item["rid"], item["split"], item["ion_mean_energy"])
    print("문제 종류별 건수:", inspection["problem_counts"] or "없음")
    print("잘못된 레시피 ID 수:", inspection["data_check"]["invalid_recipe_count"])
    print("미측정:", ", ".join(inspection["unmeasured"]))
    print("판정:", decision["action"], "-", decision["reasons"][0])
    print(
        "학습 시작:", decision["training_started"], "/ 모델 교체:", decision["active_model_changed"]
    )
    print("결과 폴더:", output)


if __name__ == "__main__":
    main()
