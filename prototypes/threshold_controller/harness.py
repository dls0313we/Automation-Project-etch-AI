"""Harness: read an evaluation result file + an editable threshold file, decide, and report.

Reuses controller.decide for every decision. This file only adds:
  - threshold selection by (model_family, eval_mode) from one human-editable JSON file,
  - a readable report (model, mode, observed values, limits, decision, reasons),
  - a new output folder per run.
Does not compute metrics, train, replace models, download data or call external services.

  python harness.py --demo
  python harness.py --eval <evaluation.json> [--candidate <evaluation.json>]
                    [--thresholds thresholds_demo.json]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from controller import decide, outcome

ROOT = Path(__file__).resolve().parent
FAMILIES = ("MGN", "FNO")
MODES = ("onestep", "rollout")
ACTION_KO = {
    "KEEP_CURRENT": "현재 모델 유지",
    "RETRAIN_REQUEST": "재학습 요청",
    "WAIT_FOR_TRAINING_DATA": "학습 데이터 준비 대기",
    "DATA_REVIEW": "데이터 재검토",
    "HOLD": "판단 보류",
    "APPROVE_CANDIDATE": "교체 승인 권고 (실제 교체는 하지 않음)",
}


class ThresholdFileError(ValueError):
    pass


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON object required")
    return value


def load_thresholds(path):
    """Index policies by (model_family, eval_mode). A broken threshold file is a hard error."""
    data = read_json(path)
    policies = data.get("policies")
    if not isinstance(policies, list) or not policies:
        raise ThresholdFileError(f'{path}: "policies" list required')
    index = {}
    for p in policies:
        key = (p.get("model_family"), p.get("eval_mode"))
        if key[0] not in FAMILIES or key[1] not in MODES:
            raise ThresholdFileError(
                f"{path}: model_family/eval_mode must be {FAMILIES}/{MODES}, got {key}"
            )
        if key in index:
            raise ThresholdFileError(f"{path}: duplicate entry for {key}")
        index[key] = p
    return index


def to_payload(evaluation):
    """Evaluation file -> controller payload. model_id is derived so family/version stay visible."""
    payload = dict(evaluation)
    payload["model_id"] = f"{evaluation.get('model_family')}/{evaluation.get('model_version')}"
    return payload


def judge(current, thresholds, candidate=None):
    """Return (policy_used_or_None, decision). Evaluation-file problems become HOLD, not crashes."""
    cur = to_payload(current)
    cand = to_payload(candidate) if candidate is not None else None
    for label, e in [("current", current), ("candidate", candidate)]:
        if e is None:
            continue
        problems = []
        if e.get("model_family") not in FAMILIES:
            problems.append(f"model_family must be one of {FAMILIES}")
        if e.get("eval_mode") not in MODES:
            problems.append(f"eval_mode must be one of {MODES}")
        if not isinstance(e.get("model_version"), str) or not e["model_version"].strip():
            problems.append("model_version required")
        if problems:
            return None, outcome("HOLD", [f"{label}: {p}" for p in problems], {}, cur, cand)
    key = (current["model_family"], current["eval_mode"])
    policy = thresholds.get(key)
    if policy is None:
        return None, outcome("HOLD", [f"No threshold entry for {key[0]}/{key[1]}."], {}, cur, cand)
    # decide() rejects a candidate or policy from a different family/mode (controller.validate).
    return policy, decide(cur, policy, cand)


def fmt(v):
    if v is None:
        return "누락"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{v:g}"
    return str(v)


def report(label, current, candidate, policy, decision):
    lines = [f"== {label}"]
    for name, e in [("현재 모델", current), ("후보 모델", candidate)]:
        if e is None:
            continue
        lines.append(
            f"{name}: {e.get('model_family')} {e.get('model_version')} | "
            f"평가 방식: {e.get('eval_mode')} | "
            f"데이터: {e.get('evaluation_dataset_id')} "
            f"({e.get('split')}, {fmt(e.get('recipe_count'))}건) | "
            f"자료: {e.get('data_origin')}"
        )
        dc = e.get("data_check") or {}
        lines.append(
            f"  데이터 점검: passed={dc.get('passed')}, "
            f"invalid_recipe_count={dc.get('invalid_recipe_count')}, "
            f"reference_verified={dc.get('reference_verified')}"
        )
        if e.get("_notice"):
            lines.append(f"  ※ {e['_notice']}")
    if policy is None:
        lines.append("임계치: 선택되지 않음")
    else:
        lines.append(f"임계치: {policy.get('version')} (status={policy.get('status')})")
        units = policy.get("metric_units") or {}
        # Korean labels are double-width in a terminal: 2 characters take 4 columns.
        head = (
            f"  {'지표':<22}{'현재':>8}"
            + (f"{'후보':>8}" if candidate else "")
            + f"{'상한':>8}  단위"
        )
        lines.append(head)
        for m, limit in (policy.get("max_metrics") or {}).items():
            row = f"  {m:<24}{fmt((current.get('metrics') or {}).get(m)):>10}"
            if candidate:
                row += f"{fmt((candidate.get('metrics') or {}).get(m)):>10}"
            lines.append(row + f"{'≤ ' + fmt(limit):>10}  {units.get(m, '')}")
    lines.append(
        f"판정: {decision['action']} ({ACTION_KO.get(decision['action'], decision['action'])})"
    )
    for r in decision["reasons"]:
        lines.append(f"이유: {r}")
    if decision["evidence"]:
        lines.append("근거: " + json.dumps(decision["evidence"], ensure_ascii=False))
    lines.append(
        f"학습 시작: {decision['training_started']} / 모델 교체: {decision['active_model_changed']}"
    )
    return "\n".join(lines)


def new_output_dir(root):
    out = Path(root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)  # never reuse/overwrite an earlier run
    return out


def run_cases(cases, thresholds_path, output_root):
    """Run case dicts with file paths; return (directory, results, report texts)."""
    thresholds = load_thresholds(thresholds_path)
    out = new_output_dir(output_root)
    results, texts = [], []
    for case in cases:
        current = read_json(case["current"])
        candidate = read_json(case["candidate"]) if case.get("candidate") else None
        policy, decision = judge(current, thresholds, candidate)
        if decision["training_started"] or decision["active_model_changed"]:
            raise AssertionError("The harness must only decide and record.")
        text = report(case["label"], current, candidate, policy, decision)
        texts.append(text)
        record = dict(
            case=case["id"],
            label=case["label"],
            inputs=dict(
                current=str(case["current"]),
                candidate=str(case["candidate"]) if case.get("candidate") else None,
                thresholds=str(thresholds_path),
            ),
            current=current,
            candidate=candidate,
            policy_used=policy,
            decision=decision,
            expected=case.get("expected"),
        )
        (out / f"{case['id']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        results.append(
            dict(
                case=case["id"],
                label=case["label"],
                action=decision["action"],
                expected=case.get("expected"),
            )
        )
    (out / "report.txt").write_text("\n\n".join(texts) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out, results, texts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the virtual scenarios in harness_inputs/scenarios.json",
    )
    parser.add_argument("--eval", help="evaluation result JSON of the current model")
    parser.add_argument(
        "--candidate", help="evaluation result JSON of a candidate model (optional)"
    )
    parser.add_argument("--thresholds", default=str(ROOT / "thresholds_demo.json"))
    parser.add_argument("--output-root", default=str(ROOT / "harness_results"))
    args = parser.parse_args()
    if args.demo == bool(args.eval):
        parser.error("use exactly one of --demo or --eval")
    if args.demo:
        spec = read_json(ROOT / "harness_inputs" / "scenarios.json")
        cases = [
            dict(
                c,
                current=ROOT / c["current"],
                candidate=ROOT / c["candidate"] if c.get("candidate") else None,
            )
            for c in spec["scenarios"]
        ]
    else:
        cases = [
            dict(
                id="decision",
                label=Path(args.eval).name,
                current=args.eval,
                candidate=args.candidate,
            )
        ]
    out, results, texts = run_cases(cases, args.thresholds, args.output_root)
    print("\n\n".join(texts))
    print()
    mismatched = [r for r in results if r["expected"] and r["expected"] != r["action"]]
    for r in mismatched:
        print(f"예상과 다름: {r['case']} expected={r['expected']} observed={r['action']}")
    print(f"완료: {len(results)}개 판정 / 결과 폴더: {out}")
    if mismatched:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
