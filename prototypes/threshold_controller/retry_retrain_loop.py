"""Bounded mock retraining retries inspired by ADR-0002. Standard library only.

Runs synthetic inputs only. No simulator, model training, checkpoint creation,
registry change or model replacement occurs. Existing judge/decide stays unchanged.
The four training configurations are demonstration proposals, not measured choices.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from harness import judge, load_thresholds, read_json
from mock_retrain_loop import MockTrainer, MockTrainingFailed

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "retry_loop_inputs"
MAX_ATTEMPTS = 4
ATTEMPT_CONFIGS = [
    dict(
        attempt=1,
        train_recipe_count=40,
        epochs=5,
        learning_rate=0.001,
        additional_recipes_planned=0,
        training_dataset_id="mock-drift-train-40",
    ),
    dict(
        attempt=2,
        train_recipe_count=100,
        epochs=5,
        learning_rate=0.001,
        additional_recipes_planned=60,
        training_dataset_id="mock-drift-train-100",
    ),
    dict(
        attempt=3,
        train_recipe_count=100,
        epochs=12,
        learning_rate=0.001,
        additional_recipes_planned=0,
        training_dataset_id="mock-drift-train-100",
    ),
    dict(
        attempt=4,
        train_recipe_count=100,
        epochs=12,
        learning_rate=0.0005,
        additional_recipes_planned=0,
        training_dataset_id="mock-drift-train-100",
    ),
]
STATUS_LABELS = {
    "RECOMMEND_REPLACEMENT": "후보 통과, 교체 권고 후 종료",
    "EXHAUSTED_KEEP_CURRENT": "횟수 상한 도달, 기존 모델 유지",
    "NO_RETRAIN_NEEDED": "현재 기준 만족, 재학습 요청 없음",
    "NOT_STARTED_HOLD": "판단 보류, 모의 학습 시작 안 함",
    "NOT_STARTED_DATA_REVIEW": "입력 데이터 검토, 시작 안 함",
    "NOT_STARTED_WAIT_FOR_TRAINING_DATA": "학습 데이터 준비 대기",
    "REJECTED_NON_DEMO_INPUT": "실제 자료 입력 거부",
    "STOPPED_CANDIDATE_EVALUATION": "후보 평가 불가, 즉시 중단",
    "STOPPED_DATA_REVIEW": "후보 데이터 문제, 즉시 중단",
    "STOPPED_TRAINING_FAILED": "모의 실행 실패, 즉시 중단",
    "STOPPED_NON_DEMO_CANDIDATE": "실제 후보 입력 거부, 중단",
    "STOPPED_PLAN_MISSING": "다음 모의 후보 계획 없음, 중단",
}


class MockRetryRunner:
    """Reuse the existing mock runner. It returns scripted scores, never a model."""

    def run(self, request, plan, base):
        return MockTrainer(plan, base).run(request)


def _result(record, status, reasons):
    return dict(record, status=status, reasons=list(reasons))


def run_case(current, thresholds, plans, *, base=INPUTS, runner=None, max_attempts=4):
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise ValueError("Attempt limit must be an integer from 1 to 4")
    if not 1 <= max_attempts <= MAX_ATTEMPTS:
        raise ValueError("Attempt limit must be an integer from 1 to 4")
    current, thresholds = copy.deepcopy(current), copy.deepcopy(thresholds)
    record = dict(
        current_model=current.get("model_version"),
        recommended_model=None,
        passed_attempt=None,
        max_attempts=max_attempts,
        attempts=[],
        real_training_executed=False,
        real_model_replaced=False,
        real_data_generation_executed=False,
        strategy_status="demo_proposal_not_calibrated",
        scope="Bounded mock controller only; not a full C8 retraining completion.",
    )
    if current.get("data_origin") != "synthetic_demo":
        return _result(
            record, "REJECTED_NON_DEMO_INPUT", ["Only synthetic_demo inputs are accepted"]
        )
    policy, initial = judge(current, thresholds)
    record["initial_decision"] = initial
    record["policy_snapshot"] = copy.deepcopy(policy)
    record["current_evaluation"] = copy.deepcopy(current)
    if initial["action"] != "RETRAIN_REQUEST":
        status = {
            "KEEP_CURRENT": "NO_RETRAIN_NEEDED",
            "HOLD": "NOT_STARTED_HOLD",
            "DATA_REVIEW": "NOT_STARTED_DATA_REVIEW",
            "WAIT_FOR_TRAINING_DATA": "NOT_STARTED_WAIT_FOR_TRAINING_DATA",
        }.get(initial["action"], "NOT_STARTED_HOLD")
        return _result(record, status, initial["reasons"])

    runner = runner or MockRetryRunner()
    for index in range(max_attempts):
        if index >= len(plans):
            return _result(record, "STOPPED_PLAN_MISSING", ["No scripted plan for next attempt"])
        config = copy.deepcopy(ATTEMPT_CONFIGS[index])
        request = dict(
            decision="RETRAIN_REQUEST",
            data_origin="synthetic_demo",
            starting_model=current["model_version"],
            config=copy.deepcopy(config),
            evaluation_dataset_id=current["evaluation_dataset_id"],
            evaluation_contract_id=current["evaluation_contract_id"],
            policy_version=(policy or {}).get("version"),
        )
        attempt = dict(
            attempt=index + 1,
            at=datetime.now().isoformat(timespec="seconds"),
            config=config,
            starting_model=current["model_version"],
            candidate_model=None,
            mock_training_executed=True,
            real_training_executed=False,
            real_data_generation_executed=False,
            real_model_replaced=False,
            mock_gate_result="STOP",
            decision=dict(
                action="NOT_EVALUATED",
                reasons=[],
                training_started=False,
                active_model_changed=False,
            ),
        )
        record["attempts"].append(attempt)
        try:
            response = runner.run(request, copy.deepcopy(plans[index]), Path(base))
            candidate = copy.deepcopy(response["candidate"])
        except (MockTrainingFailed, OSError, ValueError, KeyError, TypeError) as exc:
            attempt["runner_error"] = f"{type(exc).__name__}: {exc}"
            return _result(record, "STOPPED_TRAINING_FAILED", [attempt["runner_error"]])
        if not isinstance(candidate, dict) or candidate.get("data_origin") != "synthetic_demo":
            return _result(
                record,
                "STOPPED_NON_DEMO_CANDIDATE",
                ["Mock candidate must be a synthetic_demo evaluation"],
            )
        source_id = candidate.get("model_version")
        if not isinstance(source_id, str) or not source_id.strip():
            return _result(record, "STOPPED_CANDIDATE_EVALUATION", ["Candidate identity missing"])
        # Unique IDs identify virtual attempts; no checkpoint is produced or renamed.
        candidate["model_version"] = f"{source_id}-retry-{index + 1}"
        attempt.update(
            candidate_model=candidate["model_version"],
            source_model_version=source_id,
            candidate_file=response.get("candidate_file"),
            candidate_evaluation=copy.deepcopy(candidate),
        )
        _, decision = judge(current, thresholds, candidate)
        attempt["decision"] = decision
        if decision["training_started"] or decision["active_model_changed"]:
            raise AssertionError("Decision-only harness unexpectedly recorded real execution")
        if decision["action"] == "APPROVE_CANDIDATE":
            attempt["mock_gate_result"] = "PASS"
            record.update(recommended_model=candidate["model_version"], passed_attempt=index + 1)
            return _result(record, "RECOMMEND_REPLACEMENT", decision["reasons"])
        if decision["action"] == "KEEP_CURRENT":
            attempt["mock_gate_result"] = "BLOCK"
            continue  # Only measured quality rejection advances the mock retry schedule.
        status = (
            "STOPPED_DATA_REVIEW"
            if decision["action"] == "DATA_REVIEW"
            else "STOPPED_CANDIDATE_EVALUATION"
        )
        return _result(record, status, decision["reasons"])
    return _result(
        record,
        "EXHAUSTED_KEEP_CURRENT",
        [f"All {max_attempts} mock candidates failed quality checks; keep current model"],
    )


def _inside(base, relative):
    root = Path(base).resolve().parent
    p = (Path(base) / relative).resolve()
    if not p.is_relative_to(root):
        raise ValueError("Scenario paths must stay within the demonstration input workspace")
    return p


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_text(results):
    lines = ["최대 4회 모의 재학습 재시도", "실제 데이터 생성·학습·모델 교체는 모두 미실행", ""]
    for r in results:
        lines += [
            f"{r['label']}: {STATUS_LABELS[r['status']]}",
            f"  시도 {len(r['attempts'])}/{r['max_attempts']}, 통과 시도 {r['passed_attempt']}",
        ]
        for a in r["attempts"]:
            c = a["config"]
            lines.append(
                f"  {a['attempt']}차: train {c['train_recipe_count']}, "
                f"epoch {c['epochs']}, lr {c['learning_rate']} -> "
                f"{a['mock_gate_result']} ({a['decision']['action']})"
            )
        lines += ["  이유: " + " / ".join(r["reasons"]), ""]
    return "\n".join(lines)


def render_html(summary):
    def e(value):
        return html.escape(str(value))

    blocks = []
    for r in summary["results"]:
        rows = []
        for a in r["attempts"]:
            c = a["config"]
            score = (a.get("candidate_evaluation", {}).get("metrics") or {}).get("mean_chamfer_um")
            rows.append(
                f"<tr><td>{a['attempt']}</td><td>{c['train_recipe_count']}</td>"
                f"<td>{c['additional_recipes_planned']}</td><td>{c['epochs']}</td>"
                f"<td>{c['learning_rate']}</td>"
                f"<td>{e(score if score is not None else '미측정')}</td>"
                f"<td>{a['mock_gate_result']}</td><td>{e(a['decision']['action'])}</td></tr>"
            )
        blocks.append(
            f"<section><h2>{e(r['label'])}</h2><p><strong>"
            f"{e(STATUS_LABELS[r['status']])}</strong> · 시도 {len(r['attempts'])}회</p>"
            "<table><tr><th>시도</th><th>학습 조건 수</th><th>추가 생성 계획</th>"
            "<th>에포크</th><th>학습률</th><th>가상 오차 µm</th><th>모의 판정</th>"
            f"<th>제어기 결정</th></tr>{''.join(rows)}</table>"
            f"<p>{e(' / '.join(r['reasons']))}</p></section>"
        )
    return (
        '<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>최대 4회 모의 재학습</title><style>"
        "body{font:16px/1.6 system-ui,sans-serif;color:#213047;background:#fff;margin:24px}"
        "main{max-width:1150px;margin:auto}section{margin:32px 0}h1{font-size:28px}"
        "table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;"
        "border-bottom:1px solid #d9dfe6}.notice{background:#fff3d4;padding:14px}"
        "</style><main><h1>최대 4회 모의 재학습 재시도</h1>"
        f"<p>{summary['scenario_count']}개 상황 중 "
        f"{summary['expected_matched_count']}개 예상 일치</p>"
        '<p class="notice">실제 시뮬레이션·학습·모델 교체는 실행하지 않았습니다. '
        "학습 조건 수·에포크·학습률은 ADR-0002를 참고한 시연용 제안입니다. "
        "PASS/BLOCK은 기존 프로토타입의 후보 판정을 요약한 모의 상태이며, "
        "실제 골든셋 게이트나 레지스트리 승격을 실행한 것은 아닙니다.</p>"
        "<p>성능 미달일 때만 다음 설정을 시도합니다. 누락·데이터 문제·실행 실패는 중단합니다. "
        "모든 후보는 같은 기존 모델, 같은 평가 데이터, 같은 임계치로 비교합니다.</p>"
        + "".join(blocks)
        + "</main></html>"
    )


def run(scenarios_path=INPUTS / "scenarios.json", output_root=ROOT / "retry_loop_results"):
    scenarios_path = Path(scenarios_path)
    spec, base = read_json(scenarios_path), scenarios_path.parent
    paths = {scenarios_path.resolve()}
    for s in spec["scenarios"]:
        paths.update(_inside(base, s[k]) for k in ["current", "thresholds"])
        paths.update(_inside(base, p["candidate"]) for p in s["plans"] if "candidate" in p)
    before = {str(p): _sha(p) for p in sorted(paths)}
    results = []
    for s in spec["scenarios"]:
        current = read_json(_inside(base, s["current"]))
        thresholds = load_thresholds(_inside(base, s["thresholds"]))
        # The existing MockTrainer reads paths relative to base; validate those above.
        result = run_case(current, thresholds, s["plans"], base=base)
        results.append(
            dict(
                result,
                id=s["id"],
                label=s["label"],
                expected=s["expected"],
                expected_matched=result["status"] == s["expected"],
            )
        )
    if before != {str(p): _sha(p) for p in sorted(paths)}:
        raise AssertionError("Demonstration source inputs changed during the run")
    summary = dict(
        notice="Bounded mock retry only. No real C8 completion, generation, training or promotion.",
        scenario_count=len(results),
        expected_matched_count=sum(r["expected_matched"] for r in results),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        results=results,
        real_training_executed=False,
        real_model_replaced=False,
        real_data_generation_executed=False,
        sources_unchanged=True,
        status_counts=dict(Counter(r["status"] for r in results)),
    )
    out = Path(output_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    (out / "cases").mkdir()
    for r in results:
        (out / "cases" / (r["id"] + ".json")).write_text(
            json.dumps(r, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (out / "source_files.json").write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")
    (out / "report.txt").write_text(render_text(results), encoding="utf-8")
    (out / "report.html").write_text(render_html(summary), encoding="utf-8")
    return out, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", default=str(INPUTS / "scenarios.json"))
    p.add_argument("--output-root", default=str(ROOT / "retry_loop_results"))
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    out, summary = run(args.scenarios, args.output_root)
    print(render_text(summary["results"]))
    print("예상 일치:", summary["expected_matched_count"], "/", summary["scenario_count"])
    print("보고서:", out / "report.html")
    if summary["expected_matched_count"] != summary["scenario_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
