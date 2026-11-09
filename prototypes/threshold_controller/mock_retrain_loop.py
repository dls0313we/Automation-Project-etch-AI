"""Mock retraining loop on top of the existing decision harness.

  python mock_retrain_loop.py

Flow per scenario:
  evaluate current model (harness.judge -> controller.decide)
  -> only if the decision is RETRAIN_REQUEST: call the MOCK training runner
  -> evaluate the mock candidate together with the current model (same harness.judge)
  -> APPROVE_CANDIDATE: replacement is recommended (not executed); otherwise keep the current model
  -> every step is recorded with separate flags for mock execution and real execution.

The mock runner never trains anything: it returns a pre-written virtual candidate evaluation
(or a scripted failure). Real training and real model replacement are never executed.
Only synthetic_demo inputs under mock_loop_inputs/ are accepted; real FNO/MGN results, their
provisional thresholds and the HOLD rules are not read or changed. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from harness import judge, load_thresholds, read_json

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "mock_loop_inputs"
LOOP_STATUS_KO = {
    "NO_RETRAIN_NEEDED": "기준 만족 — 재학습 요청 안 함",
    "NOT_STARTED_HOLD": "판단 보류 — 모의 학습 시작 안 함",
    "NOT_STARTED_OTHER": "재학습 요청이 아님 — 모의 학습 시작 안 함",
    "RECOMMEND_REPLACEMENT": "후보 교체 권고 (실제 교체는 하지 않음)",
    "KEEP_CURRENT_AFTER_CANDIDATE": "후보 미달 — 기존 모델 유지",
    "STOPPED_TRAINING_FAILED": "모의 학습 실패 — 중단",
    "STOPPED_CANDIDATE_EVALUATION": "후보 평가 불가 — 중단",
}


class MockTrainingFailed(RuntimeError):
    pass


class MockTrainer:
    """Stands in for a training runner.

    Returns a scripted virtual candidate evaluation; trains nothing.
    """

    def __init__(self, plan, base=INPUTS):
        self.plan, self.base, self.calls = plan, Path(base), 0

    def run(self, request):
        self.calls += 1
        if request.get("decision") != "RETRAIN_REQUEST":
            raise MockTrainingFailed("Mock runner called without a RETRAIN_REQUEST decision")
        if request.get("data_origin") != "synthetic_demo":
            raise MockTrainingFailed("Mock runner only accepts synthetic_demo inputs")
        if self.plan.get("result") == "fail":
            raise MockTrainingFailed(self.plan.get("reason") or "scripted mock failure")
        candidate = read_json(self.base / self.plan["candidate"])
        if candidate.get("data_origin") != "synthetic_demo":
            raise MockTrainingFailed("Mock candidate must be synthetic_demo data")
        return dict(
            candidate=candidate,
            candidate_file=self.plan["candidate"],
            log=[
                "mock runner: no data loaded, no model trained, no checkpoint written",
                f"mock runner: returned virtual candidate {candidate.get('model_version')}",
            ],
        )


def _step(name, *, mock, real=False, **detail):
    return dict(
        step=name,
        at=datetime.now().isoformat(timespec="seconds"),
        mock_executed=mock,
        real_executed=real,
        **detail,
    )


def _decision_view(policy, d):
    return dict(
        action=d["action"],
        reasons=d["reasons"],
        evidence=d["evidence"],
        policy_version=(policy or {}).get("version"),
        policy_status=(policy or {}).get("status"),
        training_started=d["training_started"],
        active_model_changed=d["active_model_changed"],
    )


def run_scenario(scenario, base=INPUTS, trainer=None):
    base = Path(base)
    thresholds = load_thresholds(base / scenario["thresholds"])
    current = read_json(base / scenario["current"])
    trainer = trainer or MockTrainer(scenario.get("mock_training", {}), base)
    steps = []
    record = dict(
        id=scenario["id"],
        label=scenario["label"],
        current_file=scenario["current"],
        thresholds_file=scenario["thresholds"],
        current_model=current.get("model_version"),
        candidate_model=None,
        mock_training_executed=False,
        real_training_executed=False,
        real_model_replaced=False,
        steps=steps,
    )

    # 1. evaluate the current model with the existing harness
    policy, first = judge(current, thresholds)
    steps.append(_step("evaluate_current", mock=False, decision=_decision_view(policy, first)))

    # 2. retraining request only when the existing decision says so
    if first["action"] != "RETRAIN_REQUEST":
        status = {"KEEP_CURRENT": "NO_RETRAIN_NEEDED", "HOLD": "NOT_STARTED_HOLD"}.get(
            first["action"], "NOT_STARTED_OTHER"
        )
        steps.append(
            _step(
                "retrain_request",
                mock=False,
                requested=False,
                reason=f"decision was {first['action']}, so no training runner is called",
            )
        )
        return dict(record, loop_status=status, final_reason=first["reasons"])
    steps.append(_step("retrain_request", mock=False, requested=True, reason=first["reasons"]))

    # 3. mock training runner
    request = dict(
        decision=first["action"],
        model=current.get("model_version"),
        data_origin=current.get("data_origin"),
        failed_checks=first["evidence"].get("failed_checks"),
    )
    try:
        result = trainer.run(request)
    except MockTrainingFailed as exc:
        steps.append(_step("mock_training", mock=True, outcome="failed", reason=str(exc)))
        return dict(
            record,
            mock_training_executed=True,
            loop_status="STOPPED_TRAINING_FAILED",
            final_reason=[str(exc)],
        )
    candidate = result["candidate"]
    steps.append(
        _step(
            "mock_training",
            mock=True,
            outcome="candidate_returned",
            candidate_file=result["candidate_file"],
            log=result["log"],
        )
    )
    record = dict(
        record, mock_training_executed=True, candidate_model=candidate.get("model_version")
    )

    # 4-5. evaluate candidate against the current model with the same harness
    policy, second = judge(current, thresholds, candidate)
    steps.append(
        _step("evaluate_candidate_and_compare", mock=False, decision=_decision_view(policy, second))
    )
    if second["action"] == "APPROVE_CANDIDATE":
        status = "RECOMMEND_REPLACEMENT"
    elif second["action"] == "KEEP_CURRENT":
        status = "KEEP_CURRENT_AFTER_CANDIDATE"
    else:
        status = "STOPPED_CANDIDATE_EVALUATION"

    # 6. recommendation only; the active model is never switched here
    steps.append(
        _step(
            "replacement",
            mock=False,
            recommended=status == "RECOMMEND_REPLACEMENT",
            executed=False,
            reason="recommendation only; loading, smoke test and rollout are "
            "separate steps outside this mock loop",
        )
    )
    return dict(record, loop_status=status, final_reason=second["reasons"])


def text_report(results):
    lines = ["모의 재학습 루프 결과 (모든 학습은 모의 실행, 실제 학습·실제 교체 없음)", ""]
    for r in results:
        lines.append(f"== {r['label']}")
        lines.append(f"   결과: {r['loop_status']} ({LOOP_STATUS_KO[r['loop_status']]})")
        for s in r["steps"]:
            d = s.get("decision")
            extra = (
                f"{d['action']} (기준 {d['policy_version']}, status={d['policy_status']})"
                if d
                else s.get("outcome")
                or (
                    "요청함"
                    if s.get("requested")
                    else "권고"
                    if s.get("recommended")
                    else "요청 안 함"
                    if s.get("requested") is False
                    else "권고 안 함"
                )
            )
            lines.append(
                f"   - {s['step']}: {extra} | "
                f"모의 실행 {s['mock_executed']} / 실제 실행 {s['real_executed']}"
            )
        lines.append(f"   이유: {' / '.join(map(str, r['final_reason']))}")
        lines.append(
            f"   모의 학습 실행 {r['mock_training_executed']} · "
            f"실제 학습 실행 {r['real_training_executed']} "
            f"· 실제 모델 교체 {r['real_model_replaced']}"
        )
        lines.append("")
    return "\n".join(lines)


def run(scenarios_path=INPUTS / "scenarios.json", output_root=ROOT / "mock_loop_results"):
    spec = read_json(scenarios_path)
    base = Path(scenarios_path).parent
    results = [dict(run_scenario(s, base), expected=s.get("expected")) for s in spec["scenarios"]]
    for r in results:
        if (
            r["real_training_executed"]
            or r["real_model_replaced"]
            or any(s["real_executed"] for s in r["steps"])
        ):
            raise AssertionError("The mock loop must never execute real training or replacement.")
    out = Path(output_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    (out / "scenarios").mkdir()
    for r in results:
        (out / "scenarios" / f"{r['id']}.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    summary = dict(
        notice="모의 재학습 루프. mock_executed는 가상 실행, real_executed는 실제 실행 여부입니다.",
        scenarios_file=str(scenarios_path),
        results=[
            {
                k: r[k]
                for k in (
                    "id",
                    "label",
                    "loop_status",
                    "expected",
                    "current_model",
                    "candidate_model",
                    "mock_training_executed",
                    "real_training_executed",
                    "real_model_replaced",
                )
            }
            for r in results
        ],
        real_training_executed=False,
        real_model_replaced=False,
    )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    text = text_report(results)
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    return out, results, text


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scenarios", default=str(INPUTS / "scenarios.json"))
    p.add_argument("--output-root", default=str(ROOT / "mock_loop_results"))
    args = p.parse_args()
    out, results, text = run(Path(args.scenarios), Path(args.output_root))
    # cp949 등 UTF-8이 아닌 콘솔에서 출력 문자(예: —) 때문에 중단되지 않게 한다.
    # 파일 저장은 UTF-8 그대로다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print(text)
    mismatched = [r for r in results if r["expected"] and r["expected"] != r["loop_status"]]
    for r in mismatched:
        print(f"예상과 다름: {r['id']} expected={r['expected']} observed={r['loop_status']}")
    print(f"완료: {len(results)}개 사례 / 결과 폴더: {out}")
    if mismatched:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
