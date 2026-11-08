"""Decision-only retraining controller. Does not train or deploy a model."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path


class ContractError(ValueError):
    pass


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name}: numeric value required, not a missing value")
    if not math.isfinite(value) or value < 0:
        raise ContractError(f"{name}: finite nonnegative value required")
    if name.endswith("_count") and int(value) != value:
        raise ContractError(f"{name}: integer count required")
    return float(value)


def outcome(action, reasons, policy, current, candidate=None, **evidence):
    return dict(
        action=action,
        reasons=reasons,
        policy_version=policy.get("version"),
        policy_status=policy.get("status"),
        current_model=current.get("model_id"),
        candidate_model=candidate.get("model_id") if candidate else None,
        decided_at_utc=datetime.now(UTC).isoformat(),
        evidence=evidence,
        training_started=False,
        active_model_changed=False,
    )


def data_problem(payload):
    check = payload.get("data_check", {})
    if not isinstance(check, dict):
        return None
    if check.get("passed") is False:
        return "Data inspection failed; do not use model training to repair invalid inputs."
    if (
        isinstance(check.get("invalid_recipe_count"), (int, float))
        and check["invalid_recipe_count"] > 0
    ):
        return (
            "Invalid recipes were reported. "
            "Keep the original result and review/regenerate the data."
        )
    return None


def validate(payload, policy):
    for name in ["model_id", "evaluation_dataset_id", "evaluation_contract_id"]:
        if not isinstance(payload.get(name), str) or not payload[name].strip():
            raise ContractError(f"{name}: identifier required")
    if payload.get("split") not in ["validation", "monitor"]:
        raise ContractError(
            "Use validation/monitor results for controller design, "
            "not repeatedly tuned test results."
        )
    if payload.get("unit") != "um":
        raise ContractError("This prototype expects explicitly defined micrometre metrics.")
    # Check model family and evaluation mode when the policy declares them.
    for key in ["model_family", "eval_mode"]:
        if key in policy and payload.get(key) != policy[key]:
            raise ContractError(
                f"{key} mismatch: evaluation {payload.get(key)!r} vs policy {policy[key]!r}"
            )
    if payload["evaluation_contract_id"] != policy.get("evaluation_contract_id"):
        raise ContractError(
            "Metric protocol, aggregation or geometry definition differs from this policy."
        )
    if number(payload.get("recipe_count"), "recipe_count") < 1:
        raise ContractError("No evaluated recipes")
    check = payload.get("data_check", {})
    if not isinstance(check, dict):
        raise ContractError("data_check: JSON object required")
    if check.get("passed") is not True or check.get("reference_verified") is not True:
        raise ContractError("Data quality and the reference must have explicit passing evidence.")
    if number(check.get("invalid_recipe_count"), "invalid_recipe_count") != 0:
        raise ContractError("Invalid input recipes remain")
    required = policy.get("max_metrics")
    if not isinstance(required, dict) or not required:
        raise ContractError("No metric limits have been defined")
    if policy.get("primary_metric") not in required:
        raise ContractError("Primary metric must be included in the policy limits")
    if policy.get("status") == "demo" and payload.get("data_origin") != "synthetic_demo":
        raise ContractError("Demonstration limits must not judge actual model results.")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ContractError("metrics: JSON object required")
    units = policy.get("metric_units")
    if units is not None:
        got = payload.get("metric_units")
        if not isinstance(got, dict):
            raise ContractError("metric_units: JSON object required by this policy")
        for name in required:
            if got.get(name) != units.get(name):
                raise ContractError(
                    f"{name}: unit {got.get(name)!r} differs from policy unit {units.get(name)!r}"
                )
    values, failures = {}, []
    for name, limit in required.items():
        limit = number(limit, f"limit:{name}")
        value = number(metrics.get(name), name)
        values[name] = value
        if value > limit:
            failures.append(dict(metric=name, observed=value, maximum=limit))
    return values, failures


def decide(current, policy, candidate=None):
    """Produce an auditable request/recommendation; no execution side effects."""
    for label, payload in [("current", current), ("candidate", candidate)]:
        if payload is not None and (problem := data_problem(payload)):
            return outcome("DATA_REVIEW", [f"{label}: {problem}"], policy, current, candidate)
    if policy.get("status") not in ["approved", "demo"]:
        return outcome(
            "HOLD", ["Thresholds are not approved or calibrated."], policy, current, candidate
        )
    if not isinstance(policy.get("version"), str) or not policy["version"].strip():
        return outcome(
            "HOLD", ["A versioned threshold policy is required."], policy, current, candidate
        )
    try:
        values, failures = validate(current, policy)
        if candidate is None:
            if not failures:
                return outcome(
                    "KEEP_CURRENT",
                    ["All supplied required metrics pass this policy."],
                    policy,
                    current,
                    observed=values,
                )
            if current.get("training_data_ready") is not True:
                return outcome(
                    "WAIT_FOR_TRAINING_DATA",
                    ["Quality failed, but approved training inputs/labels are not ready."],
                    policy,
                    current,
                    failed_checks=failures,
                )
            return outcome(
                "RETRAIN_REQUEST",
                [
                    "Quality failed and approved training data is available; "
                    "request the training runner."
                ],
                policy,
                current,
                failed_checks=failures,
            )

        # A replacement decision is a separate gate from a training trigger.
        c_values, c_failures = validate(candidate, policy)
        if candidate["model_id"] == current["model_id"]:
            raise ContractError("A candidate must have a different model identity")
        if current["split"] != "validation" or candidate["split"] != "validation":
            raise ContractError("Candidate comparison requires the same fixed validation dataset.")
        for key in [
            "evaluation_dataset_id",
            "evaluation_contract_id",
            "recipe_count",
            "unit",
            "model_family",
            "eval_mode",
            "metric_units",
        ]:
            if current.get(key) != candidate.get(key):
                raise ContractError(f"Candidate comparison mismatch: {key}")
        if c_failures:
            return outcome(
                "KEEP_CURRENT",
                ["Candidate fails absolute quality limits."],
                policy,
                current,
                candidate,
                candidate_failed_checks=c_failures,
            )
        tolerance = number(policy.get("max_relative_regression"), "max_relative_regression")
        floor = number(policy.get("denominator_floor"), "denominator_floor")
        if floor == 0:
            raise ContractError("denominator_floor must be positive")
        regressions = [
            dict(metric=k, incumbent=v, candidate=c_values[k])
            for k, v in values.items()
            if c_values[k] > v + max(abs(v), floor) * tolerance
        ]
        if regressions:
            return outcome(
                "KEEP_CURRENT",
                ["Candidate regresses on a required metric."],
                policy,
                current,
                candidate,
                regressed_checks=regressions,
            )
        primary = policy["primary_metric"]
        gain = (values[primary] - c_values[primary]) / max(abs(values[primary]), floor)
        required_gain = number(
            policy.get("min_primary_improvement_relative"), "min_primary_improvement_relative"
        )
        if gain < required_gain:
            return outcome(
                "KEEP_CURRENT",
                ["Candidate improvement is below the required margin."],
                policy,
                current,
                candidate,
                primary_relative_improvement=gain,
            )
        return outcome(
            "APPROVE_CANDIDATE",
            [
                "Candidate passes absolute limits, non-regression "
                "and primary improvement requirements.",
                "Recommendation only. Loading, smoke tests, "
                "publication and rollback are separate steps.",
            ],
            policy,
            current,
            candidate,
            primary_relative_improvement=gain,
        )
    except ContractError as exc:
        return outcome("HOLD", [str(exc)], policy, current, candidate)


def read(path):
    with Path(path).open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = decide(
        read(args.metrics), read(args.policy), read(args.candidate) if args.candidate else None
    )
    body = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        out = Path(args.output)
        if out.exists():
            raise FileExistsError(
                "Use a new decision output path; previous decision logs are preserved."
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
