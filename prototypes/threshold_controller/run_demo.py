"""Run the five packaged controller cases and save the actual decisions."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from controller import decide

ROOT = Path(__file__).resolve().parent
CASE_NAMES = ["keep", "retrain", "data", "candidate", "real"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASE_NAMES], default="all")
    args = parser.parse_args()
    names = CASE_NAMES if args.case == "all" else [args.case]
    output = ROOT / "demo_results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    summary = []
    for name in names:
        case = json.loads((ROOT / "examples" / (name + ".json")).read_text(encoding="utf-8"))
        result = decide(case["current"], case["policy"], case.get("candidate"))
        if result["action"] != case["expected"]:
            raise AssertionError(
                f"{name}: expected {case['expected']}, observed {result['action']}"
            )
        if result["training_started"] or result["active_model_changed"]:
            raise AssertionError("This demo must only decide and record results.")
        (output / (name + ".json")).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(case["label"])
        print("판정:", case["headline"], "(" + result["action"] + ")")
        print("이유:", case["reason"])
        print(
            "학습 시작:", result["training_started"], "/ 모델 교체:", result["active_model_changed"]
        )
        print()
        summary.append(dict(case=name, label=case["label"], action=result["action"]))
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("완료:", len(summary), "개 사례")
    print("판정 기록:", output)


if __name__ == "__main__":
    main()
