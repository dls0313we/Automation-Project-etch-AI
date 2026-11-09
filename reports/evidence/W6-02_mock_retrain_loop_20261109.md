# W6-02: 모의 재학습 루프 기록

[확인] 2026-11-09 로컬 Windows에서 실행했다. 모든 입력은 `mock_loop_inputs/`의 가상 자료(`synthetic_demo`)이며
실제 FNO/MGN 결과, `thresholds_*_provisional.json`, 기존 판정 규칙(`controller.py`, `harness.py`)은 변경하지 않았다.

## 테스트·린트

- `cd prototypes/threshold_controller` 후 Python 3.11.9와 3.14.6에서 10개 테스트 파일 실행: 158 tests, OK.
  (기존 147 + 모의 재학습 루프 11)
- 저장소 루트 `ruff check .`(0.17.3): 통과. `prototypes/threshold_controller`는 `ruff format --check` 통과.
- `local_paths.json`이 없는 상태(저장소 반영본)에서 실행했다.

## 시연 (`python mock_retrain_loop.py`, 종료 코드 0)

| 사례 | 루프 결과 | 모의 학습 실행 | 실제 학습 실행 | 실제 모델 교체 |
|---|---|---|---|---|
| 1_meets_criteria | NO_RETRAIN_NEEDED | False | False | False |
| 2_retrain_improves | RECOMMEND_REPLACEMENT | True | False | False |
| 3_candidate_worse | KEEP_CURRENT_AFTER_CANDIDATE | True | False | False |
| 4a_candidate_metric_missing | STOPPED_CANDIDATE_EVALUATION | True | False | False |
| 4b_mock_training_failed | STOPPED_TRAINING_FAILED | True | False | False |
| 5a_provisional_thresholds | NOT_STARTED_HOLD | False | False | False |
| 5b_reference_unverified | NOT_STARTED_HOLD | False | False | False |

- 7개 사례 모두 `scenarios.json`의 기대 결과와 일치했다. 결과 폴더(`mock_loop_results/`)는 Git에서 제외한다.
- 사례 2의 교체는 권고(`recommended=true`)만 기록하며 실행(`executed=false`)하지 않는다.
- 출력을 파일로 넘기는 등 cp949 콘솔에서 `—` 출력 시 `UnicodeEncodeError`로 종료 코드 1이 나던 문제를
  `sys.stdout.reconfigure(errors="replace")`로 고친 뒤 위 결과를 다시 얻었다. 결과 파일은 계속 UTF-8로 저장된다.

## 한계

- [제안] 모의 사례의 수치·기준은 흐름 시연용 가상 값이며 실제 MGN/FNO 성능이나 공식 기준이 아니다.
- [미확인] 실제 학습 실행기 연결, 실제 후보 평가, 실제 모델 교체는 구현·실행하지 않았다. 실제 자료의 판정은 HOLD로 유지된다.
