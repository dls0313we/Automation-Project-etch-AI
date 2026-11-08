# 하네스 임계치 판정기 시연

[확인] MGN/FNO 평가 점수를 입력받아 모델 유지, 재학습 요청, 데이터 재검토,
판단 보류, 후보 모델 교체 승인 권고를 기록하는 독립 프로토타입이다.
2026-11-08 로컬 개발 상태를 공유한다. 테스트/시연 실행 증거는
[검증 기록](../../reports/evidence/W6-02_harness_prototype_20261108.md)에 있다.

[제안] 지표와 임계치는 가상 입력을 위한 임시 값이다. 실제 MGN/FNO 합격 기준이 아니다.
[미확인] 실제 평가기 연결, 공식 기준 확정, 실제 재학습 실행, 모델 교체는 이후 작업이다.
현재 명령은 모델을 학습하거나 교체하지 않는다.

## 실행

Python 3.11 이상을 준비한다. 이 폴더의 실행에는 추가 패키지나 GPU가 필요 없다.
저장소 루트에서 다음을 실행한다.

```text
cd prototypes/threshold_controller
python -m unittest -v test_controller.py test_inspect_results.py test_harness.py test_fno_adapter.py test_mgn_adapter.py test_run_review.py test_mgn_geometry.py test_mgn_gt_overlay.py test_fno_gt_overlay.py
python harness.py --demo
```

가상 사례 5개는 차례대로 KEEP_CURRENT, RETRAIN_REQUEST, DATA_REVIEW, HOLD,
APPROVE_CANDIDATE를 만든다. 결과는 `harness_results/<실행시각>/`에 새로 저장한다.
`report.txt`는 읽기용 보고서, 사례별 JSON은 입력/기준/판정 원문,
`summary.json`은 판정 요약이다. 실행 결과 폴더는 Git 추적에서 제외한다.

## 파일 구성

| 위치 | 역할 |
|---|---|
| `controller.py` | 기존 판정 규칙 및 입력 검증 |
| `harness.py` | 모델/방식별 정책 선택, 파일 입력, 보고서 저장 |
| `thresholds_demo.json` | 기본 시연용 임계치 4종 |
| `thresholds_demo_relaxed.json` | MGN rollout 상한 완화 및 기준 버전 기록 시연 |
| `harness_inputs/evaluations/` | 실제 측정값이 아닌 가상 평가 결과 |
| `harness_inputs/scenarios.json` | 기본 시연 5개 |
| `test_*.py` | 판정 15 + CSV 점검 11 + 하네스 14 + FNO 연결 21 + MGN 연결 17 + 통합 12 + 형상 검사 29 + MGN 정답 비교 11 + FNO 정답 비교 17 = 147개 검사 |
| `mgn_geometry.py` | 선택형 MGN 선분 교차·중첩·중복 연결 검사 |
| `mgn_gt_overlay.py` | validation 사례의 예측·정답 형상 비교 보고서 |
| `fno_gt_overlay.py` | FNO 샘플 SDF 경계와 정답 형상 비교 보고서 |
| `inspect_results.py` | 기존 6epoch MGN 평가 CSV의 데이터/집계 점검 |
| `examples/`, `run_demo.py` | 기존 판정기 사례 시연 |

## 임계치 수정

`thresholds_demo.json`의 `policies`에서 `model_family`와 `eval_mode`에 해당하는
항목을 복사하여 새 기준 파일로 저장한다. 원래 기준과 과거 기록은 보존한다.

| 모델 / 방식 | 형상 오차 상한 (µm) | 깊이 오차 상한 (µm) | 형상 이상 상한 (건) |
|---|---:|---:|---:|
| MGN / onestep | 0.20 | 0.50 | 0 |
| MGN / rollout | 0.50 | 1.00 | 0 |
| FNO / onestep | 0.25 | 0.60 | 0 |
| FNO / rollout | 0.60 | 1.20 | 0 |

[제안] 모든 수치는 시연용이다. FNO의 실제 지표와 단위는 미확인이다.
주 지표의 후보 개선 하한은 5%, 다른 필수 지표의 허용 악화 비율은 0으로 가정한다.
상한과 같은 점수는 통과하고, 상한보다 큰 점수는 초과다.
데이터 점검 실패가 성능 기준 비교보다 우선한다.

같은 입력의 형상 오차 0.7에 대해 MGN rollout 상한 0.5에서는 재학습 요청,
완화 상한 0.8에서는 현재 모델 유지가 나오는지 아래처럼 비교할 수 있다.

```text
python harness.py --eval harness_inputs/evaluations/mgn_rollout_v1_over_limit.json
python harness.py --eval harness_inputs/evaluations/mgn_rollout_v1_over_limit.json --thresholds thresholds_demo_relaxed.json
```

기준 숫자만 바꾸면 정책 `version`을 갱신하고 `evaluation_contract_id`는 유지한다.
지표 계산/경계 정의/집계 방식이 바뀌면 계약 ID도 입력과 정책에서 함께 갱신한다.
계산 가정과 이후 수정 위치는 [임시 기준](임시기준_및_연결방법.md)을 참조한다.

## 실제 평가 결과 연결 예정 지점

평가 담당자가 계산한 점수를 `harness_inputs/evaluations/` 예시 형식으로 변환하여
`python harness.py --eval 평가결과.json --thresholds 기준파일.json`에 전달한다.
모델 버전, 평가 방식, 데이터 ID/split, 계산 계약 ID, 지표 단위와 데이터 점검 근거가 필요하다.
누락한 필수 점수는 0으로 채우지 않는다. 현재 demo 정책은 실제 자료를 성능 판정하지 않는다.

`data_check.reference_verified=true`나 `training_data_ready=true`는 가상 사례의 설정이다.
실제 자료에 사용할 때에는 확인된 근거에 맞춰 기록한다.
기준값을 느슨하게 바꾸는 시연은 모델 정확도가 개선되었다는 증거가 아니다.

## 기존 CSV 점검 도구

```text
python inspect_results.py --input /path/to/mgn_eval_results
```

외부에서 `recipe_conditions.csv`, `per_recipe.csv`, `per_step.csv`, `summary.csv`를
별도로 제공해야 한다. 이 저장소에는 원본 평가 CSV나 예측 배열을 포함하지 않는다.
검사 결과는 `inspection_results/<실행시각>/`에 저장된다.

`run_demo.py --case real`은 과거 6epoch 자료의 음수 에너지 5건을 요약한 고정 사례다.
현재 원본 CSV를 직접 읽는 명령은 `inspect_results.py`다.
기존 CSV와 특정 체크포인트의 대응 관계나 새 모델 성능을 독립적으로 입증하지 않는다.

## 실제 FNO·MGN 평가 점수 연결 (2026-11-09)

[확인] 평가 담당자가 계산해 제공한 점수를 **새로 계산하지 않고** 읽어 하네스 입력으로 바꾸고,
4개 조합(FNO/MGN × onestep/rollout)의 현재 상태를 보고서로 만든다.
실행 증거는 [검증 기록](../../reports/evidence/W6-02_fno_mgn_review_20261109.md)에 있다.
원본 평가 자료와 예측 배열은 이 저장소에 포함하지 않는다.

```text
python fno_adapter.py      # FNO 300런(main_s1) validation 점수
python mgn_adapter.py      # MGN mgn_main_s1_nov9 점수 (validation 판정, test 참고 기록)
python run_review.py       # 두 연결기를 함께 실행 → review_results/<실행시각>/report.html, summary.json
```

입력 위치는 이 폴더의 `local_paths.json`(Git 제외)에 적는다. 없으면 `inputs/` 아래 같은 이름을 찾는다.

```json
{"fno_metrics": "<경로>/metrics5_val.csv", "fno_samples": "<경로>/samples_main_s1",
 "mgn_input": "<경로>/mgn_main_s1_nov9"}
```

| 파일 | 역할 |
|---|---|
| `fno_adapter.py` | FNO 평가표에서 300런(main_s1)만 선택, onestep=단계 평균·rollout=마지막 단계를 샘플로 재확인 |
| `mgn_adapter.py` | MGN validation/test 분리, onestep=단계별 Chamfer 평균, rollout=마지막 단계 |
| `run_review.py` | 두 연결기를 그대로 호출해 `report.html`(정적 파일), `summary.json` 저장 |
| `thresholds_fno_provisional.json`, `thresholds_mgn_provisional.json` | 사람이 수정하는 **임시 비교용** 상한 |

- [확인] FNO와 MGN의 Chamfer 계산 방식이 달라 모델별로 별도 `evaluation_contract_id`를 쓴다.
  두 모델의 점수로 우열이나 순위를 매기지 않으며 보고서 순서는 FNO → MGN 고정이다.
- [확인] 깊이오차(%)는 부호를 보존하고 크기 비교에는 절댓값을 쓴다. %를 µm로 환산하지 않는다.
- [확인] 제공되지 않은 지표(MGN의 깊이 오차·IoU, 두 모델의 형상 이상 건수)는 미측정으로 두며 0으로 채우지 않는다.
- [확인] MGN 제공 설명의 "교차 있는 레시피 남아 있음, Chamfer만으로 합격 판단 금지"를 보고서 상단에 표시한다.
- [확인] MGN test 점수는 참고 기록만 하며 임시 기준 비교·임계치 조정·판정에 쓰지 않는다.
- [확인] 임시 기준은 `status: provisional`이라 최종 판정은 `HOLD`다. 기준을 승인해도
  `reference_verified=false`와 형상 이상 건수 미측정 때문에 보류가 남으며 결과에 함께 기록한다.
- [제안] 임시 상한 숫자는 근거 없는 시작값이다. 팀 합의 후 새 파일로 확정한다.
- [미확인] MGN validation은 rollout 간격 선택에 쓰여 독립 평가가 아니다. 자기교차 건수,
  FNO CD 높이 기준, 정답 자료 대조는 확인하지 않았다.
- 입력이 없거나 한쪽 연결기가 실패하면 해당 조합을 실패와 이유로 표시하고 종료 코드 1을 돌려준다.
  과거 결과를 최신 결과처럼 보여주지 않는다.

## MGN 선분 형상 검사와 정답 비교 (2026-11-09)

[확인] `mgn_geometry.py`는 MGN 예측 NPZ의 선분 그래프를 표준 라이브러리로 읽어(pickle 거부)
교차·서로 다른 선분의 중첩·중복 연결·구조 이상을 프레임 단위로 센다(`mgn-segment-geometry-v2`, 허용치 1e-8 µm).
`run_review.py --check-mgn-geometry` 또는 `mgn_adapter.py --check-geometry`로 선택 실행하며,
제공 Chamfer 점수는 바꾸지 않는다.

[확인] `mgn_gt_overlay.py`는 선택한 validation 사례의 예측과 정답 VTP를 같은 검사로 비교하고
겹친 그림이 있는 `gt_overlay_results/<실행시각>/report.html`을 만든다.

```text
python mgn_gt_overlay.py
```

- 정답: Hugging Face `code5312/etch-ai-automation-data`의 `etch-raw-n400.zip`
  (실행 기록 revision `40fcc746ed9654d89a1d6d4aa3cc8c5d4c076868`,
  SHA256 `4df39525896536a84e5337dfe196833c7497b6215865c0fc93c1bda87f64b011`).
  저장소에 포함하지 않으며 위치는 `local_paths.json`의 `mgn_gt_zip`에 적는다.
- 단계 대응은 파일명 번호(예측 k ↔ `t<k>.vtp`, t000=초기 형상)로만 하고, 스냅샷 개수·`actual_k`·
  마지막 정답 점 수(`n_gt_points`)·t000 깊이 중 하나라도 어긋나면 그림을 만들지 않고 중단한다.
- 사례는 validation에서만 고른다. 정답에도 같은 `check_frame`과 허용치를 적용한다.
- [제안] 보고서는 검출 사실만 보여 준다. 예측 오류의 원인이나 재학습 필요성을 단정하지 않으며
  `reference_verified`는 `false`로 유지한다.
- [미확인] 중복 연결이 형상 이상인지 출력 형식 문제인지는 MGN 담당 확인이 필요하다.

## FNO 샘플 정답 형상 비교 (2026-11-09)

[확인] `fno_gt_overlay.py`는 FNO 샘플(`samples_main_s1`)의 SDF `phi`에서 φ=0 경계를 추출해
같은 레시피·단계의 정답 VTP와 겹친 `fno_gt_overlay_results/<실행시각>/report.html`을 만든다.

```text
python fno_gt_overlay.py
```

- 입력: `local_paths.json`의 `fno_samples`와 `mgn_gt_zip`(정답 ZIP). 정답 ZIP 해시가 기록값과 다르면 중단한다.
- 단계 대응: `phi[index]`의 `step[index]=k` ↔ `t<k>.vtp`. 스냅샷 번호·`actual_k`와
  샘플 README의 레시피 파라미터 7개가 정답 `meta.json`과 모두 같아야 진행한다.
- 경계 추출: marching squares(선형 보간, 음수=고체, 안장점은 음수 상 연결). 이 추출은 구성상
  선 교차를 만들지 않으므로 자기교차 여부나 형상 합격을 판단하지 않는다.
- 제공된 샘플 3개 레시피만 확인하며 전체 validation 판정이 아니다. 바닥 CD는 측정 높이 기준이 미확정이라
  재계산하지 않는다. `reference_verified`는 `false`, 형상 이상 건수는 계속 미측정이다.

## 기존 파이프라인과 관계

이 프로토타입의 판정은 `src/etchpipe`의 회귀 게이트나 레지스트리와 아직 연결되지 않았다.
기존 metrics/gate 스키마, 종료 코드, 공식 임계치 설정은 변경하지 않았다.
W5-04/W6-02 관련 선행 시연이며 체크리스트 완료나 재학습 루프 완주를 뜻하지 않는다.
기존 설계와 임시 가정의 차이는 [DOC-HARNESS-001](../../docs/threshold-controller-prototype.md)에 기록한다.
