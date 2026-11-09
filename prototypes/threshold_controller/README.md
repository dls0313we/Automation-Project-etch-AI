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
python -m unittest -v test_controller.py test_inspect_results.py test_harness.py test_fno_adapter.py test_mgn_adapter.py test_run_review.py test_mgn_geometry.py test_mgn_gt_overlay.py test_fno_gt_overlay.py test_mock_retrain_loop.py test_fno_sdf_compare.py test_fno_chamfer_compare.py
python harness.py --demo
python mock_retrain_loop.py
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
| `test_*.py` | 판정 15 + CSV 점검 11 + 하네스 14 + FNO 연결 21 + MGN 연결 17 + 통합 12 + 형상 검사 29 + MGN 정답 비교 11 + FNO 정답 비교 17 + 모의 재학습 루프 11 + FNO SDF 비교 14 + FNO Chamfer 재계산 16 = 188개 검사 |
| `mgn_geometry.py` | 선택형 MGN 선분 교차·중첩·중복 연결 검사 |
| `mgn_gt_overlay.py` | validation 사례의 예측·정답 형상 비교 보고서 |
| `fno_gt_overlay.py` | FNO 샘플 SDF 경계와 정답 형상 비교 보고서 |
| `fno_sdf_compare.py` | 받은 정답 SDF와 샘플 깊이·CD를 제공 CSV와 수치 비교 |
| `fno_chamfer_compare.py` | 같은 샘플의 Chamfer 독립 재계산·제공 CSV 비교(별도 진단, 동일 재현 아님) |
| `inspect_results.py` | 기존 6epoch MGN 평가 CSV의 데이터/집계 점검 |
| `examples/`, `run_demo.py` | 기존 판정기 사례 시연 |
| `mock_retrain_loop.py`, `mock_loop_inputs/` | 모의 재학습 루프와 그 전용 가상 입력·시연용 기준 |

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
  정답 자료 대조는 확인하지 않았다.
- [확인] 2026-11-09 민서 답변으로 FNO 정답 경계와 CD 정의를 확인했다([FNO 평가 정의](FNO_평가정의_확인_20261109.md)).
  정답은 128×256 SDF의 φ=0 경계(내부 마스크–실리콘 경계 제외)이고, CD는 두 형상 모두
  정답 깊이의 10/50/90% 높이에서 x=0을 포함하는 공기 구간 폭이다. `fno_adapter_results/`에
  `provider_definition.json`으로 기록한다. 제공자 확인이며 독립 점수 재현은 아니다.
  기존 계산·임계치를 바꾸지 않았으므로 `evaluation_contract_id`는 유지한다.
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
- 제공된 샘플 3개 레시피만 확인하며 전체 validation 판정이 아니다. CD 측정 높이는 정답 깊이의
  10/50/90%로 제공자가 확인했다. 이 VTP 비교에서는 재계산하지 않으며 받은 정답 SDF와의 수치 비교는
  아래 `fno_sdf_compare.py`로 따로 한다.
- 이 그림의 정답은 VTP이고 제공 평가의 정답은 SDF φ=0 경계다. 깊이 진단도 추출 꼭짓점의 최저점을
  사용하므로 제공자의 0.1 µm 재표본화 평가를 재현했다고 해석하지 않는다. `reference_verified`는 `false`, 형상 이상 건수는 계속 미측정이다.

## 받은 FNO 정답 SDF와 깊이·CD 수치 비교 (`fno_sdf_compare.py`, 2026-11-09)

```text
python fno_sdf_compare.py
```

`local_paths.json`의 `fno_gt_sdf`를 사용합니다. 별도 경로는 `--truth <폴더>`로 지정합니다.
압축 해제 후 `sdf/sdf/`로 중첩된 폴더도 처리합니다. 표준 라이브러리만 사용하며 원본을 수정하지 않습니다.

- 받은 파일 목록은 400개(이름 기준 train 300 / validation 40 / test 40 / OOD 20), 약 708 MB입니다.
  전체 파일의 물리 정확도를 검사한 뜻은 아닙니다.
- 실제 비교는 예측 샘플이 있는 validation 3개 레시피, 두 방식, 98개 프레임입니다.
- 정답 `t`는 t000 포함, 예측 `step`은 t001부터입니다. 명시된 번호로 대응하며 좌표 배열과 레시피 7개 조건이 같아야 합니다.
- 깊이는 선형 보간한 φ=0 경계 꼭짓점의 −min(y)로 계산합니다. 제공자의 0.1 µm 재표본화 코드와 동일하다고 주장하지 않습니다.
- CD는 정답 깊이 d의 10/50/90% 높이에서 두 SDF를 측정합니다. y 방향 선형 보간 후 x=0을 포함하는 φ>0 구간의 양끝 φ=0을 선형 보간합니다.
- 예측 중심이 고체면 폭 0, 정답 폭이 양수일 때 −100%입니다. 정답 폭 0이나 격자 밖이면 미측정으로 남깁니다.
- 깊이 1e-6 µm, 오차율 1e-4 퍼센트포인트는 **수치 재현 비교용 허용차**입니다. 모델 합격 기준·재학습 임계치가 아닙니다.
- 2026-11-09 실제 실행: 576개 숫자 일치, 빈칸 12개는 양쪽 미측정. 최대 차이는 깊이 약 1.12e-8 µm, 오차율 약 1.02e-6 퍼센트포인트입니다.
- Chamfer·IoU 및 전체 40개 validation 예측은 재계산하지 않았습니다. `reference_verified=false`와 기존 HOLD를 유지합니다.

결과는 `fno_sdf_compare_results/실행시각/`에 저장합니다.
`report.html`은 같은 SDF 표현의 정답·예측 비교 그림과 마지막 단계 수치,
`comparisons.csv`는 모든 단계의 제공값·재계산·차이,
`summary.json`은 대응 근거·일치 현황·누락·범위,
`per_step_diagnostics.json`은 CD 높이·폭·미측정 이유,
`source_files.json`은 읽은 원본의 SHA256입니다. 기존 결과는 덮어쓰지 않습니다.

추가 패키지 없이 실행합니다. 알고리즘 해석은
[scikit-image의 경계 추출 설명](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.find_contours)을 참고했고,
scikit-image를 실행 의존성으로 추가하지 않았습니다. NumPy는 별도 검증 도구에서만 사용했습니다.

실행 증거는 [검증 기록](../../reports/evidence/W6-02_fno_sdf_comparison_20261109.md)에 있다.

## 같은 샘플의 Chamfer 독립 재계산 (`fno_chamfer_compare.py`, 2026-11-09)

```text
python fno_chamfer_compare.py
```

입력은 `fno_sdf_compare.py`와 같습니다(`fno_samples`, `fno_gt_sdf`). 대응 검증(t000 포함 번호, 좌표, 7개 조건)도 같은 함수를 씁니다.
제공 점수·CSV·임계치·판정은 바꾸지 않고, 결과는 `fno_chamfer_compare_results/실행시각/`에 별도 진단으로 저장합니다. IoU는 제외합니다.

- 이 v1 방법을 정의할 당시 원본 평가 코드가 없었습니다. 이후 받은 노트북 재현은 아래 수신 기록을 참고하세요.
  근거는 샘플 README의 설명 한 줄입니다: "양쪽 윤곽선을 0.1 µm 간격으로 찍고, 각 점에서 상대 윤곽선까지 최단거리의 전체 평균".
  따라서 **민서 코드와 동일한 재현이 아닙니다**(`exact_provider_reproduction=false`).
- 사용한 방식(비교 전에 고정, 결과에 맞춰 바꾸지 않음):
  - 경계: 정답·예측 SDF의 φ=0, 기존 `marching_squares`(선형 보간, 안장점은 고체 연결). skimage 미사용.
  - 재표본화: 선분을 연결한 폴리라인마다 호 길이 0, 0.1, 0.2 … µm 위치 + 남은 길이가 있으면 끝점.
  - 거리: 각 점에서 상대 **폴리라인(선분)** 까지의 최단 유클리드 거리(상대 표본점까지가 아님).
  - 평균: 두 윤곽선의 모든 점을 합친 평균 (Σd(정답→예측)+Σd(예측→정답))/(N정답+N예측).
- 참고용 해석 3가지도 함께 기록합니다(일치 여부에 쓰지 않음): 두 방향 평균의 평균, 표본점 대 표본점,
  팀 `src/etchpipe/eval/metrics.py`의 두 방향 평균의 합.
- 수치 일치 허용차 1e-6 µm는 계산 재현 비교용이며 모델 기준이 아닙니다.
- 2026-11-09 실제 실행: 98개 프레임 모두 비교, **98개 모두 허용차 밖(불일치)**. |차이| 중앙값 0.0011 µm,
  최대 0.0232 µm(r0307 rollout t008: 제공 0.4089, 재계산 0.3856), 상대 차이 중앙값 7.3%, 최대 39.8%.
  one-step 49개는 모두 재계산이 작고(약 0.001 µm 일정한 차이), rollout은 위아래가 섞입니다.
  참고 해석은 모두 더 멀었습니다(표본점 대 표본점 중앙값 40.6%, 팀 합산식 181%).
- NumPy로 따로 작성한 재표본화·거리·평균 계산과 이 결과의 차이는 최대 4.4e-16 µm입니다(경계 추출만 공유).
  차이는 구현 오류보다 계산 방식 차이로 보이나 원인은 확인하지 않았습니다.
- 결과 파일: `report.html`(방식·요약·프레임별 표), `comparisons.csv`(프레임별 제공값·재계산·차이·점 개수·참고 해석),
  `summary.json`(방식·통계·범위), `source_files.json`(원본 SHA256). 모두 일치하지 않으면 종료 코드 1입니다.

실행 증거는 [검증 기록](../../reports/evidence/W6-02_fno_chamfer_comparison_20261109.md)에 있다.

## 원본 노트북 수신 후 Chamfer 확인 (2026-11-09)

`fno_b_colab.ipynb`의 metrics5 평가 셀(인덱스 37)을 직접 확인했습니다. 예전 셀의 합산식과 혼용하지 않습니다.

- 실제 표본화는 전체 호 길이가 아닌 각 원본 선분마다 `n=ceil(L/spacing)`로 등분합니다.
- 질의점 간격 기준은 0.1 µm, 거리 대상 경계의 표본점 간격 기준은 **0.01 µm**입니다.
- `cKDTree`가 상대 표본점까지 거리를 구하고, 양방향 거리 배열을 합쳐 평균합니다. v1의 정확한 선분 거리와 다릅니다.
- 검토한 원본 함수 `ev_contour_pts`, `ev_seg_dist`만 실행한 결과 **98개 전부 CSV와 수치 일치**, 최대 차이 약 4.98e-10 µm입니다. 노트북 전체나 학습·추론 셀은 실행하지 않았습니다.
- 따라서 Chamfer 계산 코드 확인 대기는 해소됐습니다. v1 진단·기존 결과는 보존하고 제공 점수는 수정하지 않습니다.
- 원본 노트북과 검증용 NumPy/SciPy/scikit-image는 기본 하네스 의존성에 추가하지 않았습니다. 개인 경로는 `local_paths.json`의 `fno_eval_notebook`에만 있습니다.

실행 증거는 [검증 기록](../../reports/evidence/W6-02_fno_provider_notebook_20261109.md)에 있습니다. IoU, 전체 40개 validation 예측 및 물리 참값은 검증하지 않았습니다. 임시 기준·공식 HOLD·`reference_verified=false`와 실제 학습·교체 미실행은 유지합니다.

## 모의 재학습 루프 (`mock_retrain_loop.py`, 2026-11-09)

기존 판정기(`harness.judge` → `controller.decide`)를 그대로 써서 재학습 루프 전체 흐름을 **가상으로** 시연합니다.

```text
python mock_retrain_loop.py
```

흐름: 기존 모델 평가 → (판정이 `RETRAIN_REQUEST`일 때만) 재학습 요청 → 모의 학습 실행기 호출 →
모의 후보 평가 → 기존 모델과 비교 → 교체 권고 또는 기존 모델 유지 → 전체 과정 기록.

| 사례 | 결과 |
|---|---|
| 1. 기준 만족 | `NO_RETRAIN_NEEDED` — 재학습 요청 안 함 |
| 2. 기준 초과 + 학습 데이터 준비, 개선 후보 | `RECOMMEND_REPLACEMENT` — 교체 권고(실제 교체 없음) |
| 3. 후보 성능 악화 | `KEEP_CURRENT_AFTER_CANDIDATE` — 기존 모델 유지 |
| 4a. 후보 평가값 누락 | `STOPPED_CANDIDATE_EVALUATION` — 중단, 이유 기록 |
| 4b. 모의 학습 실패 | `STOPPED_TRAINING_FAILED` — 중단, 이유 기록 |
| 5a. 기준이 임시(provisional) | `NOT_STARTED_HOLD` — 모의 학습 시작 안 함 |
| 5b. 정답 검증 미완료 | `NOT_STARTED_HOLD` — 모의 학습 시작 안 함 |

- 가상 평가값과 시연용 기준은 `mock_loop_inputs/`에 따로 있습니다(`scenarios.json`, `thresholds_mock*.json`,
  `evaluations/`). 실제 FNO/MGN 결과, `thresholds_*_provisional.json`, 기존 HOLD 규칙은 읽거나 바꾸지 않습니다.
- 모의 학습 실행기(`MockTrainer`)는 학습을 하지 않고 미리 적어 둔 가상 후보 평가 파일을 돌려줍니다.
  `synthetic_demo` 자료와 `RETRAIN_REQUEST` 판정일 때만 실행되며, 실제 자료처럼 보이면 거부합니다.
- 기록의 각 단계에 `mock_executed`(가상 실행)와 `real_executed`(실제 실행)를 따로 남깁니다.
  실제 학습·실제 모델 교체는 항상 `False`이며, 교체는 권고(`recommended`)만 하고 실행(`executed`)하지 않습니다.
- 결과: `mock_loop_results/실행시각/` — `report.txt`(읽기용), `summary.json`(사례별 결과),
  `scenarios/<사례>.json`(단계별 기록과 판정 원문). 예상 결과와 다르면 종료 코드 1입니다.
  실행 결과 폴더는 Git 추적에서 제외합니다.
- 실행 증거는 [검증 기록](../../reports/evidence/W6-02_mock_retrain_loop_20261109.md)에 있습니다.
- [제안] 모의 사례의 수치·기준은 흐름 시연용 가상 값이며 실제 MGN/FNO 성능이나 공식 기준이 아닙니다.

## 기존 파이프라인과 관계

이 프로토타입의 판정은 `src/etchpipe`의 회귀 게이트나 레지스트리와 아직 연결되지 않았다.
기존 metrics/gate 스키마, 종료 코드, 공식 임계치 설정은 변경하지 않았다.
W5-04/W6-02 관련 선행 시연이며 체크리스트 완료나 재학습 루프 완주를 뜻하지 않는다.
모의 재학습 루프는 가상 흐름 시연이며 실제 재학습 루프 완주가 아니다.
기존 설계와 임시 가정의 차이는 [DOC-HARNESS-001](../../docs/threshold-controller-prototype.md)에 기록한다.
