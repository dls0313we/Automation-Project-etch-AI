# W6-02 / W5-04: FNO·MGN 실제 점수 연결과 통합 보고서 기록

[확인] 2026-11-09 로컬 Windows에서 실행했다. 원본 평가 자료와 예측 배열은 저장소에 포함하지 않는다.

## 테스트·린트

- `cd prototypes/threshold_controller` 후 Python 3.11.9에서
  `python -m unittest -v test_controller.py test_inspect_results.py test_harness.py test_fno_adapter.py test_mgn_adapter.py test_run_review.py`:
  90 tests, OK. (15 + 11 + 14 + FNO 21 + MGN 17 + 통합 12)
- 저장소 루트 `ruff check .`(0.17.3): 통과. `prototypes/`는 `ruff format --check` 통과.
- `python scripts/check_forbidden_expressions.py`: 종료 코드 0. `prototypes/`, `.github/`에 금지 표현·로컬 절대 경로 없음(grep).
- 새 검사는 일부러 넣은 오류를 잡는지 확인했다: 절댓값 제거, 그룹 필터 제거, rollout을 평균으로 집계,
  빈칸 0 채움, reference_verified=true, MGN one-step 마지막 단계 사용, test로 판정, 형상 이상 0 채움,
  자기교차 안내 제거, HTML escape 제거, 점수순 정렬, 입력 누락 무시, 실패 시 종료 코드 0.

## 실제 자료 실행 (`python run_review.py`, 종료 코드 0)

입력: 민서 제공 `metrics5_val.csv` + `samples_main_s1`, 진주 제공 `mgn_main_s1_nov9`(README 기준 체크포인트
`mgn_h128_e35_ep34.pt`, sha256_16 `9cf49bdd636d365c`). 실행 전후 원본 256개 파일 SHA256 동일.

| 조합 | 레시피 | 평균 점수 | 임시 상한 | 상한 초과 레시피 | 최종 판정 |
|---|---:|---|---|---|---|
| FNO onestep (validation) | 40 | Chamfer 0.0112 µm, \|깊이오차\| 0.264 % | 0.05 / 1.0 | 1 / 1 (r0307) | HOLD |
| FNO rollout (validation) | 40 | Chamfer 0.1297 µm, \|깊이오차\| 1.063 % | 0.5 / 5.0 | 0 / 0 | HOLD |
| MGN onestep (validation) | 40 | Chamfer 0.0313 µm | 0.05 | 7 | HOLD |
| MGN rollout (validation) | 40 | Chamfer 0.3787 µm | 0.5 | 6 | HOLD |

- 형상 이상 건수: 네 조합 모두 미측정(제공되지 않음, 0으로 채우지 않음).
- 보류 사유: 임시 기준 미승인(`status: provisional`). 승인해도 `reference_verified=false`, 형상 이상 건수 미측정이 남는다.
- FNO 데이터 검사 문제 0건, 샘플 3개 레시피 집계 확인 42/42 일치(onestep=단계 평균, rollout=마지막 단계).
- MGN 데이터 검사 문제 0건, `summary_last_step.csv`의 onestep/rollout 값이 마지막 단계와 80/80 일치.
  summary의 `onestep_chamfer`는 마지막 단계 값이라 one-step 점수(단계 평균)로 쓰지 않았다.
- MGN test(r0340-r0379)는 참고 기록만 했다: onestep 평균 0.0331 µm, rollout 평균 0.3985 µm.
- 실제 학습 시작·모델 교체: 모두 False.

[제안] 임시 상한 숫자는 근거 없는 시작값이며 FNO와 MGN 점수로 우열을 매기지 않는다(Chamfer 정의가 다름).
[미확인] MGN validation은 rollout 간격 선택에 쓰여 독립 평가가 아님, 자기교차 건수, FNO CD 높이 기준,
Chamfer의 평가 담당 정의(점-선분)와의 차이, 정답 자료 대조, 단일 시드.
W5-04/W6-02 체크리스트 완료나 G5/G6 통과를 선언하지 않는다. GitHub CI 결과는 이 기록과 별도로 확인한다.
