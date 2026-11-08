# W6-02: 형상 검사를 포함한 FNO·MGN 통합 현황 실행 기록

[확인] 2026-11-09 05:27 로컬 Windows에서 브랜치 커밋 `7667f52`와 같은 코드로 실행했다.
평가 자료와 결과 폴더는 저장소에 포함하지 않는다.

```text
cd prototypes/threshold_controller
python run_review.py --check-mgn-geometry
```

종료 코드 0. FNO·MGN 연결 모두 실행됨. 실제 학습 시작·모델 교체 모두 False.

## 4개 조합 (validation, 임시 기준 `status: provisional`)

| 조합 | 레시피 | 평균 점수 / 임시 상한 | 형상 이상 건수 (프레임) | 최종 판정 |
|---|---:|---|---|---|
| FNO onestep | 40 | Chamfer 0.0112 / 0.05 µm, \|깊이오차\| 0.264 / 1.0 % | 미측정 | HOLD |
| FNO rollout | 40 | Chamfer 0.1297 / 0.5 µm, \|깊이오차\| 1.063 / 5.0 % | 미측정 | HOLD |
| MGN onestep | 40 | Chamfer 0.0313 / 0.05 µm | 74 (임시 상한 0 초과, 31개 레시피) | HOLD |
| MGN rollout | 40 | Chamfer 0.3787 / 0.5 µm | 330 (임시 상한 0 초과, 34개 레시피) | HOLD |

## MGN 선분 형상 검사 (`mgn-segment-geometry-v2`, 허용치 1e-8 µm)

읽기 오류 0, 원본 파일 실행 전후 동일.

| 구분 | 검사 프레임 | 이상 프레임 | 교차 프레임 | 서로 다른 선분 중첩 | 중복 연결 프레임 | 이상 레시피 |
|---|---|---:|---:|---:|---:|---:|
| validation onestep | 639/639 | 74 | 68 | 0 | 6 | 31 |
| validation rollout | 639/639 | 330 | 267 | 0 | 85 | 34 |
| test onestep (참고) | 641/641 | 87 | 79 | 0 | 8 | 34 |
| test rollout (참고) | 641/641 | 451 | 356 | 0 | 123 | 39 |

- 이상 프레임은 교차·중복 연결 등이 있는 프레임을 한 번씩 센 수이며 선분 쌍 수가 아니다.
- test 결과는 참고 기록이며 판정과 임시 기준 조정에 쓰지 않았다.
- 같은 정의로 검사한 정답 validation 639개 프레임은 문제 0건이었다
  (`W6-02_mgn_geometry_gt_overlay_20261109.md`).

[제안] 임시 상한은 근거 없는 시작값이다. FNO와 MGN은 Chamfer 정의가 달라 우열을 매기지 않는다.
[미확인] 중복 연결을 형상 이상으로 셀지(MGN 담당 확인 필요), 예측 교차의 원인, FNO 형상 이상 측정 방법.
`reference_verified`는 false로 유지했다.
