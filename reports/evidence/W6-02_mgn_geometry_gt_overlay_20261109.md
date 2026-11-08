# W6-02: MGN 선분 형상 검사와 정답 형상 비교 기록

[확인] 2026-11-09 로컬 Windows에서 실행했다. MGN 예측 자료와 정답 ZIP은 저장소에 포함하지 않는다.

## 테스트·린트

- `cd prototypes/threshold_controller` 후 Python 3.11.9에서 8개 테스트 파일 실행: 130 tests, OK.
  (기존 90 + 형상 검사 29 + 정답 비교 11)
- 저장소 루트 `ruff check .`(0.17.3): 통과. `prototypes/`는 `ruff format --check` 통과.
- 정답 비교 검사는 일부러 넣은 오류 7종을 잡았다: ZIP 저장 순서로 단계 대응, test 프레임 선택 허용,
  정답 검사 생략, reference_verified=true, 대응 실패 무시, 고정 크기 교차 표시, 교차가 있는 중복 연결을
  "중복 연결만"으로 분류.

## 정답 자료

- 출처: Hugging Face `code5312/etch-ai-automation-data`(dataset, 카드 표기 apache-2.0), 파일 `etch-raw-n400.zip`.
- revision `40fcc746ed9654d89a1d6d4aa3cc8c5d4c076868`(lastModified 2026-11-05T08:41:04Z),
  크기 42,061,104 bytes, SHA256 `4df39525896536a84e5337dfe196833c7497b6215865c0fc93c1bda87f64b011`
  (Hugging Face LFS oid와 일치). 압축을 풀지 않고 ZIP에서 직접 읽었다.
- VTP 구조: VTK XML PolyData, base64 appended, zlib, UInt32 header. `Points`(3성분, z=0),
  `Lines/connectivity`·`offsets`(2점 선분), `CellData/MaterialIds`(0=마스크, 10=Si, 선분 단위).

## 단계 대응 (`python mgn_gt_overlay.py`, 종료 코드 0)

- validation 40개 레시피 모두 통과: 스냅샷 `t000..tN`과 예측 단계 N 일치, `meta.json` `actual_k == N`,
  마지막 정답 점 수 = `summary_last_step.csv` `n_gt_points`(40/40), t000 깊이 0.0001 µm(초기 형상).
- 선택 사례에서 점 Chamfer를 t(k-1)·t(k)·t(k+1)과 재계산하면 t(k)가 약 10배 이상 작았다.
  재계산값과 제공값의 상대 차이는 0.08–3.8%로, 계산 세부가 완전히 같다고 보지 않는다.

## 사례 (validation만, 같은 `check_frame`·허용치 1e-8 µm)

| 사례 | 예측 | 정답 |
|---|---|---|
| 자기교차 one-step: r0323 t001 | 교차 5쌍 | 문제 0 |
| 자기교차 rollout: r0308 t006 | 교차 26쌍, 중복 연결 2 | 문제 0 |
| 중복 연결만: r0305 one-step t001 | 중복 연결 2 | 문제 0 |
| 문제 없음 참고: r0302 one-step t020 | 문제 0 | 문제 0 |

- 정답 validation t001–tN 전체 639개 프레임: 교차·서로 다른 선분 중첩·중복 연결·끝점 접촉 모두 0.
- r0305 중복 연결은 한 노드에만 이어진 노드(차수 2)가 같은 연결을 두 번 가진 형태였다(관찰 사실).

[제안] 그림과 표는 검출 위치를 보여 주는 진단 자료다. 예측 오류의 원인이나 재학습 필요성을 단정하지 않는다.
[미확인] 중복 연결의 생성 원인, 형상 이상을 공식 게이트에 반영할지 여부, 정답 자료의 독립 검증.
`reference_verified`는 false로 유지했다. 원본 파일·과거 결과·Chamfer 점수는 실행 전후 SHA256이 같았다.
