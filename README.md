# etch-pipeline

식각 공정 AI 대리 모델의 검증·재학습 자동화 파이프라인(데이터 생성 → 학습 → 평가 → 회귀 게이트 → 서빙 → 재학습). 정답 데이터는 ViennaPS 시뮬레이션이다. 팹에서 수집한 데이터는 쓰지 않는다.

상세 사양은 `docs/project-definition.md`(무엇을 왜), `docs/project-design.md`(컴포넌트·API), `docs/project-checklist.md`(주차별 작업)를 참조한다.

진행 중이며 설치·실행 가능한 전체 데모는 아직 없다. 진행 상황(몇 주차까지 끝났는지 포함)은 `docs/project-checklist.md`의 체크박스와 `reports/evidence/`를 기준으로 판단한다 — 이 README에 주차를 못박지 않는다(stale 방지).

## 라이선스

저장소 코드는 Apache-2.0 (`LICENSE`). ViennaPS(GPL-3.0)를 포함하는 시뮬레이션 이미지는 별도 조건이 있다 — `NOTICE` 참조.

## 임계치 판정기 선행 시연

[확인] [독립 실행 안내](prototypes/threshold_controller/README.md)에 평가 점수 입력,
모델/평가 방식별 임계치 비교, 판정 및 이유 저장 예제를 추가했다.
[제안] 입력과 임계치는 가상 값이다. 실제 평가기·재학습·모델 교체 연결은 미확인이다.
[로컬 검증 기록](reports/evidence/W6-02_harness_prototype_20261108.md)을 참조한다.
