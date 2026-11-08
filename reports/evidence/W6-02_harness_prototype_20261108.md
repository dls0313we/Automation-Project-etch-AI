# W6-02 / W5-04: 독립 하네스 프로토타입 공유 기록

[확인] 2026-11-08 로컬 Windows, Python 3.12.14에서 업로드용 복사본을 실행했다.

- `cd prototypes/threshold_controller` 후 `python -m unittest -v test_controller.py test_inspect_results.py test_harness.py`: 40 tests, OK.
- `python harness.py --demo`: 5개 예상 판정 일치, 종료 코드 0.
- `python run_demo.py`: 기존 사례 5개 예상 판정 일치, 종료 코드 0.
- 같은 MGN rollout 가상 입력을 기본 기준/완화 기준으로 각각 실행: RETRAIN_REQUEST → KEEP_CURRENT, 둘 다 종료 코드 0.
- 저장소 루트 `ruff check .`(0.17.3): 통과.
- `python scripts/check_forbidden_expressions.py`: 통과.
- 업로드 복사본의 인식 가능한 시크릿 패턴/로컬 절대 경로 검사: 발견 없음.

[제안] 입력과 기준값은 시연용이다. FNO의 실제 지표/단위는 미확인이다.
[미확인] 실제 평가기 연결, 공식 기준 확정, 재학습 실행과 모델 교체는 수행하지 않았다.
GitHub CI 상태는 이 로컬 기록과 별도로 확인해야 한다.
W5-04/W6-02 체크리스트 완료나 G5/G6 통과를 선언하지 않는다.

원본 로컬 작업 폴더는 보존하고, 복사본의 Python 코드만 저장소 린트 형식에 맞췄다.
복사 전 파일 해시는 `prototypes/threshold_controller/SOURCE_MANIFEST.json`,
실행 상세와 복사본 파일 해시는 [JSON 기록](W6-02_harness_prototype_20261108.json)에 있다.
기존 설계와 임시 규격의 차이는 `docs/threshold-controller-prototype.md`의 DOC-HARNESS-001에 기록했다.
