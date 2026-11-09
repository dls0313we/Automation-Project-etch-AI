# 독립 하네스 프로토타입

상위 저장소의 프로젝트 규칙을 따른다. README.md를 먼저 읽는다.
이 폴더에서 `python -m unittest -v test_controller.py test_inspect_results.py test_harness.py test_fno_adapter.py test_mgn_adapter.py test_run_review.py test_mgn_geometry.py test_mgn_gt_overlay.py test_fno_gt_overlay.py test_mock_retrain_loop.py`를 실행한다(158개).
실제 점수 연결은 `python run_review.py`(입력 위치는 Git 제외 `local_paths.json`)로 실행한다.
FNO·MGN Chamfer 정의가 달라 우열·순위를 매기지 않는다. MGN test 점수는 참고 기록만 한다.
제공되지 않은 지표는 0으로 채우지 않는다.
정답 비교(`mgn_gt_overlay.py`)는 파일명 번호로만 단계를 대응시키고, 불명확하면 중단한다.
사례는 validation에서만 고르며 원인이나 재학습 필요성을 단정하지 않는다.
FNO 정답 비교(`fno_gt_overlay.py`)는 제공 샘플만 다루며, 경계 추출 방식상 자기교차를 판단하지 않는다. 임시 비교표와 최종 판정(HOLD)을 구분해 보고한다.
FNO 평가 정의는 `FNO_평가정의_확인_20261109.md`를 읽는다. 정답 SDF 경계와 정답 깊이 기준 CD 높이는
제공자가 확인했으나 독립 점수 재현은 아니다. VTP 비교 그림을 SDF 대 SDF 평가로 표현하지 않는다.
시연은 `python harness.py --demo`로 실행하고 예상값 대신 실제 판정과 종료 코드를 보고한다.
가상 평가값과 임계치를 실제 모델의 검증 결과나 공식 기준으로 표현하지 않는다.
기존 파이프라인의 스키마, 게이트, 레지스트리는 이 프로토타입과 아직 연결되지 않았다.
모의 재학습 루프는 `python mock_retrain_loop.py`로 실행하며 `mock_loop_inputs/`의 가상 자료만 쓴다.
모의 실행(mock_executed)과 실제 실행(real_executed)을 구분해 보고한다. 교체는 권고만 하고 실행하지 않는다.
원본 평가 자료와 과거 실행 결과를 수정하지 않는다. 모델 학습과 교체를 실행하지 않는다.
