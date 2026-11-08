# 독립 하네스 프로토타입

상위 저장소의 프로젝트 규칙을 따른다. README.md를 먼저 읽는다.
이 폴더에서 `python -m unittest -v test_controller.py test_inspect_results.py test_harness.py`를 실행한다.
시연은 `python harness.py --demo`로 실행하고 예상값 대신 실제 판정과 종료 코드를 보고한다.
가상 평가값과 임계치를 실제 모델의 검증 결과나 공식 기준으로 표현하지 않는다.
기존 파이프라인의 스키마, 게이트, 레지스트리는 이 프로토타입과 아직 연결되지 않았다.
원본 평가 자료와 과거 실행 결과를 수정하지 않는다. 모델 학습과 교체를 실행하지 않는다.
