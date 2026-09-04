# Physics-guided-VM

`main` 브랜치는 SUM과 CWSA 모델을 모두 선택해서 실행할 수 있다.

- SUM: `python train.py --aggregation sum`
- CWSA: `python train.py --aggregation cwsa`

기본 aggregation은 `sum`이다. 별도 경로를 지정하지 않으면 결과는 선택한
방식에 따라 `checkpoints/sum` 또는 `checkpoints/cwsa`에 저장된다.

고정된 모델 버전은 각각 `model-sum`, `model-cwsa` 브랜치에서 관리한다.
