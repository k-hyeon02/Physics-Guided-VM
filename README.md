# Physics-guided-VM

이 브랜치는 **CWSA (channel-wise softmax aggregation)** 모델을 관리한다.

- encoder aggregation: `cwsa`
- physics pair reduction 기본값: `mean`
- KL weight 기본값: `beta / K`
- checkpoint 기본 경로: `checkpoints/cwsa`

`train.py`의 `--aggregation`은 이 브랜치에서 `cwsa`로 고정된다.
