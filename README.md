# Physics-guided-VM

이 브랜치는 논문 원본의 **SUM pair aggregation** 모델을 관리한다.

- encoder aggregation: `sum`
- physics pair reduction 기본값: `sum`
- KL weight 기본값: `beta`
- checkpoint 기본 경로: `checkpoints/sum`

`train.py`의 `--aggregation`은 이 브랜치에서 `sum`으로 고정된다.
