# 데이터 구성

`Physics-Guided Variational Model for Unsupervised Sound Source Tracking`의 합성·평가 조건을 기준으로 한 데이터 파이프라인.

## 합성 데이터

- 단일 이동 음원
- 16 kHz, 20초
- 동일 LibriSpeech chapter의 발화 연결
- chapter 하나당 기본 샘플 하나
- 직선 이동과 축별 사인파 진동을 결합한 3차원 궤적
- 궤적당 RIR 156개
- 진동 횟수 0–2회
- 진동 진폭 0–1 m
- RT60 0.2–1.0초
- SNR 5–30 dB
- `gpuRIR.simulateTrajectory` 기반 시간 가변 RIR 합성
- 직접경로 전파 지연이 반영된 실제 활동 마스크

방 크기, 배열 위치 범위, 궤적당 RIR 수는 논문이 시뮬레이션 조건을 참고한 Neural-SRP 구현값 사용.

## 잡음 조건

`SimulationConfig.noise_mode`로 논문 Experiment 1과 기존 AGG-RL 합성을 구분.

### `awgn` (논문 Experiment 1)

- 기본값은 논문이 참조한 Neural-SRP 공개 시뮬레이터와 같은 `direct_path`
  power reference
- 직접경로 마이크 신호의 활성 음향 파워를 512-sample window/256-sample
  hop으로 계산
- 12개 채널의 활성 파워를 평균하여 목표 SNR 5–30 dB의 공통 noise scale 결정
- 마이크마다 독립적인 표준정규 AWGN 추가
- MS-SNSD 파일을 읽거나 잡음 파일 선택으로 RNG를 소비하지 않음
- 과거 auralized-power 실험 재현에는 `--awgn-power-reference auralized` 사용
- 어느 power reference를 선택해도 최종 입력은 전체 잔향 신호이며, 직접경로
  신호는 AWGN scale 계산에만 사용됨

### `mixed` (기존 AGG-RL)

- 방 안에서 배열 중심으로부터 2.5 m 이상 떨어진 고정 잡음원 배치
- MS-SNSD 신호의 RIR 합성을 통한 공간 상관 잡음 생성
- 마이크 채널별 독립 백색 잡음 생성
- 공간 상관 잡음과 백색 잡음의 비율을 -15–15 dB에서 표본추출
- 전체 잡음을 음성 기준 5–30 dB SNR로 조절
- 최종 혼합 신호의 peak 정규화

별도 잡음 유형 설정 없음. `SimulationConfig`의 `snr_db`, `noise_sir_db`, `noise_min_distance_m`으로 수치 범위만 조절.

## 배열 설정

현재 프로젝트의 점진 학습용 배열 프로필 유지.

| 프로필 | 배열 |
|---|---|
| `stage1` | 4 cm 정사면체, 4채널 |
| `stage2` | 무작위 배열, 4채널 |
| `stage3` | 무작위 배열, 4–12채널 |
| `nao12` | LOCATA NAO robot, 12채널 |

단계 전환은 `set_profile()`로 적용. 데이터 로더를 다시 만들지 않아도 배치 샘플러가 변경된 채널 배정을 매 에폭 다시 조회.

## 사용 예

```python
from data.dataset import SyntheticDOADataset, build_dataloader
from data.simulate import SimulationConfig

dataset = SyntheticDOADataset(
    librispeech_root="datasets/librispeech/LibriSpeech/train-clean-100",
    ms_snsd_root="datasets/ms-snsd/MS-SNSD/noise_train",
    profile="stage1",
    batch_size=16,
    simulation_config=SimulationConfig(),
)

loader = build_dataloader(
    dataset,
    batch_size=16,
    num_workers=4,
    shuffle=True,
)

for epoch in range(1, 301):
    dataset.set_epoch(epoch)
    if epoch <= 10:
        dataset.set_profile("stage1")
    elif epoch <= 20:
        dataset.set_profile("stage2")
    else:
        dataset.set_profile("stage3")

    for batch in loader:
        pass
```

`num_samples` 생략 시 데이터셋 길이는 LibriSpeech chapter 수와 동일. `train-clean-100` 기준 585개, `test-clean` 기준 87개.

## 정적 단일 음원 데이터

동적 데이터와 별도로 `StaticSyntheticDOADataset` 제공.

- 16 kHz, 4초
- 샘플당 단일 LibriSpeech 발화 구간
- 배열 중심 기준 거리 0.3–2.5 m의 음원 위치 하나
- 음원 RIR 하나
- 전체 시간축에서 동일한 DOA와 거리
- 동적 데이터와 동일한 MS-SNSD·백색 잡음 혼합
- 동일한 `stage1`·`stage2`·`stage3` 배열 점진 학습

```python
from data.dataset import build_dataloader
from data.static import StaticSimulationConfig, StaticSyntheticDOADataset

dataset = StaticSyntheticDOADataset(
    librispeech_root="datasets/librispeech/LibriSpeech/train-clean-100",
    ms_snsd_root="datasets/ms-snsd/MS-SNSD/noise_train",
    num_samples=28_800,
    profile="stage1",
    batch_size=16,
    simulation_config=StaticSimulationConfig(),
)

loader = build_dataloader(
    dataset,
    batch_size=16,
    num_workers=4,
    shuffle=True,
)
```

## 출력 형식

```text
input_audio        (C, 320000)       float32
vad                (1, 320000)       float32
mic_coordinate     (C, 3)            float32
spherical_position (1, 3, 320000)    float32
polar_position     (1, 3)            float32
source_trajectory  (1, 3, 320000)    float32
n_spk              ()                int64
room_size          (3,)              float32
rt60               ()                float32
snr_db             ()                float32
```

`spherical_position`의 좌표 순서는 `[방위각, +z축 기준 극각, 거리]`. `polar_position`은 기존 정적 입력 형식과의 호환을 위한 첫 시점 값.

## LOCATA 평가

```python
from data.locata import find_recordings, load_recording

for recording in find_recordings("datasets/locata/dev", "benchmark2"):
    sample = load_recording(recording, "benchmark2")
```

`find_recordings`의 기본 과제는 논문과 동일한 Task 1·3·5. `load_recording`은 각 오디오 시점에서 가장 가까운 배열 위치·자세와 음원 위치를 사용하여 시간별 DOA 계산.
