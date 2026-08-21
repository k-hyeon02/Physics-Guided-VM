"""이동 단일 음원 학습 데이터의 음향 시뮬레이션."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

FS = 16_000
AUDIO_LEN = 20 * FS
N_SPK = 1


@dataclass(frozen=True)
class SimulationConfig:
    """논문의 공통 시뮬레이션 조건."""

    sample_rate: int = FS
    segment_seconds: float = 20.0
    max_speakers: int = N_SPK
    snr_db: tuple[float, float] = (5.0, 30.0)
    noise_sir_db: tuple[float, float] = (-15.0, 15.0)
    rt60_s: tuple[float, float] = (0.2, 1.0)
    room_size_min_m: tuple[float, float, float] = (3.0, 3.0, 2.5)
    room_size_max_m: tuple[float, float, float] = (10.0, 8.0, 6.0)
    wall_absorption_weight: tuple[float, float] = (0.5, 1.0)
    array_position_min_fraction: tuple[float, float, float] = (0.1, 0.1, 0.1)
    array_position_max_fraction: tuple[float, float, float] = (0.9, 0.9, 0.5)
    trajectory_points: int = 156
    oscillations: tuple[float, float] = (0.0, 2.0)
    oscillation_amplitude_m: tuple[float, float] = (0.0, 1.0)
    static_trajectory_probability: float = 0.0
    noise_min_distance_m: float = 2.5
    noise_wall_margin_m: float = 0.1
    microphone_position_std_m: float = 0.0
    rir_diffuse_attenuation_db: float = 12.0
    rir_end_attenuation_db: float = 40.0

    @property
    def segment_samples(self) -> int:
        """샘플당 오디오 샘플 수."""

        return int(self.sample_rate * self.segment_seconds)

    def validate(self) -> None:
        """시뮬레이션 설정 검증."""

        if self.sample_rate <= 0 or self.segment_seconds <= 0:
            raise ValueError("샘플링 주파수와 구간 길이는 양수여야 함")
        if self.max_speakers != N_SPK:
            raise ValueError(f"max_speakers는 {N_SPK}이어야 함")
        if self.trajectory_points < 2:
            raise ValueError("trajectory_points는 2 이상이어야 함")
        if not 0.0 <= self.static_trajectory_probability <= 1.0:
            raise ValueError("static_trajectory_probability는 0과 1 사이여야 함")
        if self.microphone_position_std_m < 0.0:
            raise ValueError("microphone_position_std_m은 음수가 될 수 없음")
        if self.noise_min_distance_m <= 0.0:
            raise ValueError("noise_min_distance_m은 양수여야 함")


def _trim_or_pad(signal: np.ndarray, length: int) -> np.ndarray:
    """신호의 고정 길이 변환."""

    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape[0] >= length:
        return signal[:length].copy()
    return np.pad(signal, (0, length - signal.shape[0])).astype(np.float32)


def _rms(signal: np.ndarray) -> float:
    """신호의 RMS 계산."""

    return float(np.sqrt(np.mean(np.square(signal), dtype=np.float64) + 1e-10))


def _scale_to_db(
    reference: np.ndarray,
    signal: np.ndarray,
    target_db: float,
) -> float:
    """기준 신호에 대한 목표 dB 비율의 배율 계산."""

    return _rms(reference) * 10.0 ** (target_db / 20.0) / (_rms(signal) + 1e-10)


def _normalize_peak(signal: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """최대 절댓값 기준 신호 정규화."""

    maximum = float(np.max(np.abs(signal))) + 1e-10
    return (signal * (peak / maximum)).astype(np.float32)


def _sample_room(
    rng: np.random.Generator,
    config: SimulationConfig,
) -> tuple[np.ndarray, float, np.ndarray]:
    """방 크기, 잔향 시간, 벽별 흡음 가중치 표본추출."""

    room_size = rng.uniform(
        np.asarray(config.room_size_min_m, dtype=np.float64),
        np.asarray(config.room_size_max_m, dtype=np.float64),
    )
    rt60 = float(rng.uniform(*config.rt60_s))
    absorption_weights = rng.uniform(
        config.wall_absorption_weight[0],
        config.wall_absorption_weight[1],
        size=6,
    )
    return room_size, rt60, absorption_weights


def _sample_array_center(
    room_size: np.ndarray,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> np.ndarray:
    """방 크기에 대한 비율 범위 내 배열 중심 표본추출."""

    low = np.asarray(config.array_position_min_fraction, dtype=np.float64)
    high = np.asarray(config.array_position_max_fraction, dtype=np.float64)
    return rng.uniform(low, high) * room_size


def _sample_trajectory(
    room_size: np.ndarray,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """직선 이동과 축별 사인파 진동을 결합한 3차원 궤적 생성."""

    start = rng.uniform(np.zeros(3), room_size)
    end = rng.uniform(np.zeros(3), room_size)
    clearance = np.min(
        np.stack((start, room_size - start, end, room_size - end)),
        axis=0,
    )

    amplitude_high = np.minimum(clearance, config.oscillation_amplitude_m[1])
    amplitude_low = np.minimum(amplitude_high, config.oscillation_amplitude_m[0])
    amplitude = rng.uniform(amplitude_low, amplitude_high)
    cycles = rng.uniform(*config.oscillations, size=3)
    angular_frequency = 2.0 * np.pi * cycles / config.trajectory_points

    trajectory_points = np.stack(
        [
            np.linspace(start[axis], end[axis], config.trajectory_points)
            for axis in range(3)
        ],
        axis=1,
    )
    trajectory_points += amplitude * np.sin(
        angular_frequency * np.arange(config.trajectory_points)[:, None]
    )

    if rng.random() < config.static_trajectory_probability:
        trajectory_points[:] = start

    duration = config.segment_samples / config.sample_rate
    timestamps = (
        np.arange(config.trajectory_points, dtype=np.float64)
        * duration
        / config.trajectory_points
    )
    return trajectory_points.astype(np.float64), timestamps


def _interpolate_trajectory(
    trajectory_points: np.ndarray,
    timestamps: np.ndarray,
    length: int,
    sample_rate: int,
) -> np.ndarray:
    """궤적 지점의 오디오 샘플 단위 선형 보간."""

    sample_times = np.arange(length, dtype=np.float64) / sample_rate
    return np.stack(
        [
            np.interp(sample_times, timestamps, trajectory_points[:, axis])
            for axis in range(3)
        ],
        axis=1,
    )


def _cartesian_to_spherical(relative_position: np.ndarray) -> np.ndarray:
    """상대 직교좌표의 방위각, 극각, 거리 변환."""

    x = relative_position[..., 0]
    y = relative_position[..., 1]
    z = relative_position[..., 2]
    horizontal = np.sqrt(x * x + y * y)
    distance = np.sqrt(horizontal * horizontal + z * z)
    azimuth = np.mod(np.degrees(np.arctan2(y, x)), 360.0)
    polar = np.degrees(np.arctan2(horizontal, z))
    return np.stack((azimuth, polar, distance), axis=-1).astype(np.float32)


def _rir_parameters(
    room_size: np.ndarray,
    rt60: float,
    absorption_weights: np.ndarray,
    config: SimulationConfig,
):
    """Sabine 모델 기반 gpuRIR 매개변수 계산."""

    import gpuRIR  # type: ignore

    beta = gpuRIR.beta_SabineEstimation(room_size, rt60, absorption_weights)
    t_diff = float(
        gpuRIR.att2t_SabineEstimator(config.rir_diffuse_attenuation_db, rt60)
    )
    t_max = float(gpuRIR.att2t_SabineEstimator(config.rir_end_attenuation_db, rt60))
    image_count = gpuRIR.t2n(t_diff, room_size)
    return beta, t_diff, t_max, image_count


def _render_trajectory_rirs(
    room_size: np.ndarray,
    source_positions: np.ndarray,
    microphone_positions: np.ndarray,
    beta: np.ndarray,
    t_diff: float,
    t_max: float,
    image_count: Sequence[int],
    sample_rate: int,
) -> np.ndarray:
    """GPU 메모리 사용량을 제한한 궤적 지점별 RIR 생성."""

    import gpuRIR  # type: ignore

    num_points = source_positions.shape[0]
    num_microphones = microphone_positions.shape[0]
    num_gpu_calls = min(
        max(
            1,
            int(
                np.ceil(
                    sample_rate
                    * t_diff
                    * num_microphones
                    * num_points
                    * np.prod(image_count)
                    / 1e9
                )
            ),
        ),
        num_points,
    )
    boundaries = np.ceil(
        num_points / num_gpu_calls * np.arange(num_gpu_calls + 1)
    ).astype(int)

    rir_batches = []
    for batch_index in range(num_gpu_calls):
        start, stop = boundaries[batch_index : batch_index + 2]
        rir_batches.append(
            gpuRIR.simulateRIR(
                room_sz=room_size,
                beta=beta,
                pos_src=source_positions[start:stop],
                pos_rcv=microphone_positions,
                nb_img=image_count,
                Tmax=t_max,
                fs=sample_rate,
                Tdiff=t_diff,
                spkr_pattern="omni",
                mic_pattern="omni",
            )
        )
    return np.concatenate(rir_batches, axis=0)


def _render_direct_path_rirs(
    room_size: np.ndarray,
    source_positions: np.ndarray,
    microphone_positions: np.ndarray,
    beta: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """신호 세기와 VAD 계산용 직접경로 RIR 생성."""

    import gpuRIR  # type: ignore

    return gpuRIR.simulateRIR(
        room_sz=room_size,
        beta=beta,
        pos_src=source_positions,
        pos_rcv=microphone_positions,
        nb_img=[1, 1, 1],
        Tmax=0.1,
        fs=sample_rate,
        spkr_pattern="omni",
        mic_pattern="omni",
    )


def _simulate_trajectory(
    signal: np.ndarray,
    rirs: np.ndarray,
    timestamps: np.ndarray | None,
    sample_rate: int,
    length: int,
) -> np.ndarray:
    """gpuRIR 이동 음원 합성 결과의 채널 우선 형식 변환."""

    import gpuRIR  # type: ignore

    rendered = gpuRIR.simulateTrajectory(
        np.asarray(signal, dtype=np.float32),
        np.asarray(rirs, dtype=np.float32),
        timestamps=timestamps,
        fs=sample_rate if timestamps is not None else None,
    )
    return rendered[:length].T.astype(np.float32, copy=False)


def _sample_noise_position(
    room_size: np.ndarray,
    array_center: np.ndarray,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> np.ndarray:
    """AGG-RL 공간 상관 잡음원의 고정 위치 표본추출."""

    margin = config.noise_wall_margin_m
    low = np.full(3, margin, dtype=np.float64)
    high = room_size - margin
    for _ in range(1024):
        position = rng.uniform(low, high)
        if np.linalg.norm(position - array_center) >= config.noise_min_distance_m:
            return position.astype(np.float64)

    corners = np.array(
        [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=np.float64,
    )
    distances = np.linalg.norm(corners - array_center[None], axis=1)
    farthest = int(np.argmax(distances))
    if distances[farthest] >= config.noise_min_distance_m:
        return corners[farthest]
    raise RuntimeError("AGG-RL 공간 상관 잡음원 위치 표본추출 실패")


def _mix_noise(
    microphone_signals: np.ndarray,
    coherent_noise: np.ndarray,
    room_size: np.ndarray,
    microphone_positions: np.ndarray,
    array_center: np.ndarray,
    beta: np.ndarray,
    t_diff: float,
    t_max: float,
    image_count: Sequence[int],
    snr_db: float,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> np.ndarray:
    """MS-SNSD 공간 상관 잡음과 채널별 백색 잡음의 AGG-RL 혼합."""

    noise_position = _sample_noise_position(
        room_size,
        array_center,
        rng,
        config,
    )
    coherent_rirs = _render_trajectory_rirs(
        room_size=room_size,
        source_positions=noise_position[None],
        microphone_positions=microphone_positions,
        beta=beta,
        t_diff=t_diff,
        t_max=t_max,
        image_count=image_count,
        sample_rate=config.sample_rate,
    )
    coherent_signals = _simulate_trajectory(
        _trim_or_pad(coherent_noise, config.segment_samples),
        coherent_rirs,
        timestamps=None,
        sample_rate=config.sample_rate,
        length=config.segment_samples,
    )
    white_noise = rng.standard_normal(microphone_signals.shape).astype(np.float32)

    noise_sir_db = float(rng.uniform(*config.noise_sir_db))
    white_scale = _scale_to_db(
        coherent_signals[0],
        white_noise[0],
        -noise_sir_db,
    )
    mixed_noise = coherent_signals + white_noise * white_scale
    noise_scale = _scale_to_db(
        microphone_signals[0],
        mixed_noise[0],
        -snr_db,
    )
    return _normalize_peak(microphone_signals + mixed_noise * noise_scale)


def _propagate_vad(
    vad: np.ndarray,
    direct_path_rirs: np.ndarray,
    timestamps: np.ndarray | None,
    sample_rate: int,
    length: int,
) -> np.ndarray:
    """직접경로 지연을 반영한 수신 신호 활동 마스크 생성."""

    propagated = _simulate_trajectory(
        vad,
        direct_path_rirs,
        timestamps,
        sample_rate,
        length,
    )
    mean_activity = propagated.mean(axis=0)
    threshold = float(np.max(propagated)) * 1e-3
    return (mean_activity > threshold).astype(np.float32)


def simulate_one_sample(
    speeches: Sequence[np.ndarray],
    coherent_noise: np.ndarray,
    microphone_coordinates: np.ndarray,
    rng: np.random.Generator,
    config: SimulationConfig | None = None,
    vads: Sequence[np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """단일 이동 음원의 다채널 학습 샘플 생성."""

    config = config or SimulationConfig()
    config.validate()
    if len(speeches) != N_SPK:
        raise ValueError(f"음원 신호는 {N_SPK}개여야 함")
    if vads is not None and len(vads) != N_SPK:
        raise ValueError(f"VAD는 {N_SPK}개여야 함")

    length = config.segment_samples
    speech = _trim_or_pad(speeches[0], length)
    source_vad = (
        _trim_or_pad(vads[0], length)
        if vads is not None
        else np.ones(length, dtype=np.float32)
    )

    room_size, rt60, absorption_weights = _sample_room(rng, config)
    array_center = _sample_array_center(room_size, rng, config)
    microphone_coordinates = np.asarray(microphone_coordinates, dtype=np.float64)
    if config.microphone_position_std_m > 0.0:
        microphone_coordinates = microphone_coordinates + rng.normal(
            0.0,
            config.microphone_position_std_m,
            size=microphone_coordinates.shape,
        )
        microphone_coordinates -= microphone_coordinates.mean(axis=0, keepdims=True)
    microphone_positions = array_center[None] + microphone_coordinates
    if np.any(microphone_positions <= 0.0) or np.any(microphone_positions >= room_size):
        raise RuntimeError("마이크가 방 경계 밖에 배치됨")

    trajectory_points, timestamps = _sample_trajectory(room_size, rng, config)
    trajectory = _interpolate_trajectory(
        trajectory_points,
        timestamps,
        length,
        config.sample_rate,
    )
    spherical = _cartesian_to_spherical(trajectory - array_center[None])

    beta, t_diff, t_max, image_count = _rir_parameters(
        room_size,
        rt60,
        absorption_weights,
        config,
    )
    trajectory_rirs = _render_trajectory_rirs(
        room_size=room_size,
        source_positions=trajectory_points,
        microphone_positions=microphone_positions,
        beta=beta,
        t_diff=t_diff,
        t_max=t_max,
        image_count=image_count,
        sample_rate=config.sample_rate,
    )
    microphone_signals = _simulate_trajectory(
        speech,
        trajectory_rirs,
        timestamps,
        config.sample_rate,
        length,
    )

    direct_path_rirs = _render_direct_path_rirs(
        room_size,
        trajectory_points,
        microphone_positions,
        beta,
        config.sample_rate,
    )
    snr_db = float(rng.uniform(*config.snr_db))
    mixture = _mix_noise(
        microphone_signals=microphone_signals,
        coherent_noise=coherent_noise,
        room_size=room_size,
        microphone_positions=microphone_positions,
        array_center=array_center,
        beta=beta,
        t_diff=t_diff,
        t_max=t_max,
        image_count=image_count,
        snr_db=snr_db,
        rng=rng,
        config=config,
    )

    propagated_vad = _propagate_vad(
        source_vad,
        direct_path_rirs,
        timestamps,
        config.sample_rate,
        length,
    )
    spherical_position = spherical.T[None].astype(np.float32)

    return {
        "input_audio": mixture.astype(np.float32),
        "vad": propagated_vad[None],
        "mic_coordinate": microphone_coordinates.astype(np.float32),
        "spherical_position": spherical_position,
        "polar_position": spherical_position[:, :, 0],
        "source_trajectory": trajectory.T[None].astype(np.float32),
        "n_spk": np.int64(N_SPK),
        "room_size": room_size.astype(np.float32),
        "rt60": np.float32(rt60),
        "snr_db": np.float32(snr_db),
    }
