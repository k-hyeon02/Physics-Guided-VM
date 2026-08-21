"""정적 단일 음원 합성 데이터셋."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .dataset import (
    SyntheticDOADataset,
    _clean_speech_and_vad,
    _crop_or_pad,
    _load_audio_mono,
    _resample_audio,
)
from .simulate import (
    FS,
    N_SPK,
    _mix_noise,
    _propagate_vad,
    _render_direct_path_rirs,
    _render_trajectory_rirs,
    _rir_parameters,
    _sample_array_center,
    _sample_room,
    _simulate_trajectory,
    _trim_or_pad,
)


@dataclass(frozen=True)
class StaticSimulationConfig:
    """정적 단일 음원 시뮬레이션 조건."""

    sample_rate: int = FS
    segment_seconds: float = 4.0
    max_speakers: int = N_SPK
    snr_db: tuple[float, float] = (5.0, 30.0)
    noise_sir_db: tuple[float, float] = (-15.0, 15.0)
    rt60_s: tuple[float, float] = (0.2, 1.0)
    room_size_min_m: tuple[float, float, float] = (3.0, 3.0, 2.5)
    room_size_max_m: tuple[float, float, float] = (10.0, 8.0, 6.0)
    wall_absorption_weight: tuple[float, float] = (0.5, 1.0)
    array_position_min_fraction: tuple[float, float, float] = (0.1, 0.1, 0.1)
    array_position_max_fraction: tuple[float, float, float] = (0.9, 0.9, 0.5)
    source_distance_m: tuple[float, float] = (0.3, 2.5)
    azimuth_deg: tuple[float, float] = (0.0, 360.0)
    polar_deg: tuple[float, float] = (0.0, 180.0)
    polar_vmf_mu_deg: float = 180.0
    polar_vmf_kappa: float = 2.0
    source_wall_margin_m: float = 0.1
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
        if self.source_distance_m[0] <= 0.0:
            raise ValueError("음원 거리는 양수여야 함")
        if self.source_distance_m[0] > self.source_distance_m[1]:
            raise ValueError("음원 거리 범위의 순서 오류")
        if self.source_wall_margin_m <= 0.0:
            raise ValueError("source_wall_margin_m은 양수여야 함")
        if self.noise_min_distance_m <= 0.0:
            raise ValueError("noise_min_distance_m은 양수여야 함")
        if self.microphone_position_std_m < 0.0:
            raise ValueError("microphone_position_std_m은 음수가 될 수 없음")


def _sample_polar_angle(
    rng: np.random.Generator,
    config: StaticSimulationConfig,
) -> float:
    """수평면 주변에 집중된 +z축 기준 극각 표본추출."""

    angle = rng.vonmises(
        np.radians(config.polar_vmf_mu_deg),
        config.polar_vmf_kappa,
    )
    angle = float(angle % (2.0 * np.pi))
    polar = np.degrees(angle / 2.0)
    return float(np.clip(polar, config.polar_deg[0], config.polar_deg[1]))


def _spherical_to_cartesian(
    distance: float,
    azimuth_deg: float,
    polar_deg: float,
) -> np.ndarray:
    """방위각·극각·거리의 직교좌표 변환."""

    azimuth = np.radians(azimuth_deg)
    polar = np.radians(polar_deg)
    return np.array(
        [
            distance * np.sin(polar) * np.cos(azimuth),
            distance * np.sin(polar) * np.sin(azimuth),
            distance * np.cos(polar),
        ],
        dtype=np.float64,
    )


def _sample_source_position(
    room_size: np.ndarray,
    array_center: np.ndarray,
    rng: np.random.Generator,
    config: StaticSimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """배열 중심 기준 정적 음원 위치 표본추출."""

    lower = np.full(3, config.source_wall_margin_m, dtype=np.float64)
    upper = room_size - config.source_wall_margin_m
    for _ in range(1024):
        distance = float(rng.uniform(*config.source_distance_m))
        azimuth = float(rng.uniform(*config.azimuth_deg))
        polar = _sample_polar_angle(rng, config)
        relative = _spherical_to_cartesian(distance, azimuth, polar)
        position = array_center + relative
        if np.all(position >= lower) and np.all(position <= upper):
            spherical = np.array([azimuth, polar, distance], dtype=np.float32)
            return position.astype(np.float64), spherical
    raise RuntimeError("정적 음원 위치 표본추출 실패")


def simulate_static_sample(
    speech: np.ndarray,
    coherent_noise: np.ndarray,
    microphone_coordinates: np.ndarray,
    rng: np.random.Generator,
    config: StaticSimulationConfig | None = None,
    vad: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """정적 단일 음원의 4초 다채널 학습 샘플 생성."""

    config = config or StaticSimulationConfig()
    config.validate()
    length = config.segment_samples
    speech = _trim_or_pad(speech, length)
    source_vad = (
        _trim_or_pad(vad, length)
        if vad is not None
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

    source_position, polar_position = _sample_source_position(
        room_size,
        array_center,
        rng,
        config,
    )
    beta, t_diff, t_max, image_count = _rir_parameters(
        room_size,
        rt60,
        absorption_weights,
        config,
    )
    source_rirs = _render_trajectory_rirs(
        room_size=room_size,
        source_positions=source_position[None],
        microphone_positions=microphone_positions,
        beta=beta,
        t_diff=t_diff,
        t_max=t_max,
        image_count=image_count,
        sample_rate=config.sample_rate,
    )
    microphone_signals = _simulate_trajectory(
        speech,
        source_rirs,
        timestamps=None,
        sample_rate=config.sample_rate,
        length=length,
    )
    direct_path_rirs = _render_direct_path_rirs(
        room_size,
        source_position[None],
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
        timestamps=None,
        sample_rate=config.sample_rate,
        length=length,
    )

    spherical_position = np.broadcast_to(
        polar_position[:, None],
        (3, length),
    ).copy()[None]
    source_trajectory = np.broadcast_to(
        source_position[:, None],
        (3, length),
    ).copy()[None]

    return {
        "input_audio": mixture.astype(np.float32),
        "vad": propagated_vad[None],
        "mic_coordinate": microphone_coordinates.astype(np.float32),
        "spherical_position": spherical_position.astype(np.float32),
        "polar_position": polar_position[None].astype(np.float32),
        "source_trajectory": source_trajectory.astype(np.float32),
        "n_spk": np.int64(N_SPK),
        "room_size": room_size.astype(np.float32),
        "rt60": np.float32(rt60),
        "snr_db": np.float32(snr_db),
    }


class StaticSyntheticDOADataset(SyntheticDOADataset):
    """LibriSpeech와 MS-SNSD 기반 정적 단일 음원 데이터셋."""

    def __init__(
        self,
        librispeech_root: str,
        ms_snsd_root: str,
        num_samples: int | None = None,
        profile: str = "stage1",
        batch_size: int = 16,
        seed: int = 0,
        simulation_config: StaticSimulationConfig | None = None,
        rotate_arrays: bool = True,
        channel_schedule: Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            librispeech_root=librispeech_root,
            ms_snsd_root=ms_snsd_root,
            num_samples=num_samples,
            profile=profile,
            batch_size=batch_size,
            seed=seed,
            simulation_config=simulation_config or StaticSimulationConfig(),
            rotate_arrays=rotate_arrays,
            channel_schedule=channel_schedule,
        )

    def _sample_utterance(
        self,
        path: str,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """단일 발화의 무작위 4초 구간과 활동 마스크 생성."""

        speech, source_rate = _load_audio_mono(path)
        speech = _resample_audio(
            speech,
            source_rate,
            self.simulation_config.sample_rate,
        )
        speech = _crop_or_pad(
            speech,
            self.simulation_config.segment_samples,
            rng,
        )
        return _clean_speech_and_vad(
            speech,
            self.simulation_config.sample_rate,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        seed = (
            self.seed * 1_000_003 + self._epoch * self.num_samples + int(index)
        ) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        speech_path = self.speech_files[int(rng.integers(0, len(self.speech_files)))]
        speech, source_vad = self._sample_utterance(speech_path, rng)
        noise_path = self.noise_files[int(rng.integers(0, len(self.noise_files)))]
        coherent_noise = self._sample_noise(noise_path, rng)
        microphone_coordinates = self._sample_mic_coordinates(
            int(self.channel_counts[index]),
            rng,
        )

        sample = simulate_static_sample(
            speech=speech,
            coherent_noise=coherent_noise,
            microphone_coordinates=microphone_coordinates,
            rng=rng,
            config=self.simulation_config,
            vad=source_vad,
        )
        return {
            "input_audio": torch.from_numpy(sample["input_audio"]),
            "vad": torch.from_numpy(sample["vad"]),
            "mic_coordinate": torch.from_numpy(sample["mic_coordinate"]),
            "spherical_position": torch.from_numpy(sample["spherical_position"]),
            "polar_position": torch.from_numpy(sample["polar_position"]),
            "source_trajectory": torch.from_numpy(sample["source_trajectory"]),
            "n_spk": torch.tensor(sample["n_spk"], dtype=torch.long),
            "room_size": torch.from_numpy(sample["room_size"]),
            "rt60": torch.tensor(sample["rt60"], dtype=torch.float32),
            "snr_db": torch.tensor(sample["snr_db"], dtype=torch.float32),
        }
