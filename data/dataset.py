"""LibriSpeech 기반 이동 음원 합성 데이터셋."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from glob import glob
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf
import torch
import webrtcvad
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset, Sampler

from .mic_arrays import get_fixed_array, random_rotate, sample_dynamic_array_vectorized
from .simulate import (
    N_SPK,
    SimulationConfig,
    SimulationRequest,
    prepare_simulation,
    render_simulation,
)

PROFILE_SPECS = {
    "stage1": {"array_type": "tetrahedron", "channel_range": (4, 4)},
    "stage2": {"array_type": "dynamic", "channel_range": (4, 4)},
    "stage3": {"array_type": "dynamic", "channel_range": (4, 12)},
    "tetrahedron": {"array_type": "tetrahedron", "channel_range": (4, 4)},
    "nao4": {"array_type": "nao4", "channel_range": (4, 4)},
    "nao12": {"array_type": "nao12", "channel_range": (12, 12)},
    "dynamic4": {"array_type": "dynamic", "channel_range": (4, 4)},
    "dynamic4to12": {"array_type": "dynamic", "channel_range": (4, 12)},
    "dynamic13to16": {"array_type": "dynamic", "channel_range": (13, 16)},
}

WEBRTC_VAD_FRAME_MS = 10
WEBRTC_VAD_MODES = (3, 2, 1)
MINIMUM_ACTIVE_RATIO = 0.66


def _discover_audio_files(root: str, patterns: Iterable[str]) -> list[str]:
    """하위 디렉터리의 오디오 파일 탐색."""

    files: list[str] = []
    for pattern in patterns:
        files.extend(glob(os.path.join(root, "**", pattern), recursive=True))
    return sorted(set(files))


def _discover_librispeech_chapters(root: str) -> list[list[str]]:
    """LibriSpeech 파일의 chapter 단위 그룹화."""

    files = _discover_audio_files(root, ("*.flac", "*.wav"))
    chapters: dict[str, list[str]] = {}
    for path in files:
        chapters.setdefault(os.path.dirname(path), []).append(path)
    return [sorted(chapters[key]) for key in sorted(chapters)]


def _load_audio_mono(path: str) -> tuple[np.ndarray, int]:
    """오디오의 단일 채널 float32 형식 로드."""

    audio, sample_rate = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sample_rate)


def _resample_audio(
    audio: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    """다상 필터 기반 샘플링 주파수 변환."""

    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    divisor = int(np.gcd(source_rate, target_rate))
    return resample_poly(
        audio,
        up=target_rate // divisor,
        down=source_rate // divisor,
    ).astype(np.float32)


def _crop_or_pad(
    audio: np.ndarray,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """무작위 구간 추출 또는 뒤쪽 영점 채움."""

    if audio.shape[0] >= length:
        start = int(rng.integers(0, audio.shape[0] - length + 1))
        return audio[start : start + length].astype(np.float32, copy=False)
    return np.pad(audio, (0, length - audio.shape[0])).astype(np.float32)


def _sample_vad_mask(
    audio: np.ndarray,
    sample_rate: int,
    mode: int,
    frame_ms: int = WEBRTC_VAD_FRAME_MS,
) -> np.ndarray:
    """WebRTC VAD의 샘플 단위 활동 마스크 생성."""

    frame_length = int(sample_rate * frame_ms / 1000)
    if frame_length <= 0:
        raise ValueError("VAD 프레임 길이는 양수여야 함")
    if audio.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    detector = webrtcvad.Vad(mode)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    padding = (-pcm.shape[0]) % frame_length
    padded = np.pad(pcm, (0, padding)) if padding else pcm
    activity = np.zeros(padded.shape[0], dtype=np.float32)

    try:
        for start in range(0, padded.shape[0], frame_length):
            stop = start + frame_length
            if detector.is_speech(padded[start:stop].tobytes(), sample_rate):
                activity[start:stop] = 1.0
    except ValueError as error:
        raise ValueError(
            f"WebRTC VAD 처리 실패: sample_rate={sample_rate}, frame_ms={frame_ms}"
        ) from error
    return activity[: audio.shape[0]]


def _concatenate_chapter_utterances(
    utterance_paths: Sequence[str],
    length: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """같은 chapter의 연속 발화를 무작위 시작점부터 연결."""

    if not utterance_paths:
        raise ValueError("비어 있는 LibriSpeech chapter")

    start_index = int(rng.integers(0, len(utterance_paths)))
    chunks: list[np.ndarray] = []
    accumulated = 0
    utterance_index = start_index
    while accumulated < length:
        audio, source_rate = _load_audio_mono(utterance_paths[utterance_index])
        audio = _resample_audio(audio, source_rate, sample_rate)
        chunks.append(audio)
        accumulated += audio.shape[0]
        utterance_index = (utterance_index + 1) % len(utterance_paths)

    speech = np.concatenate(chunks)[:length].astype(np.float32, copy=False)
    speech -= np.mean(speech, dtype=np.float64)
    return speech


def _clean_speech_and_vad(
    speech: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """활동 비율에 따른 단계적 VAD와 무음 제거."""

    activity = np.ones(speech.shape[0], dtype=np.float32)
    for mode in WEBRTC_VAD_MODES:
        activity = _sample_vad_mask(speech, sample_rate, mode)
        if float(np.mean(activity)) >= MINIMUM_ACTIVE_RATIO:
            break
    return (speech * activity).astype(np.float32), activity


class SyntheticDOADataset(Dataset):
    """chapter 단위 LibriSpeech 이동 음원 합성 데이터셋."""

    def __init__(
        self,
        librispeech_root: str,
        ms_snsd_root: str,
        num_samples: int | None = None,
        profile: str = "stage1",
        batch_size: int = 16,
        seed: int = 0,
        simulation_config: SimulationConfig | None = None,
        rotate_arrays: bool = True,
        channel_schedule: Sequence[int] | None = None,
    ) -> None:
        if profile not in PROFILE_SPECS:
            raise ValueError(f"지원하지 않는 배열 프로필: {profile!r}")

        self.librispeech_root = librispeech_root
        self.ms_snsd_root = ms_snsd_root
        self.profile = profile
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rotate_arrays = bool(rotate_arrays)
        self.simulation_config = simulation_config or SimulationConfig()
        self.simulation_config.validate()
        self._epoch = 0

        self.speech_chapters = _discover_librispeech_chapters(librispeech_root)
        if not self.speech_chapters:
            raise FileNotFoundError(
                f"LibriSpeech 오디오 파일을 찾을 수 없음: {librispeech_root!r}"
            )
        self.speech_files = [
            path for chapter in self.speech_chapters for path in chapter
        ]
        # Experiment 1의 AWGN은 외부 잡음 음원을 사용하지 않는다. 이 경우
        # MS-SNSD를 읽거나 잡음 파일 선택으로 RNG 상태를 소비하지 않는다.
        if getattr(self.simulation_config, "noise_mode", "mixed") == "awgn":
            self.noise_files = []
        else:
            self.noise_files = _discover_audio_files(
                ms_snsd_root,
                ("*.wav", "*.flac"),
            )
            if not self.noise_files:
                raise FileNotFoundError(
                    f"MS-SNSD 오디오 파일을 찾을 수 없음: {ms_snsd_root!r}"
                )

        self.channel_schedule = (
            np.asarray(channel_schedule, dtype=np.int32)
            if channel_schedule is not None
            else None
        )
        if self.channel_schedule is not None:
            self.num_samples = int(self.channel_schedule.shape[0])
        elif num_samples is None:
            self.num_samples = len(self.speech_chapters)
        else:
            self.num_samples = int(num_samples)
        if self.num_samples <= 0:
            raise ValueError("num_samples는 양수여야 함")

        self.channel_counts = self._assign_channel_counts(self.seed)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """에폭 변경과 채널 수 재배정."""

        self._epoch = int(epoch)
        self.channel_counts = self._assign_channel_counts(self.seed + self._epoch)

    def set_profile(self, profile: str) -> None:
        """점진 학습용 배열 프로필 변경."""

        if profile not in PROFILE_SPECS:
            raise ValueError(f"지원하지 않는 배열 프로필: {profile!r}")
        self.profile = profile
        self.channel_counts = self._assign_channel_counts(self.seed + self._epoch)

    def _assign_channel_counts(self, seed: int) -> np.ndarray:
        """배치별 동일 채널 수 배정."""

        if self.channel_schedule is not None:
            return self.channel_schedule.copy()

        minimum, maximum = PROFILE_SPECS[self.profile]["channel_range"]
        if minimum == maximum:
            return np.full(self.num_samples, minimum, dtype=np.int32)

        rng = np.random.default_rng(seed)
        num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        batch_counts = rng.integers(minimum, maximum + 1, size=num_batches)
        return np.repeat(batch_counts, self.batch_size)[: self.num_samples].astype(
            np.int32
        )

    def _sample_mic_coordinates(
        self,
        num_channels: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """프로필에 따른 배열 좌표 표본추출."""

        array_type = PROFILE_SPECS[self.profile]["array_type"]
        if array_type == "dynamic":
            coordinates = sample_dynamic_array_vectorized(num_channels, rng=rng)
        else:
            coordinates = get_fixed_array(array_type)
        if self.rotate_arrays:
            coordinates = random_rotate(coordinates, rng)
        return coordinates.astype(np.float32)

    def _sample_speech(
        self,
        chapter: Sequence[str],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """20초 chapter 음성과 실제 활동 마스크 생성."""

        speech = _concatenate_chapter_utterances(
            chapter,
            self.simulation_config.segment_samples,
            self.simulation_config.sample_rate,
            rng,
        )
        return _clean_speech_and_vad(
            speech,
            self.simulation_config.sample_rate,
        )

    def _sample_noise(
        self,
        path: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """MS-SNSD 잡음의 무작위 20초 구간 생성."""

        noise, source_rate = _load_audio_mono(path)
        noise = _resample_audio(
            noise,
            source_rate,
            self.simulation_config.sample_rate,
        )
        return _crop_or_pad(
            noise,
            self.simulation_config.segment_samples,
            rng,
        )

    def prepare_sample(self, index: int) -> SimulationRequest:
        """파일 로딩과 난수 표본추출만 수행해 gpuRIR 렌더링 요청을 준비."""

        seed = (
            self.seed * 1_000_003 + self._epoch * self.num_samples + int(index)
        ) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        chapter = self.speech_chapters[index % len(self.speech_chapters)]
        speech, source_vad = self._sample_speech(chapter, rng)
        coherent_noise = None
        if self.noise_files:
            noise_index = int(rng.integers(0, len(self.noise_files)))
            coherent_noise = self._sample_noise(self.noise_files[noise_index], rng)
        microphone_coordinates = self._sample_mic_coordinates(
            int(self.channel_counts[index]),
            rng,
        )

        return prepare_simulation(
            speeches=[speech],
            microphone_coordinates=microphone_coordinates,
            rng=rng,
            config=self.simulation_config,
            vads=[source_vad],
            coherent_noise=coherent_noise,
        )

    @staticmethod
    def render_sample(request: SimulationRequest) -> dict[str, torch.Tensor]:
        """준비된 요청을 gpuRIR로 렌더링하고 학습용 Tensor로 변환."""

        sample = render_simulation(request)
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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """기존 Dataset API를 유지하며 준비와 렌더링을 순서대로 실행."""

        return self.render_sample(self.prepare_sample(index))


class ChannelGroupBatchSampler(Sampler[list[int]]):
    """동일 채널 수의 샘플로만 구성되는 배치 샘플러."""

    def __init__(
        self,
        channel_counts: Sequence[int] | SyntheticDOADataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        self.dataset = (
            channel_counts if isinstance(channel_counts, SyntheticDOADataset) else None
        )
        self.channel_counts = (
            None
            if self.dataset is not None
            else np.asarray(channel_counts, dtype=np.int32)
        )
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

    def _build_batches(self) -> list[list[int]]:
        channel_counts = (
            self.dataset.channel_counts
            if self.dataset is not None
            else self.channel_counts
        )
        if channel_counts is None:
            raise RuntimeError("채널 수 정보 없음")
        groups: dict[int, list[int]] = {}
        for index, count in enumerate(channel_counts.tolist()):
            groups.setdefault(int(count), []).append(index)

        batches: list[list[int]] = []
        for indices in groups.values():
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        return batches

    def __iter__(self):
        batches = self._build_batches()
        if self.shuffle:
            np.random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(self._build_batches())


def build_dataloader(
    dataset: SyntheticDOADataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    prefetch_factor: int = 1,
    drop_last: bool = True,
) -> DataLoader:
    """채널 수 그룹화를 적용한 데이터 로더 생성.

    학습은 고정 배치 크기를 위해 ``drop_last=True``를 쓰고, 검증/평가는
    전체 chapter를 포함하도록 반드시 ``drop_last=False``를 지정한다.
    """

    sampler = ChannelGroupBatchSampler(
        channel_counts=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    options: dict = {
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": False,
    }
    if num_workers > 0:
        options["multiprocessing_context"] = mp.get_context("spawn")
        options["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(dataset, **options)


def _run_dataset_smoke_test(args: argparse.Namespace) -> None:
    """명령행 데이터 생성 점검."""

    config = SimulationConfig()
    dataset = SyntheticDOADataset(
        librispeech_root=args.librispeech_root,
        ms_snsd_root=args.ms_snsd_root,
        num_samples=args.num_samples,
        profile=args.profile,
        batch_size=args.batch_size,
        seed=args.seed,
        simulation_config=config,
        rotate_arrays=not args.no_rotate_arrays,
    )
    dataset.set_epoch(args.epoch)
    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    print("=== 이동 음원 합성 데이터 점검 ===")
    print(f"chapter 수: {len(dataset.speech_chapters)}")
    print(f"샘플 수: {len(dataset)}")
    print(f"배열 프로필: {dataset.profile}")
    print(f"MS-SNSD 파일 수: {len(dataset.noise_files)}")

    inspected = 0
    for batch_index, batch in enumerate(loader):
        print(f"배치 {batch_index}")
        for key, value in batch.items():
            print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        if batch["input_audio"].shape[1] != batch["mic_coordinate"].shape[1]:
            raise RuntimeError("오디오와 마이크 좌표의 채널 수 불일치")
        if batch["input_audio"].shape[-1] != batch["spherical_position"].shape[-1]:
            raise RuntimeError("오디오와 이동 위치 라벨의 시간축 불일치")
        inspected += 1
        if inspected >= args.num_batches:
            break
    if inspected == 0:
        raise RuntimeError("생성된 배치 없음")


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    parser = argparse.ArgumentParser(description="이동 음원 합성 데이터 점검")
    parser.add_argument(
        "--librispeech-root",
        default=os.path.join(
            project_root,
            "datasets",
            "librispeech",
            "LibriSpeech",
            "test-clean",
        ),
    )
    parser.add_argument(
        "--ms-snsd-root",
        default=os.path.join(
            project_root,
            "datasets",
            "ms-snsd",
            "MS-SNSD",
            "noise_test",
        ),
    )
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--profile", choices=sorted(PROFILE_SPECS), default="stage1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--no-rotate-arrays", action="store_true")
    _run_dataset_smoke_test(parser.parse_args())
