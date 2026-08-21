"""논문 평가 조건에 맞춘 LOCATA 단일 음원 로더."""

from __future__ import annotations

import os
import re
from glob import glob
from typing import Iterable

import numpy as np
import pandas as pd

LOCATA_ARRAYS = {
    "benchmark2": ("NAO robot", 12),
    "eigenmike": ("Eigenmike", 32),
    "dicit": ("DICIT", 15),
    "dummy": ("dummy", 4),
}

PAPER_TASKS = (1, 3, 5)
FS_TARGET = 16_000


def _read_wav(path: str) -> tuple[np.ndarray, int]:
    """다채널 오디오의 float32 형식 로드."""

    import soundfile as sf

    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return data, int(sample_rate)


def _resample(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """시간축 기준 다상 리샘플링."""

    if source_rate == target_rate:
        return signal.astype(np.float32, copy=False)

    from math import gcd
    from scipy.signal import resample_poly

    divisor = gcd(source_rate, target_rate)
    return resample_poly(
        signal,
        target_rate // divisor,
        source_rate // divisor,
        axis=0,
    ).astype(np.float32)


def _read_position_table(path: str) -> pd.DataFrame:
    """LOCATA 위치 표의 열 이름 정규화."""

    try:
        table = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        table = pd.read_csv(path, sep=r"\s+")
    table.columns = [column.strip() for column in table.columns]
    return table


def _columns(table: pd.DataFrame, *names: str) -> np.ndarray:
    """선택 열의 행 우선 float64 배열 변환."""

    return np.stack(
        [table[name].to_numpy(dtype=np.float64) for name in names],
        axis=1,
    )


def _timestamps(table: pd.DataFrame) -> np.ndarray:
    """날짜와 시각 열의 초 단위 절대 시간 변환."""

    dates = pd.to_datetime(
        {
            "year": table["year"].astype(int),
            "month": table["month"].astype(int),
            "day": table["day"].astype(int),
        },
        utc=True,
    )
    day_seconds = dates.astype("int64").to_numpy(dtype=np.float64) / 1e9
    return (
        day_seconds
        + table["hour"].to_numpy(dtype=np.float64) * 3600.0
        + table["minute"].to_numpy(dtype=np.float64) * 60.0
        + table["second"].to_numpy(dtype=np.float64)
    )


def _nearest_row_indices(
    metadata_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """각 오디오 시각에 가장 가까운 위치 표 행 탐색."""

    right = np.searchsorted(metadata_times, target_times, side="left")
    right = np.clip(right, 0, len(metadata_times) - 1)
    left = np.clip(right - 1, 0, len(metadata_times) - 1)
    choose_left = np.abs(target_times - metadata_times[left]) <= np.abs(
        metadata_times[right] - target_times
    )
    return np.where(choose_left, left, right).astype(np.int64)


def _rotation_matrices(table: pd.DataFrame) -> np.ndarray:
    """위치 표의 배열 자세 행렬 추출."""

    names = [f"rotation_{row}{column}" for row in range(1, 4) for column in range(1, 4)]
    if not all(name in table.columns for name in names):
        return np.broadcast_to(
            np.eye(3, dtype=np.float64),
            (len(table), 3, 3),
        ).copy()
    return _columns(table, *names).reshape(-1, 3, 3)


def _array_mic_coordinates(
    array_table: pd.DataFrame,
    num_channels: int,
) -> np.ndarray:
    """첫 위치의 마이크 좌표를 배열 좌표계로 변환."""

    microphone_positions = []
    for microphone in range(1, num_channels + 1):
        names = [
            f"mic{microphone}_x",
            f"mic{microphone}_y",
            f"mic{microphone}_z",
        ]
        if not all(name in array_table.columns for name in names):
            raise KeyError(f"마이크 {microphone}의 위치 열 없음")
        microphone_positions.append(
            [float(array_table[name].iloc[0]) for name in names]
        )

    center = _columns(array_table, "x", "y", "z")[0]
    rotation = _rotation_matrices(array_table)[0]
    relative_world = np.asarray(microphone_positions, dtype=np.float64) - center
    return (relative_world @ rotation).astype(np.float32)


def compute_gt_doa(
    array_position: np.ndarray,
    array_rotation: np.ndarray,
    source_position: np.ndarray,
) -> tuple[float, float, float]:
    """단일 시점 전역 좌표의 배열 기준 DOA 변환."""

    relative = array_rotation.T @ (source_position - array_position)
    x, y, z = relative
    horizontal = np.sqrt(x * x + y * y)
    azimuth = np.mod(np.degrees(np.arctan2(y, x)), 360.0)
    polar = np.degrees(np.arctan2(horizontal, z))
    distance = np.sqrt(horizontal * horizontal + z * z)
    return float(azimuth), float(polar), float(distance)


def _compute_gt_doa_track(
    array_table: pd.DataFrame,
    source_table: pd.DataFrame,
    audio_times: np.ndarray,
    block_size: int = 65_536,
) -> np.ndarray:
    """시간별 배열 위치·자세와 음원 위치를 반영한 DOA 궤적 계산."""

    array_indices = _nearest_row_indices(_timestamps(array_table), audio_times)
    source_indices = _nearest_row_indices(_timestamps(source_table), audio_times)
    array_positions = _columns(array_table, "x", "y", "z")
    source_positions = _columns(source_table, "x", "y", "z")
    array_rotations = _rotation_matrices(array_table)
    spherical = np.empty((3, audio_times.shape[0]), dtype=np.float64)

    for start in range(0, audio_times.shape[0], block_size):
        stop = min(start + block_size, audio_times.shape[0])
        array_rows = array_indices[start:stop]
        source_rows = source_indices[start:stop]
        relative_world = source_positions[source_rows] - array_positions[array_rows]
        relative_array = np.einsum(
            "bij,bj->bi",
            np.transpose(array_rotations[array_rows], (0, 2, 1)),
            relative_world,
        )
        x = relative_array[:, 0]
        y = relative_array[:, 1]
        z = relative_array[:, 2]
        horizontal = np.sqrt(x * x + y * y)
        spherical[0, start:stop] = np.mod(
            np.degrees(np.arctan2(y, x)),
            360.0,
        )
        spherical[1, start:stop] = np.degrees(np.arctan2(horizontal, z))
        spherical[2, start:stop] = np.sqrt(horizontal * horizontal + z * z)
    return spherical


def _read_locata_vad(path: str) -> np.ndarray | None:
    """LOCATA 공식 샘플 단위 VAD 로드."""

    if not os.path.exists(path):
        return None
    values = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            value = line.strip()
            if value and value != "VAD":
                values.append(float(value))
    if not values:
        return None
    return np.asarray(values, dtype=np.float32)


def _vad_from_source(source_audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """근접 음원 신호 기반 에너지 VAD 생성."""

    frame_length = int(sample_rate * 0.03)
    activity = np.zeros(source_audio.shape[0], dtype=np.float32)
    threshold = (np.max(np.abs(source_audio)) + 1e-8) * 10.0 ** (-40.0 / 20.0)
    for start in range(0, len(source_audio) - frame_length + 1, frame_length):
        stop = start + frame_length
        if np.sqrt(np.mean(source_audio[start:stop] ** 2)) > threshold:
            activity[start:stop] = 1.0
    return activity


def _align_mask(mask: np.ndarray, length: int) -> np.ndarray:
    """최근접 인덱스 기반 마스크 길이 정합."""

    indices = (
        np.arange(length, dtype=np.float64) * (mask.shape[0] / length)
    ).astype(np.int64)
    return mask[np.clip(indices, 0, mask.shape[0] - 1)]


def load_recording(
    recording_directory: str,
    array_name: str = "benchmark2",
    max_speakers: int = 1,
) -> dict | None:
    """LOCATA 녹음과 시간별 단일 음원 위치 라벨 로드."""

    if array_name not in LOCATA_ARRAYS:
        raise ValueError(f"지원하지 않는 LOCATA 배열: {array_name!r}")
    _, num_channels = LOCATA_ARRAYS[array_name]

    audio_path = os.path.join(
        recording_directory,
        f"audio_array_{array_name}.wav",
    )
    array_path = os.path.join(
        recording_directory,
        f"position_array_{array_name}.txt",
    )
    if not os.path.exists(audio_path) or not os.path.exists(array_path):
        return None

    source_paths = sorted(
        glob(os.path.join(recording_directory, "position_source_*.txt"))
    )
    num_speakers = len(source_paths)
    if num_speakers == 0 or num_speakers > max_speakers:
        return None

    audio, source_rate = _read_wav(audio_path)
    audio = _resample(audio, source_rate, FS_TARGET)
    num_samples = audio.shape[0]
    array_table = _read_position_table(array_path)
    start_time = _timestamps(array_table)[0]
    audio_times = start_time + np.arange(num_samples, dtype=np.float64) / FS_TARGET

    spherical = np.zeros((num_speakers, 3, num_samples), dtype=np.float64)
    activity = np.zeros((num_speakers, num_samples), dtype=np.float32)
    for source_index, source_path in enumerate(source_paths):
        source_table = _read_position_table(source_path)
        spherical[source_index] = _compute_gt_doa_track(
            array_table,
            source_table,
            audio_times,
        )

        source_tag = (
            os.path.basename(source_path)
            .replace("position_source_", "")
            .replace(".txt", "")
        )
        official_vad = _read_locata_vad(
            os.path.join(recording_directory, f"VAD_source_{source_tag}.txt")
        )
        if official_vad is not None:
            activity[source_index] = _align_mask(official_vad, num_samples)
            continue

        source_audio_path = os.path.join(
            recording_directory,
            f"audio_source_{source_tag}.wav",
        )
        if os.path.exists(source_audio_path):
            source_audio, source_audio_rate = _read_wav(source_audio_path)
            source_audio = _resample(
                source_audio[:, :1],
                source_audio_rate,
                FS_TARGET,
            )[:, 0]
            activity[source_index] = _align_mask(
                _vad_from_source(source_audio, FS_TARGET),
                num_samples,
            )
        else:
            activity[source_index] = 1.0

    return {
        "vad": activity.astype(bool),
        "n_spk": int(num_speakers),
        "n_channel": int(num_channels),
        "mic_dim": "D3",
        "input_audio": audio.T.astype(np.float32),
        "mic_coordinate": _array_mic_coordinates(
            array_table,
            num_channels,
        ).astype(np.float64),
        "spherical_position": spherical,
    }


def _task_number(path: str) -> int | None:
    """경로 구성요소의 LOCATA 과제 번호 추출."""

    for part in os.path.normpath(path).split(os.sep):
        match = re.fullmatch(r"task(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def find_recordings(
    locata_root: str,
    array_name: str = "benchmark2",
    tasks: Iterable[int] | None = PAPER_TASKS,
):
    """논문 평가 과제에 해당하는 LOCATA 녹음 경로 탐색."""

    selected_tasks = None if tasks is None else set(map(int, tasks))
    pattern = os.path.join(
        locata_root,
        "**",
        f"audio_array_{array_name}.wav",
    )
    for audio_path in sorted(glob(pattern, recursive=True)):
        recording_directory = os.path.dirname(audio_path)
        if selected_tasks is None or _task_number(recording_directory) in selected_tasks:
            yield recording_directory


def to_dataframe(sample: dict) -> pd.DataFrame:
    """샘플 딕셔너리의 단일 행 DataFrame 변환."""

    return pd.DataFrame({key: [value] for key, value in sample.items()})
