from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from data.dataset import (
    ChannelGroupBatchSampler,
    SyntheticDOADataset,
    _concatenate_chapter_utterances,
    _discover_librispeech_chapters,
)
from data.simulate import (
    SimulationConfig,
    _interpolate_trajectory,
    _sample_trajectory,
    simulate_one_sample,
)
from data.static import StaticSimulationConfig, simulate_static_sample


class _FakeGpuRIR:
    @staticmethod
    def beta_SabineEstimation(room_size, rt60, absorption_weights):
        return np.full(6, 0.5, dtype=np.float32)

    @staticmethod
    def att2t_SabineEstimator(attenuation, rt60):
        return attenuation / 60.0 * rt60

    @staticmethod
    def t2n(t_diff, room_size):
        return [1, 1, 1]

    @staticmethod
    def simulateRIR(**kwargs):
        num_sources = kwargs["pos_src"].shape[0]
        num_microphones = kwargs["pos_rcv"].shape[0]
        rirs = np.zeros((num_sources, num_microphones, 4), dtype=np.float32)
        rirs[:, :, 0] = 1.0
        return rirs

    @staticmethod
    def simulateTrajectory(source_signal, rirs, timestamps=None, fs=None):
        source_signal = np.asarray(source_signal, dtype=np.float32)
        num_microphones = rirs.shape[1]
        output = np.zeros(
            (source_signal.shape[0] + rirs.shape[2] - 1, num_microphones),
            dtype=np.float32,
        )
        output[: source_signal.shape[0]] = source_signal[:, None]
        return output


def test_paper_default_configuration() -> None:
    config = SimulationConfig()

    assert config.sample_rate == 16_000
    assert config.segment_seconds == 20.0
    assert config.trajectory_points == 156
    assert config.rt60_s == (0.2, 1.0)
    assert config.snr_db == (5.0, 30.0)
    assert config.noise_min_distance_m == 2.5


def test_sinusoidal_trajectory_stays_inside_room() -> None:
    config = SimulationConfig(segment_seconds=1.0, trajectory_points=156)
    room_size = np.array([3.0, 4.0, 2.5])
    points, timestamps = _sample_trajectory(
        room_size,
        np.random.default_rng(7),
        config,
    )
    trajectory = _interpolate_trajectory(
        points,
        timestamps,
        config.segment_samples,
        config.sample_rate,
    )

    assert points.shape == (156, 3)
    assert timestamps.shape == (156,)
    assert np.all(points >= 0.0)
    assert np.all(points <= room_size)
    assert np.all(trajectory >= 0.0)
    assert np.all(trajectory <= room_size)
    assert not np.allclose(points[0], points[-1])


def test_chapter_grouping_and_concatenation(tmp_path: Path) -> None:
    sample_rate = 16_000
    chapter_a = tmp_path / "1" / "10"
    chapter_b = tmp_path / "1" / "11"
    chapter_a.mkdir(parents=True)
    chapter_b.mkdir(parents=True)
    sf.write(chapter_a / "1-10-1.flac", np.full(80, 0.1), sample_rate)
    sf.write(chapter_a / "1-10-2.flac", np.full(80, 0.2), sample_rate)
    sf.write(chapter_b / "1-11-1.flac", np.full(80, 0.3), sample_rate)

    chapters = _discover_librispeech_chapters(str(tmp_path))
    segment = _concatenate_chapter_utterances(
        chapters[0],
        length=200,
        sample_rate=sample_rate,
        rng=np.random.default_rng(0),
    )

    assert len(chapters) == 2
    assert sorted(map(len, chapters)) == [1, 2]
    assert segment.shape == (200,)
    assert abs(float(segment.mean())) < 1e-6


def test_simulated_sample_has_time_varying_position(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "gpuRIR", _FakeGpuRIR)
    config = SimulationConfig(
        segment_seconds=0.02,
        trajectory_points=4,
        room_size_min_m=(3.0, 3.0, 2.5),
        room_size_max_m=(3.0, 3.0, 2.5),
        rt60_s=(0.2, 0.2),
        snr_db=(30.0, 30.0),
    )
    length = config.segment_samples
    sample = simulate_one_sample(
        speeches=[np.ones(length, dtype=np.float32)],
        microphone_coordinates=np.array(
            [
                [-0.02, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [0.0, -0.02, 0.0],
                [0.0, 0.02, 0.0],
            ],
            dtype=np.float32,
        ),
        rng=np.random.default_rng(11),
        config=config,
        vads=[np.ones(length, dtype=np.float32)],
        coherent_noise=np.linspace(-1.0, 1.0, length, dtype=np.float32),
    )

    assert sample["input_audio"].shape == (4, length)
    assert sample["vad"].shape == (1, length)
    assert sample["spherical_position"].shape == (1, 3, length)
    assert sample["source_trajectory"].shape == (1, 3, length)
    assert not np.allclose(
        sample["spherical_position"][0, :, 0],
        sample["spherical_position"][0, :, -1],
    )


def test_stage3_keeps_random_channel_count_per_batch() -> None:
    dataset = SyntheticDOADataset.__new__(SyntheticDOADataset)
    dataset.channel_schedule = None
    dataset.profile = "stage3"
    dataset.num_samples = 16 * 20
    dataset.batch_size = 16

    channel_counts = dataset._assign_channel_counts(seed=3)

    assert set(np.unique(channel_counts)).issubset(set(range(4, 13)))
    assert len(np.unique(channel_counts)) > 1
    for start in range(0, dataset.num_samples, dataset.batch_size):
        assert np.unique(channel_counts[start : start + dataset.batch_size]).size == 1


def test_batch_sampler_tracks_profile_channel_changes() -> None:
    dataset = SyntheticDOADataset.__new__(SyntheticDOADataset)
    dataset.channel_counts = np.array([4, 4, 4, 4], dtype=np.int32)
    sampler = ChannelGroupBatchSampler(dataset, batch_size=2, shuffle=False)
    assert list(sampler) == [[0, 1], [2, 3]]

    dataset.channel_counts = np.array([4, 8, 4, 8], dtype=np.int32)
    assert list(sampler) == [[0, 2], [1, 3]]


def test_static_configuration_uses_four_seconds() -> None:
    config = StaticSimulationConfig()

    assert config.sample_rate == 16_000
    assert config.segment_seconds == 4.0
    assert config.segment_samples == 64_000
    assert config.max_speakers == 1
    assert config.rt60_s == (0.2, 1.0)
    assert config.snr_db == (5.0, 30.0)


def test_static_sample_keeps_one_position(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "gpuRIR", _FakeGpuRIR)
    config = StaticSimulationConfig(
        segment_seconds=0.02,
        room_size_min_m=(3.0, 3.0, 2.5),
        room_size_max_m=(3.0, 3.0, 2.5),
        rt60_s=(0.2, 0.2),
        snr_db=(30.0, 30.0),
    )
    length = config.segment_samples
    sample = simulate_static_sample(
        speech=np.ones(length, dtype=np.float32),
        coherent_noise=np.linspace(-1.0, 1.0, length, dtype=np.float32),
        microphone_coordinates=np.array(
            [
                [-0.02, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [0.0, -0.02, 0.0],
                [0.0, 0.02, 0.0],
            ],
            dtype=np.float32,
        ),
        rng=np.random.default_rng(17),
        config=config,
        vad=np.ones(length, dtype=np.float32),
    )

    assert sample["input_audio"].shape == (4, length)
    assert sample["spherical_position"].shape == (1, 3, length)
    assert sample["source_trajectory"].shape == (1, 3, length)
    assert np.all(sample["spherical_position"] == sample["polar_position"][:, :, None])
    assert np.all(sample["source_trajectory"] == sample["source_trajectory"][:, :, :1])
