from __future__ import annotations

import torch

from model.gcc_phat import (
    GCCPHATProcess,
    gcc_distribution,
    microphone_pair_indices,
)


def test_pair_order_and_output_shapes() -> None:
    torch.manual_seed(0)
    audio = torch.randn(2, 4, 2_048)
    coordinates = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.04, 0.00, 0.00],
            [0.00, 0.04, 0.00],
            [0.00, 0.00, 0.04],
        ]
    )
    extractor = GCCPHATProcess(
        win_length=256,
        hop_length=64,
        num_delay_bins=65,
        distribution_scale=2.0,
    )

    gcc_phat, observation_distribution, delay_samples, pair_indices = extractor(
        audio, coordinates
    )

    assert torch.equal(
        pair_indices.cpu(),
        torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]),
    )
    assert gcc_phat.shape == (2, 6, 29, 65)
    assert observation_distribution.shape == gcc_phat.shape
    assert delay_samples.shape == (2, 65)
    assert torch.allclose(
        observation_distribution.sum(dim=-1),
        torch.ones(2, 6, 29),
        atol=1e-6,
    )
    assert torch.isfinite(observation_distribution).all()


def test_conv_stft_cross_correlation_interface() -> None:
    torch.manual_seed(2)
    extractor = GCCPHATProcess(win_length=256, hop_length=64)
    x_real, x_imag = extractor.STFT(torch.randn(1, 3, 1_024), cplx=True)

    _, _, cross_real, cross_imag, pairs = extractor.cross_correlation(
        x_real,
        x_imag,
        phat=True,
    )
    phat_cross_spectrum = torch.complex(cross_real, cross_imag)

    assert torch.equal(pairs.cpu(), torch.tensor([[0, 1], [0, 2], [1, 2]]))
    assert phat_cross_spectrum.shape == (
        1,
        3,
        extractor.fft_length // 2 + 1,
        13,
    )
    assert torch.allclose(
        phat_cross_spectrum.abs(),
        torch.ones_like(phat_cross_spectrum.real),
        atol=1e-6,
    )


def test_known_sample_delay_has_expected_gcc_peak() -> None:
    torch.manual_seed(1)
    sample_rate = 16_000
    delay = 4
    source = torch.randn(4_096)
    delayed = torch.zeros_like(source)
    delayed[delay:] = source[:-delay]
    audio = torch.stack((source, delayed))

    # The 10 cm spacing admits about 4.66 samples of propagation delay.
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]])
    extractor = GCCPHATProcess(
        sample_rate=sample_rate,
        win_length=512,
        hop_length=256,
        num_delay_bins=129,
        distribution_scale=1.0,
        normalization="std",
    )

    gcc_phat, _, delay_samples, _ = extractor(audio, coordinates)
    mean_gcc = gcc_phat[0, 0].mean(dim=0)
    peak_delay_samples = delay_samples[0, mean_gcc.argmax()].item()

    # Xi * conj(Xj) uses the cross-correlation convention whose peak is -delay
    # when channel j is a delayed copy of channel i.
    assert abs(peak_delay_samples + delay) < 0.15


def test_paper_variance_normalization_and_silent_uniform_distribution() -> None:
    gcc = torch.tensor([[[[1.0, 2.0, 4.0]]]])
    normalized, probability = gcc_distribution(
        gcc,
        distribution_scale=1.5,
        eps=1e-8,
        normalization="variance",
    )
    centered = gcc - gcc.mean(dim=-1, keepdim=True)
    expected = centered / centered.square().mean(dim=-1, keepdim=True)
    assert torch.allclose(normalized, expected)
    assert torch.allclose(probability, torch.softmax(1.5 * expected, dim=-1))

    silent = torch.zeros(2, 3, 4, 5)
    _, silent_probability = gcc_distribution(
        silent,
        distribution_scale=10.0,
    )
    assert torch.allclose(silent_probability, torch.full_like(silent, 0.2))


def test_microphone_pairs_require_two_channels() -> None:
    try:
        microphone_pair_indices(1)
    except ValueError as exc:
        assert "two microphones" in str(exc)
    else:
        raise AssertionError("Expected microphone_pair_indices(1) to fail.")
