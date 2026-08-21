"""ConvSTFT 기반 GCC-PHAT 입력 특징과 관측 시간지연 분포"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from STFT import ConvSTFT


Normalization = Literal["variance", "std"]


def microphone_pair_indices(
    num_microphones: int,
    device: torch.device | None = None,
) -> Tensor:
    """모든 ``i < j`` 마이크쌍을 ``(K, 2)`` 텐서로 반환"""

    if num_microphones < 2:
        raise ValueError("At least two microphones are required.")
    # triu_indices: (2, K) -> transpose -> (K, 2), K = M(M-1)/2
    return torch.triu_indices(
        num_microphones,
        num_microphones,
        offset=1,
        device=device,
    ).transpose(0, 1)


class GCCPHATProcess(nn.Module):
    """다채널 waveform에서 물리적으로 유효한 GCC-PHAT 분포를 추출

    처리 순서:

    ``ConvSTFT -> pairwise PHAT cross-spectrum -> IDFT -> physical delay grid
    -> normalization -> weighted softmax``

    Args:
        win_length: STFT window 길이.
        hop_length: STFT hop. ``None``이면 75% overlap인 ``win_length // 4``.
        fft_length: FFT/IDFT 길이.
        sample_rate: sampling frequency ``Fs``.
        num_delay_bins: 고정 지연 bin 개수 ``G``.
        distribution_scale: weighted softmax의 ``lambda``.
        speed_of_sound: 음속 ``c`` (m/s).
        normalization: 논문 그대로면 ``variance``.
        pair_chunk_size: 전체 IDFT를 동시에 만들 마이크쌍 개수.

    Input:
        ``input_audio``: ``(B, M, N)`` 또는 ``(M, N)``.
        ``microphone_coordinates``: 미터 단위 ``(B, M, D)`` 또는 ``(M, D)``.

    Output:
        ``(gcc_phat, observation_distribution, delay_samples, pair_indices)`` 튜플.
        ``gcc_phat``과 ``observation_distribution``의 shape은 ``(B, M(M-1)/2, T, G)``,
        ``delay_samples``는 ``(B, G)``, ``pair_indices``는 ``(M(M-1)/2, 2)``.
    """

    def __init__(
        self,
        win_length: int = 512,
        hop_length: int | None = 128,
        fft_length: int = 512,
        sample_rate: int = 16_000,
        num_delay_bins: int = 64,
        distribution_scale: float = 1.0,
        speed_of_sound: float = 343.0,
        eps: float = 1e-8,
        normalization: Normalization = "variance",
        pair_chunk_size: int = 4,
    ) -> None:
        super().__init__()

        if hop_length is None:
            resolved_hop = win_length // 4
        else:
            resolved_hop = int(hop_length)

        self.STFT = ConvSTFT(
            win_len=win_length,
            win_inc=resolved_hop,
            fft_len=fft_length,
            vad_threshold=2 / 3,
            win_type="hann",
        )
        self.win_length = int(win_length)
        self.hop_length = resolved_hop
        self.fft_length = int(self.STFT.fft_len)
        self.sample_rate = int(sample_rate)
        self.num_delay_bins = int(num_delay_bins)
        self.distribution_scale = float(distribution_scale)
        self.speed_of_sound = float(speed_of_sound)
        self.eps = float(eps)
        self.normalization = normalization
        self.pair_chunk_size = int(pair_chunk_size)

        # G가 짝수이면 중앙 lag 관례에 따라 [-G/2, ..., G/2-1],
        # 홀수이면 zero를 포함하는 대칭 grid를 사용
        center_bin = self.num_delay_bins // 2
        offsets = torch.arange(self.num_delay_bins, dtype=torch.float32) - center_bin  # (G,)
        radius = max(center_bin, self.num_delay_bins - 1 - center_bin)
        self.register_buffer(
            "unit_delay_grid",
            offsets / radius,
            persistent=False,
        )  # (G,)

    # 1. 입력 형식 통일 -----------------------------------------------------

    def _validate_and_batch_inputs(
        self,
        input_audio: Tensor,
        microphone_coordinates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """waveform과 좌표를 각각 ``(B, M, N)``, ``(B, M, D)``로 통일."""

        if input_audio.ndim == 2:
            input_audio = input_audio.unsqueeze(0)  # (M, N) -> (1, M, N)
        if microphone_coordinates.ndim == 2:
            microphone_coordinates = microphone_coordinates.unsqueeze(0)
            # (M, D) -> (1, M, D)
        if input_audio.shape[1] != microphone_coordinates.shape[1]:
            raise ValueError("Audio channels and microphone coordinates do not match.")

        batch_size = input_audio.shape[0]
        if microphone_coordinates.shape[0] == 1 and batch_size > 1:
            microphone_coordinates = microphone_coordinates.expand(
                batch_size,
                -1,
                -1,
            )  # (1, M, D) -> (B, M, D)
        elif microphone_coordinates.shape[0] != batch_size:
            raise ValueError("Audio and coordinate batch dimensions do not match.")

        microphone_coordinates = microphone_coordinates.to(
            device=input_audio.device,
            dtype=input_audio.dtype,
        )
        return input_audio, microphone_coordinates

    # 2. 논문 식 (2): pairwise PHAT cross-spectrum ------------------------

    @staticmethod
    def _cross_spectrum(
        spectrum: Tensor,
        pair_indices: Tensor,
        eps: float,
        phat: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """선택한 마이크쌍의 GCC/GCC-PHAT cross-spectrum을 계산

        ``spectrum``: ``(B, M, F, T)``
        ``pair_indices``: ``(K', 2)``
        """

        spectrum_i = spectrum[:, pair_indices[:, 0]]
        spectrum_j = spectrum[:, pair_indices[:, 1]]  # (B, K', F, T)
        cross_spectrum = spectrum_i * spectrum_j.conj()
        if phat:
            # Xi Xj* / |Xi Xj*| = (Xi / |Xi|)(Xj* / |Xj|)
            cross_spectrum = cross_spectrum / cross_spectrum.abs().clamp_min(eps)
        return spectrum_i, spectrum_j, cross_spectrum

    # 3. 논문 식 (3): 배열의 물리적 지연 범위 -----------------------------

    def _physical_delay_grid(
        self,
        microphone_coordinates: Tensor,
        pair_indices: Tensor,
    ) -> Tensor:
        pair_displacements = (
            microphone_coordinates[:, pair_indices[:, 0]]
            - microphone_coordinates[:, pair_indices[:, 1]]
        )  # (B, K, D)
        pair_max_delay_seconds = (
            torch.linalg.vector_norm(pair_displacements, dim=-1)
            / self.speed_of_sound
        )  # (B, K)
        max_delay_seconds = pair_max_delay_seconds.amax(dim=-1)  # (B,)
        if torch.any(max_delay_seconds <= 0):
            raise ValueError("Every array must contain at least two distinct positions.")

        max_delay_samples = max_delay_seconds * self.sample_rate  # (B,)
        if torch.any(max_delay_samples >= self.fft_length / 2):
            raise ValueError(
                "The physical maximum delay must be smaller than fft_length / 2 "
                "samples to avoid circular-correlation aliasing."
            )

        unit_grid = self.unit_delay_grid.to(
            device=microphone_coordinates.device,
            dtype=microphone_coordinates.dtype,
        )  # (G,)
        delay_seconds = max_delay_seconds[:, None] * unit_grid[None, :]
        return delay_seconds * self.sample_rate  # (B, G)

    # 4. 식 (2)의 IDFT 결과를 식 (3)의 지연축으로 제한 --------------------

    def _interpolate_physical_delay_grid(
        self,
        centered_gcc: Tensor,
        delay_samples: Tensor,
    ) -> Tensor:
        """정수 lag GCC를 배열별 physical fractional-delay grid에 보간

        centered_gcc: (B, K', T, L) -- L=fft_length(정수 lag 축)
        delay_samples: (B, G) -- 보간할 physical delay grid
        """

        batch_size, num_pairs, num_frames, num_lags = centered_gcc.shape
        min_lag = -(num_lags // 2)
        positions = delay_samples - min_lag  # (B, G)
        lower = positions.floor().to(torch.long).clamp(0, num_lags - 1)  # (B, G)
        upper = (lower + 1).clamp(max=num_lags - 1)  # (B, G)
        upper_weight = positions - lower.to(positions.dtype)  # (B, G)

        gather_shape = (
            batch_size,
            num_pairs,
            num_frames,
            self.num_delay_bins,
        )
        lower_index = lower[:, None, None, :].expand(gather_shape)
        upper_index = upper[:, None, None, :].expand(gather_shape)
        lower_value = torch.gather(centered_gcc, dim=-1, index=lower_index)
        upper_value = torch.gather(centered_gcc, dim=-1, index=upper_index)
        return lower_value + upper_weight[:, None, None, :] * (
            upper_value - lower_value
        )

    def _gcc_phat_features(
        self,
        spectrum: Tensor,
        pairs: Tensor,
        delay_samples: Tensor,
    ) -> Tensor:
        """식 (2)의 GCC-PHAT를 계산하고 물리적 ``G``개 지연 bin만 반환."""

        gcc_chunks = []
        for start in range(0, pairs.shape[0], self.pair_chunk_size):
            chunk_pairs = pairs[start : start + self.pair_chunk_size]
            _, _, phat_cross_spectrum = self._cross_spectrum(
                spectrum,
                chunk_pairs,
                eps=self.eps,
                phat=True,
            )  # (B, K', F, T)

            # IDFT: 주파수 F -> 정수 지연 L
            full_gcc = torch.fft.irfft(
                phat_cross_spectrum,
                n=self.fft_length,
                dim=2,
            )  # (B, K', L, T)
            full_gcc = torch.fft.fftshift(full_gcc, dim=2)
            full_gcc = full_gcc.permute(0, 1, 3, 2)  # (B, K', T, L)

            gcc_chunks.append(
                self._interpolate_physical_delay_grid(full_gcc, delay_samples)
            )  # (B, K', T, G)

        return torch.cat(gcc_chunks, dim=1)  # (B, K, T, G)

    # 5. 논문 식 (17)--(20): GCC-PHAT -> 관측 지연분포 -------------------

    @staticmethod
    def _to_observation_distribution(
        gcc_phat: Tensor,
        distribution_scale: float,
        eps: float,
        normalization: Normalization,
    ) -> tuple[Tensor, Tensor]:
        """지연축을 정규화하고 ``p(tau_k | g_k)``를 계산."""

        if gcc_phat.ndim < 1 or gcc_phat.shape[-1] < 2:
            raise ValueError("gcc_phat must contain at least two delay bins.")
        if distribution_scale <= 0:
            raise ValueError("distribution_scale (lambda) must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if normalization not in ("variance", "std"):
            raise ValueError("normalization must be either 'variance' or 'std'.")

        # 식 (17), (18): g_k - mu(g_k)
        centered = gcc_phat - gcc_phat.mean(dim=-1, keepdim=True)
        # 식 (19): Sigma(g_k), shape (B, K, T, 1)
        variance = centered.square().mean(dim=-1, keepdim=True)
        denominator = variance if normalization == "variance" else variance.sqrt()
        normalized = centered / (denominator + eps)  # 식 (17)

        # 식 (20): weighted softmax
        probability = torch.softmax(distribution_scale * normalized, dim=-1)
        return normalized, probability

    # 6. 전체 파이프라인 ---------------------------------------------------

    def forward(
        self,
        input_audio: Tensor,
        microphone_coordinates: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        input_audio, microphone_coordinates = self._validate_and_batch_inputs(
            input_audio,
            microphone_coordinates,
        )
        pairs = microphone_pair_indices(
            input_audio.shape[1],
            input_audio.device,
        )  # (K, 2)

        # STFT: x_i -> X_i
        x_real, x_imag = self.STFT(input_audio, cplx=True)  # 각각 (B, M, F, T)
        spectrum = torch.complex(x_real, x_imag)  # (B, M, F, T)

        # 식 (2), (3): GCC-PHAT와 물리적 지연축
        delay_samples = self._physical_delay_grid(microphone_coordinates, pairs)
        gcc_phat = self._gcc_phat_features(spectrum, pairs, delay_samples)

        # 식 (17)--(20): 관측 시간지연 분포
        _, observation_distribution = self._to_observation_distribution(
            gcc_phat,
            distribution_scale=self.distribution_scale,
            eps=self.eps,
            normalization=self.normalization,
        )
        return gcc_phat, observation_distribution, delay_samples, pairs

    # 보조 공개 API --------------------------------------------------------

    def cross_correlation(
        self,
        x_real: Tensor,
        x_imag: Tensor,
        phat: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """STFT로부터 pairwise cross-spectrum을 직접 반환."""

        if x_real.shape != x_imag.shape or x_real.ndim != 4:
            raise ValueError("STFT real/imag must both have shape (B, M, F, T).")
        spectrum = torch.complex(x_real, x_imag)  # (B, M, F, T)
        pairs = microphone_pair_indices(spectrum.shape[1], spectrum.device)
        spectrum_i, spectrum_j, cross_spectrum = self._cross_spectrum(
            spectrum,
            pairs,
            eps=self.eps,
            phat=phat,
        )
        return (
            spectrum_i,
            spectrum_j,
            cross_spectrum.real,
            cross_spectrum.imag,
            pairs,
        )


def gcc_distribution(
    gcc_phat: Tensor,
    distribution_scale: float,
    eps: float = 1e-8,
    normalization: Normalization = "variance",
) -> tuple[Tensor, Tensor]:
    """논문 식 (17)--(20)을 독립적으로 호출하는 공개 함수.

    입력 ``gcc_phat``의 shape은 ``(B, K, T, G)``이며, 평균과 분산은 마지막
    지연축 ``G``에서 계산한다. ``variance``가 논문 구현이고 ``std``는 일반적인
    z-score 정규화를 위한 선택 사항이다.
    """

    return GCCPHATProcess._to_observation_distribution(
        gcc_phat,
        distribution_scale=distribution_scale,
        eps=eps,
        normalization=normalization,
    )
