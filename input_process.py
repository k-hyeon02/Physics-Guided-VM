"""ConvSTFT 기반 raw GCC-PHAT 계산"""

import torch
from torch import Tensor, nn

from STFT import ConvSTFT


def microphone_pair_indices(
    num_microphones: int,
    device: torch.device | None = None,
) -> Tensor:
    """모든 i < j 마이크쌍을 (K, 2) 텐서로 반환"""

    if num_microphones < 2:
        raise ValueError("At least two microphones are required.")
    # triu_indices: (2, K) -> transpose -> (K, 2), K = M(M-1)/2
    return torch.triu_indices(
        num_microphones,
        num_microphones,
        offset=1,
        device=device,
    ).transpose(0, 1)


def pair_displacement(
    microphone_coordinates: Tensor,
    pair_indices: Tensor,
) -> Tensor:
    """마이크쌍의 상대 위치 벡터 v_i - v_j (Eq 3, 9에서 공용으로 쓰임)

    Args:
        microphone_coordinates: 미터 단위 (B, M, D) 또는 (M, D) 좌표
        pair_indices: (K, 2) 마이크쌍 인덱스

    Returns:
        (B, K, D) -- centroid 이동에 불변 (차이이므로 상쇄됨)
    """

    if microphone_coordinates.ndim == 2:
        microphone_coordinates = microphone_coordinates.unsqueeze(0)
    pair_indices = pair_indices.to(device=microphone_coordinates.device)

    return (
        microphone_coordinates[:, pair_indices[:, 0]]
        - microphone_coordinates[:, pair_indices[:, 1]]
    )  # (B, K, D)


class GCCPHATProcess(nn.Module):
    """다채널 waveform에서 물리적으로 유효한 raw GCC-PHAT를 추출

    처리 순서:

    ConvSTFT -> pairwise PHAT cross-spectrum -> IDFT -> physical delay grid

    Args:
        win_length: STFT window 길이
        hop_length: STFT hop. None이면 논문의 hop rate 0.75를 적용
        fft_length: FFT/IDFT 길이
        sample_rate: sampling frequency Fs
        num_delay_bins: 고정 지연 bin 개수 G
        speed_of_sound: 음속 c (m/s)
        pair_chunk_size: 전체 IDFT를 동시에 만들 마이크쌍 개수

    Input:
        input_audio: (B, M, N) 또는 (M, N)
        microphone_coordinates: 미터 단위 (B, M, D) 또는 (M, D)

    Output:
        (gcc_phat, delay_samples, pair_indices) 튜플
        gcc_phat: (B, K, T, G)  K=M(M-1)/2
        delay_samples: (B, G)
        pair_indices: (K, 2)
    """

    def __init__(
        self,
        win_length: int = 4_096,
        hop_length: int | None = 3_072,
        fft_length: int = 4_096,
        sample_rate: int = 16_000,
        num_delay_bins: int = 64,
        speed_of_sound: float = 343.0,
        eps: float = 1e-8,
        pair_chunk_size: int = 4,
    ) -> None:
        super().__init__()

        if hop_length is None:
            resolved_hop = int(round(win_length * 0.75))
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
        self.speed_of_sound = float(speed_of_sound)
        self.eps = float(eps)
        self.pair_chunk_size = int(pair_chunk_size)

        # 배열의 최대 물리 TDOA 범위를 G개의 공통 lag bin으로 균등 분할
        # G가 짝수이면 중앙 lag 관례에 따라 [-G/2, ..., 0, ..., G/2-1],
        # 실제 지연값은 forward에서 정규화 grid에 배열의 최대 지연을 곱해 결정
        center_bin = self.num_delay_bins // 2
        offsets = torch.arange(self.num_delay_bins, dtype=torch.float32) - center_bin  # (G,)
        radius = max(center_bin, self.num_delay_bins - 1 - center_bin)
        self.register_buffer(
            "unit_delay_grid",
            offsets / radius,  # 정규화
            persistent=False,
        )  # (G,)

    # 1. 입력 형식 통일 -----------------------------------------------------
    def _validate_and_batch_inputs(
        self,
        input_audio: Tensor,
        microphone_coordinates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """waveform과 좌표를 각각 (B, M, N), (B, M, D)로 통일."""

        if input_audio.ndim == 2:
            input_audio = input_audio.unsqueeze(0)  # (M, N) -> (1, M, N)
        if microphone_coordinates.ndim == 2:
            microphone_coordinates = microphone_coordinates.unsqueeze(0)
            # (M, D) -> (1, M, D)
        if input_audio.shape[1] != microphone_coordinates.shape[1]:
            raise ValueError("Audio channels and microphone coordinates do not match.")

        batch_size = input_audio.shape[0]
        if microphone_coordinates.shape[0] == 1 and batch_size > 1:
            microphone_coordinates = microphone_coordinates.expand(batch_size, -1, -1)  # (1, M, D) -> (B, M, D)
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

        spectrum: (B, M, F, T)
        pair_indices: (K', 2)
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
        # 마이크 쌍의 상대 위치 벡터
        pair_displacements = pair_displacement(microphone_coordinates, pair_indices)  # (B, K, D)

        # 상대 위치 벡터의 norm / 음속
        pair_max_delay_seconds = (
            torch.linalg.vector_norm(pair_displacements, dim=-1) / self.speed_of_sound
        )  # (B, K)

        # 모든 마이크 쌍 중 가장 긴 거리를 기준으로 사용
        max_delay_seconds = pair_max_delay_seconds.amax(dim=-1)  # (B,)
        # 초 단위를 sample 단위로 변환
        max_delay_samples = max_delay_seconds * self.sample_rate  # (B,)
        # 물리적 최대 지연이 GCC의 표현 가능 lag 범위 넘으면
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
        min_lag = -(num_lags // 2)  # scalar
        positions = delay_samples - min_lag  # (B, G)  # 원하는 지연의 index 값
        lower = positions.floor().to(torch.long).clamp(0, num_lags - 1)  # position보다 크지 않은 정수 index
        upper = (lower + 1).clamp(max=num_lags - 1)  # position보다 큰 정수 index
        upper_weight = positions - lower.to(positions.dtype)  # interpolate 가중치 계산

        gather_shape = (batch_size, num_pairs, num_frames,self.num_delay_bins) # (B, K', T, G)
        lower_index = lower[:, None, None, :].expand(gather_shape)  # (B, K', T, G)
        upper_index = upper[:, None, None, :].expand(gather_shape)

        # GCC 값 가져오기
        lower_value = torch.gather(centered_gcc, dim=-1, index=lower_index)
        upper_value = torch.gather(centered_gcc, dim=-1, index=upper_index)
        return lower_value + upper_weight[:, None, None, :] * (upper_value - lower_value)

    def _gcc_phat_features(
        self,
        spectrum: Tensor,
        pairs: Tensor,
        delay_samples: Tensor,
    ) -> Tensor:
        """식 (2)의 GCC-PHAT를 계산하고 물리적 G개 지연 bin만 반환."""

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

    # 5. 전체 파이프라인 ---------------------------------------------------
    def forward(
        self,
        input_audio: Tensor,
        microphone_coordinates: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:

        input_audio, microphone_coordinates = self._validate_and_batch_inputs(input_audio,
                                                                              microphone_coordinates)
        pairs = microphone_pair_indices(input_audio.shape[1], input_audio.device)  # (K, 2)

        # STFT: x_i -> X_i
        x_real, x_imag = self.STFT(input_audio, cplx=True)  # 각각 (B, M, F, T)
        spectrum = torch.complex(x_real, x_imag)  # (B, M, F, T)

        # 식 (2), (3): raw GCC-PHAT와 물리적 지연축
        delay_samples = self._physical_delay_grid(microphone_coordinates, pairs)
        gcc_phat = self._gcc_phat_features(spectrum, pairs, delay_samples)
        return gcc_phat, delay_samples, pairs

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
