import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
from scipy.signal import get_window

EPSILON = torch.finfo(torch.float32).eps


def init_kernerls(win_len, win_inc, fft_len, win_type=None, invers=False):
    """
    Initialize the kernels for STFT and iSTFT operations.
    This function generates the kernel for the convolutional layers used in the short-time Fourier transform (STFT)
    and its inverse (iSTFT). The kernel is created based on the window type and length specified.
    Args:
        win_len (int): Length of the window.
        win_inc (int): Window increment (hop length).
        fft_len (int): Length of the FFT.
        win_type (str, optional): Type of window to apply (e.g., 'hanning', 'hamming'). Default is None (rectangular window).
        invers (bool, optional): If True, computes the pseudo-inverse of the kernel. Default is False.
    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The kernel used for convolution, with shape (2K, 1, win_len). (K: Frequency bin)
            - torch.Tensor: The window applied to the kernel, with shape (1, win_len, 1).
    """
    if win_type == "None" or win_type is None:
        window = np.ones(win_len)  # 모든 표본 통과시키는 직사각 윈도우 (win_len, )
    else:
        window = get_window(win_type, win_len, fftbins=True)  # 지정 윈도우 (win_len, )

    # np.eye(N): 시간 위치별 단위 임펄스
    fourier_basis = np.fft.rfft(np.eye(fft_len))[:win_len]  # 시간축 구간의 실수 입력 DFT basis
    real_kernel = np.real(fourier_basis)  # complex DFT basis의 실수부: (win_len, K)
    imag_kernel = np.imag(fourier_basis)  # complex DFT basis의 허수부: (win_len, K)

    # 주파수(K) 축으로 실수·허수부 결합 후 전치로 출력 채널 축 생성 → (2K, win_len) = (출력채널, 커널길이)
    kernel = np.concatenate([real_kernel, imag_kernel], axis=1).T
    if invers:  # 역변환용 커널 요청 여부
        # 의사역행렬(pseudoinverse) 기반 역변환 근사 커널: (win_len, 2K).T → (2K, win_len)
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window  # 시간 표본별 분석 윈도우 적용
    kernel = kernel[:, None, :]  # conv1d용 입력 채널 축 추가 (2K, 1, win_len)
    # float32 PyTorch 커널 텐서 → STFT 계산에 사용하는 필터
    kernel_tensor = torch.from_numpy(kernel.astype(np.float32))
    # 호환 형태의 윈도우 텐서 (1, win_len, 1) → iSTFT에서 각 프레임의 겹침 정도를 계산하는 별도 데이터
    window_tensor = torch.from_numpy(window[None, :, None].astype(np.float32))

    return kernel_tensor, window_tensor

class ConvSTFT(nn.Module):
    def __init__(self,
                 win_len,
                 win_inc,
                 fft_len=None,
                 vad_threshold = 2/3,
                 win_type = "hann"
                 ):
        super().__init__()

        self.win_len = win_len
        self.stride = win_inc

        if fft_len is None:
            # win_len 이상인 가장 작은 2의 거듭제곱
            self.fft_len = int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        kernel, _ = init_kernerls(win_len, win_inc, self.fft_len, win_type)  # (2K, 1, win_len)

        # win_len으로 정규화: 해당 프레임의 몇 %가 음성인지 - vad threshold로 음성 O/X 판단
        vad_kernel=torch.ones((1,1, self.win_len), dtype=torch.float32)/self.win_len
        # vad_kernel 텐서를 학습 파라미터가 아닌 모델 상태로 등록: 학습 대상은 아니지만 모델과 같은 장치에서 계속 사용해야 하므로
        # VAD/궤적 프레임화용 커널 buffer -> self.vad_kernel=vad_kernel
        self.register_buffer('vad_kernel', vad_kernel)
        self.register_buffer('weight', kernel)
        self.vad_threshold = vad_threshold

        self.dim = self.fft_len

    def get_vad_frame(self, vad):
        """
        샘플 단위(waveform 레벨) VAD 라벨을 STFT 프레임 단위로 변환
        Arg:
            vad (Tensor): (B, Q, L) 형태의 0/1 샘플 단위 VAD| Q: 최대 화자 수, L: 샘플 길이
        Return:
            (B, Q, T) Tensor 형태의 프레임 단위 VAD | T: STFT 프레임 수
        """

        B, Q, L = vad.shape
        vad = vad.float().view(B * Q, 1, L)
        pad_size = (self.stride - L) % self.stride
        vad = F.pad(vad, [0, pad_size])  # 왼쪽 패딩 = 0, 오른쪽 패딩 = pad_size
        vad = F.conv1d(vad, self.vad_kernel, stride=self.stride)
        # 각 프레임의 음성 비율이 threshold 이상인지: gd = greater than or equal >=
        vad = vad.view(B, Q, -1).ge(self.vad_threshold).long()  # (B, Q, T)
        return vad

    def forward(self, inputs, cplx=True):
        """
        Conv1d 기반 STFT 계산
        Args:
            inputs (Tensor): (B, L) or (B, C, L) 형태의 오디오 샘플 | B: 배치수, C: 마이크 채널 수, L: 오디오 샘플 길이
            cplx (bool): True면 복소수를 실수부/허수부로 분리해서 반환, False면 magnitude/phase로 변환해서 반환
        Return:
            Tuple[Tensor, Tensor]: cplx 값에 따라
                - cplx=True: (r, i)
                - cplx=False: (mags, phase)
                shape: (B, F, T) or (B, C ,F, T)
        """
        if inputs.dim() == 2:  # (B, L): 배치 오디오 1채널인 경우
            B, L = inputs.shape
            inputs = torch.unsqueeze(inputs, 1)  # (B, 1, L): 채널 차원 추가
            pad_size = (self.stride - L) % self.stride
            inputs = F.pad(inputs, [0, pad_size])
            outputs = F.conv1d(inputs, self.weight, stride=self.stride)  # (B, 2F ,T)
            # real & imag
            r, i = torch.chunk(outputs, 2, dim=1)  # (B, F, T)
        else:
            B, C, L = inputs.shape
            inputs = inputs.view(B*C, 1, L)
            pad_size = (self.stride - L) % self.stride
            inputs = F.pad(inputs, [0, pad_size])
            outputs = F.conv1d(inputs, self.weight, stride=self.stride)  # (B*C, 2F, T )
            outputs = outputs.view(B, C, -1, outputs.shape[-1])  # (B, C, 2F, T) 

            r, i = torch.chunk(outputs, 2, dim=2)  # (B, C, F, T)

        if cplx:
            return r, i
        else:
            mags = torch.clamp(r**2 + i**2, EPSILON) ** 0.5
            phase = torch.atan2(i,r)
            return mags, phase
