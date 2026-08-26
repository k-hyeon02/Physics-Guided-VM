"""
Physics-based decoder
    VariationalDOAEncoder -> reparam.sample_von_mises_fisher를 거쳐 얻은 latent 방향 샘플 z로부터,
    마이크쌍별 예측 시간지연 tau_hat을 물리적으로 계산하고, 이를 중심으로 한 이산 가우시안 분포 p(tau_k|z)를 반환
    학습 시에만 사용되고 추론 시에는 encoder만
"""

import torch
from torch import Tensor

def pairwise_delay(
    z: Tensor,
    pair_displacement: Tensor,
    speed_of_sound: float,
    sample_rate: float
) -> Tensor:
    """
    tau_hat_k = (v_i - v_j)^T z / c * F_s

     Args:
        z: (B, T', 3) 단위벡터 (vMF에서 샘플링된 latent 방향)
        pair_displacement: (B, K, 3) -- v_i - v_j
        speed_of_sound: 음속 c (m/s)
        sample_rate: 샘플링 주파수 F_s

    Returns:
        (B, K, T') tau_hat, sample 단위 (GCCPHATProcess의 delay_samples와 동일 단위)
    """

    # (v_i - v_j): (B,K,3) x z: (B,T',3) -> (B,K,T'): 마이크쌍마다 모든 시간 프레임의 z에 내적
    projected = torch.matmul(pair_displacement, z.transpose(1,2))
    return projected / speed_of_sound * sample_rate


def physics_based_decoder(
    z: Tensor,
    pair_displacement: Tensor,
    delay_bins: Tensor,
    speed_of_sound: float,
    sample_rate: float,
    sigma: float
) -> Tensor:
    """
    tau_hat_k(z)를 중심으로 한 이산 gaussian distribution p(tau_k|z)

    1. pairwise_delay로 예측 지연 tau_hat_k(z) 구함
    2. 각 delay bin tau_k에서의 Gaussian 형태의 logit 계산
        l_k = -1/2 * ((tau_k - tau_hat_k) / sigma)^2
    3. delay bin 축(G)에 대해 softmax 적용하여 정규화 -> p(tau_k|z)

    Args:
        z: (B, T', 3) 단위벡터
        pair_displacement: (B, K, 3) -- v_i - v_j
        delay_bins: (B, G) -- GCCPHATProcess가 계산한 physical delay grid (samples 단위)
        speed_of_sound: 음속 c (m/s)
        sample_rate: 샘플링 주파수 F_s
        sigma: 가우시안 logit의 표준편차 (samples 단위, hyperparameter)

    Returns:
        (B, K, T', G) p(tau_k|z), 마지막 축(G)에 대해 합이 1인 확률분포
    """

    tau_hat = pairwise_delay(z, pair_displacement, speed_of_sound, sample_rate)  # (B,K,T')

    delay_bins = delay_bins[:, None, None, :]  # (B,1,1,G)
    tau_hat = tau_hat[..., None]  # (B,K,T',1)
    logits = -0.5 * ((delay_bins - tau_hat) / sigma) ** 2  # (B,K,T',G)

    return torch.softmax(logits, dim=-1)  # (B,K,T',G)
