"""
Loss function (Eq.17-25)
    VariationalDOAEncoder의 vMF posterior (mu, kappa)와 
    physics-based decoder가 출력하는 p(tau_k|z)를 학습 가능한 손실로 결합한다.

    1. von_mises_fisher_kl_loss: 
        vMF(mu, kappa)와 Uniform(S^2) 사이의 KL divergence (Eq.24)
    2. normalize_gcc_phat: 
        입력 GCC-PHAT를 delay-bin 축 기준으로 정규화 (Eq.17-19)
    3. interpolate_time_axis: 
        encoder의 시간축 pooling으로 어긋난 원본 시간축 T와 latent 시간축 T'를 선형보간으로 맞춤 
        (gcc_phat, activity mask 공용)
    4. input_delay_distribution: 
        정규화된 GCC-PHAT에 weighted softmax를 적용해 입력측 시간지연 분포 p(tau_k|g_k)를 얻음 (Eq.20)
    5. physics_loss: 
        p(tau_k|g_k)와 decoder가 예측한 p(tau_k|z) 사이의 cross entropy, 활동마스크(a_x)로 무음 구간을 제외하고 가중평균 (Eq.21-22)
    6. elbo_doa_loss: 
        physics_loss와 KL 정규화 항을 beta로 결합한 최종 손실 (Eq.25)

논문과 다른 점 (가변 마이크 대응):
    논문은 pair 축을 합으로 줄인다. 조건부 독립 가정에서 log-likelihood가 합이므로
    그 자체는 옳지만, 마이크쌍 개수 K가 6(4채널)에서 66(12채널)까지 변하는
    가변 마이크에서는 손실 크기가 그대로 K에 비례해 버린다.

    그래서 ELBO 전체를 K로 나눈다:

        (1/K)[sum_k CE_k + beta * KL] = mean_k CE_k + (beta/K) * KL

    physics_loss가 pair 평균을 쓰고 elbo_doa_loss가 beta를 K로 나누는 것은
    이 하나의 재척도화의 양쪽 절반이다. 한쪽만 적용하면 증거와 prior의 비율이
    K배 어긋나 (마이크가 많을수록 prior가 세지는) 다른 모델이 되어 버린다.
    전역 1/K 배이므로 gradient 방향과 최적점은 논문 그대로다.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def von_mises_fisher_kl_loss(kappa: Tensor, eps: float = 1e-6) -> Tensor:
    """vMF(mu, kappa)와 Uniform(S^2) 사이의 KL divergence (L_KL, Eq.24)

        L_KL = kappa * (coth(kappa) - 1/kappa) + log(kappa) - log(sinh(kappa))

    mu에 의존하지 않는 항이므로(균등분포는 방향에 무관) kappa만 입력음

    수치안정화:
        - coth(kappa)를 cosh/sinh로 직접 계산하면 kappa가 커질 때 overflow하므로
          coth(kappa) = 1 + 2/(e^{2*kappa} - 1) 형태로 변환
        - log(sinh(kappa))도 같은 이유로 k - log(2) + log(1 - e^{-2*kappa})로 바꿔
          sinh 자체의 overflow 회피
        - kappa -> 0 근처에서는 log(kappa)와 log1p(-exp(-2*kappa))가 각각 -inf로 발산하고
          그 차이만 유한하게 남는(상쇄오차, catastrophic cancellation) 문제 발생
          이 텐서는 크기가 작아(프레임 수 정도) 비용 부담이 거의 없으므로,
          이 계산 구간만 float64로 올려서 유효자리 손실을 줄인 뒤 원래 dtype으로 복원

    Args:
        kappa: (B, T', 1) 양수 (vMF의 집중도)
        eps: kappa=0 근방에서 log/나눗셈이 발산하지 않도록 하는 하한

    Returns:
        (B, T', 1), 항상 0 이상인 KL divergence
    """

    kappa64 = kappa.double().clamp_min(eps)

    log_sinh_kappa = kappa64 - math.log(2.0) + torch.log1p(-torch.exp(-2.0 * kappa64))
    coth_minus_inverse = 1.0 + 2.0 / torch.expm1(2.0 * kappa64) - 1.0 / kappa64

    kl = kappa64 * coth_minus_inverse + torch.log(kappa64) - log_sinh_kappa

    # 이론적으로 KL >= 0이지만 부동소수점 반올림으로 아주 작은 음수가 나올 수 있어 clamp
    return kl.clamp_min(0.0).to(kappa.dtype)


def interpolate_time_axis(x: Tensor, target_length: int, time_dim: int = -2) -> Tensor:
    """
    x의 time_dim(T) 축을 target_length 프레임 수(T')로 선형보간

    encoder가 시간축을 (5,1,1) pooling으로 줄이기 때문에,
    gcc_phat/activity mask의 원본 시간축 T와 z/mu/kappa의 latent 시간축 T'가 어긋나므로 
    둘 다 T'로 선형보간 후 Eq.17-22 계산

    Args:
        x: 임의의 shape, time_dim 위치가 시간축인 텐서 
            (예: gcc_phat (B,K,T,G), activity mask (B,T))
        target_length: 목표 시간 프레임 수 T'
        time_dim: x에서 시간축의 위치.
            기본값은 뒤에서 두번째(gcc_phat처럼 시간축 뒤에 delay-bin 축이 남는 경우)
            activity mask처럼 시간축이 마지막 축이면 -1

    Returns:
        time_dim(T)만 target_length(T')로 바뀌고 나머지 shape은 그대로인 텐서
        (B,K,T',G)
    """

    # F.interpolate(mode="linear")는 (N,C,L) 형태를 요구하므로,
    # 시간축을 마지막으로 옮기고 나머지 모든 축은 배치(N)로 접은 뒤 채널(C)=1을 끼워 넣는다
    time_dim = time_dim % x.ndim

    moved = x.movedim(time_dim, -1)  # (B,K,G,T)
    flattened = moved.reshape(-1, 1, moved.shape[-1])  # (N,1,T)

    resized = F.interpolate(
        flattened, size=target_length, mode="linear", align_corners=False
    )   # (N,1,T')
    resized = resized.reshape(*moved.shape[:-1], target_length)  # (B,K,G,T')
    return resized.movedim(-1, time_dim)  # 다시 원래 자리로 복원


def normalize_gcc_phat(gcc_phat: Tensor, eps: float = 1e-8) -> Tensor:
    """
    delay-bin 축(G) 기준으로 GCC-PHAT를 정규화 (Eq.17-19)

    Args:
        gcc_phat: (B, K, T', G) -- GCCPHATProcess가 계산한 raw GCC-PHAT
        eps: 분산이 0에 가까울 때 0-division을 막는 epsilon

    Returns:
        (B, K, T', G) 정규화된 g_tilde
    """

    mean = gcc_phat.mean(dim=-1, keepdim=True)  # (B, K, T, 1)
    variance = (gcc_phat - mean).square().mean(dim=-1, keepdim=True)  # (B, K, T, 1)

    return (gcc_phat - mean) / (variance + eps)  # (B, K, T, G)


def input_delay_distribution(g_tilde: Tensor, lambda_scale: float = 8.0) -> Tensor:
    """
    정규화된 GCC-PHAT에 weighted softmax를 적용해 입력 시간지연 분포 p(tau_k|g_k) 계산 (Eq.20)

    Args:
        g_tilde: (B, K, T', G) -- normalize_gcc_phat의 출력
            physics_loss에서 p(tau_k|z)와 비교하려면 
            시간축이 미리 interpolate_time_axis로 T'에 맞춰져 있어야 함
        lambda_scale: GCC 분포를 스케일하는 hyperparameter (8.0)
            
    Returns:
        (B, K, T', G), 마지막 축(G)에 대해 합이 1인 확률분포
    """

    return torch.softmax(lambda_scale * g_tilde, dim=-1)


def physics_loss(
    p_target: Tensor,
    p_pred: Tensor,
    activity_mask: Tensor,
    eps: float = 1e-8
) -> Tensor:
    """
    p(tau_k|g_k)와 p(tau_k|z) 사이의 cross entropy, 활동마스크로 가중평균 (Eq.21-22)

     Args:
        p_target: (B, K, T', G) -- input_delay_distribution의 출력, p(tau_k|g_k)
        p_pred: (B, K, T', G) -- physics_based_decoder의 출력, p(tau_k|z)
        activity_mask: (B, T'), 0(무음)~1(활성) 범위
            interpolate_time_axis로 시간축을 T'에 맞춤
        eps: log(0) 발산을 막는 epsilon

    Returns:
        scalar, 활동 구간에 대해 가중평균한 pair당 cross entropy (Eq.22를 K로 나눈 값)

    pair 축을 합이 아니라 평균으로 줄이므로 반환값이 마이크쌍 개수 K에 비례해
    커지지 않는다. 대신 elbo_doa_loss가 KL 쪽에 beta/K를 적용해 두 항의 비율을
    논문과 같게 유지한다 (모듈 docstring 참고).

    B와 T' 축은 원래부터 (마스크) 평균이고 G 축만이 cross entropy 정의상 합이므로,
    K를 평균으로 두는 편이 축 처리에도 일관된다.
    """

    cross_entropy = -(p_target * torch.log(p_pred.clamp_min(eps))).sum(dim=-1)  # (B,K,T'): sum_{tau_k} delay-bin(G) 축 합
    cross_entropy = cross_entropy.mean(dim=1)  # (B, T'): mean_k -- K에 비례하지 않도록

    mask = activity_mask.to(cross_entropy.dtype)  # (B, T')
    weighted_sum = (cross_entropy * mask).sum()
    normalizer = mask.sum().clamp_min(eps)  # 활성 프레임이 얼마나 있는지
    return weighted_sum / normalizer


def elbo_doa_loss(
    phy_loss: Tensor,
    kl_loss: Tensor,
    beta: float,
    num_pairs: int,
) -> Tensor:
    """
    physics loss와 KL 정규화 항을 beta로 결합한 최종 학습 손실 (Eq. 25)

    physics_loss가 pair 평균이므로 KL에는 beta 대신 beta/K를 적용한다.
    둘을 합치면 논문 ELBO를 정확히 1/K배 한 것이 되어, 손실 크기만 마이크쌍
    개수에 불변해지고 증거 대 prior의 비율은 논문 그대로 유지된다.

        (1/K)[sum_k CE_k + beta * KL] = mean_k CE_k + (beta/K) * KL

    beta를 그대로 두면 원래 목적함수에서 beta가 K배 세진 것과 같아진다.
    마이크가 많아질수록(=증거가 많아질수록) prior를 더 세게 당기는 셈이라
    가변 마이크에서는 방향이 정반대가 된다.

    Args:
        phy_loss: scalar
            Eq.21-22를 pair 평균으로 줄인 값 (physics_loss의 출력)
        kl_loss: (B, T', 1)
            Eq.24의 KL divergence (von_mises_fisher_kl_loss의 출력)
        beta: KL 항 가중치
            첫 5% epoch는 0(posterior collapse 방지), 이후 1.0
        num_pairs: 이 배치의 마이크쌍 개수 K.
            ChannelGroupBatchSampler가 배치를 채널 수로 묶으므로 배치 안에서는 상수다.

    Returns:
        scalar, 최종 학습 손실
    """

    if num_pairs < 1:
        raise ValueError(f"num_pairs must be positive, got {num_pairs}.")

    # beta=0인 warm-up 구간에서는 kl_loss가 (kappa 폭주 등으로) nan/inf여도
    # 0 * nan = nan, 0 * inf = nan이 되어 버려서 beta=0의 "KL 무시" 의도가
    # 깨지므로, beta=0일 때는 곱셈 자체를 하지 않고 phy_loss만 반환
    if beta == 0.0:
        return phy_loss
    return phy_loss + (beta / num_pairs) * kl_loss.mean()
