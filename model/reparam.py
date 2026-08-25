"""논문 Figure 1의 reparam 블록 (Eq 13-16)

VariationalDOAEncoder가 출력하는 vMF posterior 파라미터 (mu, kappa)로부터

    sample_von_mises_fisher: 미분 가능한 방식으로 z ~ vMF(mu, kappa)를 샘플링 (Eq 13-16)

을 제공한다. Eq 24의 KL 정규화 항(von_mises_fisher_kl_loss)은 인코더 파라미터가 아니라
loss 계산용이라 model/loss.py로 옮겼다.

학습 가능한 파라미터가 없으므로 nn.Module이 아닌 순수 함수로 둔다.

z가 S^2(3차원 단위구) 위에 있어야 하므로 Gaussian VAE의 z = mu + sigma * epsilon 같은
reparameterization은 쓸 수 없다. 대신 "북극(canonical mean [0,0,1])을 향한 vMF에서
샘플링한 뒤 mu 방향으로 회전"하는 방식을 쓰는데, 3차원(S^2)의 경우 특이하게도
rejection sampling이나 Bessel 함수 없이 inverse-CDF만으로 완전히 미분 가능한
샘플링이 가능하다 (Ulrich, 1984). 이 덕분에 아래 모든 연산은 log/exp/sqrt/cross 같은
평범한 미분 가능 연산으로만 구성되고, 별도의 custom autograd.Function 없이도
mu와 kappa 양쪽으로 gradient가 정상적으로 흐른다.
"""

import math

import torch
from torch import Tensor


def _orthonormal_tangent_basis(mu: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    """mu에 접하는 평면의 정규직교 기저 (e1, e2)를 구성 (Eq 15)

    mu와 평행하지 않은 기준축 reference를 하나 고른 뒤 cross product로
    e1 = normalize(reference x mu), e2 = mu x e1을 만들면 (e1, e2, mu)가
    우수(right-handed) 정규직교 기저가 된다.

    기본 기준축은 [0, 0, 1]이지만, mu가 이 축과 거의 평행한 위치(z 성분이 거의 ±1)에서는
    cross product가 0벡터에 가까워져 e1이 불안정해지므로, 그 위치만 [1, 0, 0]으로 바꾼다.
    (encoder.py의 zero-direction fallback과 동일한 torch.where 패턴)

    Args:
        mu: (B,T′,3) 단위벡터
        eps: 평행 판정 및 0-division 방지에 쓰는 epsilon

    Returns:
        (e1, e2): 각각 (B,T′,3), mu와 함께 정규직교 기저를 이룸
    """

    reference = torch.zeros_like(mu)  # (B,T′,3)
    reference[..., 2] = 1.0  # [0,0,1]

    alternate_reference = torch.zeros_like(mu)
    alternate_reference[..., 0] = 1.0  # [1,0,0]

    # mu의 z 성분이 거의 ±1이면 [0,0,1]과 거의 평행 -> [1,0,0]으로 교체
    is_near_pole = mu[..., 2:3].abs() > (1.0 - eps)  # (B,T′,1)
    # is_near_pole 참이면 [1,0,0]  | 거짓이면 [0,0,1]
    reference = torch.where(is_near_pole, alternate_reference, reference)

    e1 = torch.linalg.cross(reference, mu, dim=-1)  # (B,T′,3)
    e1 = e1 / torch.linalg.vector_norm(e1, dim=-1, keepdim=True).clamp_min(eps)
    e2 = torch.linalg.cross(mu, e1, dim=-1)  # (B,T′,3), 이미 단위벡터 (mu ⊥ e1이므로 norm=1)

    return e1, e2


def sample_von_mises_fisher(
    mu: Tensor,
    kappa: Tensor,
    eps: float = 1e-6,
    generator: torch.Generator | None = None,
) -> Tensor:
    """vMF(mu, kappa)에서 reparameterization trick으로 z를 샘플링 (Eq 13-16)

    처리 순서:
        1. u1, u2 ~ Uniform(0,1) 두 개를 뽑는다.
        2. u1으로 canonical vMF(북극이 평균인 vMF)의 고도각 성분 w를 inverse-CDF로 샘플링한다 (Eq 14).
        3. u2로 방위각 phi = 2*pi*u2를 정하고, 접평면 반지름 r = sqrt(1 - w^2)를 구한다.
        4. mu에 대한 접평면의 정규직교 기저 (e1, e2)를 구성한다 (Eq 15).
        5. z = r*cos(phi)*e1 + r*sin(phi)*e2 + w*mu로 canonical 샘플을 mu 방향으로 회전시킨다 (Eq 16).

    모든 연산이 elementary하게 미분 가능하므로 custom backward 없이
    autograd만으로 mu, kappa 양쪽에 gradient가 흐른다.

    Args:
        mu: (B,T′,3) 단위벡터 (vMF의 평균 방향)
        kappa: (B,T′,1) 양수 (vMF의 집중도)
        eps: 0-division 및 kappa -> 0 등방 극한 판정에 쓰는 epsilon
        generator: torch.rand에 전달할 난수 생성기 (재현성용, 선택)

    Returns:
        z: (B,T′,3) 단위벡터, ||z|| = 1
    """

    u1 = torch.rand(kappa.shape, dtype=kappa.dtype, device=kappa.device, generator=generator)
    u2 = torch.rand(kappa.shape, dtype=kappa.dtype, device=kappa.device, generator=generator)

    # --- w (고도각) 샘플링 - Eq 14 -----------------------------------------
    # 원래 식 w = (1/k)*log((1-u1)e^{-k} + u1 e^{k})는 e^{k}에서 overflow할 수 있으므로
    # e^{k}로 묶어 exp(-2k) 형태로 바꾼다: w = 1 + (1/k)*log(u1 + (1-u1)*exp(-2k))
    # exp(-2k)는 k가 커도 0으로 underflow만 할 뿐 overflow하지 않아 안전하다.
    kappa_safe = kappa.clamp_min(eps)  # 나눗셈용, 선택(where)과는 독립적으로 clamp
    log_term = torch.log(u1 + (1.0 - u1) * torch.exp(-2.0 * kappa_safe))
    w_general = 1.0 + log_term / kappa_safe

    # kappa -> 0 등방 극한: 0/0 형태를 피하기 위해 별도로 처리 (Eq 14 아래 명시된 극한)
    w_isotropic = 2.0 * u1 - 1.0

    is_isotropic = kappa < eps  # (..., 1)
    w = torch.where(is_isotropic, w_isotropic, w_general)

    # --- 접평면 반지름 및 방위각 --------------------------------------------
    # 부동소수점 오차로 w가 ±1을 살짝 넘어 1 - w^2가 음수가 되는 것을 방지
    tangential_radius = torch.sqrt((1.0 - w * w).clamp_min(0.0))  # (..., 1)
    azimuth = 2.0 * math.pi * u2  # (..., 1)

    # --- canonical 샘플을 mu 방향 기저로 회전, Eq 15-16 ---------------------
    e1, e2 = _orthonormal_tangent_basis(mu, eps)  # 각각 (..., 3)
    z = (tangential_radius * torch.cos(azimuth) * e1 
        + tangential_radius * torch.sin(azimuth) * e2 
        + w * mu)  # (B,T′,3)

    # 수치 오차로 인한 norm 이탈을 막는 안전장치 (encoder.py의 mu 정규화와 동일한 패턴)
    z_norm = torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(eps)
    return z / z_norm
