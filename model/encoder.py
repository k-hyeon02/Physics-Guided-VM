"""논문 Figure 2의 pairwise variational DOA encoder"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FeatureWisePReLU(nn.Module):
    """마지막 축(feature)이 채널인 텐서용 PReLU

    nn.PReLU는 항상 dim=1을 채널로 가정하지만, GRU/MLP를 거친 텐서는
    (B, T, feature) 형태로 feature가 마지막 축에 있으므로 직접 broadcast한다.
    """

    def __init__(self, num_parameters: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((num_parameters,), 0.25))

    def forward(self, x: Tensor) -> Tensor:
        return torch.where(x >= 0, x, self.weight * x)


def channelwise_softmax_aggregation(x: Tensor, pair_dim: int = 1) -> Tensor:
    """AGG-RL의 CWSA(channel-wise softmax aggregation)로 pair 축을 없앤다

    pair마다 가중치를 학습된 feature 값 자체로 정하고(softmax), 가중합과
    가중 표준편차를 이어 붙인다:

        w = softmax_K(x),  y = w * x
        out = concat(sum_K y, std_K y)

    단순 합산과 달리 softmax 가중치가 pair 축에서 1로 정규화되므로 출력 크기가
    마이크쌍 개수 K에 비례해 커지지 않는다. 가변 마이크에서는 K가 6(4채널)에서
    66(12채널)까지 11배 변하는데, 합산을 쓰면 posterior_head가 보는 입력 크기와
    kappa가 그만큼 함께 끌려간다.

    표준편차 항은 pair 사이의 의견 불일치도를 그대로 실어 나르므로, 집중도
    kappa가 "몇 쌍을 봤는가"가 아니라 "쌍들이 얼마나 합의하는가"에 근거하게 된다.

    표준편차는 K=1에서도 정의되도록 unbiased=False(모표준편차)를 쓴다.
    K >= 6인 4채널 이상에서는 unbiased 여부의 차이가 sqrt(K/(K-1)) 배로 무시할 수 있다.

    Args:
        x: (B, K, T', H) -- pair축이 pair_dim에 있는 텐서
        pair_dim: 없앨 pair 축의 위치

    Returns:
        (B, T', 2H) -- 마지막 축에 [가중합, 가중 표준편차]가 이어 붙은 텐서
    """

    weights = x.softmax(dim=pair_dim)  # (B, K, T', H) -- feature마다 pair 축에서 정규화
    weighted = weights * x  # (B, K, T', H)

    total = weighted.sum(dim=pair_dim)  # (B, T', H)
    spread = weighted.std(dim=pair_dim, unbiased=False)  # (B, T', H)

    return torch.cat((total, spread), dim=-1)  # (B, T', 2H)


class ConvBlock(nn.Module):
    """Figure 2(b): metadata로 조건화된 2D conv block

    처리 순서:
        Conv2d -> + metadata projection -> GroupNorm -> PReLU -> MaxPool2d

    Args:
        in_channels: 입력 채널 수
        out_channels: conv 출력 채널 수
        metadata_dim: pairwise metadata (v_i, v_j)의 차원
        pool_size: (time_pool, lag_pool) 형태의 MaxPool2d 커널 크기

    Input:
        x: (B, in_channels, T, G)
        metadata: (B, metadata_dim)

    Output:
        (B, out_channels, T // time_pool, G // lag_pool)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        metadata_dim: int,
        pool_size: tuple[int, int],
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.metadata_projection = nn.Linear(metadata_dim, out_channels)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.activation = nn.PReLU(num_parameters=out_channels)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x: Tensor, metadata: Tensor) -> Tensor:
        bias = self.metadata_projection(metadata)[:, :, None, None]  # (B, out_channels, 1, 1)
        conditioned = self.conv(x) + bias
        return self.pool(self.activation(self.norm(conditioned)))


class VariationalDOAEncoder(nn.Module):
    """논문 Figure 2(a): pairwise-shared variational DOA encoder

    마이크쌍마다 가중치를 공유하는 conv+GRU+MLP 인코더를 통과시킨 뒤
    pair 축을 집계하여 방향(mu)과 집중도(kappa)의 von Mises-Fisher
    posterior 파라미터를 얻는다.

    논문은 pair 축을 단순 합산하지만, 여기서는 기본값을 CWSA로 바꿨다.
    합산은 출력 크기가 마이크쌍 개수 K에 선형 비례해서 가변 마이크(4채널 K=6
    ~ 12채널 K=66, 11배)에서는 배열 크기가 곧 활성값 크기가 되어 버린다.
    CWSA는 pair 축에서 정규화된 softmax 가중합이라 K에 거의 불변이고,
    baseline 재현이 필요하면 aggregation="sum"으로 되돌릴 수 있다.

    처리 순서 (pair별 가중치 공유):

    3x ConvBlock -> GRU(2 Layer) -> pairwise MLP(2 Layer)
        -> (pair 축 CWSA 집계) -> final MLP -> (mu, kappa)

    Args:
        conv_channels: conv block 출력 채널 수
        lag_pool_sizes: 3개 conv block의 lag축(G) pooling 크기
        time_pool_sizes: 3개 conv block의 time축(T) pooling 크기
        num_delay_bins: 입력 gcc_phat의 lag축 크기 G (GRU 입력 크기 결정에 필요)
        metadata_dim: pairwise metadata (v_i, v_j) 차원
        gru_layers: unidirectional GRU 레이어 수
        hidden_size: GRU/MLP의 공통 hidden 크기
        aggregation: pair 축 집계 방식.
            "cwsa"(기본) -- softmax 가중합과 표준편차를 이어 붙임 (K에 거의 불변)
            "sum" -- 논문 원본의 단순 합산 (K에 선형 비례)
        eps: 0-division 및 zero-direction fallback 판정에 쓰는 epsilon

    Input:
        gcc_phat: (B, K, T, G) -- K = M(M-1)/2 마이크쌍 수
        metadata: (B 또는 1, K, metadata_dim) -- (B, K)와 batch 방향으로 broadcast 가능

    Output:
        (mu, kappa) 튜플
        mu: (B, T', 3) 단위벡터
        kappa: (B, T', 1) 양수
    """

    def __init__(
        self,
        conv_channels: int = 128,
        lag_pool_sizes: tuple[int, int, int] = (2, 2, 2),
        time_pool_sizes: tuple[int, int, int] = (5, 1, 1),
        num_delay_bins: int = 64,
        metadata_dim: int = 6,
        gru_layers: int = 2,
        hidden_size: int = 128,
        aggregation: str = "cwsa",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if aggregation not in ("cwsa", "sum"):
            raise ValueError(
                f"aggregation must be 'cwsa' or 'sum', got '{aggregation}'."
            )
        self.aggregation = aggregation

        self.conv_blocks = nn.ModuleList(
            ConvBlock(
                in_channels=1 if index == 0 else conv_channels,
                out_channels=conv_channels,
                metadata_dim=metadata_dim,
                pool_size=(time_pool_sizes[index], lag_pool_sizes[index]),
            )
            for index in range(3)
        )

        # GRU에 들어갈 입력 크기(input_size)를 계산: 3개의 MaxPool2d를 거치고 나면 lag축(G)이 몇 칸 남는지
        lag_downsample = 1
        for lag_pool in lag_pool_sizes:
            lag_downsample *= lag_pool  # 총 축소 비율 곱셈으로 누적 (2*2*2=8)
        remaining_lag_bins = num_delay_bins // lag_downsample  # 64 // 8 = 8

        self.gru = nn.GRU(
            input_size=conv_channels * remaining_lag_bins,
            hidden_size=hidden_size,
            num_layers=gru_layers,  # 2
            batch_first=True,
        )

        self.pairwise_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            FeatureWisePReLU(num_parameters=hidden_size),
            nn.Linear(hidden_size, hidden_size),
            FeatureWisePReLU(num_parameters=hidden_size),
        )

        # CWSA는 [가중합, 가중 표준편차]를 이어 붙이므로 집계 출력이 2*hidden_size
        aggregated_size = hidden_size * 2 if aggregation == "cwsa" else hidden_size

        self.posterior_head = nn.Sequential(
            nn.Linear(aggregated_size, hidden_size),
            FeatureWisePReLU(num_parameters=hidden_size),
            nn.Linear(hidden_size, 4),  # 앞 3개는 mu의 raw 좌표, 마지막은 raw kappa
        )

        self.eps = float(eps)

    def forward(self, gcc_phat: Tensor, metadata: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, num_pairs, num_frames, num_lags = gcc_phat.shape  # (B, K, T, G)

        if metadata.shape[0] == 1 and batch_size > 1:
            metadata = metadata.expand(batch_size, -1, -1)

        # pair마다 가중치를 공유하므로 (B, K, ...) -> (B*K, ...)로 합쳐 한 번에 처리
        hidden = gcc_phat.reshape(batch_size * num_pairs, 1, num_frames, num_lags)  # (B*K, 1, T, G)
        conditioning = metadata.reshape(batch_size * num_pairs, -1)  # (B*K, 6)

        for conv_block in self.conv_blocks: # conv block 3번
            hidden = conv_block(hidden, conditioning)  # (B*K, C, T', G')

        hidden = hidden.permute(0, 2, 1, 3).flatten(2)  # (B*K, T', C*G')
        hidden, _ = self.gru(hidden)  # (B*K, T', H)
        hidden = self.pairwise_mlp(hidden)  # (B*K, T', H)

        # pair 축 집계
        num_output_frames = hidden.shape[1]
        hidden = hidden.reshape(batch_size, num_pairs, num_output_frames, -1)  # (B, K, T', H)
        if self.aggregation == "cwsa":
            hidden = channelwise_softmax_aggregation(hidden)  # (B, T', 2H)
        else:
            hidden = hidden.sum(dim=1)  # (B, T', H)
        # 어느 쪽이든 pair 축을 대칭적으로 없애므로 pair 순서에 불변

        raw_posterior = self.posterior_head(hidden)  # (B, T', 4)
        raw_direction = raw_posterior[..., :3]  # mu
        raw_concentration = raw_posterior[..., 3:]  # kappa

        direction_norm = torch.linalg.vector_norm(raw_direction, dim=-1, keepdim=True)  # (B, T', 1)

        # raw_direction과 같은 shape (B,T′,3)의 0 텐서를 만들고, 첫 번째 성분만 1로 세팅 (방향을 못 정할 때 쓸 기본값)
        fallback_direction = torch.zeros_like(raw_direction)
        fallback_direction[..., 0] = 1.0

        # raw_direction 벡터 크기가 거의 0이면, 그 벡터 전체를 [1,0,0]으로 변경 (backpropagation gradient 차단)
        safe_direction = torch.where(direction_norm < self.eps, fallback_direction, raw_direction)

        # 방향 단위벡터
        safe_norm = torch.linalg.vector_norm(safe_direction, dim=-1, keepdim=True).clamp_min(self.eps)
        mu = safe_direction / safe_norm

        # kappa는 반드시 양수이므로 softplus(smooth한 ReLU)로 매핑 
        kappa = F.softplus(raw_concentration)

        return mu, kappa