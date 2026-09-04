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
    """CWSA로 pair 축을 softmax 가중합과 가중 표준편차로 집계한다.

    Args:
        x: 일반적으로 (B, K, T', H) 형태인 pairwise feature.
        pair_dim: 마이크 pair 축의 위치.

    Returns:
        pair 축이 제거되고 마지막 축에 가중합과 가중 표준편차가 이어 붙은 텐서.
    """

    weights = x.softmax(dim=pair_dim)
    weighted = weights * x
    mean = weighted.sum(dim=pair_dim)  # (B, 1, T', H)
    var = (weights * (x - mean).square()).sum(dim=pair_dim, keepdim=True)
    std = torch.sqrt(var)
    return torch.cat((mean, std), dim=-1)


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

    처리 순서 (pair별 가중치 공유):

    3x ConvBlock -> GRU(2 Layer) -> pairwise MLP(2 Layer)
        -> (pair 집계) -> final MLP -> (mu, kappa)

    Args:
        conv_channels: conv block 출력 채널 수
        lag_pool_sizes: 3개 conv block의 lag축(G) pooling 크기
        time_pool_sizes: 3개 conv block의 time축(T) pooling 크기
        num_delay_bins: 입력 gcc_phat의 lag축 크기 G (GRU 입력 크기 결정에 필요)
        metadata_dim: pairwise metadata (v_i, v_j) 차원
        gru_layers: unidirectional GRU 레이어 수
        hidden_size: GRU/MLP의 공통 hidden 크기
        aggregation: sum은 기존 단순 합산, cwsa는 channel-wise
            softmax 가중합과 가중 표준편차를 사용한다.
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
        aggregation: str = "sum",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if aggregation not in ("sum", "cwsa"):
            raise ValueError(
                f"aggregation must be 'sum' or 'cwsa', got {aggregation!r}."
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
        hidden = hidden.reshape(batch_size, num_pairs, num_output_frames, -1)
        if self.aggregation == "cwsa":
            hidden = channelwise_softmax_aggregation(hidden)
        else:
            hidden = hidden.sum(dim=1)
        # 어느 집계든 pair 축에 대해 대칭이므로 pair 순서에 불변

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
