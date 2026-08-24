"""마이크 배열 좌표 메타데이터 처리."""

import torch
from torch import Tensor


def mic_position_metadata(
    microphone_coordinates: Tensor,
    pair_indices: Tensor,
) -> Tensor:
    """마이크쌍의 중심 기준 좌표 [v_i, v_j]를 생성

    Args:
        microphone_coordinates: 미터 단위 (B, M, 3) 또는 (M, 3) 좌표
        pair_indices: GCC-PHAT과 같은 순서의 마이크쌍 (K, 2)

    Returns:
        (B, K, 6) 형태의 mic position metadata
    """

    if microphone_coordinates.ndim == 2:  # Batch 차원 없으면 만듦
        microphone_coordinates = microphone_coordinates.unsqueeze(0)

    # 배열의 평행이동에는 불변이고 방향·크기 정보는 보존하도록 중심만 제거
    relative_coordinates = (
        microphone_coordinates
        - microphone_coordinates.mean(dim=1, keepdim=True)
    )
    pair_indices = pair_indices.to(device=microphone_coordinates.device)

    position_i = relative_coordinates[:, pair_indices[:, 0]]
    position_j = relative_coordinates[:, pair_indices[:, 1]]
    return torch.cat((position_i, position_j), dim=-1)  # (B, K, 6)