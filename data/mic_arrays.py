from __future__ import annotations

import numpy as np  # 수치 계산 및 배열 연산용

# 상용 NAO 로봇 헤드 (4개 마이크). 좌표는 중심화되어 모든 고정 배열이
# 논문에서 사용한 "array-center" 관례를 공유하도록 설정됨
# 배열 형태: 직사각형 모양 (X: ±48mm, Y: ±36mm, Z: 30mm)
NAO_4CH = np.array(  # NAO 로봇 헤드의 4개 마이크 위치 (단위: m)
    [
        [0.048, 0.036, 0.030],    # 마이크1 (앞위쪽): X=48mm, Y=36mm, Z=30mm
        [0.048, -0.036, 0.030],   # 마이크2 (앞아래쪽): X=48mm, Y=-36mm, Z=30mm
        [-0.048, 0.036, 0.030],   # 마이크3 (뒤위쪽): X=-48mm, Y=36mm, Z=30mm
        [-0.048, -0.036, 0.030],  # 마이크4 (뒤아래쪽): X=-48mm, Y=-36mm, Z=30mm
    ],
    dtype=np.float32,
)
# 배열의 중심을 원점으로 정규화
NAO_4CH -= NAO_4CH.mean(axis=0, keepdims=True)  # 각 좌표에서 무게중심(평균)을 빼서 원점 중심화

# LOCATA / EARS 로봇 헤드 (12개 마이크)
# Source: LOCATA final-release 문서, Table 3
NAO_ROBOT_12CH = np.array(  # LOCATA/EARS 로봇 헤드의 12개 마이크 위치 (단위: m)
    [
        [-0.028, 0.030, -0.040],  # 마이크1
        [0.006, 0.057, 0.000],    # 마이크2
        [0.022, 0.022, -0.046],   # 마이크3
        [-0.055, -0.024, -0.025], # 마이크4
        [-0.031, 0.023, 0.042],   # 마이크5
        [-0.032, 0.011, 0.046],   # 마이크6
        [-0.025, -0.003, 0.051],  # 마이크7
        [-0.036, -0.027, 0.038],  # 마이크8
        [-0.035, -0.043, 0.025],  # 마이크9
        [0.029, -0.048, -0.012],  # 마이크10
        [0.034, -0.030, 0.037],   # 마이크11
        [0.035, 0.025, 0.039],    # 마이크12
    ],
    dtype=np.float32,
)
# 배열의 중심을 원점으로 정규화
NAO_ROBOT_12CH -= NAO_ROBOT_12CH.mean(axis=0, keepdims=True)  # 각 좌표에서 무게중심(평균)을 빼서 원점 중심화

# 한 변 4cm짜리 정사면체, 원점 중심 (AGG-RL 논문 stage 1 사용)
# AGG-RL(ICLR 2026) Table 6은 MSGL stage 1에서 GI-DOAEnet의 ReSpeaker 대신
# "Tetrahedron (4 cm)" 배열을 사용함. 기준 꼭짓점들의 pairwise 거리는 2*sqrt(2)이므로
# 0.04/(2*sqrt(2))를 곱하면 한 변이 4cm가 됨
TETRAHEDRON_4CM = np.array(  # 정사면체의 4개 꼭짓점 (정규화 전 좌표)
    [
        [1.0, 1.0, 1.0],        # 꼭짓점1
        [1.0, -1.0, -1.0],      # 꼭짓점2
        [-1.0, 1.0, -1.0],      # 꼭짓점3
        [-1.0, -1.0, 1.0],      # 꼭짓점4
    ],
    dtype=np.float32,
) * (0.04 / (2.0 * np.sqrt(2.0)))  # 한 변이 4cm가 되도록 스케일 조정 (0.04m / 2*sqrt(2))
TETRAHEDRON_4CM -= TETRAHEDRON_4CM.mean(axis=0, keepdims=True)  # 중심을 원점으로 정규화

# 동적으로 생성된 마이크 배열에 작은 무작위 jitter 추가
# 배열 위치에 현실적인 약간의 불확실성/노이즈 반영
# 데이터 증강: 모델이 정확한 좌표뿐만 아니라 약간의 오차도 견딜 수 있도록 학습
RORIGIN_CM = (-0.5, 0.5)  # jitter 범위 (cm 단위): -0.5cm ~ +0.5cm (=±0.005m)

C_MIN = 4   # 마이크 최소 채널 수
C_MAX = 12  # 마이크 최대 채널 수


def get_fixed_array(name: str) -> np.ndarray:  # 고정 배열 이름으로 마이크 좌표 배열 조회
    arrays = {  # 지원하는 고정 배열 종류
        "tetrahedron": TETRAHEDRON_4CM,  # AGG-RL stage1 사용
        "nao4": NAO_4CH,                  # NAO 4채널 (stage2)
        "nao12": NAO_ROBOT_12CH,          # LOCATA/EARS 12채널 (stage3)
    }
    try:
        return arrays[name].copy()  # 해당 배열 반환 (copy로 원본 보호)
    except KeyError as exc:  # 지원하지 않는 배열 이름
        raise ValueError(f"Unknown fixed array '{name}'.") from exc


def pairwise_distance_bounds_cm(num_channels: int) -> tuple[float, float]:  # 동적 배열의 마이크 간 거리 범위 (cm)
    if not (C_MIN <= num_channels <= C_MAX):  # 채널 수 범위 검증 (4~12)
        raise ValueError(
            f"num_channels must be within [{C_MIN}, {C_MAX}], got {num_channels}."
        )
    # 논문 Appendix A.10
    ratio = (num_channels - C_MIN) / (C_MAX - C_MIN)  # 정규화 비율 (4채널: 0, 12채널: 1)
    r_min = np.random.uniform(max(1.0, 4.0 - 3.0 * ratio), 6.0)  # 최소거리 범위: 채널 증가시 하한 감소 (1cm~6cm)
    r_max = np.random.uniform(7.0, max(7.0, 9.0 + 4.0 * ratio))  # 최대거리 범위: 채널 증가시 상한 증가 (7cm~13cm)
    return float(r_min), float(r_max)  # cm 단위의 (최소거리, 최대거리) 튜플 반환


def _pairwise_distance_bounds_m(
    num_channels: int, rng: np.random.Generator  # rng: 재현 가능한 난수 생성기
) -> tuple[float, float]:  # m 단위의 거리 범위 반환
    ratio = (num_channels - C_MIN) / (C_MAX - C_MIN)  # 정규화: 4ch→0, 12ch→1
    r_min = rng.uniform(max(1.0, 4.0 - 3.0 * ratio), 6.0) / 100.0  # 1~6cm를 m로 변환 (/100)
    r_max = rng.uniform(7.0, max(7.0, 9.0 + 4.0 * ratio)) / 100.0  # 7~13cm를 m로 변환 (/100)
    # 채널이 많을수록 마이크 배치가 넓어짐: r_min↓, r_max↑ (더 먼 거리에서도 배치 가능)
    return float(r_min), float(r_max)  # m 단위의 (최소거리, 최대거리) 튜플


# 동적 배열 마이크를 3D 구에 균등하게 배치하기 위한 무작위 단위 벡터 생성
def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:  # 길이 1인 무작위 3D 벡터 반환
    vec = rng.normal(size=3)  # 표준정규분포 N(0,1)에서 3개 난수 샘플링 → [x, y, z]
    norm = np.linalg.norm(vec) + 1e-12  # 벡터의 L2 norm 계산 (1e-12로 0 방지)
    return (vec / norm).astype(np.float32)  # 벡터 정규화: 길이가 1인 단위벡터로 변환


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:  # 무작위 3×3 회전 행렬 생성 (균일 분포)
    u1, u2, u3 = rng.random(3)  # [0, 1) 범위의 균일 난수 3개 생성
    # Quaternion (4개 성분: x, y, z, w) 생성: 회전을 나타내는 수학적 표현
    q = np.array(  # 균일분포의 Quaternion (3D 회전 공간에서 균등하게 샘플링)
        [
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),  # q_x
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),  # q_y
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),  # q_z
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),  # q_w (스칼라부)
        ],
        dtype=np.float32,
    )
    x, y, z, w = q  # Quaternion 성분 분해
    # Quaternion을 3×3 회전 행렬로 변환 (수학 공식)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],  # 첫 번째 행
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],  # 두 번째 행
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],  # 세 번째 행
        ],
        dtype=np.float32,
    )  # 회전 행렬은 직교행렬(orthogonal): R^T * R = I


def random_rotate(coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:  # 마이크 배열 좌표에 무작위 회전 적용
    rotation = random_rotation_matrix(rng)  # 3×3 회전 행렬 생성
    return (coords @ rotation.T).astype(np.float32)  # 좌표 × 회전행렬의 전치 = 회전된 좌표


# 동적으로 생성된 마이크 배열(4~12개)을 3D 공간에 배치
def sample_dynamic_array(num_channels: int,
                         rng: np.random.Generator | None = None,  # 재현 가능한 난수 생성기
                         max_attempts: int = 4096) -> np.ndarray:  # 배치된 마이크 좌표 (num_channels, 3)
    if rng is None:  # 난수 생성기가 없으면
        rng = np.random.default_rng()  # 새로 생성

    if num_channels < 2:  # 최소 2개 마이크 필요
        raise ValueError("num_channels must be at least 2.")

    r_min, r_max = _pairwise_distance_bounds_m(num_channels, rng)  # 채널 수에 맞는 거리 범위 결정
    max_radius = 0.5 * r_max  # 배치할 구의 최대 반지름 = 최대거리 / 2

    positions: list[np.ndarray] = []  # 배치된 마이크 위치 저장
    for _ in range(num_channels):  # 각 마이크마다 위치 결정
        placed = False  # 현재 마이크 배치 성공 여부
        for _ in range(max_attempts):  # 최대 4096번 시도
            # 무작위 방향의 단위벡터 × 반지름 = 구 내부의 무작위 점
            candidate = _random_unit_vector(rng) * rng.uniform(0.0, max_radius)

            if not positions:  # 첫 번째 마이크
                positions.append(candidate.astype(np.float32))  # 제약 없이 바로 배치
                placed = True
                break  # 첫 마이크는 거리 제약이 없으므로 곧바로 확정하고 max_attempts 루프 종료 -> 다음 마이크 배치

            # 기존 마이크들과의 거리 계산
            dists = np.linalg.norm(np.stack(positions, axis=0) - candidate, axis=1)
            # 모든 기존 마이크와 거리가 [r_min, r_max] 범위 내인지 확인
            if np.all(dists >= r_min) and np.all(dists <= r_max):
                positions.append(candidate.astype(np.float32))  # 조건 만족하면 배치
                placed = True
                break  # 거리 제약 통과하면 확정하고 max_attempts 루프 종료 -> 다음 마이크 배치

        if placed:  # 조건에 맞게 잘 배치됐으면
            continue  # 다음 마이크로

        # 4096번 시도해도 배치 못한 경우 → fallback: 원 위에 배치 (Z=0)
        ring_radius = min(  # 원의 반지름 (구의 반지름보다 작거나 거리 범위 고려)
            max_radius,  # 최대 반지름
            # 2*R*sin(pi/N)= r_min: 이웃 간 최소 거리 -> R = r_min/2sin(pi/N): 마이크 간격이 딱 r_min 되는 반지름
            # max(): 이론 반지름을 쓰되, 너무 작으면(N이 작을 때) 0.5*r_min - 하한 적용
            max(r_min / (2.0 * np.sin(np.pi / num_channels) + 1e-6), 0.5 * r_min),
        )  # 상한을 max_radius로 제한

        # 원 위의 각도 (균등 분배)
        angle = 2.0 * np.pi * len(positions) / num_channels  # len(positions): 지금까지 놓인 마이크 수
        fallback = np.array(  # 원 위의 점 (Z=0은 수평면)
            [
                ring_radius * np.cos(angle),  # X 좌표
                ring_radius * np.sin(angle),  # Y 좌표
                0.0,                          # Z 좌표 (수평면)
            ],
            dtype=np.float32,
        )
        positions.append(fallback)

    coords = np.stack(positions, axis=0)  # (num_channels, 3)로 쌓기
    coords -= coords.mean(axis=0, keepdims=True)  # 1단계 중심화: 무게중심을 원점으로
    # jitter 추가 (±0.5cm). cm → m로 변환
    coords += rng.uniform(RORIGIN_CM[0], RORIGIN_CM[1], size=coords.shape).astype(np.float32) / 100.0
    coords -= coords.mean(axis=0, keepdims=True)  # 2단계 중심화: jitter 후 다시 중심화
    return coords.astype(np.float32)  # 배치된 마이크 배열 반환


def _sample_dynamic_array_candidates(
    rng: np.random.Generator,
    count: int,
    max_radius: float,
) -> np.ndarray:
    """동적 배열 후보 좌표를 한 번에 생성.

    후보 방향과 반지름의 확률분포는 기존 scalar sampler와 동일하고,
    반환 shape은 (count, 3).
    """
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    radii = rng.uniform(0.0, max_radius, size=(count, 1))
    return (directions * radii).astype(np.float32)


def sample_dynamic_array_vectorized(
    num_channels: int,
    rng: np.random.Generator | None = None,
    max_attempts: int = 4096,
    chunk_size: int = 256,
) -> np.ndarray:
    """기존 거리 분포를 유지하는 벡터화 동적 마이크 배열 생성.

    후보 하나마다 Python loop와 ``np.stack``을 수행하던 기존 구현 대신,
    후보를 ``chunk_size``개씩 생성해 모든 기존 마이크와의 거리를 한 번에
    계산. 각 chunk에서 첫 번째 유효 후보를 선택하고 최대 시도 횟수와
    fallback, 중심화, jitter는 기존 구현과 동일하게 유지.
    """
    if rng is None:
        rng = np.random.default_rng()
    if num_channels < 2:
        raise ValueError("num_channels must be at least 2.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    r_min, r_max = _pairwise_distance_bounds_m(num_channels, rng)
    max_radius = 0.5 * r_max

    # 반복적인 list -> np.stack 변환 제거를 위한 고정 크기 좌표 버퍼
    positions = np.empty((num_channels, 3), dtype=np.float32)
    positions[0] = _sample_dynamic_array_candidates(rng, 1, max_radius)[0]
    num_placed = 1

    while num_placed < num_channels:
        accepted_candidate: np.ndarray | None = None
        attempts_used = 0

        while attempts_used < max_attempts:
            current_chunk = min(chunk_size, max_attempts - attempts_used)
            candidates = _sample_dynamic_array_candidates(
                rng,
                current_chunk,
                max_radius,
            )  # (candidate, 3)

            # (candidate, 1, 3) - (1, placed_mic, 3)
            # -> 후보별 기존 모든 마이크와의 거리 (candidate, placed_mic)
            differences = (
                candidates[:, None, :]
                - positions[None, :num_placed, :]
            )
            distances = np.linalg.norm(differences, axis=2)
            valid = np.all(
                (distances >= r_min) & (distances <= r_max),
                axis=1,
            )
            valid_indices = np.flatnonzero(valid)

            if valid_indices.size > 0:
                # scalar rejection sampling과 동일하게 첫 유효 후보 선택
                accepted_candidate = candidates[int(valid_indices[0])]
                break

            attempts_used += current_chunk

        if accepted_candidate is None:
            # 최대 시도 횟수 소진 시 기존 원형 fallback 유지
            ring_radius = min(
                max_radius,
                max(
                    r_min
                    / (2.0 * np.sin(np.pi / num_channels) + 1e-6),
                    0.5 * r_min,
                ),
            )
            angle = 2.0 * np.pi * num_placed / num_channels
            accepted_candidate = np.array(
                [
                    ring_radius * np.cos(angle),
                    ring_radius * np.sin(angle),
                    0.0,
                ],
                dtype=np.float32,
            )

        positions[num_placed] = accepted_candidate
        num_placed += 1

    # 기존 sampler와 동일한 중심화 -> jitter -> 재중심화 순서
    positions -= positions.mean(axis=0, keepdims=True)
    positions += rng.uniform(
        RORIGIN_CM[0],
        RORIGIN_CM[1],
        size=positions.shape,
    ).astype(np.float32) / 100.0
    positions -= positions.mean(axis=0, keepdims=True)
    return positions.astype(np.float32)
