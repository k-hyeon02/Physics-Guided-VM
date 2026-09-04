"""
Physics-guided variational model 학습

파이프라인:
    SyntheticDOADataset -> GCCPHATProcess(frontend) -> mic_position_metadata
        -> VariationalDOAEncoder -> sample_von_mises_fisher -> physics_based_decoder
        -> physics_loss + beta * von_mises_fisher_kl_loss -> elbo_doa_loss

논문에서 확인한 학습 설정 :
    - 300 epoch, batch size 16
    - learning rate는 5e-4 -> 5e-5로 epoch에 대해 지수 감소
    - beta(KL 가중치)는 처음 5% epoch은 0, 이후 1.0 (posterior collapse 방지 warm-up)
    - lambda(입력 delay 분포 softmax scale) = 8.0
    - sigma(decoder Gaussian 표준편차)는 학습 가능한 스칼라, softplus로 양수 보장
    - G=64 delay bin, STFT window 4096 samples, hop rate 0.75
"""
import argparse
import csv
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from data.dataset import PROFILE_SPECS, SyntheticDOADataset, build_dataloader
from data.simulate import SimulationConfig
from data.static import StaticSyntheticDOADataset, StaticSimulationConfig
from data.streaming_pipeline import build_streaming_dataloader
from input_process import GCCPHATProcess, pair_displacement
from mic_metadata import mic_position_metadata
from model.encoder import VariationalDOAEncoder
from model.decoder import physics_based_decoder
from model.reparam import sample_von_mises_fisher
from model.loss import (
    von_mises_fisher_kl_loss,
    interpolate_time_axis,
    normalize_gcc_phat,     
    input_delay_distribution,
    physics_loss,
    elbo_doa_loss
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_VARIANT = "sum"

def inverse_softplus(value: float) -> float:
    """softplus(x) = value가 되는 x를 계산 (sigma 초기값 설정용)"""
    return math.log(math.expm1(value))

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Training Physics-based variational model (SUM branch)"
    )

    # 데이터 경로
    parser.add_argument(
        "--train-librispeech-root",
        default=os.path.join(PROJECT_ROOT, "datasets", "librispeech", "LibriSpeech", "train-clean-100")
    )
    parser.add_argument(
        "--train-ms-snsd-root",
        default=os.path.join(PROJECT_ROOT, "datasets", "ms-snsd", "MS-SNSD", "noise_train")
    )
    parser.add_argument(
        "--val-librispeech-root",
        default=os.path.join(PROJECT_ROOT, "datasets", "librispeech", "LibriSpeech", "test-clean")
    )
    parser.add_argument(
        "--val-ms-snsd-root",
        default=os.path.join(PROJECT_ROOT, "datasets", "ms-snsd", "MS-SNSD", "noise_test")
    )
    parser.add_argument("--train-num-samples", type=int, default=None)
    parser.add_argument("--val-num-samples", type=int, default=None)
    parser.add_argument(
        "--train-profile",
        default=None,
        choices=sorted(PROFILE_SPECS),
        help="지정하면 stage curriculum 대신 이 배열 profile을 전 epoch에 고정",
    )
    parser.add_argument(
        "--val-profile", default="stage3", choices=sorted(PROFILE_SPECS)
    )
    parser.add_argument(
        "--no-rotate-arrays",
        action="store_true",
        help="훈련·검증 배열의 무작위 3D 회전을 비활성화",
    )
    parser.add_argument(
        "--noise-mode",
        default="mixed",
        choices=["mixed", "awgn"],
        help="moving-source 잡음: mixed=기존 MS-SNSD+백색, awgn=원논문 Experiment 1",
    )
    parser.add_argument(
        "--awgn-power-reference",
        default="direct_path",
        choices=["auralized", "direct_path"],
        help=(
            "AWGN SNR 파워 기준(기본 direct_path): "
            "auralized=잔향 포함 마이크 신호, "
            "direct_path=Neural-SRP 공개 시뮬레이터의 직접경로 신호"
        ),
    )
    parser.add_argument(
        "--static", action="store_true",
        help="이동 음원(SyntheticDOADataset) 대신 정적 단일 음원(StaticSyntheticDOADataset)으로 학습"
    )

    # 채널 수 커리큘럼
    parser.add_argument("--stage1-end-epoch", type=int, default=10)
    parser.add_argument("--stage2-end-epoch", type=int, default=20)

    # 학습 하이퍼파라미터
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-start", type=float, default=5e-4)
    parser.add_argument("--lr-end", type=float, default=5e-5)
    parser.add_argument("--beta-warmup-fraction", type=float, default=0.05)
    parser.add_argument(
        "--beta-after-warmup",
        type=float,
        default=1.0,
        help="warm-up 종료 후 KL 가중치 beta (기본값: 논문 설정 1.0)",
    )
    parser.add_argument(
        "--physics-pair-reduction",
        default=None,
        choices=["sum", "mean"],
        help=(
            "physics loss의 pair 축 reduction. 지정하지 않으면 "
            "SUM encoder는 sum, CWSA encoder는 mean을 사용"
        ),
    )
    parser.add_argument("--lambda-scale", type=float, default=8.0)
    parser.add_argument("--sigma-init", type=float, default=1.0)
    parser.add_argument(
        "--fixed-sigma",
        type=float,
        default=None,
        help="지정하면 decoder sigma를 이 값으로 고정하고 optimizer에서 제외",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=0.0, help="0이면 비활성화")

    # frontend / encoder 구성
    parser.add_argument("--win-length", type=int, default=4096)
    parser.add_argument("--hop-length", type=int, default=None, help="None이면 win_length*0.75 사용")
    parser.add_argument("--fft-length", type=int, default=4096)
    parser.add_argument("--num-delay-bins", type=int, default=64)
    parser.add_argument(
        "--aggregation",
        default=MODEL_VARIANT,
        choices=[MODEL_VARIANT],
        help="encoder pair 집계 (이 브랜치는 기존 논문 방식인 sum으로 고정)",
    )
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--speed-of-sound", type=float, default=343.0)

    # 실행/로깅
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--streaming-simulation",
        action="store_true",
        help=(
            "CPU simulation 준비 worker와 단일 gpuRIR renderer를 분리한 "
            "streaming pipeline 사용"
        ),
    )
    parser.add_argument(
        "--cpu-prep-workers",
        type=int,
        default=None,
        help=(
            "streaming pipeline의 CPU 준비 worker 수 "
            "(미지정 시 --num-workers 사용)"
        ),
    )
    parser.add_argument(
        "--simulation-prefetch-batches",
        type=int,
        default=2,
        help="streaming pipeline이 미리 준비할 batch 수",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint-dir",
        default=os.path.join(PROJECT_ROOT, "checkpoints", MODEL_VARIANT),
    )
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=50, help="배치 단위 콘솔 로그 주기")
    parser.add_argument(
        "--log-csv",
        default=os.path.join(
            PROJECT_ROOT, "checkpoints", MODEL_VARIANT, "train_log.csv"
        ),
    )
    parser.add_argument("--resume", default=None, help="이어서 학습할 checkpoint 경로")

    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA를 사용할 수 없어 CPU로 대체")
        return torch.device("cpu")
    return torch.device(requested)


def profile_for_epoch(epoch: int, stage1_end: int, stage2_end: int) -> str:
    """Gradual Curriculum Learning: stage1 -> stage2 -> stage3."""

    if epoch <= stage1_end:
        return "stage1"
    if epoch <= stage2_end:
        return "stage2"
    return "stage3"


def beta_for_epoch(
    epoch: int,
    total_epochs: int,
    warmup_fraction: float,
    beta_after_warmup: float = 1.0,
) -> float:
    """Eq.25의 beta: 처음 warmup_fraction만큼은 0, 이후 지정한 값."""

    warmup_epochs = round(total_epochs * warmup_fraction)
    return 0.0 if epoch <= warmup_epochs else beta_after_warmup


def build_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[GCCPHATProcess, VariationalDOAEncoder, nn.Parameter]:
    """
    학습에 필요한 3가지 구성요소(frontend, encoder, sigma)를 한 번에 만들어서 리턴
    """

    frontend = GCCPHATProcess(
        win_length=args.win_length,
        hop_length=args.hop_length,
        fft_length=args.fft_length,
        sample_rate=args.sample_rate,
        num_delay_bins=args.num_delay_bins,
        speed_of_sound=args.speed_of_sound,
    ).to(device)

    encoder = VariationalDOAEncoder(
        num_delay_bins=args.num_delay_bins,
        aggregation=args.aggregation,
    ).to(device)

    sigma_value = args.fixed_sigma if args.fixed_sigma is not None else args.sigma_init
    if sigma_value <= 0.0:
        raise ValueError("sigma는 0보다 커야 합니다")

    # 기본값은 논문 5.3절대로 학습 가능한 스칼라다. --fixed-sigma를 지정한
    # 대조 실험에서는 같은 표현을 유지하되 optimizer에서 제외한다.
    raw_sigma = nn.Parameter(
        torch.tensor(inverse_softplus(sigma_value), device=device),
        requires_grad=args.fixed_sigma is None,
    )

    return frontend, encoder, raw_sigma


def move_batch_to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        "input_audio": batch["input_audio"].to(device),
        "mic_coordinate": batch["mic_coordinate"].to(device),
        "vad": batch["vad"].to(device)
    }


def forward_losses(
    batch: dict[str, Tensor],
    frontend: GCCPHATProcess,
    encoder: VariationalDOAEncoder,
    raw_sigma: nn.Parameter,
    lambda_scale: float,
    beta: float,
    physics_pair_reduction: str | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    """
    frontend -> encoder -> reparam -> decoder -> physics + KL loss

    Returns:
        (phy_loss, kl_loss, kappa, sigma, num_pairs)
    """

    audio = batch["input_audio"]
    mic_coordinate = batch["mic_coordinate"]

    # frontend - gcc_phat: (B,K,T,G) | delay_bins: (B,G) | pairs: (K,2) | metadata: (B,K,6)
    gcc_phat, delay_bins, pairs = frontend(audio, mic_coordinate)
    metadata = mic_position_metadata(mic_coordinate, pairs)
    # encoder - mu: (B,T',3) | kappa: (B,T',1)
    mu, kappa = encoder(gcc_phat, metadata)
    # reparameterization - z: (B,T′,3)  |  warm-up 중엔 샘플링하지 않음
    z = mu if beta == 0.0 else sample_von_mises_fisher(mu, kappa)
    # decoder - displacement: (B,K,D) | p_pred: (B,K,T',G)
    displacement = pair_displacement(mic_coordinate, pairs)
    sigma = F.softplus(raw_sigma)
    p_pred = physics_based_decoder(
        z, displacement, delay_bins, 
        speed_of_sound=frontend.speed_of_sound, 
        sample_rate=frontend.sample_rate, 
        sigma=sigma
        )

    latent_frames = mu.shape[1]  # T'
    g_tilde = normalize_gcc_phat(interpolate_time_axis(gcc_phat, target_length=latent_frames))
    p_target = input_delay_distribution(g_tilde, lambda_scale)

    activity_mask = frontend.STFT.get_vad_frame(batch["vad"]).squeeze(1).float()  # (B,T)
    activity_mask = interpolate_time_axis(activity_mask, target_length=latent_frames, time_dim=-1)

    # Loss
    num_pairs = pairs.shape[0]
    if physics_pair_reduction is None:
        physics_pair_reduction = (
            "mean" if encoder.aggregation == "cwsa" else "sum"
        )
    phy_loss = physics_loss(
        p_target,
        p_pred,
        activity_mask,
        pair_reduction=physics_pair_reduction,
    )
    kl_loss = von_mises_fisher_kl_loss(kappa)
    return phy_loss, kl_loss, kappa, sigma, num_pairs


def train_1epoch(
    loader,
    frontend: GCCPHATProcess,
    encoder: VariationalDOAEncoder,
    raw_sigma: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    beta: float,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int
) -> dict[str, float]:
    encoder.train()
    totals = {"loss": 0.0, "phy": 0.0, "kl": 0.0, "kappa": 0.0}
    num_batches = 0

    for step, raw_batch in enumerate(loader):
        batch = move_batch_to_device(raw_batch, device)

        phy_loss, kl_loss, kappa, sigma, num_pairs = forward_losses(
            batch,
            frontend,
            encoder,
            raw_sigma,
            args.lambda_scale,
            beta,
            physics_pair_reduction=args.physics_pair_reduction,
        )
        average_pair_loss = (
            args.physics_pair_reduction == "mean"
            or (
                args.physics_pair_reduction is None
                and encoder.aggregation == "cwsa"
            )
        )
        normalize_beta_by_pairs = (
            encoder.aggregation == "cwsa" and average_pair_loss
        )
        beta_num_pairs = num_pairs if normalize_beta_by_pairs else None
        effective_beta = beta if beta_num_pairs is None else beta / beta_num_pairs
        loss = elbo_doa_loss(
            phy_loss,
            kl_loss,
            beta,
            num_pairs=beta_num_pairs,
        )

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                optimizer.param_groups[0]["params"], args.grad_clip_norm
            )
        optimizer.step()

        totals["loss"] += loss.item()
        totals["phy"] += phy_loss.item()
        totals["kl"] += kl_loss.mean().item()
        totals["kappa"] += kappa.mean().item()
        num_batches += 1

        if step % args.log_every == 0:
            phy_label = "phy/pair" if average_pair_loss else "phy"
            beta_label = "beta_eff" if normalize_beta_by_pairs else "beta"
            print(
                f"[epoch {epoch}][step {step}/{len(loader)}] "
                f"loss={loss.item():.4f}  |  "
                f"{phy_label}={phy_loss.item():.4f}  |  kl={kl_loss.mean().item():.4f}  |  "
                f"sigma={sigma.item():.4f}  |  K={num_pairs}  |  "
                f"{beta_label}={effective_beta:.6f}"
            )

    num_batches = max(num_batches, 1)
    return {key: value / num_batches for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    loader,
    frontend: GCCPHATProcess,
    encoder: VariationalDOAEncoder,
    raw_sigma: nn.Parameter,
    args: argparse.Namespace,
    device: torch.device
) -> dict[str, float]:

    encoder.eval()
    totals = {"phy": 0.0, "kl": 0.0}
    num_batches = 0

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        phy_loss, kl_loss, _, _, _ = forward_losses(
            batch,
            frontend,
            encoder,
            raw_sigma,
            args.lambda_scale,
            beta=args.beta_after_warmup,
            physics_pair_reduction=args.physics_pair_reduction,
        )
        totals["phy"] += phy_loss.item()
        totals["kl"] += kl_loss.mean().item()
        num_batches += 1

    num_batches = max(num_batches, 1)
    return {key: value / num_batches for key, value in totals.items()}


def save_checkpoint(
    path: str,
    epoch: int,
    encoder: VariationalDOAEncoder,
    raw_sigma: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "aggregation": encoder.aggregation,
            "encoder": encoder.state_dict(),
            "raw_sigma": raw_sigma.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        path
    )


def load_checkpoint(
    path: str,
    encoder: VariationalDOAEncoder,
    raw_sigma: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device
) -> int:
    # 학습 코드가 직접 저장한 RNG/optimizer 상태까지 복원해야 하므로
    # PyTorch 2.6+의 weights_only 기본값에 의존하지 않는다.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    checkpoint_aggregation = checkpoint.get("aggregation")
    if (
        checkpoint_aggregation is not None
        and checkpoint_aggregation != encoder.aggregation
    ):
        raise ValueError(
            "checkpoint aggregation mismatch: "
            f"checkpoint={checkpoint_aggregation!r}, model={encoder.aggregation!r}"
        )
    encoder.load_state_dict(checkpoint["encoder"])
    with torch.no_grad():
        raw_sigma.copy_(checkpoint["raw_sigma"].to(device))
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state_all])
    return checkpoint["epoch"] + 1


def append_log_row(csv_path: str, row: dict[str, float | int | str]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_training_loaders(args, train_dataset, val_dataset):
    """선택한 simulation 방식으로 train/validation loader를 생성한다."""

    if not args.streaming_simulation:
        print(f"data loader: standard (workers={args.num_workers})")
        train_loader = build_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=True,
        )
        val_loader = build_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return train_loader, val_loader

    if (
        type(train_dataset) is not SyntheticDOADataset
        or type(val_dataset) is not SyntheticDOADataset
    ):
        raise ValueError(
            "--streaming-simulation은 moving SyntheticDOADataset만 지원합니다"
        )

    num_prepare_workers = (
        args.num_workers
        if args.cpu_prep_workers is None
        else args.cpu_prep_workers
    )
    if num_prepare_workers < 1:
        raise ValueError("streaming simulation의 CPU 준비 worker 수는 1 이상이어야 합니다")
    if args.simulation_prefetch_batches < 1:
        raise ValueError("--simulation-prefetch-batches는 1 이상이어야 합니다")

    print(
        "data loader: streaming "
        f"(CPU prepare workers={num_prepare_workers}, gpuRIR renderers=1, "
        f"prefetch batches={args.simulation_prefetch_batches})"
    )
    train_loader = build_streaming_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_prepare_workers=num_prepare_workers,
        shuffle=True,
        prefetch_batches=args.simulation_prefetch_batches,
        drop_last=True,
    )
    val_loader = build_streaming_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        num_prepare_workers=num_prepare_workers,
        shuffle=False,
        prefetch_batches=args.simulation_prefetch_batches,
        drop_last=False,
    )
    return train_loader, val_loader


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)

    dataset_cls = StaticSyntheticDOADataset if args.static else SyntheticDOADataset
    simulation_config = (
        StaticSimulationConfig(sample_rate=args.sample_rate)
        if args.static
        else SimulationConfig(
            sample_rate=args.sample_rate,
            noise_mode=args.noise_mode,
            awgn_power_reference=args.awgn_power_reference,
        )
    )

    initial_train_profile = args.train_profile or "stage1"
    rotate_arrays = not args.no_rotate_arrays

    train_dataset = dataset_cls(
        librispeech_root=args.train_librispeech_root,
        ms_snsd_root=args.train_ms_snsd_root,
        num_samples=args.train_num_samples,
        profile=initial_train_profile,
        batch_size=args.batch_size,
        seed=args.seed,
        simulation_config=simulation_config,
        rotate_arrays=rotate_arrays,
    )
    val_dataset = dataset_cls(
        librispeech_root=args.val_librispeech_root,
        ms_snsd_root=args.val_ms_snsd_root,
        num_samples=args.val_num_samples,
        profile=args.val_profile,
        batch_size=args.batch_size,
        seed=args.seed,
        simulation_config=simulation_config,
        rotate_arrays=rotate_arrays,
    )

    train_loader, val_loader = build_training_loaders(
        args, train_dataset, val_dataset
    )

    frontend, encoder, raw_sigma = build_model(args, device)
    optimizer_parameters = list(encoder.parameters())
    if raw_sigma.requires_grad:
        optimizer_parameters.append(raw_sigma)
    optimizer = torch.optim.Adam(optimizer_parameters, lr=args.lr_start)
    # gamma: 매 epoch마다 현재 lr에 곱해지는 감쇠 비율
    gamma = (args.lr_end / args.lr_start) ** (1.0 / args.epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    start_epoch = 1
    if args.resume is not None:
        start_epoch = load_checkpoint(args.resume, encoder, raw_sigma, optimizer, scheduler, device)
        print(f"resume checkpoint: {args.resume} (epoch {start_epoch}부터)")

    for epoch in range(start_epoch, args.epochs + 1):
        profile = (
            args.train_profile
            if args.train_profile is not None
            else profile_for_epoch(epoch, args.stage1_end_epoch, args.stage2_end_epoch)
        )
        train_dataset.set_epoch(epoch)
        train_dataset.set_profile(profile)
        beta = beta_for_epoch(
            epoch,
            args.epochs,
            args.beta_warmup_fraction,
            args.beta_after_warmup,
        )

        train_stats = train_1epoch(
            train_loader, frontend, encoder, raw_sigma, optimizer, beta, args, device, epoch
            )
        scheduler.step()

        val_stats = {"phy": "", "kl": ""}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_stats = evaluate(val_loader, frontend, encoder, raw_sigma, args, device)
            print(f"  val: phy={val_stats['phy']:.4f}  |  kl={val_stats['kl']:.4f}")

        log_row = {
            "epoch": epoch,
            "aggregation": encoder.aggregation,
            "profile": profile,
            "beta": beta,
            "lr": optimizer.param_groups[0]["lr"],
            "sigma": F.softplus(raw_sigma).item(),
            "train_loss": train_stats["loss"],
            "train_phy": train_stats["phy"],
            "train_kl": train_stats["kl"],
            "train_kappa": train_stats["kappa"],
            "val_phy": val_stats["phy"],
            "val_kl": val_stats["kl"],
        }
        append_log_row(args.log_csv, log_row)

        print(
            f"epoch {epoch}/{args.epochs} [{profile}]  "
            f"loss={train_stats['loss']:.4f}  |  "
            f"phy={train_stats['phy']:.4f}  |  kl={train_stats['kl']:.4f}  |  "
            f"kappa={train_stats['kappa']:.2f}  |  beta={beta:.6f}  |  lr={log_row['lr']:.2e}"
        )

        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch:04d}.pt")
            save_checkpoint(ckpt_path, epoch, encoder, raw_sigma, optimizer, scheduler)
            last_path = os.path.join(args.checkpoint_dir, "last.pt")
            save_checkpoint(last_path, epoch, encoder, raw_sigma, optimizer, scheduler)


if __name__ == "__main__":
    main()
