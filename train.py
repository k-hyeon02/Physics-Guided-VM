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

논문과 다른 점:
    - encoder의 pair 축 집계를 단순 합산에서 CWSA로 교체 (--aggregation, 기본 cwsa).
      합산은 출력이 마이크쌍 개수 K에 선형 비례해서 가변 마이크(K=6~66)에서
      배열 크기가 그대로 활성값·kappa 크기가 되어버린다.
      --aggregation sum으로 논문 원본 동작을 재현할 수 있다.
    - physics_loss의 pair 축을 합에서 평균으로 바꾸고, KL에 beta/K를 적용.
      ELBO 전체를 K로 나눈 것이라 gradient 방향과 최적점은 논문 그대로이고,
      손실 크기만 마이크쌍 개수에 불변해진다. 이제 train/val, stage 전후,
      채널 수가 다른 실험끼리 phy 값을 직접 비교할 수 있다.
"""
import argparse
import csv
import math
import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from data.dataset import SyntheticDOADataset, build_dataloader
from data.simulate import SimulationConfig
from data.static import StaticSyntheticDOADataset, StaticSimulationConfig
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

def inverse_softplus(value: float) -> float:
    """softplus(x) = value가 되는 x를 계산 (sigma 초기값 설정용)"""
    return math.log(math.expm1(value))

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training Physics-based variational model")

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
    parser.add_argument("--val-profile", default="stage3", choices=["stage1", "stage2", "stage3"])
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
    parser.add_argument("--lambda-scale", type=float, default=8.0)
    parser.add_argument("--sigma-init", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0, help="0이면 비활성화")

    # frontend / encoder 구성
    parser.add_argument("--win-length", type=int, default=4096)
    parser.add_argument("--hop-length", type=int, default=None, help="None이면 win_length*0.75 사용")
    parser.add_argument("--fft-length", type=int, default=4096)
    parser.add_argument("--num-delay-bins", type=int, default=64)
    parser.add_argument(
        "--aggregation", default="cwsa", choices=["cwsa", "sum"],
        help="encoder의 pair 축 집계 방식. cwsa=softmax 가중합+표준편차(K에 거의 불변), sum=논문 원본"
    )
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--speed-of-sound", type=float, default=343.0)

    # 실행/로깅
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", default=os.path.join(PROJECT_ROOT, "checkpoints"))
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=50, help="배치 단위 콘솔 로그 주기")
    parser.add_argument("--log-csv", default=os.path.join(PROJECT_ROOT, "checkpoints", "train_log.csv"))
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


def beta_for_epoch(epoch: int, total_epochs: int, warmup_fraction: float) -> float:
    """Eq.25의 beta: 처음 warmup_fraction만큼은 0, 이후 1.0 (posterior collapse 방지)."""

    warmup_epochs = round(total_epochs * warmup_fraction)
    return 0.0 if epoch <= warmup_epochs else 1.0


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

    # sigma는 논문 5.3절대로 학습 가능한 스칼라: softplus(raw_sigma)로 양수 보장 (decoder.py 참고)
    raw_sigma = nn.Parameter(
        torch.tensor(inverse_softplus(args.sigma_init), device=device)
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
    beta: float
) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    """
    frontend -> encoder -> reparam -> decoder -> physics + KL loss

    Returns:
        (phy_loss, kl_loss, kappa, sigma, num_pairs)
        num_pairs는 elbo_doa_loss가 beta/K를 적용하는 데 필요한 이 배치의 K다.
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
    phy_loss = physics_loss(p_target, p_pred, activity_mask)
    kl_loss = von_mises_fisher_kl_loss(kappa)
    return phy_loss, kl_loss, kappa, sigma, pairs.shape[0]


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
    totals = {"loss": 0.0, "phy": 0.0, "kl": 0.0, "kappa": 0.0, "pairs": 0.0}
    num_batches = 0

    for step, raw_batch in enumerate(loader):
        batch = move_batch_to_device(raw_batch, device)

        phy_loss, kl_loss, kappa, sigma, num_pairs = forward_losses(
            batch, frontend, encoder, raw_sigma, args.lambda_scale, beta
        )
        loss = elbo_doa_loss(phy_loss, kl_loss, beta, num_pairs)

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + [raw_sigma], args.grad_clip_norm
            )
        optimizer.step()

        totals["loss"] += loss.item()
        totals["phy"] += phy_loss.item()
        totals["kl"] += kl_loss.mean().item()
        totals["kappa"] += kappa.mean().item()
        totals["pairs"] += float(num_pairs)
        num_batches += 1

        if step % args.log_every == 0:
            print(
                f"[epoch {epoch}][step {step}/{len(loader)}] "
                f"loss={loss.item():.4f}  |  "
                f"phy/pair={phy_loss.item():.4f}  |  kl={kl_loss.mean().item():.4f}  |  "
                f"sigma={sigma.item():.4f}  |  K={num_pairs}  |  beta_eff={beta / num_pairs:.4f}"
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
            batch, frontend, encoder, raw_sigma, args.lambda_scale, beta=1.0
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
            "encoder": encoder.state_dict(),
            "raw_sigma": raw_sigma.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
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
    checkpoint = torch.load(path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    with torch.no_grad():
        raw_sigma.copy_(checkpoint["raw_sigma"].to(device))
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["epoch"] + 1


def append_log_row(csv_path: str, row: dict[str, float | int | str]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)  # PyTorch의 전역 랜덤 시드를 고정
    device = resolve_device(args.device)

    dataset_cls = StaticSyntheticDOADataset if args.static else SyntheticDOADataset
    simulation_config = (
        StaticSimulationConfig(sample_rate=args.sample_rate)
        if args.static
        else SimulationConfig(sample_rate=args.sample_rate)
    )

    train_dataset = dataset_cls(
        librispeech_root=args.train_librispeech_root,
        ms_snsd_root=args.train_ms_snsd_root,
        num_samples=args.train_num_samples,
        profile="stage1",
        batch_size=args.batch_size,
        seed=args.seed,
        simulation_config=simulation_config,
    )
    val_dataset = dataset_cls(
        librispeech_root=args.val_librispeech_root,
        ms_snsd_root=args.val_ms_snsd_root,
        num_samples=args.val_num_samples,
        profile=args.val_profile,
        batch_size=args.batch_size,
        seed=args.seed,
        simulation_config=simulation_config,
    )

    train_loader = build_dataloader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    val_loader = build_dataloader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    frontend, encoder, raw_sigma = build_model(args, device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + [raw_sigma], lr=args.lr_start
    )
    # gamma: 매 epoch마다 현재 lr에 곱해지는 감쇠 비율
    gamma = (args.lr_end / args.lr_start) ** (1.0 / args.epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    start_epoch = 1
    if args.resume is not None:
        start_epoch = load_checkpoint(args.resume, encoder, raw_sigma, optimizer, scheduler, device)
        print(f"resume checkpoint: {args.resume} (epoch {start_epoch}부터)")

    for epoch in range(start_epoch, args.epochs + 1):
        profile = profile_for_epoch(epoch, args.stage1_end_epoch, args.stage2_end_epoch)
        train_dataset.set_epoch(epoch)
        train_dataset.set_profile(profile)
        beta = beta_for_epoch(epoch, args.epochs, args.beta_warmup_fraction)

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
            "profile": profile,
            "beta": beta,
            "lr": optimizer.param_groups[0]["lr"],
            "sigma": F.softplus(raw_sigma).item(),
            "train_loss": train_stats["loss"],
            "train_phy": train_stats["phy"],
            "train_kl": train_stats["kl"],
            "train_kappa": train_stats["kappa"],
            "train_pairs": train_stats["pairs"],
            "val_phy": val_stats["phy"],
            "val_kl": val_stats["kl"],
        }
        append_log_row(args.log_csv, log_row)

        print(
            f"epoch {epoch}/{args.epochs} [{profile}]  "
            f"loss={train_stats['loss']:.4f}  |  "
            f"phy/pair={train_stats['phy']:.4f}  |  kl={train_stats['kl']:.4f}  |  "
            f"kappa={train_stats['kappa']:.2f}  |  K={train_stats['pairs']:.1f}  |  "
            f"beta={beta:.2f}  |  lr={log_row['lr']:.2e}"
        )

        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch:04d}.pt")
            save_checkpoint(ckpt_path, epoch, encoder, raw_sigma, optimizer, scheduler)
            last_path = os.path.join(args.checkpoint_dir, "last.pt")
            save_checkpoint(last_path, epoch, encoder, raw_sigma, optimizer, scheduler)


if __name__ == "__main__":
    main()
