"""
FAAP Training - Stabilized Gender-Aware Contrastive Learning (4th Version)

=============================================================================
3rd 버전 성공/실패 분석 기반 설계:
=============================================================================

[3rd 성공 요소 → 유지]
- Gender-Aware 구조: Anchor=Female, Positive=Male, Negative=Other Females
- 비대칭 학습: F→M 방향이 핵심
- 최고 AP Gap 달성 (-1.8%, epoch 3)

[3rd 실패 요소 → 제거/수정]
1. Epoch 3 이후 과적합 → Male Detach + 정규화로 해결
2. Score Gap Reversal → Adaptive Weighting 완전 제거
3. M→F 방향 (0.5) → Male AP 하락 → 완전 제거

=============================================================================
4th 핵심 개선 (3개 버전 장점 결합):
=============================================================================

[3rd에서 채택]
- Gender-Aware InfoNCE: F→M contrastive loss
- Wasserstein score alignment (보조)

[fix2에서 채택]
- Male Detach: proj_m.detach() → Female만 학습, Male representation 보호
  → 3rd의 과적합 원인인 양방향 gradient 제거
  → fix2는 epoch 29까지 안정적이었음

[1st에서 채택]
- LayerNorm in ProjectionHead → 더 안정적인 feature normalization
- Feature Mean Alignment Loss → 직접적인 분포 중심 정렬

[새로운 개선]
1. Adaptive Weighting 완전 제거 (Score Gap Reversal 근본 해결)
2. M→F 방향 완전 제거 (Male AP 보호)
3. Dropout(0.1) in ProjectionHead → 과적합 방지
4. Temperature 0.07 → 0.1 (안정적인 gradient)
5. Epsilon Schedule (0.05→0.10→0.09) → 점진적 학습
6. Contrastive Warmup (3 epochs) → 초기 안정화
7. Augmentation "medium" → "weak" (detection 성능 보호)
8. Gradient clipping 0.1 → 0.5 (유연한 학습)
9. Cosine LR Schedule + Warmup

=============================================================================
설계 근거 (기존 MoCo 4th 대비):
=============================================================================
MoCo의 Memory Bank/Momentum Centroid는 3rd의 핵심 문제를 직접 해결하지 않음.
3rd의 실패는 "batch size가 작아서"가 아니라:
  - Male gradient가 M→F 방향으로 흘러 Male AP 하락
  - Adaptive Weighting이 Score Gap Reversal로 역효과
  - ProjectionHead 정규화 부재로 과적합
이 세 가지를 직접 해결하는 것이 더 효과적.

기대: 3rd의 AP Gap 개선력 (-1.8%) + fix2의 안정성 (29 epochs)
=============================================================================
"""

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

# Allow running as a script
if __package__ is None or __package__ == "":
    import sys
    pkg_dir = Path(__file__).resolve().parent
    parent = pkg_dir.parent
    if str(parent) not in sys.path:
        sys.path.append(str(parent))
    if str(pkg_dir) not in sys.path:
        sys.path.append(str(pkg_dir))
    __package__ = "faap_gan"

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .datasets import build_faap_dataloader, inspect_faap_dataset
from .models import FrozenDETR, PerturbationGenerator, clamp_normalized
from .path_utils import DETR_REPO, default_detr_checkpoint, ensure_detr_repo_on_path
import util.misc as utils
from util.misc import NestedTensor


# =============================================================================
# SimCLR-Style Data Augmentation (3rd와 동일 구조, 기본값 "weak")
# =============================================================================

class SimCLRAugmentation(nn.Module):
    """
    3rd는 "medium" 사용 → 4th는 "weak"로 약화.
    이유: perturbation 자체가 augmentation 역할을 하므로,
    추가 augmentation이 너무 강하면 detection feature를 왜곡.
    """

    def __init__(self, strength: str = "weak"):
        super().__init__()
        self.strength = strength

        if strength == "none":
            self.transform = None
        elif strength == "weak":
            self.transform = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        elif strength == "medium":
            self.transform = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        elif strength == "strong":
            self.transform = T.Compose([
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                T.RandomGrayscale(p=0.2),
            ])
        else:
            raise ValueError(f"Unknown augmentation strength: {strength}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform is None:
            return x
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_denorm = torch.clamp(x * std + mean, 0, 1)
        augmented = torch.stack([self.transform(img) for img in x_denorm])
        return (augmented - mean) / std


# =============================================================================
# Stabilized Projection Head (1st의 LayerNorm + 신규 Dropout)
# =============================================================================

class StabilizedProjectionHead(nn.Module):
    """
    3rd/fix2의 SimCLRProjectionHead를 정규화 강화 버전으로 교체.

    변경점:
    - LayerNorm 추가 (1st 버전에서 검증): 입력 feature scale 정규화
    - Dropout 추가 (신규): 과적합 방지

    3rd의 epoch 3 과적합 원인 중 하나는 ProjectionHead가
    training distribution에 과적합된 것. LayerNorm은 feature scale
    차이를 안정화하고, Dropout은 co-adaptation을 방지함.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256,
                 output_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)  # (batch, num_queries, D) → (batch, D)
        proj = self.net(pooled)
        return F.normalize(proj, dim=-1, p=2)


# =============================================================================
# Detach InfoNCE Loss (핵심: 3rd 장점 + fix2 안정성 + 문제 제거)
# =============================================================================

class DetachInfoNCELoss(nn.Module):
    """
    Male-Detached Gender-Aware InfoNCE Loss.

    3rd 대비 3가지 핵심 변경:

    1. Male Detach (fix2에서 검증):
       - proj_m.detach() → contrastive loss의 gradient가 Male로 흐르지 않음
       - Female projection만 Male 방향으로 이동
       - fix2는 이 방식으로 epoch 29까지 안정적이었음

    2. Adaptive Weighting 제거:
       - 3rd의 `w = 0.5 + sigmoid((score_m - score_f) * 5)` 제거
       - Score Gap Reversal 문제: 학습 중 score_m < score_f 발생
       → weight < 1.0 → 필요한 쌍의 학습이 오히려 약해짐
       - 단순 균일 가중치가 더 안정적

    3. M→F 방향 제거:
       - 3rd: 1.5 * F→M + 0.5 * M→F
       - 4th: F→M only
       - M→F는 Male feature를 Female 방향으로 끌어 Male AP 하락 기여
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        proj_f: torch.Tensor,
        proj_m: torch.Tensor,
        scores_f: torch.Tensor,
        scores_m: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        n_f = proj_f.size(0)
        n_m = proj_m.size(0)

        if n_f < 2 or n_m < 1:
            return proj_f.new_tensor(0.0), {
                "n_f": n_f, "n_m": n_m, "score_gap": 0.0,
                "sim_f2m_mean": 0.0, "sim_f2f_mean": 0.0,
            }

        # Male Detach: gradient가 Male projection으로 흐르지 않음
        proj_m_detached = proj_m.detach()

        # F→M similarity (positive pairs)
        sim_f2m = torch.mm(proj_f, proj_m_detached.t()) / self.temperature  # (N_f, N_m)

        # F→F similarity (negative pairs)
        sim_f2f = torch.mm(proj_f, proj_f.t()) / self.temperature  # (N_f, N_f)
        mask_self = torch.eye(n_f, device=proj_f.device, dtype=torch.bool)
        sim_f2f_masked = sim_f2f.masked_fill(mask_self, float('-inf'))

        # InfoNCE: all males are positive, other females are negative
        all_sims = torch.cat([sim_f2m, sim_f2f_masked], dim=1)
        numerator = torch.logsumexp(sim_f2m, dim=1)
        denominator = torch.logsumexp(all_sims, dim=1)
        loss = -(numerator - denominator).mean()

        # Monitoring metrics (no grad)
        with torch.no_grad():
            sim_f2m_raw = torch.mm(proj_f, proj_m_detached.t())
            sim_f2m_mean = sim_f2m_raw.mean().item()
            sim_f2f_raw = torch.mm(proj_f, proj_f.t())
            sim_f2f_mean = sim_f2f_raw[~mask_self].mean().item() if n_f > 1 else 0.0

        info = {
            "n_f": n_f,
            "n_m": n_m,
            "score_f_mean": scores_f.detach().mean().item(),
            "score_m_mean": scores_m.detach().mean().item(),
            "score_gap": (scores_m.detach().mean() - scores_f.detach().mean()).item(),
            "sim_f2m_mean": sim_f2m_mean,
            "sim_f2f_mean": sim_f2f_mean,
        }

        return loss, info


# =============================================================================
# Feature Mean Alignment (1st 버전에서 채택)
# =============================================================================

def _feature_mean_alignment(features_f: torch.Tensor, features_m: torch.Tensor) -> torch.Tensor:
    """
    Female feature 분포 중심을 Male 방향으로 이동 (1st에서 검증).

    Contrastive loss는 projection space에서 작용하고,
    이 loss는 원본 DETR feature space에서 작용하여 상호 보완적.
    Male 측은 detach하여 Female만 이동.
    """
    if features_f.size(0) == 0 or features_m.size(0) == 0:
        return features_f.new_tensor(0.0)

    pooled_f = features_f.mean(dim=1)             # (N_f, D)
    pooled_m = features_m.mean(dim=1).detach()     # (N_m, D) - Male gradient 차단

    mean_f = pooled_f.mean(dim=0)   # (D,)
    mean_m = pooled_m.mean(dim=0)   # (D,)

    return F.mse_loss(mean_f, mean_m)


# =============================================================================
# Utility Functions
# =============================================================================

def _image_level_detection_score(outputs: dict, top_k: int = 10) -> torch.Tensor:
    """DETR logits에서 이미지 단위 score 계산"""
    probs = outputs["pred_logits"].softmax(dim=-1)[..., :-1]
    max_probs = probs.max(dim=-1).values
    if top_k > 0 and top_k < max_probs.size(1):
        topk_probs = max_probs.topk(top_k, dim=1).values
        return topk_probs.mean(dim=1)
    return max_probs.mean(dim=1)


def _wasserstein_1d_asymmetric(female_scores: torch.Tensor, male_scores: torch.Tensor) -> torch.Tensor:
    """단방향 Wasserstein: Female score를 Male 수준으로 끌어올림"""
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0)
    sorted_f = female_scores.sort().values
    sorted_m = male_scores.detach().sort().values
    k = max(sorted_f.numel(), sorted_m.numel())

    if k != sorted_f.numel():
        idx = torch.linspace(0, sorted_f.numel() - 1, k, device=sorted_f.device)
        idx_low, idx_high = idx.floor().long(), idx.ceil().long()
        weight = idx - idx_low
        sorted_f = sorted_f[idx_low] * (1 - weight) + sorted_f[idx_high] * weight

    if k != sorted_m.numel():
        idx = torch.linspace(0, sorted_m.numel() - 1, k, device=sorted_m.device)
        idx_low, idx_high = idx.floor().long(), idx.ceil().long()
        weight = idx - idx_low
        sorted_m = sorted_m[idx_low] * (1 - weight) + sorted_m[idx_high] * weight

    return F.relu(sorted_m - sorted_f).mean()


# =============================================================================
# Schedule Functions
# =============================================================================

def _epsilon_schedule(
    epoch: int, total_epochs: int,
    eps_start: float = 0.05, eps_peak: float = 0.10, eps_final: float = 0.09,
    warmup_epochs: int = 6, hold_epochs: int = 6,
) -> float:
    """
    3단계 Epsilon Schedule (WGAN 7th에서 검증).
    Warmup → Hold → Cooldown.
    """
    cooldown_start = warmup_epochs + hold_epochs
    if epoch < warmup_epochs:
        progress = epoch / max(1, warmup_epochs)
        return eps_start + (eps_peak - eps_start) * progress
    elif epoch < cooldown_start:
        return eps_peak
    else:
        remaining = total_epochs - cooldown_start
        if remaining <= 0:
            return eps_final
        progress = min((epoch - cooldown_start) / remaining, 1.0)
        return eps_peak + (eps_final - eps_peak) * progress


def _contrastive_warmup(epoch: int, warmup_epochs: int = 3) -> float:
    """Contrastive loss 가중치를 서서히 증가 (초기 안정화)."""
    if epoch < warmup_epochs:
        return (epoch + 1) / (warmup_epochs + 1)
    return 1.0


def _scheduled_beta(epoch: int, total_epochs: int, beta_start: float, beta_final: float) -> float:
    """Detection loss weight linear schedule"""
    if total_epochs <= 1:
        return beta_start
    progress = min(epoch / max(1, total_epochs - 1), 1.0)
    return beta_start + (beta_final - beta_start) * progress


def _cosine_lr(optimizer: torch.optim.Optimizer, epoch: int, total_epochs: int,
               lr_base: float, lr_min: float = 1e-6, warmup_epochs: int = 2) -> float:
    """Cosine LR schedule with linear warmup."""
    if epoch < warmup_epochs:
        lr = lr_base * (epoch + 1) / (warmup_epochs + 1)
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = lr_min + 0.5 * (lr_base - lr_min) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# =============================================================================
# Other Utilities
# =============================================================================

def _default_output_dir(script_path: Path) -> str:
    stem = script_path.stem
    for prefix in ("train_faap_simclr_", "train_faap_", "train_"):
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix):]
            break
    return str(Path("faap_outputs") / f"faap_outputs_{stem.lower()}")


def _apply_generator(generator: nn.Module, samples: NestedTensor) -> NestedTensor:
    tensors = samples.tensors
    delta = generator(tensors)
    perturbed = clamp_normalized(tensors + delta)
    return NestedTensor(perturbed, samples.mask)


def _unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def _set_generator_epsilon(generator: nn.Module, epsilon: float) -> None:
    _unwrap_ddp(generator).epsilon = epsilon


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "FAAP Stabilized Gender-Aware Contrastive (4th Version)"
    )

    # Paths
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()))
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(Path(__file__)))

    # Training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--lr_warmup_epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # Perturbation (epsilon schedule)
    parser.add_argument("--epsilon_start", type=float, default=0.05)
    parser.add_argument("--epsilon_peak", type=float, default=0.10)
    parser.add_argument("--epsilon_final", type=float, default=0.09)
    parser.add_argument("--epsilon_warmup_epochs", type=int, default=6)
    parser.add_argument("--epsilon_hold_epochs", type=int, default=6)

    # Loss weights
    parser.add_argument("--lambda_contrastive", type=float, default=1.0)
    parser.add_argument("--lambda_align", type=float, default=0.3,
                        help="Feature mean alignment weight (1st에서 채택)")
    parser.add_argument("--lambda_wass", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--beta_final", type=float, default=0.6)

    # Contrastive settings
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="3rd의 0.07보다 높여 gradient 안정화")
    parser.add_argument("--contrastive_warmup_epochs", type=int, default=3)
    parser.add_argument("--score_top_k", type=int, default=10)

    # Projection Head
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--proj_dropout", type=float, default=0.1)

    # Augmentation
    parser.add_argument("--aug_strength", type=str, default="weak",
                        choices=["none", "weak", "medium", "strong"])

    # Other
    parser.add_argument("--max_norm", type=float, default=0.5,
                        help="3rd의 0.1보다 유연하게")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--dist_url", default="env://")

    return parser.parse_args()


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    args = parse_args()
    utils.init_distributed_mode(args)

    if not hasattr(args, "gpu"):
        args.gpu = None

    detr_repo = ensure_detr_repo_on_path(Path(args.detr_repo))
    ckpt_path = Path(args.detr_checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = detr_repo / ckpt_path

    if args.distributed:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    args.world_size = utils.get_world_size()
    args.rank = utils.get_rank()

    output_dir = Path(args.output_dir)
    if utils.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        with (output_dir / "config.json").open("w") as f:
            json.dump(vars(args), f, indent=2)

    if args.distributed:
        dist.barrier()

    if utils.is_main_process():
        dataset_info = inspect_faap_dataset(Path(args.dataset_root))
        with (output_dir / "dataset_layout.json").open("w") as f:
            json.dump(dataset_info, f, indent=2)

        print("=" * 70)
        print("Stabilized Gender-Aware Contrastive Learning (4th Version)")
        print("=" * 70)
        print()
        print("[3rd 대비 핵심 변경]")
        print("  1. Male Detach (fix2) → Female만 학습, 과적합 방지")
        print("  2. Adaptive Weighting 제거 → Score Gap Reversal 해결")
        print("  3. M→F 방향 제거 → Male AP 보호")
        print("  4. LayerNorm + Dropout(0.1) → ProjectionHead 정규화")
        print("  5. Temperature 0.07→0.1 → gradient 안정화")
        print("  6. Epsilon Schedule (0.05→0.10→0.09)")
        print("  7. Feature Mean Alignment (1st) → 분포 중심 정렬")
        print("  8. Cosine LR Schedule + Warmup")
        print("  9. Augmentation medium→weak → detection 보호")
        print("-" * 70)
        print(f"Temperature: {args.temperature}")
        print(f"Epsilon: {args.epsilon_start} → {args.epsilon_peak} → {args.epsilon_final}")
        print(f"Contrastive warmup: {args.contrastive_warmup_epochs} epochs")
        print(f"LR: {args.lr_g} (cosine → {args.lr_min})")
        print(f"Projection dropout: {args.proj_dropout}")
        print(f"Augmentation: {args.aug_strength}")
        print(f"Gradient clip: {args.max_norm}")
        print(f"Loss: C={args.lambda_contrastive}, A={args.lambda_align}, "
              f"W={args.lambda_wass}, D={args.beta}→{args.beta_final}")
        print("=" * 70)

    # ==========================================================================
    # Model Initialization
    # ==========================================================================

    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon_start).to(device)

    proj_head = StabilizedProjectionHead(
        input_dim=detr.hidden_dim,
        hidden_dim=detr.hidden_dim,
        output_dim=args.proj_dim,
        dropout=args.proj_dropout,
    ).to(device)

    contrastive_loss_fn = DetachInfoNCELoss(
        temperature=args.temperature,
    ).to(device)

    simclr_aug = SimCLRAugmentation(strength=args.aug_strength).to(device)

    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)
        proj_head = DDP(proj_head, device_ids=[args.gpu] if args.gpu is not None else None)

    params = list(_unwrap_ddp(generator).parameters()) + list(_unwrap_ddp(proj_head).parameters())
    opt_g = torch.optim.AdamW(params, lr=args.lr_g, weight_decay=0.01)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if "generator" in ckpt:
            _unwrap_ddp(generator).load_state_dict(ckpt["generator"])
        if "proj_head" in ckpt:
            _unwrap_ddp(proj_head).load_state_dict(ckpt["proj_head"])
        if "opt_g" in ckpt:
            opt_g.load_state_dict(ckpt["opt_g"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if utils.is_main_process():
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # DataLoader
    train_loader, _ = build_faap_dataloader(
        Path(args.dataset_root),
        "train",
        args.batch_size,
        include_gender=True,
        balance_genders=True,
        num_workers=args.num_workers,
        distributed=args.distributed,
        rank=args.rank,
        world_size=args.world_size,
    )

    log_path = output_dir / "train_log.jsonl"

    # ==========================================================================
    # Training Loop
    # ==========================================================================

    for epoch in range(start_epoch, args.epochs):
        metrics_logger = utils.MetricLogger(delimiter="  ")
        generator.train()
        proj_head.train()

        if args.distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # --- Schedules ---
        current_eps = _epsilon_schedule(
            epoch, args.epochs,
            args.epsilon_start, args.epsilon_peak, args.epsilon_final,
            args.epsilon_warmup_epochs, args.epsilon_hold_epochs,
        )
        _set_generator_epsilon(generator, current_eps)

        current_beta = _scheduled_beta(epoch, args.epochs, args.beta, args.beta_final)
        contrastive_weight = _contrastive_warmup(epoch, args.contrastive_warmup_epochs)
        current_lr = _cosine_lr(
            opt_g, epoch, args.epochs, args.lr_g, args.lr_min, args.lr_warmup_epochs
        )

        for samples, targets, genders in metrics_logger.log_every(
            train_loader, args.log_every, f"Epoch {epoch}"
        ):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            genders = [g.lower() for g in genders]

            female_idx = [i for i, g in enumerate(genders) if g == "female"]
            male_idx = [i for i, g in enumerate(genders) if g == "male"]

            if len(female_idx) < 2 or len(male_idx) < 1:
                continue

            # =================================================================
            # Forward Pass
            # =================================================================
            opt_g.zero_grad()

            perturbed = _apply_generator(generator, samples)

            if generator.training and args.aug_strength != "none":
                perturbed = NestedTensor(
                    simclr_aug(perturbed.tensors),
                    perturbed.mask,
                )

            outputs, features = detr.forward_with_features(perturbed)

            # =================================================================
            # 1. Detach InfoNCE Contrastive Loss (핵심)
            # =================================================================
            image_scores = _image_level_detection_score(outputs, top_k=args.score_top_k)

            proj_all = proj_head(features)
            proj_f = proj_all[female_idx]
            proj_m = proj_all[male_idx]
            scores_f = image_scores[female_idx]
            scores_m = image_scores[male_idx]

            loss_contrastive, contrastive_info = contrastive_loss_fn(
                proj_f, proj_m, scores_f, scores_m
            )

            # =================================================================
            # 2. Feature Mean Alignment (1st에서 채택)
            # =================================================================
            features_f = features[female_idx]
            features_m = features[male_idx]
            loss_align = _feature_mean_alignment(features_f, features_m)

            # =================================================================
            # 3. Score-Level Wasserstein (보조)
            # =================================================================
            loss_wasserstein = _wasserstein_1d_asymmetric(scores_f, scores_m)

            # =================================================================
            # 4. Detection Loss
            # =================================================================
            loss_det, _ = detr.detection_loss(outputs, targets)

            # =================================================================
            # Total Loss
            # =================================================================
            total_g = (
                args.lambda_contrastive * contrastive_weight * loss_contrastive
                + args.lambda_align * loss_align
                + args.lambda_wass * loss_wasserstein
                + current_beta * loss_det
            )

            # =================================================================
            # Metrics
            # =================================================================
            with torch.no_grad():
                delta = perturbed.tensors - samples.tensors
                delta_linf = delta.abs().amax(dim=(1, 2, 3)).mean()
                delta_l2 = delta.flatten(1).norm(p=2, dim=1).mean()

            # =================================================================
            # Backward & Optimize
            # =================================================================
            total_g.backward()
            if args.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(generator.parameters()) + list(proj_head.parameters()),
                    args.max_norm,
                )
            opt_g.step()

            metrics_logger.update(
                loss_contrastive=loss_contrastive.item(),
                loss_align=loss_align.item(),
                loss_wasserstein=loss_wasserstein.item(),
                loss_det=loss_det.item(),
                total_g=total_g.item(),
                c_weight=contrastive_weight,
                epsilon=current_eps,
                beta=current_beta,
                lr=current_lr,
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                score_f=contrastive_info.get("score_f_mean", 0.0),
                score_m=contrastive_info.get("score_m_mean", 0.0),
                score_gap=contrastive_info.get("score_gap", 0.0),
                sim_f2m=contrastive_info.get("sim_f2m_mean", 0.0),
                sim_f2f=contrastive_info.get("sim_f2f_mean", 0.0),
                n_f=contrastive_info.get("n_f", 0),
                n_m=contrastive_info.get("n_m", 0),
            )

        # =====================================================================
        # End of Epoch
        # =====================================================================
        metrics_logger.synchronize_between_processes()

        if utils.is_main_process():
            log_entry = {
                "epoch": epoch,
                "loss_contrastive": metrics_logger.meters["loss_contrastive"].global_avg,
                "loss_align": metrics_logger.meters["loss_align"].global_avg,
                "loss_wasserstein": metrics_logger.meters["loss_wasserstein"].global_avg,
                "loss_det": metrics_logger.meters["loss_det"].global_avg,
                "total_g": metrics_logger.meters["total_g"].global_avg,
                "contrastive_weight": contrastive_weight,
                "epsilon": current_eps,
                "beta": current_beta,
                "lr": current_lr,
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "score_f": metrics_logger.meters["score_f"].global_avg,
                "score_m": metrics_logger.meters["score_m"].global_avg,
                "score_gap": metrics_logger.meters["score_gap"].global_avg,
                "sim_f2m": metrics_logger.meters["sim_f2m"].global_avg,
                "sim_f2f": metrics_logger.meters["sim_f2f"].global_avg,
                "n_f_avg": metrics_logger.meters["n_f"].global_avg,
                "n_m_avg": metrics_logger.meters["n_m"].global_avg,
            }

            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(f"\n[Epoch {epoch}] Summary:")
            print(f"  Contrastive: {log_entry['loss_contrastive']:.4f}"
                  f" (weight: {contrastive_weight:.2f})")
            print(f"  Align: {log_entry['loss_align']:.6f}")
            print(f"  Wasserstein: {log_entry['loss_wasserstein']:.4f}")
            print(f"  Detection: {log_entry['loss_det']:.4f}")
            print(f"  Total: {log_entry['total_g']:.4f}")
            print(f"  Score (F/M): {log_entry['score_f']:.4f} / {log_entry['score_m']:.4f}"
                  f"  Gap: {log_entry['score_gap']:.4f}")
            print(f"  Sim F→M: {log_entry['sim_f2m']:.4f}  |  F→F: {log_entry['sim_f2f']:.4f}")
            print(f"  Eps: {current_eps:.4f}  Beta: {current_beta:.4f}"
                  f"  LR: {current_lr:.2e}")

            if (epoch + 1) % args.save_every == 0:
                ckpt_path_save = output_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "generator": _unwrap_ddp(generator).state_dict(),
                        "proj_head": _unwrap_ddp(proj_head).state_dict(),
                        "opt_g": opt_g.state_dict(),
                        "args": vars(args),
                    },
                    ckpt_path_save,
                )
                print(f"  Saved: {ckpt_path_save}")

        if args.distributed:
            dist.barrier()

    # =========================================================================
    # Training Complete
    # =========================================================================
    if utils.is_main_process():
        print("\n" + "=" * 70)
        print("Stabilized Gender-Aware Contrastive (4th) Complete!")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print("\n[4th = 3rd의 AP Gap 개선력 + fix2의 안정성]")
        print("  - Male Detach → 과적합 방지 (fix2: 29 epochs 안정)")
        print("  - Adaptive Weighting 제거 → Score Gap Reversal 해결")
        print("  - M→F 제거 → Male AP 보호")
        print("  - LayerNorm + Dropout → 정규화")
        print("  - Feature Mean Alignment → 분포 중심 정렬")
        print("\n성공 기준 (vs Baseline 0.1063 / 0.0081):")
        print("  - AP Gap < 0.100 (~6% 개선)")
        print("  - AR Gap < 0.005 (~38% 개선)")
        print("  - Female AP > 0.410, Male AP >= 0.511")


if __name__ == "__main__":
    main()
