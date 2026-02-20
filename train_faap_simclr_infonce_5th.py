"""
FAAP Training - Controlled Aggression Contrastive Learning (5th Version)

=============================================================================
4th 실패 원인 분석:
=============================================================================
- Male Detach + Triple Warmup → 3rd의 빠른 초기 개선 메커니즘 제거
- Adaptive Weighting 제거는 올바른 판단이었으나, 양방향 신호까지 제거
- 과도한 정규화 (LayerNorm, Dropout, weak aug, temperature 0.1)
- 결과: 안정적이지만 AP Gap 개선 미미 (best -1.2% vs 3rd의 -1.8%)

=============================================================================
5th 핵심 아이디어: 3rd의 Power + 선별적 안정화
=============================================================================

[3rd에서 유지]
  1. 양방향 F→M (1.5x) + M→F (변동)  ← 검증된 gradient 엔진
  2. Male Detach 없음  ← 양방향 신호 보존
  3. Temperature 0.07  ← 날카로운 gradient
  4. Epsilon 0.10 고정  ← warmup 없이 전력
  5. LR 1e-4 고정  ← warmup/cosine 없이 전력
  6. Medium augmentation  ← contrastive 다양성 확보
  7. ProjectionHead 정규화 없음  ← 과도한 정규화 방지

[3rd에서 수정 (수술적 변경)]
  1. Adaptive Weighting 제거  ← Score Gap Reversal 해결 (4th에서 검증)
  2. M→F 감쇠 스케줄  ← ep0-3: 0.5 (3rd처럼), ep4-8: 감쇠→0, ep9+: 0
     → 초기에는 양방향으로 빠른 개선, 후반에는 M→F 제거로 Male AP 보호

[핵심 가설]
  3rd의 epoch 3 peak는 "과적합"이 아니라 "빠른 수렴"
  Epoch 3 이후 하락의 원인:
    1. Adaptive Weighting의 Score Gap Reversal → 해결 (제거)
    2. M→F의 누적 간섭 → 해결 (감쇠 스케줄)
  → 5th는 epoch 3의 성능을 유지하면서 이후에도 안정적

=============================================================================
"""

import argparse
import json
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
# SimCLR-Style Data Augmentation (3rd와 동일)
# =============================================================================

class SimCLRAugmentation(nn.Module):
    def __init__(self, strength: str = "medium"):
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
# SimCLR Projection Head (3rd와 동일 - 정규화 없음)
# =============================================================================

class SimCLRProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        proj = self.net(pooled)
        return F.normalize(proj, dim=-1, p=2)


# =============================================================================
# Controlled Aggression Contrastive Loss (5th: 핵심 변경)
# =============================================================================

class ControlledAggressionLoss(nn.Module):
    """
    3rd의 검증된 양방향 구조 + Adaptive Weighting 제거 + M→F 감쇠

    3rd 대비 변경:
    1. Adaptive Weighting (sigmoid score_diff) 완전 제거
       → Score Gap Reversal 문제 해결
    2. w_m2f 파라미터로 M→F 가중치를 외부에서 제어
       → epoch별 감쇠 스케줄 적용 가능
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        proj_f: torch.Tensor,
        proj_m: torch.Tensor,
        scores_f: torch.Tensor,
        scores_m: torch.Tensor,
        w_m2f: float = 0.5,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            proj_f: (N_f, D) Female projections (L2-normalized)
            proj_m: (N_m, D) Male projections (L2-normalized)
            scores_f: (N_f,) Female detection scores
            scores_m: (N_m,) Male detection scores
            w_m2f: M→F 방향 가중치 (스케줄에 의해 감쇠)

        Returns:
            loss: scalar
            info: dict with debugging info
        """
        n_f = proj_f.size(0)
        n_m = proj_m.size(0)

        if n_f < 2 or n_m < 1:
            return proj_f.new_tensor(0.0), {
                "n_f": n_f, "n_m": n_m, "score_gap": 0.0,
                "sim_f2m": 0.0, "sim_f2f": 0.0,
            }

        scores_f = scores_f.detach()
        scores_m = scores_m.detach()

        # =================================================================
        # 1. F→M Contrastive (핵심 - 3rd와 동일, Adaptive Weighting 제거)
        # =================================================================
        sim_f2m = torch.mm(proj_f, proj_m.t()) / self.temperature  # (N_f, N_m)
        sim_f2f = torch.mm(proj_f, proj_f.t()) / self.temperature  # (N_f, N_f)

        mask_self = torch.eye(n_f, device=proj_f.device, dtype=torch.bool)
        sim_f2f_masked = sim_f2f.masked_fill(mask_self, float('-inf'))

        # 균일 가중치 (Adaptive Weighting 제거 - 4th에서 검증)
        all_sims = torch.cat([sim_f2m, sim_f2f_masked], dim=1)
        numerator = torch.logsumexp(sim_f2m, dim=1)
        denominator = torch.logsumexp(all_sims, dim=1)
        loss_f2m = -(numerator - denominator).mean()

        # =================================================================
        # 2. M→F Contrastive (감쇠 스케줄 적용)
        # =================================================================
        loss_m2f = proj_f.new_tensor(0.0)
        if w_m2f > 0 and n_m >= 2:
            sim_m2f = sim_f2m.t()  # (N_m, N_f)
            sim_m2m = torch.mm(proj_m, proj_m.t()) / self.temperature
            mask_m = torch.eye(n_m, device=proj_m.device, dtype=torch.bool)
            sim_m2m_masked = sim_m2m.masked_fill(mask_m, float('-inf'))

            all_sims_m = torch.cat([sim_m2f, sim_m2m_masked], dim=1)
            numerator_m = torch.logsumexp(sim_m2f, dim=1)
            denominator_m = torch.logsumexp(all_sims_m, dim=1)
            loss_m2f = -(numerator_m - denominator_m).mean()

        # 비대칭 결합: F→M 고정(1.5), M→F 감쇠(w_m2f: 0.5→0.0)
        loss = 1.5 * loss_f2m + w_m2f * loss_m2f

        # Monitoring (cosine sim, not temperature-scaled)
        with torch.no_grad():
            cos_f2m = torch.mm(proj_f, proj_m.t()).mean().item()
            cos_f2f_vals = torch.mm(proj_f, proj_f.t())
            cos_f2f_vals = cos_f2f_vals.masked_fill(mask_self, 0.0)
            cos_f2f_mean = cos_f2f_vals.sum().item() / max(n_f * (n_f - 1), 1)

        score_gap = (scores_m.mean() - scores_f.mean()).item()
        info = {
            "n_f": n_f,
            "n_m": n_m,
            "score_f_mean": scores_f.mean().item(),
            "score_m_mean": scores_m.mean().item(),
            "score_gap": score_gap,
            "loss_f2m": loss_f2m.item(),
            "loss_m2f": loss_m2f.item() if isinstance(loss_m2f, torch.Tensor) else 0.0,
            "sim_f2m": cos_f2m,
            "sim_f2f": cos_f2f_mean,
            "w_m2f": w_m2f,
        }

        return loss, info


# =============================================================================
# Schedule Functions
# =============================================================================

def _m2f_weight_schedule(
    epoch: int,
    full_epochs: int = 4,    # ep0-3: 0.5 유지 (3rd의 peak 구간)
    decay_epochs: int = 5,   # ep4-8: 선형 감쇠
    w_start: float = 0.5,
) -> float:
    """
    M→F 가중치 감쇠 스케줄.

    3rd의 epoch 3 peak를 재현하되, 이후 M→F 간섭을 점진적 제거.
    - Epoch 0-3: w=0.5 (3rd와 동일, 양방향 full power)
    - Epoch 4-8: 0.5→0.0 선형 감쇠
    - Epoch 9+: w=0.0 (F→M only, fix2처럼 안정)
    """
    if epoch < full_epochs:
        return w_start
    decay_start = full_epochs
    decay_end = full_epochs + decay_epochs
    if epoch >= decay_end:
        return 0.0
    progress = (epoch - decay_start) / decay_epochs
    return w_start * (1.0 - progress)


def _scheduled_beta(
    epoch: int, total_epochs: int,
    beta_start: float, beta_final: float,
) -> float:
    """Detection loss 가중치 스케줄 (3rd와 동일)."""
    if total_epochs <= 1:
        return beta_start
    progress = min(epoch / max(1, total_epochs - 1), 1.0)
    return beta_start + (beta_final - beta_start) * progress


# =============================================================================
# Utility Functions (3rd와 동일)
# =============================================================================

def _image_level_detection_score(outputs: dict, top_k: int = 10) -> torch.Tensor:
    probs = outputs["pred_logits"].softmax(dim=-1)[..., :-1]
    max_probs = probs.max(dim=-1).values
    if top_k > 0 and top_k < max_probs.size(1):
        topk_probs = max_probs.topk(top_k, dim=1).values
        return topk_probs.mean(dim=1)
    return max_probs.mean(dim=1)


def _wasserstein_1d_asymmetric(female_scores: torch.Tensor, male_scores: torch.Tensor) -> torch.Tensor:
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


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "FAAP Controlled Aggression Contrastive (5th Version)"
    )

    # Paths
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()))
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(Path(__file__)))

    # Training (3rd와 동일)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    # Perturbation (고정 - 3rd와 동일)
    parser.add_argument("--epsilon", type=float, default=0.10)

    # Loss weights (3rd와 동일)
    parser.add_argument("--lambda_contrastive", type=float, default=1.0)
    parser.add_argument("--lambda_wass", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--beta_final", type=float, default=0.6)

    # Contrastive settings (3rd와 동일)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--score_top_k", type=int, default=10)
    parser.add_argument("--proj_dim", type=int, default=128)

    # M→F 감쇠 스케줄 (5th 핵심)
    parser.add_argument("--m2f_full_epochs", type=int, default=4,
                        help="M→F w=0.5 유지 epoch 수 (ep0-3)")
    parser.add_argument("--m2f_decay_epochs", type=int, default=5,
                        help="M→F 선형 감쇠 epoch 수 (ep4-8)")
    parser.add_argument("--m2f_weight", type=float, default=0.5,
                        help="M→F 초기 가중치")

    # Augmentation (3rd와 동일: medium)
    parser.add_argument("--aug_strength", type=str, default="medium",
                        choices=["none", "weak", "medium", "strong"])

    # Other (3rd와 동일)
    parser.add_argument("--max_norm", type=float, default=0.1)
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

    # Setup
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

    # Output directory
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
        print("Controlled Aggression Contrastive Learning (5th Version)")
        print("=" * 70)
        print()
        print("[4th 실패 원인]")
        print("  - Male Detach → 양방향 gradient 신호 제거")
        print("  - Triple Warmup → epoch 0-3 유효 신호 3rd의 4%로 억제")
        print("  - 과도한 정규화 (LayerNorm, Dropout, weak aug)")
        print()
        print("[5th = 3rd의 Power + 수술적 안정화]")
        print("  1. Adaptive Weighting 제거 (4th에서 검증)")
        print("  2. M→F 감쇠 스케줄 (ep0-3: 0.5, ep4-8: 감쇠→0)")
        print("  3. 나머지 전부 3rd와 동일")
        print("-" * 70)
        print(f"Temperature: {args.temperature}")
        print(f"Epsilon: {args.epsilon} (fixed)")
        print(f"M→F weight: {args.m2f_weight} → 0.0 "
              f"(full={args.m2f_full_epochs}ep, decay={args.m2f_decay_epochs}ep)")
        print(f"LR: {args.lr_g} (fixed)")
        print(f"Augmentation: {args.aug_strength}")
        print(f"Gradient clip: {args.max_norm}")
        print(f"Loss: C=1.0 (F→M=1.5, M→F=sched), W=0.2, D=0.5→0.6")
        print("=" * 70)

    # ======================================================================
    # Model Initialization
    # ======================================================================

    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)

    proj_head = SimCLRProjectionHead(
        input_dim=detr.hidden_dim,
        hidden_dim=detr.hidden_dim,
        output_dim=args.proj_dim,
    ).to(device)

    contrastive_loss_fn = ControlledAggressionLoss(
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

    # ======================================================================
    # Training Loop
    # ======================================================================

    for epoch in range(start_epoch, args.epochs):
        metrics_logger = utils.MetricLogger(delimiter="  ")
        generator.train()
        proj_head.train()

        if args.distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        current_beta = _scheduled_beta(epoch, args.epochs, args.beta, args.beta_final)
        current_w_m2f = _m2f_weight_schedule(
            epoch,
            full_epochs=args.m2f_full_epochs,
            decay_epochs=args.m2f_decay_epochs,
            w_start=args.m2f_weight,
        )

        if utils.is_main_process():
            print(f"\n--- Epoch {epoch} | w_m2f={current_w_m2f:.3f} | "
                  f"beta={current_beta:.3f} ---")

        for samples, targets, genders in metrics_logger.log_every(
            train_loader, args.log_every, f"Epoch {epoch}"
        ):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            genders = [g.lower() for g in genders]

            # Gender split
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
            # 1. Contrastive Loss (Adaptive Weighting 제거, M→F 감쇠)
            # =================================================================
            image_scores = _image_level_detection_score(outputs, top_k=args.score_top_k)

            proj_all = proj_head(features)
            proj_f = proj_all[female_idx]
            proj_m = proj_all[male_idx]
            scores_f = image_scores[female_idx]
            scores_m = image_scores[male_idx]

            loss_contrastive, contrastive_info = contrastive_loss_fn(
                proj_f, proj_m, scores_f, scores_m,
                w_m2f=current_w_m2f,
            )

            # =================================================================
            # 2. Wasserstein Loss (3rd와 동일)
            # =================================================================
            loss_wasserstein = _wasserstein_1d_asymmetric(scores_f, scores_m)

            # =================================================================
            # 3. Detection Loss (3rd와 동일)
            # =================================================================
            loss_det, _ = detr.detection_loss(outputs, targets)

            # =================================================================
            # Total Loss
            # =================================================================
            total_g = (
                args.lambda_contrastive * loss_contrastive
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

            # Log
            metrics_logger.update(
                loss_contrastive=loss_contrastive.item(),
                loss_wasserstein=loss_wasserstein.item(),
                loss_det=loss_det.item(),
                total_g=total_g.item(),
                w_m2f=current_w_m2f,
                beta=current_beta,
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                score_f=contrastive_info.get("score_f_mean", 0.0),
                score_m=contrastive_info.get("score_m_mean", 0.0),
                score_gap=contrastive_info.get("score_gap", 0.0),
                sim_f2m=contrastive_info.get("sim_f2m", 0.0),
                sim_f2f=contrastive_info.get("sim_f2f", 0.0),
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
                "loss_wasserstein": metrics_logger.meters["loss_wasserstein"].global_avg,
                "loss_det": metrics_logger.meters["loss_det"].global_avg,
                "total_g": metrics_logger.meters["total_g"].global_avg,
                "w_m2f": current_w_m2f,
                "beta": current_beta,
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
            print(f"  Contrastive: {log_entry['loss_contrastive']:.4f} "
                  f"(F→M: {contrastive_info.get('loss_f2m', 0):.4f}, "
                  f"M→F: {contrastive_info.get('loss_m2f', 0):.4f})")
            print(f"  Wasserstein: {log_entry['loss_wasserstein']:.4f}")
            print(f"  Detection: {log_entry['loss_det']:.4f}")
            print(f"  Total: {log_entry['total_g']:.4f}")
            print(f"  Score (F/M): {log_entry['score_f']:.4f} / {log_entry['score_m']:.4f}")
            print(f"  Score Gap (M-F): {log_entry['score_gap']:.4f}")
            print(f"  Cosine Sim: F→M={log_entry['sim_f2m']:.4f}, "
                  f"F→F={log_entry['sim_f2f']:.4f}")
            print(f"  M→F weight: {current_w_m2f:.3f}")
            print(f"  Beta: {current_beta:.4f}")
            print(f"  Samples (F/M): {log_entry['n_f_avg']:.1f} / {log_entry['n_m_avg']:.1f}")

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
        print("Controlled Aggression Contrastive (5th) Complete!")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print("\n[5th = 3rd의 Power + 수술적 안정화]")
        print("  - Adaptive Weighting 제거 → Score Gap Reversal 해결")
        print("  - M→F 감쇠 스케줄 → 초기 양방향 power + 후반 안정성")
        print("  - 나머지 전부 3rd 유지 → 검증된 gradient 엔진 보존")
        print(f"\n[M→F Schedule]")
        print(f"  ep0-{args.m2f_full_epochs-1}: w={args.m2f_weight} (full)")
        print(f"  ep{args.m2f_full_epochs}-{args.m2f_full_epochs+args.m2f_decay_epochs-1}: "
              f"w={args.m2f_weight}→0.0 (decay)")
        print(f"  ep{args.m2f_full_epochs+args.m2f_decay_epochs}+: w=0.0 (F→M only)")


if __name__ == "__main__":
    main()
