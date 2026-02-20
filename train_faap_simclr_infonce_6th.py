"""
FAAP Training - Clamped Adaptive Contrastive Learning (6th Version)

=============================================================================
5th 실패 원인 분석:
=============================================================================
- Adaptive Weighting 제거 → 초기 epoch에서 3rd 대비 Female AP 개선 약화
  (3rd ep3: Female 0.413 vs 5th ep3: Female 0.406)
- M→F 완전 제거(ep9+) → AP Gap 즉시 악화 (0.1054→0.1126)
- M→F 감쇠가 아닌 적절한 고정값(0.2)이 필요

=============================================================================
전 버전 교훈 정리:
=============================================================================
3rd: Adaptive Weighting이 초기(ep0-3)에 강력한 Female 개선 → ep3 이후 Score Gap Reversal로 붕괴
4th: Male Detach + Triple Warmup → 초기 신호 4%로 억제, AP Gap 개선 미미
5th: Adaptive 제거 → 초기 약화, M→F 완전 제거 시 악화 확인, best w_m2f=0.2

=============================================================================
6th 핵심 아이디어: 3rd의 Power + Score Gap Reversal 원천 차단
=============================================================================

[3rd에서 유지]
  1. 양방향 F→M (1.5x) + M→F (0.2x 고정)
  2. Male Detach 없음
  3. Temperature 0.07
  4. Epsilon 0.10 고정
  5. LR 1e-4 고정
  6. Medium augmentation
  7. ProjectionHead 정규화 없음

[3rd에서 수정 (수술적 변경)]
  1. Clamped Adaptive Weighting
     - 3rd: weights ∈ [0.5, 1.5] → Reversal시 약화 (0.8 등)
     - 6th: weights ∈ [1.0, 1.5] → Reversal시 균일(1.0), 정상시 증폭(>1.0)
     → Score Gap Reversal 원천 차단, 유용한 증폭은 보존
  2. M→F 고정 0.2 (3rd: 0.5, 5th: 0.5→0.0)
     - 5th 실험에서 w_m2f=0.2(ep7)이 best 확인
     - 0.5는 Male AP 간섭, 0.0은 Male AP 과잉 상승

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
# Clamped Adaptive Contrastive Loss (6th: 핵심 변경)
# =============================================================================

class ClampedAdaptiveLoss(nn.Module):
    """
    3rd의 Adaptive Weighting + Score Gap Reversal 원천 차단

    3rd 대비 변경:
    - 3rd: weights = 0.5 + sigmoid(score_diff * 5)  → [0.5, 1.5]
      - 정상: >1.0 (증폭) ← OK
      - Reversal: <1.0 (약화) ← 문제!
    - 6th: weights = 1.0 + relu(sigmoid(score_diff * 5) - 0.5)  → [1.0, 1.5]
      - 정상: >1.0 (증폭) ← 동일
      - Reversal: =1.0 (균일) ← 해결!

    M→F는 0.2로 고정 (5th ep7 best에서 검증).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        score_weight_alpha: float = 1.0,
        w_m2f: float = 0.2,
    ):
        super().__init__()
        self.temperature = temperature
        self.score_weight_alpha = score_weight_alpha
        self.w_m2f = w_m2f

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
                "sim_f2m": 0.0, "sim_f2f": 0.0, "avg_weight": 1.0,
            }

        scores_f = scores_f.detach()
        scores_m = scores_m.detach()

        # =================================================================
        # 1. F→M Contrastive + Clamped Adaptive Weighting
        # =================================================================
        sim_f2m = torch.mm(proj_f, proj_m.t()) / self.temperature  # (N_f, N_m)
        sim_f2f = torch.mm(proj_f, proj_f.t()) / self.temperature  # (N_f, N_f)

        mask_self = torch.eye(n_f, device=proj_f.device, dtype=torch.bool)
        sim_f2f_masked = sim_f2f.masked_fill(mask_self, float('-inf'))

        # Clamped Adaptive Weighting (6th 핵심)
        # score_diff > 0: score_m > score_f (정상) → weight > 1.0 (증폭)
        # score_diff < 0: score_f > score_m (역전) → weight = 1.0 (균일, 약화 방지)
        score_diff = scores_m.unsqueeze(0) - scores_f.unsqueeze(1)  # (N_f, N_m)
        raw_weights = torch.sigmoid(score_diff * 5)  # [0, 1]
        clamped_weights = 1.0 + F.relu(raw_weights - 0.5)  # [1.0, 1.5]

        sim_f2m_weighted = sim_f2m + self.score_weight_alpha * torch.log(clamped_weights + 1e-8)

        # InfoNCE
        all_sims = torch.cat([sim_f2m_weighted, sim_f2f_masked], dim=1)
        numerator = torch.logsumexp(sim_f2m_weighted, dim=1)
        denominator = torch.logsumexp(all_sims, dim=1)
        loss_f2m = -(numerator - denominator).mean()

        # =================================================================
        # 2. M→F Contrastive (고정 0.2)
        # =================================================================
        loss_m2f = proj_f.new_tensor(0.0)
        if self.w_m2f > 0 and n_m >= 2:
            sim_m2f = sim_f2m.t()  # (N_m, N_f) - unweighted
            sim_m2m = torch.mm(proj_m, proj_m.t()) / self.temperature
            mask_m = torch.eye(n_m, device=proj_m.device, dtype=torch.bool)
            sim_m2m_masked = sim_m2m.masked_fill(mask_m, float('-inf'))

            all_sims_m = torch.cat([sim_m2f, sim_m2m_masked], dim=1)
            numerator_m = torch.logsumexp(sim_m2f, dim=1)
            denominator_m = torch.logsumexp(all_sims_m, dim=1)
            loss_m2f = -(numerator_m - denominator_m).mean()

        # F→M (1.5) + M→F (0.2)
        loss = 1.5 * loss_f2m + self.w_m2f * loss_m2f

        # Monitoring
        with torch.no_grad():
            cos_f2m = torch.mm(proj_f, proj_m.t()).mean().item()
            cos_f2f_vals = torch.mm(proj_f, proj_f.t())
            cos_f2f_vals = cos_f2f_vals.masked_fill(mask_self, 0.0)
            cos_f2f_mean = cos_f2f_vals.sum().item() / max(n_f * (n_f - 1), 1)
            avg_weight = clamped_weights.mean().item()

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
            "avg_weight": avg_weight,
        }

        return loss, info


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


def _scheduled_beta(
    epoch: int, total_epochs: int,
    beta_start: float, beta_final: float,
) -> float:
    if total_epochs <= 1:
        return beta_start
    progress = min(epoch / max(1, total_epochs - 1), 1.0)
    return beta_start + (beta_final - beta_start) * progress


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "FAAP Clamped Adaptive Contrastive (6th Version)"
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

    # Perturbation (고정)
    parser.add_argument("--epsilon", type=float, default=0.10)

    # Loss weights (3rd와 동일)
    parser.add_argument("--lambda_contrastive", type=float, default=1.0)
    parser.add_argument("--lambda_wass", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--beta_final", type=float, default=0.6)

    # Contrastive settings
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--score_weight_alpha", type=float, default=1.0,
                        help="Clamped adaptive weighting 강도")
    parser.add_argument("--w_m2f", type=float, default=0.2,
                        help="M→F 고정 가중치 (5th ep7 best에서 검증)")
    parser.add_argument("--score_top_k", type=int, default=10)
    parser.add_argument("--proj_dim", type=int, default=128)

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
        print("Clamped Adaptive Contrastive Learning (6th Version)")
        print("=" * 70)
        print()
        print("[5th 교훈]")
        print("  - Adaptive Weighting 제거 → 초기 Female 개선 약화")
        print("  - M→F=0 → AP Gap 즉시 악화, M→F=0.2 → best")
        print()
        print("[6th = 3rd의 Adaptive Power + Score Gap Reversal 차단]")
        print("  1. Clamped Adaptive: [0.5,1.5] → [1.0,1.5]")
        print("     - Reversal시 약화 방지 (weight >= 1.0)")
        print("     - 정상시 증폭 유지 (weight > 1.0)")
        print("  2. M→F 고정 0.2 (5th ep7 best에서 검증)")
        print("-" * 70)
        print(f"Temperature: {args.temperature}")
        print(f"Epsilon: {args.epsilon} (fixed)")
        print(f"F→M: 1.5 (with Clamped Adaptive Weighting)")
        print(f"M→F: {args.w_m2f} (fixed)")
        print(f"Score weight alpha: {args.score_weight_alpha}")
        print(f"LR: {args.lr_g} (fixed)")
        print(f"Augmentation: {args.aug_strength}")
        print(f"Gradient clip: {args.max_norm}")
        print(f"Loss: C=1.0, W=0.2, D=0.5→0.6")
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

    contrastive_loss_fn = ClampedAdaptiveLoss(
        temperature=args.temperature,
        score_weight_alpha=args.score_weight_alpha,
        w_m2f=args.w_m2f,
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

        if utils.is_main_process():
            print(f"\n--- Epoch {epoch} | beta={current_beta:.3f} ---")

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
            # 1. Clamped Adaptive Contrastive Loss
            # =================================================================
            image_scores = _image_level_detection_score(outputs, top_k=args.score_top_k)

            proj_all = proj_head(features)
            proj_f = proj_all[female_idx]
            proj_m = proj_all[male_idx]
            scores_f = image_scores[female_idx]
            scores_m = image_scores[male_idx]

            loss_contrastive, contrastive_info = contrastive_loss_fn(
                proj_f, proj_m, scores_f, scores_m,
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
                beta=current_beta,
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                score_f=contrastive_info.get("score_f_mean", 0.0),
                score_m=contrastive_info.get("score_m_mean", 0.0),
                score_gap=contrastive_info.get("score_gap", 0.0),
                sim_f2m=contrastive_info.get("sim_f2m", 0.0),
                sim_f2f=contrastive_info.get("sim_f2f", 0.0),
                avg_weight=contrastive_info.get("avg_weight", 1.0),
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
                "beta": current_beta,
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "score_f": metrics_logger.meters["score_f"].global_avg,
                "score_m": metrics_logger.meters["score_m"].global_avg,
                "score_gap": metrics_logger.meters["score_gap"].global_avg,
                "sim_f2m": metrics_logger.meters["sim_f2m"].global_avg,
                "sim_f2f": metrics_logger.meters["sim_f2f"].global_avg,
                "avg_weight": metrics_logger.meters["avg_weight"].global_avg,
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
            print(f"  Avg Adaptive Weight: {log_entry['avg_weight']:.4f} "
                  f"(clamped [1.0, 1.5])")
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
        print("Clamped Adaptive Contrastive (6th) Complete!")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print("\n[6th = 3rd의 Adaptive Power + SGR 차단]")
        print("  - Clamped [1.0,1.5]: Reversal시 약화 방지, 정상시 증폭 유지")
        print(f"  - M→F 고정 {args.w_m2f}: 5th ep7에서 검증된 최적값")
        print("  - 나머지 전부 3rd 유지: 검증된 gradient 엔진 보존")


if __name__ == "__main__":
    main()
