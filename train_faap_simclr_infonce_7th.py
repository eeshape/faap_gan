"""
FAAP Training - Dual Gap Direct Minimization (7th Version)

=============================================================================
3rd~6th 실패 분석 종합:
=============================================================================
- 3rd (Best): AP Gap -1.8%, AR Gap -67.9% @ ep3
- 4th~6th: 3rd의 loss 변형 시도 → 모두 3rd 미달
- 공통 문제: Contrastive Loss는 AP Gap의 간접적 proxy
  → Feature 유사 ≠ AP 동일
  → Perturbation이 "범용 이미지 향상기"로 작동
  → Male/Female 모두 비슷하게 개선 → Gap 거의 불변

=============================================================================
7th 핵심 전략: 간접 Proxy → 직접 Gap 공격
=============================================================================

[근본적 전환]
- 기존: Contrastive(주력) + Wasserstein(보조 0.2) + Detection(0.5)
- 7th:  ScoreGap(주력 1.0) + RecallGap(0.5) + Contrastive(보조 0.3) + Detection(0.3)

[ScoreGapLoss] - AP Gap 직접 공격
- loss = relu(mean_score_m - mean_score_f)^2
- Male score > Female score일 때만 패널티
- AP는 confidence score에 직접 의존 → score gap 줄이면 AP gap 감소

[RecallGapLoss] - AR Gap 직접 공격
- Soft detection count: sigmoid((score - threshold) * sharpness)
- loss = relu(soft_count_m - soft_count_f)^2
- Male detection 수 > Female detection 수일 때만 패널티

[Contrastive (보조)]
- 3rd의 Adaptive Weighting 그대로 유지 (검증된 설계)
- 가중치 0.3으로 약화 → feature 정규화 역할만

[Detection Loss (약화)]
- 0.5→0.6 → 0.3 고정
- Generator에 더 많은 자유도 부여

[Cosine LR Decay]
- 초반 강한 학습 + 후반 안정화
- 3rd~6th 공통 문제인 "후반 악화" 방지
=============================================================================
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

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
# SimCLR-Style Projection Head (3rd와 동일)
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
# Score Gap Loss (AP Gap 직접 공격)
# =============================================================================

class ScoreGapLoss(nn.Module):
    """
    Detection score의 성별 격차를 직접 최소화.
    AP는 confidence score에 의존하므로, score gap 감소 → AP gap 감소.

    loss = relu(mean_score_m - mean_score_f)^2
    - Male > Female일 때만 패널티 (hinge-style)
    - 양쪽이 같거나 Female이 높으면 loss = 0
    """
    def forward(
        self,
        scores_f: torch.Tensor,
        scores_m: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        if scores_f.numel() == 0 or scores_m.numel() == 0:
            return scores_f.new_tensor(0.0), {"score_gap": 0.0}

        mean_f = scores_f.mean()
        mean_m = scores_m.mean()
        gap = mean_m - mean_f  # positive = Male이 높음

        # Hinge: Male > Female일 때만 패널티
        loss = F.relu(gap) ** 2

        return loss, {
            "score_gap_raw": gap.item(),
            "score_gap_loss": loss.item(),
            "score_f_mean": mean_f.item(),
            "score_m_mean": mean_m.item(),
        }


# =============================================================================
# Recall Gap Loss (AR Gap 직접 공격)
# =============================================================================

class RecallGapLoss(nn.Module):
    """
    Detection recall의 성별 격차를 직접 최소화.
    Soft counting으로 differentiable하게 detection 수 비교.

    soft_count = sigmoid((score - threshold) * sharpness).mean()
    loss = relu(soft_count_m - soft_count_f)^2
    """
    def __init__(self, threshold: float = 0.3, sharpness: float = 10.0):
        super().__init__()
        self.threshold = threshold
        self.sharpness = sharpness

    def forward(
        self,
        all_scores_f: torch.Tensor,  # (N_f, num_queries) 모든 query의 score
        all_scores_m: torch.Tensor,  # (N_m, num_queries)
    ) -> Tuple[torch.Tensor, dict]:
        if all_scores_f.numel() == 0 or all_scores_m.numel() == 0:
            return all_scores_f.new_tensor(0.0), {"recall_gap": 0.0}

        # Soft detection count per image
        soft_det_f = torch.sigmoid(
            (all_scores_f - self.threshold) * self.sharpness
        ).mean(dim=-1).mean()  # scalar

        soft_det_m = torch.sigmoid(
            (all_scores_m - self.threshold) * self.sharpness
        ).mean(dim=-1).mean()  # scalar

        gap = soft_det_m - soft_det_f  # positive = Male이 더 많이 detect

        loss = F.relu(gap) ** 2

        return loss, {
            "recall_gap_raw": gap.item(),
            "recall_gap_loss": loss.item(),
            "soft_det_f": soft_det_f.item(),
            "soft_det_m": soft_det_m.item(),
        }


# =============================================================================
# Gender-Aware Contrastive Loss (3rd 그대로, 보조 역할)
# =============================================================================

class GenderAwareContrastiveLoss(nn.Module):
    """3rd의 Adaptive Weighting 그대로 유지 (검증된 설계). 7th에서는 보조 역할."""

    def __init__(
        self,
        temperature: float = 0.07,
        score_weight_alpha: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.score_weight_alpha = score_weight_alpha

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
            return proj_f.new_tensor(0.0), {"n_f": n_f, "n_m": n_m}

        scores_f = scores_f.detach()
        scores_m = scores_m.detach()

        # F→M Contrastive
        sim_f2m = torch.mm(proj_f, proj_m.t()) / self.temperature
        sim_f2f = torch.mm(proj_f, proj_f.t()) / self.temperature
        mask_self = torch.eye(n_f, device=proj_f.device, dtype=torch.bool)
        sim_f2f_masked = sim_f2f.masked_fill(mask_self, float('-inf'))

        # Adaptive Weighting [0.5, 1.5]
        score_diff = scores_m.unsqueeze(0) - scores_f.unsqueeze(1)
        score_diff_normalized = torch.sigmoid(score_diff * 5)
        weights = 0.5 + score_diff_normalized

        sim_f2m_weighted = sim_f2m + self.score_weight_alpha * torch.log(weights + 1e-8)

        # InfoNCE
        all_sims = torch.cat([sim_f2m_weighted, sim_f2f_masked], dim=1)
        numerator = torch.logsumexp(sim_f2m_weighted, dim=1)
        denominator = torch.logsumexp(all_sims, dim=1)
        loss_f2m = -(numerator - denominator).mean()

        # M→F (약하게)
        if n_m >= 2:
            sim_m2f = sim_f2m.t()
            sim_m2m = torch.mm(proj_m, proj_m.t()) / self.temperature
            mask_m = torch.eye(n_m, device=proj_m.device, dtype=torch.bool)
            sim_m2m_masked = sim_m2m.masked_fill(mask_m, float('-inf'))

            all_sims_m = torch.cat([sim_m2f, sim_m2m_masked], dim=1)
            numerator_m = torch.logsumexp(sim_m2f, dim=1)
            denominator_m = torch.logsumexp(all_sims_m, dim=1)
            loss_m2f = -(numerator_m - denominator_m).mean()
        else:
            loss_m2f = proj_f.new_tensor(0.0)

        loss = 1.5 * loss_f2m + 0.5 * loss_m2f

        info = {
            "n_f": n_f,
            "n_m": n_m,
            "loss_f2m": loss_f2m.item(),
            "loss_m2f": loss_m2f.item() if isinstance(loss_m2f, torch.Tensor) else 0.0,
        }

        return loss, info


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


def _per_query_detection_scores(outputs: dict) -> torch.Tensor:
    """DETR logits에서 query별 max score (RecallGapLoss용)"""
    probs = outputs["pred_logits"].softmax(dim=-1)[..., :-1]  # (B, num_queries, num_classes)
    max_probs = probs.max(dim=-1).values  # (B, num_queries)
    return max_probs


def _default_output_dir(script_path: Path) -> str:
    stem = script_path.stem
    for prefix in ("train_faap_simclr_", "train_faap_", "train_"):
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix):]
            break
    return str(Path("faap_outputs") / f"faap_outputs_{stem.lower()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FAAP Dual Gap Direct Minimization (7th)")

    # Paths
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()))
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(Path(__file__)))

    # Training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)

    # Perturbation
    parser.add_argument("--epsilon", type=float, default=0.10)

    # Loss weights (7th: Gap Loss 주력, Contrastive 보조)
    parser.add_argument("--lambda_score_gap", type=float, default=1.0,
                        help="Score Gap Loss weight (AP Gap 직접 공격)")
    parser.add_argument("--lambda_recall_gap", type=float, default=0.5,
                        help="Recall Gap Loss weight (AR Gap 직접 공격)")
    parser.add_argument("--lambda_contrastive", type=float, default=0.3,
                        help="Contrastive Loss weight (feature 정규화, 보조)")
    parser.add_argument("--lambda_det", type=float, default=0.3,
                        help="Detection Loss weight (검출 품질 유지)")

    # Recall Gap settings
    parser.add_argument("--recall_threshold", type=float, default=0.3,
                        help="Soft detection threshold for RecallGapLoss")
    parser.add_argument("--recall_sharpness", type=float, default=10.0,
                        help="Sigmoid sharpness for soft counting")

    # Contrastive settings (3rd와 동일)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--score_weight_alpha", type=float, default=1.0)
    parser.add_argument("--score_top_k", type=int, default=10)
    parser.add_argument("--proj_dim", type=int, default=128)

    # Augmentation
    parser.add_argument("--aug_strength", type=str, default="medium",
                        choices=["none", "weak", "medium", "strong"])

    # Other
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


def _apply_generator(generator: nn.Module, samples: NestedTensor) -> NestedTensor:
    tensors = samples.tensors
    delta = generator(tensors)
    perturbed = clamp_normalized(tensors + delta)
    return NestedTensor(perturbed, samples.mask)


def _unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def _cosine_lr(epoch: int, total_epochs: int, lr_max: float, lr_min: float) -> float:
    """Cosine annealing LR schedule"""
    if total_epochs <= 1:
        return lr_max
    progress = min(epoch / max(1, total_epochs - 1), 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


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
        print("Dual Gap Direct Minimization (7th Version)")
        print("=" * 70)
        print()
        print("[3rd~6th 실패 교훈]")
        print("  - Contrastive Loss = AP Gap의 간접 proxy → 1.8%밖에 감소 못함")
        print("  - Feature 유사 ≠ AP 동일")
        print("  - Perturbation이 범용 이미지 향상기로 작동")
        print()
        print("[7th = Gap을 직접 공격]")
        print("  1. ScoreGapLoss: relu(mean_score_m - mean_score_f)^2")
        print("     → AP Gap 직접 최소화")
        print("  2. RecallGapLoss: relu(soft_count_m - soft_count_f)^2")
        print("     → AR Gap 직접 최소화")
        print("  3. Contrastive: 3rd의 Adaptive Weighting (보조)")
        print("  4. Detection: 약화 (Generator 자유도 증가)")
        print("  5. Cosine LR: 후반 악화 방지")
        print("-" * 70)
        print(f"Temperature: {args.temperature}")
        print(f"Epsilon: {args.epsilon} (fixed)")
        print(f"LR: {args.lr_g} → {args.lr_min} (cosine)")
        print(f"Loss: ScoreGap={args.lambda_score_gap}, RecallGap={args.lambda_recall_gap}, "
              f"Contrastive={args.lambda_contrastive}, Det={args.lambda_det}")
        print(f"Recall threshold: {args.recall_threshold}, sharpness: {args.recall_sharpness}")
        print(f"Augmentation: {args.aug_strength}")
        print(f"Gradient clip: {args.max_norm}")
        print("=" * 70)

    # ==========================================================================
    # Model Initialization
    # ==========================================================================

    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)

    proj_head = SimCLRProjectionHead(
        input_dim=detr.hidden_dim,
        hidden_dim=detr.hidden_dim,
        output_dim=args.proj_dim,
    ).to(device)

    contrastive_loss_fn = GenderAwareContrastiveLoss(
        temperature=args.temperature,
        score_weight_alpha=args.score_weight_alpha,
    ).to(device)

    score_gap_loss_fn = ScoreGapLoss()
    recall_gap_loss_fn = RecallGapLoss(
        threshold=args.recall_threshold,
        sharpness=args.recall_sharpness,
    )

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

        # Cosine LR schedule
        current_lr = _cosine_lr(epoch, args.epochs, args.lr_g, args.lr_min)
        for pg in opt_g.param_groups:
            pg["lr"] = current_lr

        if utils.is_main_process():
            print(f"\n--- Epoch {epoch} | lr={current_lr:.6f} ---")

        for samples, targets, genders in metrics_logger.log_every(
            train_loader, args.log_every, f"Epoch {epoch}"
        ):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            genders = [g.lower() for g in genders]

            # Gender split indices
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
            # Score 계산
            # =================================================================
            image_scores = _image_level_detection_score(outputs, top_k=args.score_top_k)
            per_query_scores = _per_query_detection_scores(outputs)

            scores_f = image_scores[female_idx]
            scores_m = image_scores[male_idx]
            query_scores_f = per_query_scores[female_idx]  # (N_f, num_queries)
            query_scores_m = per_query_scores[male_idx]    # (N_m, num_queries)

            # =================================================================
            # 1. Score Gap Loss (AP Gap 직접 공격) - 주력
            # =================================================================
            loss_score_gap, score_gap_info = score_gap_loss_fn(scores_f, scores_m)

            # =================================================================
            # 2. Recall Gap Loss (AR Gap 직접 공격)
            # =================================================================
            loss_recall_gap, recall_gap_info = recall_gap_loss_fn(
                query_scores_f, query_scores_m
            )

            # =================================================================
            # 3. Contrastive Loss (보조: feature 정규화)
            # =================================================================
            proj_all = proj_head(features)
            proj_f = proj_all[female_idx]
            proj_m = proj_all[male_idx]

            loss_contrastive, contrastive_info = contrastive_loss_fn(
                proj_f, proj_m, scores_f.detach(), scores_m.detach()
            )

            # =================================================================
            # 4. Detection Loss (약화: 검출 품질 유지)
            # =================================================================
            loss_det, _ = detr.detection_loss(outputs, targets)

            # =================================================================
            # Total Loss
            # =================================================================
            total_g = (
                args.lambda_score_gap * loss_score_gap
                + args.lambda_recall_gap * loss_recall_gap
                + args.lambda_contrastive * loss_contrastive
                + args.lambda_det * loss_det
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
                    args.max_norm
                )
            opt_g.step()

            # Log
            metrics_logger.update(
                loss_score_gap=loss_score_gap.item(),
                loss_recall_gap=loss_recall_gap.item(),
                loss_contrastive=loss_contrastive.item(),
                loss_det=loss_det.item(),
                total_g=total_g.item(),
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                score_f=score_gap_info.get("score_f_mean", 0.0),
                score_m=score_gap_info.get("score_m_mean", 0.0),
                score_gap=score_gap_info.get("score_gap_raw", 0.0),
                recall_gap=recall_gap_info.get("recall_gap_raw", 0.0),
                soft_det_f=recall_gap_info.get("soft_det_f", 0.0),
                soft_det_m=recall_gap_info.get("soft_det_m", 0.0),
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
                "lr": current_lr,
                "loss_score_gap": metrics_logger.meters["loss_score_gap"].global_avg,
                "loss_recall_gap": metrics_logger.meters["loss_recall_gap"].global_avg,
                "loss_contrastive": metrics_logger.meters["loss_contrastive"].global_avg,
                "loss_det": metrics_logger.meters["loss_det"].global_avg,
                "total_g": metrics_logger.meters["total_g"].global_avg,
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "score_f": metrics_logger.meters["score_f"].global_avg,
                "score_m": metrics_logger.meters["score_m"].global_avg,
                "score_gap": metrics_logger.meters["score_gap"].global_avg,
                "recall_gap": metrics_logger.meters["recall_gap"].global_avg,
                "soft_det_f": metrics_logger.meters["soft_det_f"].global_avg,
                "soft_det_m": metrics_logger.meters["soft_det_m"].global_avg,
                "n_f_avg": metrics_logger.meters["n_f"].global_avg,
                "n_m_avg": metrics_logger.meters["n_m"].global_avg,
            }

            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(f"\n[Epoch {epoch}] Summary:")
            print(f"  Score Gap Loss: {log_entry['loss_score_gap']:.6f}")
            print(f"  Recall Gap Loss: {log_entry['loss_recall_gap']:.6f}")
            print(f"  Contrastive Loss: {log_entry['loss_contrastive']:.4f}")
            print(f"  Detection Loss: {log_entry['loss_det']:.4f}")
            print(f"  Total: {log_entry['total_g']:.4f}")
            print(f"  Score (F/M): {log_entry['score_f']:.4f} / {log_entry['score_m']:.4f}")
            print(f"  Score Gap (M-F): {log_entry['score_gap']:.4f}")
            print(f"  Recall Gap (M-F): {log_entry['recall_gap']:.4f}")
            print(f"  Soft Det (F/M): {log_entry['soft_det_f']:.4f} / {log_entry['soft_det_m']:.4f}")
            print(f"  LR: {current_lr:.6f}")

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
        print("Dual Gap Direct Minimization Complete!")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print("\n[7th 핵심 변경]")
        print("  - 간접 proxy (Contrastive) → 직접 Gap 공격")
        print("  - ScoreGapLoss + RecallGapLoss = AP/AR Gap 동시 최소화")
        print("  - Cosine LR로 후반 악화 방지")
        print("\n목표: AP Gap = 0, AR Gap = 0")


if __name__ == "__main__":
    main()
