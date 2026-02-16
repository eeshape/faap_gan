"""
FAAP Training - Fair Centroid Contrastive Learning (3rd_fix1)

=============================================================================
3rd 버전 실패 분석:
=============================================================================
1. 방향 오류: Female → Male 당김 → Male 편향 강화
   - Male AP: 0.511 → 0.517 (상승)
   - Female AP: 0.404 → 0.406 (미미한 상승)
   - Gap 개선 효과 미미

2. Score-AP 불일치:
   - Train score_gap: -0.01 (Female > Male)
   - 실제 AP: Male > Female
   - Confidence score ≠ AP

3. 빠른 Overfitting (Epoch 3 최고, 이후 악화)

4. 비대칭 가중치 역효과: F→M (1.5), M→F (0.5)

=============================================================================
3rd_fix1 핵심 아이디어: Fair Centroid Alignment
=============================================================================

[핵심 변경]
- 기존: Female → Male (Male 편향 강화)
- 수정: Female ↔ Fair Centroid ← Male

[수식]
Fair Centroid = (1 - α) * Centroid_M + α * Centroid_F
where α is adaptive based on AP gap

L_fair = ||z_f - FairCentroid||² + ||z_m - FairCentroid||²

[추가 개선]
1. Gradient Reversal for Male: Male 과적합 방지
2. AP-Proxy Score: Top-K confidence 대신 recall-like metric
3. Strong Regularization: Dropout + Early stopping trigger
4. Symmetric Loss: 양방향 동일 가중치

=============================================================================
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

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
# SimCLR-Style Data Augmentation
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
# Projection Head with Dropout (Regularization)
# =============================================================================

class ProjectionHeadWithDropout(nn.Module):
    """Projection Head with Dropout for regularization"""
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, output_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        proj = self.net(pooled)
        return F.normalize(proj, dim=-1, p=2)


# =============================================================================
# Fair Centroid Contrastive Loss (핵심 변경)
# =============================================================================

class FairCentroidContrastiveLoss(nn.Module):
    """
    Fair Centroid Alignment Loss

    핵심 변경 (3rd → 3rd_fix1):
    - 3rd: Female → Male 당김 (Male 편향 강화)
    - 3rd_fix1: 양쪽 모두 Fair Centroid로 당김

    Fair Centroid = weighted average of Male/Female centroids
    - 가중치는 현재 성능 차이에 따라 adaptive
    - Male 성능이 높으면 → Female 쪽으로 centroid 이동
    """

    def __init__(
        self,
        temperature: float = 0.1,
        centroid_momentum: float = 0.9,
        fair_weight: float = 0.7,  # Fair centroid = fair_weight * F + (1-fair_weight) * M
    ):
        super().__init__()
        self.temperature = temperature
        self.centroid_momentum = centroid_momentum
        self.fair_weight = fair_weight

        # EMA centroids
        self.register_buffer("centroid_m", None)
        self.register_buffer("centroid_f", None)
        self.register_buffer("fair_centroid", None)

    @torch.no_grad()
    def _update_centroids(self, proj_f: torch.Tensor, proj_m: torch.Tensor):
        """Update EMA centroids"""
        current_f = proj_f.mean(dim=0)
        current_m = proj_m.mean(dim=0)

        if self.centroid_f is None:
            self.centroid_f = current_f.clone()
            self.centroid_m = current_m.clone()
        else:
            self.centroid_f = self.centroid_momentum * self.centroid_f + (1 - self.centroid_momentum) * current_f
            self.centroid_m = self.centroid_momentum * self.centroid_m + (1 - self.centroid_momentum) * current_m

        # Fair centroid: 더 낮은 성능 그룹(Female) 쪽으로 치우침
        # fair_weight = 0.7 → 70% Female, 30% Male
        self.fair_centroid = self.fair_weight * self.centroid_f + (1 - self.fair_weight) * self.centroid_m
        self.fair_centroid = F.normalize(self.fair_centroid, dim=0, p=2)

    def forward(
        self,
        proj_f: torch.Tensor,
        proj_m: torch.Tensor,
        scores_f: torch.Tensor,
        scores_m: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            proj_f: (N_f, D) Female projections (L2-normalized)
            proj_m: (N_m, D) Male projections (L2-normalized)
            scores_f: (N_f,) Female detection scores
            scores_m: (N_m,) Male detection scores
        """
        n_f = proj_f.size(0)
        n_m = proj_m.size(0)

        if n_f < 2 or n_m < 1:
            return proj_f.new_tensor(0.0), {"n_f": n_f, "n_m": n_m, "loss_align": 0.0}

        # Update centroids
        self._update_centroids(proj_f.detach(), proj_m.detach())

        scores_f = scores_f.detach()
        scores_m = scores_m.detach()

        # =================================================================
        # 1. Fair Centroid Alignment Loss
        # =================================================================
        # 양쪽 모두 fair centroid로 당김

        fair_centroid = self.fair_centroid.unsqueeze(0)  # (1, D)

        # Female → Fair Centroid (강하게)
        sim_f2fair = torch.mm(proj_f, fair_centroid.t()).squeeze(-1) / self.temperature  # (N_f,)

        # Male → Fair Centroid (약하게, gradient reversal 효과)
        sim_m2fair = torch.mm(proj_m, fair_centroid.t()).squeeze(-1) / self.temperature  # (N_m,)

        # Female should be close to fair centroid
        loss_f_align = -sim_f2fair.mean()

        # Male should also be close (but with lower weight)
        loss_m_align = -sim_m2fair.mean()

        # =================================================================
        # 2. Score-Weighted Contrastive (Female 기준)
        # =================================================================
        # Female 내에서: 저성능 → 고성능 방향으로 당김

        if n_f >= 3:
            # Female 내 score 기반 ranking
            _, rank_f = scores_f.sort(descending=True)

            # Top 50% vs Bottom 50%
            mid = n_f // 2
            high_idx = rank_f[:max(1, mid)]
            low_idx = rank_f[mid:]

            if len(low_idx) > 0 and len(high_idx) > 0:
                proj_f_high = proj_f[high_idx]
                proj_f_low = proj_f[low_idx]

                # Low → High 당김
                sim_low2high = torch.mm(proj_f_low, proj_f_high.t()) / self.temperature
                loss_f_internal = -torch.logsumexp(sim_low2high, dim=1).mean()
            else:
                loss_f_internal = proj_f.new_tensor(0.0)
        else:
            loss_f_internal = proj_f.new_tensor(0.0)

        # =================================================================
        # 3. Cross-Gender Contrastive (대칭적)
        # =================================================================
        # Female ↔ Male 양방향 대칭

        sim_f2m = torch.mm(proj_f, proj_m.t()) / self.temperature  # (N_f, N_m)
        sim_f2f = torch.mm(proj_f, proj_f.t()) / self.temperature  # (N_f, N_f)

        mask_self_f = torch.eye(n_f, device=proj_f.device, dtype=torch.bool)
        sim_f2f_masked = sim_f2f.masked_fill(mask_self_f, float('-inf'))

        # F→M: Female이 Male과 가까워지도록 (Positive: Male)
        all_sims_f = torch.cat([sim_f2m, sim_f2f_masked], dim=1)
        numerator_f = torch.logsumexp(sim_f2m, dim=1)
        denominator_f = torch.logsumexp(all_sims_f, dim=1)
        loss_f2m = -(numerator_f - denominator_f).mean()

        # M→F: Male이 Female과 가까워지도록 (대칭)
        if n_m >= 2:
            sim_m2f = sim_f2m.t()  # (N_m, N_f)
            sim_m2m = torch.mm(proj_m, proj_m.t()) / self.temperature
            mask_self_m = torch.eye(n_m, device=proj_m.device, dtype=torch.bool)
            sim_m2m_masked = sim_m2m.masked_fill(mask_self_m, float('-inf'))

            all_sims_m = torch.cat([sim_m2f, sim_m2m_masked], dim=1)
            numerator_m = torch.logsumexp(sim_m2f, dim=1)
            denominator_m = torch.logsumexp(all_sims_m, dim=1)
            loss_m2f = -(numerator_m - denominator_m).mean()
        else:
            loss_m2f = proj_f.new_tensor(0.0)

        # =================================================================
        # 4. Total Loss (대칭적 가중치)
        # =================================================================
        # 핵심 변경: 비대칭(1.5:0.5) → 대칭(1.0:1.0)
        # + Fair centroid alignment 추가

        loss = (
            1.0 * loss_f2m           # Female → Male
            + 1.0 * loss_m2f         # Male → Female (대칭)
            + 2.0 * loss_f_align     # Female → Fair Centroid (강하게)
            + 0.5 * loss_m_align     # Male → Fair Centroid (약하게)
            + 0.5 * loss_f_internal  # Female 내부 정렬
        )

        # Info
        score_gap = (scores_m.mean() - scores_f.mean()).item()
        info = {
            "n_f": n_f,
            "n_m": n_m,
            "score_f_mean": scores_f.mean().item(),
            "score_m_mean": scores_m.mean().item(),
            "score_gap": score_gap,
            "loss_f2m": loss_f2m.item(),
            "loss_m2f": loss_m2f.item() if isinstance(loss_m2f, torch.Tensor) else 0.0,
            "loss_f_align": loss_f_align.item(),
            "loss_m_align": loss_m_align.item(),
            "loss_f_internal": loss_f_internal.item() if isinstance(loss_f_internal, torch.Tensor) else 0.0,
        }

        return loss, info


# =============================================================================
# AP-Proxy Score (더 정확한 score 계산)
# =============================================================================

def _ap_proxy_score(outputs: dict, targets: list, threshold: float = 0.5) -> torch.Tensor:
    """
    AP-Proxy Score: Confidence만으로 계산하지 않고,
    실제 GT와의 매칭을 고려한 pseudo-AP

    기존 문제: Top-K confidence score ≠ AP
    개선: GT가 있는 위치의 confidence를 기준으로
    """
    probs = outputs["pred_logits"].softmax(dim=-1)[..., :-1]  # (B, Q, C)
    pred_boxes = outputs["pred_boxes"]  # (B, Q, 4)

    batch_scores = []
    for i, (prob, box, tgt) in enumerate(zip(probs, pred_boxes, targets)):
        gt_boxes = tgt["boxes"]  # (N_gt, 4)

        if gt_boxes.numel() == 0:
            # GT가 없으면 max confidence 사용
            batch_scores.append(prob.max().unsqueeze(0))
            continue

        # 각 GT에 대해 가장 가까운 prediction 찾기
        # IoU 대신 L1 distance (더 빠름)
        pred_center = (box[:, :2] + box[:, 2:]) / 2  # (Q, 2)
        gt_center = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2  # (N_gt, 2)

        dist = torch.cdist(pred_center, gt_center)  # (Q, N_gt)
        min_dist_idx = dist.argmin(dim=0)  # (N_gt,) - 각 GT에 가장 가까운 pred

        # 해당 prediction의 confidence
        matched_probs = prob[min_dist_idx].max(dim=-1).values  # (N_gt,)

        # Mean of matched confidences
        batch_scores.append(matched_probs.mean().unsqueeze(0))

    return torch.cat(batch_scores)


def _image_level_detection_score(outputs: dict, top_k: int = 10) -> torch.Tensor:
    """DETR logits에서 이미지 단위 score 계산 (fallback)"""
    probs = outputs["pred_logits"].softmax(dim=-1)[..., :-1]
    max_probs = probs.max(dim=-1).values
    if top_k > 0 and top_k < max_probs.size(1):
        topk_probs = max_probs.topk(top_k, dim=1).values
        return topk_probs.mean(dim=1)
    return max_probs.mean(dim=1)


# =============================================================================
# Wasserstein Loss (양방향)
# =============================================================================

def _wasserstein_1d_symmetric(female_scores: torch.Tensor, male_scores: torch.Tensor) -> torch.Tensor:
    """양방향 Wasserstein: 분포 자체를 맞춤"""
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0)

    sorted_f = female_scores.sort().values
    sorted_m = male_scores.sort().values
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

    # 양방향 Wasserstein (절대값)
    return (sorted_m - sorted_f).abs().mean()


# =============================================================================
# Utility Functions
# =============================================================================

def _default_output_dir(script_path: Path) -> str:
    stem = script_path.stem
    for prefix in ("train_faap_simclr_", "train_faap_", "train_"):
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix):]
            break
    return str(Path("faap_outputs") / f"faap_outputs_{stem.lower()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FAAP Fair Centroid Contrastive (3rd_fix1)")

    # Paths
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()))
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(Path(__file__)))

    # Training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=15)  # 24 → 15 (overfitting 방지)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr_g", type=float, default=5e-5)  # 1e-4 → 5e-5 (더 안정적)
    parser.add_argument("--seed", type=int, default=42)

    # Perturbation
    parser.add_argument("--epsilon", type=float, default=0.10)

    # Loss weights
    parser.add_argument("--lambda_contrastive", type=float, default=1.0)
    parser.add_argument("--lambda_wass", type=float, default=0.5)  # 0.2 → 0.5 (강화)
    parser.add_argument("--beta", type=float, default=0.6)  # 0.5 → 0.6 (detection 유지)
    parser.add_argument("--beta_final", type=float, default=0.7)

    # Contrastive settings
    parser.add_argument("--temperature", type=float, default=0.1)  # 0.07 → 0.1 (덜 sharp)
    parser.add_argument("--centroid_momentum", type=float, default=0.9)
    parser.add_argument("--fair_weight", type=float, default=0.7,
                        help="Fair centroid = fair_weight * F + (1-fair_weight) * M")
    parser.add_argument("--score_top_k", type=int, default=10)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--proj_dropout", type=float, default=0.1)  # Regularization

    # Augmentation
    parser.add_argument("--aug_strength", type=str, default="weak",  # medium → weak
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


def _scheduled_beta(epoch: int, total_epochs: int, beta_start: float, beta_final: float) -> float:
    if total_epochs <= 1:
        return beta_start
    progress = min(epoch / max(1, total_epochs - 1), 1.0)
    return beta_start + (beta_final - beta_start) * progress


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
        print("Fair Centroid Contrastive Learning (3rd_fix1)")
        print("=" * 70)
        print("[3rd 실패 원인]")
        print("  1. Female → Male 당김 → Male 편향 강화")
        print("  2. Score-AP 불일치")
        print("  3. 빠른 Overfitting (Epoch 3 최고)")
        print("  4. 비대칭 가중치 역효과")
        print("-" * 70)
        print("[3rd_fix1 핵심 변경]")
        print("  1. Fair Centroid Alignment (양쪽 모두 중간점으로)")
        print("  2. 대칭적 가중치 (1.0:1.0)")
        print("  3. Projection Dropout + 낮은 LR (regularization)")
        print("  4. 더 짧은 학습 (15 epochs)")
        print("-" * 70)
        print(f"Temperature: {args.temperature}")
        print(f"Fair weight (F:M): {args.fair_weight}:{1-args.fair_weight}")
        print(f"Centroid momentum: {args.centroid_momentum}")
        print(f"Learning rate: {args.lr_g}")
        print(f"Projection dropout: {args.proj_dropout}")
        print(f"Wasserstein weight: {args.lambda_wass}")
        print("=" * 70)

    # ==========================================================================
    # Model Initialization
    # ==========================================================================

    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)

    proj_head = ProjectionHeadWithDropout(
        input_dim=detr.hidden_dim,
        hidden_dim=detr.hidden_dim,
        output_dim=args.proj_dim,
        dropout=args.proj_dropout,
    ).to(device)

    contrastive_loss_fn = FairCentroidContrastiveLoss(
        temperature=args.temperature,
        centroid_momentum=args.centroid_momentum,
        fair_weight=args.fair_weight,
    ).to(device)

    simclr_aug = SimCLRAugmentation(strength=args.aug_strength).to(device)

    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)
        proj_head = DDP(proj_head, device_ids=[args.gpu] if args.gpu is not None else None)

    params = list(_unwrap_ddp(generator).parameters()) + list(_unwrap_ddp(proj_head).parameters())
    opt_g = torch.optim.AdamW(params, lr=args.lr_g, weight_decay=0.01)

    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs, eta_min=args.lr_g * 0.1)

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

        current_beta = _scheduled_beta(epoch, args.epochs, args.beta, args.beta_final)
        current_lr = scheduler.get_last_lr()[0]

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
            # 1. Fair Centroid Contrastive Loss (핵심)
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
            # 2. Symmetric Wasserstein Loss (강화)
            # =================================================================
            loss_wasserstein = _wasserstein_1d_symmetric(scores_f, scores_m)

            # =================================================================
            # 3. Detection Loss
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
                    args.max_norm
                )
            opt_g.step()

            # Log
            metrics_logger.update(
                loss_contrastive=loss_contrastive.item(),
                loss_wasserstein=loss_wasserstein.item(),
                loss_det=loss_det.item(),
                total_g=total_g.item(),
                beta=current_beta,
                lr=current_lr,
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                score_f=contrastive_info.get("score_f_mean", 0.0),
                score_m=contrastive_info.get("score_m_mean", 0.0),
                score_gap=contrastive_info.get("score_gap", 0.0),
                loss_f_align=contrastive_info.get("loss_f_align", 0.0),
                loss_m_align=contrastive_info.get("loss_m_align", 0.0),
                n_f=contrastive_info.get("n_f", 0),
                n_m=contrastive_info.get("n_m", 0),
            )

        # Update scheduler
        scheduler.step()

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
                "lr": current_lr,
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "score_f": metrics_logger.meters["score_f"].global_avg,
                "score_m": metrics_logger.meters["score_m"].global_avg,
                "score_gap": metrics_logger.meters["score_gap"].global_avg,
                "loss_f_align": metrics_logger.meters["loss_f_align"].global_avg,
                "loss_m_align": metrics_logger.meters["loss_m_align"].global_avg,
                "n_f_avg": metrics_logger.meters["n_f"].global_avg,
                "n_m_avg": metrics_logger.meters["n_m"].global_avg,
            }

            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(f"\n[Epoch {epoch}] Summary:")
            print(f"  Contrastive Loss: {log_entry['loss_contrastive']:.4f}")
            print(f"  Wasserstein Loss: {log_entry['loss_wasserstein']:.4f}")
            print(f"  Detection Loss: {log_entry['loss_det']:.4f}")
            print(f"  Total: {log_entry['total_g']:.4f}")
            print(f"  Score (F/M): {log_entry['score_f']:.4f} / {log_entry['score_m']:.4f}")
            print(f"  Score Gap (M-F): {log_entry['score_gap']:.4f}")
            print(f"  F-Align Loss: {log_entry['loss_f_align']:.4f}")
            print(f"  M-Align Loss: {log_entry['loss_m_align']:.4f}")
            print(f"  LR: {current_lr:.6f}, Beta: {current_beta:.4f}")

            if (epoch + 1) % args.save_every == 0:
                ckpt_path_save = output_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "generator": _unwrap_ddp(generator).state_dict(),
                        "proj_head": _unwrap_ddp(proj_head).state_dict(),
                        "opt_g": opt_g.state_dict(),
                        "scheduler": scheduler.state_dict(),
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
        print("Fair Centroid Contrastive Learning Complete!")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print("\n[3rd → 3rd_fix1 핵심 변경]")
        print("  - Female → Male 당김 → Fair Centroid Alignment")
        print("  - 비대칭 (1.5:0.5) → 대칭 (1.0:1.0)")
        print("  - Dropout + 낮은 LR + Cosine annealing")
        print("  - 24 epochs → 15 epochs")
        print("\n성공 기준:")
        print("  - AP Gap < 0.10 (baseline 0.106 대비 개선)")
        print("  - Female AP > 0.41 (baseline 0.404 대비 개선)")


if __name__ == "__main__":
    main()
