"""
FAAP Training - Adaptive Score-based Contrastive Learning (v2)
=============================================================================

핵심 개선: Adaptive Percentile Split
=============================================================================

[v1 문제점]
- 고정 threshold (0.5) 사용
- 실제 score가 대부분 0.9 이상 → Anchor 없음 → Loss 작동 안 함

[v2 해결책]
- 배치 내 상대적 ranking 사용
- 상위 K% = Positive, 하위 K% = Anchor
- 항상 균형 있는 split 보장

[추가 개선]
- Contrastive loss warmup (처음엔 약하게, 점차 강하게)
- Temperature 0.1 (더 안정적)
- Margin 영역 (중간 샘플) 제외 옵션

=============================================================================
"""

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

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
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .datasets import build_faap_dataloader, inspect_faap_dataset
from .models import FrozenDETR, PerturbationGenerator, clamp_normalized
from .path_utils import DETR_REPO, default_detr_checkpoint, ensure_detr_repo_on_path
import util.misc as utils
from util.misc import NestedTensor


# =============================================================================
# Projection Head
# =============================================================================

class ProjectionHead(nn.Module):
    """
    Feature를 contrastive space로 매핑하는 MLP.
    SimCLR 논문에서 2-layer MLP가 효과적임을 입증.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        proj = self.net(x)
        return F.normalize(proj, dim=-1, p=2)


# =============================================================================
# Adaptive Score-based Contrastive Loss (핵심 개선)
# =============================================================================

class AdaptiveScoreContrastiveLoss(nn.Module):
    """
    Adaptive Percentile-based Contrastive Loss.

    고정 threshold 대신 배치 내 상대적 ranking 사용:
    - 상위 top_k_percent = Positive (고성능, 목표)
    - 하위 bottom_k_percent = Anchor (저성능, 이동 대상)
    - 중간 = 무시 (margin)

    이점:
    - 데이터 분포에 robust
    - 항상 균형 있는 Positive/Anchor 보장
    - Hard example mining 효과
    """

    def __init__(
        self,
        temperature: float = 0.1,
        top_k_percent: float = 0.4,
        bottom_k_percent: float = 0.4,
        min_samples: int = 2,
    ):
        super().__init__()
        self.temperature = temperature
        self.top_k_percent = top_k_percent
        self.bottom_k_percent = bottom_k_percent
        self.min_samples = min_samples

    def forward(
        self,
        projections: torch.Tensor,
        scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            projections: (N, D) L2-normalized projections
            scores: (N,) detection scores
        Returns:
            loss (scalar), info (dict)
        """
        n = projections.size(0)
        info = {"n_positive": 0, "n_anchor": 0, "score_gap": 0.0}

        if n < self.min_samples * 2:
            return projections.new_tensor(0.0), info

        # =================================================================
        # Adaptive Split: 배치 내 상대적 ranking
        # =================================================================
        sorted_indices = scores.argsort(descending=True)

        n_top = max(self.min_samples, int(n * self.top_k_percent))
        n_bottom = max(self.min_samples, int(n * self.bottom_k_percent))

        # 겹치지 않도록 조정
        if n_top + n_bottom > n:
            n_top = n // 2
            n_bottom = n - n_top

        top_indices = sorted_indices[:n_top]
        bottom_indices = sorted_indices[-n_bottom:]

        positive_feats = projections[top_indices]   # 고성능 = Positive
        anchor_feats = projections[bottom_indices]  # 저성능 = Anchor

        info["n_positive"] = n_top
        info["n_anchor"] = n_bottom
        info["score_gap"] = (scores[top_indices].mean() - scores[bottom_indices].mean()).item()

        # =================================================================
        # InfoNCE: Anchor → Positive 방향으로 당김
        # =================================================================

        # Anchor와 Positive 간 similarity
        sim_anchor_to_pos = torch.mm(anchor_feats, positive_feats.t()) / self.temperature
        # (n_anchor, n_positive)

        # Anchor와 전체 샘플 간 similarity (분모용)
        sim_anchor_to_all = torch.mm(anchor_feats, projections.t()) / self.temperature
        # (n_anchor, N)

        # 자기 자신 제외
        for i, idx in enumerate(bottom_indices):
            sim_anchor_to_all[i, idx] = float('-inf')

        # InfoNCE Loss
        # 분자: Anchor와 모든 Positive의 similarity
        numerator = torch.logsumexp(sim_anchor_to_pos, dim=1)

        # 분모: Anchor와 모든 샘플의 similarity
        denominator = torch.logsumexp(sim_anchor_to_all, dim=1)

        # Loss: -log(positive / all)
        loss = -(numerator - denominator).mean()

        return loss, info


# =============================================================================
# Bidirectional Score Contrastive Loss (양방향)
# =============================================================================

class BidirectionalScoreContrastiveLoss(nn.Module):
    """
    양방향 Contrastive Loss:
    1. Anchor(저성능) → Positive(고성능) 방향
    2. 고성능 내에서 더 tight한 cluster 형성

    비대칭 가중치로 저성능 개선에 집중.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        top_k_percent: float = 0.4,
        bottom_k_percent: float = 0.4,
        anchor_weight: float = 1.0,
        positive_weight: float = 0.3,
        min_samples: int = 2,
    ):
        super().__init__()
        self.temperature = temperature
        self.top_k_percent = top_k_percent
        self.bottom_k_percent = bottom_k_percent
        self.anchor_weight = anchor_weight
        self.positive_weight = positive_weight
        self.min_samples = min_samples

    def forward(
        self,
        projections: torch.Tensor,
        scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        n = projections.size(0)
        info = {"n_positive": 0, "n_anchor": 0, "score_gap": 0.0, "loss_a2p": 0.0, "loss_p2p": 0.0}

        if n < self.min_samples * 2:
            return projections.new_tensor(0.0), info

        # Adaptive Split
        sorted_indices = scores.argsort(descending=True)

        n_top = max(self.min_samples, int(n * self.top_k_percent))
        n_bottom = max(self.min_samples, int(n * self.bottom_k_percent))

        if n_top + n_bottom > n:
            n_top = n // 2
            n_bottom = n - n_top

        top_indices = sorted_indices[:n_top]
        bottom_indices = sorted_indices[-n_bottom:]

        positive_feats = projections[top_indices]
        anchor_feats = projections[bottom_indices]

        info["n_positive"] = n_top
        info["n_anchor"] = n_bottom
        info["score_gap"] = (scores[top_indices].mean() - scores[bottom_indices].mean()).item()

        # =================================================================
        # Loss 1: Anchor → Positive (핵심)
        # =================================================================
        sim_a2p = torch.mm(anchor_feats, positive_feats.t()) / self.temperature
        sim_a2all = torch.mm(anchor_feats, projections.t()) / self.temperature

        for i, idx in enumerate(bottom_indices):
            sim_a2all[i, idx] = float('-inf')

        num_a2p = torch.logsumexp(sim_a2p, dim=1)
        denom_a2p = torch.logsumexp(sim_a2all, dim=1)
        loss_a2p = -(num_a2p - denom_a2p).mean()

        # =================================================================
        # Loss 2: Positive 내 clustering (보조)
        # =================================================================
        if n_top >= 2:
            sim_p2p = torch.mm(positive_feats, positive_feats.t()) / self.temperature
            # 대각선(자기 자신) 제외
            mask_p = torch.eye(n_top, device=projections.device, dtype=torch.bool)
            sim_p2p_masked = sim_p2p.masked_fill(mask_p, float('-inf'))

            sim_p2all = torch.mm(positive_feats, projections.t()) / self.temperature
            for i, idx in enumerate(top_indices):
                sim_p2all[i, idx] = float('-inf')

            num_p2p = torch.logsumexp(sim_p2p_masked, dim=1)
            denom_p2p = torch.logsumexp(sim_p2all, dim=1)
            loss_p2p = -(num_p2p - denom_p2p).mean()
        else:
            loss_p2p = projections.new_tensor(0.0)

        info["loss_a2p"] = loss_a2p.item()
        info["loss_p2p"] = loss_p2p.item()

        # 비대칭 가중합
        total_loss = self.anchor_weight * loss_a2p + self.positive_weight * loss_p2p

        return total_loss, info


# =============================================================================
# Wasserstein Loss (기존 유지)
# =============================================================================

def _resize_sorted(scores: torch.Tensor, target_len: int) -> torch.Tensor:
    if target_len <= 0:
        return scores.new_zeros(0)
    if scores.numel() == 0:
        return scores.new_zeros(target_len)
    if scores.numel() == target_len:
        return scores
    idx = torch.linspace(0, scores.numel() - 1, target_len, device=scores.device)
    idx_low = idx.floor().long()
    idx_high = idx.ceil().long()
    weight = idx - idx_low
    return scores[idx_low] * (1 - weight) + scores[idx_high] * weight


def _wasserstein_1d_asymmetric(
    female_scores: torch.Tensor,
    male_scores: torch.Tensor,
) -> torch.Tensor:
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0)
    sorted_f = female_scores.sort().values
    sorted_m = male_scores.detach().sort().values
    k = max(sorted_f.numel(), sorted_m.numel())
    sorted_f = _resize_sorted(sorted_f, k)
    sorted_m = _resize_sorted(sorted_m, k)
    return F.relu(sorted_m - sorted_f).mean()


def _get_image_level_scores(
    detr: FrozenDETR,
    outputs: dict,
    targets: Sequence[dict],
) -> torch.Tensor:
    """이미지 단위 detection score (매칭된 detection의 평균)"""
    if len(targets) == 0:
        return outputs["pred_logits"].new_zeros(0)

    indices = detr.criterion.matcher(outputs, targets)
    probs = outputs["pred_logits"].softmax(dim=-1)

    image_scores = []
    for b, (src_idx, tgt_idx) in enumerate(indices):
        if len(src_idx) == 0:
            image_scores.append(torch.tensor(0.0, device=probs.device))
        else:
            tgt_labels = targets[b]["labels"][tgt_idx]
            scores = probs[b, src_idx, tgt_labels]
            image_scores.append(scores.mean())

    return torch.stack(image_scores)


# =============================================================================
# Utility Functions
# =============================================================================

def _default_output_dir(script_path: Path) -> str:
    stem = script_path.stem
    stem_lower = stem.lower()
    suffix = stem
    for prefix in ("train_faap_", "train_"):
        if stem_lower.startswith(prefix):
            suffix = stem[len(prefix):]
            break
    suffix = suffix.lower()
    return str(Path("faap_outputs") / f"faap_outputs_{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "FAAP Adaptive Score-based Contrastive Learning (v2)",
        add_help=True,
    )

    # Paths
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()))
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(Path(__file__)))

    # Training basics
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    # Perturbation settings
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--epsilon_final", type=float, default=0.10)
    parser.add_argument("--epsilon_min", type=float, default=0.09)
    parser.add_argument("--epsilon_warmup_epochs", type=int, default=8)
    parser.add_argument("--epsilon_hold_epochs", type=int, default=6)
    parser.add_argument("--epsilon_cooldown_epochs", type=int, default=16)

    # =================================================================
    # Loss weights
    # =================================================================
    parser.add_argument("--lambda_contrastive", type=float, default=1.0)
    parser.add_argument("--lambda_wass", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--beta_final", type=float, default=0.6)

    # =================================================================
    # Contrastive settings (v2 개선)
    # =================================================================
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="InfoNCE temperature (0.1 for stability)")
    parser.add_argument("--top_k_percent", type=float, default=0.4,
                        help="상위 K%를 Positive로 (배치 내 상대적)")
    parser.add_argument("--bottom_k_percent", type=float, default=0.4,
                        help="하위 K%를 Anchor로 (배치 내 상대적)")
    parser.add_argument("--contrastive_warmup_epochs", type=int, default=3,
                        help="Contrastive loss warmup epochs")
    parser.add_argument("--bidirectional", action="store_true",
                        help="양방향 contrastive loss 사용")

    # Projection head
    parser.add_argument("--proj_hidden_dim", type=int, default=512)
    parser.add_argument("--proj_output_dim", type=int, default=128)

    # Other
    parser.add_argument("--max_norm", type=float, default=0.1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--max_train_per_gender", type=int, default=0)
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


def _set_generator_epsilon(generator: nn.Module, epsilon: float) -> None:
    _unwrap_ddp(generator).epsilon = epsilon


def _scheduled_epsilon(
    epoch: int,
    warmup_epochs: int,
    hold_epochs: int,
    cooldown_epochs: int,
    eps_start: float,
    eps_peak: float,
    eps_min: float,
) -> float:
    warmup_end = max(0, warmup_epochs - 1) if warmup_epochs > 1 else 0
    if epoch <= warmup_end:
        progress = min(epoch / max(1, warmup_epochs - 1), 1.0)
        return eps_start + (eps_peak - eps_start) * progress
    hold_end = warmup_end + max(0, hold_epochs)
    if epoch <= hold_end:
        return eps_peak
    if cooldown_epochs <= 0:
        return eps_peak
    progress = (epoch - hold_end) / max(1, cooldown_epochs)
    if progress >= 1.0:
        return eps_min
    return eps_peak + (eps_min - eps_peak) * progress


def _scheduled_beta(epoch: int, total_epochs: int, beta_start: float, beta_final: float) -> float:
    if total_epochs <= 1:
        return beta_start
    progress = min(epoch / max(1, total_epochs - 1), 1.0)
    return beta_start + (beta_final - beta_start) * progress


def _contrastive_warmup_weight(epoch: int, warmup_epochs: int) -> float:
    """Contrastive loss warmup: 0 → 1"""
    if warmup_epochs <= 0:
        return 1.0
    if epoch >= warmup_epochs:
        return 1.0
    return epoch / warmup_epochs


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
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA")
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
        print("Adaptive Score-based Contrastive Learning (v2)")
        print("=" * 70)
        print(f"핵심 개선: 고정 threshold → 배치 내 상대적 ranking")
        print(f"  - Positive: 상위 {args.top_k_percent*100:.0f}% (고성능)")
        print(f"  - Anchor: 하위 {args.bottom_k_percent*100:.0f}% (저성능)")
        print(f"  - 중간 {(1-args.top_k_percent-args.bottom_k_percent)*100:.0f}%: margin (무시)")
        print(f"Temperature: {args.temperature}")
        print(f"Contrastive warmup: {args.contrastive_warmup_epochs} epochs")
        print(f"Bidirectional: {args.bidirectional}")
        print(f"Loss weights: Contrastive={args.lambda_contrastive}, Wass={args.lambda_wass}")
        print("=" * 70)

    # ==========================================================================
    # Model Initialization
    # ==========================================================================

    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)

    proj_head = ProjectionHead(
        input_dim=detr.hidden_dim,
        hidden_dim=args.proj_hidden_dim,
        output_dim=args.proj_output_dim,
    ).to(device)

    # Contrastive Loss 선택
    if args.bidirectional:
        contrastive_loss_fn = BidirectionalScoreContrastiveLoss(
            temperature=args.temperature,
            top_k_percent=args.top_k_percent,
            bottom_k_percent=args.bottom_k_percent,
        ).to(device)
    else:
        contrastive_loss_fn = AdaptiveScoreContrastiveLoss(
            temperature=args.temperature,
            top_k_percent=args.top_k_percent,
            bottom_k_percent=args.bottom_k_percent,
        ).to(device)

    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)
        proj_head = DDP(proj_head, device_ids=[args.gpu] if args.gpu is not None else None)

    # Optimizer
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
    max_per_gender = args.max_train_per_gender if args.max_train_per_gender > 0 else None
    train_loader, _ = build_faap_dataloader(
        Path(args.dataset_root),
        "train",
        args.batch_size,
        include_gender=True,
        balance_genders=True,
        max_per_gender=max_per_gender,
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

        # Schedules
        current_eps = _scheduled_epsilon(
            epoch,
            args.epsilon_warmup_epochs,
            args.epsilon_hold_epochs,
            args.epsilon_cooldown_epochs,
            args.epsilon,
            args.epsilon_final,
            args.epsilon_min,
        )
        current_beta = _scheduled_beta(epoch, args.epochs, args.beta, args.beta_final)
        contrastive_weight = _contrastive_warmup_weight(epoch, args.contrastive_warmup_epochs)
        _set_generator_epsilon(generator, current_eps)

        for samples, targets, genders in metrics_logger.log_every(
            train_loader, args.log_every, f"Epoch {epoch}"
        ):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            genders = [g.lower() for g in genders]

            female_idx = [i for i, g in enumerate(genders) if g == "female"]
            male_idx = [i for i, g in enumerate(genders) if g == "male"]

            # =================================================================
            # Forward Pass
            # =================================================================
            opt_g.zero_grad()

            perturbed = _apply_generator(generator, samples)
            outputs, features = detr.forward_with_features(perturbed)

            # =================================================================
            # Image-level Detection Scores
            # =================================================================
            image_scores = _get_image_level_scores(detr, outputs, targets)

            # =================================================================
            # Adaptive Score-based Contrastive Loss (핵심)
            # =================================================================
            projections = proj_head(features)
            loss_contrastive, contrastive_info = contrastive_loss_fn(projections, image_scores)

            # =================================================================
            # Wasserstein Loss (성별 기반, 보조)
            # =================================================================
            loss_wasserstein = torch.tensor(0.0, device=device)
            if len(female_idx) > 0 and len(male_idx) > 0:
                female_scores = image_scores[female_idx]
                male_scores = image_scores[male_idx]
                loss_wasserstein = _wasserstein_1d_asymmetric(female_scores, male_scores)

            # =================================================================
            # Detection Loss
            # =================================================================
            loss_det, _ = detr.detection_loss(outputs, targets)

            # =================================================================
            # Total Loss (with contrastive warmup)
            # =================================================================
            effective_lambda_contrastive = args.lambda_contrastive * contrastive_weight

            total_g = (
                effective_lambda_contrastive * loss_contrastive
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

                score_mean = image_scores.mean()
                score_min = image_scores.min()
                score_max = image_scores.max()

                if len(female_idx) > 0:
                    score_f = image_scores[female_idx].mean()
                else:
                    score_f = torch.tensor(0.0, device=device)
                if len(male_idx) > 0:
                    score_m = image_scores[male_idx].mean()
                else:
                    score_m = torch.tensor(0.0, device=device)

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
                eps=current_eps,
                beta=current_beta,
                c_weight=contrastive_weight,
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                n_pos=contrastive_info["n_positive"],
                n_anc=contrastive_info["n_anchor"],
                score_gap=contrastive_info["score_gap"],
                score_mean=score_mean.item(),
                score_min=score_min.item(),
                score_max=score_max.item(),
                score_f=score_f.item(),
                score_m=score_m.item(),
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
                "epsilon": current_eps,
                "beta": current_beta,
                "contrastive_weight": contrastive_weight,
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "n_positive": metrics_logger.meters["n_pos"].global_avg,
                "n_anchor": metrics_logger.meters["n_anc"].global_avg,
                "score_gap_within_batch": metrics_logger.meters["score_gap"].global_avg,
                "score_mean": metrics_logger.meters["score_mean"].global_avg,
                "score_min": metrics_logger.meters["score_min"].global_avg,
                "score_max": metrics_logger.meters["score_max"].global_avg,
                "score_f": metrics_logger.meters["score_f"].global_avg,
                "score_m": metrics_logger.meters["score_m"].global_avg,
            }

            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            gender_gap = log_entry["score_m"] - log_entry["score_f"]
            print(f"\n[Epoch {epoch}] Summary:")
            print(f"  Contrastive Loss: {log_entry['loss_contrastive']:.4f} (weight: {contrastive_weight:.2f})")
            print(f"  Wasserstein Loss: {log_entry['loss_wasserstein']:.4f}")
            print(f"  Detection Loss: {log_entry['loss_det']:.4f}")
            print(f"  Total: {log_entry['total_g']:.4f}")
            print(f"  Positive/Anchor: {log_entry['n_positive']:.1f} / {log_entry['n_anchor']:.1f}")
            print(f"  Score within batch gap: {log_entry['score_gap_within_batch']:.4f}")
            print(f"  Score (F/M): {log_entry['score_f']:.4f} / {log_entry['score_m']:.4f}")
            print(f"  Gender Score Gap (M-F): {gender_gap:.4f}")
            print(f"  Epsilon: {current_eps:.4f}, Beta: {current_beta:.4f}")

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
        print("Adaptive Score-based Contrastive Learning Complete!")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print("\nv2 핵심 개선:")
        print(f"  - 고정 threshold → 배치 내 상대적 ranking")
        print(f"  - Positive: 상위 {args.top_k_percent*100:.0f}%")
        print(f"  - Anchor: 하위 {args.bottom_k_percent*100:.0f}%")
        print(f"  - Contrastive warmup: {args.contrastive_warmup_epochs} epochs")
        print("\n성공 기준:")
        print("  - AP Gap < 0.09 (15% 개선)")
        print("  - Female AP > 0.41")


if __name__ == "__main__":
    main()
