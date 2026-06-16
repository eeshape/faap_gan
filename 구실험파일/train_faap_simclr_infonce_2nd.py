"""
FAAP Training - Score-Based Contrastive Learning (2nd Version)

=============================================================================
핵심 아이디어: Detection Score 기반 고성능/저성능 이미지 대조학습
=============================================================================

[1st 버전의 한계]
- 성별 기반으로만 고성능/저성능 구분 (암묵적 가정)
- 실제 Detection Score를 contrastive learning에 미활용

[본 연구: Score-Based Contrastive Learning]
- 성능 = Detection Score (logits)
- 고성능 = Detection Score가 높은 이미지 (자연스럽게 남성 이미지 다수)
- 저성능 = Detection Score가 낮은 이미지 (자연스럽게 여성 이미지 다수)

핵심 수식:
L = -log(exp(sim(z_low, z_high)/τ) / [exp(sim(z_low, z_high)/τ) + Σ exp(sim(z_low, z_low')/τ)])

- Anchor: 저성능 이미지 (Detection Score 하위)
- Positive: 고성능 이미지 (Detection Score 상위)
- Negative: 다른 저성능 이미지

[W거리와의 차이점]
- W거리: 단순 분포 이동 (score 분포만 정렬)
- 대조학습: Feature 자체가 이동 (representation level에서 변화)

=============================================================================
"""

import argparse
import json
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
# SimCLR-Style Data Augmentation (Detection 친화적)
# =============================================================================

class SimCLRAugmentation(nn.Module):
    """
    SimCLR 스타일 augmentation (train 파일 내 구현).

    Detection 친화적으로 설계:
    - ColorJitter: 조명/색상 변화에 대한 불변성 학습
    - GaussianBlur, Grayscale: strong 모드에서만 사용 (detection 성능 저하 위험)

    Args:
        strength: "none", "weak", "medium", "strong" 중 선택
    """

    def __init__(self, strength: str = "medium"):
        super().__init__()
        self.strength = strength

        if strength == "none":
            self.transform = None
        elif strength == "weak":
            # 약한 ColorJitter (detection 안전)
            self.transform = T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            )
        elif strength == "medium":
            # Detection 친화적 ColorJitter (추천)
            self.transform = T.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            )
        elif strength == "strong":
            # 표준 SimCLR (detection 성능 저하 가능)
            self.transform = T.Compose([
                T.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                ),
                T.RandomGrayscale(p=0.2),
            ])
        else:
            raise ValueError(f"Unknown augmentation strength: {strength}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) normalized tensor (ImageNet mean/std)
        Returns:
            (B, C, H, W) augmented and re-normalized tensor
        """
        if self.transform is None:
            return x

        # ImageNet normalization 상수
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

        # Denormalize → [0, 1] 범위
        x_denorm = x * std + mean
        x_denorm = torch.clamp(x_denorm, 0, 1)

        # ColorJitter 적용 (배치 단위로 각 이미지에 독립 적용)
        augmented = torch.stack([self.transform(img) for img in x_denorm])

        # 다시 normalize
        return (augmented - mean) / std


# =============================================================================
# SimCLR-Style Projection Head
# =============================================================================

class SimCLRProjectionHead(nn.Module):
    """
    SimCLR 스타일 2-layer MLP projection head.

    Image-level pooling 후 projection하여 contrastive learning에 사용.
    L2 normalize된 output을 반환하여 cosine similarity 계산에 최적화.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_queries, feature_dim) DETR decoder features
        Returns:
            (batch, output_dim) L2-normalized projections
        """
        # Image-level pooling (average over queries)
        pooled = x.mean(dim=1)  # (batch, feature_dim)
        proj = self.net(pooled)
        return F.normalize(proj, dim=-1, p=2)


# =============================================================================
# Score-Based Contrastive Loss (핵심: Detection Score 기반)
# =============================================================================

class ScoreBasedContrastiveLoss(nn.Module):
    """
    Detection Score 기반 대조학습

    핵심 아이디어:
    - Anchor: 저성능 이미지 (Detection Score 하위)
    - Positive: 고성능 이미지 (Detection Score 상위)
    - Negative: 다른 저성능 이미지

    수식:
    L = -log(exp(sim(z_low, z_high)/τ) / [exp(sim(z_low, z_high)/τ) + Σ exp(sim(z_low, z_low')/τ)])

    W거리와 차이:
    - W거리: score 분포만 정렬
    - 본 방법: Feature 자체가 이동 (representation level)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        score_margin: float = 0.0,  # 고성능/저성능 구분 마진 (0이면 median 기준)
    ):
        super().__init__()
        self.temperature = temperature
        self.score_margin = score_margin

    def forward(
        self,
        projections: torch.Tensor,
        scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            projections: (N, D) L2-normalized projections
            scores: (N,) Detection scores for each image

        Returns:
            loss: scalar
            info: dict with debugging info
        """
        N = projections.size(0)
        if N < 4:
            # 최소 4개 샘플 필요 (low 2개 + high 2개)
            return projections.new_tensor(0.0), {"n_low": 0, "n_high": 0}

        # =================================================================
        # 1. Score 기준으로 고성능/저성능 분리
        # =================================================================
        # scores는 분리 기준으로만 사용 (gradient 불필요)
        scores = scores.detach()

        # Median 기준으로 분리 (margin 적용 가능)
        median_score = scores.median()

        # 저성능: median - margin 이하
        # 고성능: median + margin 이상
        low_mask = scores <= (median_score - self.score_margin)
        high_mask = scores >= (median_score + self.score_margin)

        # margin으로 인해 빈 그룹이 생기면 단순 median 기준으로 fallback
        if low_mask.sum() == 0 or high_mask.sum() == 0:
            low_mask = scores <= median_score
            high_mask = scores > median_score

        n_low = low_mask.sum().item()
        n_high = high_mask.sum().item()

        # 각 그룹에 최소 2개 필요 (자기 자신 제외한 negative 필요)
        if n_low < 2 or n_high < 2:
            return projections.new_tensor(0.0), {"n_low": n_low, "n_high": n_high}

        proj_low = projections[low_mask]   # (N_low, D) - Anchor
        proj_high = projections[high_mask]  # (N_high, D) - Positive

        # =================================================================
        # 2. 양방향 InfoNCE: 저성능↔고성능 상호 학습
        # =================================================================
        # (a) 저성능(Anchor) → 고성능(Positive) 방향
        sim_low2high = torch.mm(proj_low, proj_high.t()) / self.temperature  # (N_low, N_high)
        sim_low2low = torch.mm(proj_low, proj_low.t()) / self.temperature    # (N_low, N_low)

        # 자기 자신 마스킹
        mask_low = torch.eye(n_low, device=proj_low.device, dtype=torch.bool)
        sim_low2low_masked = sim_low2low.masked_fill(mask_low, float('-inf'))

        # Low → High InfoNCE
        all_sims_low = torch.cat([sim_low2high, sim_low2low_masked], dim=1)
        numerator_low = torch.logsumexp(sim_low2high, dim=1)
        denominator_low = torch.logsumexp(all_sims_low, dim=1)
        loss_low2high = -(numerator_low - denominator_low).mean()

        # (b) 고성능(Anchor) → 저성능(Positive) 방향 (역방향, 균형 학습)
        sim_high2low = sim_low2high.t()  # (N_high, N_low)
        sim_high2high = torch.mm(proj_high, proj_high.t()) / self.temperature  # (N_high, N_high)

        mask_high = torch.eye(n_high, device=proj_high.device, dtype=torch.bool)
        sim_high2high_masked = sim_high2high.masked_fill(mask_high, float('-inf'))

        # High → Low InfoNCE (약하게)
        all_sims_high = torch.cat([sim_high2low, sim_high2high_masked], dim=1)
        numerator_high = torch.logsumexp(sim_high2low, dim=1)
        denominator_high = torch.logsumexp(all_sims_high, dim=1)
        loss_high2low = -(numerator_high - denominator_high).mean()

        # 비대칭 결합: 저성능→고성능 방향 강조 (1.5:0.5)
        loss = 1.5 * loss_low2high + 0.5 * loss_high2low

        info = {
            "n_low": n_low,
            "n_high": n_high,
            "score_low_mean": scores[low_mask].mean().item(),
            "score_high_mean": scores[high_mask].mean().item(),
            "score_gap": (scores[high_mask].mean() - scores[low_mask].mean()).item(),
        }

        return loss, info


# =============================================================================
# Score-Level Wasserstein Loss (유지)
# =============================================================================

def _resize_sorted(scores: torch.Tensor, target_len: int) -> torch.Tensor:
    """분위수 정렬을 위한 크기 조정"""
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
    """
    단방향 Wasserstein: 여성 score → 남성 score.

    여성 score가 남성보다 낮을 때만 패널티를 부여하여
    여성 detection 품질만 향상시킴.
    """
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0)

    sorted_f = female_scores.sort().values
    sorted_m = male_scores.detach().sort().values  # 남성은 detach (타겟 역할)

    k = max(sorted_f.numel(), sorted_m.numel())
    sorted_f = _resize_sorted(sorted_f, k)
    sorted_m = _resize_sorted(sorted_m, k)

    # 단방향: 여성 < 남성일 때만 손실 발생
    return F.relu(sorted_m - sorted_f).mean()


def _matched_detection_scores(
    detr: "FrozenDETR",
    outputs: dict,
    targets: Sequence[dict],
) -> torch.Tensor:
    """Hungarian matching을 통해 GT와 매칭된 query의 detection score를 추출"""
    if len(targets) == 0:
        return outputs["pred_logits"].new_zeros(0)

    indices = detr.criterion.matcher(outputs, targets)
    probs = outputs["pred_logits"].softmax(dim=-1)

    matched_scores = []
    for b, (src_idx, tgt_idx) in enumerate(indices):
        if len(src_idx) == 0:
            continue
        tgt_labels = targets[b]["labels"][tgt_idx]
        matched_scores.append(probs[b, src_idx, tgt_labels])

    if matched_scores:
        return torch.cat(matched_scores, dim=0)
    return outputs["pred_logits"].new_zeros(0)


def _image_level_detection_score(outputs: dict, top_k: int = 10) -> torch.Tensor:
    """
    이미지 단위 Detection Score 계산 (DETR logits 직접 사용).

    GT matching 없이 DETR가 예측한 confidence만 사용.
    각 이미지에서 top-k query의 max class probability 평균.

    Args:
        outputs: DETR output dict (pred_logits 포함)
        top_k: 상위 k개 query만 사용 (default: 10)

    Returns:
        (B,) 각 이미지의 detection score
    """
    # pred_logits: (B, 100, num_classes+1)
    # softmax 후 no-object class 제외
    probs = outputs["pred_logits"].softmax(dim=-1)[..., :-1]  # (B, 100, num_classes)

    # 각 query의 max class probability
    max_probs = probs.max(dim=-1).values  # (B, 100)

    # 상위 top_k query의 평균 (더 의미있는 score)
    if top_k > 0 and top_k < max_probs.size(1):
        topk_probs = max_probs.topk(top_k, dim=1).values  # (B, top_k)
        return topk_probs.mean(dim=1)  # (B,)
    else:
        return max_probs.mean(dim=1)  # (B,)


# =============================================================================
# Utility Functions
# =============================================================================

def _default_output_dir(script_path: Path) -> str:
    stem = script_path.stem
    stem_lower = stem.lower()
    suffix = stem
    for prefix in ("train_faap_simclr_", "train_faap_", "train_"):
        if stem_lower.startswith(prefix):
            suffix = stem[len(prefix):]
            break
    suffix = suffix.lower()
    return str(Path("faap_outputs") / f"faap_outputs_{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "FAAP Score-Based Contrastive Learning (2nd Version)",
        add_help=True,
    )

    # Paths
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()))
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(Path(__file__)))

    # Training basics
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    # Perturbation settings
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--epsilon_final", type=float, default=0.10)
    parser.add_argument("--epsilon_min", type=float, default=0.09)
    parser.add_argument("--epsilon_warmup_epochs", type=int, default=8)
    parser.add_argument("--epsilon_hold_epochs", type=int, default=6)
    parser.add_argument("--epsilon_cooldown_epochs", type=int, default=10)

    # =================================================================
    # Loss weights
    # =================================================================
    parser.add_argument("--lambda_contrastive", type=float, default=1.0,
                        help="Score-based contrastive loss weight")
    parser.add_argument("--lambda_wass", type=float, default=0.2,
                        help="Score-level Wasserstein loss weight")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Detection loss weight start")
    parser.add_argument("--beta_final", type=float, default=0.6,
                        help="Detection loss weight final")

    # =================================================================
    # Contrastive learning settings
    # =================================================================
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature for contrastive loss")
    parser.add_argument("--score_margin", type=float, default=0.0,
                        help="Margin for high/low score separation (0=median)")
    parser.add_argument("--score_top_k", type=int, default=10,
                        help="Top-k queries for image-level score (0=all)")

    # Projection head
    parser.add_argument("--proj_dim", type=int, default=128,
                        help="Projection output dimension")

    # =================================================================
    # SimCLR-Style Data Augmentation
    # =================================================================
    parser.add_argument("--aug_strength", type=str, default="medium",
                        choices=["none", "weak", "medium", "strong"],
                        help="SimCLR augmentation strength")

    # Other
    parser.add_argument("--max_norm", type=float, default=0.1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--max_train_per_gender", type=int, default=0)
    parser.add_argument("--obj_conf_thresh", type=float, default=0.5)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--dist_url", default="env://")

    return parser.parse_args()


def _split_nested(samples: NestedTensor, targets: Sequence[dict], keep: List[int]):
    if len(keep) == 0:
        return None, []
    tensor = samples.tensors[keep]
    mask = samples.mask[keep] if samples.mask is not None else None
    return NestedTensor(tensor, mask), [targets[i] for i in keep]


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
        print("Score-Based Contrastive Learning (2nd Version)")
        print("=" * 70)
        print("[핵심 변경] Detection Score 기반 고성능/저성능 구분")
        print("  - Anchor: 저성능 이미지 (score 하위)")
        print("  - Positive: 고성능 이미지 (score 상위)")
        print("  - Feature 자체가 고성능 방향으로 이동")
        print("-" * 70)
        print(f"Temperature: {args.temperature}")
        print(f"Score margin: {args.score_margin}")
        print(f"Loss weights: Contrastive={args.lambda_contrastive}, Wass={args.lambda_wass}")
        print(f"Beta: {args.beta} → {args.beta_final}")
        print(f"SimCLR Augmentation: {args.aug_strength}")
        print("=" * 70)

    # ==========================================================================
    # Model Initialization
    # ==========================================================================

    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)

    # SimCLR Projection Head (image-level)
    proj_head = SimCLRProjectionHead(
        input_dim=detr.hidden_dim,
        hidden_dim=detr.hidden_dim,
        output_dim=args.proj_dim,
    ).to(device)

    # Score-Based Contrastive Loss
    contrastive_loss_fn = ScoreBasedContrastiveLoss(
        temperature=args.temperature,
        score_margin=args.score_margin,
    ).to(device)

    # SimCLR-Style Data Augmentation
    simclr_aug = SimCLRAugmentation(strength=args.aug_strength).to(device)

    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)
        proj_head = DDP(proj_head, device_ids=[args.gpu] if args.gpu is not None else None)

    # Optimizer (Generator + Projection Head)
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
        _set_generator_epsilon(generator, current_eps)

        for samples, targets, genders in metrics_logger.log_every(
            train_loader, args.log_every, f"Epoch {epoch}"
        ):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            genders = [g.lower() for g in genders]

            B = samples.tensors.size(0)
            if B < 2:
                continue

            # Gender indices (for Wasserstein loss)
            female_idx = [i for i, g in enumerate(genders) if g == "female"]
            male_idx = [i for i, g in enumerate(genders) if g == "male"]

            # =================================================================
            # Forward Pass
            # =================================================================
            opt_g.zero_grad()

            # Apply perturbation (전체 배치)
            perturbed = _apply_generator(generator, samples)

            # Apply SimCLR augmentation
            if generator.training and args.aug_strength != "none":
                perturbed = NestedTensor(
                    simclr_aug(perturbed.tensors),
                    perturbed.mask,
                )

            # DETR forward (with features)
            outputs, features = detr.forward_with_features(perturbed)

            # =================================================================
            # 1. Score-Based Contrastive Loss (핵심)
            # =================================================================
            # 이미지 단위 detection score 계산 (DETR logits 직접 사용)
            image_scores = _image_level_detection_score(outputs, top_k=args.score_top_k)

            # Projection
            projections = proj_head(features)  # (B, proj_dim)

            # Contrastive loss: 저성능 → 고성능 방향으로 feature 이동
            loss_contrastive, contrastive_info = contrastive_loss_fn(projections, image_scores)

            # =================================================================
            # 2. Score-Level Wasserstein (성별 기반, 유지)
            # =================================================================
            loss_wasserstein = torch.tensor(0.0, device=device)
            if len(female_idx) > 0 and len(male_idx) > 0:
                # tensor만 인덱싱 (aux_outputs 등 제외)
                female_outputs = {
                    k: v[female_idx] for k, v in outputs.items()
                    if isinstance(v, torch.Tensor)
                }
                male_outputs = {
                    k: v[male_idx] for k, v in outputs.items()
                    if isinstance(v, torch.Tensor)
                }
                female_scores = _matched_detection_scores(
                    detr,
                    female_outputs,
                    [targets[i] for i in female_idx],
                )
                male_scores = _matched_detection_scores(
                    detr,
                    male_outputs,
                    [targets[i] for i in male_idx],
                )
                loss_wasserstein = _wasserstein_1d_asymmetric(female_scores, male_scores)

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

                # Objectness scores (전체)
                probs = outputs["pred_logits"].softmax(dim=-1)[..., :-1]
                max_scores = probs.max(dim=-1).values
                obj_mean = max_scores.mean()
                obj_frac = (max_scores > args.obj_conf_thresh).float().mean()

                # 성별별 objectness (로깅용)
                obj_mean_f = torch.tensor(0.0, device=device)
                obj_mean_m = torch.tensor(0.0, device=device)
                if len(female_idx) > 0:
                    obj_mean_f = max_scores[female_idx].mean()
                if len(male_idx) > 0:
                    obj_mean_m = max_scores[male_idx].mean()

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
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                obj_score=obj_mean.item(),
                obj_frac=obj_frac.item(),
                obj_score_f=obj_mean_f.item(),
                obj_score_m=obj_mean_m.item(),
                n_low=contrastive_info.get("n_low", 0),
                n_high=contrastive_info.get("n_high", 0),
                score_gap=contrastive_info.get("score_gap", 0.0),
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
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "obj_score": metrics_logger.meters["obj_score"].global_avg,
                "obj_frac": metrics_logger.meters["obj_frac"].global_avg,
                "obj_score_f": metrics_logger.meters["obj_score_f"].global_avg,
                "obj_score_m": metrics_logger.meters["obj_score_m"].global_avg,
                "n_low_avg": metrics_logger.meters["n_low"].global_avg,
                "n_high_avg": metrics_logger.meters["n_high"].global_avg,
                "score_gap_avg": metrics_logger.meters["score_gap"].global_avg,
            }

            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Print summary
            obj_gap = log_entry["obj_score_m"] - log_entry["obj_score_f"]
            print(f"\n[Epoch {epoch}] Summary:")
            print(f"  Contrastive Loss: {log_entry['loss_contrastive']:.4f} (핵심)")
            print(f"  Wasserstein Loss: {log_entry['loss_wasserstein']:.4f}")
            print(f"  Detection Loss: {log_entry['loss_det']:.4f}")
            print(f"  Total: {log_entry['total_g']:.4f}")
            print(f"  Score-based split: low={log_entry['n_low_avg']:.1f}, high={log_entry['n_high_avg']:.1f}")
            print(f"  Score gap (high-low): {log_entry['score_gap_avg']:.4f}")
            print(f"  Obj Score (F/M): {log_entry['obj_score_f']:.4f} / {log_entry['obj_score_m']:.4f}")
            print(f"  Obj Score Gap (M-F): {obj_gap:.4f}")
            print(f"  Epsilon: {current_eps:.4f}, Beta: {current_beta:.4f}")

            # Save checkpoint
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
        print("Score-Based Contrastive Learning Complete!")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print("\n[1st 대비 핵심 변경]")
        print("  - 성별 기반 → Detection Score 기반 분리")
        print("  - Anchor: 저성능 이미지")
        print("  - Positive: 고성능 이미지")
        print("  - Feature 자체가 고성능 방향으로 이동")
        print("\n[W거리와 차이점]")
        print("  - W거리: score 분포만 정렬")
        print("  - 본 방법: Feature representation 자체 변화")
        print("\n성공 기준:")
        print("  - AP Gap < 0.09 (15% 개선)")
        print("  - Female AP > 0.41")


if __name__ == "__main__":
    main()
