"""
FAAP Training - MoCo-Inspired Gender-Aware Contrastive Learning (4th Version)

=============================================================================
3rd 버전 실패 분석:
=============================================================================
- Epoch 3에서 최고 성능 (AP Gap -0.0019), 이후 급격한 과적합
- 원인:
  1. Small batch (6) → 높은 variance
  2. Batch 내에서만 비교 → 불안정한 target
  3. 적은 negative samples → weak contrastive signal

=============================================================================
4th 버전 핵심 아이디어: MoCo-Inspired Contrastive Learning
=============================================================================

[MoCo 논문 핵심 3가지]
1. Dictionary as Queue: 큰 memory bank로 batch size 제한 극복
2. Momentum Update: stable target으로 급격한 변화 방지
3. Consistent Dictionary: 일관된 representation 유지

[FAAP 적용]
1. Memory Bank (Queue):
   - 과거 female feature 저장 (size=2048)
   - 풍부한 negative samples 제공
   - FIFO 방식 업데이트

2. Momentum Male Centroid:
   - Male feature의 EMA (Exponential Moving Average)
   - c = m·c + (1-m)·mean(current_male), m=0.99
   - 안정적인 positive target

3. MoCo-Style InfoNCE:
   - Query: perturbed female feature
   - Positive: momentum male centroid
   - Negative: memory bank의 과거 female features

[수식]
L = -log[exp(sim(q, k+)/τ) / (exp(sim(q, k+)/τ) + Σ exp(sim(q, k-)/τ))]

[기대 효과]
- 큰 negative pool → 강한 contrastive signal
- Stable target → 과적합 방지
- Consistent learning → 안정적인 수렴

=============================================================================
"""

import argparse
import json
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

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
# SimCLR-Style Data Augmentation (from 3rd)
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
# SimCLR-Style Projection Head
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
# MoCo-Style Memory Bank
# =============================================================================

class MoCoMemoryBank:
    """
    MoCo-style memory bank for storing past features.

    Key differences from original MoCo:
    - We don't have momentum encoder (DETR is frozen)
    - We use memory bank for negative samples only
    - Separate banks for male and female features
    """

    def __init__(self, feature_dim: int = 128, queue_size: int = 2048, device: str = "cuda"):
        self.feature_dim = feature_dim
        self.queue_size = queue_size
        self.device = device

        # Initialize memory banks
        self.female_queue = torch.randn(queue_size, feature_dim, device=device)
        self.female_queue = F.normalize(self.female_queue, dim=1)
        self.female_ptr = 0
        self.female_filled = 0  # Track how many entries are valid

        self.male_queue = torch.randn(queue_size, feature_dim, device=device)
        self.male_queue = F.normalize(self.male_queue, dim=1)
        self.male_ptr = 0
        self.male_filled = 0

        # Momentum male centroid (positive target)
        self.male_centroid = None

    @torch.no_grad()
    def update_female_queue(self, features: torch.Tensor):
        """FIFO update of female queue."""
        batch_size = features.size(0)
        features = F.normalize(features.detach(), dim=1)

        if batch_size >= self.queue_size:
            # If batch is larger than queue, just use last queue_size features
            self.female_queue = features[-self.queue_size:].clone()
            self.female_ptr = 0
            self.female_filled = self.queue_size
        else:
            # FIFO update
            end_ptr = self.female_ptr + batch_size
            if end_ptr <= self.queue_size:
                self.female_queue[self.female_ptr:end_ptr] = features
            else:
                # Wrap around
                first_part = self.queue_size - self.female_ptr
                self.female_queue[self.female_ptr:] = features[:first_part]
                self.female_queue[:batch_size - first_part] = features[first_part:]

            self.female_ptr = end_ptr % self.queue_size
            self.female_filled = min(self.female_filled + batch_size, self.queue_size)

    @torch.no_grad()
    def update_male_queue(self, features: torch.Tensor):
        """FIFO update of male queue."""
        batch_size = features.size(0)
        features = F.normalize(features.detach(), dim=1)

        if batch_size >= self.queue_size:
            self.male_queue = features[-self.queue_size:].clone()
            self.male_ptr = 0
            self.male_filled = self.queue_size
        else:
            end_ptr = self.male_ptr + batch_size
            if end_ptr <= self.queue_size:
                self.male_queue[self.male_ptr:end_ptr] = features
            else:
                first_part = self.queue_size - self.male_ptr
                self.male_queue[self.male_ptr:] = features[:first_part]
                self.male_queue[:batch_size - first_part] = features[first_part:]

            self.male_ptr = end_ptr % self.queue_size
            self.male_filled = min(self.male_filled + batch_size, self.queue_size)

    @torch.no_grad()
    def update_male_centroid(self, features: torch.Tensor, momentum: float = 0.99):
        """
        Update male centroid with momentum (EMA).
        This is the stable positive target.
        """
        features = F.normalize(features.detach(), dim=1)
        current_centroid = features.mean(dim=0)
        current_centroid = F.normalize(current_centroid, dim=0)

        if self.male_centroid is None:
            self.male_centroid = current_centroid
        else:
            self.male_centroid = momentum * self.male_centroid + (1 - momentum) * current_centroid
            self.male_centroid = F.normalize(self.male_centroid, dim=0)

    def get_female_negatives(self) -> torch.Tensor:
        """Get valid female features from queue for negative samples."""
        if self.female_filled == 0:
            return None
        return self.female_queue[:self.female_filled]

    def get_male_centroid(self) -> Optional[torch.Tensor]:
        """Get momentum-updated male centroid."""
        return self.male_centroid


# =============================================================================
# MoCo-Style Contrastive Loss (4th: 핵심 변경)
# =============================================================================

class MoCoContrastiveLoss(nn.Module):
    """
    MoCo-inspired contrastive loss for FAAP.

    Key differences from 3rd version:
    1. Positive: momentum-updated male centroid (stable target)
    2. Negative: memory bank의 과거 female features (large pool)
    3. Score-weighted: 저성능 female에 더 강한 gradient

    수식:
    L = -log[exp(sim(q, k+)/τ) / (exp(sim(q, k+)/τ) + Σ exp(sim(q, k-)/τ))]

    where:
      q  = perturbed female projection (query)
      k+ = momentum male centroid (positive key)
      k- = memory bank's past female projections (negative keys)
    """

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
        query_female: torch.Tensor,      # (N_f, D) current female projections
        male_centroid: torch.Tensor,     # (D,) momentum male centroid
        negative_female: torch.Tensor,   # (K, D) memory bank female features
        scores_f: torch.Tensor,          # (N_f,) female detection scores
        scores_m_mean: float,            # scalar: mean male score
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            query_female: (N_f, D) current batch's female projections
            male_centroid: (D,) momentum-updated male centroid
            negative_female: (K, D) past female features from memory bank
            scores_f: (N_f,) detection scores for current females
            scores_m_mean: mean detection score of males
        """
        n_f = query_female.size(0)
        n_neg = negative_female.size(0)

        if n_f < 1 or male_centroid is None or n_neg < 1:
            return query_female.new_tensor(0.0), {"n_f": n_f, "n_neg": 0}

        # =================================================================
        # 1. Positive Similarity: query vs male_centroid
        # =================================================================
        # (N_f, D) @ (D,) -> (N_f,)
        sim_pos = torch.mv(query_female, male_centroid) / self.temperature  # (N_f,)

        # =================================================================
        # 2. Negative Similarity: query vs memory bank
        # =================================================================
        # (N_f, D) @ (D, K) -> (N_f, K)
        sim_neg = torch.mm(query_female, negative_female.t()) / self.temperature  # (N_f, K)

        # =================================================================
        # 3. Score-based Weighting (from 3rd version)
        # =================================================================
        # 저성능 female에 더 강한 gradient
        score_diff = scores_m_mean - scores_f.detach()  # (N_f,)
        score_weight = 1.0 + self.score_weight_alpha * torch.sigmoid(score_diff * 5)  # [1, 2]

        # =================================================================
        # 4. InfoNCE Loss
        # =================================================================
        # logits: [positive, negatives]
        # (N_f, 1+K)
        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)

        # Labels: positive is at index 0
        labels = torch.zeros(n_f, dtype=torch.long, device=query_female.device)

        # Cross entropy with score weighting
        loss_per_sample = F.cross_entropy(logits, labels, reduction='none')  # (N_f,)
        loss = (loss_per_sample * score_weight).mean()

        # Info
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == 0).float().mean().item()

        info = {
            "n_f": n_f,
            "n_neg": n_neg,
            "sim_pos_mean": sim_pos.mean().item(),
            "sim_neg_mean": sim_neg.mean().item(),
            "score_weight_mean": score_weight.mean().item(),
            "accuracy": acc,
        }

        return loss, info


# =============================================================================
# Bidirectional MoCo Loss (Optional: M→F도 학습)
# =============================================================================

class BidirectionalMoCoLoss(nn.Module):
    """
    양방향 MoCo loss:
    1. F→M: Female이 Male centroid로 당겨지도록 (주요)
    2. M→F: Male이 Female centroid에서 밀려나도록 (보조, 약하게)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        score_weight_alpha: float = 1.0,
        f2m_weight: float = 1.5,  # F→M 가중치 (주요)
        m2f_weight: float = 0.5,  # M→F 가중치 (보조)
    ):
        super().__init__()
        self.temperature = temperature
        self.score_weight_alpha = score_weight_alpha
        self.f2m_weight = f2m_weight
        self.m2f_weight = m2f_weight

        self.moco_loss = MoCoContrastiveLoss(temperature, score_weight_alpha)

    def forward(
        self,
        proj_f: torch.Tensor,           # (N_f, D) current female projections
        proj_m: torch.Tensor,           # (N_m, D) current male projections
        male_centroid: torch.Tensor,    # (D,) momentum male centroid
        female_centroid: torch.Tensor,  # (D,) momentum female centroid (for M→F)
        neg_female: torch.Tensor,       # (K, D) memory bank female features
        neg_male: torch.Tensor,         # (K, D) memory bank male features
        scores_f: torch.Tensor,         # (N_f,) female detection scores
        scores_m: torch.Tensor,         # (N_m,) male detection scores
    ) -> Tuple[torch.Tensor, dict]:

        n_f = proj_f.size(0)
        n_m = proj_m.size(0)

        # =================================================================
        # 1. F→M Loss (주요): Female이 Male centroid로
        # =================================================================
        loss_f2m = torch.tensor(0.0, device=proj_f.device)
        info_f2m = {}

        if n_f >= 1 and male_centroid is not None and neg_female is not None and neg_female.size(0) > 0:
            loss_f2m, info_f2m = self.moco_loss(
                query_female=proj_f,
                male_centroid=male_centroid,
                negative_female=neg_female,
                scores_f=scores_f,
                scores_m_mean=scores_m.mean().item() if n_m > 0 else 0.5,
            )

        # =================================================================
        # 2. M→F Loss (보조, 약하게): Male이 다양성 유지
        # =================================================================
        # 이 부분은 optional - Male이 Female centroid에서 멀어지도록
        # (실험적으로 효과 확인 필요)
        loss_m2f = torch.tensor(0.0, device=proj_m.device)

        # 단순화: M→F는 3rd처럼 batch 내에서만
        if n_m >= 2 and n_f >= 1:
            sim_m2f = torch.mm(proj_m, proj_f.t()) / self.temperature  # (N_m, N_f)
            sim_m2m = torch.mm(proj_m, proj_m.t()) / self.temperature  # (N_m, N_m)

            # Self-mask
            mask_m = torch.eye(n_m, device=proj_m.device, dtype=torch.bool)
            sim_m2m_masked = sim_m2m.masked_fill(mask_m, float('-inf'))

            # M→F: Male이 Female에 가까워지면 diversity 유지에 도움
            all_sims = torch.cat([sim_m2f, sim_m2m_masked], dim=1)
            numerator = torch.logsumexp(sim_m2f, dim=1)
            denominator = torch.logsumexp(all_sims, dim=1)
            loss_m2f = -(numerator - denominator).mean()

        # =================================================================
        # 3. Total Loss
        # =================================================================
        total_loss = self.f2m_weight * loss_f2m + self.m2f_weight * loss_m2f

        info = {
            "loss_f2m": loss_f2m.item() if torch.is_tensor(loss_f2m) else 0.0,
            "loss_m2f": loss_m2f.item() if torch.is_tensor(loss_m2f) else 0.0,
            "n_f": n_f,
            "n_m": n_m,
            **{f"f2m_{k}": v for k, v in info_f2m.items()},
        }

        return total_loss, info


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
    """단방향 Wasserstein: 여성 score → 남성 score"""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FAAP MoCo-Inspired Contrastive (4th)")

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
    parser.add_argument("--seed", type=int, default=42)

    # Perturbation
    parser.add_argument("--epsilon", type=float, default=0.10)

    # Loss weights
    parser.add_argument("--lambda_contrastive", type=float, default=1.0)
    parser.add_argument("--lambda_wass", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--beta_final", type=float, default=0.6)

    # MoCo-specific settings
    parser.add_argument("--queue_size", type=int, default=2048,
                        help="Size of memory bank (MoCo queue)")
    parser.add_argument("--momentum", type=float, default=0.99,
                        help="Momentum for EMA update of male centroid")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--score_weight_alpha", type=float, default=1.0)
    parser.add_argument("--f2m_weight", type=float, default=1.5,
                        help="Weight for F→M contrastive loss")
    parser.add_argument("--m2f_weight", type=float, default=0.5,
                        help="Weight for M→F contrastive loss")

    # Contrastive settings (from 3rd)
    parser.add_argument("--score_top_k", type=int, default=10)
    parser.add_argument("--proj_dim", type=int, default=128)

    # Augmentation
    parser.add_argument("--aug_strength", type=str, default="medium",
                        choices=["none", "weak", "medium", "strong"])

    # Warmup (memory bank이 채워지기 전에는 약하게 학습)
    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="Epochs to fill memory bank before full contrastive")

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
        print("MoCo-Inspired Gender-Aware Contrastive Learning (4th Version)")
        print("=" * 70)
        print("[3rd 실패 원인] Small batch + 불안정한 target = 빠른 과적합")
        print("[4th 핵심 변경] MoCo의 3가지 핵심 적용")
        print("  1. Memory Bank: 큰 negative pool (size={})".format(args.queue_size))
        print("  2. Momentum Centroid: EMA로 안정적인 target (m={})".format(args.momentum))
        print("  3. Score-Weighted InfoNCE: 저성능 female에 강한 gradient")
        print("-" * 70)
        print(f"Temperature: {args.temperature}")
        print(f"Queue Size: {args.queue_size}")
        print(f"Momentum: {args.momentum}")
        print(f"F2M/M2F Weight: {args.f2m_weight}/{args.m2f_weight}")
        print(f"Warmup Epochs: {args.warmup_epochs}")
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

    # MoCo Memory Bank
    memory_bank = MoCoMemoryBank(
        feature_dim=args.proj_dim,
        queue_size=args.queue_size,
        device=device,
    )

    # Bidirectional MoCo Loss
    contrastive_loss_fn = BidirectionalMoCoLoss(
        temperature=args.temperature,
        score_weight_alpha=args.score_weight_alpha,
        f2m_weight=args.f2m_weight,
        m2f_weight=args.m2f_weight,
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
        # Note: Memory bank state is not saved/loaded (starts fresh)
        if utils.is_main_process():
            print(f"Resumed from {args.resume} at epoch {start_epoch}")
            print("Note: Memory bank starts fresh after resume")

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

        # Warmup: memory bank이 충분히 채워지기 전에는 contrastive 약하게
        warmup_scale = min(1.0, (epoch + 1) / max(1, args.warmup_epochs))

        for samples, targets, genders in metrics_logger.log_every(
            train_loader, args.log_every, f"Epoch {epoch}"
        ):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            genders = [g.lower() for g in genders]

            # Gender split indices
            female_idx = [i for i, g in enumerate(genders) if g == "female"]
            male_idx = [i for i, g in enumerate(genders) if g == "male"]

            if len(female_idx) < 1 or len(male_idx) < 1:
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
            # 1. MoCo-Style Contrastive Loss (핵심)
            # =================================================================
            # Score 계산
            image_scores = _image_level_detection_score(outputs, top_k=args.score_top_k)

            # 성별별 분리
            proj_all = proj_head(features)
            proj_f = proj_all[female_idx]
            proj_m = proj_all[male_idx]
            scores_f = image_scores[female_idx]
            scores_m = image_scores[male_idx]

            # Update memory bank (no grad)
            with torch.no_grad():
                memory_bank.update_female_queue(proj_f)
                memory_bank.update_male_queue(proj_m)
                memory_bank.update_male_centroid(proj_m, momentum=args.momentum)

            # Get negatives and target
            neg_female = memory_bank.get_female_negatives()
            neg_male = memory_bank.male_queue[:memory_bank.male_filled]
            male_centroid = memory_bank.get_male_centroid()

            # Female centroid (for M→F, computed from current batch for simplicity)
            female_centroid = F.normalize(proj_f.mean(0), dim=0) if proj_f.size(0) > 0 else None

            # MoCo contrastive loss
            loss_contrastive, contrastive_info = contrastive_loss_fn(
                proj_f=proj_f,
                proj_m=proj_m,
                male_centroid=male_centroid,
                female_centroid=female_centroid,
                neg_female=neg_female,
                neg_male=neg_male,
                scores_f=scores_f,
                scores_m=scores_m,
            )

            # Apply warmup scaling
            loss_contrastive = loss_contrastive * warmup_scale

            # =================================================================
            # 2. Score-Level Wasserstein (보조)
            # =================================================================
            loss_wasserstein = _wasserstein_1d_asymmetric(scores_f, scores_m)

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
                warmup_scale=warmup_scale,
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                score_f=scores_f.mean().item(),
                score_m=scores_m.mean().item(),
                n_f=len(female_idx),
                n_m=len(male_idx),
                queue_filled=memory_bank.female_filled,
                loss_f2m=contrastive_info.get("loss_f2m", 0.0),
                loss_m2f=contrastive_info.get("loss_m2f", 0.0),
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
                "warmup_scale": warmup_scale,
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "score_f": metrics_logger.meters["score_f"].global_avg,
                "score_m": metrics_logger.meters["score_m"].global_avg,
                "n_f_avg": metrics_logger.meters["n_f"].global_avg,
                "n_m_avg": metrics_logger.meters["n_m"].global_avg,
                "queue_filled": memory_bank.female_filled,
                "loss_f2m": metrics_logger.meters["loss_f2m"].global_avg,
                "loss_m2f": metrics_logger.meters["loss_m2f"].global_avg,
            }

            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(f"\n[Epoch {epoch}] Summary:")
            print(f"  Contrastive Loss: {log_entry['loss_contrastive']:.4f} (F2M: {log_entry['loss_f2m']:.4f}, M2F: {log_entry['loss_m2f']:.4f})")
            print(f"  Wasserstein Loss: {log_entry['loss_wasserstein']:.4f}")
            print(f"  Detection Loss: {log_entry['loss_det']:.4f}")
            print(f"  Total: {log_entry['total_g']:.4f}")
            print(f"  Score (F/M): {log_entry['score_f']:.4f} / {log_entry['score_m']:.4f}")
            print(f"  Memory Bank: {memory_bank.female_filled}/{args.queue_size} filled")
            print(f"  Warmup Scale: {warmup_scale:.2f}, Beta: {current_beta:.4f}")

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
        print("MoCo-Inspired Contrastive Learning Complete!")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print("\n[3rd → 4th 핵심 변경]")
        print("  - Batch 내 비교 → Memory Bank (size={})".format(args.queue_size))
        print("  - 불안정한 target → Momentum Centroid (m={})".format(args.momentum))
        print("  - 과적합 방지: 안정적인 학습 dynamics")
        print("\n기대 효과:")
        print("  - 3rd보다 늦은 peak (Epoch 3 → Epoch 8-12)")
        print("  - 더 안정적인 수렴")
        print("  - AP Gap < 0.09 목표")


if __name__ == "__main__":
    main()
