"""
FAAP Training - Standard InfoNCE: Cross-Gender Contrastive Learning

=============================================================================
핵심 아이디어: GenderDiscriminator를 표준 InfoNCE로 대체
=============================================================================

[7th (Adversarial) 방식]
- GenderDiscriminator로 성별 구분
- d_loss = cross_entropy(discriminator(feat), gender_labels)
- fairness_loss = -(ce + α * entropy)
- 한계: AP Gap 미개선 (AR Gap만 60% 개선)

[본 연구: 표준 InfoNCE]
L_InfoNCE = -log(exp(sim(z_f, z_m)/τ) / [exp(sim(z_f, z_m)/τ) + Σ exp(sim(z_f, z_f')/τ)])

- Positive: cross-gender (여성 ↔ 남성)
- Negative: same-gender (여성 ↔ 여성, 남성 ↔ 남성)
- 비대칭 가중치: F→M (1.5) > M→F (0.5)

[7th 대비 차이점]
- GenderDiscriminator 제거
- D/G 번갈아 학습 → G만 학습
- Wasserstein 및 Detection loss는 7th 유지

=============================================================================
"""

import argparse
import json
from pathlib import Path
from typing import List, Sequence

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
# Cross-Gender InfoNCE Loss (핵심: GenderDiscriminator 대체)
# =============================================================================

class CrossGenderInfoNCELoss(nn.Module):
    """
    표준 InfoNCE: cross-gender=positive, same-gender=negative

    수식:
    L = -log(exp(sim(z_f, z_m)/τ) / [exp(sim(z_f, z_m)/τ) + Σ exp(sim(z_f, z_f')/τ)])

    - z_f: 여성 feature의 projection
    - z_m: 남성 feature의 projection
    - z_f': 다른 여성 feature (negative)
    - τ: temperature (0.07, SimCLR 표준)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        asymmetric_f: float = 1.5,  # 여성 → 남성 방향 강화
        asymmetric_m: float = 0.5,  # 남성 → 여성 방향 약화
    ):
        super().__init__()
        self.temperature = temperature
        self.asymmetric_f = asymmetric_f
        self.asymmetric_m = asymmetric_m

    def forward(
        self,
        proj_f: torch.Tensor,
        proj_m: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            proj_f: (N_f, D) 여성 projections (L2-normalized)
            proj_m: (N_m, D) 남성 projections (L2-normalized)
        Returns:
            loss (scalar)
        """
        if proj_f.size(0) == 0 or proj_m.size(0) == 0:
            return proj_f.new_tensor(0.0)

        # =================================================================
        # 1. Female → Male InfoNCE
        # Positive: 모든 남성 샘플
        # Negative: 같은 성별 (다른 여성 샘플)
        # =================================================================
        sim_f2m = torch.mm(proj_f, proj_m.t()) / self.temperature  # (N_f, N_m)
        sim_f2f = torch.mm(proj_f, proj_f.t()) / self.temperature  # (N_f, N_f)

        # 자기 자신과의 similarity 제거 (대각선 마스킹)
        mask_f = torch.eye(proj_f.size(0), device=proj_f.device, dtype=torch.bool)
        sim_f2f = sim_f2f.masked_fill(mask_f, float('-inf'))

        # InfoNCE: -log(exp(pos) / (exp(pos) + exp(neg)))
        all_sims_f = torch.cat([sim_f2m, sim_f2f], dim=1)
        numerator_f = torch.logsumexp(sim_f2m, dim=1)  # positive들의 logsumexp
        denominator_f = torch.logsumexp(all_sims_f, dim=1)  # all의 logsumexp
        loss_f2m = -(numerator_f - denominator_f).mean()

        # =================================================================
        # 2. Male → Female InfoNCE (대칭적으로, 약하게)
        # =================================================================
        sim_m2f = torch.mm(proj_m, proj_f.t()) / self.temperature  # (N_m, N_f)
        sim_m2m = torch.mm(proj_m, proj_m.t()) / self.temperature  # (N_m, N_m)

        mask_m = torch.eye(proj_m.size(0), device=proj_m.device, dtype=torch.bool)
        sim_m2m = sim_m2m.masked_fill(mask_m, float('-inf'))

        all_sims_m = torch.cat([sim_m2f, sim_m2m], dim=1)
        numerator_m = torch.logsumexp(sim_m2f, dim=1)
        denominator_m = torch.logsumexp(all_sims_m, dim=1)
        loss_m2f = -(numerator_m - denominator_m).mean()

        # 비대칭 가중합
        return self.asymmetric_f * loss_f2m + self.asymmetric_m * loss_m2f


# =============================================================================
# Score-Level Wasserstein Loss (7th에서 유지)
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
    detr: FrozenDETR,
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
        "FAAP Standard InfoNCE: Cross-Gender Contrastive Learning",
        add_help=True,
    )

    # Paths
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()))
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(Path(__file__)))

    # Training basics
    # CUDA_VISIBLE_DEVICES=2 환경에서는 cuda:0 또는 cuda 사용
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=24)  # 7th와 동일
    parser.add_argument("--batch_size", type=int, default=4)  # augmentation 포함 시 8은 OOM 발생
    parser.add_argument("--num_workers", type=int, default=6)  # 최적화 결과: 6 (88.9 samples/s)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    # Perturbation settings (7th 기반)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--epsilon_final", type=float, default=0.10)
    parser.add_argument("--epsilon_min", type=float, default=0.09)
    parser.add_argument("--epsilon_warmup_epochs", type=int, default=8)
    parser.add_argument("--epsilon_hold_epochs", type=int, default=6)
    parser.add_argument("--epsilon_cooldown_epochs", type=int, default=10)

    # =================================================================
    # Loss weights (계획서 하이퍼파라미터)
    # =================================================================
    parser.add_argument("--lambda_infonce", type=float, default=1.0,
                        help="InfoNCE loss weight (GenderDiscriminator 대체)")
    parser.add_argument("--lambda_wass", type=float, default=0.2,
                        help="Score-level Wasserstein loss weight (7th 유지)")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Detection loss weight start (7th 유지)")
    parser.add_argument("--beta_final", type=float, default=0.6,
                        help="Detection loss weight final (7th 유지)")

    # =================================================================
    # InfoNCE settings (계획서 하이퍼파라미터)
    # =================================================================
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature for InfoNCE (SimCLR standard: 0.07)")
    parser.add_argument("--asymmetric_f", type=float, default=1.5,
                        help="Female→Male direction weight")
    parser.add_argument("--asymmetric_m", type=float, default=0.5,
                        help="Male→Female direction weight")

    # Projection head
    parser.add_argument("--proj_dim", type=int, default=128,
                        help="Projection output dimension")

    # =================================================================
    # SimCLR-Style Data Augmentation
    # =================================================================
    parser.add_argument("--aug_strength", type=str, default="medium",
                        choices=["none", "weak", "medium", "strong"],
                        help="SimCLR augmentation strength: "
                             "none (off), weak (0.2), medium (0.3, 추천), strong (0.4+grayscale)")

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
        print("Standard InfoNCE: Cross-Gender Contrastive Learning")
        print("(GenderDiscriminator 대체)")
        print("=" * 70)
        print(f"Temperature: {args.temperature} (SimCLR standard)")
        print(f"Asymmetric weights: F→M={args.asymmetric_f}, M→F={args.asymmetric_m}")
        print(f"Loss weights: InfoNCE={args.lambda_infonce}, Wass={args.lambda_wass}")
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

    # CrossGender InfoNCE Loss
    infonce_loss_fn = CrossGenderInfoNCELoss(
        temperature=args.temperature,
        asymmetric_f=args.asymmetric_f,
        asymmetric_m=args.asymmetric_m,
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

        # Schedules (7th와 동일)
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

            # Gender split
            female_idx = [i for i, g in enumerate(genders) if g == "female"]
            male_idx = [i for i, g in enumerate(genders) if g == "male"]
            female_batch, female_targets = _split_nested(samples, targets, female_idx)
            male_batch, male_targets = _split_nested(samples, targets, male_idx)

            # Initialize metrics
            loss_infonce = torch.tensor(0.0, device=device)
            loss_wasserstein = torch.tensor(0.0, device=device)
            loss_det = torch.tensor(0.0, device=device)
            total_g = torch.tensor(0.0, device=device)
            delta_linf = torch.tensor(0.0, device=device)
            delta_l2 = torch.tensor(0.0, device=device)
            obj_mean_f = torch.tensor(0.0, device=device)
            obj_mean_m = torch.tensor(0.0, device=device)
            obj_frac_f = torch.tensor(0.0, device=device)
            obj_frac_m = torch.tensor(0.0, device=device)

            # Skip if either gender missing
            if female_batch is None or male_batch is None:
                metrics_logger.update(
                    loss_infonce=0.0, loss_wasserstein=0.0, loss_det=0.0,
                    total_g=0.0, eps=current_eps, beta=current_beta,
                    delta_linf=0.0, delta_l2=0.0,
                    obj_score_f=0.0, obj_score_m=0.0,
                    obj_frac_f=0.0, obj_frac_m=0.0,
                )
                continue

            # =================================================================
            # Forward Pass
            # =================================================================
            opt_g.zero_grad()

            # Apply perturbation
            female_perturbed = _apply_generator(generator, female_batch)
            male_perturbed = _apply_generator(generator, male_batch)

            # Apply SimCLR augmentation (training only)
            if generator.training and args.aug_strength != "none":
                female_perturbed = NestedTensor(
                    simclr_aug(female_perturbed.tensors),
                    female_perturbed.mask,
                )
                male_perturbed = NestedTensor(
                    simclr_aug(male_perturbed.tensors),
                    male_perturbed.mask,
                )

            # DETR forward (with features)
            outputs_f, feat_f = detr.forward_with_features(female_perturbed)
            outputs_m, feat_m = detr.forward_with_features(male_perturbed)

            # =================================================================
            # 1. InfoNCE Loss (핵심: GenderDiscriminator 대체)
            # =================================================================
            proj_f = proj_head(feat_f)  # (N_f, proj_dim)
            proj_m = proj_head(feat_m)  # (N_m, proj_dim)
            loss_infonce = infonce_loss_fn(proj_f, proj_m)

            # =================================================================
            # 2. Score-Level Wasserstein (7th 유지)
            # =================================================================
            female_scores = _matched_detection_scores(detr, outputs_f, female_targets)
            male_scores = _matched_detection_scores(detr, outputs_m, male_targets)
            loss_wasserstein = _wasserstein_1d_asymmetric(female_scores, male_scores)

            # =================================================================
            # 3. Detection Loss (7th 유지)
            # =================================================================
            loss_det_f, _ = detr.detection_loss(outputs_f, female_targets)
            loss_det_m, _ = detr.detection_loss(outputs_m, male_targets)
            loss_det = loss_det_f + loss_det_m

            # =================================================================
            # Total Loss
            # =================================================================
            total_g = (
                args.lambda_infonce * loss_infonce
                + args.lambda_wass * loss_wasserstein
                + current_beta * loss_det
            )

            # =================================================================
            # Metrics (7th 스타일 유지)
            # =================================================================
            with torch.no_grad():
                delta_f = female_perturbed.tensors - female_batch.tensors
                delta_m = male_perturbed.tensors - male_batch.tensors
                delta_cat = torch.cat([delta_f, delta_m], dim=0)
                delta_linf = delta_cat.abs().amax(dim=(1, 2, 3)).mean()
                delta_l2 = delta_cat.flatten(1).norm(p=2, dim=1).mean()

                # Objectness scores
                probs_f = outputs_f["pred_logits"].softmax(dim=-1)[..., :-1]
                max_scores_f = probs_f.max(dim=-1).values
                obj_mean_f = max_scores_f.mean()
                obj_frac_f = (max_scores_f > args.obj_conf_thresh).float().mean()

                probs_m = outputs_m["pred_logits"].softmax(dim=-1)[..., :-1]
                max_scores_m = probs_m.max(dim=-1).values
                obj_mean_m = max_scores_m.mean()
                obj_frac_m = (max_scores_m > args.obj_conf_thresh).float().mean()

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
                loss_infonce=loss_infonce.item(),
                loss_wasserstein=loss_wasserstein.item(),
                loss_det=loss_det.item(),
                total_g=total_g.item(),
                eps=current_eps,
                beta=current_beta,
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                obj_score_f=obj_mean_f.item(),
                obj_score_m=obj_mean_m.item(),
                obj_frac_f=obj_frac_f.item(),
                obj_frac_m=obj_frac_m.item(),
            )

        # =====================================================================
        # End of Epoch
        # =====================================================================
        metrics_logger.synchronize_between_processes()

        if utils.is_main_process():
            log_entry = {
                "epoch": epoch,
                "loss_infonce": metrics_logger.meters["loss_infonce"].global_avg,
                "loss_wasserstein": metrics_logger.meters["loss_wasserstein"].global_avg,
                "loss_det": metrics_logger.meters["loss_det"].global_avg,
                "total_g": metrics_logger.meters["total_g"].global_avg,
                "epsilon": current_eps,
                "beta": current_beta,
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "obj_score_f": metrics_logger.meters["obj_score_f"].global_avg,
                "obj_score_m": metrics_logger.meters["obj_score_m"].global_avg,
                "obj_frac_f": metrics_logger.meters["obj_frac_f"].global_avg,
                "obj_frac_m": metrics_logger.meters["obj_frac_m"].global_avg,
            }

            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Print summary
            obj_gap = log_entry["obj_score_m"] - log_entry["obj_score_f"]
            print(f"\n[Epoch {epoch}] Summary:")
            print(f"  InfoNCE Loss: {log_entry['loss_infonce']:.4f} (핵심)")
            print(f"  Wasserstein Loss: {log_entry['loss_wasserstein']:.4f}")
            print(f"  Detection Loss: {log_entry['loss_det']:.4f}")
            print(f"  Total: {log_entry['total_g']:.4f}")
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
        print("Standard InfoNCE Training Complete!")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print("\n7th (Adversarial) 대비 변경 사항:")
        print("  - GenderDiscriminator 제거")
        print("  - D/G 번갈아 학습 → G만 학습")
        print("  - Adversarial → 표준 InfoNCE")
        print("  - Positive: cross-gender, Negative: same-gender")
        print(f"  - SimCLR Augmentation: {args.aug_strength}")
        print("\n7th에서 유지:")
        print("  - Wasserstein score alignment")
        print("  - Detection loss")
        print("  - Epsilon/Beta schedules")
        print("\n성공 기준:")
        print("  - AP Gap < 0.09 (15% 개선)")
        print("  - Female AP > 0.41")


if __name__ == "__main__":
    main()
