"""
FAAP Training - 14th: AP Gap 직접 개선을 위한 Score-Aware Fairness

=============================================================================
7th 분석 결과:
- AR Gap: 0.0081 → 0.0032 (60% 개선) ✅
- AP Gap: 0.1063 → 0.1059 (거의 변화 없음) ❌
- Female AP: 0.4045 → 0.4078 (+0.0034) ✅

Contrastive IoU 실패 원인:
- IoU gap이 원래 없었음 (0.766 vs 0.770)
- Feature alignment가 AP에 직접 영향 못 줌
- Projection head가 평가 시 미사용

=============================================================================
14th 핵심 전략: AP Gap 직접 공략
=============================================================================

1. Quantile-Weighted Wasserstein
   - 기존: 전체 score 분포 평균 정렬
   - 개선: High-score 영역(상위 분위)에 가중치 부여
   - 이유: AP는 high-confidence prediction의 precision에 민감

2. Confidence Margin Loss
   - Female TP의 confidence가 특정 margin 이상 되도록 강제
   - Male TP confidence를 타겟으로 margin 학습

3. Score Gap Penalty
   - 배치 내에서 Female/Male 평균 score gap에 직접 패널티
   - 단순하지만 직접적인 AP gap 감소 효과 기대

4. Detection Loss 비대칭
   - Female detection loss에 더 높은 가중치
   - Female의 localization + classification 품질 향상

=============================================================================
목표:
- AP Gap: 0.106 → 0.08 (25% 개선)
- Female AP: 0.404 → 0.42 (4% 향상)
- Male AP 유지 또는 소폭 상승
=============================================================================
"""

import argparse
import json
from pathlib import Path
from typing import List, Sequence

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
from .models import FrozenDETR, GenderDiscriminator, PerturbationGenerator, clamp_normalized
from .path_utils import DETR_REPO, default_detr_checkpoint, ensure_detr_repo_on_path
import util.misc as utils
from util.misc import NestedTensor


def _default_output_dir(script_path: Path) -> str:
    stem = script_path.stem
    stem_lower = stem.lower()
    suffix = stem
    for prefix in ("train_faap_wgan_", "train_"):
        if stem_lower.startswith(prefix):
            suffix = stem[len(prefix):]
            break
    suffix = suffix.lower()
    return str(Path("faap_outputs") / f"faap_outputs_{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FAAP 14th - AP Gap Direct Optimization", add_help=True)

    # Paths
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()))
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(Path(__file__)))

    # Training basics
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--k_d", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Epsilon schedule (7th와 동일)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--epsilon_final", type=float, default=0.10)
    parser.add_argument("--epsilon_warmup_epochs", type=int, default=8)
    parser.add_argument("--epsilon_hold_epochs", type=int, default=6)
    parser.add_argument("--epsilon_cooldown_epochs", type=int, default=16)
    parser.add_argument("--epsilon_min", type=float, default=0.09)

    # Loss weights
    parser.add_argument("--alpha", type=float, default=0.2, help="entropy weight")
    parser.add_argument("--beta", type=float, default=0.5, help="detection loss weight start")
    parser.add_argument("--beta_final", type=float, default=0.6, help="detection loss weight final")
    parser.add_argument("--lambda_fair", type=float, default=2.0, help="fairness loss weight")
    parser.add_argument("--fair_f_scale", type=float, default=1.0)
    parser.add_argument("--fair_m_scale", type=float, default=0.5)

    # === 14th 핵심: AP Gap 관련 하이퍼파라미터 ===
    parser.add_argument("--lambda_w", type=float, default=0.3,
                        help="Wasserstein loss weight (7th: 0.2 → 14th: 0.3)")
    parser.add_argument("--lambda_quantile_w", type=float, default=0.2,
                        help="Quantile-weighted Wasserstein weight (NEW)")
    parser.add_argument("--lambda_margin", type=float, default=0.1,
                        help="Confidence margin loss weight (NEW)")
    parser.add_argument("--lambda_gap", type=float, default=0.15,
                        help="Direct score gap penalty weight (NEW)")
    parser.add_argument("--margin_target", type=float, default=0.05,
                        help="Target margin for female confidence")
    parser.add_argument("--quantile_focus", type=float, default=0.7,
                        help="Focus on top X quantile (0.7 = top 30%)")
    parser.add_argument("--det_f_scale", type=float, default=1.2,
                        help="Female detection loss scaling (>1 = more weight)")

    # Other
    parser.add_argument("--max_train_per_gender", type=int, default=0)
    parser.add_argument("--obj_conf_thresh", type=float, default=0.5)
    parser.add_argument("--max_norm", type=float, default=0.1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")

    # Distributed
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--dist_url", default="env://")

    return parser.parse_args()


# =============================================================================
# Utility Functions
# =============================================================================

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


def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    return -(probs * log_probs).sum(dim=-1).mean()


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


def _unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def _set_generator_epsilon(generator: nn.Module, epsilon: float) -> None:
    _unwrap_ddp(generator).epsilon = epsilon


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


def _matched_detection_scores(detr: FrozenDETR, outputs: dict, targets: Sequence[dict]) -> torch.Tensor:
    """Hungarian matching으로 매칭된 detection의 confidence score 추출"""
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
# 14th NEW: AP Gap 직접 공략 Loss Functions
# =============================================================================

def _wasserstein_1d_asymmetric(female_scores: torch.Tensor, male_scores: torch.Tensor) -> torch.Tensor:
    """단방향 Wasserstein: Female → Male 방향으로만 끌어올림 (7th와 동일)"""
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0)

    sorted_f = female_scores.sort().values
    sorted_m = male_scores.detach().sort().values

    k = max(sorted_f.numel(), sorted_m.numel())
    sorted_f = _resize_sorted(sorted_f, k)
    sorted_m = _resize_sorted(sorted_m, k)

    return F.relu(sorted_m - sorted_f).mean()


def _quantile_weighted_wasserstein(
    female_scores: torch.Tensor,
    male_scores: torch.Tensor,
    quantile_focus: float = 0.7,
) -> torch.Tensor:
    """
    상위 분위에 가중치를 부여한 Wasserstein Loss.

    AP는 high-confidence prediction의 precision에 민감하므로,
    상위 score 영역의 정렬에 더 집중함.

    Args:
        quantile_focus: 이 값 이상의 분위수에 집중 (0.7 = 상위 30%)
    """
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0)

    sorted_f = female_scores.sort().values
    sorted_m = male_scores.detach().sort().values

    k = max(sorted_f.numel(), sorted_m.numel())
    sorted_f = _resize_sorted(sorted_f, k)
    sorted_m = _resize_sorted(sorted_m, k)

    # 분위수 기반 가중치: 상위 분위일수록 높은 가중치
    positions = torch.linspace(0, 1, k, device=female_scores.device)
    weights = torch.where(
        positions >= quantile_focus,
        torch.tensor(2.0, device=female_scores.device),  # 상위 분위: 2배 가중치
        torch.tensor(0.5, device=female_scores.device),  # 하위 분위: 0.5배 가중치
    )

    gaps = F.relu(sorted_m - sorted_f)
    weighted_loss = (gaps * weights).sum() / weights.sum()

    return weighted_loss


def _confidence_margin_loss(
    female_scores: torch.Tensor,
    male_scores: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    """
    Female confidence가 Male 대비 최소 margin만큼 되도록 유도.

    Female mean ≥ Male mean - margin 이면 loss = 0
    그렇지 않으면 차이에 비례한 패널티
    """
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0)

    mean_f = female_scores.mean()
    mean_m = male_scores.detach().mean()

    # Female이 Male - margin보다 낮으면 패널티
    target = mean_m - margin
    loss = F.relu(target - mean_f)

    return loss


def _direct_score_gap_penalty(
    female_scores: torch.Tensor,
    male_scores: torch.Tensor,
) -> torch.Tensor:
    """
    배치 내 Female/Male 평균 score gap에 직접 패널티.

    가장 단순하지만 직접적인 방법.
    """
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0)

    mean_f = female_scores.mean()
    mean_m = male_scores.detach().mean()

    # Gap이 양수면(Male > Female) 패널티
    gap = F.relu(mean_m - mean_f)

    return gap


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

    output_dir = Path(args.output_dir)
    if utils.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)

        # Save config
        with (output_dir / "config.json").open("w") as f:
            json.dump(vars(args), f, indent=2)

    if args.distributed:
        dist.barrier()

    if utils.is_main_process():
        dataset_info = inspect_faap_dataset(Path(args.dataset_root))
        with (output_dir / "dataset_layout.json").open("w") as f:
            json.dump(dataset_info, f, indent=2)

        print("=" * 70)
        print("FAAP 14th - AP Gap Direct Optimization")
        print("=" * 70)
        print(f"Lambda Wasserstein: {args.lambda_w}")
        print(f"Lambda Quantile-W:  {args.lambda_quantile_w}")
        print(f"Lambda Margin:      {args.lambda_margin}")
        print(f"Lambda Gap:         {args.lambda_gap}")
        print(f"Quantile Focus:     {args.quantile_focus}")
        print(f"Det Female Scale:   {args.det_f_scale}")
        print("=" * 70)

    # Models
    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)
    discriminator = GenderDiscriminator(feature_dim=detr.hidden_dim).to(device)

    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)
        discriminator = DDP(discriminator, device_ids=[args.gpu] if args.gpu is not None else None)

    opt_g = torch.optim.Adam(_unwrap_ddp(generator).parameters(), lr=args.lr_g)
    opt_d = torch.optim.Adam(_unwrap_ddp(discriminator).parameters(), lr=args.lr_d)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if "generator" in ckpt:
            _unwrap_ddp(generator).load_state_dict(ckpt["generator"])
        if "discriminator" in ckpt:
            _unwrap_ddp(discriminator).load_state_dict(ckpt["discriminator"])
        if "opt_g" in ckpt:
            opt_g.load_state_dict(ckpt["opt_g"])
        if "opt_d" in ckpt:
            opt_d.load_state_dict(ckpt["opt_d"])
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
        discriminator.train()

        if args.distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

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

            female_idx = [i for i, g in enumerate(genders) if g == "female"]
            male_idx = [i for i, g in enumerate(genders) if g == "male"]
            female_batch, female_targets = _split_nested(samples, targets, female_idx)
            male_batch, male_targets = _split_nested(samples, targets, male_idx)

            # Initialize metrics
            delta_linf = torch.tensor(0.0, device=device)
            delta_l2 = torch.tensor(0.0, device=device)
            obj_mean_f = torch.tensor(0.0, device=device)
            obj_mean_m = torch.tensor(0.0, device=device)
            score_gap = torch.tensor(0.0, device=device)

            # ================================================================
            # Discriminator Update
            # ================================================================
            for _ in range(args.k_d):
                d_losses = []
                opt_d.zero_grad()

                if female_batch is not None:
                    with torch.no_grad():
                        female_perturbed_d = _apply_generator(generator, female_batch)
                        _, feat_f = detr.forward_with_features(female_perturbed_d)
                    logits_f = discriminator(feat_f.detach())
                    labels_f = torch.ones(logits_f.size(0), device=device, dtype=torch.long)
                    d_losses.append(F.cross_entropy(logits_f, labels_f))

                if male_batch is not None:
                    with torch.no_grad():
                        male_perturbed_d = _apply_generator(generator, male_batch)
                        _, feat_m = detr.forward_with_features(male_perturbed_d)
                    logits_m = discriminator(feat_m.detach())
                    labels_m = torch.zeros(logits_m.size(0), device=device, dtype=torch.long)
                    d_losses.append(F.cross_entropy(logits_m, labels_m))

                if d_losses:
                    d_loss = torch.stack(d_losses).mean()
                    d_loss.backward()
                    opt_d.step()
                else:
                    d_loss = torch.tensor(0.0, device=device)

            # ================================================================
            # Generator Update
            # ================================================================
            if female_batch is not None or male_batch is not None:
                opt_g.zero_grad()

                fairness_loss = torch.tensor(0.0, device=device)
                fairness_f = torch.tensor(0.0, device=device)
                fairness_m = torch.tensor(0.0, device=device)
                det_loss = torch.tensor(0.0, device=device)
                wasserstein_loss = torch.tensor(0.0, device=device)
                quantile_w_loss = torch.tensor(0.0, device=device)
                margin_loss = torch.tensor(0.0, device=device)
                gap_loss = torch.tensor(0.0, device=device)

                female_scores = torch.tensor([], device=device)
                male_scores = torch.tensor([], device=device)

                # Female forward
                if female_batch is not None:
                    female_perturbed = _apply_generator(generator, female_batch)
                    outputs_f, feat_f = detr.forward_with_features(female_perturbed)

                    # Fairness (adversarial)
                    logits_f = discriminator(feat_f)
                    ce_f = F.cross_entropy(logits_f, torch.ones(logits_f.size(0), device=device, dtype=torch.long))
                    ent_f = _entropy_loss(logits_f)
                    fairness_f = -(ce_f + args.alpha * ent_f)

                    # Detection loss (with female scaling)
                    det_f, _ = detr.detection_loss(outputs_f, female_targets)
                    det_loss = det_loss + args.det_f_scale * det_f

                    # Matched scores
                    female_scores = _matched_detection_scores(detr, outputs_f, female_targets)

                # Male forward
                if male_batch is not None:
                    male_perturbed = _apply_generator(generator, male_batch)
                    outputs_m, feat_m = detr.forward_with_features(male_perturbed)

                    # Fairness (adversarial)
                    logits_m = discriminator(feat_m)
                    ce_m = F.cross_entropy(logits_m, torch.zeros(logits_m.size(0), device=device, dtype=torch.long))
                    ent_m = _entropy_loss(logits_m)
                    fairness_m = -(ce_m + args.alpha * ent_m)

                    # Detection loss (normal scaling)
                    det_m, _ = detr.detection_loss(outputs_m, male_targets)
                    det_loss = det_loss + det_m

                    # Matched scores
                    male_scores = _matched_detection_scores(detr, outputs_m, male_targets)

                # ============================================================
                # 14th NEW: AP Gap 직접 공략 Losses
                # ============================================================

                # 1. 기본 Wasserstein (7th와 동일)
                wasserstein_loss = _wasserstein_1d_asymmetric(female_scores, male_scores)

                # 2. Quantile-Weighted Wasserstein (상위 분위 집중)
                quantile_w_loss = _quantile_weighted_wasserstein(
                    female_scores, male_scores, args.quantile_focus
                )

                # 3. Confidence Margin Loss
                margin_loss = _confidence_margin_loss(
                    female_scores, male_scores, args.margin_target
                )

                # 4. Direct Score Gap Penalty
                gap_loss = _direct_score_gap_penalty(female_scores, male_scores)

                # Combine fairness
                fairness_loss = args.fair_f_scale * fairness_f + args.fair_m_scale * fairness_m

                # Total Generator Loss
                total_g = (
                    args.lambda_fair * fairness_loss
                    + current_beta * det_loss
                    + args.lambda_w * wasserstein_loss
                    + args.lambda_quantile_w * quantile_w_loss
                    + args.lambda_margin * margin_loss
                    + args.lambda_gap * gap_loss
                )

                # ============================================================
                # Metrics
                # ============================================================
                with torch.no_grad():
                    deltas = []
                    if female_batch is not None:
                        delta_f = female_perturbed.tensors - female_batch.tensors
                        deltas.append(delta_f)
                    if male_batch is not None:
                        delta_m = male_perturbed.tensors - male_batch.tensors
                        deltas.append(delta_m)

                    if deltas:
                        delta_cat = torch.cat(deltas, dim=0)
                        delta_linf = delta_cat.abs().amax(dim=(1, 2, 3)).mean()
                        delta_l2 = delta_cat.flatten(1).norm(p=2, dim=1).mean()

                    if female_scores.numel() > 0:
                        obj_mean_f = female_scores.mean()
                    if male_scores.numel() > 0:
                        obj_mean_m = male_scores.mean()

                    if female_scores.numel() > 0 and male_scores.numel() > 0:
                        score_gap = male_scores.mean() - female_scores.mean()

                # Backward
                total_g.backward()
                if args.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.max_norm)
                opt_g.step()

            else:
                fairness_loss = torch.tensor(0.0, device=device)
                det_loss = torch.tensor(0.0, device=device)
                total_g = torch.tensor(0.0, device=device)
                wasserstein_loss = torch.tensor(0.0, device=device)
                quantile_w_loss = torch.tensor(0.0, device=device)
                margin_loss = torch.tensor(0.0, device=device)
                gap_loss = torch.tensor(0.0, device=device)

            # Log metrics
            metrics_logger.update(
                d_loss=d_loss.item(),
                g_fair=fairness_loss.item(),
                g_det=det_loss.item(),
                g_total=total_g.item(),
                g_w=wasserstein_loss.item(),
                g_qw=quantile_w_loss.item(),
                g_margin=margin_loss.item(),
                g_gap=gap_loss.item(),
                eps=current_eps,
                beta=current_beta,
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                score_f=obj_mean_f.item(),
                score_m=obj_mean_m.item(),
                score_gap=score_gap.item(),
            )

        # =====================================================================
        # End of Epoch
        # =====================================================================
        metrics_logger.synchronize_between_processes()

        if utils.is_main_process():
            log_entry = {
                "epoch": epoch,
                "d_loss": metrics_logger.meters["d_loss"].global_avg,
                "g_fair": metrics_logger.meters["g_fair"].global_avg,
                "g_det": metrics_logger.meters["g_det"].global_avg,
                "g_total": metrics_logger.meters["g_total"].global_avg,
                "g_w": metrics_logger.meters["g_w"].global_avg,
                "g_qw": metrics_logger.meters["g_qw"].global_avg,
                "g_margin": metrics_logger.meters["g_margin"].global_avg,
                "g_gap": metrics_logger.meters["g_gap"].global_avg,
                "epsilon": current_eps,
                "beta": current_beta,
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "score_f": metrics_logger.meters["score_f"].global_avg,
                "score_m": metrics_logger.meters["score_m"].global_avg,
                "score_gap": metrics_logger.meters["score_gap"].global_avg,
            }

            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Print summary
            print(f"\n[Epoch {epoch}] Summary:")
            print(f"  Score F/M: {log_entry['score_f']:.4f} / {log_entry['score_m']:.4f}")
            print(f"  Score Gap (M-F): {log_entry['score_gap']:.4f}")
            print(f"  Wasserstein: {log_entry['g_w']:.4f}, Quantile-W: {log_entry['g_qw']:.4f}")
            print(f"  Margin: {log_entry['g_margin']:.4f}, Gap: {log_entry['g_gap']:.4f}")
            print(f"  Detection: {log_entry['g_det']:.4f}, Fairness: {log_entry['g_fair']:.4f}")
            print(f"  Epsilon: {current_eps:.4f}, Beta: {current_beta:.4f}")

            # Save checkpoint
            if (epoch + 1) % args.save_every == 0:
                ckpt_path_save = output_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "generator": _unwrap_ddp(generator).state_dict(),
                        "discriminator": _unwrap_ddp(discriminator).state_dict(),
                        "opt_g": opt_g.state_dict(),
                        "opt_d": opt_d.state_dict(),
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
        print("14th Training Complete!")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print("\nKey improvements over 7th:")
        print("  1. Quantile-Weighted Wasserstein (top 30% focus)")
        print("  2. Confidence Margin Loss")
        print("  3. Direct Score Gap Penalty")
        print("  4. Female Detection Loss Scaling (1.2x)")
        print("\nExpected improvements:")
        print("  - AP Gap: 0.106 → 0.08 (25% reduction)")
        print("  - Female AP: 0.404 → 0.42 (4% increase)")


if __name__ == "__main__":
    main()
