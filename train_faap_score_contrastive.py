"""
FAAP Training - Score-based Contrastive Learning
=============================================================================

핵심 아이디어: Detection Score 기반 Positive/Anchor 정의
=============================================================================

[기존 방식 (실패)]
- Positive: cross-gender (여성 ↔ 남성)
- Negative: same-gender
- 문제: semantic 유사성 없음, 목표와 연결 안 됨

[제안 방식]
- Positive: Detection score 높은 샘플 (고성능, 주로 남성)
- Anchor: Detection score 낮은 샘플 (저성능, 주로 여성)
- 효과: 저성능 feature → 고성능 feature 방향으로 이동

[Wasserstein과 차이]
- Wasserstein: Score(1D) 분포 정렬 → 결과만 맞춤
- Contrastive: Feature(256D) 정렬 → 원인 해결

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
from .models import FrozenDETR, PerturbationGenerator, clamp_normalized
from .path_utils import DETR_REPO, default_detr_checkpoint, ensure_detr_repo_on_path
import util.misc as utils
from util.misc import NestedTensor


# =============================================================================
# Projection Head (SimCLR 스타일)
# =============================================================================

class ProjectionHead(nn.Module):
    """
    Feature를 contrastive space로 매핑하는 2-layer MLP.
    L2 normalize된 output 반환.
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
            x: (batch, num_queries, feature_dim) or (batch, feature_dim)
        Returns:
            (batch, output_dim) L2-normalized projections
        """
        if x.dim() == 3:
            # (batch, num_queries, feature_dim) → (batch, feature_dim)
            x = x.mean(dim=1)
        proj = self.net(x)
        return F.normalize(proj, dim=-1, p=2)


# =============================================================================
# Score-based Contrastive Loss (핵심)
# =============================================================================

class ScoreBasedContrastiveLoss(nn.Module):
    """
    Detection Score 기반 Contrastive Loss.

    - Anchor: 저성능 샘플 (score < threshold)
    - Positive: 고성능 샘플 (score > threshold)
    - 저성능 feature를 고성능 feature 방향으로 당김

    InfoNCE:
    L = -log( exp(sim(anchor, positive)/τ) / Σ exp(sim(anchor, all)/τ) )
    """

    def __init__(
        self,
        temperature: float = 0.07,
        score_threshold: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.score_threshold = score_threshold

    def forward(
        self,
        projections: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            projections: (N, D) L2-normalized projections
            scores: (N,) detection scores for each sample
        Returns:
            loss (scalar)
        """
        if projections.size(0) < 2:
            return projections.new_tensor(0.0)

        # 고성능/저성능 분리
        high_mask = scores > self.score_threshold
        low_mask = scores <= self.score_threshold

        high_feats = projections[high_mask]  # Positive (목표)
        low_feats = projections[low_mask]    # Anchor (이동 대상)

        n_high = high_feats.size(0)
        n_low = low_feats.size(0)

        # 둘 다 있어야 학습 가능
        if n_high == 0 or n_low == 0:
            return projections.new_tensor(0.0)

        # =================================================================
        # InfoNCE: Anchor → Positive 방향으로 당김
        # =================================================================

        # Anchor와 Positive 간 similarity (가까워져야 함)
        sim_anchor_to_pos = torch.mm(low_feats, high_feats.t()) / self.temperature
        # (n_low, n_high)

        # Anchor와 모든 샘플 간 similarity (분모용)
        sim_anchor_to_all = torch.mm(low_feats, projections.t()) / self.temperature
        # (n_low, N)

        # 자기 자신 제외 (Anchor끼리의 대각선)
        # low_feats의 원래 index 찾기
        low_indices = torch.where(low_mask)[0]
        for i, idx in enumerate(low_indices):
            sim_anchor_to_all[i, idx] = float('-inf')

        # InfoNCE Loss
        # 분자: anchor와 모든 positive의 similarity (logsumexp로 합침)
        numerator = torch.logsumexp(sim_anchor_to_pos, dim=1)  # (n_low,)

        # 분모: anchor와 모든 샘플의 similarity
        denominator = torch.logsumexp(sim_anchor_to_all, dim=1)  # (n_low,)

        # Loss: -log(positive / all)
        loss = -(numerator - denominator).mean()

        return loss


# =============================================================================
# Score-Level Wasserstein Loss (기존 유지)
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
    """단방향 Wasserstein: 여성 score → 남성 score."""
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0)

    sorted_f = female_scores.sort().values
    sorted_m = male_scores.detach().sort().values

    k = max(sorted_f.numel(), sorted_m.numel())
    sorted_f = _resize_sorted(sorted_f, k)
    sorted_m = _resize_sorted(sorted_m, k)

    return F.relu(sorted_m - sorted_f).mean()


def _matched_detection_scores(
    detr: FrozenDETR,
    outputs: dict,
    targets: Sequence[dict],
) -> torch.Tensor:
    """Hungarian matching으로 GT와 매칭된 query의 detection score 추출"""
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


def _get_image_level_scores(
    detr: FrozenDETR,
    outputs: dict,
    targets: Sequence[dict],
) -> torch.Tensor:
    """
    이미지 단위 detection score 계산.
    각 이미지의 매칭된 detection score 평균 반환.
    """
    if len(targets) == 0:
        return outputs["pred_logits"].new_zeros(0)

    indices = detr.criterion.matcher(outputs, targets)
    probs = outputs["pred_logits"].softmax(dim=-1)

    image_scores = []
    for b, (src_idx, tgt_idx) in enumerate(indices):
        if len(src_idx) == 0:
            # 매칭된 object가 없으면 0
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
        "FAAP Score-based Contrastive Learning",
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
    parser.add_argument("--batch_size", type=int, default=8)
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
    parser.add_argument("--lambda_contrastive", type=float, default=1.0,
                        help="Score-based contrastive loss weight")
    parser.add_argument("--lambda_wass", type=float, default=0.2,
                        help="Wasserstein loss weight (score-level)")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Detection loss weight start")
    parser.add_argument("--beta_final", type=float, default=0.6,
                        help="Detection loss weight final")

    # =================================================================
    # Contrastive settings
    # =================================================================
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature for InfoNCE")
    parser.add_argument("--score_threshold", type=float, default=0.5,
                        help="Threshold to split high/low score samples")

    # Projection head
    parser.add_argument("--proj_dim", type=int, default=128,
                        help="Projection output dimension")

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
        print("Score-based Contrastive Learning")
        print("=" * 70)
        print(f"핵심: Detection Score 기반 Positive/Anchor 분리")
        print(f"  - Positive: score > {args.score_threshold} (고성능)")
        print(f"  - Anchor: score <= {args.score_threshold} (저성능)")
        print(f"Temperature: {args.temperature}")
        print(f"Loss weights: Contrastive={args.lambda_contrastive}, Wass={args.lambda_wass}")
        print(f"Beta: {args.beta} → {args.beta_final}")
        print("=" * 70)

    # ==========================================================================
    # Model Initialization
    # ==========================================================================

    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)

    # Projection Head
    proj_head = ProjectionHead(
        input_dim=detr.hidden_dim,
        hidden_dim=detr.hidden_dim,
        output_dim=args.proj_dim,
    ).to(device)

    # Score-based Contrastive Loss
    contrastive_loss_fn = ScoreBasedContrastiveLoss(
        temperature=args.temperature,
        score_threshold=args.score_threshold,
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
        _set_generator_epsilon(generator, current_eps)

        for samples, targets, genders in metrics_logger.log_every(
            train_loader, args.log_every, f"Epoch {epoch}"
        ):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            genders = [g.lower() for g in genders]

            # Gender indices (for Wasserstein loss)
            female_idx = [i for i, g in enumerate(genders) if g == "female"]
            male_idx = [i for i, g in enumerate(genders) if g == "male"]

            # Initialize metrics
            loss_contrastive = torch.tensor(0.0, device=device)
            loss_wasserstein = torch.tensor(0.0, device=device)
            loss_det = torch.tensor(0.0, device=device)
            total_g = torch.tensor(0.0, device=device)

            # =================================================================
            # Forward Pass
            # =================================================================
            opt_g.zero_grad()

            # Apply perturbation
            perturbed = _apply_generator(generator, samples)

            # DETR forward
            outputs, features = detr.forward_with_features(perturbed)

            # =================================================================
            # 1. Image-level Detection Scores
            # =================================================================
            image_scores = _get_image_level_scores(detr, outputs, targets)

            # =================================================================
            # 2. Score-based Contrastive Loss (핵심)
            # =================================================================
            projections = proj_head(features)  # (batch, proj_dim)
            loss_contrastive = contrastive_loss_fn(projections, image_scores)

            # =================================================================
            # 3. Wasserstein Loss (성별 기반, 기존 유지)
            # =================================================================
            if len(female_idx) > 0 and len(male_idx) > 0:
                female_scores = image_scores[female_idx]
                male_scores = image_scores[male_idx]
                loss_wasserstein = _wasserstein_1d_asymmetric(female_scores, male_scores)

            # =================================================================
            # 4. Detection Loss
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

                # Score statistics
                n_high = (image_scores > args.score_threshold).sum().item()
                n_low = (image_scores <= args.score_threshold).sum().item()
                score_mean = image_scores.mean()
                score_std = image_scores.std() if image_scores.numel() > 1 else torch.tensor(0.0)

                # Gender-wise scores
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
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                n_high=n_high,
                n_low=n_low,
                score_mean=score_mean.item(),
                score_std=score_std.item(),
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
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "n_high": metrics_logger.meters["n_high"].global_avg,
                "n_low": metrics_logger.meters["n_low"].global_avg,
                "score_mean": metrics_logger.meters["score_mean"].global_avg,
                "score_std": metrics_logger.meters["score_std"].global_avg,
                "score_f": metrics_logger.meters["score_f"].global_avg,
                "score_m": metrics_logger.meters["score_m"].global_avg,
            }

            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Print summary
            score_gap = log_entry["score_m"] - log_entry["score_f"]
            print(f"\n[Epoch {epoch}] Summary:")
            print(f"  Contrastive Loss: {log_entry['loss_contrastive']:.4f} (핵심)")
            print(f"  Wasserstein Loss: {log_entry['loss_wasserstein']:.4f}")
            print(f"  Detection Loss: {log_entry['loss_det']:.4f}")
            print(f"  Total: {log_entry['total_g']:.4f}")
            print(f"  High/Low score samples: {log_entry['n_high']:.1f} / {log_entry['n_low']:.1f}")
            print(f"  Score (F/M): {log_entry['score_f']:.4f} / {log_entry['score_m']:.4f}")
            print(f"  Score Gap (M-F): {score_gap:.4f}")
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
        print("Score-based Contrastive Learning Complete!")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print("\n기존 방식 대비 변경:")
        print("  - Cross-gender positive → Score-based positive")
        print("  - 성별 기준 → Detection score 기준")
        print("  - 목표와 직접 연결")
        print("\n핵심 메커니즘:")
        print(f"  - Positive: score > {args.score_threshold} (고성능)")
        print(f"  - Anchor: score <= {args.score_threshold} (저성능)")
        print("  - Anchor를 Positive 방향으로 당김")
        print("\n성공 기준:")
        print("  - AP Gap < 0.09 (15% 개선)")
        print("  - Female AP > 0.41")


if __name__ == "__main__":
    main()
