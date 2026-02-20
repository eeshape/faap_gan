"""
FAAP Training - Asymmetric Detection Loss (8th Version)

=============================================================================
3rd~7th 실험 분석 요약:
=============================================================================
- 3rd ep3이 유일한 AP Gap 감소 성공 (0.1063 → 0.1044, -1.8%)
- 성공 핵심: Male AP 거의 불변 (+0.0001), Female AP만 상승 (+0.0020)
- 4th~7th 실패 원인: Detection loss가 Male/Female AP를 동시에 올림 (universal enhancer)

=============================================================================
8th 버전 핵심 아이디어: Asymmetric Detection Loss
=============================================================================

[핵심 변경]
- 3rd: 단일 detection_loss(outputs, targets) → Male/Female AP 동시 상승
- 8th: Gender별 분리 detection loss
  - Female: lambda_det_f * detection_loss(outputs_f, targets_f)  (정상 - Female 검출 향상)
  - Male:   lambda_det_m * detection_loss(outputs_m, targets_m)  (약화 - Male AP 상승 억제)

[Architect 리뷰 반영]
- aux_outputs (5개 intermediate decoder layer) 반드시 gender별 slicing
- aux_outputs 누락 시 detection loss의 ~50% 손실

[Critic 리뷰 반영]
- lambda_det_m = 0.1 (완전 제거 아닌 약화, fix3 교훈)
- 성공 기준 현실적 조정: AP Gap < 0.100

[유지 컴포넌트 (3rd 검증)]
- GenderAwareContrastiveLoss (Adaptive Weighting [0.5, 1.5])
- Wasserstein 1D Asymmetric
- SimCLR Augmentation (medium)
- SimCLR Projection Head

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
# Gender-Aware Score-Based Contrastive Loss (3rd 검증, 변경 없음)
# =============================================================================

class GenderAwareContrastiveLoss(nn.Module):
    """
    Gender + Score 결합 대조학습 (3rd에서 검증, 8th에서 그대로 유지)

    수식:
    L = -log(exp(sim(z_f, z_m)/t) / [exp(sim(z_f, z_m)/t) + Sum exp(sim(z_f, z_f')/t)])

    Adaptive Weighting:
    - Score 차이에 비례한 가중치 적용 [0.5, 1.5]
    - 비대칭 결합: F->M 1.5, M->F 0.5
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
        proj_f: torch.Tensor,
        proj_m: torch.Tensor,
        scores_f: torch.Tensor,
        scores_m: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        n_f = proj_f.size(0)
        n_m = proj_m.size(0)

        if n_f < 2 or n_m < 1:
            return proj_f.new_tensor(0.0), {"n_f": n_f, "n_m": n_m, "score_gap": 0.0}

        scores_f = scores_f.detach()
        scores_m = scores_m.detach()

        # Female -> Male Contrastive
        sim_f2m = torch.mm(proj_f, proj_m.t()) / self.temperature
        sim_f2f = torch.mm(proj_f, proj_f.t()) / self.temperature

        mask_self = torch.eye(n_f, device=proj_f.device, dtype=torch.bool)
        sim_f2f_masked = sim_f2f.masked_fill(mask_self, float('-inf'))

        # Adaptive Weighting
        score_diff = scores_m.unsqueeze(0) - scores_f.unsqueeze(1)
        score_diff_normalized = torch.sigmoid(score_diff * 5)
        weights = 0.5 + score_diff_normalized  # [0.5, 1.5]

        sim_f2m_weighted = sim_f2m + self.score_weight_alpha * torch.log(weights + 1e-8)

        # InfoNCE Loss
        all_sims = torch.cat([sim_f2m_weighted, sim_f2f_masked], dim=1)
        numerator = torch.logsumexp(sim_f2m_weighted, dim=1)
        denominator = torch.logsumexp(all_sims, dim=1)
        loss_f2m = -(numerator - denominator).mean()

        # Male -> Female (약하게)
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

        # 비대칭 결합: F->M 강하게 (1.5), M->F 약하게 (0.5)
        loss = 1.5 * loss_f2m + 0.5 * loss_m2f

        score_gap = (scores_m.mean() - scores_f.mean()).item()
        info = {
            "n_f": n_f,
            "n_m": n_m,
            "score_f_mean": scores_f.mean().item(),
            "score_m_mean": scores_m.mean().item(),
            "score_gap": score_gap,
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


def _wasserstein_1d_asymmetric(female_scores: torch.Tensor, male_scores: torch.Tensor) -> torch.Tensor:
    """단방향 Wasserstein: 여성 score -> 남성 score"""
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


def _split_outputs_by_gender(outputs: dict, indices: list) -> dict:
    """DETR outputs를 gender indices로 분리 (aux_outputs 포함)"""
    split = {
        "pred_logits": outputs["pred_logits"][indices],
        "pred_boxes": outputs["pred_boxes"][indices],
    }
    if "aux_outputs" in outputs:
        split["aux_outputs"] = [
            {
                "pred_logits": ao["pred_logits"][indices],
                "pred_boxes": ao["pred_boxes"][indices],
            }
            for ao in outputs["aux_outputs"]
        ]
    return split


def _default_output_dir(script_path: Path) -> str:
    stem = script_path.stem
    for prefix in ("train_faap_simclr_", "train_faap_", "train_"):
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix):]
            break
    return str(Path("faap_outputs") / f"faap_outputs_{stem.lower()}")


def _get_lr_scheduler(optimizer, args):
    """Cosine LR scheduler with warmup support"""
    if args.lr_scheduler == "cosine":
        t_max = max(args.epochs - args.warmup_epochs, 1)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=1e-6
        )
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FAAP Asymmetric Detection Loss (8th)")

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
    parser.add_argument("--seed", type=int, default=42)

    # Perturbation
    parser.add_argument("--epsilon", type=float, default=0.10)

    # Loss weights (8th: Asymmetric Detection)
    parser.add_argument("--lambda_contrastive", type=float, default=1.0)
    parser.add_argument("--lambda_wass", type=float, default=0.2)
    parser.add_argument("--lambda_det_f", type=float, default=0.5,
                        help="Female detection loss weight (boost Female AP)")
    parser.add_argument("--lambda_det_m", type=float, default=0.1,
                        help="Male detection loss weight (suppress Male AP rise)")

    # Contrastive settings
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--score_weight_alpha", type=float, default=1.0,
                        help="Score difference weighting strength")
    parser.add_argument("--score_top_k", type=int, default=10)
    parser.add_argument("--proj_dim", type=int, default=128)

    # LR Scheduler (8th: Cosine decay)
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["cosine", "none"])
    parser.add_argument("--warmup_epochs", type=int, default=2)

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
        print("Asymmetric Detection Loss (8th Version)")
        print("=" * 70)
        print("[3rd~7th 실패 원인] Detection loss가 Male/Female AP를 동시에 올림")
        print("[8th 핵심 변경] Gender별 비대칭 Detection Loss")
        print(f"  - Female det loss weight: {args.lambda_det_f} (정상 - Female 검출 향상)")
        print(f"  - Male det loss weight:   {args.lambda_det_m} (약화 - Male AP 상승 억제)")
        print("-" * 70)
        print(f"Contrastive: {args.lambda_contrastive} (Adaptive [0.5, 1.5])")
        print(f"Wasserstein: {args.lambda_wass}")
        print(f"Temperature: {args.temperature}")
        print(f"Epsilon: {args.epsilon} (fixed)")
        print(f"LR: {args.lr_g}, Scheduler: {args.lr_scheduler}, Warmup: {args.warmup_epochs} epochs")
        print(f"Batch size: {args.batch_size}")
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

    simclr_aug = SimCLRAugmentation(strength=args.aug_strength).to(device)

    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)
        proj_head = DDP(proj_head, device_ids=[args.gpu] if args.gpu is not None else None)

    params = list(_unwrap_ddp(generator).parameters()) + list(_unwrap_ddp(proj_head).parameters())
    opt_g = torch.optim.AdamW(params, lr=args.lr_g, weight_decay=0.01)

    # LR Scheduler
    scheduler = _get_lr_scheduler(opt_g, args)

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
        if "scheduler" in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
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

        current_lr = opt_g.param_groups[0]["lr"]

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
            # 1. Gender-Aware Contrastive Loss (3rd 검증, 변경 없음)
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
            # 2. Score-Level Wasserstein (보조, 3rd 동일)
            # =================================================================
            loss_wasserstein = _wasserstein_1d_asymmetric(scores_f, scores_m)

            # =================================================================
            # 3. Asymmetric Detection Loss (8th 핵심 변경)
            # =================================================================
            # Split DETR outputs by gender (aux_outputs 포함!)
            outputs_f = _split_outputs_by_gender(outputs, female_idx)
            outputs_m = _split_outputs_by_gender(outputs, male_idx)
            targets_f = [targets[i] for i in female_idx]
            targets_m = [targets[i] for i in male_idx]

            # Female detection loss (정상 적용 - Female 검출 향상)
            if len(female_idx) > 0:
                loss_det_f, _ = detr.detection_loss(outputs_f, targets_f)
            else:
                loss_det_f = torch.tensor(0.0, device=device)

            # Male detection loss (약화 적용 - Male AP 상승 억제)
            if len(male_idx) > 0:
                loss_det_m, _ = detr.detection_loss(outputs_m, targets_m)
            else:
                loss_det_m = torch.tensor(0.0, device=device)

            # =================================================================
            # Total Loss
            # =================================================================
            total_g = (
                args.lambda_contrastive * loss_contrastive
                + args.lambda_wass * loss_wasserstein
                + args.lambda_det_f * loss_det_f
                + args.lambda_det_m * loss_det_m
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
                loss_det_f=loss_det_f.item(),
                loss_det_m=loss_det_m.item(),
                total_g=total_g.item(),
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                score_f=contrastive_info.get("score_f_mean", 0.0),
                score_m=contrastive_info.get("score_m_mean", 0.0),
                score_gap=contrastive_info.get("score_gap", 0.0),
                n_f=contrastive_info.get("n_f", 0),
                n_m=contrastive_info.get("n_m", 0),
            )

        # =====================================================================
        # End of Epoch
        # =====================================================================
        metrics_logger.synchronize_between_processes()

        # LR Scheduler step (after warmup)
        if scheduler is not None and epoch >= args.warmup_epochs:
            scheduler.step()

        current_lr = opt_g.param_groups[0]["lr"]

        if utils.is_main_process():
            log_entry = {
                "epoch": epoch,
                "loss_contrastive": metrics_logger.meters["loss_contrastive"].global_avg,
                "loss_wasserstein": metrics_logger.meters["loss_wasserstein"].global_avg,
                "loss_det_f": metrics_logger.meters["loss_det_f"].global_avg,
                "loss_det_m": metrics_logger.meters["loss_det_m"].global_avg,
                "total_g": metrics_logger.meters["total_g"].global_avg,
                "lr": current_lr,
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "score_f": metrics_logger.meters["score_f"].global_avg,
                "score_m": metrics_logger.meters["score_m"].global_avg,
                "score_gap": metrics_logger.meters["score_gap"].global_avg,
                "n_f_avg": metrics_logger.meters["n_f"].global_avg,
                "n_m_avg": metrics_logger.meters["n_m"].global_avg,
            }

            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(f"\n[Epoch {epoch}] Summary:")
            print(f"  Contrastive Loss: {log_entry['loss_contrastive']:.4f}")
            print(f"  Wasserstein Loss: {log_entry['loss_wasserstein']:.4f}")
            print(f"  Det Loss (F): {log_entry['loss_det_f']:.4f}  (weight: {args.lambda_det_f})")
            print(f"  Det Loss (M): {log_entry['loss_det_m']:.4f}  (weight: {args.lambda_det_m})")
            print(f"  Total: {log_entry['total_g']:.4f}")
            print(f"  Score (F/M): {log_entry['score_f']:.4f} / {log_entry['score_m']:.4f}")
            print(f"  Score Gap (M-F): {log_entry['score_gap']:.4f}")
            print(f"  Samples (F/M): {log_entry['n_f_avg']:.1f} / {log_entry['n_m_avg']:.1f}")
            print(f"  LR: {current_lr:.6f}")

            if (epoch + 1) % args.save_every == 0:
                ckpt_path_save = output_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
                save_dict = {
                    "epoch": epoch,
                    "generator": _unwrap_ddp(generator).state_dict(),
                    "proj_head": _unwrap_ddp(proj_head).state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "args": vars(args),
                }
                if scheduler is not None:
                    save_dict["scheduler"] = scheduler.state_dict()
                torch.save(save_dict, ckpt_path_save)
                print(f"  Saved: {ckpt_path_save}")

        if args.distributed:
            dist.barrier()

    # =========================================================================
    # Training Complete
    # =========================================================================
    if utils.is_main_process():
        print("\n" + "=" * 70)
        print("Asymmetric Detection Loss Training Complete! (8th Version)")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print(f"\n[8th 핵심 변경]")
        print(f"  - Gender별 비대칭 Detection Loss")
        print(f"  - Female det weight: {args.lambda_det_f}, Male det weight: {args.lambda_det_m}")
        print(f"  - Cosine LR decay: {args.lr_scheduler}, warmup: {args.warmup_epochs} epochs")
        print(f"\n성공 기준:")
        print(f"  - AP Gap < 0.100 (baseline 0.1063 대비 6%+ 감소)")
        print(f"  - Male AP >= 0.505 (baseline 유지)")
        print(f"  - Female AP >= 0.410 (baseline 0.404 대비 상승)")


if __name__ == "__main__":
    main()
