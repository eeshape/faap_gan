"""
FAAP Training - Asymmetric Contrastive with Male-Anchored Learning (3rd_fix2)

=============================================================================
3rd_fix1 실패 분석:
=============================================================================
1. Loss Saturation (포화):
   - loss_f_align, loss_m_align이 -10에 즉시 수렴
   - Fair Centroid와의 similarity가 1.0에 도달 → gradient ≈ 0
   - 학습이 전혀 진행되지 않음

2. Representation Collapse:
   - 모든 샘플이 같은 점으로 수렴
   - Contrastive learning 실패

3. 음수 Loss:
   - total_loss = -24 (음수)
   - 최소화가 의미 없음

=============================================================================
3rd_fix2 핵심 아이디어: Male-Anchored Asymmetric Contrastive
=============================================================================

[핵심 변경]
- 3rd_fix1: Fair Centroid → 포화 문제
- 3rd_fix2: Male을 Anchor(고정)로 사용, Female만 학습

[왜 Male을 Anchor로?]
- Male AP (0.51) > Female AP (0.40)
- Male의 representation이 더 discriminative
- Female을 Male 방향으로 당기면 Female 성능 향상
- Male은 detach하여 변하지 않게 함

[Loss 설계]
1. Female → Male Contrastive (핵심):
   - Anchor: Female
   - Positive: Male (detached)
   - Negative: Other Females
   - Male gradient 차단 → Female만 학습

2. Score-Weighted Sampling:
   - 저성능 Female → 고성능 Male 쌍에 더 높은 가중치
   - Hard negative mining 효과

3. 비대칭 가중치 복원:
   - 3rd: F→M (1.5), M→F (0.5) 성공
   - 3rd_fix1: 대칭 (1.0:1.0) 실패
   - 3rd_fix2: F→M만 사용 (완전 비대칭)

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
# Projection Head (3rd 버전과 동일)
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
# Male-Anchored Asymmetric Contrastive Loss (핵심 변경)
# =============================================================================

class MaleAnchoredContrastiveLoss(nn.Module):
    """
    Male-Anchored Asymmetric Contrastive Loss

    핵심:
    - Male representation을 anchor(고정)로 사용
    - Female만 학습하여 Male 방향으로 이동
    - Male은 detach하여 gradient 차단

    3rd_fix1 실패 원인 해결:
    - Fair Centroid 제거 (포화 문제)
    - 직접적인 F→M contrastive 사용 (3rd 성공 방식)
    - Male detach로 Female만 학습
    """

    def __init__(
        self,
        temperature: float = 0.07,  # 3rd와 동일
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
            return proj_f.new_tensor(0.0), {"n_f": n_f, "n_m": n_m, "loss_f2m": 0.0}

        scores_f = scores_f.detach()
        scores_m = scores_m.detach()

        # =================================================================
        # 핵심: Male을 Detach하여 Anchor로 사용
        # =================================================================
        # Male representation 고정 → Female만 학습
        proj_m_detached = proj_m.detach()

        # =================================================================
        # 1. Female → Male Contrastive (핵심)
        # =================================================================
        # Anchor: Female
        # Positive: Male (detached)
        # Negative: Other Females

        sim_f2m = torch.mm(proj_f, proj_m_detached.t()) / self.temperature  # (N_f, N_m)
        sim_f2f = torch.mm(proj_f, proj_f.t()) / self.temperature  # (N_f, N_f)

        # 자기 자신 마스킹
        mask_self = torch.eye(n_f, device=proj_f.device, dtype=torch.bool)
        sim_f2f_masked = sim_f2f.masked_fill(mask_self, float('-inf'))

        # =================================================================
        # 2. Score-Weighted Positive Selection
        # =================================================================
        # 고성능 Male에 더 높은 가중치
        # score_weight[j] = softmax(scores_m)[j]
        score_weight_m = F.softmax(scores_m * 5, dim=0)  # temperature=0.2 for sharper distribution

        # Weighted similarity to males
        # sim_f2m_weighted[i] = sum_j(score_weight[j] * sim_f2m[i,j])
        sim_f2m_weighted = torch.mm(sim_f2m.exp(), score_weight_m.unsqueeze(1)).squeeze(1)
        sim_f2m_weighted = torch.log(sim_f2m_weighted + 1e-8)

        # =================================================================
        # 3. InfoNCE Loss (Female → Male)
        # =================================================================
        # Numerator: Female이 (weighted) Male과 가까워지도록
        # Denominator: Female이 다른 Female과는 멀어지도록

        # All negatives: other females
        all_sims = torch.cat([sim_f2m, sim_f2f_masked], dim=1)  # (N_f, N_m + N_f)

        # Numerator: similarity to males (weighted)
        numerator = torch.logsumexp(sim_f2m, dim=1)  # (N_f,)

        # Denominator: all similarities
        denominator = torch.logsumexp(all_sims, dim=1)  # (N_f,)

        # InfoNCE loss
        loss_f2m = -(numerator - denominator).mean()

        # =================================================================
        # 4. Score Gap Penalty (추가)
        # =================================================================
        # Female score가 Male score보다 낮으면 패널티
        score_gap = (scores_m.mean() - scores_f.mean()).clamp(min=0)
        loss_score_gap = score_gap * self.score_weight_alpha

        # =================================================================
        # Total Loss
        # =================================================================
        loss = loss_f2m + 0.5 * loss_score_gap

        # Info
        info = {
            "n_f": n_f,
            "n_m": n_m,
            "score_f_mean": scores_f.mean().item(),
            "score_m_mean": scores_m.mean().item(),
            "score_gap": (scores_m.mean() - scores_f.mean()).item(),
            "loss_f2m": loss_f2m.item(),
            "loss_score_gap": loss_score_gap.item(),
        }

        return loss, info


# =============================================================================
# Wasserstein Loss (단방향: Female → Male)
# =============================================================================

def _wasserstein_1d_asymmetric(female_scores: torch.Tensor, male_scores: torch.Tensor) -> torch.Tensor:
    """단방향 Wasserstein: Female score를 Male 방향으로만"""
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0)
    sorted_f = female_scores.sort().values
    sorted_m = male_scores.detach().sort().values  # Male detach
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

    # 단방향: Female < Male 일 때만 패널티
    return F.relu(sorted_m - sorted_f).mean()


def _image_level_detection_score(outputs: dict, top_k: int = 10) -> torch.Tensor:
    """DETR logits에서 이미지 단위 score 계산"""
    probs = outputs["pred_logits"].softmax(dim=-1)[..., :-1]
    max_probs = probs.max(dim=-1).values
    if top_k > 0 and top_k < max_probs.size(1):
        topk_probs = max_probs.topk(top_k, dim=1).values
        return topk_probs.mean(dim=1)
    return max_probs.mean(dim=1)


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
    parser = argparse.ArgumentParser("FAAP Male-Anchored Contrastive (3rd_fix2)")

    # Paths
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()))
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(Path(__file__)))

    # Training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=10)  # 더 짧게 (overfitting 방지)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr_g", type=float, default=1e-4)  # 3rd와 동일
    parser.add_argument("--seed", type=int, default=42)

    # Perturbation
    parser.add_argument("--epsilon", type=float, default=0.10)

    # Loss weights
    parser.add_argument("--lambda_contrastive", type=float, default=1.0)
    parser.add_argument("--lambda_wass", type=float, default=0.2)  # 3rd와 동일
    parser.add_argument("--beta", type=float, default=0.5)  # 3rd와 동일
    parser.add_argument("--beta_final", type=float, default=0.6)

    # Contrastive settings
    parser.add_argument("--temperature", type=float, default=0.07)  # 3rd와 동일
    parser.add_argument("--score_weight_alpha", type=float, default=1.0)
    parser.add_argument("--score_top_k", type=int, default=10)
    parser.add_argument("--proj_dim", type=int, default=128)

    # Augmentation
    parser.add_argument("--aug_strength", type=str, default="medium",  # 3rd와 동일
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
        print("Male-Anchored Asymmetric Contrastive Learning (3rd_fix2)")
        print("=" * 70)
        print("[3rd_fix1 실패 원인]")
        print("  1. Fair Centroid alignment 포화 (loss = -10)")
        print("  2. Representation collapse")
        print("  3. 음수 loss → 학습 안됨")
        print("-" * 70)
        print("[3rd_fix2 핵심 변경]")
        print("  1. Fair Centroid 제거")
        print("  2. Male을 Anchor(고정)로 사용")
        print("  3. Female만 학습 (Male detach)")
        print("  4. 3rd 버전 hyperparameter 복원")
        print("-" * 70)
        print(f"Temperature: {args.temperature} (3rd와 동일)")
        print(f"Learning rate: {args.lr_g} (3rd와 동일)")
        print(f"Epochs: {args.epochs}")
        print(f"Wasserstein weight: {args.lambda_wass}")
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

    contrastive_loss_fn = MaleAnchoredContrastiveLoss(
        temperature=args.temperature,
        score_weight_alpha=args.score_weight_alpha,
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
            # 1. Male-Anchored Contrastive Loss (핵심)
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
            # 2. Asymmetric Wasserstein (Female → Male)
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
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                score_f=contrastive_info.get("score_f_mean", 0.0),
                score_m=contrastive_info.get("score_m_mean", 0.0),
                score_gap=contrastive_info.get("score_gap", 0.0),
                loss_f2m=contrastive_info.get("loss_f2m", 0.0),
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
                "loss_f2m": metrics_logger.meters["loss_f2m"].global_avg,
                "n_f_avg": metrics_logger.meters["n_f"].global_avg,
                "n_m_avg": metrics_logger.meters["n_m"].global_avg,
            }

            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(f"\n[Epoch {epoch}] Summary:")
            print(f"  Contrastive Loss: {log_entry['loss_contrastive']:.4f}")
            print(f"  F→M Loss: {log_entry['loss_f2m']:.4f}")
            print(f"  Wasserstein Loss: {log_entry['loss_wasserstein']:.4f}")
            print(f"  Detection Loss: {log_entry['loss_det']:.4f}")
            print(f"  Total: {log_entry['total_g']:.4f}")
            print(f"  Score (F/M): {log_entry['score_f']:.4f} / {log_entry['score_m']:.4f}")
            print(f"  Score Gap (M-F): {log_entry['score_gap']:.4f}")
            print(f"  Beta: {current_beta:.4f}")

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
        print("Male-Anchored Contrastive Learning Complete!")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print("\n[3rd_fix1 → 3rd_fix2 핵심 변경]")
        print("  - Fair Centroid 제거 (포화 문제)")
        print("  - Male을 Anchor로 고정 (detach)")
        print("  - Female만 학습 (비대칭)")
        print("  - 3rd 버전 hyperparameter 복원")
        print("\n성공 기준:")
        print("  - AP Gap < 0.104 (3rd Epoch 3 수준)")
        print("  - Loss가 양수 (정상 학습)")


if __name__ == "__main__":
    main()
