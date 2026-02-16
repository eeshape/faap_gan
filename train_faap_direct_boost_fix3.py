"""
FAAP Training - Direct Confidence Boosting (fix3)

=============================================================================
이전 접근법들의 근본적 한계:
=============================================================================
1. Representation-Performance Gap:
   - InfoNCE는 cosine similarity 최적화 (표현 공간)
   - Detection confidence는 absolute logit magnitude에 의존
   - 표현이 비슷해져도 detection score는 개선되지 않음

2. Information Bottleneck:
   - 100개 object query → mean pooling → 1개 vector
   - 공간 정보/per-object confidence 손실
   - Female의 낮은 confidence 정보가 평균에 묻힘

3. Detection Score에 직접 gradient 없음:
   - Contrastive loss: representation level (간접)
   - Female detection을 직접 높이는 신호 없음

=============================================================================
fix3: Direct Confidence Boosting
=============================================================================

[핵심 아이디어]
- Contrastive loss 제거
- Female detection confidence를 **직접** 높이는 loss
- Per-object confidence 사용 (mean pooling 우회)
- Hard sample mining: 낮은 confidence에 더 높은 가중치

[Loss 설계]
1. Direct Confidence Boosting Loss:
   - L_boost = -log(female_conf + eps) * weight
   - weight = softmax((threshold - conf) * beta)  # Hard sample mining

2. Gap Reduction Loss:
   - L_gap = max(0, male_conf - female_conf - margin)

3. Detection Preservation Loss:
   - L_det = DETR detection loss (기존)

[예상 개선]
- AP Gap: 0.08-0.09 (현재 0.104 대비 20-25% 개선)
- F_AP Δ: > 0.02 (현재 0.01 대비 2배)

=============================================================================
"""

import argparse
import json
from pathlib import Path
from typing import Tuple, Dict

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
# Direct Confidence Boosting Loss (핵심)
# =============================================================================

class DirectConfidenceBoostLoss(nn.Module):
    """
    Direct Confidence Boosting Loss

    핵심:
    - Female detection confidence를 직접 높임
    - Per-object confidence 사용 (mean pooling 우회)
    - Hard sample mining: 낮은 confidence에 더 높은 가중치

    이전 접근법 대비 장점:
    - Detection logit에 직접 gradient 전달
    - Representation bottleneck 우회
    - Per-object 수준에서 최적화
    """

    def __init__(
        self,
        target_conf: float = 0.7,       # 목표 confidence
        hard_mining_beta: float = 5.0,  # Hard sample mining 강도
        gap_margin: float = 0.0,        # Gap 허용 margin
        top_k: int = 10,                # Top-k objects 사용
    ):
        super().__init__()
        self.target_conf = target_conf
        self.hard_mining_beta = hard_mining_beta
        self.gap_margin = gap_margin
        self.top_k = top_k

    def _get_object_confidences(self, outputs: dict) -> torch.Tensor:
        """
        DETR outputs에서 per-object confidence 추출

        Returns:
            (batch, num_queries) confidence scores
        """
        # pred_logits: (batch, num_queries, num_classes+1)
        # 마지막 클래스는 "no object"
        probs = outputs["pred_logits"].softmax(dim=-1)
        # 실제 object classes에 대한 max confidence
        obj_conf = probs[..., :-1].max(dim=-1).values  # (batch, num_queries)
        return obj_conf

    def _get_topk_confidences(self, obj_conf: torch.Tensor) -> torch.Tensor:
        """
        Top-k confident objects 선택

        Args:
            obj_conf: (batch, num_queries)
        Returns:
            (batch, top_k) top-k confidence scores
        """
        k = min(self.top_k, obj_conf.size(1))
        topk_conf = obj_conf.topk(k, dim=1).values  # (batch, k)
        return topk_conf

    def forward(
        self,
        outputs_f: dict,
        outputs_m: dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            outputs_f: DETR outputs for female images
            outputs_m: DETR outputs for male images
        """
        # Per-object confidence
        conf_f_all = self._get_object_confidences(outputs_f)  # (N_f, 100)
        conf_m_all = self._get_object_confidences(outputs_m)  # (N_m, 100)

        # Top-k confidence
        conf_f = self._get_topk_confidences(conf_f_all)  # (N_f, k)
        conf_m = self._get_topk_confidences(conf_m_all)  # (N_m, k)

        n_f = conf_f.size(0)
        n_m = conf_m.size(0)

        if n_f == 0:
            return conf_f_all.new_tensor(0.0), {"loss_boost": 0.0, "loss_gap": 0.0}

        # =================================================================
        # 1. Direct Confidence Boosting Loss
        # =================================================================
        # 목표: Female confidence를 target_conf까지 높임
        # L_boost = -log(conf) * weight

        # Hard sample mining weight: 낮은 confidence에 더 높은 가중치
        with torch.no_grad():
            # (target - conf)가 클수록 weight 높음
            diff = (self.target_conf - conf_f).clamp(min=0)
            weight = F.softmax(diff * self.hard_mining_beta, dim=-1)  # (N_f, k)

        # Confidence boosting loss
        # -log(conf)는 conf가 낮을수록 큼
        loss_boost_per_obj = -torch.log(conf_f + 1e-8)  # (N_f, k)
        loss_boost = (loss_boost_per_obj * weight).sum(dim=-1).mean()  # scalar

        # =================================================================
        # 2. Gap Reduction Loss (Optional)
        # =================================================================
        # 목표: Female mean conf가 Male mean conf에 가까워지도록

        mean_conf_f = conf_f.mean()
        mean_conf_m = conf_m.mean().detach()  # Male은 고정

        # max(0, male - female - margin)
        gap = (mean_conf_m - mean_conf_f - self.gap_margin).clamp(min=0)
        loss_gap = gap

        # =================================================================
        # 3. Confidence Threshold Loss (추가)
        # =================================================================
        # 목표: Female의 low-confidence objects를 threshold 이상으로

        threshold = 0.3
        below_threshold = (conf_f < threshold).float()
        loss_threshold = (below_threshold * (threshold - conf_f).clamp(min=0)).mean()

        # =================================================================
        # Total Loss
        # =================================================================
        loss = loss_boost + 0.5 * loss_gap + 0.3 * loss_threshold

        info = {
            "loss_boost": loss_boost.item(),
            "loss_gap": loss_gap.item(),
            "loss_threshold": loss_threshold.item(),
            "conf_f_mean": mean_conf_f.item(),
            "conf_m_mean": mean_conf_m.item(),
            "conf_gap": (mean_conf_m - mean_conf_f).item(),
            "n_f": n_f,
            "n_m": n_m,
        }

        return loss, info


# =============================================================================
# Wasserstein Loss (Female → Male, per-object)
# =============================================================================

def _wasserstein_1d_per_object(conf_f: torch.Tensor, conf_m: torch.Tensor) -> torch.Tensor:
    """
    Per-object Wasserstein distance

    Args:
        conf_f: (N_f, k) Female top-k confidences
        conf_m: (N_m, k) Male top-k confidences
    """
    if conf_f.numel() == 0 or conf_m.numel() == 0:
        return conf_f.new_tensor(0.0)

    # Flatten and sort
    flat_f = conf_f.flatten().sort().values
    flat_m = conf_m.flatten().detach().sort().values

    # Interpolate to same size
    k = max(flat_f.numel(), flat_m.numel())

    if k != flat_f.numel():
        idx = torch.linspace(0, flat_f.numel() - 1, k, device=flat_f.device)
        idx_low, idx_high = idx.floor().long().clamp(max=flat_f.numel()-1), idx.ceil().long().clamp(max=flat_f.numel()-1)
        weight = idx - idx.floor()
        flat_f = flat_f[idx_low] * (1 - weight) + flat_f[idx_high] * weight

    if k != flat_m.numel():
        idx = torch.linspace(0, flat_m.numel() - 1, k, device=flat_m.device)
        idx_low, idx_high = idx.floor().long().clamp(max=flat_m.numel()-1), idx.ceil().long().clamp(max=flat_m.numel()-1)
        weight = idx - idx.floor()
        flat_m = flat_m[idx_low] * (1 - weight) + flat_m[idx_high] * weight

    # 단방향: Female < Male 일 때만 패널티
    return F.relu(flat_m - flat_f).mean()


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
    for prefix in ("train_faap_direct_boost_", "train_faap_", "train_"):
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix):]
            break
    return str(Path("faap_outputs") / f"faap_outputs_{stem.lower()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FAAP Direct Confidence Boosting (fix3)")

    # Paths
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()))
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(Path(__file__)))

    # Training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr_g", type=float, default=5e-5)  # 더 낮은 lr (직접 최적화)
    parser.add_argument("--seed", type=int, default=42)

    # Perturbation
    parser.add_argument("--epsilon", type=float, default=0.10)

    # Loss weights
    parser.add_argument("--lambda_boost", type=float, default=1.0)
    parser.add_argument("--lambda_wass", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.3)       # Detection loss weight (낮춤)
    parser.add_argument("--beta_final", type=float, default=0.5)

    # Direct Boost settings
    parser.add_argument("--target_conf", type=float, default=0.7)
    parser.add_argument("--hard_mining_beta", type=float, default=5.0)
    parser.add_argument("--gap_margin", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=10)

    # Scheduler
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    parser.add_argument("--warmup_epochs", type=int, default=2)

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


def _get_lr_scheduler(optimizer, args):
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6
        )
    elif args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        return None


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
        print("Direct Confidence Boosting (fix3)")
        print("=" * 70)
        print("[이전 접근법 한계]")
        print("  1. Contrastive loss: representation level (간접)")
        print("  2. Mean pooling: per-object info 손실")
        print("  3. Detection score에 직접 gradient 없음")
        print("-" * 70)
        print("[fix3 핵심 변경]")
        print("  1. Contrastive loss 제거")
        print("  2. Female confidence 직접 boosting")
        print("  3. Per-object confidence 사용")
        print("  4. Hard sample mining")
        print("-" * 70)
        print(f"Target confidence: {args.target_conf}")
        print(f"Hard mining beta: {args.hard_mining_beta}")
        print(f"Learning rate: {args.lr_g}")
        print(f"Epochs: {args.epochs}")
        print(f"Detection loss weight (beta): {args.beta} → {args.beta_final}")
        print("=" * 70)

    # ==========================================================================
    # Model Initialization
    # ==========================================================================

    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)

    boost_loss_fn = DirectConfidenceBoostLoss(
        target_conf=args.target_conf,
        hard_mining_beta=args.hard_mining_beta,
        gap_margin=args.gap_margin,
        top_k=args.top_k,
    ).to(device)

    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)

    opt_g = torch.optim.AdamW(_unwrap_ddp(generator).parameters(), lr=args.lr_g, weight_decay=0.01)
    scheduler = _get_lr_scheduler(opt_g, args)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if "generator" in ckpt:
            _unwrap_ddp(generator).load_state_dict(ckpt["generator"])
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

        if args.distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        current_beta = _scheduled_beta(epoch, args.epochs, args.beta, args.beta_final)
        current_lr = opt_g.param_groups[0]['lr']

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
            outputs = detr(perturbed)

            # Split outputs by gender
            outputs_f = {
                "pred_logits": outputs["pred_logits"][female_idx],
                "pred_boxes": outputs["pred_boxes"][female_idx],
            }
            outputs_m = {
                "pred_logits": outputs["pred_logits"][male_idx],
                "pred_boxes": outputs["pred_boxes"][male_idx],
            }

            # =================================================================
            # 1. Direct Confidence Boosting Loss (핵심)
            # =================================================================
            loss_boost, boost_info = boost_loss_fn(outputs_f, outputs_m)

            # =================================================================
            # 2. Wasserstein Loss (per-object)
            # =================================================================
            conf_f = outputs_f["pred_logits"].softmax(-1)[..., :-1].max(-1).values
            conf_m = outputs_m["pred_logits"].softmax(-1)[..., :-1].max(-1).values

            # Top-k for Wasserstein
            k = min(args.top_k, conf_f.size(1))
            conf_f_topk = conf_f.topk(k, dim=1).values
            conf_m_topk = conf_m.topk(k, dim=1).values

            loss_wasserstein = _wasserstein_1d_per_object(conf_f_topk, conf_m_topk)

            # =================================================================
            # 3. Detection Loss
            # =================================================================
            loss_det, _ = detr.detection_loss(outputs, targets)

            # =================================================================
            # Total Loss
            # =================================================================
            total_g = (
                args.lambda_boost * loss_boost
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

                # Image-level scores for logging
                scores = _image_level_detection_score(outputs, top_k=args.top_k)
                score_f = scores[female_idx].mean().item() if female_idx else 0.0
                score_m = scores[male_idx].mean().item() if male_idx else 0.0

            # =================================================================
            # Backward & Optimize
            # =================================================================
            total_g.backward()
            if args.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), args.max_norm)
            opt_g.step()

            # Log
            metrics_logger.update(
                loss_boost=loss_boost.item(),
                loss_wasserstein=loss_wasserstein.item(),
                loss_det=loss_det.item(),
                total_g=total_g.item(),
                beta=current_beta,
                lr=current_lr,
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                score_f=score_f,
                score_m=score_m,
                score_gap=score_m - score_f,
                conf_f=boost_info.get("conf_f_mean", 0.0),
                conf_m=boost_info.get("conf_m_mean", 0.0),
                conf_gap=boost_info.get("conf_gap", 0.0),
                n_f=boost_info.get("n_f", 0),
                n_m=boost_info.get("n_m", 0),
            )

        # =====================================================================
        # End of Epoch
        # =====================================================================
        metrics_logger.synchronize_between_processes()

        # Update scheduler
        if scheduler is not None and epoch >= args.warmup_epochs:
            scheduler.step()

        if utils.is_main_process():
            log_entry = {
                "epoch": epoch,
                "loss_boost": metrics_logger.meters["loss_boost"].global_avg,
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
                "conf_f": metrics_logger.meters["conf_f"].global_avg,
                "conf_m": metrics_logger.meters["conf_m"].global_avg,
                "conf_gap": metrics_logger.meters["conf_gap"].global_avg,
                "n_f_avg": metrics_logger.meters["n_f"].global_avg,
                "n_m_avg": metrics_logger.meters["n_m"].global_avg,
            }

            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(f"\n[Epoch {epoch}] Summary:")
            print(f"  Boost Loss: {log_entry['loss_boost']:.4f}")
            print(f"  Wasserstein Loss: {log_entry['loss_wasserstein']:.4f}")
            print(f"  Detection Loss: {log_entry['loss_det']:.4f}")
            print(f"  Total: {log_entry['total_g']:.4f}")
            print(f"  Conf (F/M): {log_entry['conf_f']:.4f} / {log_entry['conf_m']:.4f}")
            print(f"  Conf Gap (M-F): {log_entry['conf_gap']:.4f}")
            print(f"  Score (F/M): {log_entry['score_f']:.4f} / {log_entry['score_m']:.4f}")
            print(f"  Score Gap (M-F): {log_entry['score_gap']:.4f}")
            print(f"  Beta: {current_beta:.4f}, LR: {current_lr:.6f}")

            if (epoch + 1) % args.save_every == 0:
                ckpt_path_save = output_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "generator": _unwrap_ddp(generator).state_dict(),
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
        print("Direct Confidence Boosting Training Complete!")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print("\n[fix3 핵심 특징]")
        print("  - Contrastive loss 제거 (간접 → 직접)")
        print("  - Female confidence 직접 boosting")
        print("  - Per-object confidence 사용")
        print("  - Hard sample mining")
        print("\n예상 개선:")
        print("  - AP Gap: 0.08-0.09 (현재 0.104 대비 20-25% 개선)")
        print("  - F_AP Δ: > 0.02")


if __name__ == "__main__":
    main()
