"""
FAAP Training - Fix11 Contrastive + L2 Anchoring [2026-04-10]

=============================================================================
Loss 구조 (fix11_loss_analysis.html Case 1 기반):
=============================================================================

Total Loss = lambda_con * L_contrastive
           + beta     * L_det_female
           + beta_m   * L_det_male
           + gamma    * L_L2_anchor

(1) L_contrastive: Score-Weighted Contrastive Loss
    - Anchor: Female, Positive: Male, Negative: other Females
    - .detach() 제거 → gradient 양쪽 흐름
    - detection score로 가중치 부여

(2) L_det_female: 여성 검출 보존 (DETR criterion)
(3) L_det_male:   남성 검출 보존 (DETR criterion)

(4) L_L2_anchor: Male Feature L2 Anchoring (NEW)
    - clean male feature vs perturbed male feature
    - F.mse_loss(z_pert_male, z_clean_male)
    - 남성 feature를 원래 위치에 고정 ("못 박기")
    - Contrastive가 양쪽으로 당기지만, L2가 남성을 고정
      → 결과적으로 여성만 끌려옴

=============================================================================
기존 fix11 대비 변경:
=============================================================================
- .detach() 제거 (교수님 요구)
- 대신 L2 anchoring loss 추가로 남성 feature 고정
- gamma = 0.5 ~ 1.0 추천 (fix11_loss_analysis.html 참조)

=============================================================================
참고 파일:
=============================================================================
- fix11_loss_analysis.html: Loss 구조 분석 (Case 1: L2)
- train_faap_simclr_infonce_3rd_fix5_gpu_20260305.py: 학습 루프 구조
- train_faap_simclr_infonce_3rd_fix6_gpu_20260306.py: BF16/TF32 설정
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
    # Allow absolute imports like "from faap_gan..." when run as a script.
    __package__ = "faap_gan"

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from faap_gan.datasets import build_faap_dataloader, inspect_faap_dataset
from faap_gan.models import FrozenDETR, PerturbationGenerator, clamp_normalized
from faap_gan.path_utils import DETR_REPO, default_detr_checkpoint, ensure_detr_repo_on_path

# util.misc lives in the DETR repo; register its path before importing util.*
ensure_detr_repo_on_path(DETR_REPO)

import util.misc as utils
from util.misc import NestedTensor


# =============================================================================
# Projection Head
# =============================================================================

class ProjectionHead(nn.Module):
    """DETR decoder features -> contrastive embedding space"""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256,
                 output_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_queries, feature_dim)
        pooled = x.mean(dim=1)  # (batch, feature_dim)
        proj = self.net(pooled)
        return F.normalize(proj, dim=-1, p=2)


# =============================================================================
# Score-Weighted Contrastive Loss (.detach() 제거 버전)
# =============================================================================

class ScoreWeightedContrastiveLoss(nn.Module):
    """
    Fix11 Score-Weighted Contrastive Loss

    Anchor: Female, Positive: Male, Negative: other Females
    .detach() 제거 → gradient가 male 쪽으로도 흐름
    → L2 anchoring loss가 male을 고정하는 역할

    w(i,j) = softmax(score_j) : 높은 detection score를 가진 male에 더 강하게 당김
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        proj_f: torch.Tensor,   # (N_f, D) Female projections, L2-normalized
        proj_m: torch.Tensor,   # (N_m, D) Male projections, L2-normalized (NO detach)
        scores_f: torch.Tensor, # (N_f,) Female detection scores
        scores_m: torch.Tensor, # (N_m,) Male detection scores
    ) -> tuple:
        n_f = proj_f.size(0)
        n_m = proj_m.size(0)

        if n_f < 1 or n_m < 1:
            return proj_f.new_tensor(0.0), {"n_f": n_f, "n_m": n_m}

        # Score-based weights for male targets (높은 score → 더 강하게 당김)
        weights_m = torch.softmax(scores_m.detach() / 0.1, dim=0)  # (N_m,)

        # Female(anchor) vs Male(positive) similarity
        sim_f2m = torch.mm(proj_f, proj_m.t()) / self.temperature  # (N_f, N_m)

        # Female(anchor) vs Female(negative) similarity
        sim_f2f = torch.mm(proj_f, proj_f.t()) / self.temperature  # (N_f, N_f)
        mask_self = torch.eye(n_f, device=proj_f.device, dtype=torch.bool)
        sim_f2f = sim_f2f.masked_fill(mask_self, float('-inf'))

        # Weighted numerator: sum of score-weighted positive similarities
        # weights_m: (N_m,) -> broadcast to (N_f, N_m)
        weighted_pos = (sim_f2m * weights_m.unsqueeze(0)).sum(dim=1)  # (N_f,)

        # Denominator: log-sum-exp over all (positives + negatives)
        all_sims = torch.cat([sim_f2m, sim_f2f], dim=1)  # (N_f, N_m + N_f)
        log_denom = torch.logsumexp(all_sims, dim=1)  # (N_f,)

        # InfoNCE-style loss
        loss = -(weighted_pos - log_denom).mean()

        info = {
            "n_f": n_f,
            "n_m": n_m,
            "score_f_mean": scores_f.detach().mean().item(),
            "score_m_mean": scores_m.detach().mean().item(),
            "score_gap": (scores_m.detach().mean() - scores_f.detach().mean()).item(),
            "sim_f2m_mean": sim_f2m.detach().mean().item(),
        }
        return loss, info


# =============================================================================
# Matched Detection Scores (Hungarian matching 기반)
# =============================================================================

def _matched_detection_scores(
    detr: FrozenDETR,
    outputs: dict,
    targets: Sequence[dict],
) -> torch.Tensor:
    """Hungarian matching으로 TP prediction의 confidence score 추출."""
    if len(targets) == 0:
        return outputs["pred_logits"].new_zeros(0)

    # matcher 내부 torch.cdist는 CUDA BF16을 지원하지 않으므로 FP32로 계산.
    matcher_outputs = {
        "pred_logits": outputs["pred_logits"].float(),
        "pred_boxes": outputs["pred_boxes"].float(),
    }
    indices = detr.criterion.matcher(matcher_outputs, targets)
    probs = matcher_outputs["pred_logits"].softmax(dim=-1)
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
    """DETR logits에서 이미지 단위 score 계산 (fallback)"""
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
    parser = argparse.ArgumentParser(
        "FAAP fix11: Contrastive + L2 Anchoring (no .detach(), gamma-anchored male)"
    )

    # Paths
    parser.add_argument("--dataset_root", type=str, default="/workspace/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()))
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(Path(__file__)))

    # Training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr_g", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)

    # Perturbation (fix11: 낮은 epsilon)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--epsilon_final", type=float, default=0.10)
    parser.add_argument("--epsilon_warmup_epochs", type=int, default=5)
    parser.add_argument("--epsilon_hold_epochs", type=int, default=4)
    parser.add_argument("--epsilon_cooldown_epochs", type=int, default=6)
    parser.add_argument("--epsilon_min", type=float, default=0.09)

    # Loss weights
    parser.add_argument("--lambda_con", type=float, default=1.0,
                        help="Contrastive loss weight")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Female detection loss weight (start)")
    parser.add_argument("--beta_final", type=float, default=0.6,
                        help="Female detection loss weight (end)")
    parser.add_argument("--beta_m", type=float, default=0.5,
                        help="Male detection loss weight (fixed)")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="L2 anchoring loss weight (0.5~1.0 추천)")

    # Contrastive settings
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--proj_dropout", type=float, default=0.1)
    parser.add_argument("--score_top_k", type=int, default=10)

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
    """3단계 epsilon 스케줄: warmup -> hold -> cooldown"""
    warmup_end = 0 if warmup_epochs <= 1 else warmup_epochs - 1
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


def _scheduled_beta(epoch: int, total_epochs: int,
                    beta_start: float, beta_final: float) -> float:
    """beta 선형 증가"""
    if total_epochs <= 1 or beta_start == beta_final:
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

    # A100 BF16/TF32 최적화
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Setup
    detr_repo = ensure_detr_repo_on_path(Path(args.detr_repo))
    ckpt_path = Path(args.detr_checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = detr_repo / ckpt_path

    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA.")
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
        print("FAAP fix11: Contrastive + L2 Anchoring")
        print("=" * 70)
        print("[Loss 구조]")
        print(f"  (1) lambda_con={args.lambda_con} * L_contrastive  (Score-Weighted, no .detach())")
        print(f"  (2) beta={args.beta}->{args.beta_final} * L_det_female")
        print(f"  (3) beta_m={args.beta_m} * L_det_male")
        print(f"  (4) gamma={args.gamma} * L_L2_anchor  (Male Feature L2 Anchoring)")
        print("-" * 70)
        print("[설계 원리]")
        print("  Contrastive: 여성과 남성을 당기는 줄 (gradient 양쪽)")
        print("  L2 Anchor:   남성 발에 박힌 못 (남성 고정)")
        print("  -> 줄로 당기면 못 박힌 쪽은 가만히 -> 여성만 끌려옴")
        print("-" * 70)
        print(f"  Temperature: {args.temperature}")
        print(f"  Epsilon: {args.epsilon} -> {args.epsilon_final} -> {args.epsilon_min}")
        print(f"  LR: {args.lr_g}, Batch: {args.batch_size}")
        print("=" * 70)

    # =========================================================================
    # Model Initialization
    # =========================================================================

    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)

    proj_head = ProjectionHead(
        input_dim=detr.hidden_dim,
        hidden_dim=detr.hidden_dim,
        output_dim=args.proj_dim,
        dropout=args.proj_dropout,
    ).to(device)

    contrastive_loss_fn = ScoreWeightedContrastiveLoss(
        temperature=args.temperature,
    ).to(device)

    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)
        proj_head = DDP(proj_head, device_ids=[args.gpu] if args.gpu is not None else None)

    params = list(_unwrap_ddp(generator).parameters()) + list(_unwrap_ddp(proj_head).parameters())
    opt_g = torch.optim.AdamW(params, lr=args.lr_g, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_g, T_max=args.epochs, eta_min=args.lr_g * 0.1
    )

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
        if "scheduler" in ckpt:
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

    # =========================================================================
    # Training Loop
    # =========================================================================

    for epoch in range(start_epoch, args.epochs):
        metrics_logger = utils.MetricLogger(delimiter="  ")
        generator.train()
        proj_head.train()

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
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, '_last_lr') else args.lr_g

        for samples, targets, genders in metrics_logger.log_every(
            train_loader, args.log_every, f"Epoch {epoch}"
        ):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            genders = [g.lower() for g in genders]

            female_idx = [i for i, g in enumerate(genders) if g == "female"]
            male_idx = [i for i, g in enumerate(genders) if g == "male"]

            if len(female_idx) < 1 or len(male_idx) < 1:
                continue

            # =================================================================
            # Forward Pass (BF16 AMP)
            # =================================================================
            opt_g.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

                # ----- Clean features (고정 target, no_grad) -----
                with torch.no_grad():
                    _, feat_clean = detr.forward_with_features(samples)
                    z_clean = proj_head(feat_clean)  # (B, D), L2-normalized

                # ----- Perturbed features -----
                perturbed = _apply_generator(generator, samples)
                outputs, feat_pert = detr.forward_with_features(perturbed)
                z_pert = proj_head(feat_pert)  # (B, D), L2-normalized

                # =============================================================
                # (1) Score-Weighted Contrastive Loss
                # =============================================================
                image_scores = _image_level_detection_score(outputs, top_k=args.score_top_k)

                # NOTE: proj_m에 .detach() 없음 → gradient 양쪽 흐름
                proj_f = z_pert[female_idx]
                proj_m = z_pert[male_idx]
                scores_f = image_scores[female_idx]
                scores_m = image_scores[male_idx]

                loss_contrastive, con_info = contrastive_loss_fn(
                    proj_f, proj_m, scores_f, scores_m
                )

                # =============================================================
                # (2) Female Detection Loss
                # =============================================================
                outputs_f = {
                    "pred_logits": outputs["pred_logits"][female_idx],
                    "pred_boxes": outputs["pred_boxes"][female_idx],
                }
                targets_f = [targets[i] for i in female_idx]
                loss_det_f, _ = detr.detection_loss(outputs_f, targets_f)

                # =============================================================
                # (3) Male Detection Loss
                # =============================================================
                outputs_m = {
                    "pred_logits": outputs["pred_logits"][male_idx],
                    "pred_boxes": outputs["pred_boxes"][male_idx],
                }
                targets_m = [targets[i] for i in male_idx]
                loss_det_m, _ = detr.detection_loss(outputs_m, targets_m)

                # =============================================================
                # (4) L2 Anchoring Loss (남성 feature 고정)
                # =============================================================
                # z_pert_male: gradient O (Generator를 통해 학습)
                # z_clean_male: gradient X (no_grad block에서 계산)
                z_pert_male = z_pert[male_idx]
                z_clean_male = z_clean[male_idx]
                loss_l2_anchor = F.mse_loss(z_pert_male, z_clean_male)

                # =============================================================
                # Total Loss
                # =============================================================
                total_g = (
                    args.lambda_con * loss_contrastive
                    + current_beta * loss_det_f
                    + args.beta_m * loss_det_m
                    + args.gamma * loss_l2_anchor
                )

            # =================================================================
            # Metrics (autocast 밖, no_grad)
            # =================================================================
            with torch.no_grad():
                delta = perturbed.tensors - samples.tensors
                delta_linf = delta.abs().amax(dim=(1, 2, 3)).mean()
                delta_l2 = delta.flatten(1).norm(p=2, dim=1).mean()

                # 성별별 perturbation 크기
                if male_idx:
                    delta_m = delta[male_idx]
                    delta_linf_m = delta_m.abs().amax(dim=(1, 2, 3)).mean()
                else:
                    delta_linf_m = torch.tensor(0.0, device=device)
                if female_idx:
                    delta_f = delta[female_idx]
                    delta_linf_f = delta_f.abs().amax(dim=(1, 2, 3)).mean()
                else:
                    delta_linf_f = torch.tensor(0.0, device=device)

                # Matched detection scores for monitoring
                matched_f = _matched_detection_scores(detr, outputs_f, targets_f)
                matched_m = _matched_detection_scores(detr, outputs_m, targets_m)
                mscore_f = matched_f.float().mean() if matched_f.numel() > 0 else torch.tensor(0.0, device=device)
                mscore_m = matched_m.float().mean() if matched_m.numel() > 0 else torch.tensor(0.0, device=device)

            # =================================================================
            # Backward & Optimize
            # =================================================================
            total_g.backward()
            if args.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(generator.parameters()) + list(proj_head.parameters()),
                    args.max_norm,
                )
            opt_g.step()

            # Log
            metrics_logger.update(
                loss_con=loss_contrastive.item(),
                loss_det_f=loss_det_f.item(),
                loss_det_m=loss_det_m.item(),
                loss_l2=loss_l2_anchor.item(),
                total_g=total_g.item(),
                eps=current_eps,
                beta=current_beta,
                lr=current_lr,
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                delta_linf_f=delta_linf_f.item(),
                delta_linf_m=delta_linf_m.item(),
                matched_score_f=mscore_f.item(),
                matched_score_m=mscore_m.item(),
                score_gap=con_info.get("score_gap", 0.0),
                n_f=con_info.get("n_f", 0),
                n_m=con_info.get("n_m", 0),
            )

        # Update scheduler
        scheduler.step()

        # =====================================================================
        # End of Epoch
        # =====================================================================
        metrics_logger.synchronize_between_processes()

        if utils.is_main_process():
            mf = metrics_logger.meters["matched_score_f"].global_avg
            mm = metrics_logger.meters["matched_score_m"].global_avg

            log_entry = {
                "epoch": epoch,
                "loss_con": metrics_logger.meters["loss_con"].global_avg,
                "loss_det_f": metrics_logger.meters["loss_det_f"].global_avg,
                "loss_det_m": metrics_logger.meters["loss_det_m"].global_avg,
                "loss_l2": metrics_logger.meters["loss_l2"].global_avg,
                "total_g": metrics_logger.meters["total_g"].global_avg,
                "epsilon": current_eps,
                "beta": current_beta,
                "lr": current_lr,
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "delta_linf_f": metrics_logger.meters["delta_linf_f"].global_avg,
                "delta_linf_m": metrics_logger.meters["delta_linf_m"].global_avg,
                "matched_score_f": mf,
                "matched_score_m": mm,
                "matched_score_gap": mm - mf,
                "score_gap": metrics_logger.meters["score_gap"].global_avg,
                "n_f_avg": metrics_logger.meters["n_f"].global_avg,
                "n_m_avg": metrics_logger.meters["n_m"].global_avg,
            }

            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(f"\n[Epoch {epoch}] Summary:")
            print(f"  Contrastive: {log_entry['loss_con']:.4f}  |  L2 Anchor: {log_entry['loss_l2']:.4f}")
            print(f"  Det Female:  {log_entry['loss_det_f']:.4f}  |  Det Male:  {log_entry['loss_det_m']:.4f}")
            print(f"  Total: {log_entry['total_g']:.4f}")
            print(f"  Matched Score (F/M): {mf:.4f} / {mm:.4f}  |  Gap(M-F): {log_entry['matched_score_gap']:.4f}")
            print(f"  Delta L_inf (F/M): {log_entry['delta_linf_f']:.4f} / {log_entry['delta_linf_m']:.4f}")
            print(f"  Epsilon: {current_eps:.4f}  |  Beta: {current_beta:.4f}  |  LR: {current_lr:.6f}")

            if (epoch + 1) % args.save_every == 0:
                ckpt_save_path = output_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "generator": _unwrap_ddp(generator).state_dict(),
                        "proj_head": _unwrap_ddp(proj_head).state_dict(),
                        "opt_g": opt_g.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "args": vars(args),
                    },
                    ckpt_save_path,
                )
                print(f"  Saved: {ckpt_save_path}")

        if args.distributed:
            dist.barrier()

    # =========================================================================
    # Training Complete
    # =========================================================================
    if utils.is_main_process():
        print("\n" + "=" * 70)
        print("fix11 Contrastive + L2 Anchoring 학습 완료")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print("\n[Loss 구조]")
        print(f"  lambda_con={args.lambda_con} * L_contrastive  (Score-Weighted, no .detach())")
        print(f"  beta={args.beta}->{args.beta_final} * L_det_female")
        print(f"  beta_m={args.beta_m} * L_det_male")
        print(f"  gamma={args.gamma} * L_L2_anchor  (Male Feature L2 Anchoring)")
        print("\n성공 기준:")
        print("  AP Gap < 0.103 (baseline 0.106 대비 개선)")
        print("  delta_linf_m << delta_linf_f (남성 perturbation 억제 확인)")


if __name__ == "__main__":
    main()
