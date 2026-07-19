"""
FAAP Training - Anchor 3-Way 비교 (B) L2 ANCHOR: male feature 점별 고정 [2026-07-08]

=============================================================================
3-Way 통제 비교 (anchor 항 외 코드·하이퍼파라미터 완전 동일):
  (A) 20260708_contrastive_baseline.py        : anchor 없음 (통제군)
  (B) 20260708_contrastive_l2_anchor.py       : + gamma * L_L2_anchor (점별 고정)
  (C) 20260708_contrastive_centroid_anchor.py : + gamma * L_centroid_anchor (평균 방향 정렬)
=============================================================================

배경 (20260617 OFAT 스윕 진단):
- no-anchor contrastive만으로는 AP gap 이 노이즈 수준(~2%)에서 정체
- best epoch 이 0~1 → 학습이 공정성을 지속적으로 개선하지 못함
- 가설: contrastive 가 female→male 로 당길 때 male feature(또는 centroid)가
  함께 끌려가는 드리프트로 개선이 상쇄됨 → anchor 복원으로 검증

공통 기본값 (3파일 동일 — 스윕 결과 반영):
- lambda_con=2.5  : 20260617 OFAT 스윕 최적 (AP gap·유틸리티 모두 1위)
- temperature=0.1 : 스윕 중심점 (tau 스윕은 미완이므로 검증된 값 유지)
- epochs=8        : 스윕 프로토콜과 동일 (lambda=2.5 가 검증된 조건)
- batch_size=5    : 스윕과 동일 (centroid 안정화에는 10 권장 — 올릴 땐 3파일 모두 동일하게)

원본 계보:
- 코어: 20260617_contrastive_baseline_hyperparameter.py (fix11 no-L2 ablation + 스윕 로깅)
- L2 anchor: 구실험파일/train_faap_fix11_contrastive_gpu_20260410.py 의 수식 그대로
- Centroid anchor: 20260627_loss_centroid_fix1 (pyc 디스어셈블로 설계 복원)
=============================================================================

Total Loss = lambda_con * L_contrastive
           + beta     * L_det_female
           + beta_m   * L_det_male
           + gamma    * L_L2_anchor

(1) L_contrastive: Score-Weighted Contrastive Loss (.detach() 없음, 양쪽 gradient)
(2) L_det_female : 여성 검출 보존 (DETR criterion)
(3) L_det_male   : 남성 검출 보존 (DETR criterion)
(4) L_L2_anchor  : F.mse_loss(z_pert_male, z_clean_male)
    - teacher(z_clean_male)는 no_grad 고정 target — clean forward 로 계산
    - 개별 male feature 를 점별로 clean 위치에 고정 (가장 강한 제약)
    - gamma=0.5 기본 (원본 fix11 권장 0.5~1.0)
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

from faap_gan.datasets import build_faap_dataloader, inspect_faap_dataset
from faap_gan.models import FrozenDETR, PerturbationGenerator, clamp_normalized
from faap_gan.path_utils import DETR_REPO, default_detr_checkpoint, ensure_detr_repo_on_path

ensure_detr_repo_on_path(DETR_REPO)

import util.misc as utils
from util.misc import NestedTensor

try:
    import wandb
except ImportError:
    wandb = None


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
        pooled = x.mean(dim=1)
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
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        proj_f: torch.Tensor,
        proj_m: torch.Tensor,
        scores_f: torch.Tensor,
        scores_m: torch.Tensor,
    ) -> tuple:
        n_f = proj_f.size(0)
        n_m = proj_m.size(0)

        if n_f < 1 or n_m < 1:
            return proj_f.new_tensor(0.0), {"n_f": n_f, "n_m": n_m}

        weights_m = torch.softmax(scores_m.detach() / 0.1, dim=0)

        sim_f2m = torch.mm(proj_f, proj_m.t()) / self.temperature
        sim_f2f = torch.mm(proj_f, proj_f.t()) / self.temperature
        mask_self = torch.eye(n_f, device=proj_f.device, dtype=torch.bool)
        sim_f2f = sim_f2f.masked_fill(mask_self, float('-inf'))

        weighted_pos = (sim_f2m * weights_m.unsqueeze(0)).sum(dim=1)
        all_sims = torch.cat([sim_f2m, sim_f2f], dim=1)
        log_denom = torch.logsumexp(all_sims, dim=1)

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
    if len(targets) == 0:
        return outputs["pred_logits"].new_zeros(0)

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
        "FAAP 3-way anchor comparison (B) L2 anchor (point-wise male fix)"
    )

    # Paths
    parser.add_argument("--dataset_root", type=str, default="/workspace/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()))
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(Path(__file__)))

    # Training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=8)   # 20260617 OFAT 스윕 프로토콜과 동일 (lambda=2.5 검증 조건)
    parser.add_argument("--stop_epoch", type=int, default=-1,
                        help="이 epoch까지만 학습 후 조기 종료 (-1=비활성). LR/epsilon/beta 스케줄은 --epochs 기준 유지")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr_g", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)

    # Perturbation
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--epsilon_final", type=float, default=0.10)
    parser.add_argument("--epsilon_warmup_epochs", type=int, default=5)
    parser.add_argument("--epsilon_hold_epochs", type=int, default=4)
    parser.add_argument("--epsilon_cooldown_epochs", type=int, default=6)
    parser.add_argument("--epsilon_min", type=float, default=0.09)

    # Loss weights (+ gamma: L2 anchor)
    parser.add_argument("--lambda_con", type=float, default=2.5,   # 20260617 OFAT 스윕 최적값
                        help="Contrastive loss weight (2.5 = OFAT 스윕에서 AP gap·유틸리티 모두 1위)")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Female detection loss weight (start)")
    parser.add_argument("--beta_final", type=float, default=0.6,
                        help="Female detection loss weight (end)")
    parser.add_argument("--beta_m", type=float, default=0.5,
                        help="Male detection loss weight (fixed)")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="L2 anchor loss weight (원본 fix11 권장 0.5~1.0)")

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

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="faap-anchor-3way-20260708")
    parser.add_argument("--wandb_entity", type=str,
                        default="eeshape-incheon-national-university")
    parser.add_argument("--wandb_run_name", type=str, default="",
                        help="Run name (default: output_dir name)")
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_group", type=str, default="",
                        help="W&B group (e.g. lambda_sweep / tau_sweep) for sweep filtering")
    parser.add_argument("--wandb_tags", type=str, default="",
                        help="Comma-separated W&B tags")

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

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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

    output_dir = Path(args.output_dir)
    if utils.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        with (output_dir / "config.json").open("w") as f:
            json.dump(vars(args), f, indent=2)
    if args.distributed:
        dist.barrier()

    use_wandb = args.wandb and utils.is_main_process()
    if use_wandb and wandb is None:
        raise RuntimeError(
            "wandb 가 설치되어 있지 않습니다. `pip install wandb` 후 다시 실행하세요."
        )
    if use_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name or output_dir.name,
            mode=args.wandb_mode,
            config=vars(args),
            dir=str(output_dir),
            group=args.wandb_group or None,
            tags=[t for t in (args.wandb_tags or "").split(",") if t],
        )
        # epoch 단위 지표는 epoch 을 x축으로 고정 → 여러 run 곡선을 깔끔하게 겹쳐 비교.
        wandb.define_metric("epoch")
        wandb.define_metric("epoch/*", step_metric="epoch")

    if utils.is_main_process():
        dataset_info = inspect_faap_dataset(Path(args.dataset_root))
        with (output_dir / "dataset_layout.json").open("w") as f:
            json.dump(dataset_info, f, indent=2)

        print("=" * 70)
        print("FAAP 3-Way Anchor 비교 (B) L2 ANCHOR: male 점별 고정")
        print("=" * 70)
        print("[Loss 구조]")
        print(f"  (1) lambda_con={args.lambda_con} * L_contrastive  (Score-Weighted, no .detach())")
        print(f"  (2) beta={args.beta}->{args.beta_final} * L_det_female")
        print(f"  (3) beta_m={args.beta_m} * L_det_male")
        print(f"  (4) gamma={args.gamma} * L_L2_anchor  (male feature 점별 MSE 고정)")
        print("-" * 70)
        print("[실험 목적]")
        print("  contrastive가 female→male로 당길 때 male feature를 점별 고정")
        print("  baseline(no anchor) 대비 AP gap 개선 여부 검증")
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

    last_epoch_log = None  # 학습 종료 후 wandb summary 스칼라로 사용
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

                # ----- Clean features (teacher: 고정 target, no_grad) -----
                with torch.no_grad():
                    _, feat_clean = detr.forward_with_features(samples)
                    z_clean = proj_head(feat_clean)  # (B, D), L2-normalized

                # ----- Perturbed features -----
                perturbed = _apply_generator(generator, samples)
                outputs, feat_pert = detr.forward_with_features(perturbed)
                z_pert = proj_head(feat_pert)

                # =============================================================
                # (1) Score-Weighted Contrastive Loss
                # =============================================================
                image_scores = _image_level_detection_score(outputs, top_k=args.score_top_k)

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
                # (4) L2 Anchoring Loss (male feature 점별 고정)
                # =============================================================
                # z_pert_male: gradient O (generator 통해 학습)
                # z_clean_male: gradient X (no_grad teacher)
                z_pert_male = z_pert[male_idx]
                z_clean_male = z_clean[male_idx]
                loss_anchor = F.mse_loss(z_pert_male, z_clean_male)

                # =============================================================
                # Total Loss
                # =============================================================
                total_g = (
                    args.lambda_con * loss_contrastive
                    + current_beta * loss_det_f
                    + args.beta_m * loss_det_m
                    + args.gamma * loss_anchor
                )

            # =================================================================
            # Metrics (autocast 밖, no_grad)
            # =================================================================
            with torch.no_grad():
                delta = perturbed.tensors - samples.tensors
                delta_linf = delta.abs().amax(dim=(1, 2, 3)).mean()
                delta_l2 = delta.flatten(1).norm(p=2, dim=1).mean()

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
                loss_anchor=loss_anchor.item(),
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

            if use_wandb:
                wandb.log({
                    "train/loss_con": loss_contrastive.item(),
                    "train/loss_det_f": loss_det_f.item(),
                    "train/loss_det_m": loss_det_m.item(),
                    "train/loss_anchor": loss_anchor.item(),
                    "train/total_g": total_g.item(),
                    "train/matched_score_f": mscore_f.item(),
                    "train/matched_score_m": mscore_m.item(),
                    "train/delta_linf": delta_linf.item(),
                    "train/delta_linf_f": delta_linf_f.item(),
                    "train/delta_linf_m": delta_linf_m.item(),
                    "train/eps": current_eps,
                    "train/beta": current_beta,
                    "train/lr": current_lr,
                    "epoch": epoch,
                })

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
                "loss_anchor": metrics_logger.meters["loss_anchor"].global_avg,
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

            if use_wandb:
                epoch_log = {f"epoch/{k}": v for k, v in log_entry.items()
                             if k != "epoch"}
                epoch_log["epoch"] = epoch
                wandb.log(epoch_log)

            last_epoch_log = log_entry

            print(f"\n[Epoch {epoch}] Summary (3-way L2 ANCHOR):")
            print(f"  Contrastive: {log_entry['loss_con']:.4f}  |  L2 Anchor: {log_entry['loss_anchor']:.4f}")
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

        # 조기 종료: 스케줄은 --epochs 기준 유지하되 stop_epoch에서 학습만 중단
        if args.stop_epoch >= 0 and epoch >= args.stop_epoch:
            if utils.is_main_process():
                print(f"\n[조기 종료] stop_epoch={args.stop_epoch} 도달 → 학습 중단 "
                      f"(LR/epsilon/beta 스케줄은 epochs={args.epochs} 기준)")
            break

    # =========================================================================
    # Training Complete
    # =========================================================================
    if utils.is_main_process():
        print("\n" + "=" * 70)
        print("3-Way (B) L2 ANCHOR 학습 완료")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print("\n[Loss 구조]")
        print(f"  lambda_con={args.lambda_con} * L_contrastive  (Score-Weighted, no .detach())")
        print(f"  beta={args.beta}->{args.beta_final} * L_det_female")
        print(f"  beta_m={args.beta_m} * L_det_male")
        print(f"  gamma={args.gamma} * L_L2_anchor  (male 점별 MSE 고정)")
        print("\n비교 포인트 (vs baseline/centroid):")
        print("  1. AP Gap: baseline 대비 개선되는가 / centroid 대비 어느 쪽이 나은가")
        print("  2. matched_score_m: male 성능이 점별 고정으로 더 잘 보존되는가")
        print("  3. loss_con: 점별 고정이 contrastive 학습을 방해하지 않는가")

    if use_wandb:
        # run당 단일 요약 스칼라(마지막 epoch proxy 지표) → 스윕 Scatter 보조 축.
        # 최종 공정성 지표(AP gap)는 eval_faap.py 가 같은 project 에 별도 기록.
        if last_epoch_log is not None:
            wandb.run.summary["final/matched_score_gap"] = last_epoch_log["matched_score_gap"]
            wandb.run.summary["final/delta_linf_m"] = last_epoch_log["delta_linf_m"]
            wandb.run.summary["final/delta_linf_f"] = last_epoch_log["delta_linf_f"]
            wandb.run.summary["final/loss_con"] = last_epoch_log["loss_con"]
            wandb.run.summary["final/loss_anchor"] = last_epoch_log["loss_anchor"]
        wandb.finish()


if __name__ == "__main__":
    main()
