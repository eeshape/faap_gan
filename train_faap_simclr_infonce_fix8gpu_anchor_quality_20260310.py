"""
FAAP Training - Quality SupCon + Unidirectional Wasserstein (fix7gpu_20260310)

성능 중심의 anchor설정입니다.  "같은 detection 품질이면 당기고, 다른 품질이면 민다"
=============================================================================
이전 버전 문제 분석 (3rd_fix1_gpu):
=============================================================================
1. Fair Centroid Alignment이 Contrastive Learning을 파괴
   - Centroid로 collapse → representation 구조 소멸
   - E0~E3 좋다가 E3 이후 성능 퇴화의 근본 원인
2. Cross-Gender Contrastive의 semantic basis 부재
   - "성별이 다르면 무조건 positive" → 품질 무관하게 당김
3. _ap_proxy_score 정의만 하고 미사용
4. Wasserstein 양방향(abs) → 7th 논문의 단방향(ReLU) 무시
5. Epsilon 고정 0.10, LR warmup 없음

=============================================================================
fix7 핵심 변경: Quality SupCon + Unidirectional Wasserstein
=============================================================================

[구조적 변경]
1. Fair Centroid + Cross-Gender Contrastive 전부 제거
2. QualitySupConLoss 도입
   - Label = detection score quantile bin (gender 아님)
   - 같은 quality bin = positive, 다른 bin = negative
   - 모든 샘플이 anchor (양방향)
   - Gender invariance가 데이터 분포 비대칭성에서 자연 달성
3. Hungarian Matcher Score (score_contrastive.py에서 차용)
   - DETR criterion.matcher로 GT-Pred 매칭
   - 매칭된 prediction의 confidence → AP와 상관성 높음

[역할 분담]
- SupCon: feature 공간에서 quality별 구조 형성 (표현 학습)
- Wasserstein: Female score를 Male 수준으로 끌어올림 (성능 정렬)
- Detection: 전체 검출 성능 유지 (성능 보존)
→ 세 loss가 충돌하지 않고 보완

[학습 안정성]
4. Epsilon 3단계 스케줄링 (warmup→hold→cooldown)
5. LR Warmup + Cosine Annealing
6. Contrastive Warmup (처음엔 약하게)
7. EMA Generator
8. Best model 추적 (|score_gap| 기준)

=============================================================================
GPU 최적화 (A100 / RTX 5090)
=============================================================================
- BF16 AMP (torch.autocast, GradScaler 불필요)
- TF32 활성화 (Tensor Core)
- torch.compile (reduce-overhead, graph-level 최적화)
- channels_last memory format (conv Tensor Core 최적)
- cudnn.benchmark (conv autotuning)
- zero_grad(set_to_none=True) (메모리 절약)
=============================================================================
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Sequence, Tuple

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
# Projection Head
# =============================================================================

class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, output_dim: int = 128, dropout: float = 0.1):
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
# Quality SupCon Loss (핵심)
# =============================================================================

class QualitySupConLoss(nn.Module):
    """
    Detection quality 기반 Supervised Contrastive Loss.

    - Label: score quantile bin (성별이 아님)
    - Positive: 같은 quality bin의 모든 샘플
    - Negative: 다른 quality bin의 모든 샘플
    - Gender invariance가 데이터 분포 비대칭성에서 자연 달성

    현재 데이터 분포:
      HIGH bin: Male 많음, Female 적음
      LOW bin:  Female 많음, Male 적음
    → Generator가 Female에 perturbation → DETR feature가 HIGH 방향으로 이동
    """

    def __init__(self, temperature: float = 0.1, n_bins: int = 3):
        super().__init__()
        self.temperature = temperature
        self.n_bins = n_bins

    def forward(
        self,
        projections: torch.Tensor,  # (N, D) L2-normalized
        scores: torch.Tensor,       # (N,) detection scores
        genders: list,              # 로깅용
    ) -> Tuple[torch.Tensor, dict]:

        n = projections.size(0)
        if n < 4:
            return projections.new_tensor(0.0), {"n_bins_used": 0, "score_gap": 0.0}

        # Quality bin 할당 (batch 내 quantile)
        n_bins = min(self.n_bins, n // 2)  # 샘플 수 부족 시 bin 수 축소
        boundaries = torch.quantile(
            scores.detach(),
            torch.linspace(0, 1, n_bins + 1, device=scores.device)[1:-1],
        )
        labels = torch.bucketize(scores.detach(), boundaries)  # (N,)

        # Similarity matrix
        sim = torch.mm(projections, projections.t()) / self.temperature  # (N, N)

        # Masks
        mask_self = torch.eye(n, device=projections.device, dtype=torch.bool)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask_self

        # Per-anchor SupCon loss
        sim_masked = sim.masked_fill(mask_self, float('-inf'))
        log_denom = torch.logsumexp(sim_masked, dim=1)  # (N,)

        loss = projections.new_tensor(0.0)
        valid_anchors = 0

        for i in range(n):
            pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            pos_loss = (sim[i, pos_idx] - log_denom[i]).mean()
            loss = loss - pos_loss
            valid_anchors += 1

        if valid_anchors > 0:
            loss = loss / valid_anchors

        # 로깅
        female_idx = [i for i, g in enumerate(genders) if g == "female"]
        male_idx = [i for i, g in enumerate(genders) if g == "male"]

        score_gap = 0.0
        if female_idx and male_idx:
            score_gap = (scores[male_idx].mean() - scores[female_idx].mean()).item()

        info = {
            "n_bins_used": len(labels.unique()),
            "score_gap": score_gap,
            "score_f_mean": scores[female_idx].mean().item() if female_idx else 0.0,
            "score_m_mean": scores[male_idx].mean().item() if male_idx else 0.0,
            "valid_anchors": valid_anchors,
        }

        return loss, info


# =============================================================================
# Hungarian Matcher Score (score_contrastive.py에서 차용)
# =============================================================================

def _hungarian_matcher_score(
    detr: FrozenDETR,
    outputs: dict,
    targets: Sequence[dict],
) -> torch.Tensor:
    """
    DETR Hungarian matcher로 GT-Pred 매칭 후 매칭된 prediction의 confidence 반환.
    Top-K confidence보다 AP와의 상관성이 높음.
    """
    if len(targets) == 0:
        return outputs["pred_logits"].new_zeros(0)

    indices = detr.criterion.matcher(outputs, targets)
    probs = outputs["pred_logits"].softmax(dim=-1)

    image_scores = []
    for b, (src_idx, tgt_idx) in enumerate(indices):
        if len(src_idx) == 0:
            # GT가 없거나 매칭 실패 → top-5 fallback
            max_probs = probs[b, :, :-1].max(dim=-1).values
            topk = min(5, max_probs.size(0))
            image_scores.append(max_probs.topk(topk).values.mean())
        else:
            tgt_labels = targets[b]["labels"][tgt_idx]
            matched_scores = probs[b, src_idx, tgt_labels]
            image_scores.append(matched_scores.mean())

    return torch.stack(image_scores)


def _image_level_detection_score(outputs: dict, top_k: int = 10) -> torch.Tensor:
    """Fallback: top-K confidence score"""
    probs = outputs["pred_logits"].softmax(dim=-1)[..., :-1]
    max_probs = probs.max(dim=-1).values
    if top_k > 0 and top_k < max_probs.size(1):
        topk_probs = max_probs.topk(top_k, dim=1).values
        return topk_probs.mean(dim=1)
    return max_probs.mean(dim=1)


# =============================================================================
# Wasserstein Loss (단방향 - 7th 논문 버전)
# =============================================================================

def _resize_sorted(sorted_vals: torch.Tensor, k: int) -> torch.Tensor:
    if sorted_vals.numel() == k:
        return sorted_vals
    idx = torch.linspace(0, sorted_vals.numel() - 1, k, device=sorted_vals.device)
    idx_low, idx_high = idx.floor().long(), idx.ceil().long()
    weight = idx - idx_low.float()
    return sorted_vals[idx_low] * (1 - weight) + sorted_vals[idx_high] * weight


def _wasserstein_1d_unidirectional(female_scores: torch.Tensor, male_scores: torch.Tensor) -> torch.Tensor:
    """
    단방향 Wasserstein: Female score를 Male 수준으로 끌어올림.
    - male_scores.detach(): Male을 타겟으로 고정
    - F.relu(sorted_m - sorted_f): Female < Male일 때만 패널티
    """
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0)

    sorted_f = female_scores.sort().values
    sorted_m = male_scores.detach().sort().values

    k = max(sorted_f.numel(), sorted_m.numel())
    sorted_f = _resize_sorted(sorted_f, k)
    sorted_m = _resize_sorted(sorted_m, k)

    return F.relu(sorted_m - sorted_f).mean()


# =============================================================================
# Epsilon Scheduling (3단계)
# =============================================================================

def _scheduled_epsilon(
    epoch: int,
    eps_start: float, eps_peak: float, eps_min: float,
    warmup_epochs: int, hold_epochs: int, cooldown_epochs: int,
) -> float:
    warmup_end = warmup_epochs
    hold_end = warmup_end + hold_epochs

    if epoch < warmup_end:
        progress = epoch / max(1, warmup_end)
        return eps_start + (eps_peak - eps_start) * progress
    elif epoch < hold_end:
        return eps_peak
    else:
        cooldown_progress = min((epoch - hold_end) / max(1, cooldown_epochs), 1.0)
        return eps_peak + (eps_min - eps_peak) * cooldown_progress


# =============================================================================
# EMA Model
# =============================================================================

class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: dict):
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


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


def _unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def _scheduled_beta(epoch: int, total_epochs: int, beta_start: float, beta_final: float) -> float:
    if total_epochs <= 1:
        return beta_start
    progress = min(epoch / max(1, total_epochs - 1), 1.0)
    return beta_start + (beta_final - beta_start) * progress


def _contrastive_warmup_weight(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    if epoch >= warmup_epochs:
        return 1.0
    return epoch / warmup_epochs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FAAP Quality SupCon (fix7gpu_20260310)")

    # Paths
    parser.add_argument("--dataset_root", type=str, default="/workspace/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()))
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(Path(__file__)))

    # Training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr_g", type=float, default=5e-5)
    parser.add_argument("--lr_warmup_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    # Perturbation (3단계 스케줄링)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--epsilon_final", type=float, default=0.10)
    parser.add_argument("--epsilon_min", type=float, default=0.09)
    parser.add_argument("--epsilon_warmup_epochs", type=int, default=8)
    parser.add_argument("--epsilon_hold_epochs", type=int, default=6)
    parser.add_argument("--epsilon_cooldown_epochs", type=int, default=10)

    # Loss weights
    parser.add_argument("--lambda_supcon", type=float, default=1.0)
    parser.add_argument("--lambda_wass", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--beta_final", type=float, default=0.6)

    # SupCon settings
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--n_bins", type=int, default=3)
    parser.add_argument("--contrastive_warmup_epochs", type=int, default=3)

    # Score computation
    parser.add_argument("--use_hungarian_score", action="store_true", default=True)
    parser.add_argument("--no_hungarian_score", dest="use_hungarian_score", action="store_false")
    parser.add_argument("--score_top_k", type=int, default=10)

    # Projection head
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--proj_dropout", type=float, default=0.1)

    # EMA
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--no_ema", action="store_true", default=False)

    # torch.compile
    parser.add_argument("--no_compile", action="store_true", default=False,
                        help="Disable torch.compile (디버깅 시 사용)")

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


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    args = parse_args()
    utils.init_distributed_mode(args)

    if not hasattr(args, "gpu"):
        args.gpu = None

    # ==========================================================================
    # GPU Optimization (A100 / RTX 5090)
    # ==========================================================================
    torch.backends.cuda.matmul.allow_tf32 = True       # TF32 matmul (A100+)
    torch.backends.cudnn.allow_tf32 = True              # TF32 cuDNN
    torch.backends.cudnn.benchmark = True               # Conv autotuning
    torch.set_float32_matmul_precision('high')          # torch.compile 호환

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
        print("Quality SupCon + Unidirectional Wasserstein (fix7gpu_20260310)")
        print("=" * 70)
        print("[핵심 변경]")
        print("  1. Fair Centroid + Cross-Gender Contrastive 제거")
        print("  2. QualitySupConLoss (score bin 기반, 성별 무관)")
        print("  3. Hungarian Matcher Score (AP 상관성 높음)")
        print("  4. 단방향 Wasserstein (Female → Male 수준)")
        print("[역할 분담]")
        print("  SupCon:      feature quality 구조 형성")
        print("  Wasserstein: Female score → Male 수준으로")
        print("  Detection:   검출 성능 유지")
        print("-" * 70)
        print(f"Temperature: {args.temperature}, Bins: {args.n_bins}")
        print(f"LR: {args.lr_g} (warmup: {args.lr_warmup_epochs} epochs)")
        print(f"Epsilon: {args.epsilon} → {args.epsilon_final} → {args.epsilon_min}")
        print(f"Score: {'Hungarian Matcher' if args.use_hungarian_score else 'Top-K'}")
        print(f"EMA: {'ON (decay={args.ema_decay})' if not args.no_ema else 'OFF'}")
        print(f"Batch size: {args.batch_size}")
        print(f"torch.compile: {'OFF' if args.no_compile else 'ON (reduce-overhead)'}")
        print(f"channels_last: ON, cudnn.benchmark: ON")
        print("=" * 70)

    # ==========================================================================
    # Model Initialization
    # ==========================================================================

    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device, memory_format=torch.channels_last)

    proj_head = ProjectionHead(
        input_dim=detr.hidden_dim,
        hidden_dim=detr.hidden_dim,
        output_dim=args.proj_dim,
        dropout=args.proj_dropout,
    ).to(device)

    supcon_loss_fn = QualitySupConLoss(
        temperature=args.temperature,
        n_bins=args.n_bins,
    ).to(device)

    # torch.compile (PyTorch 2.0+) — graph-level 최적화
    if not args.no_compile:
        generator = torch.compile(generator, mode="reduce-overhead")
        proj_head = torch.compile(proj_head, mode="reduce-overhead")
        if utils.is_main_process():
            print("[torch.compile] Generator + ProjHead compiled (reduce-overhead)")

    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)
        proj_head = DDP(proj_head, device_ids=[args.gpu] if args.gpu is not None else None)

    params = list(_unwrap_ddp(generator).parameters()) + list(_unwrap_ddp(proj_head).parameters())
    opt_g = torch.optim.AdamW(params, lr=args.lr_g, weight_decay=0.01)

    # LR Warmup + Cosine Annealing
    def lr_lambda(epoch):
        if epoch < args.lr_warmup_epochs:
            return (epoch + 1) / args.lr_warmup_epochs
        progress = (epoch - args.lr_warmup_epochs) / max(1, args.epochs - args.lr_warmup_epochs)
        return 0.1 + 0.9 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item()) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)

    # EMA
    ema_gen = None
    ema_proj = None
    if not args.no_ema:
        ema_gen = EMAModel(_unwrap_ddp(generator), decay=args.ema_decay)
        ema_proj = EMAModel(_unwrap_ddp(proj_head), decay=args.ema_decay)

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
        if "ema_gen" in ckpt and ema_gen is not None:
            ema_gen.load_state_dict(ckpt["ema_gen"])
        if "ema_proj" in ckpt and ema_proj is not None:
            ema_proj.load_state_dict(ckpt["ema_proj"])
        if utils.is_main_process():
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # DataLoader
    train_loader, _ = build_faap_dataloader(
        Path(args.dataset_root),
        "train",
        args.batch_size,
        include_gender=True,
        balance_genders=False,
        num_workers=args.num_workers,
        distributed=args.distributed,
        rank=args.rank,
        world_size=args.world_size,
    )

    log_path = output_dir / "train_log.jsonl"

    # Best model tracking
    best_score_gap = float('inf')
    best_epoch = -1

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
        current_epsilon = _scheduled_epsilon(
            epoch,
            args.epsilon, args.epsilon_final, args.epsilon_min,
            args.epsilon_warmup_epochs, args.epsilon_hold_epochs, args.epsilon_cooldown_epochs,
        )
        contrastive_weight = _contrastive_warmup_weight(epoch, args.contrastive_warmup_epochs)
        current_lr = scheduler.get_last_lr()[0]

        # 동적 epsilon 설정
        _unwrap_ddp(generator).epsilon = current_epsilon

        for samples, targets, genders in metrics_logger.log_every(
            train_loader, args.log_every, f"Epoch {epoch}"
        ):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            genders = [g.lower() for g in genders]

            female_idx = [i for i, g in enumerate(genders) if g == "female"]
            male_idx = [i for i, g in enumerate(genders) if g == "male"]

            # =================================================================
            # Forward Pass (BF16 AMP)
            # =================================================================
            opt_g.zero_grad(set_to_none=True)  # set_to_none: 메모리 절약

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Perturbation 적용 (channels_last for Tensor Core)
                tensors = samples.tensors.to(memory_format=torch.channels_last)
                delta = generator(tensors)
                perturbed_tensors = clamp_normalized(tensors + delta)
                perturbed = NestedTensor(perturbed_tensors, samples.mask)

                # DETR forward
                outputs, features = detr.forward_with_features(perturbed)

                # =============================================================
                # 1. Score 계산 (Hungarian Matcher)
                # =============================================================
                if args.use_hungarian_score:
                    image_scores = _hungarian_matcher_score(detr, outputs, targets)
                else:
                    image_scores = _image_level_detection_score(outputs, top_k=args.score_top_k)

                # =============================================================
                # 2. Quality SupCon Loss (핵심)
                # =============================================================
                projections = proj_head(features)
                loss_supcon, supcon_info = supcon_loss_fn(projections, image_scores, genders)

                # =============================================================
                # 3. Wasserstein Loss (단방향: Female → Male)
                # =============================================================
                loss_wasserstein = perturbed_tensors.new_tensor(0.0)
                if len(female_idx) > 0 and len(male_idx) > 0:
                    scores_f = image_scores[female_idx]
                    scores_m = image_scores[male_idx]
                    loss_wasserstein = _wasserstein_1d_unidirectional(scores_f, scores_m)

                # =============================================================
                # 4. Detection Loss
                # =============================================================
                loss_det, _ = detr.detection_loss(outputs, targets)

                # =============================================================
                # Total Loss
                # =============================================================
                effective_lambda_supcon = args.lambda_supcon * contrastive_weight

                total_g = (
                    effective_lambda_supcon * loss_supcon
                    + args.lambda_wass * loss_wasserstein
                    + current_beta * loss_det
                )

            # Metrics
            with torch.no_grad():
                delta_val = perturbed_tensors - tensors
                delta_linf = delta_val.abs().amax(dim=(1, 2, 3)).mean()
                delta_l2 = delta_val.flatten(1).norm(p=2, dim=1).mean()

            # Backward & Optimize (BF16은 GradScaler 불필요)
            total_g.backward()
            if args.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(generator.parameters()) + list(proj_head.parameters()),
                    args.max_norm
                )
            opt_g.step()

            # EMA update
            if ema_gen is not None:
                ema_gen.update(_unwrap_ddp(generator))
                ema_proj.update(_unwrap_ddp(proj_head))

            # Log
            metrics_logger.update(
                loss_supcon=loss_supcon.item(),
                loss_wasserstein=loss_wasserstein.item(),
                loss_det=loss_det.item(),
                total_g=total_g.item(),
                beta=current_beta,
                epsilon=current_epsilon,
                lr=current_lr,
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                score_gap=supcon_info.get("score_gap", 0.0),
                score_f=supcon_info.get("score_f_mean", 0.0),
                score_m=supcon_info.get("score_m_mean", 0.0),
                n_bins_used=supcon_info.get("n_bins_used", 0),
                n_f=len(female_idx),
                n_m=len(male_idx),
            )

        # Update scheduler
        scheduler.step()

        # =====================================================================
        # End of Epoch
        # =====================================================================
        metrics_logger.synchronize_between_processes()

        if utils.is_main_process():
            avg_score_gap = abs(metrics_logger.meters["score_gap"].global_avg)

            log_entry = {
                "epoch": epoch,
                "loss_supcon": metrics_logger.meters["loss_supcon"].global_avg,
                "loss_wasserstein": metrics_logger.meters["loss_wasserstein"].global_avg,
                "loss_det": metrics_logger.meters["loss_det"].global_avg,
                "total_g": metrics_logger.meters["total_g"].global_avg,
                "beta": current_beta,
                "epsilon": current_epsilon,
                "lr": current_lr,
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "score_f": metrics_logger.meters["score_f"].global_avg,
                "score_m": metrics_logger.meters["score_m"].global_avg,
                "score_gap": metrics_logger.meters["score_gap"].global_avg,
                "abs_score_gap": avg_score_gap,
                "n_bins_used": metrics_logger.meters["n_bins_used"].global_avg,
                "n_f_avg": metrics_logger.meters["n_f"].global_avg,
                "n_m_avg": metrics_logger.meters["n_m"].global_avg,
            }

            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Best model tracking
            is_best = avg_score_gap < best_score_gap
            if is_best:
                best_score_gap = avg_score_gap
                best_epoch = epoch

            print(f"\n[Epoch {epoch}] Summary:")
            print(f"  SupCon Loss: {log_entry['loss_supcon']:.4f} (bins: {log_entry['n_bins_used']:.0f})")
            print(f"  Wasserstein Loss: {log_entry['loss_wasserstein']:.4f} (unidirectional)")
            print(f"  Detection Loss: {log_entry['loss_det']:.4f}")
            print(f"  Total: {log_entry['total_g']:.4f}")
            print(f"  Score (F/M): {log_entry['score_f']:.4f} / {log_entry['score_m']:.4f}")
            print(f"  Score Gap (M-F): {log_entry['score_gap']:.4f}")
            print(f"  Epsilon: {current_epsilon:.4f}, LR: {current_lr:.6f}, Beta: {current_beta:.4f}")
            if is_best:
                print(f"  *** NEW BEST (|gap|={avg_score_gap:.4f}) ***")

            if (epoch + 1) % args.save_every == 0:
                save_dict = {
                    "epoch": epoch,
                    "generator": _unwrap_ddp(generator).state_dict(),
                    "proj_head": _unwrap_ddp(proj_head).state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": vars(args),
                }
                if ema_gen is not None:
                    save_dict["ema_gen"] = ema_gen.state_dict()
                    save_dict["ema_proj"] = ema_proj.state_dict()

                ckpt_path_save = output_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
                torch.save(save_dict, ckpt_path_save)
                print(f"  Saved: {ckpt_path_save}")

                if is_best:
                    best_save_dict = copy.deepcopy(save_dict)
                    if ema_gen is not None:
                        best_save_dict["generator"] = ema_gen.state_dict()
                        best_save_dict["proj_head"] = ema_proj.state_dict()
                    best_path = output_dir / "checkpoints" / "best_model.pth"
                    torch.save(best_save_dict, best_path)
                    print(f"  Best model saved: {best_path}")

        if args.distributed:
            dist.barrier()

    # =========================================================================
    # Training Complete
    # =========================================================================
    if utils.is_main_process():
        print("\n" + "=" * 70)
        print("Quality SupCon Training Complete! (fix7gpu_20260310)")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print(f"Best epoch: {best_epoch} (|score_gap|={best_score_gap:.4f})")
        print("\n[핵심 구조]")
        print("  SupCon:      quality bin 기반 contrastive (성별 무관)")
        print("  Wasserstein: Female score → Male 수준 (단방향)")
        print("  Detection:   검출 성능 유지")
        print("\n성공 기준:")
        print("  - AP Gap < 0.10 (baseline 0.106)")
        print("  - Female AP > 0.41 (baseline 0.404)")
        print("  - AR Gap < 0.003 (7th 논문 수준)")


if __name__ == "__main__":
    main()
