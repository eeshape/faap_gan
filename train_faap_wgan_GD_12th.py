"""
FAAP Training - 12th Version: Discriminator-Free Progressive Alignment

================================================================================
                         11th와의 핵심 차별점
================================================================================

[11th 방식 - Unified Framework]
- Discriminator 사용 (GAN 학습)
- epsilon = 0.01 고정 (매우 작음)
- Contrastive + GAN + Quantile 모두 융합
- 복잡한 손실 함수 구조

[12th 방식 - Discriminator-Free Fixed] ⭐ NEW ⭐
- Discriminator 완전 제거 → 학습 안정성 극대화
- epsilon = 0.06 고정 (스케줄 없음 - 단순화)
- Focal Fairness Loss: 어려운 샘플에 더 집중
- 모든 하이퍼파라미터 고정 (재현성 극대화)
- 단순하지만 효과적인 손실 함수 구조

================================================================================
                         12th 설계 철학
================================================================================

1. Discriminator 제거의 이점:
   - GAN 학습 불안정성 완전 제거
   - Mode collapse 위험 없음
   - Contrastive 1st에서 AR Gap -61.73% 달성한 핵심 요소
   - 더 단순한 학습 다이나믹스

2. 고정 Epsilon 전략:
   - epsilon = 0.06 고정 (스케줄 제거)
   - Contrastive 1st의 0.08과 11th의 0.01 사이 중간값
   - 단순하고 재현 가능한 학습
   - 스케줄 복잡도 제거로 안정성 증가

3. Focal Fairness Loss (핵심 혁신):
   - 일반 손실: 모든 샘플에 동일한 가중치
   - Focal 손실: 예측이 어려운(score 낮은) 샘플에 더 높은 가중치
   - Female 그룹의 어려운 샘플을 집중적으로 개선
   - 공식: FocalLoss = -α × (1 - p)^γ × log(p)

4. 고정 가중치 전략:
   - 모든 epoch에서 동일한 가중치 사용
   - β = 0.55 고정 (Detection 보호)
   - λ_focal = 0.4 고정 (Focal weight)
   - 재현성과 안정성 극대화

================================================================================
                         손실 함수 구조 (단순화)
================================================================================

G_Loss = β × Detection_Loss                        # Detection 보호
       + λ_contrast × Cross_Gender_Contrastive     # 특징 공간 정렬 (핵심)
       + λ_align × Mean_Alignment                  # 평균 특징 벡터 정렬
       + λ_var × Variance_Alignment                # 분산 정렬
       + λ_focal × Focal_Score_Alignment           # Focal 기반 score 정렬 (NEW)
       + λ_w × Wasserstein_1D (단방향)              # 분포 정렬

주요 변화:
- Discriminator 관련 손실 제거 (λ_fair × Adversarial 제거)
- Focal Score Alignment 추가 (어려운 샘플 집중)
- Quantile Matching → Focal로 대체 (더 직관적)

하이퍼파라미터 (모두 고정):
- epsilon = 0.06 (고정, 스케줄 없음)
- β = 0.55 (Detection 보호, 고정)
- λ_contrast = 1.2 (Contrastive, 핵심 - Discriminator 없으므로 증가)
- λ_align = 0.6 (Mean Alignment)
- λ_var = 0.15 (Variance Alignment)
- λ_focal = 0.4 (Focal Score Alignment, NEW, 고정)
- λ_w = 0.3 (Wasserstein, 단방향)
- temperature = 0.07 (Contrastive)
- focal_gamma = 2.0 (Focal exponent)
- focal_alpha = 0.75 (Female 가중치)

================================================================================
"""

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

# Allow running as a script: add package root to sys.path
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


def _default_output_dir(script_path: Path) -> str:
    """스크립트 파일 이름을 기반으로 기본 output_dir을 생성한다."""
    stem = script_path.stem
    stem_lower = stem.lower()
    suffix = stem
    for prefix in ("train_faap_wgan_", "train_faap_", "train_"):
        if stem_lower.startswith(prefix):
            suffix = stem[len(prefix):]
            break
    suffix = suffix.lower()
    return str(Path("faap_outputs") / f"faap_outputs_{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "FAAP Training 12th - Discriminator-Free Fixed Hyperparameters",
        add_help=True
    )
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()))
    parser.add_argument(
        "--output_dir",
        type=str,
        default=_default_output_dir(Path(__file__)),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=7)  # Discriminator 없어 메모리 여유
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--lr_g", type=float, default=1e-4, help="generator learning rate")
    
    # ===== Epsilon (고정값) =====
    parser.add_argument("--epsilon", type=float, default=0.06, help="perturbation bound (고정)")
    
    # ===== Detection Loss 가중치 (고정값) =====
    parser.add_argument("--beta", type=float, default=0.55, help="detection loss weight (고정)")
    
    # ===== Contrastive Fairness 설정 (핵심) =====
    parser.add_argument(
        "--lambda_contrast",
        type=float,
        default=1.2,
        help="weight for contrastive fairness loss (핵심 - Discriminator 없으므로 증가)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="temperature for contrastive loss (낮을수록 sharp)",
    )
    parser.add_argument(
        "--lambda_align",
        type=float,
        default=0.6,
        help="weight for feature mean alignment loss",
    )
    parser.add_argument(
        "--lambda_var",
        type=float,
        default=0.15,
        help="weight for feature variance alignment loss",
    )
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=128,
        help="projection dimension for contrastive learning",
    )
    
    # ===== Focal Fairness 설정 (NEW) =====
    parser.add_argument(
        "--lambda_focal",
        type=float,
        default=0.4,
        help="weight for focal score alignment loss (어려운 샘플 집중)",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="focal loss exponent (높을수록 어려운 샘플에 집중)",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.75,
        help="focal loss female weight (Female 그룹 중점)",
    )
    
    # ===== Distribution Alignment 설정 =====
    parser.add_argument(
        "--lambda_w",
        type=float,
        default=0.3,
        help="weight for Wasserstein alignment (단방향: F→M)",
    )

    parser.add_argument(
        "--max_train_per_gender",
        type=int,
        default=0,
        help="per-epoch cap per gender; 0 to disable",
    )
    parser.add_argument("--obj_conf_thresh", type=float, default=0.5)
    parser.add_argument("--max_norm", type=float, default=0.1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--dist_url", default="env://")
    return parser.parse_args()


# ============================================================
# Note: Schedule 함수 제거 - 모든 하이퍼파라미터 고정
# ============================================================


# ============================================================
# Projection Head (Contrastive Learning)
# ============================================================

class ProjectionHead(nn.Module):
    """Contrastive Learning을 위한 Projection Head.
    
    DETR 특징을 저차원 공간으로 투영하여 contrastive loss 계산.
    SimCLR 스타일의 2-layer MLP + L2 정규화.
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_queries, feature_dim)
        pooled = x.mean(dim=1)  # (batch, feature_dim) - 쿼리 평균
        return F.normalize(self.net(pooled), dim=-1)  # L2 정규화


# ============================================================
# 유틸리티 함수들
# ============================================================

def _split_nested(samples: NestedTensor, targets: Sequence[dict], keep: List[int]):
    """배치를 성별로 분리"""
    if len(keep) == 0:
        return None, []
    tensor = samples.tensors[keep]
    mask = samples.mask[keep] if samples.mask is not None else None
    return NestedTensor(tensor, mask), [targets[i] for i in keep]


def _apply_generator(generator: nn.Module, samples: NestedTensor) -> NestedTensor:
    """Generator로 perturbation 적용"""
    tensors = samples.tensors
    delta = generator(tensors)
    perturbed = clamp_normalized(tensors + delta)
    return NestedTensor(perturbed, samples.mask)


def _unwrap_ddp(module: nn.Module) -> nn.Module:
    """DDP wrapper 제거"""
    return module.module if isinstance(module, DDP) else module


def _set_generator_epsilon(generator: nn.Module, epsilon: float) -> None:
    """Generator의 epsilon 설정"""
    _unwrap_ddp(generator).epsilon = epsilon


def _resize_sorted(scores: torch.Tensor, target_len: int) -> torch.Tensor:
    """점수 배열을 target_len 크기로 리사이즈 (선형 보간)"""
    if target_len <= 0:
        return scores.new_zeros(0, device=scores.device)
    if scores.numel() == 0:
        return scores.new_zeros(target_len, device=scores.device)
    if scores.numel() == target_len:
        return scores
    idx = torch.linspace(0, scores.numel() - 1, target_len, device=scores.device)
    idx_low = idx.floor().long()
    idx_high = idx.ceil().long()
    weight = idx - idx_low
    return scores[idx_low] * (1 - weight) + scores[idx_high] * weight


def _matched_detection_scores(detr: FrozenDETR, outputs: dict, targets: Sequence[dict]) -> torch.Tensor:
    """Hungarian matching을 통해 매칭된 detection score 추출"""
    if len(targets) == 0:
        return outputs["pred_logits"].new_zeros(0, device=outputs["pred_logits"].device)
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
    return outputs["pred_logits"].new_zeros(0, device=outputs["pred_logits"].device)


# ============================================================
# 핵심 손실 함수들
# ============================================================

def _cross_gender_contrastive_loss(
    proj_f: torch.Tensor,
    proj_m: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Cross-Gender Contrastive Loss
    
    핵심 아이디어:
    - 다른 성별 샘플을 positive pair로 취급
    - 성별 정보가 특징에서 제거되도록 유도
    - AR Gap 감소에 매우 효과적 (-61.73% 달성)
    """
    if proj_f.size(0) == 0 or proj_m.size(0) == 0:
        return proj_f.new_tensor(0.0)
    
    n_f, n_m = proj_f.size(0), proj_m.size(0)
    
    # 모든 샘플 간 유사도 행렬 (코사인 유사도)
    sim_f_to_m = torch.mm(proj_f, proj_m.t()) / temperature  # (N_f, N_m)
    sim_m_to_f = sim_f_to_m.t()  # (N_m, N_f)
    
    # 여성→남성: 각 여성이 모든 남성과 유사해지도록
    loss_f_to_m = -torch.logsumexp(sim_f_to_m, dim=1).mean() + \
                   torch.log(torch.tensor(n_m, dtype=torch.float, device=proj_f.device))
    
    # 남성→여성: 각 남성이 모든 여성과 유사해지도록
    loss_m_to_f = -torch.logsumexp(sim_m_to_f, dim=1).mean() + \
                   torch.log(torch.tensor(n_f, dtype=torch.float, device=proj_f.device))
    
    return (loss_f_to_m + loss_m_to_f) / 2


def _feature_alignment_loss(
    feat_f: torch.Tensor,
    feat_m: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Feature Alignment Loss: 성별 간 특징 분포 정렬
    
    1. Mean Alignment: 평균 특징 벡터를 일치시킴
    2. Variance Alignment: 분산을 일치시킴
    """
    if feat_f.size(0) == 0 or feat_m.size(0) == 0:
        zero = feat_f.new_tensor(0.0)
        return zero, zero
    
    # 쿼리 차원에서 평균 → (N, feature_dim)
    pooled_f = feat_f.mean(dim=1)
    pooled_m = feat_m.mean(dim=1)
    
    # 배치 평균
    mean_f = pooled_f.mean(dim=0)
    mean_m = pooled_m.mean(dim=0)
    
    # Mean Alignment: L2 거리
    mean_loss = F.mse_loss(mean_f, mean_m)
    
    # Variance Alignment (최소 2개 샘플 필요)
    if pooled_f.size(0) >= 2 and pooled_m.size(0) >= 2:
        var_f = pooled_f.var(dim=0)
        var_m = pooled_m.var(dim=0)
        var_loss = F.mse_loss(var_f, var_m)
    else:
        var_loss = feat_f.new_tensor(0.0)
    
    return mean_loss, var_loss


def _wasserstein_1d(female_scores: torch.Tensor, male_scores: torch.Tensor) -> torch.Tensor:
    """단방향 Wasserstein 손실
    
    여성 score가 남성보다 낮을 때만 패널티.
    남성 성능은 유지하면서 여성만 향상시킴.
    """
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0, device=female_scores.device)
    
    sorted_f = female_scores.sort().values
    sorted_m = male_scores.detach().sort().values  # 남성은 detach (보호)
    
    k = max(sorted_f.numel(), sorted_m.numel())
    sorted_f = _resize_sorted(sorted_f, k)
    sorted_m = _resize_sorted(sorted_m, k)
    
    # 단방향: 여성이 남성보다 낮을 때만 손실
    return F.relu(sorted_m - sorted_f).mean()


def _focal_score_alignment_loss(
    female_scores: torch.Tensor,
    male_scores: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    """Focal Score Alignment Loss (NEW - 12th 핵심 혁신)
    
    핵심 아이디어:
    - 일반 MSE는 모든 샘플에 동일한 가중치
    - Focal Loss는 어려운(score 낮은) 샘플에 더 높은 가중치
    - Female 그룹의 어려운 샘플을 집중적으로 개선
    
    공식:
    - p = normalized score (0~1)
    - focal_weight = (1 - p)^gamma
    - loss = alpha × focal_weight × (target - p)^2
    
    Args:
        female_scores: Female detection scores
        male_scores: Male detection scores (target distribution)
        gamma: Focal exponent (높을수록 어려운 샘플에 집중)
        alpha: Female 그룹 가중치
    
    Returns:
        Focal-weighted score alignment loss
    """
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0, device=female_scores.device)
    
    # Male 평균 score를 목표로 설정
    target_score = male_scores.detach().mean()
    
    # Female score를 0-1로 정규화 (clamp로 안정화)
    p = female_scores.clamp(0.01, 0.99)
    
    # Focal weight: 낮은 score에 높은 가중치
    # (1 - p)^gamma → p가 낮을수록 (1-p)가 커지고, gamma 제곱으로 더 증폭
    focal_weight = (1 - p) ** gamma
    
    # 손실: Male 평균과의 차이를 Focal weight로 가중
    # 단방향: Female이 Male보다 낮을 때만 패널티
    diff = F.relu(target_score - p)
    loss = alpha * focal_weight * (diff ** 2)
    
    return loss.mean()


# ============================================================
# 메인 학습 루프
# ============================================================

def main():
    args = parse_args()
    utils.init_distributed_mode(args)
    if not hasattr(args, "gpu"):
        args.gpu = None
    detr_repo = ensure_detr_repo_on_path(Path(args.detr_repo))
    ckpt_path = Path(args.detr_checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = detr_repo / ckpt_path
    
    # 디바이스 설정
    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA.")
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 시드 설정
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    args.world_size = utils.get_world_size()
    args.rank = utils.get_rank()

    # 출력 디렉터리 생성
    output_dir = Path(args.output_dir)
    if utils.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
    if args.distributed:
        dist.barrier()

    # 데이터셋 정보 저장
    if utils.is_main_process():
        dataset_info = inspect_faap_dataset(Path(args.dataset_root))
        with (output_dir / "dataset_layout.json").open("w") as f:
            json.dump(dataset_info, f, indent=2)

    # ===== 모델 초기화 (Discriminator 없음!) =====
    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)
    
    # Projection Head만 사용 (Discriminator 제거!)
    proj_head = ProjectionHead(
        input_dim=detr.hidden_dim,
        hidden_dim=detr.hidden_dim,
        output_dim=args.proj_dim,
    ).to(device)
    
    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)
        proj_head = DDP(proj_head, device_ids=[args.gpu] if args.gpu is not None else None)

    # Generator + Projection Head만 최적화 (Discriminator 없음)
    g_params = list(_unwrap_ddp(generator).parameters()) + list(_unwrap_ddp(proj_head).parameters())
    opt_g = torch.optim.Adam(g_params, lr=args.lr_g)
    
    # Learning rate scheduler (CosineAnnealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs, eta_min=1e-6)

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

    # 데이터로더 생성
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
    
    # 설정 출력
    if utils.is_main_process():
        print("=" * 70)
        print("FAAP Training 12th - Discriminator-Free Fixed Hyperparameters")
        print("=" * 70)
        print(f"Epsilon (fixed): {args.epsilon}")
        print(f"Beta (fixed): {args.beta}")
        print(f"Lambda contrast: {args.lambda_contrast}")
        print(f"Lambda align: {args.lambda_align}")
        print(f"Lambda var: {args.lambda_var}")
        print(f"Lambda focal: {args.lambda_focal}")
        print(f"Lambda W (Wasserstein): {args.lambda_w}")
        print(f"Focal gamma: {args.focal_gamma}, alpha: {args.focal_alpha}")
        print(f"Temperature: {args.temperature}")
        print(f"NO Discriminator - Pure Contrastive + Focal Learning!")
        print(f"NO Schedule - All Hyperparameters Fixed!")
        print("=" * 70)

    # ===== 학습 루프 =====
    for epoch in range(start_epoch, args.epochs):
        metrics_logger = utils.MetricLogger(delimiter="  ")
        generator.train()
        proj_head.train()
        
        if args.distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        
        # 고정값 사용 (스케줄 없음)
        current_eps = args.epsilon
        current_beta = args.beta
        _set_generator_epsilon(generator, current_eps)
        current_lr = opt_g.param_groups[0]['lr']

        for samples, targets, genders in metrics_logger.log_every(train_loader, args.log_every, f"Epoch {epoch}"):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            genders = [g.lower() for g in genders]

            # 성별로 배치 분리
            female_idx = [i for i, g in enumerate(genders) if g == "female"]
            male_idx = [i for i, g in enumerate(genders) if g == "male"]
            female_batch, female_targets = _split_nested(samples, targets, female_idx)
            male_batch, male_targets = _split_nested(samples, targets, male_idx)

            # 메트릭 초기화
            delta_linf = torch.tensor(0.0, device=device)
            delta_l2 = torch.tensor(0.0, device=device)
            obj_mean = torch.tensor(0.0, device=device)
            obj_frac = torch.tensor(0.0, device=device)
            obj_mean_f = torch.tensor(0.0, device=device)
            obj_frac_f = torch.tensor(0.0, device=device)
            obj_mean_m = torch.tensor(0.0, device=device)
            obj_frac_m = torch.tensor(0.0, device=device)
            contrast_loss = torch.tensor(0.0, device=device)
            align_loss = torch.tensor(0.0, device=device)
            var_loss = torch.tensor(0.0, device=device)
            wasserstein_loss = torch.tensor(0.0, device=device)
            focal_loss = torch.tensor(0.0, device=device)

            # ===== Generator 업데이트 (Discriminator-Free!) =====
            if female_batch is not None or male_batch is not None:
                opt_g.zero_grad()
                
                det_losses = []
                feat_f, feat_m = None, None
                proj_f, proj_m = None, None
                female_scores = torch.tensor([], device=device)
                male_scores = torch.tensor([], device=device)
                
                # 여성 배치 처리
                if female_batch is not None:
                    female_perturbed = _apply_generator(generator, female_batch)
                    outputs_f, feat_f = detr.forward_with_features(female_perturbed)
                    proj_f = proj_head(feat_f)
                    
                    # Detection Loss (유효한 타겟만)
                    valid_f_idx = [i for i, t in enumerate(female_targets) if t["boxes"].numel() > 0]
                    valid_f_targets = [female_targets[i] for i in valid_f_idx]
                    if valid_f_targets:
                        valid_outputs_f = {
                            "pred_logits": outputs_f["pred_logits"][valid_f_idx],
                            "pred_boxes": outputs_f["pred_boxes"][valid_f_idx],
                        }
                        det_f, _ = detr.detection_loss(valid_outputs_f, valid_f_targets)
                        det_losses.append(det_f)
                        female_scores = _matched_detection_scores(detr, valid_outputs_f, valid_f_targets)
                
                # 남성 배치 처리
                if male_batch is not None:
                    male_perturbed = _apply_generator(generator, male_batch)
                    outputs_m, feat_m = detr.forward_with_features(male_perturbed)
                    proj_m = proj_head(feat_m)
                    
                    # Detection Loss (유효한 타겟만)
                    valid_m_idx = [i for i, t in enumerate(male_targets) if t["boxes"].numel() > 0]
                    valid_m_targets = [male_targets[i] for i in valid_m_idx]
                    if valid_m_targets:
                        valid_outputs_m = {
                            "pred_logits": outputs_m["pred_logits"][valid_m_idx],
                            "pred_boxes": outputs_m["pred_boxes"][valid_m_idx],
                        }
                        det_m, _ = detr.detection_loss(valid_outputs_m, valid_m_targets)
                        det_losses.append(det_m)
                        male_scores = _matched_detection_scores(detr, valid_outputs_m, valid_m_targets)
                
                # Detection Loss 합산
                det_loss = torch.stack(det_losses).sum() if det_losses else torch.tensor(0.0, device=device)
                
                # ===== Contrastive Fairness Losses (핵심) =====
                if proj_f is not None and proj_m is not None:
                    # 1. Cross-Gender Contrastive Loss (AR Gap 감소에 핵심)
                    contrast_loss = _cross_gender_contrastive_loss(
                        proj_f, proj_m, args.temperature
                    )
                    
                    # 2. Feature Alignment Loss
                    align_loss, var_loss = _feature_alignment_loss(feat_f, feat_m)
                
                # ===== Distribution Alignment Losses =====
                if female_scores.numel() > 0 and male_scores.numel() > 0:
                    # 3. Wasserstein Loss (단방향)
                    wasserstein_loss = _wasserstein_1d(female_scores, male_scores)
                    
                    # 4. Focal Score Alignment Loss (NEW - 어려운 샘플 집중)
                    focal_loss = _focal_score_alignment_loss(
                        female_scores, male_scores, 
                        gamma=args.focal_gamma, alpha=args.focal_alpha
                    )
                
                # ===== 최종 손실 (Discriminator-Free, 모든 고정값) =====
                total_g = (
                    current_beta * det_loss                       # Detection 보호
                    + args.lambda_contrast * contrast_loss        # Contrastive (AR Gap)
                    + args.lambda_align * align_loss              # Mean Alignment
                    + args.lambda_var * var_loss                  # Variance Alignment
                    + args.lambda_w * wasserstein_loss            # Wasserstein (AP Gap)
                    + args.lambda_focal * focal_loss              # Focal (어려운 샘플, 고정)
                )
                
                # 메트릭 계산
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
                    
                    max_scores_list = []
                    if female_batch is not None:
                        probs_f = outputs_f["pred_logits"].softmax(dim=-1)[..., :-1]
                        max_scores_f = probs_f.max(dim=-1).values
                        obj_mean_f = max_scores_f.mean()
                        obj_frac_f = (max_scores_f > args.obj_conf_thresh).float().mean()
                        max_scores_list.append(max_scores_f)
                    if male_batch is not None:
                        probs_m = outputs_m["pred_logits"].softmax(dim=-1)[..., :-1]
                        max_scores_m = probs_m.max(dim=-1).values
                        obj_mean_m = max_scores_m.mean()
                        obj_frac_m = (max_scores_m > args.obj_conf_thresh).float().mean()
                        max_scores_list.append(max_scores_m)
                    if max_scores_list:
                        max_scores = torch.cat(max_scores_list, dim=0)
                        obj_mean = max_scores.mean()
                        obj_frac = (max_scores > args.obj_conf_thresh).float().mean()

                total_g.backward()
                if args.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.max_norm)
                    torch.nn.utils.clip_grad_norm_(proj_head.parameters(), args.max_norm)
                opt_g.step()
            else:
                det_loss = torch.tensor(0.0, device=device)
                total_g = torch.tensor(0.0, device=device)

            # 메트릭 업데이트 (Discriminator 관련 항목 제거, focal_w 고정)
            metrics_logger.update(
                g_contrast=contrast_loss.item(),
                g_align=align_loss.item(),
                g_var=var_loss.item(),
                g_focal=focal_loss.item(),
                g_w=wasserstein_loss.item(),
                g_det=det_loss.item(),
                g_total=total_g.item(),
                eps=current_eps,
                beta=current_beta,
                lr_g=current_lr,
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                obj_score=obj_mean.item(),
                obj_frac=obj_frac.item(),
                obj_score_f=obj_mean_f.item(),
                obj_frac_f=obj_frac_f.item(),
                obj_score_m=obj_mean_m.item(),
                obj_frac_m=obj_frac_m.item(),
            )

        # LR scheduler step
        scheduler.step()
        
        metrics_logger.synchronize_between_processes()

        # 에폭 종료 로깅 및 저장
        if utils.is_main_process():
            log_entry = {
                "epoch": epoch,
                "g_contrast": metrics_logger.meters["g_contrast"].global_avg,
                "g_align": metrics_logger.meters["g_align"].global_avg,
                "g_var": metrics_logger.meters["g_var"].global_avg,
                "g_focal": metrics_logger.meters["g_focal"].global_avg,
                "g_w": metrics_logger.meters["g_w"].global_avg,
                "g_det": metrics_logger.meters["g_det"].global_avg,
                "g_total": metrics_logger.meters["g_total"].global_avg,
                "epsilon": current_eps,
                "beta": current_beta,
                "lr_g": current_lr,
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "obj_score": metrics_logger.meters["obj_score"].global_avg,
                "obj_frac": metrics_logger.meters["obj_frac"].global_avg,
                "obj_score_f": metrics_logger.meters["obj_score_f"].global_avg,
                "obj_frac_f": metrics_logger.meters["obj_frac_f"].global_avg,
                "obj_score_m": metrics_logger.meters["obj_score_m"].global_avg,
                "obj_frac_m": metrics_logger.meters["obj_frac_m"].global_avg,
            }
            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if (epoch + 1) % args.save_every == 0:
                ckpt_path_save = output_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "generator": _unwrap_ddp(generator).state_dict(),
                        "proj_head": _unwrap_ddp(proj_head).state_dict(),
                        "opt_g": opt_g.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "args": vars(args),
                    },
                    ckpt_path_save,
                )

        if args.distributed:
            dist.barrier()

    if utils.is_main_process():
        print("=" * 70)
        print("Training complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
