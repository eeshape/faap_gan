"""
FAAP Training - 11th Version: Unified Fairness Framework

================================================================================
                         세 가지 방법론 융합 분석
================================================================================

[Contrastive 1st 핵심 강점]
- AR Gap -61.73% 달성 (최대 감소)
- 특징 공간에서 성별 정보 직접 제거
- InfoNCE Loss로 다른 성별을 positive pair로 취급
- Discriminator 없이 안정적 학습

[GD 7th 핵심 강점]
- AP Gap -0.42% 달성 (유일한 감소!)
- 비대칭 Fairness 스케일링 (Female 1.0, Male 0.5)
- 단방향 Wasserstein: Female→Male만 정렬 (Male 성능 보호)
- 적대적 학습으로 성별 예측 불가능하게

[GD 10th 핵심 강점]
- 최고 절대 성능 (Male 51.98%, Female 41.23%)
- Quantile Matching: AP 계산에 중요한 높은 score 영역 정밀 정렬
- Detection Guard: 성별별 Detection 하락 방지

================================================================================
                         11th 설계 철학
================================================================================

1. Epsilon 0.01 환경의 특성:
   - 매우 작은 perturbation → Detection 성능 유지 용이
   - 특징 공간 정렬이 더 중요해짐 (Contrastive 우선)
   - 작은 변화로도 효과를 내야 함 → 손실 함수 효율성 극대화

2. 핵심 융합 전략:
   - Contrastive의 ProjectionHead + InfoNCE (AR Gap 감소)
   - GD 7th의 비대칭 스케일링 + 단방향 Wasserstein (AP Gap 감소)
   - GD 10th의 Quantile Matching (정밀 분포 정렬)
   - Discriminator 사용 (단, 작은 epsilon에 맞게 가중치 조정)

3. 스케줄 미사용, 하이퍼파라미터 고정:
   - 단순하고 재현 가능한 학습
   - epsilon=0.01 고정

================================================================================
                         손실 함수 구조
================================================================================

G_Loss = β × Detection_Loss                        # Detection 보호
       + λ_contrast × Cross_Gender_Contrastive     # 특징 공간 정렬 (핵심)
       + λ_align × Mean_Alignment                  # 평균 특징 벡터 정렬
       + λ_var × Variance_Alignment                # 분산 정렬
       + λ_fair × (Adversarial + α × Entropy)      # 적대적 공정성
       + λ_w × Wasserstein_1D (단방향)              # 분포 정렬
       + λ_q × Quantile_Matching                   # 분위수 정렬

하이퍼파라미터 (모두 고정):
- epsilon = 0.01 (매우 작은 perturbation)
- β = 0.3 (Detection 보호, epsilon 작으므로 낮춤)
- λ_contrast = 1.0 (Contrastive, 핵심 - AR Gap 감소)
- λ_align = 0.5 (Mean Alignment)
- λ_var = 0.15 (Variance Alignment)
- λ_fair = 1.5 (Adversarial Fairness)
- α = 0.2 (Entropy 가중치)
- λ_w = 0.3 (Wasserstein, 단방향)
- λ_q = 0.25 (Quantile Matching)
- fair_f_scale = 1.0 (Female 중점)
- fair_m_scale = 0.4 (Male 낮춤 - GD 7th 성공 요소)
- temperature = 0.07 (Contrastive, 낮을수록 sharp)
- proj_dim = 128 (Projection 차원)
- k_d = 3 (Discriminator 업데이트 횟수)

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
from .models import FrozenDETR, GenderDiscriminator, PerturbationGenerator, clamp_normalized
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
        "FAAP Training 11th - Unified Fairness Framework (Contrastive + GAN + Quantile)",
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
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr_g", type=float, default=1e-4, help="generator learning rate")
    parser.add_argument("--lr_d", type=float, default=5e-5, help="discriminator learning rate (낮춤)")
    
    # ===== Epsilon (고정값) =====
    parser.add_argument("--epsilon", type=float, default=0.01, help="perturbation bound (고정)")
    
    # ===== Detection Loss 가중치 (고정값) =====
    parser.add_argument("--beta", type=float, default=0.3, help="detection loss weight (epsilon 작으므로 낮춤)")
    
    # ===== Discriminator 설정 =====
    parser.add_argument("--k_d", type=int, default=3, help="discriminator steps per iteration")
    
    # ===== Contrastive Fairness 설정 (Contrastive 1st 핵심) =====
    parser.add_argument(
        "--lambda_contrast",
        type=float,
        default=1.0,
        help="weight for contrastive fairness loss (핵심 - AR Gap 감소)",
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
        default=0.5,
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
    
    # ===== Adversarial Fairness 설정 (GD 7th 핵심) =====
    parser.add_argument(
        "--lambda_fair",
        type=float,
        default=1.5,
        help="weight for adversarial fairness loss",
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="entropy weight for fairness")
    parser.add_argument(
        "--fair_f_scale",
        type=float,
        default=1.0,
        help="female fairness scaling factor (GD 7th 핵심)",
    )
    parser.add_argument(
        "--fair_m_scale",
        type=float,
        default=0.4,
        help="male fairness scaling factor (비대칭 - GD 7th 핵심)",
    )
    
    # ===== Distribution Alignment 설정 =====
    parser.add_argument(
        "--lambda_w",
        type=float,
        default=0.3,
        help="weight for Wasserstein alignment (단방향: F→M)",
    )
    parser.add_argument(
        "--lambda_q",
        type=float,
        default=0.25,
        help="weight for quantile matching loss (GD 10th 핵심)",
    )
    parser.add_argument(
        "--num_quantiles",
        type=int,
        default=5,
        help="number of quantile levels for matching",
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
# Projection Head (Contrastive 1st에서 가져옴)
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
            nn.GELU(),  # ReLU 대신 GELU 사용 (더 부드러운 활성화)
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


def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Entropy 손실: 판별기 출력의 불확실성 최대화"""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    return -(probs * log_probs).sum(dim=-1).mean()


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
    """Cross-Gender Contrastive Loss (Contrastive 1st 핵심)
    
    핵심 아이디어:
    - 다른 성별 샘플을 positive pair로 취급
    - 성별 정보가 특징에서 제거되도록 유도
    - AR Gap 감소에 매우 효과적
    
    Args:
        proj_f: 여성 샘플의 투영된 특징 (N_f, proj_dim), L2 정규화됨
        proj_m: 남성 샘플의 투영된 특징 (N_m, proj_dim), L2 정규화됨
        temperature: softmax temperature (낮을수록 sharp)
    
    Returns:
        Contrastive loss (여성→남성 + 남성→여성 평균)
    """
    if proj_f.size(0) == 0 or proj_m.size(0) == 0:
        return proj_f.new_tensor(0.0)
    
    n_f, n_m = proj_f.size(0), proj_m.size(0)
    
    # 모든 샘플 간 유사도 행렬 (코사인 유사도, 이미 L2 정규화됨)
    sim_f_to_m = torch.mm(proj_f, proj_m.t()) / temperature  # (N_f, N_m)
    sim_m_to_f = sim_f_to_m.t()  # (N_m, N_f)
    
    # 여성→남성: 각 여성이 모든 남성과 유사해지도록
    # log_softmax 기반 손실 (모든 남성을 동등하게 positive로)
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
    """Feature Alignment Loss: 성별 간 특징 분포 정렬 (Contrastive 1st)
    
    1. Mean Alignment: 평균 특징 벡터를 일치시킴
    2. Variance Alignment: 분산을 일치시킴 (분포 형태 유사화)
    
    Args:
        feat_f: 여성 샘플의 특징 (N_f, num_queries, feature_dim)
        feat_m: 남성 샘플의 특징 (N_m, num_queries, feature_dim)
    
    Returns:
        (mean_loss, var_loss)
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
    """단방향 Wasserstein 손실 (GD 7th 핵심)
    
    여성 score가 남성보다 낮을 때만 패널티.
    이렇게 하면 남성 성능은 유지하면서 여성만 향상시킴.
    
    AP Gap 감소에 효과적 (GD 7th에서 유일하게 AP Gap 감소 달성)
    """
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0, device=female_scores.device)
    
    sorted_f = female_scores.sort().values
    sorted_m = male_scores.detach().sort().values  # 남성은 detach로 타겟 역할 (보호)
    
    k = max(sorted_f.numel(), sorted_m.numel())
    sorted_f = _resize_sorted(sorted_f, k)
    sorted_m = _resize_sorted(sorted_m, k)
    
    # 단방향: 여성이 남성보다 낮을 때만 손실 발생 (여성 → 남성 방향 이동)
    return F.relu(sorted_m - sorted_f).mean()


def _quantile_matching_loss(
    source_scores: torch.Tensor,
    target_scores: torch.Tensor,
    num_quantiles: int = 5
) -> torch.Tensor:
    """Quantile Matching Loss (GD 10th 핵심)
    
    특정 분위수에서의 차이를 직접 최소화.
    높은 분위수에 더 높은 가중치 부여 → AP 개선에 효과적.
    
    Wasserstein보다 해석 가능하고, AP 계산에 중요한 영역 집중.
    
    Args:
        source_scores: 정렬 대상 그룹의 detection confidence scores (Female)
        target_scores: 타겟 그룹의 detection confidence scores (Male)
        num_quantiles: 사용할 분위수 개수 (기본 5 = [0.1, 0.3, 0.5, 0.7, 0.9])
    
    Returns:
        단방향 quantile 차이 손실
    """
    if source_scores.numel() < num_quantiles or target_scores.numel() < num_quantiles:
        return source_scores.new_tensor(0.0, device=source_scores.device)
    
    # 분위수 레벨 생성 (예: [0.167, 0.333, 0.5, 0.667, 0.833] for num_quantiles=5)
    quantile_levels = torch.linspace(
        1.0 / (num_quantiles + 1),
        num_quantiles / (num_quantiles + 1),
        num_quantiles,
        device=source_scores.device
    )
    
    # 각 그룹의 분위수 계산
    q_source = torch.quantile(source_scores, quantile_levels)
    q_target = torch.quantile(target_scores.detach(), quantile_levels)  # target은 detach
    
    # 단방향 손실: source 분위수가 target보다 낮을 때만 패널티
    # 높은 분위수(예: 70%, 90%)에 더 높은 가중치 부여 → AP 개선에 효과적
    weights = quantile_levels  # [0.17, 0.33, 0.5, 0.67, 0.83] - 높은 분위수에 높은 가중치
    weights = weights / weights.sum()  # 정규화
    weighted_diff = weights * F.relu(q_target - q_source)
    
    return weighted_diff.sum()


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

    # ===== 모델 초기화 =====
    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)
    
    # Discriminator (GD 7th 방식)
    discriminator = GenderDiscriminator(feature_dim=detr.hidden_dim).to(device)
    
    # Projection Head (Contrastive 1st 방식)
    proj_head = ProjectionHead(
        input_dim=detr.hidden_dim,
        hidden_dim=detr.hidden_dim,
        output_dim=args.proj_dim,
    ).to(device)
    
    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)
        discriminator = DDP(discriminator, device_ids=[args.gpu] if args.gpu is not None else None)
        proj_head = DDP(proj_head, device_ids=[args.gpu] if args.gpu is not None else None)

    # Generator + Projection Head를 함께 최적화
    g_params = list(_unwrap_ddp(generator).parameters()) + list(_unwrap_ddp(proj_head).parameters())
    opt_g = torch.optim.Adam(g_params, lr=args.lr_g)
    opt_d = torch.optim.Adam(_unwrap_ddp(discriminator).parameters(), lr=args.lr_d)

    start_epoch = 0
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if "generator" in ckpt:
            _unwrap_ddp(generator).load_state_dict(ckpt["generator"])
        if "discriminator" in ckpt:
            _unwrap_ddp(discriminator).load_state_dict(ckpt["discriminator"])
        if "proj_head" in ckpt:
            _unwrap_ddp(proj_head).load_state_dict(ckpt["proj_head"])
        if "opt_g" in ckpt:
            opt_g.load_state_dict(ckpt["opt_g"])
        if "opt_d" in ckpt:
            opt_d.load_state_dict(ckpt["opt_d"])
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
        print("FAAP Training 11th - Unified Fairness Framework")
        print("=" * 70)
        print(f"Epsilon (fixed): {args.epsilon}")
        print(f"Beta (detection): {args.beta}")
        print(f"Lambda contrast: {args.lambda_contrast}")
        print(f"Lambda align: {args.lambda_align}")
        print(f"Lambda var: {args.lambda_var}")
        print(f"Lambda fair: {args.lambda_fair}")
        print(f"Lambda W (Wasserstein): {args.lambda_w}")
        print(f"Lambda Q (Quantile): {args.lambda_q}")
        print(f"Fair F scale: {args.fair_f_scale}, Fair M scale: {args.fair_m_scale}")
        print(f"Temperature: {args.temperature}")
        print("=" * 70)

    # ===== 학습 루프 =====
    for epoch in range(start_epoch, args.epochs):
        metrics_logger = utils.MetricLogger(delimiter="  ")
        generator.train()
        discriminator.train()
        proj_head.train()
        
        if args.distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        
        # 고정값 사용 (스케줄 없음)
        current_eps = args.epsilon
        current_beta = args.beta
        _set_generator_epsilon(generator, current_eps)
        current_lr_g = opt_g.param_groups[0]['lr']
        current_lr_d = opt_d.param_groups[0]['lr']

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
            quantile_loss = torch.tensor(0.0, device=device)
            d_loss = torch.tensor(0.0, device=device)

            # ===== Discriminator 업데이트 (GD 7th 방식) =====
            d_losses_list = []
            for _ in range(args.k_d):
                opt_d.zero_grad()
                d_batch_losses = []
                
                if female_batch is not None:
                    with torch.no_grad():
                        female_perturbed_d = _apply_generator(generator, female_batch)
                        _, feat_f_d = detr.forward_with_features(female_perturbed_d)
                    logits_f = discriminator(feat_f_d.detach())
                    labels_f = torch.ones(logits_f.size(0), device=device, dtype=torch.long)
                    d_batch_losses.append(F.cross_entropy(logits_f, labels_f))
                
                if male_batch is not None:
                    with torch.no_grad():
                        male_perturbed_d = _apply_generator(generator, male_batch)
                        _, feat_m_d = detr.forward_with_features(male_perturbed_d)
                    logits_m = discriminator(feat_m_d.detach())
                    labels_m = torch.zeros(logits_m.size(0), device=device, dtype=torch.long)
                    d_batch_losses.append(F.cross_entropy(logits_m, labels_m))

                if d_batch_losses:
                    d_loss_step = torch.stack(d_batch_losses).mean()
                    d_loss_step.backward()
                    opt_d.step()
                    d_losses_list.append(d_loss_step.item())
            
            d_loss = torch.tensor(sum(d_losses_list) / len(d_losses_list) if d_losses_list else 0.0, device=device)

            # ===== Generator 업데이트 (Unified Framework) =====
            if female_batch is not None or male_batch is not None:
                opt_g.zero_grad()
                
                # Discriminator gradient 비활성화 (G 업데이트 시 D weights 수정 방지)
                for p in discriminator.parameters():
                    p.requires_grad = False
                
                # 손실들을 리스트로 수집 (inplace 연산 방지)
                det_losses = []
                fairness_f = torch.tensor(0.0, device=device)
                fairness_m = torch.tensor(0.0, device=device)
                feat_f, feat_m = None, None
                proj_f, proj_m = None, None
                female_scores = torch.tensor([], device=device)
                male_scores = torch.tensor([], device=device)
                
                # 여성 배치 처리
                if female_batch is not None:
                    female_perturbed = _apply_generator(generator, female_batch)
                    outputs_f, feat_f = detr.forward_with_features(female_perturbed)
                    proj_f = proj_head(feat_f)
                    
                    # Adversarial Fairness (GD 7th) - clone logits to avoid inplace issues
                    logits_f = discriminator(feat_f).clone()
                    ce_f = F.cross_entropy(logits_f, torch.ones(logits_f.size(0), device=device, dtype=torch.long))
                    ent_f = _entropy_loss(logits_f)
                    fairness_f = -(ce_f + args.alpha * ent_f)
                    
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
                    
                    # Adversarial Fairness (GD 7th) - clone logits to avoid inplace issues
                    logits_m = discriminator(feat_m).clone()
                    ce_m = F.cross_entropy(logits_m, torch.zeros(logits_m.size(0), device=device, dtype=torch.long))
                    ent_m = _entropy_loss(logits_m)
                    fairness_m = -(ce_m + args.alpha * ent_m)
                    
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
                
                # ===== Contrastive Fairness Losses (Contrastive 1st 핵심) =====
                if proj_f is not None and proj_m is not None:
                    # 1. Cross-Gender Contrastive Loss (AR Gap 감소에 핵심)
                    contrast_loss = _cross_gender_contrastive_loss(
                        proj_f, proj_m, args.temperature
                    )
                    
                    # 2. Feature Alignment Loss
                    align_loss, var_loss = _feature_alignment_loss(feat_f, feat_m)
                
                # ===== Distribution Alignment Losses =====
                if female_scores.numel() > 0 and male_scores.numel() > 0:
                    # 3. Wasserstein Loss (GD 7th - 단방향, AP Gap 감소에 핵심)
                    wasserstein_loss = _wasserstein_1d(female_scores, male_scores)
                    
                    # 4. Quantile Matching Loss (GD 10th - 정밀 분포 정렬)
                    quantile_loss = _quantile_matching_loss(
                        female_scores, male_scores, args.num_quantiles
                    )
                
                # 비대칭 Fairness (GD 7th 핵심 - AP Gap 감소에 기여)
                fairness_loss = args.fair_f_scale * fairness_f + args.fair_m_scale * fairness_m
                
                # ===== 최종 손실 (Unified) =====
                total_g = (
                    current_beta * det_loss                       # Detection 보호
                    + args.lambda_contrast * contrast_loss        # Contrastive (AR Gap)
                    + args.lambda_align * align_loss              # Mean Alignment
                    + args.lambda_var * var_loss                  # Variance Alignment
                    + args.lambda_fair * fairness_loss            # Adversarial Fairness
                    + args.lambda_w * wasserstein_loss            # Wasserstein (AP Gap)
                    + args.lambda_q * quantile_loss               # Quantile (AP Gap 정밀)
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

                # Fairness loss 계산
                fairness_loss = args.fair_f_scale * fairness_f + args.fair_m_scale * fairness_m
                
                total_g.backward()
                if args.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.max_norm)
                    torch.nn.utils.clip_grad_norm_(proj_head.parameters(), args.max_norm)
                opt_g.step()
                
                # Discriminator gradient 다시 활성화
                for p in discriminator.parameters():
                    p.requires_grad = True
            else:
                det_loss = torch.tensor(0.0, device=device)
                fairness_loss = torch.tensor(0.0, device=device)
                total_g = torch.tensor(0.0, device=device)

            # 메트릭 업데이트
            metrics_logger.update(
                d_loss=d_loss.item(),
                g_contrast=contrast_loss.item(),
                g_align=align_loss.item(),
                g_var=var_loss.item(),
                g_fair=fairness_loss.item(),
                g_w=wasserstein_loss.item(),
                g_q=quantile_loss.item(),
                g_det=det_loss.item(),
                g_total=total_g.item(),
                eps=current_eps,
                beta=current_beta,
                lr_g=current_lr_g,
                lr_d=current_lr_d,
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                obj_score=obj_mean.item(),
                obj_frac=obj_frac.item(),
                obj_score_f=obj_mean_f.item(),
                obj_frac_f=obj_frac_f.item(),
                obj_score_m=obj_mean_m.item(),
                obj_frac_m=obj_frac_m.item(),
            )

        metrics_logger.synchronize_between_processes()

        # 에폭 종료 로깅 및 저장
        if utils.is_main_process():
            log_entry = {
                "epoch": epoch,
                "d_loss": metrics_logger.meters["d_loss"].global_avg,
                "g_contrast": metrics_logger.meters["g_contrast"].global_avg,
                "g_align": metrics_logger.meters["g_align"].global_avg,
                "g_var": metrics_logger.meters["g_var"].global_avg,
                "g_fair": metrics_logger.meters["g_fair"].global_avg,
                "g_w": metrics_logger.meters["g_w"].global_avg,
                "g_q": metrics_logger.meters["g_q"].global_avg,
                "g_det": metrics_logger.meters["g_det"].global_avg,
                "g_total": metrics_logger.meters["g_total"].global_avg,
                "epsilon": current_eps,
                "beta": current_beta,
                "lr_g": current_lr_g,
                "lr_d": current_lr_d,
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
                        "discriminator": _unwrap_ddp(discriminator).state_dict(),
                        "proj_head": _unwrap_ddp(proj_head).state_dict(),
                        "opt_g": opt_g.state_dict(),
                        "opt_d": opt_d.state_dict(),
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
