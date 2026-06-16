"""
FAAP Training - 13th Version: Adaptive Multi-Scale Contrastive Fairness

================================================================================
                         7th + 11th 핵심 강점 융합 분석
================================================================================

[GD 7th 핵심 성공 요소] - AP Gap -0.38%, AR Gap -60.5%
- 비대칭 Fairness 스케일링: Female 1.0, Male 0.5
- 단방향 Wasserstein: Female→Male만 정렬 (Male 성능 보호)
- Epsilon 스케줄: warmup → hold → cooldown
- 긴 학습 (24 epochs)

[GD 11th 핵심 강점] - Unified Framework
- Contrastive + GAN + Quantile 융합
- Projection Head를 통한 특징 공간 정렬
- Cross-Gender Contrastive Loss (AR Gap 감소)

[Contrastive 1st] - AR Gap -61.73% (최대 감소)
- Discriminator 없이 안정적 학습
- InfoNCE Loss로 성별 정보 제거
- 단순한 손실 함수 구조

================================================================================
                         13th 혁신 설계
================================================================================

1. **Discriminator-Free** (Contrastive 1st 계승):
   - GAN 학습 불안정성 완전 제거
   - Mode collapse 위험 없음
   - 더 큰 배치 사이즈 가능 (메모리 여유)

2. **Multi-Scale Contrastive Learning** (NEW):
   - Query-level: 각 query 특징 정렬
   - Image-level: 이미지 전체 특징 정렬  
   - Batch-level: 배치 통계 정렬
   → 더 정교한 특징 공간 정렬

3. **Asymmetric Contrastive Loss** (7th 아이디어 확장):
   - Female→Male: 강하게 정렬 (weight 1.0)
   - Male→Female: 약하게 정렬 (weight 0.3)
   → Male 성능 보호하면서 Female 향상

4. **Hard Negative Mining** (NEW):
   - 성별 간 가장 멀리 있는 샘플 쌍에 집중
   - 어려운 경우를 먼저 해결 → 전체 성능 향상

5. **Adaptive Loss Weighting** (NEW):
   - Detection Loss vs Fairness Loss 자동 균형
   - Detection 성능이 떨어지면 Detection Loss 가중치 증가
   - 공정성 목표 달성 시 Detection 보호 강화

6. **Curriculum Epsilon Schedule** (7th 계승 + 개선):
   - Warmup: 0.03 → 0.08 (안정적 시작)
   - Peak: 0.08 유지 (최대 학습 효과)
   - Cooldown: 0.08 → 0.06 (Detection 회복)

================================================================================
                         손실 함수 구조
================================================================================

G_Loss = β × Detection_Loss                              # Detection 보호
       + λ_contrast × Multi_Scale_Contrastive            # 다중 스케일 정렬
       + λ_asym × Asymmetric_Alignment                   # 비대칭 정렬
       + λ_hard × Hard_Negative_Mining                   # 어려운 샘플 집중
       + λ_w × Wasserstein_1D (단방향)                    # 분포 정렬

하이퍼파라미터:
- epsilon: 0.03 → 0.08 → 0.06 (Curriculum)
- β: 0.4 → 0.6 (Detection 보호 점진 증가)
- λ_contrast: 1.2 (핵심 - Multi-Scale)
- λ_asym: 0.8 (비대칭 정렬)
- λ_hard: 0.4 (Hard Negative Mining)
- λ_w: 0.3 (Wasserstein)
- temperature: 0.05 (더 sharp한 contrastive)
- proj_dim: 128 (Projection 차원)

================================================================================
"""

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

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
        "FAAP Training 13th - Adaptive Multi-Scale Contrastive Fairness",
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
    parser.add_argument("--batch_size", type=int, default=6)  # Discriminator 없어 메모리 여유
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--lr_g", type=float, default=1e-4, help="generator learning rate")
    
    # ===== Epsilon Schedule (7th 스타일 Curriculum) =====
    parser.add_argument("--epsilon_start", type=float, default=0.03, help="starting epsilon for warmup")
    parser.add_argument("--epsilon_peak", type=float, default=0.08, help="peak epsilon")
    parser.add_argument("--epsilon_final", type=float, default=0.06, help="final epsilon after cooldown")
    parser.add_argument("--warmup_epochs", type=int, default=8, help="epochs to warm up epsilon")
    parser.add_argument("--hold_epochs", type=int, default=8, help="epochs to hold peak epsilon")
    parser.add_argument("--cooldown_epochs", type=int, default=14, help="epochs to cool down epsilon")
    
    # ===== Detection Loss Schedule (7th 스타일) =====
    parser.add_argument("--beta_start", type=float, default=0.4, help="starting detection loss weight")
    parser.add_argument("--beta_final", type=float, default=0.6, help="final detection loss weight")
    
    # ===== Multi-Scale Contrastive (NEW) =====
    parser.add_argument(
        "--lambda_contrast",
        type=float,
        default=1.2,
        help="weight for multi-scale contrastive loss (핵심)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="temperature for contrastive loss (더 sharp)",
    )
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=128,
        help="projection dimension for contrastive learning",
    )
    
    # ===== Asymmetric Alignment (7th 아이디어 확장) =====
    parser.add_argument(
        "--lambda_asym",
        type=float,
        default=0.8,
        help="weight for asymmetric alignment loss",
    )
    parser.add_argument(
        "--asym_f_weight",
        type=float,
        default=1.0,
        help="Female→Male alignment weight (강하게)",
    )
    parser.add_argument(
        "--asym_m_weight",
        type=float,
        default=0.3,
        help="Male→Female alignment weight (약하게, Male 보호)",
    )
    
    # ===== Hard Negative Mining (NEW) =====
    parser.add_argument(
        "--lambda_hard",
        type=float,
        default=0.4,
        help="weight for hard negative mining loss",
    )
    parser.add_argument(
        "--hard_ratio",
        type=float,
        default=0.3,
        help="ratio of hardest samples to focus on",
    )
    
    # ===== Distribution Alignment (7th 계승) =====
    parser.add_argument(
        "--lambda_w",
        type=float,
        default=0.3,
        help="weight for Wasserstein alignment (단방향: F→M)",
    )
    
    # ===== Adaptive Loss Weighting (NEW) =====
    parser.add_argument(
        "--use_adaptive_weights",
        action="store_true",
        default=True,
        help="use adaptive loss weighting based on detection performance",
    )
    parser.add_argument(
        "--detection_threshold",
        type=float,
        default=0.45,
        help="detection score threshold for adaptive weighting",
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
# Schedule Functions (7th 스타일)
# ============================================================

def _scheduled_epsilon(
    epoch: int,
    warmup_epochs: int,
    hold_epochs: int,
    cooldown_epochs: int,
    eps_start: float,
    eps_peak: float,
    eps_final: float,
) -> float:
    """7th 스타일 Epsilon Schedule: warmup → hold → cooldown"""
    warmup_end = warmup_epochs - 1 if warmup_epochs > 1 else 0
    
    # Warmup Phase
    if epoch <= warmup_end:
        progress = min(epoch / max(1, warmup_epochs - 1), 1.0)
        return eps_start + (eps_peak - eps_start) * progress
    
    # Hold Phase
    hold_end = warmup_end + max(0, hold_epochs)
    if epoch <= hold_end:
        return eps_peak
    
    # Cooldown Phase
    if cooldown_epochs <= 0:
        return eps_peak
    
    progress = (epoch - hold_end) / max(1, cooldown_epochs)
    if progress >= 1.0:
        return eps_final
    return eps_peak + (eps_final - eps_peak) * progress


def _scheduled_beta(epoch: int, total_epochs: int, beta_start: float, beta_final: float) -> float:
    """Detection 가중치 선형 증가 (후반부에서 Detection 보호 강화)"""
    if total_epochs <= 1:
        return beta_start
    progress = min(epoch / max(1, total_epochs - 1), 1.0)
    return beta_start + (beta_final - beta_start) * progress


# ============================================================
# Multi-Scale Projection Head (NEW - 13th 핵심)
# ============================================================

class MultiScaleProjectionHead(nn.Module):
    """다중 스케일 Projection Head for Contrastive Learning.
    
    세 가지 스케일의 특징을 추출하여 contrastive loss 계산:
    1. Query-level: 각 query의 개별 특징 (num_queries, proj_dim)
    2. Image-level: 이미지 전체 특징 (1, proj_dim)
    3. Statistical-level: 평균/분산 기반 통계적 특징 (1, proj_dim*2)
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim
        
        # Query-level projection
        self.query_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Image-level projection (pooled features)
        self.image_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Statistical projection (mean + std features)
        self.stat_proj = nn.Sequential(
            nn.LayerNorm(input_dim * 2),  # mean concat std
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, num_queries, feature_dim)
        
        Returns:
            query_feat: (batch, num_queries, proj_dim) - L2 normalized
            image_feat: (batch, proj_dim) - L2 normalized  
            stat_feat: (batch, proj_dim) - L2 normalized
        """
        batch_size, num_queries, feat_dim = x.shape
        
        # Query-level: 각 query 독립적으로 투영
        query_feat = self.query_proj(x)  # (batch, num_queries, proj_dim)
        query_feat = F.normalize(query_feat, dim=-1)
        
        # Image-level: query 평균 후 투영
        pooled = x.mean(dim=1)  # (batch, feat_dim)
        image_feat = self.image_proj(pooled)  # (batch, proj_dim)
        image_feat = F.normalize(image_feat, dim=-1)
        
        # Statistical-level: 평균 + 표준편차 연결 후 투영
        mean_feat = x.mean(dim=1)  # (batch, feat_dim)
        std_feat = x.std(dim=1)    # (batch, feat_dim)
        stat_input = torch.cat([mean_feat, std_feat], dim=-1)  # (batch, feat_dim*2)
        stat_feat = self.stat_proj(stat_input)  # (batch, proj_dim)
        stat_feat = F.normalize(stat_feat, dim=-1)
        
        return query_feat, image_feat, stat_feat


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
# 핵심 손실 함수들 (13th 핵심 혁신)
# ============================================================

def _multi_scale_contrastive_loss(
    query_f: torch.Tensor,
    query_m: torch.Tensor,
    image_f: torch.Tensor,
    image_m: torch.Tensor,
    stat_f: torch.Tensor,
    stat_m: torch.Tensor,
    temperature: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-Scale Contrastive Loss (13th 핵심 혁신)
    
    세 가지 스케일에서 동시에 성별 간 특징 정렬:
    1. Image-level: 이미지 전체 표현 정렬 (가장 중요)
    2. Query-level: 개별 query 특징 정렬
    3. Statistical-level: 분포 통계 정렬
    
    Args:
        query_f, query_m: (batch, num_queries, proj_dim)
        image_f, image_m: (batch, proj_dim)
        stat_f, stat_m: (batch, proj_dim)
        temperature: softmax temperature
    
    Returns:
        (image_loss, query_loss, stat_loss)
    """
    device = image_f.device
    zero = torch.tensor(0.0, device=device)
    
    # ===== Image-level Contrastive (핵심) =====
    if image_f.size(0) > 0 and image_m.size(0) > 0:
        n_f, n_m = image_f.size(0), image_m.size(0)
        
        # 유사도 행렬
        sim_f_to_m = torch.mm(image_f, image_m.t()) / temperature  # (N_f, N_m)
        sim_m_to_f = sim_f_to_m.t()  # (N_m, N_f)
        
        # InfoNCE-style loss: 다른 성별과 유사하게 만듦
        loss_f = -torch.logsumexp(sim_f_to_m, dim=1).mean() + \
                  torch.log(torch.tensor(n_m, dtype=torch.float, device=device))
        loss_m = -torch.logsumexp(sim_m_to_f, dim=1).mean() + \
                  torch.log(torch.tensor(n_f, dtype=torch.float, device=device))
        
        image_loss = (loss_f + loss_m) / 2
    else:
        image_loss = zero
    
    # ===== Query-level Contrastive =====
    if query_f.size(0) > 0 and query_m.size(0) > 0:
        # Query 평균으로 축소 후 비교
        q_f_mean = query_f.mean(dim=1)  # (batch_f, proj_dim)
        q_m_mean = query_m.mean(dim=1)  # (batch_m, proj_dim)
        
        n_f, n_m = q_f_mean.size(0), q_m_mean.size(0)
        sim_q = torch.mm(q_f_mean, q_m_mean.t()) / temperature
        
        loss_qf = -torch.logsumexp(sim_q, dim=1).mean() + \
                   torch.log(torch.tensor(n_m, dtype=torch.float, device=device))
        loss_qm = -torch.logsumexp(sim_q.t(), dim=1).mean() + \
                   torch.log(torch.tensor(n_f, dtype=torch.float, device=device))
        
        query_loss = (loss_qf + loss_qm) / 2
    else:
        query_loss = zero
    
    # ===== Statistical-level Contrastive =====
    if stat_f.size(0) > 0 and stat_m.size(0) > 0:
        # 통계 특징 간 유사도
        n_f, n_m = stat_f.size(0), stat_m.size(0)
        sim_stat = torch.mm(stat_f, stat_m.t()) / temperature
        
        loss_sf = -torch.logsumexp(sim_stat, dim=1).mean() + \
                   torch.log(torch.tensor(n_m, dtype=torch.float, device=device))
        loss_sm = -torch.logsumexp(sim_stat.t(), dim=1).mean() + \
                   torch.log(torch.tensor(n_f, dtype=torch.float, device=device))
        
        stat_loss = (loss_sf + loss_sm) / 2
    else:
        stat_loss = zero
    
    return image_loss, query_loss, stat_loss


def _asymmetric_alignment_loss(
    feat_f: torch.Tensor,
    feat_m: torch.Tensor,
    f_weight: float = 1.0,
    m_weight: float = 0.3,
) -> torch.Tensor:
    """Asymmetric Alignment Loss (7th 비대칭 스케일링 아이디어 확장)
    
    핵심 아이디어:
    - Female → Male 방향: 강하게 정렬 (f_weight = 1.0)
    - Male → Female 방향: 약하게 정렬 (m_weight = 0.3)
    → Male 성능 보호하면서 Female만 집중 개선
    
    Args:
        feat_f: 여성 특징 (N_f, num_queries, feature_dim)
        feat_m: 남성 특징 (N_m, num_queries, feature_dim)
        f_weight: Female → Male 가중치
        m_weight: Male → Female 가중치
    
    Returns:
        비대칭 정렬 손실
    """
    if feat_f.size(0) == 0 or feat_m.size(0) == 0:
        return feat_f.new_tensor(0.0)
    
    # Query 평균으로 풀링
    pooled_f = feat_f.mean(dim=1)  # (N_f, feature_dim)
    pooled_m = feat_m.mean(dim=1)  # (N_m, feature_dim)
    
    # 그룹 평균
    mean_f = pooled_f.mean(dim=0)  # (feature_dim,)
    mean_m = pooled_m.mean(dim=0)  # (feature_dim,)
    
    # 비대칭 손실: Female→Male은 강하게, Male→Female은 약하게
    # Female이 Male 중심으로 이동하도록 유도
    diff = mean_f - mean_m.detach()  # Male은 detach (타겟 역할)
    f_to_m_loss = (diff ** 2).mean()  # Female → Male
    
    diff_rev = mean_m - mean_f.detach()  # Female은 detach
    m_to_f_loss = (diff_rev ** 2).mean()  # Male → Female
    
    return f_weight * f_to_m_loss + m_weight * m_to_f_loss


def _hard_negative_mining_loss(
    image_f: torch.Tensor,
    image_m: torch.Tensor,
    hard_ratio: float = 0.3,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Hard Negative Mining Loss (NEW - 13th)
    
    가장 다른(어려운) 성별 쌍에 집중하여 학습.
    쉬운 샘플은 이미 정렬되어 있으므로, 어려운 샘플에 집중.
    
    Args:
        image_f, image_m: (batch, proj_dim) - L2 normalized
        hard_ratio: 가장 어려운 샘플의 비율 (0.3 = 상위 30%)
        temperature: softmax temperature
    
    Returns:
        Hard negative mining 손실
    """
    if image_f.size(0) == 0 or image_m.size(0) == 0:
        return image_f.new_tensor(0.0)
    
    n_f, n_m = image_f.size(0), image_m.size(0)
    
    # 유사도 계산 (높을수록 유사)
    sim = torch.mm(image_f, image_m.t())  # (N_f, N_m)
    
    # Hard negatives: 유사도가 가장 낮은 쌍 (가장 다른 샘플)
    # Female 관점에서: 각 female에 대해 가장 다른 male 찾기
    k_m = max(1, int(n_m * hard_ratio))
    _, hard_m_idx = sim.topk(k_m, dim=1, largest=False)  # 가장 낮은 유사도
    
    # Male 관점에서: 각 male에 대해 가장 다른 female 찾기
    k_f = max(1, int(n_f * hard_ratio))
    _, hard_f_idx = sim.t().topk(k_f, dim=1, largest=False)
    
    # Hard pairs에 대해 contrastive loss 강화
    # Female → hardest Males
    hard_sim_f = torch.gather(sim, 1, hard_m_idx) / temperature
    loss_f = -torch.logsumexp(hard_sim_f, dim=1).mean() + \
              torch.log(torch.tensor(k_m, dtype=torch.float, device=image_f.device))
    
    # Male → hardest Females  
    hard_sim_m = torch.gather(sim.t(), 1, hard_f_idx) / temperature
    loss_m = -torch.logsumexp(hard_sim_m, dim=1).mean() + \
              torch.log(torch.tensor(k_f, dtype=torch.float, device=image_f.device))
    
    return (loss_f + loss_m) / 2


def _wasserstein_1d(female_scores: torch.Tensor, male_scores: torch.Tensor) -> torch.Tensor:
    """단방향 Wasserstein 손실 (7th 핵심)
    
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


def _adaptive_weight(
    current_score: float,
    threshold: float,
    base_weight: float,
    scale: float = 2.0,
) -> float:
    """Adaptive Loss Weighting (NEW - 13th)
    
    Detection 성능이 threshold 아래로 떨어지면 detection 가중치 증가.
    
    Args:
        current_score: 현재 detection score (obj_mean)
        threshold: 목표 threshold
        base_weight: 기본 가중치
        scale: 스케일링 팩터
    
    Returns:
        조정된 가중치
    """
    if current_score >= threshold:
        return base_weight
    
    # 성능이 떨어지면 가중치 증가 (최대 scale배)
    ratio = (threshold - current_score) / threshold
    return base_weight * (1 + ratio * (scale - 1))


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
    generator = PerturbationGenerator(epsilon=args.epsilon_start).to(device)
    
    # Multi-Scale Projection Head (13th 핵심)
    proj_head = MultiScaleProjectionHead(
        input_dim=detr.hidden_dim,
        hidden_dim=detr.hidden_dim,
        output_dim=args.proj_dim,
    ).to(device)
    
    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)
        proj_head = DDP(proj_head, device_ids=[args.gpu] if args.gpu is not None else None)

    # Generator + Projection Head만 최적화 (Discriminator 없음)
    g_params = list(_unwrap_ddp(generator).parameters()) + list(_unwrap_ddp(proj_head).parameters())
    opt_g = torch.optim.AdamW(g_params, lr=args.lr_g, weight_decay=1e-4)
    
    # Learning rate scheduler (CosineAnnealing with Warmup)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_g, T_0=10, T_mult=2, eta_min=1e-6
    )

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
        print("=" * 80)
        print("FAAP Training 13th - Adaptive Multi-Scale Contrastive Fairness")
        print("=" * 80)
        print(f"Epsilon Schedule: {args.epsilon_start} → {args.epsilon_peak} → {args.epsilon_final}")
        print(f"  Warmup: {args.warmup_epochs} epochs")
        print(f"  Hold: {args.hold_epochs} epochs")
        print(f"  Cooldown: {args.cooldown_epochs} epochs")
        print(f"Beta Schedule: {args.beta_start} → {args.beta_final}")
        print(f"Lambda contrast (Multi-Scale): {args.lambda_contrast}")
        print(f"Lambda asym (Asymmetric): {args.lambda_asym}")
        print(f"  Female→Male weight: {args.asym_f_weight}")
        print(f"  Male→Female weight: {args.asym_m_weight}")
        print(f"Lambda hard (Hard Mining): {args.lambda_hard}")
        print(f"Lambda W (Wasserstein): {args.lambda_w}")
        print(f"Temperature: {args.temperature}")
        print(f"Adaptive Weights: {args.use_adaptive_weights}")
        print(f"NO Discriminator - Pure Multi-Scale Contrastive Learning!")
        print("=" * 80)

    # Running average for adaptive weighting
    running_obj_score = 0.5  # 초기값

    # ===== 학습 루프 =====
    for epoch in range(start_epoch, args.epochs):
        metrics_logger = utils.MetricLogger(delimiter="  ")
        generator.train()
        proj_head.train()
        
        if args.distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        
        # Schedule 계산
        current_eps = _scheduled_epsilon(
            epoch,
            args.warmup_epochs,
            args.hold_epochs,
            args.cooldown_epochs,
            args.epsilon_start,
            args.epsilon_peak,
            args.epsilon_final,
        )
        current_beta = _scheduled_beta(epoch, args.epochs, args.beta_start, args.beta_final)
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
            
            # 손실 초기화
            image_contrast = torch.tensor(0.0, device=device)
            query_contrast = torch.tensor(0.0, device=device)
            stat_contrast = torch.tensor(0.0, device=device)
            asym_loss = torch.tensor(0.0, device=device)
            hard_loss = torch.tensor(0.0, device=device)
            wasserstein_loss = torch.tensor(0.0, device=device)

            # ===== Generator 업데이트 (Discriminator-Free!) =====
            if female_batch is not None or male_batch is not None:
                opt_g.zero_grad()
                
                det_losses = []
                feat_f, feat_m = None, None
                query_f, query_m = None, None
                image_f, image_m = None, None
                stat_f, stat_m = None, None
                female_scores = torch.tensor([], device=device)
                male_scores = torch.tensor([], device=device)
                
                # 여성 배치 처리
                if female_batch is not None:
                    female_perturbed = _apply_generator(generator, female_batch)
                    outputs_f, feat_f = detr.forward_with_features(female_perturbed)
                    query_f, image_f, stat_f = proj_head(feat_f)
                    
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
                    query_m, image_m, stat_m = proj_head(feat_m)
                    
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
                
                # ===== Multi-Scale Contrastive Loss (13th 핵심) =====
                if image_f is not None and image_m is not None:
                    image_contrast, query_contrast, stat_contrast = _multi_scale_contrastive_loss(
                        query_f, query_m,
                        image_f, image_m,
                        stat_f, stat_m,
                        args.temperature,
                    )
                
                # ===== Asymmetric Alignment Loss (7th 아이디어 확장) =====
                if feat_f is not None and feat_m is not None:
                    asym_loss = _asymmetric_alignment_loss(
                        feat_f, feat_m,
                        args.asym_f_weight, args.asym_m_weight,
                    )
                
                # ===== Hard Negative Mining Loss (NEW) =====
                if image_f is not None and image_m is not None:
                    hard_loss = _hard_negative_mining_loss(
                        image_f, image_m,
                        args.hard_ratio, args.temperature,
                    )
                
                # ===== Wasserstein Loss (7th 핵심) =====
                if female_scores.numel() > 0 and male_scores.numel() > 0:
                    wasserstein_loss = _wasserstein_1d(female_scores, male_scores)
                
                # ===== Adaptive Weight 계산 (NEW) =====
                if args.use_adaptive_weights:
                    adaptive_beta = _adaptive_weight(
                        running_obj_score,
                        args.detection_threshold,
                        current_beta,
                        scale=1.5,
                    )
                else:
                    adaptive_beta = current_beta
                
                # ===== Multi-Scale Contrastive 합산 =====
                # Image-level이 가장 중요 (0.5), Query-level (0.3), Stat-level (0.2)
                contrast_total = 0.5 * image_contrast + 0.3 * query_contrast + 0.2 * stat_contrast
                
                # ===== 최종 손실 =====
                total_g = (
                    adaptive_beta * det_loss                      # Detection 보호
                    + args.lambda_contrast * contrast_total       # Multi-Scale Contrastive
                    + args.lambda_asym * asym_loss                # Asymmetric Alignment
                    + args.lambda_hard * hard_loss                # Hard Negative Mining
                    + args.lambda_w * wasserstein_loss            # Wasserstein (단방향)
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
                    
                    # Running average 업데이트
                    running_obj_score = 0.9 * running_obj_score + 0.1 * obj_mean.item()

                total_g.backward()
                if args.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.max_norm)
                    torch.nn.utils.clip_grad_norm_(proj_head.parameters(), args.max_norm)
                opt_g.step()
            else:
                det_loss = torch.tensor(0.0, device=device)
                total_g = torch.tensor(0.0, device=device)
                adaptive_beta = current_beta
                contrast_total = torch.tensor(0.0, device=device)

            # 메트릭 업데이트
            metrics_logger.update(
                g_img_con=image_contrast.item(),
                g_qry_con=query_contrast.item(),
                g_stat_con=stat_contrast.item(),
                g_asym=asym_loss.item(),
                g_hard=hard_loss.item(),
                g_w=wasserstein_loss.item(),
                g_det=det_loss.item(),
                g_total=total_g.item(),
                eps=current_eps,
                beta=current_beta,
                adaptive_beta=adaptive_beta,
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
                "g_img_con": metrics_logger.meters["g_img_con"].global_avg,
                "g_qry_con": metrics_logger.meters["g_qry_con"].global_avg,
                "g_stat_con": metrics_logger.meters["g_stat_con"].global_avg,
                "g_asym": metrics_logger.meters["g_asym"].global_avg,
                "g_hard": metrics_logger.meters["g_hard"].global_avg,
                "g_w": metrics_logger.meters["g_w"].global_avg,
                "g_det": metrics_logger.meters["g_det"].global_avg,
                "g_total": metrics_logger.meters["g_total"].global_avg,
                "epsilon": current_eps,
                "beta": current_beta,
                "adaptive_beta": metrics_logger.meters["adaptive_beta"].global_avg,
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
            
            print(f"Epoch {epoch} | eps={current_eps:.4f} | beta={current_beta:.3f} | "
                  f"det={metrics_logger.meters['g_det'].global_avg:.4f} | "
                  f"contrast={metrics_logger.meters['g_img_con'].global_avg:.4f} | "
                  f"obj_f={metrics_logger.meters['obj_score_f'].global_avg:.4f} | "
                  f"obj_m={metrics_logger.meters['obj_score_m'].global_avg:.4f}")

        if args.distributed:
            dist.barrier()

    if utils.is_main_process():
        print("=" * 80)
        print("Training complete!")
        print("=" * 80)


if __name__ == "__main__":
    main()
