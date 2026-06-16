"""
FAAP Training - Contrastive 3rd Version: Asymmetric Contrastive Fairness

핵심 변경점 (1st 버전 대비):
1. 비대칭 Contrastive Loss 도입:
   - 여성→남성 방향: 가중치 1.5 (더 강하게)
   - 남성→여성 방향: 가중치 0.5 (약하게)
   - 7th WGAN-GD의 fair_f_scale/fair_m_scale 아이디어 차용

2. 설계 근거:
   - 편향 문제: 여성 탐지 성능이 남성보다 낮음
   - 해결 방향: 여성 특징을 남성 방향으로 더 강하게 이동
   - 대칭 처리는 양쪽을 동등하게 이동 → 비효율적
   - 비대칭 처리는 문제가 있는 그룹에 집중 → 효율적

3. 수학적 변경:
   - 1st: L_contrast = (L_f→m + L_m→f) / 2
   - 3rd: L_contrast = 1.5 * L_f→m + 0.5 * L_m→f

4. 기대 효과:
   - 여성 특징이 남성 특징 공간으로 더 빠르게 이동
   - 남성 특징은 상대적으로 안정 유지
   - 7th WGAN-GD의 성능 장점을 Contrastive 방식으로 구현
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
    for prefix in ("train_faap_contrastive_", "train_faap_", "train_"):
        if stem_lower.startswith(prefix):
            suffix = stem[len(prefix):]
            break
    suffix = suffix.lower()
    return str(Path("faap_outputs") / f"faap_outputs_{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "FAAP-style training for DETR (Contrastive 3rd - Asymmetric)",
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
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--lr_g", type=float, default=1e-4, help="generator learning rate")
    
    # ===== Epsilon (고정값) =====
    parser.add_argument("--epsilon", type=float, default=0.08, help="perturbation bound (fixed)")
    
    # ===== Detection Loss 가중치 (고정값) =====
    parser.add_argument("--beta", type=float, default=0.6, help="detection loss weight (fixed)")
    
    # ===== Contrastive Fairness 설정 =====
    parser.add_argument(
        "--lambda_contrast",
        type=float,
        default=1.0,
        help="weight for contrastive fairness loss",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="temperature for contrastive loss (lower = sharper)",
    )
    parser.add_argument(
        "--lambda_align",
        type=float,
        default=0.5,
        help="weight for feature alignment loss (mean matching)",
    )
    parser.add_argument(
        "--lambda_var",
        type=float,
        default=0.1,
        help="weight for variance matching loss",
    )
    
    # ===== 3rd 신규: 비대칭 가중치 =====
    parser.add_argument(
        "--contrast_f_scale",
        type=float,
        default=1.5,
        help="weight for female→male contrastive direction (3rd: 1.5, 1st: 1.0)",
    )
    parser.add_argument(
        "--contrast_m_scale",
        type=float,
        default=0.5,
        help="weight for male→female contrastive direction (3rd: 0.5, 1st: 1.0)",
    )
    
    # ===== Score Distribution Alignment =====
    parser.add_argument(
        "--lambda_score",
        type=float,
        default=0.3,
        help="weight for detection score alignment",
    )
    
    # ===== Projection Head 설정 =====
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=128,
        help="projection dimension for contrastive learning",
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


class ProjectionHead(nn.Module):
    """Contrastive Learning을 위한 Projection Head.
    
    DETR 특징을 저차원 공간으로 투영하여 contrastive loss 계산.
    SimCLR 스타일의 2-layer MLP.
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_queries, feature_dim)
        pooled = x.mean(dim=1)  # (batch, feature_dim)
        return F.normalize(self.net(pooled), dim=-1)  # L2 정규화


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


# ============================================================
# Asymmetric Contrastive Fairness Loss (3rd 핵심 변경)
# ============================================================

def _asymmetric_cross_gender_contrastive_loss(
    proj_f: torch.Tensor,
    proj_m: torch.Tensor,
    temperature: float = 0.1,
    f_scale: float = 1.5,
    m_scale: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric Cross-Gender Contrastive Loss (3rd 핵심).
    
    핵심 변경 (1st 대비):
    - 1st: (loss_f_to_m + loss_m_to_f) / 2  (대칭)
    - 3rd: f_scale * loss_f_to_m + m_scale * loss_m_to_f  (비대칭)
    
    설계 근거:
    - 여성 탐지 성능이 낮음 → 여성 특징을 남성 방향으로 더 강하게 이동
    - f_scale=1.5: 여성→남성 방향 강화
    - m_scale=0.5: 남성→여성 방향 약화
    
    Args:
        proj_f: 여성 샘플의 투영된 특징 (N_f, proj_dim), L2 정규화됨
        proj_m: 남성 샘플의 투영된 특징 (N_m, proj_dim), L2 정규화됨
        temperature: softmax temperature (낮을수록 sharp)
        f_scale: 여성→남성 방향 가중치 (기본: 1.5)
        m_scale: 남성→여성 방향 가중치 (기본: 0.5)
    
    Returns:
        (total_loss, loss_f_to_m, loss_m_to_f) - 총 손실과 개별 손실
    """
    if proj_f.size(0) == 0 or proj_m.size(0) == 0:
        zero = proj_f.new_tensor(0.0)
        return zero, zero, zero
    
    # 모든 샘플 간 유사도 행렬
    sim_f_to_m = torch.mm(proj_f, proj_m.t()) / temperature  # (N_f, N_m)
    sim_m_to_f = sim_f_to_m.t()  # (N_m, N_f)
    
    n_f, n_m = proj_f.size(0), proj_m.size(0)
    
    # 여성→남성: 각 여성이 남성 전체와 유사해지도록 (강화됨)
    loss_f_to_m = -torch.logsumexp(sim_f_to_m, dim=1).mean() + torch.log(torch.tensor(n_m, dtype=torch.float, device=proj_f.device))
    
    # 남성→여성: 각 남성이 여성 전체와 유사해지도록 (약화됨)
    loss_m_to_f = -torch.logsumexp(sim_m_to_f, dim=1).mean() + torch.log(torch.tensor(n_f, dtype=torch.float, device=proj_f.device))
    
    # 비대칭 가중치 적용 (3rd 핵심)
    # 1st: (loss_f_to_m + loss_m_to_f) / 2
    # 3rd: f_scale * loss_f_to_m + m_scale * loss_m_to_f
    total_loss = f_scale * loss_f_to_m + m_scale * loss_m_to_f
    
    return total_loss, loss_f_to_m, loss_m_to_f


def _feature_alignment_loss(
    feat_f: torch.Tensor,
    feat_m: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Feature Alignment Loss: 성별 간 특징 분포 정렬.
    
    1. Mean Alignment: 평균 특징 벡터를 일치시킴
    2. Variance Alignment: 분산을 일치시킴 (분포 형태 유사화)
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


def _score_distribution_loss(
    female_scores: torch.Tensor,
    male_scores: torch.Tensor,
) -> torch.Tensor:
    """Score Distribution Alignment.
    
    Detection confidence score 분포를 정렬.
    단방향: 여성 score가 남성보다 낮을 때만 패널티.
    """
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0)
    
    # 평균 score 차이
    mean_f = female_scores.mean()
    mean_m = male_scores.detach().mean()
    
    # 단방향: 여성이 남성보다 낮을 때만 손실
    gap_loss = F.relu(mean_m - mean_f)
    
    # 분위수 정렬 (간소화 버전)
    if female_scores.numel() >= 3 and male_scores.numel() >= 3:
        q_levels = torch.tensor([0.25, 0.5, 0.75], device=female_scores.device)
        q_f = torch.quantile(female_scores, q_levels)
        q_m = torch.quantile(male_scores.detach(), q_levels)
        quantile_loss = F.relu(q_m - q_f).mean()
        return gap_loss + quantile_loss
    
    return gap_loss


def _matched_detection_scores(detr: FrozenDETR, outputs: dict, targets: Sequence[dict]) -> torch.Tensor:
    """Hungarian matching을 통해 매칭된 detection score 추출"""
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
    if args.distributed:
        dist.barrier()

    if utils.is_main_process():
        dataset_info = inspect_faap_dataset(Path(args.dataset_root))
        with (output_dir / "dataset_layout.json").open("w") as f:
            json.dump(dataset_info, f, indent=2)
        # 3rd: 비대칭 가중치 로깅
        print(f"[3rd] Asymmetric Contrastive: f_scale={args.contrast_f_scale}, m_scale={args.contrast_m_scale}")

    # ===== 모델 초기화 =====
    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)
    
    proj_head = ProjectionHead(
        input_dim=detr.hidden_dim,
        hidden_dim=detr.hidden_dim,
        output_dim=args.proj_dim,
    ).to(device)
    
    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)
        proj_head = DDP(proj_head, device_ids=[args.gpu] if args.gpu is not None else None)

    # Generator + Projection Head를 함께 최적화
    params = list(_unwrap_ddp(generator).parameters()) + list(_unwrap_ddp(proj_head).parameters())
    opt_g = torch.optim.Adam(params, lr=args.lr_g)

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
    
    for epoch in range(start_epoch, args.epochs):
        metrics_logger = utils.MetricLogger(delimiter="  ")
        generator.train()
        proj_head.train()
        
        if args.distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        
        # 고정값 사용
        current_eps = args.epsilon
        current_beta = args.beta
        _set_generator_epsilon(generator, current_eps)
        current_lr_g = opt_g.param_groups[0]['lr']

        for samples, targets, genders in metrics_logger.log_every(train_loader, args.log_every, f"Epoch {epoch}"):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            genders = [g.lower() for g in genders]

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
            contrast_f2m = torch.tensor(0.0, device=device)  # 3rd: 개별 손실 추적
            contrast_m2f = torch.tensor(0.0, device=device)  # 3rd: 개별 손실 추적
            align_loss = torch.tensor(0.0, device=device)
            var_loss = torch.tensor(0.0, device=device)
            score_loss = torch.tensor(0.0, device=device)

            # ===== Generator 업데이트 (Asymmetric Contrastive Fairness) =====
            if female_batch is not None or male_batch is not None:
                opt_g.zero_grad()
                
                det_loss = torch.tensor(0.0, device=device)
                feat_f, feat_m = None, None
                proj_f, proj_m = None, None
                female_scores = torch.tensor([], device=device)
                male_scores = torch.tensor([], device=device)
                
                # 여성 배치 처리
                if female_batch is not None:
                    female_perturbed = _apply_generator(generator, female_batch)
                    outputs_f, feat_f = detr.forward_with_features(female_perturbed)
                    proj_f = proj_head(feat_f)
                    valid_f_idx = [i for i, t in enumerate(female_targets) if t["boxes"].numel() > 0]
                    valid_f_targets = [female_targets[i] for i in valid_f_idx]
                    if valid_f_targets:
                        valid_outputs_f = {
                            "pred_logits": outputs_f["pred_logits"][valid_f_idx],
                            "pred_boxes": outputs_f["pred_boxes"][valid_f_idx],
                        }
                        det_f, _ = detr.detection_loss(valid_outputs_f, valid_f_targets)
                        det_loss = det_loss + det_f
                        female_scores = _matched_detection_scores(detr, valid_outputs_f, valid_f_targets)
                
                # 남성 배치 처리
                if male_batch is not None:
                    male_perturbed = _apply_generator(generator, male_batch)
                    outputs_m, feat_m = detr.forward_with_features(male_perturbed)
                    proj_m = proj_head(feat_m)
                    valid_m_idx = [i for i, t in enumerate(male_targets) if t["boxes"].numel() > 0]
                    valid_m_targets = [male_targets[i] for i in valid_m_idx]
                    if valid_m_targets:
                        valid_outputs_m = {
                            "pred_logits": outputs_m["pred_logits"][valid_m_idx],
                            "pred_boxes": outputs_m["pred_boxes"][valid_m_idx],
                        }
                        det_m, _ = detr.detection_loss(valid_outputs_m, valid_m_targets)
                        det_loss = det_loss + det_m
                        male_scores = _matched_detection_scores(detr, valid_outputs_m, valid_m_targets)
                
                # ===== Asymmetric Contrastive Fairness Losses (3rd 핵심) =====
                if proj_f is not None and proj_m is not None:
                    # 3rd 변경: 비대칭 가중치 적용
                    contrast_loss, contrast_f2m, contrast_m2f = _asymmetric_cross_gender_contrastive_loss(
                        proj_f, proj_m, 
                        args.temperature,
                        f_scale=args.contrast_f_scale,  # 1.5 (강화)
                        m_scale=args.contrast_m_scale,  # 0.5 (약화)
                    )
                    
                    # Feature Alignment (동일)
                    align_loss, var_loss = _feature_alignment_loss(feat_f, feat_m)
                    
                    # Score Distribution (동일)
                    score_loss = _score_distribution_loss(female_scores, male_scores)
                
                # 최종 손실
                total_g = (
                    args.lambda_contrast * contrast_loss
                    + args.lambda_align * align_loss
                    + args.lambda_var * var_loss
                    + args.lambda_score * score_loss
                    + current_beta * det_loss
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

            # 3rd: 비대칭 손실 개별 추적 추가
            metrics_logger.update(
                g_contrast=contrast_loss.item(),
                g_contrast_f2m=contrast_f2m.item(),  # 3rd 신규
                g_contrast_m2f=contrast_m2f.item(),  # 3rd 신규
                g_align=align_loss.item(),
                g_var=var_loss.item(),
                g_score=score_loss.item(),
                g_det=det_loss.item(),
                g_total=total_g.item(),
                eps=current_eps,
                beta=current_beta,
                contrast_f_scale=args.contrast_f_scale,  # 3rd 신규
                contrast_m_scale=args.contrast_m_scale,  # 3rd 신규
                lr_g=current_lr_g,
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

        if utils.is_main_process():
            log_entry = {
                "epoch": epoch,
                "g_contrast": metrics_logger.meters["g_contrast"].global_avg,
                "g_contrast_f2m": metrics_logger.meters["g_contrast_f2m"].global_avg,  # 3rd 신규
                "g_contrast_m2f": metrics_logger.meters["g_contrast_m2f"].global_avg,  # 3rd 신규
                "g_align": metrics_logger.meters["g_align"].global_avg,
                "g_var": metrics_logger.meters["g_var"].global_avg,
                "g_score": metrics_logger.meters["g_score"].global_avg,
                "g_det": metrics_logger.meters["g_det"].global_avg,
                "g_total": metrics_logger.meters["g_total"].global_avg,
                "epsilon": current_eps,
                "beta": current_beta,
                "contrast_f_scale": args.contrast_f_scale,  # 3rd 신규
                "contrast_m_scale": args.contrast_m_scale,  # 3rd 신규
                "lr_g": current_lr_g,
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
                        "args": vars(args),
                    },
                    ckpt_path_save,
                )

        if args.distributed:
            dist.barrier()

    if utils.is_main_process():
        print("Training complete!")


if __name__ == "__main__":
    main()
