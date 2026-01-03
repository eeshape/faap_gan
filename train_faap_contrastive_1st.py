"""
FAAP Training - 10th Version: Contrastive Fairness

핵심 변경점 (9th WGAN 방식 대비):
1. Discriminator 완전 제거 → 불안정한 적대적 학습 회피
2. Contrastive Learning 기반 Fairness:
   - InfoNCE Loss: 성별 간 특징을 구별 불가능하게 만듦
   - Feature Alignment: 남성/여성 그룹의 평균 특징 벡터 정렬
   - Cross-Gender Similarity: 다른 성별 샘플을 positive pair로 취급
3. 더 안정적인 학습 (GAN 모드 붕괴 위험 없음)
4. Detection 성능 유지를 위한 손실 함수 설계

Contrastive Fairness 원리:
- 전통적 Contrastive: 같은 클래스 = positive, 다른 클래스 = negative
- Fairness Contrastive: 다른 성별 = positive (가깝게), 같은 성별 = neutral
- 결과: 특징 공간에서 성별 정보가 제거됨
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
        "FAAP-style training for DETR (10th version - Contrastive Fairness)",
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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr_g", type=float, default=1e-4, help="generator learning rate")
    
    # ===== Epsilon (고정값) =====
    parser.add_argument("--epsilon", type=float, default=0.08, help="perturbation bound (fixed)")
    
    # ===== Detection Loss 가중치 (고정값) =====
    parser.add_argument("--beta", type=float, default=0.6, help="detection loss weight (fixed)")
    
    # ===== Contrastive Fairness 설정 (10th 신규) =====
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
    
    # ===== Score Distribution Alignment (9th에서 효과적) =====
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
# Contrastive Fairness Loss Functions (10th 핵심)
# ============================================================

def _cross_gender_contrastive_loss(
    proj_f: torch.Tensor,
    proj_m: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Cross-Gender Contrastive Loss (InfoNCE 변형).
    
    핵심 아이디어:
    - 다른 성별 샘플을 positive pair로 취급
    - 성별 정보가 특징에서 제거되도록 유도
    
    Args:
        proj_f: 여성 샘플의 투영된 특징 (N_f, proj_dim), L2 정규화됨
        proj_m: 남성 샘플의 투영된 특징 (N_m, proj_dim), L2 정규화됨
        temperature: softmax temperature (낮을수록 sharp)
    
    Returns:
        Contrastive loss (여성→남성 + 남성→여성 평균)
    """
    if proj_f.size(0) == 0 or proj_m.size(0) == 0:
        return proj_f.new_tensor(0.0)
    
    # 모든 샘플 간 유사도 행렬
    # proj_f: (N_f, D), proj_m: (N_m, D)
    sim_f_to_m = torch.mm(proj_f, proj_m.t()) / temperature  # (N_f, N_m)
    sim_m_to_f = sim_f_to_m.t()  # (N_m, N_f)
    
    # 여성 → 남성: 각 여성 샘플에 대해 가장 유사한 남성 샘플 찾기
    # InfoNCE: -log(exp(pos) / sum(exp(all)))
    # 여기서는 모든 다른 성별 샘플을 positive로 취급 (평균)
    
    # Soft assignment: 모든 다른 성별 샘플에 균등하게 분포
    n_f, n_m = proj_f.size(0), proj_m.size(0)
    
    # 여성→남성: 각 여성이 남성 전체와 유사해지도록
    loss_f_to_m = -torch.logsumexp(sim_f_to_m, dim=1).mean() + torch.log(torch.tensor(n_m, dtype=torch.float, device=proj_f.device))
    
    # 남성→여성: 각 남성이 여성 전체와 유사해지도록
    loss_m_to_f = -torch.logsumexp(sim_m_to_f, dim=1).mean() + torch.log(torch.tensor(n_f, dtype=torch.float, device=proj_f.device))
    
    return (loss_f_to_m + loss_m_to_f) / 2


def _feature_alignment_loss(
    feat_f: torch.Tensor,
    feat_m: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Feature Alignment Loss: 성별 간 특징 분포 정렬.
    
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


def _score_distribution_loss(
    female_scores: torch.Tensor,
    male_scores: torch.Tensor,
) -> torch.Tensor:
    """Score Distribution Alignment (9th에서 검증된 방식).
    
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

    # ===== 모델 초기화 =====
    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)
    
    # 10th 신규: Projection Head (Discriminator 대신)
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
        
        # 고정값 사용 (스케줄 제거)
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
            align_loss = torch.tensor(0.0, device=device)
            var_loss = torch.tensor(0.0, device=device)
            score_loss = torch.tensor(0.0, device=device)

            # ===== Generator 업데이트 (Contrastive Fairness) =====
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
                    # 유효한 타겟만 필터링 (빈 박스 제외) + outputs도 함께 필터링
                    valid_f_idx = [i for i, t in enumerate(female_targets) if t["boxes"].numel() > 0]
                    valid_f_targets = [female_targets[i] for i in valid_f_idx]
                    if valid_f_targets:
                        # outputs를 유효한 인덱스만 추출
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
                    # 유효한 타겟만 필터링 (빈 박스 제외) + outputs도 함께 필터링
                    valid_m_idx = [i for i, t in enumerate(male_targets) if t["boxes"].numel() > 0]
                    valid_m_targets = [male_targets[i] for i in valid_m_idx]
                    if valid_m_targets:
                        # outputs를 유효한 인덱스만 추출
                        valid_outputs_m = {
                            "pred_logits": outputs_m["pred_logits"][valid_m_idx],
                            "pred_boxes": outputs_m["pred_boxes"][valid_m_idx],
                        }
                        det_m, _ = detr.detection_loss(valid_outputs_m, valid_m_targets)
                        det_loss = det_loss + det_m
                        male_scores = _matched_detection_scores(detr, valid_outputs_m, valid_m_targets)
                
                # ===== Contrastive Fairness Losses (10th 핵심) =====
                if proj_f is not None and proj_m is not None:
                    # 1. Cross-Gender Contrastive Loss
                    contrast_loss = _cross_gender_contrastive_loss(
                        proj_f, proj_m, args.temperature
                    )
                    
                    # 2. Feature Alignment Loss
                    align_loss, var_loss = _feature_alignment_loss(feat_f, feat_m)
                    
                    # 3. Score Distribution Loss
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

            metrics_logger.update(
                g_contrast=contrast_loss.item(),
                g_align=align_loss.item(),
                g_var=var_loss.item(),
                g_score=score_loss.item(),
                g_det=det_loss.item(),
                g_total=total_g.item(),
                eps=current_eps,
                beta=current_beta,
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
                "g_align": metrics_logger.meters["g_align"].global_avg,
                "g_var": metrics_logger.meters["g_var"].global_avg,
                "g_score": metrics_logger.meters["g_score"].global_avg,
                "g_det": metrics_logger.meters["g_det"].global_avg,
                "g_total": metrics_logger.meters["g_total"].global_avg,
                "epsilon": current_eps,
                "beta": current_beta,
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
