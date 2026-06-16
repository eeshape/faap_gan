"""
FAAP Training - IoU-Aware Contrastive Fairness with Prototype Alignment

=============================================================================
핵심 아이디어: AP Gap 개선을 위한 IoU-Aware Contrastive Learning
=============================================================================

[기존 방법의 한계]
1. WGAN-GD (7th): AR Gap 60% 개선, but AP Gap 미개선 (0.106 유지)
2. Contrastive (1st~3rd): Feature 정렬 ≠ AP 개선
3. DINO: Score 정렬만으로는 localization 품질 개선 안 됨

[본 연구의 차별점]
1. IoU-Aware Contrastive: High-IoU detection을 positive로, Low-IoU를 negative로
2. Prototype Alignment: 성별별 대표 feature를 EMA로 학습하고 정렬
3. Dual-Level Loss: Feature-level + Score-level 동시 최적화
4. Hard Sample Mining: 어려운 샘플에 집중하여 효율적 학습

[목표]
- AP Gap: 0.106 → 0.08 (25% 개선)
- AR Gap: 0.0081 → 0.002 (75% 개선)
- Female AP: 0.404 → 0.42 (4% 향상)

=============================================================================
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Allow running as a script
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
# IoU-Aware Projection Head
# =============================================================================

class IoUAwareProjectionHead(nn.Module):
    """
    IoU 정보를 활용하는 Projection Head.

    Feature + IoU를 결합하여 projection하므로,
    contrastive learning이 localization 품질을 반영하게 됨.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 128,
        use_iou: bool = True,
    ):
        super().__init__()
        self.use_iou = use_iou
        input_dim = feature_dim + 1 if use_iou else feature_dim

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        ious: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: (N, feature_dim) 매칭된 query features
            ious: (N,) 각 detection의 IoU 값 (optional)
        Returns:
            (N, output_dim) L2-normalized projections
        """
        if self.use_iou and ious is not None:
            # IoU를 feature에 concatenate
            ious_expanded = ious.unsqueeze(-1)  # (N, 1)
            x = torch.cat([features, ious_expanded], dim=-1)
        else:
            x = features

        proj = self.net(x)
        return F.normalize(proj, dim=-1, p=2)


# =============================================================================
# Prototype Bank for Alignment
# =============================================================================

class PrototypeBank(nn.Module):
    """
    성별별 Prototype을 EMA로 관리.

    여성 feature를 남성 prototype 방향으로 정렬하면서,
    개별 다양성은 유지 (collapse 방지).
    """

    def __init__(self, feature_dim: int = 256, momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("prototype_m", torch.zeros(feature_dim))
        self.register_buffer("prototype_f", torch.zeros(feature_dim))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update(self, feat_f: torch.Tensor, feat_m: torch.Tensor):
        """Prototype EMA 업데이트"""
        if feat_f.numel() == 0 or feat_m.numel() == 0:
            return

        mean_f = feat_f.mean(dim=0)
        mean_m = feat_m.mean(dim=0)

        if not self.initialized:
            self.prototype_f.copy_(mean_f)
            self.prototype_m.copy_(mean_m)
            self.initialized.fill_(True)
        else:
            self.prototype_f.copy_(
                self.momentum * self.prototype_f + (1 - self.momentum) * mean_f
            )
            self.prototype_m.copy_(
                self.momentum * self.prototype_m + (1 - self.momentum) * mean_m
            )

    def alignment_loss(self, feat_f: torch.Tensor) -> torch.Tensor:
        """여성 feature를 남성 prototype으로 정렬"""
        if feat_f.numel() == 0 or not self.initialized:
            return feat_f.new_tensor(0.0)

        # Cosine similarity (높을수록 좋음 → 음수 손실)
        feat_f_norm = F.normalize(feat_f.mean(dim=0), dim=0)
        proto_m_norm = F.normalize(self.prototype_m, dim=0)
        similarity = F.cosine_similarity(feat_f_norm.unsqueeze(0), proto_m_norm.unsqueeze(0))

        return 1.0 - similarity.mean()  # 1 - cos_sim

    def diversity_loss(self, feat_f: torch.Tensor) -> torch.Tensor:
        """Collapse 방지: 분산 유지"""
        if feat_f.size(0) < 2:
            return feat_f.new_tensor(0.0)

        # 분산이 작아지면 패널티
        var = feat_f.var(dim=0).mean()
        target_var = 0.5  # 목표 분산
        return F.relu(target_var - var)


# =============================================================================
# IoU-Aware Contrastive Loss
# =============================================================================

class IoUAwareContrastiveLoss(nn.Module):
    """
    IoU를 고려한 Contrastive Loss.

    핵심 아이디어:
    - High-IoU 여성 detection을 High-IoU 남성 detection과 positive pair로
    - Low-IoU detection을 negative로 활용
    - 이를 통해 AP와 직접 연관된 localization 품질 최적화
    """

    def __init__(
        self,
        temperature: float = 0.1,
        iou_threshold_high: float = 0.5,
        iou_threshold_low: float = 0.3,
        asymmetric_weight_f: float = 1.5,
        asymmetric_weight_m: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.iou_threshold_high = iou_threshold_high
        self.iou_threshold_low = iou_threshold_low
        self.asymmetric_weight_f = asymmetric_weight_f
        self.asymmetric_weight_m = asymmetric_weight_m

    def forward(
        self,
        proj_f: torch.Tensor,
        proj_m: torch.Tensor,
        iou_f: Optional[torch.Tensor] = None,
        iou_m: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            proj_f: (N_f, D) 여성 projections
            proj_m: (N_m, D) 남성 projections
            iou_f: (N_f,) 여성 IoU 값
            iou_m: (N_m,) 남성 IoU 값
        Returns:
            loss, info_dict
        """
        if proj_f.size(0) == 0 or proj_m.size(0) == 0:
            return proj_f.new_tensor(0.0), {}

        # IoU가 없으면 기본 contrastive loss
        if iou_f is None or iou_m is None:
            return self._basic_contrastive(proj_f, proj_m)

        # IoU 기반 샘플 분류
        high_iou_f_mask = iou_f >= self.iou_threshold_high
        high_iou_m_mask = iou_m >= self.iou_threshold_high
        low_iou_f_mask = iou_f < self.iou_threshold_low

        high_f = proj_f[high_iou_f_mask]
        high_m = proj_m[high_iou_m_mask]
        low_f = proj_f[low_iou_f_mask]

        losses = {}
        total_loss = proj_f.new_tensor(0.0)

        # 1. High-IoU 여성 → High-IoU 남성 (positive)
        if high_f.size(0) > 0 and high_m.size(0) > 0:
            sim_pos = torch.mm(high_f, high_m.t()) / self.temperature

            # 각 여성 샘플에 대해 모든 남성과의 유사도 평균
            loss_f_to_m = -torch.logsumexp(sim_pos, dim=1).mean() + torch.log(
                torch.tensor(high_m.size(0), dtype=torch.float, device=proj_f.device)
            )
            losses["loss_high_f2m"] = loss_f_to_m.item()
            total_loss = total_loss + self.asymmetric_weight_f * loss_f_to_m

        # 2. High-IoU 남성 → High-IoU 여성 (약하게)
        if high_m.size(0) > 0 and high_f.size(0) > 0:
            sim_pos_m = torch.mm(high_m, high_f.t()) / self.temperature
            loss_m_to_f = -torch.logsumexp(sim_pos_m, dim=1).mean() + torch.log(
                torch.tensor(high_f.size(0), dtype=torch.float, device=proj_f.device)
            )
            losses["loss_high_m2f"] = loss_m_to_f.item()
            total_loss = total_loss + self.asymmetric_weight_m * loss_m_to_f

        # 3. High-IoU를 Low-IoU와 분리 (negative contrastive)
        if high_f.size(0) > 0 and low_f.size(0) > 0:
            sim_neg = torch.mm(high_f, low_f.t()) / self.temperature
            # Low-IoU와의 유사도를 낮추고 싶음
            loss_separation = torch.logsumexp(sim_neg, dim=1).mean()
            losses["loss_separation"] = loss_separation.item()
            total_loss = total_loss + 0.3 * loss_separation

        losses["n_high_f"] = high_f.size(0)
        losses["n_high_m"] = high_m.size(0)
        losses["n_low_f"] = low_f.size(0)

        return total_loss, losses

    def _basic_contrastive(
        self,
        proj_f: torch.Tensor,
        proj_m: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """IoU 정보 없을 때 기본 cross-gender contrastive"""
        sim = torch.mm(proj_f, proj_m.t()) / self.temperature

        n_f, n_m = proj_f.size(0), proj_m.size(0)

        loss_f2m = -torch.logsumexp(sim, dim=1).mean() + torch.log(
            torch.tensor(n_m, dtype=torch.float, device=proj_f.device)
        )
        loss_m2f = -torch.logsumexp(sim.t(), dim=1).mean() + torch.log(
            torch.tensor(n_f, dtype=torch.float, device=proj_f.device)
        )

        total = self.asymmetric_weight_f * loss_f2m + self.asymmetric_weight_m * loss_m2f

        return total, {"loss_f2m": loss_f2m.item(), "loss_m2f": loss_m2f.item()}


# =============================================================================
# Score-Level Wasserstein Loss (7th에서 가져옴)
# =============================================================================

def _resize_sorted(scores: torch.Tensor, target_len: int) -> torch.Tensor:
    """분위수 정렬을 위한 크기 조정"""
    if target_len <= 0:
        return scores.new_zeros(0)
    if scores.numel() == 0:
        return scores.new_zeros(target_len)
    if scores.numel() == target_len:
        return scores
    idx = torch.linspace(0, scores.numel() - 1, target_len, device=scores.device)
    idx_low = idx.floor().long()
    idx_high = idx.ceil().long()
    weight = idx - idx_low
    return scores[idx_low] * (1 - weight) + scores[idx_high] * weight


def _wasserstein_1d_asymmetric(
    female_scores: torch.Tensor,
    male_scores: torch.Tensor,
) -> torch.Tensor:
    """단방향 Wasserstein: 여성 score → 남성 score"""
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0)

    sorted_f = female_scores.sort().values
    sorted_m = male_scores.detach().sort().values

    k = max(sorted_f.numel(), sorted_m.numel())
    sorted_f = _resize_sorted(sorted_f, k)
    sorted_m = _resize_sorted(sorted_m, k)

    # 단방향: 여성 < 남성일 때만 패널티
    return F.relu(sorted_m - sorted_f).mean()


# =============================================================================
# Utility Functions
# =============================================================================

def _default_output_dir(script_path: Path) -> str:
    stem = script_path.stem
    stem_lower = stem.lower()
    suffix = stem
    for prefix in ("train_faap_contrastive_", "train_faap_", "train_"):
        if stem_lower.startswith(prefix):
            suffix = stem[len(prefix):]
            break
    suffix = suffix.lower()
    return str(Path("faap_outputs") / f"faap_outputs_contrastive_{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "FAAP IoU-Aware Contrastive Fairness Training",
        add_help=True,
    )

    # Paths
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()))
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(Path(__file__)))

    # Training basics
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    # Perturbation settings (7th 기반)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--epsilon_final", type=float, default=0.10)
    parser.add_argument("--epsilon_min", type=float, default=0.09)
    parser.add_argument("--epsilon_warmup_epochs", type=int, default=8)
    parser.add_argument("--epsilon_hold_epochs", type=int, default=6)
    parser.add_argument("--epsilon_cooldown_epochs", type=int, default=16)

    # Loss weights
    parser.add_argument("--lambda_iou_contrast", type=float, default=1.0,
                        help="IoU-aware contrastive loss weight")
    parser.add_argument("--lambda_prototype", type=float, default=0.5,
                        help="Prototype alignment loss weight")
    parser.add_argument("--lambda_diversity", type=float, default=0.1,
                        help="Diversity (anti-collapse) loss weight")
    parser.add_argument("--lambda_wasserstein", type=float, default=0.3,
                        help="Score-level Wasserstein loss weight")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Detection loss weight start")
    parser.add_argument("--beta_final", type=float, default=0.6,
                        help="Detection loss weight final")

    # IoU-aware contrastive settings
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--iou_threshold_high", type=float, default=0.5)
    parser.add_argument("--iou_threshold_low", type=float, default=0.3)
    parser.add_argument("--asymmetric_weight_f", type=float, default=1.5,
                        help="Female→Male direction weight")
    parser.add_argument("--asymmetric_weight_m", type=float, default=0.5,
                        help="Male→Female direction weight")

    # Projection head
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--use_iou_in_proj", type=bool, default=True,
                        help="Include IoU value in projection input (default: True)")

    # Prototype bank
    parser.add_argument("--prototype_momentum", type=float, default=0.99)

    # Other
    parser.add_argument("--max_norm", type=float, default=0.1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--max_train_per_gender", type=int, default=0)
    parser.add_argument("--obj_conf_thresh", type=float, default=0.5)
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
    warmup_end = max(0, warmup_epochs - 1) if warmup_epochs > 1 else 0

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


def _scheduled_beta(epoch: int, total_epochs: int, beta_start: float, beta_final: float) -> float:
    if total_epochs <= 1:
        return beta_start
    progress = min(epoch / max(1, total_epochs - 1), 1.0)
    return beta_start + (beta_final - beta_start) * progress


def _extract_matched_features_and_ious(
    detr: FrozenDETR,
    outputs: dict,
    features: torch.Tensor,
    targets: Sequence[dict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hungarian matching을 통해 GT와 매칭된 query의 feature, score, IoU를 추출.

    Returns:
        matched_features: (N_matched, feature_dim)
        matched_scores: (N_matched,)
        matched_ious: (N_matched,)
    """
    if len(targets) == 0:
        device = outputs["pred_logits"].device
        return (
            features.new_zeros(0, features.size(-1)),
            outputs["pred_logits"].new_zeros(0),
            outputs["pred_logits"].new_zeros(0),
        )

    indices = detr.criterion.matcher(outputs, targets)
    probs = outputs["pred_logits"].softmax(dim=-1)
    pred_boxes = outputs["pred_boxes"]

    matched_features_list = []
    matched_scores_list = []
    matched_ious_list = []

    for b, (src_idx, tgt_idx) in enumerate(indices):
        if len(src_idx) == 0:
            continue

        # Features
        batch_feat = features[b, src_idx]  # (num_matched, feature_dim)
        matched_features_list.append(batch_feat)

        # Scores
        tgt_labels = targets[b]["labels"][tgt_idx]
        batch_scores = probs[b, src_idx, tgt_labels]
        matched_scores_list.append(batch_scores)

        # IoU 계산
        pred_box = pred_boxes[b, src_idx]  # (num_matched, 4) - cxcywh format
        tgt_box = targets[b]["boxes"][tgt_idx]  # (num_matched, 4) - cxcywh format

        # cxcywh → xyxy 변환
        pred_xyxy = _cxcywh_to_xyxy(pred_box)
        tgt_xyxy = _cxcywh_to_xyxy(tgt_box)

        # IoU 계산 (element-wise)
        batch_ious = _compute_iou(pred_xyxy, tgt_xyxy)
        matched_ious_list.append(batch_ious)

    if matched_features_list:
        return (
            torch.cat(matched_features_list, dim=0),
            torch.cat(matched_scores_list, dim=0),
            torch.cat(matched_ious_list, dim=0),
        )

    device = outputs["pred_logits"].device
    return (
        features.new_zeros(0, features.size(-1)),
        outputs["pred_logits"].new_zeros(0),
        outputs["pred_logits"].new_zeros(0),
    )


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Center format to corner format"""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Element-wise IoU 계산"""
    # boxes: (N, 4) in xyxy format
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1 + area2 - inter
    iou = inter / (union + 1e-8)

    return iou


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
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA")
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
        print("IoU-Aware Contrastive Fairness Training")
        print("=" * 70)
        print(f"Lambda IoU Contrast: {args.lambda_iou_contrast}")
        print(f"Lambda Prototype: {args.lambda_prototype}")
        print(f"Lambda Wasserstein: {args.lambda_wasserstein}")
        print(f"Asymmetric weights: F→M={args.asymmetric_weight_f}, M→F={args.asymmetric_weight_m}")
        print(f"IoU thresholds: high={args.iou_threshold_high}, low={args.iou_threshold_low}")
        print("=" * 70)

    # ==========================================================================
    # Model Initialization
    # ==========================================================================

    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)

    # IoU-Aware Projection Head
    proj_head = IoUAwareProjectionHead(
        feature_dim=detr.hidden_dim,
        hidden_dim=detr.hidden_dim,
        output_dim=args.proj_dim,
        use_iou=args.use_iou_in_proj,
    ).to(device)

    # Prototype Bank
    prototype_bank = PrototypeBank(
        feature_dim=detr.hidden_dim,
        momentum=args.prototype_momentum,
    ).to(device)

    # IoU-Aware Contrastive Loss
    iou_contrastive_loss = IoUAwareContrastiveLoss(
        temperature=args.temperature,
        iou_threshold_high=args.iou_threshold_high,
        iou_threshold_low=args.iou_threshold_low,
        asymmetric_weight_f=args.asymmetric_weight_f,
        asymmetric_weight_m=args.asymmetric_weight_m,
    ).to(device)

    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)
        proj_head = DDP(proj_head, device_ids=[args.gpu] if args.gpu is not None else None)

    # Optimizer
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
        if "prototype_bank" in ckpt:
            prototype_bank.load_state_dict(ckpt["prototype_bank"])
        if "opt_g" in ckpt:
            opt_g.load_state_dict(ckpt["opt_g"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if utils.is_main_process():
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # DataLoader
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

    # ==========================================================================
    # Training Loop
    # ==========================================================================

    for epoch in range(start_epoch, args.epochs):
        metrics_logger = utils.MetricLogger(delimiter="  ")
        generator.train()
        proj_head.train()

        if args.distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # Schedules
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

        for samples, targets, genders in metrics_logger.log_every(
            train_loader, args.log_every, f"Epoch {epoch}"
        ):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            genders = [g.lower() for g in genders]

            # Gender split
            female_idx = [i for i, g in enumerate(genders) if g == "female"]
            male_idx = [i for i, g in enumerate(genders) if g == "male"]
            female_batch, female_targets = _split_nested(samples, targets, female_idx)
            male_batch, male_targets = _split_nested(samples, targets, male_idx)

            # Initialize metrics
            loss_iou_contrast = torch.tensor(0.0, device=device)
            loss_prototype = torch.tensor(0.0, device=device)
            loss_diversity = torch.tensor(0.0, device=device)
            loss_wasserstein = torch.tensor(0.0, device=device)
            loss_det = torch.tensor(0.0, device=device)
            total_g = torch.tensor(0.0, device=device)
            delta_linf = torch.tensor(0.0, device=device)
            delta_l2 = torch.tensor(0.0, device=device)
            mean_iou_f = torch.tensor(0.0, device=device)
            mean_iou_m = torch.tensor(0.0, device=device)

            # Skip if either gender missing
            if female_batch is None or male_batch is None:
                metrics_logger.update(
                    loss_iou_contrast=0.0, loss_prototype=0.0, loss_diversity=0.0,
                    loss_wasserstein=0.0, loss_det=0.0, total_g=0.0,
                    eps=current_eps, beta=current_beta,
                    delta_linf=0.0, delta_l2=0.0, mean_iou_f=0.0, mean_iou_m=0.0,
                )
                continue

            # =================================================================
            # Forward Pass
            # =================================================================
            opt_g.zero_grad()

            # Apply perturbation
            female_perturbed = _apply_generator(generator, female_batch)
            male_perturbed = _apply_generator(generator, male_batch)

            # DETR forward
            outputs_f, feat_f = detr.forward_with_features(female_perturbed)
            outputs_m, feat_m = detr.forward_with_features(male_perturbed)

            # Extract matched features, scores, IoUs
            matched_feat_f, matched_scores_f, matched_ious_f = _extract_matched_features_and_ious(
                detr, outputs_f, feat_f, female_targets
            )
            matched_feat_m, matched_scores_m, matched_ious_m = _extract_matched_features_and_ious(
                detr, outputs_m, feat_m, male_targets
            )

            # =================================================================
            # Loss Computation
            # =================================================================

            # 1. IoU-Aware Contrastive Loss
            if matched_feat_f.size(0) > 0 and matched_feat_m.size(0) > 0:
                proj_f = proj_head(
                    matched_feat_f,
                    matched_ious_f if args.use_iou_in_proj else None
                )
                proj_m = proj_head(
                    matched_feat_m,
                    matched_ious_m if args.use_iou_in_proj else None
                )

                loss_iou_contrast, contrast_info = iou_contrastive_loss(
                    proj_f, proj_m, matched_ious_f, matched_ious_m
                )

            # 2. Prototype Alignment
            if matched_feat_f.size(0) > 0 and matched_feat_m.size(0) > 0:
                prototype_bank.update(matched_feat_f.detach(), matched_feat_m.detach())
                loss_prototype = prototype_bank.alignment_loss(matched_feat_f)
                loss_diversity = prototype_bank.diversity_loss(matched_feat_f)

            # 3. Score-Level Wasserstein
            if matched_scores_f.numel() > 0 and matched_scores_m.numel() > 0:
                loss_wasserstein = _wasserstein_1d_asymmetric(matched_scores_f, matched_scores_m)

            # 4. Detection Loss
            loss_det_f, _ = detr.detection_loss(outputs_f, female_targets)
            loss_det_m, _ = detr.detection_loss(outputs_m, male_targets)
            loss_det = (loss_det_f + loss_det_m) / 2

            # Total Loss
            total_g = (
                args.lambda_iou_contrast * loss_iou_contrast
                + args.lambda_prototype * loss_prototype
                + args.lambda_diversity * loss_diversity
                + args.lambda_wasserstein * loss_wasserstein
                + current_beta * loss_det
            )

            # =================================================================
            # Metrics
            # =================================================================
            with torch.no_grad():
                delta_f = female_perturbed.tensors - female_batch.tensors
                delta_m = male_perturbed.tensors - male_batch.tensors
                delta_cat = torch.cat([delta_f, delta_m], dim=0)
                delta_linf = delta_cat.abs().amax(dim=(1, 2, 3)).mean()
                delta_l2 = delta_cat.flatten(1).norm(p=2, dim=1).mean()

                if matched_ious_f.numel() > 0:
                    mean_iou_f = matched_ious_f.mean()
                if matched_ious_m.numel() > 0:
                    mean_iou_m = matched_ious_m.mean()

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
                loss_iou_contrast=loss_iou_contrast.item(),
                loss_prototype=loss_prototype.item(),
                loss_diversity=loss_diversity.item(),
                loss_wasserstein=loss_wasserstein.item(),
                loss_det=loss_det.item(),
                total_g=total_g.item(),
                eps=current_eps,
                beta=current_beta,
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                mean_iou_f=mean_iou_f.item(),
                mean_iou_m=mean_iou_m.item(),
            )

        # =====================================================================
        # End of Epoch
        # =====================================================================
        metrics_logger.synchronize_between_processes()

        if utils.is_main_process():
            log_entry = {
                "epoch": epoch,
                "loss_iou_contrast": metrics_logger.meters["loss_iou_contrast"].global_avg,
                "loss_prototype": metrics_logger.meters["loss_prototype"].global_avg,
                "loss_diversity": metrics_logger.meters["loss_diversity"].global_avg,
                "loss_wasserstein": metrics_logger.meters["loss_wasserstein"].global_avg,
                "loss_det": metrics_logger.meters["loss_det"].global_avg,
                "total_g": metrics_logger.meters["total_g"].global_avg,
                "epsilon": current_eps,
                "beta": current_beta,
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "mean_iou_f": metrics_logger.meters["mean_iou_f"].global_avg,
                "mean_iou_m": metrics_logger.meters["mean_iou_m"].global_avg,
            }

            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Print summary
            iou_gap = log_entry["mean_iou_m"] - log_entry["mean_iou_f"]
            print(f"\n[Epoch {epoch}] Summary:")
            print(f"  IoU Contrast Loss: {log_entry['loss_iou_contrast']:.4f}")
            print(f"  Prototype Loss: {log_entry['loss_prototype']:.4f}")
            print(f"  Wasserstein Loss: {log_entry['loss_wasserstein']:.4f}")
            print(f"  Detection Loss: {log_entry['loss_det']:.4f}")
            print(f"  Total: {log_entry['total_g']:.4f}")
            print(f"  Mean IoU (F/M): {log_entry['mean_iou_f']:.4f} / {log_entry['mean_iou_m']:.4f}")
            print(f"  IoU Gap (M-F): {iou_gap:.4f}")
            print(f"  Epsilon: {current_eps:.4f}, Beta: {current_beta:.4f}")

            # Save checkpoint
            if (epoch + 1) % args.save_every == 0:
                ckpt_path_save = output_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "generator": _unwrap_ddp(generator).state_dict(),
                        "proj_head": _unwrap_ddp(proj_head).state_dict(),
                        "prototype_bank": prototype_bank.state_dict(),
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
        print("IoU-Aware Contrastive Fairness Training Complete!")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print("\nKey innovations:")
        print("  - IoU-Aware Contrastive: High-IoU = positive, Low-IoU = negative")
        print("  - Prototype Alignment: EMA gender prototypes for stable alignment")
        print("  - Asymmetric weights: F→M stronger than M→F")
        print("  - Score-level Wasserstein: Complementary distribution alignment")


if __name__ == "__main__":
    main()
