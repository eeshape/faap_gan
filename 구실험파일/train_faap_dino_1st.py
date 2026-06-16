"""
DINO-style Self-Distillation for Fair Object Detection

=============================================================================
핵심 아이디어: DINO (Self-DIstillation with NO labels) 방식을 성별 공정성에 적용
=============================================================================

[문제 정의]
기존 Wasserstein/Contrastive 방식의 한계:
1. 단순 거리 최소화는 분포 붕괴(Collapse) 위험
2. 여성/남성 분포를 억지로 맞추면 각 그룹의 다양성이 손실됨

[DINO 논문의 핵심 기법]
1. Teacher-Student 구조
   - Teacher: EMA(Exponential Moving Average)로 천천히 업데이트 → 안정적인 타겟 제공
   - Student: 직접 학습 → Teacher를 따라가도록

2. Centering (평균 보정)
   - Teacher 출력에서 running mean(center)을 빼줌
   - 효과: 한 모드로 붕괴하는 것을 방지
   - 수식: teacher_out = teacher_out - center

3. Sharpening (분포 날카롭게 하기)
   - Temperature를 낮게 설정하여 확률 분포를 peak로 만듦
   - Teacher temp (τ_t) < Student temp (τ_s)
   - 수식: softmax(z / τ)

4. Cross-Entropy Distillation Loss
   - H(p_t, p_s) = -Σ p_t * log(p_s)
   - Student가 Teacher의 출력을 모방하도록 학습

[본 구현에서의 적용]
- Teacher: 남성 이미지의 detection score 분포 (EMA로 업데이트)
- Student: 여성 이미지의 detection score 분포 (직접 학습)
- Centering: 남성 score의 running mean을 빼서 collapse 방지
- Sharpening: Teacher에 낮은 temperature 적용

[논문 인용 포인트]
"단순히 거리를 좁히는 Contrastive Learning을 넘어, DINO의 Self-Distillation
프레임워크를 도입하여 여성/남성 detection 분포의 다양성은 유지하면서
(Collapse 방지 via Centering) 공정한 성능 분포만 일치시켰다."

=============================================================================
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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


# =============================================================================
# DINO-specific Components
# =============================================================================

class DINOHead(nn.Module):
    """
    DINO-style projection head for detection scores.
    
    Detection score를 더 높은 차원의 특징 공간으로 투영한 뒤,
    다시 적절한 차원으로 줄여서 distillation에 사용합니다.
    
    Original DINO uses: input → hidden → hidden → output → L2 norm
    
    Note: 
    - weight_norm을 사용하지 않음 (deepcopy 호환성 문제)
    - BatchNorm 대신 LayerNorm 사용 (배치 크기 1에서도 동작)
    대신 L2 normalization으로 유사한 효과를 얻음
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_layers: int = 3,
        use_bn: bool = False,  # Changed to False by default (LayerNorm used instead)
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.use_bn = use_bn
        
        layers = []
        for i in range(num_layers - 1):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            # Use LayerNorm instead of BatchNorm (works with batch_size=1)
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        
        # Final layer (no activation)
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)
        
        # Final projection layer (without weight_norm for deepcopy compatibility)
        # L2 normalization in forward pass provides similar regularization effect
        self.last_layer = nn.Linear(out_dim, out_dim, bias=False)
        # Initialize with orthogonal initialization for better training
        nn.init.orthogonal_(self.last_layer.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N,) or (N, D) shaped tensor of detection scores/features
        Returns:
            (N, out_dim) L2-normalized features
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # (N,) → (N, 1)
        
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        # Final L2 normalization (replaces weight_norm effect)
        x = F.normalize(x, dim=-1, p=2)
        return x


class DINOCenter:
    """
    DINO의 Centering 메커니즘을 구현합니다.
    
    Teacher의 출력에서 exponential moving average center를 빼줌으로써
    모든 출력이 한 점으로 붕괴하는 것을 방지합니다.
    
    center = m * center + (1 - m) * mean(teacher_output)
    """
    
    def __init__(self, out_dim: int, center_momentum: float = 0.9):
        self.center = torch.zeros(out_dim)
        self.center_momentum = center_momentum
        self.initialized = False
    
    def to(self, device: torch.device):
        self.center = self.center.to(device)
        return self
    
    @torch.no_grad()
    def update(self, teacher_output: torch.Tensor):
        """
        Teacher 출력의 batch mean으로 center를 업데이트합니다.
        
        Args:
            teacher_output: (batch_size, out_dim) teacher 출력
        """
        batch_center = teacher_output.mean(dim=0)
        
        if not self.initialized:
            self.center = batch_center.clone()
            self.initialized = True
        else:
            self.center = (
                self.center_momentum * self.center + 
                (1 - self.center_momentum) * batch_center
            )
    
    def apply(self, teacher_output: torch.Tensor) -> torch.Tensor:
        """
        Teacher 출력에서 center를 빼줍니다 (Centering).
        
        Args:
            teacher_output: (batch_size, out_dim)
        Returns:
            Centered output
        """
        return teacher_output - self.center


class DINOLoss(nn.Module):
    """
    DINO Self-Distillation Loss
    
    핵심 수식:
    L = -Σ softmax(teacher/τ_t - center) * log(softmax(student/τ_s))
    
    여기서:
    - τ_t: teacher temperature (낮음, 날카로운 분포)
    - τ_s: student temperature (높음, 부드러운 분포)
    - center: teacher 출력의 running mean
    
    Teacher는 날카로운(sharp) 분포를, Student는 부드러운(soft) 분포를 갖게 하여
    Student가 Teacher의 confident한 예측을 따라가도록 유도합니다.
    """
    
    def __init__(
        self,
        out_dim: int,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center = DINOCenter(out_dim, center_momentum)
    
    def to(self, device: torch.device):
        self.center.to(device)
        return self
    
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        update_center: bool = True,
    ) -> torch.Tensor:
        """
        DINO distillation loss 계산
        
        Args:
            student_output: (N, D) student의 projection 출력
            teacher_output: (N, D) teacher의 projection 출력
            update_center: center 업데이트 여부
            
        Returns:
            Cross-entropy loss scalar
        """
        # Centering (Teacher에만 적용)
        teacher_centered = self.center.apply(teacher_output)
        
        # Sharpening via temperature
        teacher_probs = F.softmax(teacher_centered / self.teacher_temp, dim=-1)
        student_log_probs = F.log_softmax(student_output / self.student_temp, dim=-1)
        
        # Cross-entropy: H(p_teacher, p_student) = -Σ p_t * log(p_s)
        loss = -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()
        
        # Update center with teacher output (before centering)
        if update_center:
            self.center.update(teacher_output)
        
        return loss


class EMATeacher:
    """
    Exponential Moving Average Teacher for DINO
    
    Teacher 네트워크는 Student의 EMA로 업데이트됩니다.
    이렇게 하면 Teacher가 Student보다 천천히 변화하여 안정적인 타겟을 제공합니다.
    
    teacher = m * teacher + (1 - m) * student
    
    DINO 논문에서는 momentum schedule을 사용:
    - 초기: m = 0.996 (빠른 업데이트)
    - 후기: m → 1.0 (느린 업데이트)
    
    Note: deepcopy 대신 새 인스턴스 + state_dict 복사 방식 사용
    (weight_norm 등과의 호환성 문제 해결)
    """
    
    def __init__(self, student: nn.Module, momentum: float = 0.996):
        # deepcopy 대신 새 인스턴스 생성 후 state_dict 복사
        self.teacher = self._create_teacher_from_student(student)
        self.momentum = momentum
        
        # Teacher는 gradient 불필요
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    @staticmethod
    def _create_teacher_from_student(student: nn.Module) -> nn.Module:
        """
        Student와 동일한 구조의 Teacher를 생성하고 가중치를 복사합니다.
        deepcopy를 사용하지 않아 weight_norm 등과 호환됩니다.
        """
        # DINOHead인 경우 동일한 구조로 새로 생성
        if isinstance(student, DINOHead):
            teacher = DINOHead(
                in_dim=student.in_dim,
                hidden_dim=student.hidden_dim,
                out_dim=student.out_dim,
                num_layers=student.num_layers,
                use_bn=student.use_bn,
            )
        elif isinstance(student, DDP):
            # DDP로 감싸진 경우 내부 모듈 사용
            return EMATeacher._create_teacher_from_student(student.module)
        else:
            # 일반적인 경우: 같은 클래스로 새 인스턴스 생성 시도
            # 이 경우 __init__ 인자가 필요없는 모듈만 지원
            raise TypeError(
                f"EMATeacher does not support module type: {type(student)}. "
                "Please extend _create_teacher_from_student method."
            )
        
        # state_dict 복사
        teacher.load_state_dict(student.state_dict())
        return teacher
    
    def to(self, device: torch.device):
        self.teacher = self.teacher.to(device)
        return self
    
    @torch.no_grad()
    def update(self, student: nn.Module):
        """Student의 파라미터로 Teacher를 EMA 업데이트합니다."""
        # DDP wrapper 처리
        if isinstance(student, DDP):
            student = student.module
        
        student_params = dict(student.named_parameters())
        teacher_params = dict(self.teacher.named_parameters())
        
        for name, teacher_param in teacher_params.items():
            if name in student_params:
                student_param = student_params[name]
                teacher_param.data = (
                    self.momentum * teacher_param.data + 
                    (1 - self.momentum) * student_param.data
                )
    
    def forward(self, *args, **kwargs):
        """Teacher forward pass (no gradient)"""
        self.teacher.eval()
        with torch.no_grad():
            return self.teacher(*args, **kwargs)


def _cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    current_epoch: int,
) -> float:
    """
    Cosine annealing schedule (DINO에서 사용하는 momentum schedule)
    
    momentum이 base_value에서 final_value로 cosine 형태로 증가합니다.
    """
    if epochs <= 1:
        return final_value
    
    progress = current_epoch / (epochs - 1)
    return final_value + 0.5 * (base_value - final_value) * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())


# =============================================================================
# Utility Functions (기존 코드에서 유지)
# =============================================================================

def _default_output_dir(script_path: Path) -> str:
    """스크립트 파일 이름을 기반으로 기본 output_dir을 생성합니다."""
    stem = script_path.stem
    stem_lower = stem.lower()
    suffix = stem
    for prefix in ("train_faap_dino_", "train_faap_wgan_", "train_"):
        if stem_lower.startswith(prefix):
            suffix = stem[len(prefix):]
            break
    suffix = suffix.lower()
    return str(Path("faap_outputs") / f"faap_outputs_dino_{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "FAAP-DINO: Self-Distillation for Fair Object Detection",
        add_help=True,
    )
    
    # Dataset & Model paths
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()))
    parser.add_argument(
        "--output_dir",
        type=str,
        default=_default_output_dir(Path(__file__)),
    )
    
    # Training basics
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size (optimized for L40S 46GB with gradient computation)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    
    # Generator settings
    parser.add_argument("--lr_g", type=float, default=1e-4, help="Generator learning rate")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Perturbation bound start")
    parser.add_argument("--epsilon_final", type=float, default=0.10, help="Perturbation bound peak")
    parser.add_argument("--epsilon_warmup_epochs", type=int, default=8)
    parser.add_argument("--epsilon_hold_epochs", type=int, default=8)
    parser.add_argument("--epsilon_cooldown_epochs", type=int, default=14)
    parser.add_argument("--epsilon_min", type=float, default=0.08)
    
    # DINO-specific hyperparameters
    parser.add_argument("--dino_out_dim", type=int, default=128, help="DINO projection output dimension")
    parser.add_argument("--dino_hidden_dim", type=int, default=256, help="DINO projection hidden dimension")
    parser.add_argument("--teacher_temp", type=float, default=0.04, help="Teacher temperature (sharp)")
    parser.add_argument("--student_temp", type=float, default=0.1, help="Student temperature (soft)")
    parser.add_argument("--teacher_temp_warmup", type=float, default=0.04, help="Teacher temp warmup start")
    parser.add_argument("--teacher_temp_final", type=float, default=0.07, help="Teacher temp final")
    parser.add_argument("--center_momentum", type=float, default=0.9, help="Center EMA momentum")
    parser.add_argument("--ema_momentum", type=float, default=0.996, help="Teacher EMA momentum start")
    parser.add_argument("--ema_momentum_final", type=float, default=1.0, help="Teacher EMA momentum final")
    
    # Loss weights
    parser.add_argument("--lambda_dino", type=float, default=1.0, help="DINO distillation loss weight")
    parser.add_argument("--lambda_det", type=float, default=0.5, help="Detection loss weight start")
    parser.add_argument("--lambda_det_final", type=float, default=0.6, help="Detection loss weight final")
    parser.add_argument("--lambda_entropy", type=float, default=0.1, help="Entropy regularization weight")
    
    # Discriminator (optional, for additional fairness signal)
    parser.add_argument("--use_discriminator", action="store_true", help="Use adversarial discriminator additionally")
    parser.add_argument("--lr_d", type=float, default=1e-4, help="Discriminator learning rate")
    parser.add_argument("--k_d", type=int, default=2, help="Discriminator update steps")
    parser.add_argument("--lambda_adv", type=float, default=0.5, help="Adversarial loss weight")
    
    # Other training settings
    parser.add_argument("--max_norm", type=float, default=0.1, help="Gradient clipping")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--max_train_per_gender", type=int, default=0)
    parser.add_argument("--obj_conf_thresh", type=float, default=0.5)
    
    # Resume & Distributed
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--dist_url", default="env://")
    
    return parser.parse_args()


def _split_nested(samples: NestedTensor, targets: Sequence[dict], keep: List[int]):
    """배치에서 특정 인덱스만 추출합니다."""
    if len(keep) == 0:
        return None, []
    tensor = samples.tensors[keep]
    mask = samples.mask[keep] if samples.mask is not None else None
    return NestedTensor(tensor, mask), [targets[i] for i in keep]


def _apply_generator(generator: nn.Module, samples: NestedTensor) -> NestedTensor:
    """Generator를 적용하여 perturbed 이미지를 생성합니다."""
    tensors = samples.tensors
    delta = generator(tensors)
    perturbed = clamp_normalized(tensors + delta)
    return NestedTensor(perturbed, samples.mask)


def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """엔트로피 손실: 예측의 불확실성을 높입니다."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    return -(probs * log_probs).sum(dim=-1).mean()


def _scheduled_epsilon(
    epoch: int,
    warmup_epochs: int,
    hold_epochs: int,
    cooldown_epochs: int,
    eps_start: float,
    eps_peak: float,
    eps_min: float,
) -> float:
    """Epsilon scheduler: warmup → hold → cooldown"""
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


def _scheduled_value(
    epoch: int, 
    total_epochs: int, 
    start_value: float, 
    final_value: float
) -> float:
    """Linear schedule from start to final value."""
    if total_epochs <= 1:
        return start_value
    progress = min(epoch / max(1, total_epochs - 1), 1.0)
    return start_value + (final_value - start_value) * progress


def _unwrap_ddp(module: nn.Module) -> nn.Module:
    """DDP wrapper를 제거합니다."""
    return module.module if isinstance(module, DDP) else module


def _set_generator_epsilon(generator: nn.Module, epsilon: float) -> None:
    """Generator의 epsilon을 설정합니다."""
    _unwrap_ddp(generator).epsilon = epsilon


def _matched_detection_scores(
    detr: FrozenDETR, 
    outputs: dict, 
    targets: Sequence[dict]
) -> torch.Tensor:
    """Ground truth와 매칭된 detection score를 추출합니다."""
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


def _get_all_detection_scores(outputs: dict) -> torch.Tensor:
    """모든 query의 detection score (max class prob)를 추출합니다."""
    # pred_logits: (batch, num_queries, num_classes+1)
    # 마지막 클래스는 no-object
    probs = outputs["pred_logits"].softmax(dim=-1)
    # 배경 클래스 제외한 최대 확률
    max_probs = probs[..., :-1].max(dim=-1).values  # (batch, num_queries)
    return max_probs.flatten()  # (batch * num_queries,)


# =============================================================================
# Main Training Loop with DINO
# =============================================================================

def main():
    args = parse_args()
    utils.init_distributed_mode(args)
    
    if not hasattr(args, "gpu"):
        args.gpu = None
    
    # Setup DETR
    detr_repo = ensure_detr_repo_on_path(Path(args.detr_repo))
    ckpt_path = Path(args.detr_checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = detr_repo / ckpt_path
    
    # Device setup
    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA")
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Random seed
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
        
        # Save training config
        with (output_dir / "config.json").open("w") as f:
            json.dump(vars(args), f, indent=2)
    
    if args.distributed:
        dist.barrier()
    
    # Dataset inspection
    if utils.is_main_process():
        dataset_info = inspect_faap_dataset(Path(args.dataset_root))
        with (output_dir / "dataset_layout.json").open("w") as f:
            json.dump(dataset_info, f, indent=2)
    
    # ===========================================================
    # Model Initialization
    # ===========================================================
    
    # Frozen DETR backbone
    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    
    # Perturbation Generator (Student component)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)
    
    # DINO Projection Head (for Student)
    # Input: matched detection scores (scalar per detection)
    student_head = DINOHead(
        in_dim=1,  # Single score per detection
        hidden_dim=args.dino_hidden_dim,
        out_dim=args.dino_out_dim,
    ).to(device)
    
    # Teacher Head (EMA of Student)
    teacher_ema = EMATeacher(student_head, momentum=args.ema_momentum).to(device)
    
    # DINO Loss with Centering
    dino_loss_fn = DINOLoss(
        out_dim=args.dino_out_dim,
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp,
        center_momentum=args.center_momentum,
    ).to(device)
    
    # Optional: Gender Discriminator
    discriminator = None
    opt_d = None
    if args.use_discriminator:
        discriminator = GenderDiscriminator(feature_dim=detr.hidden_dim).to(device)
    
    # DDP wrapping
    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)
        student_head = DDP(student_head, device_ids=[args.gpu] if args.gpu is not None else None)
        if discriminator is not None:
            discriminator = DDP(discriminator, device_ids=[args.gpu] if args.gpu is not None else None)
    
    # Optimizers
    opt_g = torch.optim.AdamW(
        list(_unwrap_ddp(generator).parameters()) + 
        list(_unwrap_ddp(student_head).parameters()),
        lr=args.lr_g,
        weight_decay=0.04,  # DINO uses weight decay
    )
    
    if discriminator is not None:
        opt_d = torch.optim.Adam(_unwrap_ddp(discriminator).parameters(), lr=args.lr_d)
    
    # ===========================================================
    # Resume from checkpoint
    # ===========================================================
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if "generator" in ckpt:
            _unwrap_ddp(generator).load_state_dict(ckpt["generator"])
        if "student_head" in ckpt:
            _unwrap_ddp(student_head).load_state_dict(ckpt["student_head"])
        if "teacher_head" in ckpt:
            teacher_ema.teacher.load_state_dict(ckpt["teacher_head"])
        if "dino_center" in ckpt:
            dino_loss_fn.center.center = ckpt["dino_center"].to(device)
            dino_loss_fn.center.initialized = True
        if "opt_g" in ckpt:
            opt_g.load_state_dict(ckpt["opt_g"])
        if "discriminator" in ckpt and discriminator is not None:
            _unwrap_ddp(discriminator).load_state_dict(ckpt["discriminator"])
        if "opt_d" in ckpt and opt_d is not None:
            opt_d.load_state_dict(ckpt["opt_d"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        
        if utils.is_main_process():
            print(f"Resumed from {args.resume} at epoch {start_epoch}")
    
    # ===========================================================
    # DataLoader
    # ===========================================================
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
    
    # ===========================================================
    # Training Loop
    # ===========================================================
    log_path = output_dir / "train_log.jsonl"
    
    for epoch in range(start_epoch, args.epochs):
        metrics_logger = utils.MetricLogger(delimiter="  ")
        generator.train()
        student_head.train()
        if discriminator is not None:
            discriminator.train()
        
        if args.distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        
        # Schedule values
        current_eps = _scheduled_epsilon(
            epoch,
            args.epsilon_warmup_epochs,
            args.epsilon_hold_epochs,
            args.epsilon_cooldown_epochs,
            args.epsilon,
            args.epsilon_final,
            args.epsilon_min,
        )
        _set_generator_epsilon(generator, current_eps)
        
        current_lambda_det = _scheduled_value(
            epoch, args.epochs, args.lambda_det, args.lambda_det_final
        )
        
        # DINO-specific schedules
        current_ema_momentum = _cosine_scheduler(
            args.ema_momentum, args.ema_momentum_final, args.epochs, epoch
        )
        teacher_ema.momentum = current_ema_momentum
        
        current_teacher_temp = _scheduled_value(
            epoch, args.epochs, args.teacher_temp_warmup, args.teacher_temp_final
        )
        dino_loss_fn.teacher_temp = current_teacher_temp
        
        # Training iteration
        for samples, targets, genders in metrics_logger.log_every(
            train_loader, args.log_every, f"Epoch {epoch}"
        ):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            genders = [g.lower() for g in genders]
            
            # Split by gender
            female_idx = [i for i, g in enumerate(genders) if g == "female"]
            male_idx = [i for i, g in enumerate(genders) if g == "male"]
            female_batch, female_targets = _split_nested(samples, targets, female_idx)
            male_batch, male_targets = _split_nested(samples, targets, male_idx)
            
            # Initialize metrics
            dino_loss = torch.tensor(0.0, device=device)
            det_loss = torch.tensor(0.0, device=device)
            adv_loss = torch.tensor(0.0, device=device)
            d_loss = torch.tensor(0.0, device=device)
            entropy_loss = torch.tensor(0.0, device=device)
            total_g = torch.tensor(0.0, device=device)
            delta_linf = torch.tensor(0.0, device=device)
            delta_l2 = torch.tensor(0.0, device=device)
            obj_mean_f = torch.tensor(0.0, device=device)
            obj_mean_m = torch.tensor(0.0, device=device)
            obj_frac_f = torch.tensor(0.0, device=device)
            obj_frac_m = torch.tensor(0.0, device=device)
            
            # Skip if either gender is missing (need both for DINO)
            if female_batch is None or male_batch is None:
                metrics_logger.update(
                    dino_loss=0.0, det_loss=0.0, adv_loss=0.0, d_loss=0.0,
                    entropy_loss=0.0, total_g=0.0, eps=current_eps,
                    lambda_det=current_lambda_det, ema_momentum=current_ema_momentum,
                    teacher_temp=current_teacher_temp, delta_linf=0.0, delta_l2=0.0,
                    obj_score_f=0.0, obj_score_m=0.0, obj_frac_f=0.0, obj_frac_m=0.0,
                )
                continue
            
            # =========================================================
            # Step 1: Optional Discriminator Update
            # =========================================================
            if discriminator is not None and args.use_discriminator:
                for _ in range(args.k_d):
                    opt_d.zero_grad()
                    
                    with torch.no_grad():
                        female_perturbed = _apply_generator(generator, female_batch)
                        male_perturbed = _apply_generator(generator, male_batch)
                        _, feat_f = detr.forward_with_features(female_perturbed)
                        _, feat_m = detr.forward_with_features(male_perturbed)
                    
                    logits_f = discriminator(feat_f.detach())
                    logits_m = discriminator(feat_m.detach())
                    labels_f = torch.ones(logits_f.size(0), device=device, dtype=torch.long)
                    labels_m = torch.zeros(logits_m.size(0), device=device, dtype=torch.long)
                    
                    d_loss = (
                        F.cross_entropy(logits_f, labels_f) +
                        F.cross_entropy(logits_m, labels_m)
                    ) / 2
                    
                    d_loss.backward()
                    opt_d.step()
            
            # =========================================================
            # Step 2: Generator + Student Head Update with DINO Loss
            # =========================================================
            opt_g.zero_grad()
            
            # Generate perturbed images
            female_perturbed = _apply_generator(generator, female_batch)
            male_perturbed = _apply_generator(generator, male_batch)
            
            # Get DETR outputs
            outputs_f, feat_f = detr.forward_with_features(female_perturbed)
            outputs_m, feat_m = detr.forward_with_features(male_perturbed)
            
            # Extract matched detection scores
            female_scores = _matched_detection_scores(detr, outputs_f, female_targets)
            male_scores = _matched_detection_scores(detr, outputs_m, male_targets)
            
            # =========================================================
            # DINO Distillation Loss
            # Teacher (남성) → Student (여성) distillation
            # =========================================================
            if female_scores.numel() > 0 and male_scores.numel() > 0:
                # Align dimensions for DINO
                min_scores = min(female_scores.numel(), male_scores.numel())
                
                # Sample to match dimensions (or use all if similar)
                if female_scores.numel() > min_scores:
                    perm_f = torch.randperm(female_scores.numel(), device=device)[:min_scores]
                    female_scores_sampled = female_scores[perm_f]
                else:
                    female_scores_sampled = female_scores
                
                if male_scores.numel() > min_scores:
                    perm_m = torch.randperm(male_scores.numel(), device=device)[:min_scores]
                    male_scores_sampled = male_scores[perm_m]
                else:
                    male_scores_sampled = male_scores
                
                # Student projection (여성 scores)
                student_proj = student_head(female_scores_sampled.unsqueeze(-1))
                
                # Teacher projection (남성 scores) - no gradient
                with torch.no_grad():
                    teacher_proj = teacher_ema.forward(male_scores_sampled.unsqueeze(-1))
                
                # DINO Loss: Student가 Teacher를 따라가도록
                dino_loss = dino_loss_fn(student_proj, teacher_proj, update_center=True)
                
                # Entropy regularization: 다양성 유지
                # Student 출력의 엔트로피를 높여서 collapse 방지
                student_probs = F.softmax(student_proj / args.student_temp, dim=-1)
                entropy_loss = -torch.sum(
                    student_probs * torch.log(student_probs + 1e-8), dim=-1
                ).mean()
                # 엔트로피를 높이고 싶으므로 음수 부호 (maximize entropy)
                entropy_loss = -entropy_loss
            
            # =========================================================
            # Detection Preserving Loss
            # =========================================================
            det_loss_f, _ = detr.detection_loss(outputs_f, female_targets)
            det_loss_m, _ = detr.detection_loss(outputs_m, male_targets)
            det_loss = (det_loss_f + det_loss_m) / 2
            
            # =========================================================
            # Optional: Adversarial Loss
            # =========================================================
            if discriminator is not None and args.use_discriminator:
                logits_f_adv = discriminator(feat_f)
                logits_m_adv = discriminator(feat_m)
                # Fool the discriminator: female → male label (0)
                adv_loss = F.cross_entropy(
                    logits_f_adv, 
                    torch.zeros(logits_f_adv.size(0), device=device, dtype=torch.long)
                )
            
            # =========================================================
            # Total Generator Loss
            # =========================================================
            total_g = (
                args.lambda_dino * dino_loss +
                current_lambda_det * det_loss +
                args.lambda_entropy * entropy_loss
            )
            if args.use_discriminator:
                total_g = total_g + args.lambda_adv * adv_loss
            
            # =========================================================
            # Compute Metrics
            # =========================================================
            with torch.no_grad():
                # Perturbation magnitude
                delta_f = female_perturbed.tensors - female_batch.tensors
                delta_m = male_perturbed.tensors - male_batch.tensors
                delta_cat = torch.cat([delta_f, delta_m], dim=0)
                delta_linf = delta_cat.abs().amax(dim=(1, 2, 3)).mean()
                delta_l2 = delta_cat.flatten(1).norm(p=2, dim=1).mean()
                
                # Objectness scores
                probs_f = outputs_f["pred_logits"].softmax(dim=-1)[..., :-1]
                probs_m = outputs_m["pred_logits"].softmax(dim=-1)[..., :-1]
                max_scores_f = probs_f.max(dim=-1).values
                max_scores_m = probs_m.max(dim=-1).values
                obj_mean_f = max_scores_f.mean()
                obj_mean_m = max_scores_m.mean()
                obj_frac_f = (max_scores_f > args.obj_conf_thresh).float().mean()
                obj_frac_m = (max_scores_m > args.obj_conf_thresh).float().mean()
            
            # =========================================================
            # Backward & Optimize
            # =========================================================
            total_g.backward()
            if args.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(generator.parameters()) + list(student_head.parameters()),
                    args.max_norm,
                )
            opt_g.step()
            
            # =========================================================
            # Update Teacher (EMA)
            # =========================================================
            teacher_ema.update(_unwrap_ddp(student_head))
            
            # =========================================================
            # Log Metrics
            # =========================================================
            metrics_logger.update(
                dino_loss=dino_loss.item(),
                det_loss=det_loss.item(),
                adv_loss=adv_loss.item() if args.use_discriminator else 0.0,
                d_loss=d_loss.item() if args.use_discriminator else 0.0,
                entropy_loss=entropy_loss.item(),
                total_g=total_g.item(),
                eps=current_eps,
                lambda_det=current_lambda_det,
                ema_momentum=current_ema_momentum,
                teacher_temp=current_teacher_temp,
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                obj_score_f=obj_mean_f.item(),
                obj_score_m=obj_mean_m.item(),
                obj_frac_f=obj_frac_f.item(),
                obj_frac_m=obj_frac_m.item(),
            )
        
        # ===========================================================
        # End of Epoch
        # ===========================================================
        metrics_logger.synchronize_between_processes()
        
        if utils.is_main_process():
            # Log to file
            log_entry = {
                "epoch": epoch,
                "dino_loss": metrics_logger.meters["dino_loss"].global_avg,
                "det_loss": metrics_logger.meters["det_loss"].global_avg,
                "adv_loss": metrics_logger.meters["adv_loss"].global_avg,
                "d_loss": metrics_logger.meters["d_loss"].global_avg,
                "entropy_loss": metrics_logger.meters["entropy_loss"].global_avg,
                "total_g": metrics_logger.meters["total_g"].global_avg,
                "epsilon": current_eps,
                "lambda_det": current_lambda_det,
                "ema_momentum": current_ema_momentum,
                "teacher_temp": current_teacher_temp,
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "obj_score_f": metrics_logger.meters["obj_score_f"].global_avg,
                "obj_score_m": metrics_logger.meters["obj_score_m"].global_avg,
                "obj_frac_f": metrics_logger.meters["obj_frac_f"].global_avg,
                "obj_frac_m": metrics_logger.meters["obj_frac_m"].global_avg,
            }
            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            # Print summary
            print(f"\n[Epoch {epoch}] Summary:")
            print(f"  DINO Loss: {log_entry['dino_loss']:.4f}")
            print(f"  Detection Loss: {log_entry['det_loss']:.4f}")
            print(f"  Entropy Loss: {log_entry['entropy_loss']:.4f}")
            print(f"  Total G: {log_entry['total_g']:.4f}")
            print(f"  Epsilon: {current_eps:.4f}")
            print(f"  EMA Momentum: {current_ema_momentum:.4f}")
            print(f"  Teacher Temp: {current_teacher_temp:.4f}")
            print(f"  Female Obj Score: {log_entry['obj_score_f']:.4f}")
            print(f"  Male Obj Score: {log_entry['obj_score_m']:.4f}")
            print(f"  Score Gap (M-F): {log_entry['obj_score_m'] - log_entry['obj_score_f']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % args.save_every == 0:
                ckpt_path = output_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
                save_dict = {
                    "epoch": epoch,
                    "generator": _unwrap_ddp(generator).state_dict(),
                    "student_head": _unwrap_ddp(student_head).state_dict(),
                    "teacher_head": teacher_ema.teacher.state_dict(),
                    "dino_center": dino_loss_fn.center.center,
                    "opt_g": opt_g.state_dict(),
                    "args": vars(args),
                }
                if discriminator is not None:
                    save_dict["discriminator"] = _unwrap_ddp(discriminator).state_dict()
                    save_dict["opt_d"] = opt_d.state_dict()
                
                torch.save(save_dict, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")
        
        if args.distributed:
            dist.barrier()
    
    # ===========================================================
    # Training Complete
    # ===========================================================
    if utils.is_main_process():
        print("\n" + "=" * 60)
        print("DINO-based Fair Object Detection Training Complete!")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Total epochs: {args.epochs}")
        print("\nKey DINO components used:")
        print("  - Centering: Prevented mode collapse")
        print("  - Sharpening: Teacher temp < Student temp")
        print("  - EMA Teacher: Stable target distribution")
        print("  - Cross-entropy distillation: Female → Male distribution")


if __name__ == "__main__":
    main()
