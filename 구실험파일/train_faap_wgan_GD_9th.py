import argparse  # 커맨드라인 인자 파싱용 표준 라이브러리
import json  # JSON 파일 입출력 처리
from pathlib import Path  # 경로를 객체로 다루기 위한 표준 라이브러리
from typing import List, Sequence  # 타입 힌트: 리스트와 시퀀스 타입 지정

# Allow running as a script: add package root to sys.path
if __package__ is None or __package__ == "":  # 스크립트 단독 실행 시 패키지 경로 보정
    import sys  # sys.path 수정 위해 임포트

    pkg_dir = Path(__file__).resolve().parent  # 현재 파일이 있는 디렉터리 절대경로
    parent = pkg_dir.parent  # 상위 디렉터리 (패키지 루트)
    if str(parent) not in sys.path:  # 모듈 검색 경로에 상위 디렉터리가 없으면
        sys.path.append(str(parent))  # 상위 디렉터리 추가
    if str(pkg_dir) not in sys.path:  # 현재 디렉터리가 없으면
        sys.path.append(str(pkg_dir))  # 현재 디렉터리 추가
    __package__ = "faap_gan"  # 패키지 이름 설정해 상대 임포트가 동작하도록 지정

import torch  # PyTorch 핵심 라이브러리
import torch.distributed as dist  # 분산 학습 유틸리티
import torch.nn.functional as F  # 다양한 손실/활성화 함수 모음
from torch import nn  # 신경망 모듈 베이스
from torch.nn.parallel import DistributedDataParallel as DDP  # 분산 학습용 래퍼

from .datasets import build_faap_dataloader, inspect_faap_dataset  # 데이터로더 생성/점검 함수
from .models import FrozenDETR, GenderDiscriminator, PerturbationGenerator, clamp_normalized  # DETR 래퍼, 생성기, 성별 판별기, 정규화 보정 함수
from .path_utils import DETR_REPO, default_detr_checkpoint, ensure_detr_repo_on_path  # DETR 경로/체크포인트 관련 헬퍼
import util.misc as utils  # DETR 프로젝트의 공용 유틸리티 (분산, 로깅 등)
from util.misc import NestedTensor  # 이미지와 마스크를 함께 담는 자료형


def _default_output_dir(script_path: Path) -> str:
    """스크립트 파일 이름을 기반으로 기본 output_dir을 생성한다.

    예) train_faap_wgan_GD_2nd.py → faap_outputs/faap_outputs_gd_2nd
    """
    stem = script_path.stem
    stem_lower = stem.lower()
    suffix = stem
    for prefix in ("train_faap_wgan_", "train_"):
        if stem_lower.startswith(prefix):
            suffix = stem[len(prefix) :]
            break
    suffix = suffix.lower()
    return str(Path("faap_outputs") / f"faap_outputs_{suffix}")


def parse_args() -> argparse.Namespace:  # 커맨드라인 인자를 파싱해 Namespace로 반환
    parser = argparse.ArgumentParser(
        "FAAP-style training for DETR (9th version - Quantile Matching + Adaptive LR)",
        add_help=True
    )
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")  # 데이터셋 루트 경로
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO), help="path to DETR repository (for imports/checkpoint)")  # DETR 코드 위치
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()), help="path to DETR pretrained checkpoint")  # 사전학습 체크포인트 경로
    parser.add_argument(
        "--output_dir",
        type=str,
        default=_default_output_dir(Path(__file__)),
        help="출력 저장 디렉터리 (기본: 스크립트 이름 기반 faap_outputs/faap_outputs_*)",
    )
    parser.add_argument("--device", type=str, default="cuda")  # 기본 디바이스 설정
    parser.add_argument("--epochs", type=int, default=28)  # 학습 에폭 수 (7th 24 → 28로 증가)
    parser.add_argument("--batch_size", type=int, default=4)  # 배치 크기
    parser.add_argument("--num_workers", type=int, default=6)  # 데이터로더 워커 수
    parser.add_argument("--lr_g", type=float, default=1e-4)  # 생성기 학습률
    parser.add_argument("--lr_d", type=float, default=1e-4)  # 판별기 학습률
    parser.add_argument("--k_d", type=int, default=4, help="discriminator steps per iteration")  # 판별기 업데이트 반복 횟수
    
    # ===== 7th 성공 요소 유지: Epsilon 스케줄 (warmup + hold + cooldown) =====
    parser.add_argument("--epsilon", type=float, default=0.05, help="starting epsilon for warmup")  # 교란 크기 초기값
    parser.add_argument("--epsilon_final", type=float, default=0.10, help="peak epsilon after warmup")  # 워밍업 후 최대 epsilon
    parser.add_argument("--epsilon_warmup_epochs", type=int, default=8, help="epochs to linearly warm epsilon")  # epsilon 선형 증가 에폭
    parser.add_argument("--epsilon_hold_epochs", type=int, default=8, help="epochs to keep epsilon_final before cooldown")  # epsilon 유지 에폭 (6→8)
    parser.add_argument("--epsilon_cooldown_epochs", type=int, default=12, help="epochs to linearly cool epsilon")  # epsilon 감소 에폭 (10→12)
    parser.add_argument("--epsilon_min", type=float, default=0.08, help="minimum epsilon after cooldown")  # 쿨다운 후 최소 epsilon (0.09→0.08)

    parser.add_argument("--alpha", type=float, default=0.2, help="entropy weight for fairness term")  # 엔트로피 가중치

    # ===== 7th 성공 요소 유지: Beta 스케줄 =====
    parser.add_argument("--beta", type=float, default=0.5, help="detection-preserving loss weight")  # 검출 손실 시작 가중치
    parser.add_argument("--beta_final", type=float, default=0.65, help="final detection-preserving loss weight")  # 검출 손실 최종 가중치 (0.6→0.65)

    parser.add_argument(
        "--lambda_fair",
        type=float,
        default=2.0,
        help="weight for fairness loss (adversarial + entropy)",
    )
    
    # ===== 7th 성공 요소 유지: 비대칭 fairness 스케일링 =====
    parser.add_argument(
        "--fair_f_scale",
        type=float,
        default=1.0,
        help="female fairness scaling factor",
    )
    parser.add_argument(
        "--fair_m_scale",
        type=float,
        default=0.5,
        help="male fairness scaling factor",
    )

    # ===== Wasserstein 손실 설정 (7th와 동일) =====
    parser.add_argument(
        "--lambda_w",
        type=float,
        default=0.2,
        help="weight for Wasserstein alignment (단방향)",
    )
    
    # ===== 9th 신규: Quantile Matching Loss =====
    parser.add_argument(
        "--lambda_q",
        type=float,
        default=0.15,
        help="weight for quantile matching loss (AP Gap 개선)",
    )
    parser.add_argument(
        "--quantile_levels",
        type=int,
        default=5,
        help="number of quantile levels for matching (e.g., 5 = [0.1, 0.3, 0.5, 0.7, 0.9])",
    )
    
    # ===== 9th 신규: Learning Rate Decay (후반부 안정화) =====
    parser.add_argument(
        "--lr_decay_epoch",
        type=int,
        default=18,
        help="epoch to start learning rate decay",
    )
    parser.add_argument(
        "--lr_decay_factor",
        type=float,
        default=0.5,
        help="learning rate decay factor (applied once)",
    )
    
    # ===== 9th 신규: Score Gap Penalty =====
    parser.add_argument(
        "--lambda_gap",
        type=float,
        default=0.1,
        help="weight for direct score gap penalty",
    )
    
    parser.add_argument(
        "--max_train_per_gender",
        type=int,
        default=0,
        help="per-epoch cap per gender (without replacement) for balanced sampling; set to 0 to disable",
    )

    parser.add_argument("--obj_conf_thresh", type=float, default=0.5, help="objectness threshold for logging recall proxy")  # 객체성 점수 임계값
    parser.add_argument("--max_norm", type=float, default=0.1, help="gradient clipping for G")  # 그래디언트 클리핑 한계
    parser.add_argument("--log_every", type=int, default=10)  # 로깅 간격
    parser.add_argument("--save_every", type=int, default=1)  # 체크포인트 저장 간격(에폭 단위)
    parser.add_argument("--seed", type=int, default=42)  # 랜덤 시드
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint to resume G/D/optim state")  # 재시작용 체크포인트
    parser.add_argument("--distributed", action="store_true", help="force distributed mode even if env vars are missing")  # 분산 학습 강제 플래그
    parser.add_argument("--world_size", default=1, type=int, help="number of processes for distributed training")  # 전체 프로세스 수
    parser.add_argument("--rank", default=0, type=int, help="rank of the process")  # 현재 프로세스 랭크
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for distributed launchers")  # 로컬 랭크(런처용)
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")  # 분산 초기화 URL
    return parser.parse_args()  # 파싱 수행 후 Namespace 반환


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


def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:
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
    """Epsilon 스케줄: warmup → hold → cooldown (7th와 동일한 구조)"""
    if warmup_epochs <= 1:
        warmup_end = 0
    else:
        warmup_end = warmup_epochs - 1

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
    """Beta 스케줄: 선형 증가 (7th와 동일)"""
    if total_epochs <= 1 or beta_start == beta_final:
        return beta_start
    progress = min(epoch / max(1, total_epochs - 1), 1.0)
    return beta_start + (beta_final - beta_start) * progress


def _unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def _set_generator_epsilon(generator: nn.Module, epsilon: float) -> None:
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


def _wasserstein_1d(female_scores: torch.Tensor, male_scores: torch.Tensor) -> torch.Tensor:
    """단방향 Wasserstein 손실: 여성 score가 남성보다 낮을 때만 패널티.
    
    7th에서 검증된 핵심 메커니즘 유지:
    - Female < Male일 때만 손실 발생
    - 남성 성능은 detach로 타겟 역할 (보호)
    """
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0, device=female_scores.device)
    sorted_f = female_scores.sort().values
    sorted_m = male_scores.detach().sort().values  # 남성은 detach로 타겟 역할
    k = max(sorted_f.numel(), sorted_m.numel())
    sorted_f = _resize_sorted(sorted_f, k)
    sorted_m = _resize_sorted(sorted_m, k)
    # 단방향: 여성이 남성보다 낮을 때만 손실 발생 (여성 → 남성 방향 이동)
    return F.relu(sorted_m - sorted_f).mean()


def _quantile_matching_loss(
    female_scores: torch.Tensor,
    male_scores: torch.Tensor,
    num_quantiles: int = 5
) -> torch.Tensor:
    """9th 신규: Quantile Matching Loss
    
    Wasserstein은 전체 분포를 정렬하지만, Quantile Matching은
    특정 분위수(예: 10%, 30%, 50%, 70%, 90%)에서의 차이를 직접 최소화.
    
    이점:
    - 분포의 특정 부분(예: 하위 score)에 더 집중 가능
    - AP 계산에 중요한 높은 score 영역도 정밀 정렬
    - Wasserstein보다 해석 가능하고 안정적
    
    Args:
        female_scores: 여성 그룹의 detection confidence scores
        male_scores: 남성 그룹의 detection confidence scores  
        num_quantiles: 사용할 분위수 개수 (기본 5 = [0.1, 0.3, 0.5, 0.7, 0.9])
    
    Returns:
        단방향 quantile 차이 손실 (female을 male 방향으로 끌어올림)
    """
    if female_scores.numel() < num_quantiles or male_scores.numel() < num_quantiles:
        return female_scores.new_tensor(0.0, device=female_scores.device)
    
    # 분위수 레벨 생성 (예: [0.1, 0.3, 0.5, 0.7, 0.9] for num_quantiles=5)
    quantile_levels = torch.linspace(
        1.0 / (num_quantiles + 1),
        num_quantiles / (num_quantiles + 1),
        num_quantiles,
        device=female_scores.device
    )
    
    # 각 그룹의 분위수 계산
    q_female = torch.quantile(female_scores, quantile_levels)
    q_male = torch.quantile(male_scores.detach(), quantile_levels)  # 남성은 타겟
    
    # 단방향 손실: 여성 분위수가 남성보다 낮을 때만 패널티
    # 높은 분위수(예: 70%, 90%)에 더 높은 가중치 부여 → AP 개선에 효과적
    weights = quantile_levels  # [0.1, 0.3, 0.5, 0.7, 0.9] - 높은 분위수에 높은 가중치
    weighted_diff = weights * F.relu(q_male - q_female)
    
    return weighted_diff.mean()


def _score_gap_penalty(
    female_scores: torch.Tensor,
    male_scores: torch.Tensor
) -> torch.Tensor:
    """9th 신규: Score Gap Penalty
    
    성별 간 평균 score 차이를 직접 패널티로 부여.
    Wasserstein/Quantile은 분포 형태를 정렬하지만,
    이 손실은 단순히 평균 차이를 줄이는 데 집중.
    
    단방향: 여성 평균이 남성보다 낮을 때만 손실 발생
    """
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0, device=female_scores.device)
    
    mean_f = female_scores.mean()
    mean_m = male_scores.detach().mean()  # 남성은 타겟
    
    # 단방향: 여성 평균이 남성보다 낮을 때만 패널티
    return F.relu(mean_m - mean_f)


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
            raise RuntimeError("Distributed training requires CUDA to be available.")
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

    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)
    discriminator = GenderDiscriminator(feature_dim=detr.hidden_dim).to(device)
    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)
        discriminator = DDP(discriminator, device_ids=[args.gpu] if args.gpu is not None else None)

    opt_g = torch.optim.Adam(_unwrap_ddp(generator).parameters(), lr=args.lr_g)
    opt_d = torch.optim.Adam(_unwrap_ddp(discriminator).parameters(), lr=args.lr_d)

    start_epoch = 0
    lr_decayed = False  # LR decay 적용 여부 추적
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if "generator" in ckpt:
            _unwrap_ddp(generator).load_state_dict(ckpt["generator"])
        if "discriminator" in ckpt:
            _unwrap_ddp(discriminator).load_state_dict(ckpt["discriminator"])
        if "opt_g" in ckpt:
            opt_g.load_state_dict(ckpt["opt_g"])
        if "opt_d" in ckpt:
            opt_d.load_state_dict(ckpt["opt_d"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "lr_decayed" in ckpt:
            lr_decayed = ckpt["lr_decayed"]
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
    metrics_logger = utils.MetricLogger(delimiter="  ")
    
    for epoch in range(start_epoch, args.epochs):
        metrics_logger = utils.MetricLogger(delimiter="  ")
        generator.train()
        discriminator.train()
        if args.distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        
        # ===== 9th 신규: Learning Rate Decay (한 번만 적용) =====
        if epoch >= args.lr_decay_epoch and not lr_decayed:
            for param_group in opt_g.param_groups:
                param_group['lr'] *= args.lr_decay_factor
            for param_group in opt_d.param_groups:
                param_group['lr'] *= args.lr_decay_factor
            lr_decayed = True
            if utils.is_main_process():
                print(f"[Epoch {epoch}] Learning rate decayed by factor {args.lr_decay_factor}")
        
        # ===== 7th 스케줄링 유지 =====
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
        
        # 현재 LR 추적
        current_lr_g = opt_g.param_groups[0]['lr']
        current_lr_d = opt_d.param_groups[0]['lr']

        for samples, targets, genders in metrics_logger.log_every(train_loader, args.log_every, f"Epoch {epoch}"):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            genders = [g.lower() for g in genders]

            female_idx = [i for i, g in enumerate(genders) if g == "female"]
            male_idx = [i for i, g in enumerate(genders) if g == "male"]
            female_batch, female_targets = _split_nested(samples, targets, female_idx)
            male_batch, male_targets = _split_nested(samples, targets, male_idx)

            delta_linf = torch.tensor(0.0, device=device)
            delta_l2 = torch.tensor(0.0, device=device)
            obj_mean = torch.tensor(0.0, device=device)
            obj_frac = torch.tensor(0.0, device=device)
            obj_mean_f = torch.tensor(0.0, device=device)
            obj_frac_f = torch.tensor(0.0, device=device)
            obj_mean_m = torch.tensor(0.0, device=device)
            obj_frac_m = torch.tensor(0.0, device=device)
            wasserstein_loss = torch.tensor(0.0, device=device)
            quantile_loss = torch.tensor(0.0, device=device)
            gap_loss = torch.tensor(0.0, device=device)

            # -- update discriminator --
            for _ in range(args.k_d):
                d_losses = []
                opt_d.zero_grad()
                if female_batch is not None:
                    with torch.no_grad():
                        female_perturbed_d = _apply_generator(generator, female_batch)
                        _, feat_f = detr.forward_with_features(female_perturbed_d)
                    logits_f = discriminator(feat_f.detach())
                    labels_f = torch.ones(logits_f.size(0), device=device, dtype=torch.long)
                    d_losses.append(F.cross_entropy(logits_f, labels_f))
                if male_batch is not None:
                    with torch.no_grad():
                        male_perturbed_d = _apply_generator(generator, male_batch)
                        _, feat_m = detr.forward_with_features(male_perturbed_d)
                    logits_m = discriminator(feat_m.detach())
                    labels_m = torch.zeros(logits_m.size(0), device=device, dtype=torch.long)
                    d_losses.append(F.cross_entropy(logits_m, labels_m))

                if d_losses:
                    d_loss = torch.stack(d_losses).mean()
                    d_loss.backward()
                    opt_d.step()
                else:
                    d_loss = torch.tensor(0.0, device=device)

            # -- update generator (female + male) --
            if female_batch is not None or male_batch is not None:
                opt_g.zero_grad()
                fairness_loss = torch.tensor(0.0, device=device)
                fairness_f = torch.tensor(0.0, device=device)
                fairness_m = torch.tensor(0.0, device=device)
                det_loss = torch.tensor(0.0, device=device)
                total_g = torch.tensor(0.0, device=device)

                female_scores = torch.tensor([], device=device)
                male_scores = torch.tensor([], device=device)

                if female_batch is not None:
                    female_perturbed = _apply_generator(generator, female_batch)
                    outputs_f, feat_f = detr.forward_with_features(female_perturbed)
                    logits_f = discriminator(feat_f)
                    ce_f = F.cross_entropy(logits_f, torch.ones(logits_f.size(0), device=device, dtype=torch.long))
                    ent_f = _entropy_loss(logits_f)
                    fairness_f = -(ce_f + args.alpha * ent_f)
                    det_f, _ = detr.detection_loss(outputs_f, female_targets)
                    female_scores = _matched_detection_scores(detr, outputs_f, female_targets)
                    det_loss = det_loss + det_f

                if male_batch is not None:
                    male_perturbed = _apply_generator(generator, male_batch)
                    outputs_m, feat_m = detr.forward_with_features(male_perturbed)
                    logits_m = discriminator(feat_m)
                    ce_m = F.cross_entropy(logits_m, torch.zeros(logits_m.size(0), device=device, dtype=torch.long))
                    ent_m = _entropy_loss(logits_m)
                    fairness_m = -(ce_m + args.alpha * ent_m)
                    det_m, _ = detr.detection_loss(outputs_m, male_targets)
                    male_scores = _matched_detection_scores(detr, outputs_m, male_targets)
                    det_loss = det_loss + det_m

                # ===== 7th 핵심: 비대칭 fairness 스케일링 (유지) =====
                fairness_loss = args.fair_f_scale * fairness_f + args.fair_m_scale * fairness_m
                
                # ===== Score Alignment Losses =====
                # 1. Wasserstein Loss (7th 유지)
                wasserstein_loss = _wasserstein_1d(female_scores, male_scores)
                
                # 2. Quantile Matching Loss (9th 신규)
                quantile_loss = _quantile_matching_loss(
                    female_scores, male_scores, args.quantile_levels
                )
                
                # 3. Score Gap Penalty (9th 신규)
                gap_loss = _score_gap_penalty(female_scores, male_scores)

                # 최종 Generator 손실
                total_g = (
                    args.lambda_fair * fairness_loss
                    + current_beta * det_loss
                    + args.lambda_w * wasserstein_loss
                    + args.lambda_q * quantile_loss
                    + args.lambda_gap * gap_loss
                )

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
                    else:
                        obj_mean = torch.tensor(0.0, device=device)
                        obj_frac = torch.tensor(0.0, device=device)

                total_g.backward()
                if args.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.max_norm)
                opt_g.step()
            else:
                fairness_loss = torch.tensor(0.0, device=device)
                det_loss = torch.tensor(0.0, device=device)
                total_g = torch.tensor(0.0, device=device)

            metrics_logger.update(
                d_loss=d_loss.item(),
                g_fair=fairness_loss.item(),
                g_det=det_loss.item(),
                g_total=total_g.item(),
                g_w=wasserstein_loss.item(),
                g_q=quantile_loss.item(),
                g_gap=gap_loss.item(),
                eps=current_eps,
                beta=current_beta,
                lr_g=current_lr_g,
                lr_d=current_lr_d,
                fair_f_scale=args.fair_f_scale,
                fair_m_scale=args.fair_m_scale,
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
                "d_loss": metrics_logger.meters["d_loss"].global_avg,
                "g_fair": metrics_logger.meters["g_fair"].global_avg,
                "g_det": metrics_logger.meters["g_det"].global_avg,
                "g_total": metrics_logger.meters["g_total"].global_avg,
                "g_w": metrics_logger.meters["g_w"].global_avg,
                "g_q": metrics_logger.meters["g_q"].global_avg,
                "g_gap": metrics_logger.meters["g_gap"].global_avg,
                "epsilon": current_eps,
                "beta": current_beta,
                "lr_g": current_lr_g,
                "lr_d": current_lr_d,
                "fair_f_scale": metrics_logger.meters["fair_f_scale"].global_avg,
                "fair_m_scale": metrics_logger.meters["fair_m_scale"].global_avg,
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
                        "opt_g": opt_g.state_dict(),
                        "opt_d": opt_d.state_dict(),
                        "lr_decayed": lr_decayed,
                        "args": vars(args),
                    },
                    ckpt_path_save,
                )

        if args.distributed:
            dist.barrier()


if __name__ == "__main__":
    main()
