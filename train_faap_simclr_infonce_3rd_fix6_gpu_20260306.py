"""
FAAP Training - GAN + Matched Wasserstein (fix6) [2026-03-06]

=============================================================================
fix5 → fix6 변경사항:
=============================================================================
1. 기본 구조 교체: InfoNCE + Fair Centroid Contrastive → GAN Discriminator
   - Fair Centroid Loss 제거 (fix5에서 즉시 -10.0으로 포화되어 gradient 없음)
   - GenderDiscriminator 복원 (7th 구조)

2. delta_linf 측정 버그 수정
   - fix5: SimCLR aug 적용 후 perturbed - samples 계산 → aug 영향으로 0.91 같은 오염값
   - fix6: 7th와 동일하게 aug 없이 올바르게 측정

=============================================================================
7th → fix6 변경사항:
=============================================================================
1. lambda_w: 0.2 → 0.5  (AP Gap 더 강하게 공략)
2. BF16 AMP 추가: torch.autocast(dtype=torch.bfloat16)  (A100 GPU 효율)
3. TF32 활성화: allow_tf32=True  (A100 Tensor Core 최대 활용)
4. matched_score_gap 로그 추가: matched_score_f, matched_score_m, matched_score_gap
   (학습 중 AP Gap proxy를 직접 모니터링)

=============================================================================
7th에서 그대로 유지한 것:
=============================================================================
- GAN Discriminator 구조 (k_d=4, lambda_fair=2.0)
- 단방향 Wasserstein: ReLU(sorted_m - sorted_f) (male.detach())
- matched_detection_scores (7th에 이미 있었음)
- epsilon 3단계 스케줄 (warmup → hold → cooldown)
- beta 선형 증가 (0.5 → 0.6)
- gradient clipping (max_norm=0.1)
- balance_genders=True
=============================================================================
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

from .datasets import build_faap_dataloader, inspect_faap_dataset
from .models import FrozenDETR, GenderDiscriminator, PerturbationGenerator, clamp_normalized
from .path_utils import DETR_REPO, default_detr_checkpoint, ensure_detr_repo_on_path
import util.misc as utils
from util.misc import NestedTensor


def _default_output_dir(script_path: Path) -> str:
    stem = script_path.stem
    stem_lower = stem.lower()
    suffix = stem
    for prefix in ("train_faap_simclr_infonce_", "train_faap_wgan_", "train_faap_", "train_"):
        if stem_lower.startswith(prefix):
            suffix = stem[len(prefix):]
            break
    suffix = suffix.lower()
    return str(Path("faap_outputs") / f"faap_outputs_{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FAAP fix6: 7th GAN + Matched Wasserstein (lambda_w=0.5, BF16)")
    parser.add_argument("--dataset_root", type=str, default="/workspace/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()))
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(Path(__file__)))

    # 학습 기본 설정 (7th와 동일)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--k_d", type=int, default=4, help="discriminator steps per iteration")
    parser.add_argument("--seed", type=int, default=42)

    # Epsilon 3단계 스케줄 (7th와 동일)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--epsilon_final", type=float, default=0.10)
    parser.add_argument("--epsilon_warmup_epochs", type=int, default=8)
    parser.add_argument("--epsilon_hold_epochs", type=int, default=6)
    parser.add_argument("--epsilon_cooldown_epochs", type=int, default=10)
    parser.add_argument("--epsilon_min", type=float, default=0.09)

    # 손실 가중치
    parser.add_argument("--alpha", type=float, default=0.2, help="entropy weight for fairness term")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--beta_final", type=float, default=0.6)
    parser.add_argument("--lambda_fair", type=float, default=2.0)
    parser.add_argument("--fair_f_scale", type=float, default=1.0)
    parser.add_argument("--fair_m_scale", type=float, default=0.5)

    # [fix6 변경 1/4] lambda_w: 0.2 → 0.5
    # 7th에서 AP Gap 개선이 -0.4%로 미미했던 원인 중 하나가 Wasserstein 신호가 약했기 때문.
    # matched_detection_scores 기반 Wasserstein을 더 강하게 밀기 위해 0.5로 상향.
    parser.add_argument("--lambda_w", type=float, default=0.5,
                        help="[fix6 변경] Wasserstein 가중치: 7th 0.2 → fix6 0.5 (AP Gap 더 강하게 공략)")

    parser.add_argument("--max_train_per_gender", type=int, default=0)
    parser.add_argument("--obj_conf_thresh", type=float, default=0.5)
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
    """3단계 epsilon 스케줄: warmup → hold → cooldown (7th와 동일)"""
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


def _scheduled_beta(epoch: int, total_epochs: int, beta_start: float, beta_final: float) -> float:
    """beta 선형 증가: 0.5 → 0.6 (7th와 동일)"""
    if total_epochs <= 1 or beta_start == beta_final:
        return beta_start
    progress = min(epoch / max(1, total_epochs - 1), 1.0)
    return beta_start + (beta_final - beta_start) * progress


def _unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def _set_generator_epsilon(generator: nn.Module, epsilon: float) -> None:
    _unwrap_ddp(generator).epsilon = epsilon


def _resize_sorted(scores: torch.Tensor, target_len: int) -> torch.Tensor:
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


def _matched_detection_scores(
    detr: FrozenDETR,
    outputs: dict,
    targets: Sequence[dict],
) -> torch.Tensor:
    """Hungarian matching으로 TP prediction의 confidence score 추출.

    image-level top-k score 대신 실제 GT에 매칭된 prediction의 confidence를 사용.
    AP는 matched prediction의 confidence rank에 민감하므로 Wasserstein 신호가 더 직접적.
    (7th에 이미 있던 함수 — fix6에서 변경 없음)
    """
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


def _wasserstein_1d(female_scores: torch.Tensor, male_scores: torch.Tensor) -> torch.Tensor:
    """단방향 Wasserstein: 여성 score가 남성보다 낮을 때만 패널티.

    ReLU(sorted_m - sorted_f): 여성 < 남성일 때만 양수 → 남성 성능 유지 + 여성만 향상.
    male_scores.detach(): 남성 score는 타겟으로 고정, gradient 흐르지 않음.
    (7th와 동일 — fix6에서 변경 없음. 단, lambda_w를 0.2→0.5로 높여 신호 강화)
    """
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0)
    sorted_f = female_scores.sort().values
    sorted_m = male_scores.detach().sort().values
    k = max(sorted_f.numel(), sorted_m.numel())
    sorted_f = _resize_sorted(sorted_f, k)
    sorted_m = _resize_sorted(sorted_m, k)
    return F.relu(sorted_m - sorted_f).mean()


def main():
    args = parse_args()
    utils.init_distributed_mode(args)
    if not hasattr(args, "gpu"):
        args.gpu = None

    # [fix6 변경 2/4] TF32 활성화
    # A100 GPU에서 matmul/cudnn에 TF32 사용 → 속도 향상, 정확도 영향 미미
    # (7th에는 없던 설정)
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

    if utils.is_main_process():
        dataset_info = inspect_faap_dataset(Path(args.dataset_root))
        with (output_dir / "dataset_layout.json").open("w") as f:
            json.dump(dataset_info, f, indent=2)

        print("=" * 70)
        print("FAAP fix6: 7th GAN + Matched Wasserstein (lambda_w=0.5, BF16)")
        print("=" * 70)
        print("[7th 대비 변경사항]")
        print(f"  1. lambda_w: 0.2 → {args.lambda_w}  (AP Gap 더 강하게)")
        print("  2. BF16 AMP: torch.autocast 적용")
        print("  3. TF32 활성화")
        print("  4. matched_score_gap 로그 추가")
        print("[fix5 대비 변경사항]")
        print("  - InfoNCE/Fair Centroid 제거 (포화 문제)")
        print("  - GAN Discriminator 복원 (7th 구조)")
        print("  - delta_linf 측정 버그 수정")
        print("=" * 70)

    # 모델 초기화 (7th와 동일)
    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)
    discriminator = GenderDiscriminator(feature_dim=detr.hidden_dim).to(device)

    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)
        discriminator = DDP(discriminator, device_ids=[args.gpu] if args.gpu is not None else None)

    opt_g = torch.optim.Adam(_unwrap_ddp(generator).parameters(), lr=args.lr_g)
    opt_d = torch.optim.Adam(_unwrap_ddp(discriminator).parameters(), lr=args.lr_d)

    start_epoch = 0
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
        discriminator.train()

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

        for samples, targets, genders in metrics_logger.log_every(
            train_loader, args.log_every, f"Epoch {epoch}"
        ):
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
            # [fix6 변경 4/4] matched_score_gap 모니터링 변수 초기화
            matched_score_f = torch.tensor(0.0, device=device)
            matched_score_m = torch.tensor(0.0, device=device)

            # ================================================================
            # Discriminator 업데이트 (7th와 동일, k_d=4회)
            # ================================================================
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

            # ================================================================
            # Generator 업데이트
            # ================================================================
            if female_batch is not None or male_batch is not None:
                opt_g.zero_grad()
                fairness_loss = torch.tensor(0.0, device=device)
                fairness_f = torch.tensor(0.0, device=device)
                fairness_m = torch.tensor(0.0, device=device)
                det_loss = torch.tensor(0.0, device=device)
                female_scores = torch.tensor([], device=device)
                male_scores = torch.tensor([], device=device)

                # [fix6 변경 3/4] BF16 AMP 적용
                # 7th에는 autocast가 없었음. A100에서 BF16으로 forward 수행 → 속도 향상.
                # backward()는 autocast 밖에서 수행 (BF16은 GradScaler 불필요).
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if female_batch is not None:
                        female_perturbed = _apply_generator(generator, female_batch)
                        outputs_f, feat_f = detr.forward_with_features(female_perturbed)
                        logits_f = discriminator(feat_f)
                        ce_f = F.cross_entropy(
                            logits_f,
                            torch.ones(logits_f.size(0), device=device, dtype=torch.long),
                        )
                        ent_f = _entropy_loss(logits_f)
                        fairness_f = -(ce_f + args.alpha * ent_f)
                        det_f, _ = detr.detection_loss(outputs_f, female_targets)
                        det_loss = det_loss + det_f
                        female_scores = _matched_detection_scores(detr, outputs_f, female_targets)

                    if male_batch is not None:
                        male_perturbed = _apply_generator(generator, male_batch)
                        outputs_m, feat_m = detr.forward_with_features(male_perturbed)
                        logits_m = discriminator(feat_m)
                        ce_m = F.cross_entropy(
                            logits_m,
                            torch.zeros(logits_m.size(0), device=device, dtype=torch.long),
                        )
                        ent_m = _entropy_loss(logits_m)
                        fairness_m = -(ce_m + args.alpha * ent_m)
                        det_m, _ = detr.detection_loss(outputs_m, male_targets)
                        det_loss = det_loss + det_m
                        male_scores = _matched_detection_scores(detr, outputs_m, male_targets)

                    wasserstein_loss = _wasserstein_1d(female_scores, male_scores)
                    fairness_loss = args.fair_f_scale * fairness_f + args.fair_m_scale * fairness_m

                    total_g = (
                        args.lambda_fair * fairness_loss
                        + current_beta * det_loss
                        + args.lambda_w * wasserstein_loss  # lambda_w=0.5 (7th의 0.2에서 상향)
                    )

                # ============================================================
                # 메트릭 계산 (autocast 밖, no_grad)
                # delta_linf: 7th와 동일하게 perturbation만 측정 (aug 오염 없음)
                # ============================================================
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
                        probs_f = outputs_f["pred_logits"].float().softmax(dim=-1)[..., :-1]
                        max_scores_f = probs_f.max(dim=-1).values
                        obj_mean_f = max_scores_f.mean()
                        obj_frac_f = (max_scores_f > args.obj_conf_thresh).float().mean()
                        max_scores_list.append(max_scores_f)
                    if male_batch is not None:
                        probs_m = outputs_m["pred_logits"].float().softmax(dim=-1)[..., :-1]
                        max_scores_m = probs_m.max(dim=-1).values
                        obj_mean_m = max_scores_m.mean()
                        obj_frac_m = (max_scores_m > args.obj_conf_thresh).float().mean()
                        max_scores_list.append(max_scores_m)
                    if max_scores_list:
                        max_scores = torch.cat(max_scores_list, dim=0)
                        obj_mean = max_scores.mean()
                        obj_frac = (max_scores > args.obj_conf_thresh).float().mean()

                    # [fix6 변경 4/4] matched_score_gap 계산
                    # AP Gap proxy: train 중 matched score의 성별 차이를 직접 추적
                    # score_gap > 0 이면 Male 우위 (실제 AP와 같은 방향)
                    if female_scores.numel() > 0:
                        matched_score_f = female_scores.float().mean()
                    if male_scores.numel() > 0:
                        matched_score_m = male_scores.float().mean()

                total_g.backward()
                if args.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.max_norm)
                opt_g.step()

            else:
                d_loss = torch.tensor(0.0, device=device)
                fairness_loss = torch.tensor(0.0, device=device)
                det_loss = torch.tensor(0.0, device=device)
                total_g = torch.tensor(0.0, device=device)

            metrics_logger.update(
                d_loss=d_loss.item(),
                g_fair=fairness_loss.item(),
                g_det=det_loss.item(),
                g_total=total_g.item(),
                g_w=wasserstein_loss.item(),
                eps=current_eps,
                beta=current_beta,
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                obj_score=obj_mean.item(),
                obj_frac=obj_frac.item(),
                obj_score_f=obj_mean_f.item(),
                obj_frac_f=obj_frac_f.item(),
                obj_score_m=obj_mean_m.item(),
                obj_frac_m=obj_frac_m.item(),
                # [fix6 추가] matched score 모니터링
                matched_score_f=matched_score_f.item(),
                matched_score_m=matched_score_m.item(),
            )

        metrics_logger.synchronize_between_processes()

        if utils.is_main_process():
            mf = metrics_logger.meters["matched_score_f"].global_avg
            mm = metrics_logger.meters["matched_score_m"].global_avg
            log_entry = {
                "epoch": epoch,
                "d_loss": metrics_logger.meters["d_loss"].global_avg,
                "g_fair": metrics_logger.meters["g_fair"].global_avg,
                "g_det": metrics_logger.meters["g_det"].global_avg,
                "g_total": metrics_logger.meters["g_total"].global_avg,
                "g_w": metrics_logger.meters["g_w"].global_avg,
                "epsilon": current_eps,
                "beta": current_beta,
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "obj_score": metrics_logger.meters["obj_score"].global_avg,
                "obj_frac": metrics_logger.meters["obj_frac"].global_avg,
                "obj_score_f": metrics_logger.meters["obj_score_f"].global_avg,
                "obj_frac_f": metrics_logger.meters["obj_frac_f"].global_avg,
                "obj_score_m": metrics_logger.meters["obj_score_m"].global_avg,
                "obj_frac_m": metrics_logger.meters["obj_frac_m"].global_avg,
                # [fix6 추가] matched score gap 로그
                "matched_score_f": mf,
                "matched_score_m": mm,
                "matched_score_gap": mm - mf,  # 양수면 Male 우위 (AP Gap과 같은 방향)
            }
            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(f"\n[Epoch {epoch}] Summary:")
            print(f"  D Loss: {log_entry['d_loss']:.4f}")
            print(f"  G Fair: {log_entry['g_fair']:.4f}  |  G Det: {log_entry['g_det']:.4f}  |  G Wass: {log_entry['g_w']:.4f}")
            print(f"  G Total: {log_entry['g_total']:.4f}")
            print(f"  Matched Score (F/M): {mf:.4f} / {mm:.4f}  |  Gap(M-F): {log_entry['matched_score_gap']:.4f}")
            print(f"  Epsilon: {current_eps:.4f}  |  Beta: {current_beta:.4f}")
            print(f"  Delta L∞: {log_entry['delta_linf']:.4f}  |  Delta L2: {log_entry['delta_l2']:.2f}")

            if (epoch + 1) % args.save_every == 0:
                ckpt_save_path = output_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "generator": _unwrap_ddp(generator).state_dict(),
                        "discriminator": _unwrap_ddp(discriminator).state_dict(),
                        "opt_g": opt_g.state_dict(),
                        "opt_d": opt_d.state_dict(),
                        "args": vars(args),
                    },
                    ckpt_save_path,
                )
                print(f"  Saved: {ckpt_save_path}")

        if args.distributed:
            dist.barrier()

    if utils.is_main_process():
        print("\n" + "=" * 70)
        print("fix6 학습 완료")
        print(f"Output: {output_dir}")
        print("\n[7th 대비 변경 요약]")
        print(f"  lambda_w: 0.2 → {args.lambda_w}")
        print("  BF16 AMP, TF32 활성화")
        print("  matched_score_gap 로그 추가")
        print("\n성공 기준:")
        print("  AP Gap < 0.103 (7th의 0.106 대비 개선)")
        print("  AR Gap < 0.004 (7th의 0.003 유지 또는 개선)")


if __name__ == "__main__":
    main()
