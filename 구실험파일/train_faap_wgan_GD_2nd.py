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


def parse_args() -> argparse.Namespace:  # 커맨드라인 인자를 파싱해 Namespace로 반환
    parser = argparse.ArgumentParser("FAAP-style training for DETR", add_help=True)  # 설명과 함께 파서 생성
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")  # 데이터셋 루트 경로
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO), help="path to DETR repository (for imports/checkpoint)")  # DETR 코드 위치
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()), help="path to DETR pretrained checkpoint")  # 사전학습 체크포인트 경로
    parser.add_argument("--output_dir", type=str, default="faap_outputs")  # 출력 저장 디렉터리
    parser.add_argument("--device", type=str, default="cuda")  # 기본 디바이스 설정
    parser.add_argument("--epochs", type=int, default=12)  # 학습 에폭 수
    parser.add_argument("--batch_size", type=int, default=4)  # 배치 크기
    parser.add_argument("--num_workers", type=int, default=6)  # 데이터로더 워커 수
    parser.add_argument("--lr_g", type=float, default=1e-4)  # 생성기 학습률
    parser.add_argument("--lr_d", type=float, default=1e-4)  # 판별기 학습률
    parser.add_argument("--k_d", type=int, default=2, help="discriminator steps per iteration")  # 판별기 업데이트 반복 횟수
    parser.add_argument("--epsilon", type=float, default=0.05, help="starting epsilon for warmup")  # 교란 크기 초기값
    parser.add_argument("--epsilon_final", type=float, default=0.12, help="target epsilon after warmup")  # 워밍업 후 목표 epsilon
    parser.add_argument("--epsilon_warmup_epochs", type=int, default=5, help="epochs to linearly warm epsilon")  # epsilon 선형 증가 에폭

    parser.add_argument("--alpha", type=float, default=0.2, help="entropy weight for fairness term")  # 엔트로피 가중치

    # 수정: detection-preserving loss weight 기본값을 0.7 -> 0.3으로 낮춤
    parser.add_argument("--beta", type=float, default=0.3, help="detection-preserving loss weight")  # 검출 손실 가중치

    # 수정: 공정성 손실 가중치 lambda_fair 추가
    parser.add_argument(
        "--lambda_fair",
        type=float,
        default=5.0,
        help="weight for fairness loss (adversarial + entropy)",
    )

    # 수정: 워서슈타인 손실 가중치 기본값을 0.05 -> 0.1으로 소폭 상향
    parser.add_argument(
        "--lambda_w",
        type=float,
        default=0.1,
        help="weight for Wasserstein alignment (female→male scores)",
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


def _scheduled_epsilon(epoch: int, warmup_epochs: int, eps_start: float, eps_final: float) -> float:
    if warmup_epochs <= 1:
        return eps_final
    progress = min(epoch / max(1, warmup_epochs - 1), 1.0)
    return eps_start + (eps_final - eps_start) * progress


def _unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def _set_generator_epsilon(generator: nn.Module, epsilon: float) -> None:
    _unwrap_ddp(generator).epsilon = epsilon


def _resize_sorted(scores: torch.Tensor, target_len: int) -> torch.Tensor:
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
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0, device=female_scores.device)
    sorted_f = female_scores.sort().values
    sorted_m = male_scores.detach().sort().values
    k = max(sorted_f.numel(), sorted_m.numel())
    sorted_f = _resize_sorted(sorted_f, k)
    sorted_m = _resize_sorted(sorted_m, k)
    return (sorted_f - sorted_m).abs().mean()


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

    train_loader, _ = build_faap_dataloader(
        Path(args.dataset_root),
        "train",
        args.batch_size,
        include_gender=True,
        balance_genders=True,
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
        current_eps = _scheduled_epsilon(epoch, args.epsilon_warmup_epochs, args.epsilon, args.epsilon_final)
        _set_generator_epsilon(generator, current_eps)

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
            wasserstein_loss = torch.tensor(0.0, device=device)

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
                    fairness_loss = fairness_loss + fairness_f
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
                    fairness_loss = fairness_loss + fairness_m
                    det_loss = det_loss + det_m

                wasserstein_loss = _wasserstein_1d(female_scores, male_scores)

                # 수정: fairness_loss에 lambda_fair 가중치 적용
                total_g = (
                    args.lambda_fair * fairness_loss
                    + args.beta * det_loss
                    + args.lambda_w * wasserstein_loss
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
                    probs_list = []
                    if female_batch is not None:
                        probs_list.append(outputs_f["pred_logits"].softmax(dim=-1)[..., :-1])
                    if male_batch is not None:
                        probs_list.append(outputs_m["pred_logits"].softmax(dim=-1)[..., :-1])
                    if probs_list:
                        probs_cat = torch.cat(probs_list, dim=0)
                        max_scores = probs_cat.max(dim=-1).values
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
                eps=current_eps,
                delta_linf=delta_linf.item(),
                delta_l2=delta_l2.item(),
                obj_score=obj_mean.item(),
                obj_frac=obj_frac.item(),
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
                "epsilon": current_eps,
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "obj_score": metrics_logger.meters["obj_score"].global_avg,
                "obj_frac": metrics_logger.meters["obj_frac"].global_avg,
            }
            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if (epoch + 1) % args.save_every == 0:
                ckpt_path = output_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "generator": _unwrap_ddp(generator).state_dict(),
                        "discriminator": _unwrap_ddp(discriminator).state_dict(),
                        "opt_g": opt_g.state_dict(),
                        "opt_d": opt_d.state_dict(),
                        "args": vars(args),
                    },
                    ckpt_path,
                )

        if args.distributed:
            dist.barrier()


if __name__ == "__main__":
    main()
