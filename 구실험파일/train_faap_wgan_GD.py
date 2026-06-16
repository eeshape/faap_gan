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
    parser.add_argument("--beta", type=float, default=0.7, help="detection-preserving loss weight")  # 검출 손실 가중치
    parser.add_argument(
        "--lambda_w", type=float, default=0.05, help="weight for Wasserstein alignment (female→male scores)"  # 워서슈타인 손실 가중치
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


def _split_nested(samples: NestedTensor, targets: Sequence[dict], keep: List[int]):  # 배치에서 지정 인덱스만 추출
    if len(keep) == 0:  # 유지할 인덱스가 없으면
        return None, []  # 빈 결과 반환
    tensor = samples.tensors[keep]  # 이미지 텐서에서 선택 인덱스만 슬라이스
    mask = samples.mask[keep] if samples.mask is not None else None  # 마스크가 있으면 동일 인덱스로 슬라이스
    return NestedTensor(tensor, mask), [targets[i] for i in keep]  # 새 NestedTensor와 선택된 타깃 리스트 반환


def _apply_generator(generator: nn.Module, samples: NestedTensor) -> NestedTensor:  # 생성기로 교란을 적용하는 헬퍼
    tensors = samples.tensors  # 원본 이미지 텐서
    delta = generator(tensors)  # 생성기가 생성한 교란(동일 크기)
    perturbed = clamp_normalized(tensors + delta)  # 교란을 더하고 정규화 범위로 클램프
    return NestedTensor(perturbed, samples.mask)  # 마스크는 그대로 두고 새 NestedTensor 생성


def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:  # 예측 분포의 엔트로피 계산
    probs = torch.softmax(logits, dim=-1)  # 로짓을 확률로 변환
    log_probs = torch.log(probs + 1e-8)  # 로그 확률 (언더플로 방지 상수 추가)
    return -(probs * log_probs).sum(dim=-1).mean()  # 샘플별 엔트로피 평균


def _scheduled_epsilon(epoch: int, warmup_epochs: int, eps_start: float, eps_final: float) -> float:  # 에폭에 따른 epsilon 선형 스케줄
    if warmup_epochs <= 1:  # 워밍업 에폭이 1 이하이면
        return eps_final  # 바로 최종 epsilon 사용
    progress = min(epoch / max(1, warmup_epochs - 1), 1.0)  # 0~1 사이 진행도 계산
    return eps_start + (eps_final - eps_start) * progress  # 선형 보간된 epsilon 반환


def _unwrap_ddp(module: nn.Module) -> nn.Module:  # DDP 래퍼가 씌워져 있으면 원본 모듈 추출
    return module.module if isinstance(module, DDP) else module  # 조건부 반환


def _set_generator_epsilon(generator: nn.Module, epsilon: float) -> None:  # 생성기 내부 epsilon 속성 설정
    _unwrap_ddp(generator).epsilon = epsilon  # DDP 여부에 상관없이 속성 적용


def _resize_sorted(scores: torch.Tensor, target_len: int) -> torch.Tensor:  # 정렬된 스코어를 선형보간으로 target 길이에 맞춤
    if target_len <= 0:  # 목표 길이가 0 이하인 경우
        return scores.new_zeros(0, device=scores.device)  # 빈 텐서 반환
    if scores.numel() == 0:  # 입력이 비었으면
        return scores.new_zeros(target_len, device=scores.device)  # 0으로 채운 텐서 반환
    if scores.numel() == target_len:  # 길이가 이미 맞으면
        return scores  # 그대로 반환
    idx = torch.linspace(0, scores.numel() - 1, target_len, device=scores.device)  # 원본 인덱스 범위 선형 공간
    idx_low = idx.floor().long()  # 아래쪽 정수 인덱스
    idx_high = idx.ceil().long()  # 위쪽 정수 인덱스
    weight = idx - idx_low  # 위쪽 가중치
    return scores[idx_low] * (1 - weight) + scores[idx_high] * weight  # 선형 보간 결과


def _matched_detection_scores(detr: FrozenDETR, outputs: dict, targets: Sequence[dict]) -> torch.Tensor:  # DETR 매칭된 예측 점수 추출
    if len(targets) == 0:  # 타깃이 없으면
        return outputs["pred_logits"].new_zeros(0, device=outputs["pred_logits"].device)  # 빈 텐서
    indices = detr.criterion.matcher(outputs, targets)  # DETR Hungarian matcher로 예측-타깃 매칭
    probs = outputs["pred_logits"].softmax(dim=-1)  # 로짓을 확률로 변환
    matched_scores = []  # 매칭된 점수 저장 리스트
    for b, (src_idx, tgt_idx) in enumerate(indices):  # 배치별 매칭 인덱스 순회
        if len(src_idx) == 0:  # 매칭이 없으면
            continue  # 건너뛰기
        tgt_labels = targets[b]["labels"][tgt_idx]  # 대응되는 타깃 클래스 라벨
        matched_scores.append(probs[b, src_idx, tgt_labels])  # 해당 확률 추출해 저장
    if matched_scores:  # 하나라도 있으면
        return torch.cat(matched_scores, dim=0)  # 이어붙여 반환
    return outputs["pred_logits"].new_zeros(0, device=outputs["pred_logits"].device)  # 매칭 없으면 빈 텐서


def _wasserstein_1d(female_scores: torch.Tensor, male_scores: torch.Tensor) -> torch.Tensor:  # 1D 워서슈타인 거리 근사
    if female_scores.numel() == 0 or male_scores.numel() == 0:  # 둘 중 하나라도 비면
        return female_scores.new_tensor(0.0, device=female_scores.device)  # 0 반환
    sorted_f = female_scores.sort().values  # 여성 점수 오름차순 정렬
    sorted_m = male_scores.detach().sort().values  # 남성 점수 정렬 (그래디언트 차단)
    k = max(sorted_f.numel(), sorted_m.numel())  # 더 긴 길이 선택
    sorted_f = _resize_sorted(sorted_f, k)  # 길이 맞추기
    sorted_m = _resize_sorted(sorted_m, k)  # 길이 맞추기
    return (sorted_f - sorted_m).abs().mean()  # 평균 절대 차이로 거리 계산


def main():  # 학습 파이프라인의 엔트리 포인트
    args = parse_args()  # 커맨드라인 인자 파싱
    utils.init_distributed_mode(args)  # 분산 학습 환경 초기화 (utils에서 env 읽음)
    if not hasattr(args, "gpu"):  # 런처가 gpu 속성을 안 넣은 경우 대비
        args.gpu = None  # 기본 None 설정
    detr_repo = ensure_detr_repo_on_path(Path(args.detr_repo))  # DETR 경로를 sys.path에 추가하고 실제 경로 반환
    ckpt_path = Path(args.detr_checkpoint)  # 체크포인트 경로 객체화
    if not ckpt_path.is_absolute():  # 절대경로가 아니면
        ckpt_path = detr_repo / ckpt_path  # DETR 레포 내 상대경로로 해석
    if args.distributed:  # 분산 모드일 때
        if not torch.cuda.is_available():  # CUDA 필수
            raise RuntimeError("Distributed training requires CUDA to be available.")  # 오류 발생
        device = torch.device(f"cuda:{args.gpu}")  # 현재 프로세스용 CUDA 디바이스
        torch.cuda.set_device(args.gpu)  # 현재 프로세스 디바이스 설정
    else:  # 단일 프로세스
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")  # 요청 디바이스 또는 CPU
    seed = args.seed + utils.get_rank()  # 랭크마다 다른 시드 확보
    torch.manual_seed(seed)  # CPU 시드 설정
    if torch.cuda.is_available():  # CUDA 가능 시
        torch.cuda.manual_seed(seed)  # CUDA 시드 설정
    args.world_size = utils.get_world_size()  # 분산 세계 크기 저장
    args.rank = utils.get_rank()  # 현재 랭크 저장

    output_dir = Path(args.output_dir)  # 출력 디렉터리 경로 객체
    if utils.is_main_process():  # 마스터 프로세스만
        output_dir.mkdir(parents=True, exist_ok=True)  # 출력 디렉터리 생성
        (output_dir / "checkpoints").mkdir(exist_ok=True)  # 체크포인트 디렉터리 생성
    if args.distributed:  # 분산 모드이면
        dist.barrier()  # 모든 프로세스 동기화

    # keep a lightweight snapshot of the dataset layout for reproducibility
    if utils.is_main_process():  # 데이터셋 구조를 기록 (마스터만)
        dataset_info = inspect_faap_dataset(Path(args.dataset_root))  # 데이터셋 파일 구조 요약 수집
        with (output_dir / "dataset_layout.json").open("w") as f:  # JSON 파일로 저장
            json.dump(dataset_info, f, indent=2)  # 들여쓰기 2로 기록

    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)  # 동결된 DETR 로드
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)  # 입력 교란 생성기 초기화 후 디바이스 이동
    discriminator = GenderDiscriminator(feature_dim=detr.hidden_dim).to(device)  # 성별 판별기 초기화 후 디바이스 이동
    if args.distributed:  # 분산 모드 시
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)  # 생성기를 DDP로 래핑
        discriminator = DDP(discriminator, device_ids=[args.gpu] if args.gpu is not None else None)  # 판별기 DDP 래핑

    opt_g = torch.optim.Adam(_unwrap_ddp(generator).parameters(), lr=args.lr_g)  # 생성기 최적화기 Adam
    opt_d = torch.optim.Adam(_unwrap_ddp(discriminator).parameters(), lr=args.lr_d)  # 판별기 최적화기 Adam

    start_epoch = 0  # 기본 시작 에폭
    if args.resume:  # 체크포인트에서 재개 요청 시
        ckpt = torch.load(args.resume, map_location=device)  # 체크포인트 로드 (디바이스 매핑)
        if "generator" in ckpt:  # 생성기 가중치가 있으면
            _unwrap_ddp(generator).load_state_dict(ckpt["generator"])  # 로드
        if "discriminator" in ckpt:  # 판별기 가중치가 있으면
            _unwrap_ddp(discriminator).load_state_dict(ckpt["discriminator"])  # 로드
        if "opt_g" in ckpt:  # 생성기 옵티마이저 상태
            opt_g.load_state_dict(ckpt["opt_g"])  # 로드
        if "opt_d" in ckpt:  # 판별기 옵티마이저 상태
            opt_d.load_state_dict(ckpt["opt_d"])  # 로드
        if "epoch" in ckpt:  # 에폭 정보가 있으면
            start_epoch = ckpt["epoch"] + 1  # 다음 에폭부터 시작
        if utils.is_main_process():  # 마스터에서
            print(f"Resumed from {args.resume} at epoch {start_epoch}")  # 재개 정보 출력

    train_loader, _ = build_faap_dataloader(  # 학습용 데이터로더 생성
        Path(args.dataset_root),  # 데이터셋 루트 경로
        "train",  # 분할: train
        args.batch_size,  # 배치 크기
        include_gender=True,  # 성별 정보 포함
        balance_genders=True,  # 성별 균형 샘플링
        num_workers=args.num_workers,  # 워커 수
        distributed=args.distributed,  # 분산 모드 여부
        rank=args.rank,  # 현재 랭크
        world_size=args.world_size,  # 전체 프로세스 수
    )

    log_path = output_dir / "train_log.jsonl"  # 로그 저장 경로
    metrics_logger = utils.MetricLogger(delimiter="  ")  # 메트릭 로거 초기화
    for epoch in range(start_epoch, args.epochs):  # 에폭 루프
        metrics_logger = utils.MetricLogger(delimiter="  ")  # 에폭마다 로거 새로 생성
        generator.train()  # 생성기 학습 모드
        discriminator.train()  # 판별기 학습 모드
        if args.distributed and hasattr(train_loader.sampler, "set_epoch"):  # 분산일 때 샘플러 에폭 설정 지원 시
            train_loader.sampler.set_epoch(epoch)  # 시드 변화로 셔플 동기화
        current_eps = _scheduled_epsilon(epoch, args.epsilon_warmup_epochs, args.epsilon, args.epsilon_final)  # 현재 epsilon 계산
        _set_generator_epsilon(generator, current_eps)  # 생성기에 epsilon 적용
        for samples, targets, genders in metrics_logger.log_every(train_loader, args.log_every, f"Epoch {epoch}"):  # 배치 루프 + 로깅
            samples = samples.to(device)  # 이미지/마스크 디바이스 이동
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # 타깃 텐서 디바이스 이동
            genders = [g.lower() for g in genders]  # 성별 문자열 소문자화

            female_idx = [i for i, g in enumerate(genders) if g == "female"]  # 여성 인덱스 목록
            male_idx = [i for i, g in enumerate(genders) if g == "male"]  # 남성 인덱스 목록
            female_batch, female_targets = _split_nested(samples, targets, female_idx)  # 여성 샘플 분리
            male_batch, male_targets = _split_nested(samples, targets, male_idx)  # 남성 샘플 분리

            delta_linf = torch.tensor(0.0, device=device)  # 교란 L-∞ 초기값
            delta_l2 = torch.tensor(0.0, device=device)  # 교란 L2 초기값
            obj_mean = torch.tensor(0.0, device=device)  # 객체성 평균 초기값
            obj_frac = torch.tensor(0.0, device=device)  # 객체성 임계 초과 비율 초기값
            wasserstein_loss = torch.tensor(0.0, device=device)  # 워서슈타인 손실 초기값

            # -- update discriminator --
            # 변경: 여성/남성 모두 생성기 교란본을 사용해 D가 여=1, 남=0 분류를 학습
            for _ in range(args.k_d):  # 판별기 업데이트 횟수 반복
                d_losses = []  # 배치별 손실 모음
                opt_d.zero_grad()  # 판별기 그래디언트 초기화
                if female_batch is not None:  # 여성 샘플이 있을 때
                    with torch.no_grad():  # 생성기/DETR 그래디언트 계산 방지
                        female_perturbed_d = _apply_generator(generator, female_batch)  # 여성 교란 샘플 생성
                        _, feat_f = detr.forward_with_features(female_perturbed_d)  # DETR 통해 특징 추출
                    logits_f = discriminator(feat_f.detach())  # 특징을 판별기에 넣어 로짓 계산
                    labels_f = torch.ones(logits_f.size(0), device=device, dtype=torch.long)  # 여성 레이블(1) 생성
                    d_losses.append(F.cross_entropy(logits_f, labels_f))  # 크로스 엔트로피 손실 추가
                if male_batch is not None:  # 남성 샘플이 있을 때
                    with torch.no_grad():  # 생성기/DETR 고정
                        male_perturbed_d = _apply_generator(generator, male_batch)  # 남성 교란 샘플 생성
                        _, feat_m = detr.forward_with_features(male_perturbed_d)  # 특징 추출
                    logits_m = discriminator(feat_m.detach())  # 로짓 계산
                    labels_m = torch.zeros(logits_m.size(0), device=device, dtype=torch.long)  # 남성 레이블(0)
                    d_losses.append(F.cross_entropy(logits_m, labels_m))  # 손실 추가

                if d_losses:  # 손실이 모였으면
                    d_loss = torch.stack(d_losses).mean()  # 평균 손실 계산
                    d_loss.backward()  # 판별기 그래디언트 역전파
                    opt_d.step()  # 파라미터 업데이트
                else:  # 배치가 없을 때
                    d_loss = torch.tensor(0.0, device=device)  # 0 손실 유지

            # -- update generator (female + male) --
            # 변경: 여성/남성 모두 G로 교란해 공정성+검출+Wasserstein 손실을 합산
            if female_batch is not None or male_batch is not None:  # 둘 중 하나라도 있을 때만 G 업데이트
                opt_g.zero_grad()  # 생성기 그래디언트 초기화
                fairness_loss = torch.tensor(0.0, device=device)  # 공정성 손실 초기값
                det_loss = torch.tensor(0.0, device=device)  # 검출 손실 초기값
                total_g = torch.tensor(0.0, device=device)  # 총 손실 초기값

                female_scores = torch.tensor([], device=device)  # 여성 매칭 스코어 초기 텐서
                male_scores = torch.tensor([], device=device)  # 남성 매칭 스코어 초기 텐서

                if female_batch is not None:  # 여성 배치 있을 때
                    female_perturbed = _apply_generator(generator, female_batch)  # 교란 적용
                    outputs_f, feat_f = detr.forward_with_features(female_perturbed)  # DETR 결과 및 특징
                    logits_f = discriminator(feat_f)  # 판별 로짓
                    ce_f = F.cross_entropy(logits_f, torch.ones(logits_f.size(0), device=device, dtype=torch.long))  # 여성으로 예측하도록 손실
                    ent_f = _entropy_loss(logits_f)  # 엔트로피 계산
                    fairness_f = -(ce_f + args.alpha * ent_f)  # 공정성 손실(부호 반전)
                    det_f, _ = detr.detection_loss(outputs_f, female_targets)  # DETR 검출 손실
                    female_scores = _matched_detection_scores(detr, outputs_f, female_targets)  # 매칭된 객체 점수
                    fairness_loss = fairness_loss + fairness_f  # 누적
                    det_loss = det_loss + det_f  # 누적
                if male_batch is not None:  # 남성 배치 있을 때
                    male_perturbed = _apply_generator(generator, male_batch)  # 교란 적용
                    outputs_m, feat_m = detr.forward_with_features(male_perturbed)  # DETR 결과 및 특징
                    logits_m = discriminator(feat_m)  # 판별 로짓
                    ce_m = F.cross_entropy(logits_m, torch.zeros(logits_m.size(0), device=device, dtype=torch.long))  # 남성으로 예측하도록 손실
                    ent_m = _entropy_loss(logits_m)  # 엔트로피 계산
                    fairness_m = -(ce_m + args.alpha * ent_m)  # 공정성 손실(부호 반전)
                    det_m, _ = detr.detection_loss(outputs_m, male_targets)  # DETR 검출 손실
                    male_scores = _matched_detection_scores(detr, outputs_m, male_targets)  # 매칭된 객체 점수
                    fairness_loss = fairness_loss + fairness_m  # 누적
                    det_loss = det_loss + det_m  # 누적

                wasserstein_loss = _wasserstein_1d(female_scores, male_scores)  # 성별 간 점수 분포 차이
                total_g = fairness_loss + args.beta * det_loss + args.lambda_w * wasserstein_loss  # 총 생성기 손실

                with torch.no_grad():  # 로깅용 계산에서 그래디언트 제외
                    deltas = []  # 교란 모음
                    if female_batch is not None:  # 여성 교란 차이 저장
                        delta_f = female_perturbed.tensors - female_batch.tensors  # 교란량
                        deltas.append(delta_f)  # 리스트 추가
                    if male_batch is not None:  # 남성 교란 차이 저장
                        delta_m = male_perturbed.tensors - male_batch.tensors  # 교란량
                        deltas.append(delta_m)  # 리스트 추가
                    if deltas:  # 교란이 있으면
                        delta_cat = torch.cat(deltas, dim=0)  # 배치 방향 결합
                        delta_linf = delta_cat.abs().amax(dim=(1, 2, 3)).mean()  # 샘플별 L-∞ 후 평균
                        delta_l2 = delta_cat.flatten(1).norm(p=2, dim=1).mean()  # 샘플별 L2 후 평균
                    probs_list = []  # 객체성 확률 모음
                    if female_batch is not None:  # 여성 예측 확률 저장 (배경 제외)
                        probs_list.append(outputs_f["pred_logits"].softmax(dim=-1)[..., :-1])  # 소프트맥스 후 마지막 클래스(배경) 제외
                    if male_batch is not None:  # 남성 예측 확률 저장
                        probs_list.append(outputs_m["pred_logits"].softmax(dim=-1)[..., :-1])  # 동일 처리
                    if probs_list:  # 예측이 있으면
                        probs_cat = torch.cat(probs_list, dim=0)  # 결합
                        max_scores = probs_cat.max(dim=-1).values  # 각 쿼리의 최고 확률
                        obj_mean = max_scores.mean()  # 평균 객체성
                        obj_frac = (max_scores > args.obj_conf_thresh).float().mean()  # 임계 이상 비율
                    else:  # 예측 없을 때
                        obj_mean = torch.tensor(0.0, device=device)  # 0 유지
                        obj_frac = torch.tensor(0.0, device=device)  # 0 유지

                total_g.backward()  # 생성기 손실 역전파
                if args.max_norm > 0:  # 그래디언트 클리핑 사용 시
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.max_norm)  # 클리핑 적용
                opt_g.step()  # 생성기 파라미터 업데이트
            else:  # 배치 없을 때
                fairness_loss = torch.tensor(0.0, device=device)  # 손실 0 유지
                det_loss = torch.tensor(0.0, device=device)  # 손실 0 유지
                total_g = torch.tensor(0.0, device=device)  # 손실 0 유지

            metrics_logger.update(  # 로거에 메트릭 기록
                d_loss=d_loss.item(),  # 판별기 손실
                g_fair=fairness_loss.item(),  # 공정성 손실
                g_det=det_loss.item(),  # 검출 손실
                g_total=total_g.item(),  # 생성기 총 손실
                g_w=wasserstein_loss.item(),  # 워서슈타인 손실
                eps=current_eps,  # 현재 epsilon
                delta_linf=delta_linf.item(),  # 교란 L-∞
                delta_l2=delta_l2.item(),  # 교란 L2
                obj_score=obj_mean.item(),  # 객체성 평균
                obj_frac=obj_frac.item(),  # 객체성 임계 초과 비율
            )

        # end of epoch bookkeeping
        metrics_logger.synchronize_between_processes()  # 모든 프로세스 메트릭 동기화

        if utils.is_main_process():  # 마스터만 파일 기록/저장
            log_entry = {  # 에폭 로그 딕셔너리
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
            with log_path.open("a") as f:  # 로그 파일 append 모드
                f.write(json.dumps(log_entry) + "\n")  # JSON 라인 기록

            if (epoch + 1) % args.save_every == 0:  # 저장 주기에 해당하면
                ckpt_path = output_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"  # 체크포인트 경로
                torch.save(  # 체크포인트 저장
                    {
                        "epoch": epoch,  # 현재 에폭
                        "generator": _unwrap_ddp(generator).state_dict(),  # 생성기 가중치 상태
                        "discriminator": _unwrap_ddp(discriminator).state_dict(),  # 판별기 가중치 상태
                        "opt_g": opt_g.state_dict(),  # 생성기 옵티마이저 상태
                        "opt_d": opt_d.state_dict(),  # 판별기 옵티마이저 상태
                        "args": vars(args),  # 사용한 인자 기록
                    },
                    ckpt_path,
                )

        if args.distributed:  # 에폭 끝 동기화
            dist.barrier()  # 모든 프로세스가 다음 에폭 전에 정렬


if __name__ == "__main__":  # 스크립트 직접 실행 시
    main()  # main 함수 호출
