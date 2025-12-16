import argparse
import json
from pathlib import Path
from typing import List, Sequence

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
    stem = script_path.stem
    stem_lower = stem.lower()
    suffix = stem
    for prefix in ("train_faap_wgan_", "train_"):
        if stem_lower.startswith(prefix):
            suffix = stem[len(prefix) :]
            break
    suffix = suffix.lower()
    return str(Path("faap_outputs") / f"faap_outputs_{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FAAP-style training for DETR (MMD version)", add_help=True)
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO), help="path to DETR repository")
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()), help="path to DETR pretrained checkpoint")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=_default_output_dir(Path(__file__)),
        help="output directory",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    
    # Removed discriminator args (lr_d, k_d)
    
    parser.add_argument("--epsilon", type=float, default=0.05, help="starting epsilon for warmup")
    parser.add_argument("--epsilon_final", type=float, default=0.10, help="target epsilon after warmup")
    parser.add_argument("--epsilon_warmup_epochs", type=int, default=10, help="epochs to linearly warm epsilon")

    # Removed alpha (entropy weight) as it's specific to adversarial training

    parser.add_argument("--beta", type=float, default=0.5, help="detection-preserving loss weight")

    parser.add_argument(
        "--lambda_fair",
        type=float,
        default=2.0,
        help="weight for fairness loss (MMD)",
    )

    parser.add_argument(
        "--lambda_w",
        type=float,
        default=0.2,
        help="weight for Wasserstein alignment (female->male scores)",
    )

    parser.add_argument("--obj_conf_thresh", type=float, default=0.5, help="objectness threshold for logging recall proxy")
    parser.add_argument("--max_norm", type=float, default=0.1, help="gradient clipping for G")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint to resume G/optim state")
    parser.add_argument("--distributed", action="store_true", help="force distributed mode")
    parser.add_argument("--world_size", default=1, type=int, help="number of processes for distributed training")
    parser.add_argument("--rank", default=0, type=int, help="rank of the process")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for distributed launchers")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
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
    return F.relu(sorted_m - sorted_f).mean()


def _mmd_rbf(X: torch.Tensor, Y: torch.Tensor, sigma_list: List[float] = [1.0, 2.0, 4.0, 8.0, 16.0]) -> torch.Tensor:
    """Gaussian Kernel MMD between two sets of features X and Y.
    X, Y: (Batch, Num_Queries, Hidden_Dim) -> will be pooled to (Batch, Hidden_Dim)
    """
    if X.size(0) == 0 or Y.size(0) == 0:
        return torch.tensor(0.0, device=X.device)
    
    # Global Average Pooling over queries: (B, N, D) -> (B, D)
    if X.dim() == 3:
        X_flat = X.mean(dim=1)
        Y_flat = Y.mean(dim=1)
    elif X.dim() == 4:
        X_flat = X.mean(dim=(2, 3))
        Y_flat = Y.mean(dim=(2, 3))
    else:
        X_flat = X
        Y_flat = Y
    
    # Compute pairwise squared Euclidean distances
    # cdist computes p-norm distance, so we square it for squared euclidean
    XX = torch.cdist(X_flat, X_flat, p=2).pow(2)
    YY = torch.cdist(Y_flat, Y_flat, p=2).pow(2)
    XY = torch.cdist(X_flat, Y_flat, p=2).pow(2)
    
    mmd = torch.tensor(0.0, device=X.device)
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K_XX = torch.exp(-gamma * XX)
        K_YY = torch.exp(-gamma * YY)
        K_XY = torch.exp(-gamma * XY)
        
        mmd = mmd + K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        
    return mmd


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
    
    # No Discriminator in MMD version
    
    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)

    opt_g = torch.optim.Adam(_unwrap_ddp(generator).parameters(), lr=args.lr_g)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if "generator" in ckpt:
            _unwrap_ddp(generator).load_state_dict(ckpt["generator"])
        if "opt_g" in ckpt:
            opt_g.load_state_dict(ckpt["opt_g"])
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

            # -- update generator (female + male) --
            if female_batch is not None or male_batch is not None:
                opt_g.zero_grad()
                fairness_loss = torch.tensor(0.0, device=device)
                det_loss = torch.tensor(0.0, device=device)
                total_g = torch.tensor(0.0, device=device)

                female_scores = torch.tensor([], device=device)
                male_scores = torch.tensor([], device=device)
                
                feat_f = None
                feat_m = None

                if female_batch is not None:
                    female_perturbed = _apply_generator(generator, female_batch)
                    outputs_f, feat_f = detr.forward_with_features(female_perturbed)
                    
                    det_f, _ = detr.detection_loss(outputs_f, female_targets)
                    female_scores = _matched_detection_scores(detr, outputs_f, female_targets)
                    det_loss = det_loss + det_f

                if male_batch is not None:
                    male_perturbed = _apply_generator(generator, male_batch)
                    outputs_m, feat_m = detr.forward_with_features(male_perturbed)
                    
                    det_m, _ = detr.detection_loss(outputs_m, male_targets)
                    male_scores = _matched_detection_scores(detr, outputs_m, male_targets)
                    det_loss = det_loss + det_m

                # MMD Loss for Feature Alignment
                if feat_f is not None and feat_m is not None:
                    fairness_loss = _mmd_rbf(feat_f, feat_m)
                else:
                    fairness_loss = torch.tensor(0.0, device=device)

                wasserstein_loss = _wasserstein_1d(female_scores, male_scores)

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
                        "opt_g": opt_g.state_dict(),
                        "args": vars(args),
                    },
                    ckpt_path,
                )

        if args.distributed:
            dist.barrier()


if __name__ == "__main__":
    main()
