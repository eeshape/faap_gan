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
import torch.nn.functional as F
from torch import nn

from .datasets import build_faap_dataloader, inspect_faap_dataset
from .models import FrozenDETR, GenderDiscriminator, PerturbationGenerator, clamp_normalized
from .path_utils import DETR_REPO, default_detr_checkpoint, ensure_detr_repo_on_path
import util.misc as utils
from util.misc import NestedTensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FAAP-style training for DETR", add_help=True)
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO), help="path to DETR repository (for imports/checkpoint)")
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()), help="path to DETR pretrained checkpoint")
    parser.add_argument("--output_dir", type=str, default="faap_outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--k_d", type=int, default=1, help="discriminator steps per iteration")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.1, help="entropy weight for fairness term")
    parser.add_argument("--beta", type=float, default=1.0, help="detection-preserving loss weight")
    parser.add_argument("--max_norm", type=float, default=0.1, help="gradient clipping for G")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint to resume G/D/optim state")
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


def main():
    args = parse_args()
    detr_repo = ensure_detr_repo_on_path(Path(args.detr_repo))
    ckpt_path = Path(args.detr_checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = detr_repo / ckpt_path
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    # keep a lightweight snapshot of the dataset layout for reproducibility
    dataset_info = inspect_faap_dataset(Path(args.dataset_root))
    with (output_dir / "dataset_layout.json").open("w") as f:
        json.dump(dataset_info, f, indent=2)

    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)
    discriminator = GenderDiscriminator(feature_dim=detr.hidden_dim).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if "generator" in ckpt:
            generator.load_state_dict(ckpt["generator"])
        if "discriminator" in ckpt:
            discriminator.load_state_dict(ckpt["discriminator"])
        if "opt_g" in ckpt:
            opt_g.load_state_dict(ckpt["opt_g"])
        if "opt_d" in ckpt:
            opt_d.load_state_dict(ckpt["opt_d"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    train_loader, _ = build_faap_dataloader(
        Path(args.dataset_root),
        "train",
        args.batch_size,
        include_gender=True,
        balance_genders=True,
        num_workers=args.num_workers,
    )

    log_path = output_dir / "train_log.jsonl"
    metrics_logger = utils.MetricLogger(delimiter="  ")
    for epoch in range(start_epoch, args.epochs):
        metrics_logger = utils.MetricLogger(delimiter="  ")
        generator.train()
        discriminator.train()

        for samples, targets, genders in metrics_logger.log_every(train_loader, args.log_every, f"Epoch {epoch}"):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            genders = [g.lower() for g in genders]

            female_idx = [i for i, g in enumerate(genders) if g == "female"]
            male_idx = [i for i, g in enumerate(genders) if g == "male"]
            female_batch, female_targets = _split_nested(samples, targets, female_idx)
            male_batch, male_targets = _split_nested(samples, targets, male_idx)

            # -- update discriminator --
            for _ in range(args.k_d):
                d_losses = []
                opt_d.zero_grad()
                if female_batch is not None:
                    with torch.no_grad():
                        female_perturbed = _apply_generator(generator, female_batch)
                        _, feat_f = detr.forward_with_features(female_perturbed)
                    logits_f = discriminator(feat_f.detach())
                    labels_f = torch.ones(logits_f.size(0), device=device, dtype=torch.long)
                    d_losses.append(F.cross_entropy(logits_f, labels_f))
                if male_batch is not None:
                    with torch.no_grad():
                        _, feat_m = detr.forward_with_features(male_batch)
                    logits_m = discriminator(feat_m.detach())
                    labels_m = torch.zeros(logits_m.size(0), device=device, dtype=torch.long)
                    d_losses.append(F.cross_entropy(logits_m, labels_m))

                if d_losses:
                    d_loss = torch.stack(d_losses).mean()
                    d_loss.backward()
                    opt_d.step()
                else:
                    d_loss = torch.tensor(0.0, device=device)

            # -- update generator (female only) --
            if female_batch is not None:
                opt_g.zero_grad()
                female_perturbed = _apply_generator(generator, female_batch)
                outputs_f, feat_f = detr.forward_with_features(female_perturbed)
                logits_f = discriminator(feat_f)
                ce_f = F.cross_entropy(logits_f, torch.ones(logits_f.size(0), device=device, dtype=torch.long))
                ent_f = _entropy_loss(logits_f)
                fairness_loss = -(ce_f + args.alpha * ent_f)
                det_loss, _ = detr.detection_loss(outputs_f, female_targets)
                total_g = fairness_loss + args.beta * det_loss
                total_g.backward()
                if args.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.max_norm)
                opt_g.step()
            else:
                fairness_loss = torch.tensor(0.0, device=device)
                det_loss = torch.tensor(0.0, device=device)
                total_g = torch.tensor(0.0, device=device)

            metrics_logger.update(d_loss=d_loss.item(), g_fair=fairness_loss.item(), g_det=det_loss.item(), g_total=total_g.item())

        # end of epoch bookkeeping
        log_entry = {
            "epoch": epoch,
            "d_loss": metrics_logger.meters["d_loss"].global_avg,
            "g_fair": metrics_logger.meters["g_fair"].global_avg,
            "g_det": metrics_logger.meters["g_det"].global_avg,
            "g_total": metrics_logger.meters["g_total"].global_avg,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = output_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )


if __name__ == "__main__":
    main()
