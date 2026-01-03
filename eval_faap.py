import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

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

from .datasets import build_eval_loader, build_gender_datasets
from .models import FrozenDETR, PerturbationGenerator, clamp_normalized
from .path_utils import DETR_REPO, default_detr_checkpoint, ensure_detr_repo_on_path
import util.misc as utils
from util.misc import NestedTensor
from datasets import get_coco_api_from_dataset
from datasets.coco_eval import CocoEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate baseline vs. FAAP perturbation on DETR", add_help=True)
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO), help="path to DETR repository (for imports/checkpoint)")
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()), help="path to DETR pretrained checkpoint")
    parser.add_argument("--generator_checkpoint", type=str, default=None, help="trained generator checkpoint")
    parser.add_argument("--epsilon", type=float, default=0.05, help="perturbation bound (should match training)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results_path", type=str, default="faap_outputs/faap_metrics.json")
    return parser.parse_args()


def _apply_generator_eval(generator: PerturbationGenerator, samples: NestedTensor) -> NestedTensor:
    tensors = samples.tensors
    delta = generator(tensors)
    perturbed = clamp_normalized(tensors + delta)
    return NestedTensor(perturbed, samples.mask)


def _coco_stats_dict(stats: Sequence[float]) -> Dict[str, float]:
    keys = [
        "AP@[0.50:0.95]_all_100",
        "AP@[0.50]_all_100",
        "AP@[0.75]_all_100",
        "AP@[0.50:0.95]_small_100",
        "AP@[0.50:0.95]_medium_100",
        "AP@[0.50:0.95]_large_100",
        "AR@[0.50:0.95]_all_1",
        "AR@[0.50:0.95]_all_10",
        "AR@[0.50:0.95]_all_100",
        "AR@[0.50:0.95]_small_100",
        "AR@[0.50:0.95]_medium_100",
        "AR@[0.50:0.95]_large_100",
    ]
    return {k: float(v) for k, v in zip(keys, stats)}


def _coco_stats_text(stats: Sequence[float]) -> str:
    lines = [
        f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[0]:.3f}",
        f" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {stats[1]:.3f}",
        f" Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {stats[2]:.3f}",
        f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[3]:.3f}",
        f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats[4]:.3f}",
        f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats[5]:.3f}",
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {stats[6]:.3f}",
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {stats[7]:.3f}",
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[8]:.3f}",
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[9]:.3f}",
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats[10]:.3f}",
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats[11]:.3f}",
    ]
    return "\n".join(lines)


@torch.no_grad()
def evaluate_split(
    detr: FrozenDETR,
    data_loader,
    *,
    device: torch.device,
    generator: Optional[PerturbationGenerator] = None,
) -> Tuple[float, float, Dict[str, float], str]:
    if generator is not None:
        generator.eval()

    coco = get_coco_api_from_dataset(data_loader.dataset)
    # Some custom annotations may miss the optional "info" field; COCOeval expects it.
    coco.dataset.setdefault("info", {})
    coco_evaluator = CocoEvaluator(coco, ("bbox",))
    metric_logger = utils.MetricLogger(delimiter="  ")

    for samples, targets in metric_logger.log_every(data_loader, 20, "eval"):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if generator is not None:
            samples = _apply_generator_eval(generator, samples)

        outputs = detr.forward(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = detr.postprocessors["bbox"](outputs, orig_target_sizes)
        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    stats = coco_evaluator.coco_eval["bbox"].stats
    ap = float(stats[0])
    ar = float(stats[8])  # AR@100
    return ap, ar, _coco_stats_dict(stats), _coco_stats_text(stats)


def main_legacy():
    """Legacy main function - evaluates baseline first, then perturbed."""
    args = parse_args()
    detr_repo = ensure_detr_repo_on_path(Path(args.detr_repo))
    ckpt_path = Path(args.detr_checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = detr_repo / ckpt_path
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_path = Path(args.results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)

    generator = None
    if args.generator_checkpoint:
        generator = PerturbationGenerator(epsilon=args.epsilon).to(device)
        state = torch.load(args.generator_checkpoint, map_location=device)
        if "generator" in state:
            generator.load_state_dict(state["generator"])
        else:
            generator.load_state_dict(state)

    gender_ds = build_gender_datasets(Path(args.dataset_root), args.split, include_gender=False)
    male_loader = build_eval_loader(gender_ds["male"], args.batch_size, args.num_workers)
    female_loader = build_eval_loader(gender_ds["female"], args.batch_size, args.num_workers)

    print(f"=== Evaluating baseline (male) split={args.split} ===", flush=True)
    baseline_male_ap, baseline_male_ar, baseline_male_stats, baseline_male_text = evaluate_split(
        detr, male_loader, device=device, generator=None
    )
    print(f"=== Evaluating baseline (female) split={args.split} ===", flush=True)
    baseline_female_ap, baseline_female_ar, baseline_female_stats, baseline_female_text = evaluate_split(
        detr, female_loader, device=device, generator=None
    )

    pert_male_ap, pert_male_ar = baseline_male_ap, baseline_male_ar
    pert_female_ap, pert_female_ar = baseline_female_ap, baseline_female_ar
    pert_male_stats = baseline_male_stats
    pert_female_stats = baseline_female_stats
    pert_male_text = baseline_male_text
    pert_female_text = baseline_female_text
    if generator is not None:
        print(f"=== Evaluating perturbed (male) split={args.split} ===", flush=True)
        pert_male_ap, pert_male_ar, pert_male_stats, pert_male_text = evaluate_split(
            detr, male_loader, device=device, generator=generator
        )
        print(f"=== Evaluating perturbed (female) split={args.split} ===", flush=True)
        pert_female_ap, pert_female_ar, pert_female_stats, pert_female_text = evaluate_split(
            detr, female_loader, device=device, generator=generator
        )

    deltas = {
        "male": {"AP": pert_male_ap - baseline_male_ap, "AR": pert_male_ar - baseline_male_ar},
        "female": {"AP": pert_female_ap - baseline_female_ap, "AR": pert_female_ar - baseline_female_ar},
    }

    results = {
        "baseline": {
            "male": {"AP": baseline_male_ap, "AR": baseline_male_ar},
            "female": {"AP": baseline_female_ap, "AR": baseline_female_ar},
        },
        "perturbed": {
            "male": {"AP": pert_male_ap, "AR": pert_male_ar},
            "female": {"AP": pert_female_ap, "AR": pert_female_ar},
        },
        "deltas": deltas,
        "details": {
            "baseline": {"male": baseline_male_stats, "female": baseline_female_stats},
            "perturbed": {"male": pert_male_stats, "female": pert_female_stats},
        },
        "details_text": {
            "baseline": {"male": baseline_male_text, "female": baseline_female_text},
            "perturbed": {"male": pert_male_text, "female": pert_female_text},
        },
        "gaps": {
            "AP": {"baseline": baseline_male_ap - baseline_female_ap, "perturbed": pert_male_ap - pert_female_ap},
            "AR": {"baseline": baseline_male_ar - baseline_female_ar, "perturbed": pert_male_ar - pert_female_ar},
        },
        "hyperparams": {
            "epsilon": args.epsilon,
            "generator_checkpoint": args.generator_checkpoint,
            "detr_checkpoint": args.detr_checkpoint,
            "split": args.split,
            "batch_size": args.batch_size,
        },
        "notes": {
            "perturbation_applied_to_male": generator is not None,
            "perturbation_applied_to_female": generator is not None,
        },
        "generated_at": datetime.now().astimezone().isoformat(),
    }

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {output_path}")


def main():
    """Main function - evaluates perturbed first, then baseline."""
    args = parse_args()
    detr_repo = ensure_detr_repo_on_path(Path(args.detr_repo))
    ckpt_path = Path(args.detr_checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = detr_repo / ckpt_path
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_path = Path(args.results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)

    generator = None
    if args.generator_checkpoint:
        generator = PerturbationGenerator(epsilon=args.epsilon).to(device)
        state = torch.load(args.generator_checkpoint, map_location=device)
        if "generator" in state:
            generator.load_state_dict(state["generator"])
        else:
            generator.load_state_dict(state)

    gender_ds = build_gender_datasets(Path(args.dataset_root), args.split, include_gender=False)
    male_loader = build_eval_loader(gender_ds["male"], args.batch_size, args.num_workers)
    female_loader = build_eval_loader(gender_ds["female"], args.batch_size, args.num_workers)

    # Initialize perturbed results with baseline defaults
    pert_male_ap, pert_male_ar = 0.0, 0.0
    pert_female_ap, pert_female_ar = 0.0, 0.0
    pert_male_stats, pert_female_stats = {}, {}
    pert_male_text, pert_female_text = "", ""

    # Evaluate perturbed FIRST
    if generator is not None:
        print(f"=== Evaluating perturbed (male) split={args.split} ===", flush=True)
        pert_male_ap, pert_male_ar, pert_male_stats, pert_male_text = evaluate_split(
            detr, male_loader, device=device, generator=generator
        )
        print(f"=== Evaluating perturbed (female) split={args.split} ===", flush=True)
        pert_female_ap, pert_female_ar, pert_female_stats, pert_female_text = evaluate_split(
            detr, female_loader, device=device, generator=generator
        )

    # Evaluate baseline SECOND
    print(f"=== Evaluating baseline (male) split={args.split} ===", flush=True)
    baseline_male_ap, baseline_male_ar, baseline_male_stats, baseline_male_text = evaluate_split(
        detr, male_loader, device=device, generator=None
    )
    print(f"=== Evaluating baseline (female) split={args.split} ===", flush=True)
    baseline_female_ap, baseline_female_ar, baseline_female_stats, baseline_female_text = evaluate_split(
        detr, female_loader, device=device, generator=None
    )

    # If no generator, copy baseline to perturbed
    if generator is None:
        pert_male_ap, pert_male_ar = baseline_male_ap, baseline_male_ar
        pert_female_ap, pert_female_ar = baseline_female_ap, baseline_female_ar
        pert_male_stats = baseline_male_stats
        pert_female_stats = baseline_female_stats
        pert_male_text = baseline_male_text
        pert_female_text = baseline_female_text

    deltas = {
        "male": {"AP": pert_male_ap - baseline_male_ap, "AR": pert_male_ar - baseline_male_ar},
        "female": {"AP": pert_female_ap - baseline_female_ap, "AR": pert_female_ar - baseline_female_ar},
    }

    results = {
        "baseline": {
            "male": {"AP": baseline_male_ap, "AR": baseline_male_ar},
            "female": {"AP": baseline_female_ap, "AR": baseline_female_ar},
        },
        "perturbed": {
            "male": {"AP": pert_male_ap, "AR": pert_male_ar},
            "female": {"AP": pert_female_ap, "AR": pert_female_ar},
        },
        "deltas": deltas,
        "details": {
            "baseline": {"male": baseline_male_stats, "female": baseline_female_stats},
            "perturbed": {"male": pert_male_stats, "female": pert_female_stats},
        },
        "details_text": {
            "baseline": {"male": baseline_male_text, "female": baseline_female_text},
            "perturbed": {"male": pert_male_text, "female": pert_female_text},
        },
        "gaps": {
            "AP": {"baseline": baseline_male_ap - baseline_female_ap, "perturbed": pert_male_ap - pert_female_ap},
            "AR": {"baseline": baseline_male_ar - baseline_female_ar, "perturbed": pert_male_ar - pert_female_ar},
        },
        "hyperparams": {
            "epsilon": args.epsilon,
            "generator_checkpoint": args.generator_checkpoint,
            "detr_checkpoint": args.detr_checkpoint,
            "split": args.split,
            "batch_size": args.batch_size,
        },
        "notes": {
            "perturbation_applied_to_male": generator is not None,
            "perturbation_applied_to_female": generator is not None,
        },
        "generated_at": datetime.now().astimezone().isoformat(),
    }

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
