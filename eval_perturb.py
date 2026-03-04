import argparse
import json
import re
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
    parser = argparse.ArgumentParser("Evaluate FAAP perturbation only (skip baseline)", add_help=True)
    parser.add_argument("--dataset_root", type=str, default="/workspace/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO), help="path to DETR repository (for imports/checkpoint)")
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()), help="path to DETR pretrained checkpoint")
    parser.add_argument("--generator_checkpoint", type=str, required=True, help="trained generator checkpoint")
    parser.add_argument("--epsilon", type=float, default=0.10, help="perturbation bound (should match training)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results_path", type=str, default="", help="output path for metrics (auto-generated if empty)")
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


def _infer_output_path(generator_checkpoint: Optional[str], split: str, default_dir: str = "faap_outputs") -> Path:
    """generator_checkpoint 경로에서 output_dir, 방법명, epoch 번호를 추출하여 결과 파일 경로 생성."""
    if not generator_checkpoint:
        return Path(default_dir) / "faap_metrics_perturb.json"

    ckpt_path = Path(generator_checkpoint)

    epoch_match = re.search(r'epoch_(\d{4})', ckpt_path.name)
    epoch_str = epoch_match.group(1) if epoch_match else "0000"

    if "checkpoints" in ckpt_path.parts:
        idx = ckpt_path.parts.index("checkpoints")
        output_dir = Path(*ckpt_path.parts[:idx])
    else:
        output_dir = ckpt_path.parent

    method_name = ""
    for part in output_dir.parts:
        if part.startswith("faap_outputs_"):
            method_name = part[13:]
            break

    if method_name:
        filename = f"{split}_metrics_perturb_{method_name}_epoch_{epoch_str}.json"
    else:
        filename = f"{split}_metrics_perturb_epoch_{epoch_str}.json"

    return output_dir / filename


def main():
    args = parse_args()
    detr_repo = ensure_detr_repo_on_path(Path(args.detr_repo))
    ckpt_path = Path(args.detr_checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = detr_repo / ckpt_path
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.results_path:
        output_path = Path(args.results_path)
    else:
        output_path = _infer_output_path(args.generator_checkpoint, args.split)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)

    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)
    state = torch.load(args.generator_checkpoint, map_location=device)
    if "generator" in state:
        generator.load_state_dict(state["generator"])
    else:
        generator.load_state_dict(state)

    gender_ds = build_gender_datasets(Path(args.dataset_root), args.split, include_gender=False)
    male_loader = build_eval_loader(gender_ds["male"], args.batch_size, args.num_workers)
    female_loader = build_eval_loader(gender_ds["female"], args.batch_size, args.num_workers)

    print(f"=== Evaluating perturbed (male) split={args.split} ===", flush=True)
    pert_male_ap, pert_male_ar, pert_male_stats, pert_male_text = evaluate_split(
        detr, male_loader, device=device, generator=generator
    )
    print(f"=== Evaluating perturbed (female) split={args.split} ===", flush=True)
    pert_female_ap, pert_female_ar, pert_female_stats, pert_female_text = evaluate_split(
        detr, female_loader, device=device, generator=generator
    )

    results = {
        "perturbed": {
            "male": {"AP": pert_male_ap, "AR": pert_male_ar},
            "female": {"AP": pert_female_ap, "AR": pert_female_ar},
        },
        "details": {
            "perturbed": {"male": pert_male_stats, "female": pert_female_stats},
        },
        "details_text": {
            "perturbed": {"male": pert_male_text, "female": pert_female_text},
        },
        "gaps": {
            "AP": {"perturbed": pert_male_ap - pert_female_ap},
            "AR": {"perturbed": pert_male_ar - pert_female_ar},
        },
        "hyperparams": {
            "epsilon": args.epsilon,
            "generator_checkpoint": args.generator_checkpoint,
            "detr_checkpoint": args.detr_checkpoint,
            "split": args.split,
            "batch_size": args.batch_size,
        },
        "generated_at": datetime.now().astimezone().isoformat(),
    }

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
