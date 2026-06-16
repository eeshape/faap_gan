import argparse
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

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

from .path_utils import DETR_REPO, ensure_detr_repo_on_path
ensure_detr_repo_on_path(DETR_REPO)

import util.misc as utils

from .datasets import build_gender_datasets
from .models import PerturbationGenerator, clamp_normalized, _IMAGENET_MEAN, _IMAGENET_STD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate perturbed FAAP images (original / perturbed / combined)", add_help=True)
    parser.add_argument("--dataset_root", type=str, default="/home/dohyeong/Desktop/faap_dataset")
    parser.add_argument("--generator_checkpoint", type=str, required=True, help="trained generator checkpoint")
    parser.add_argument("--epsilon", type=float, default=0.05, help="perturbation bound (should match training)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_root", type=str, default="faap_outputs/generated_images")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _denormalize(img: torch.Tensor) -> torch.Tensor:
    """
    Inverse of ImageNet normalization. Expects (3, H, W) normalized tensor.
    """
    mean = _IMAGENET_MEAN.to(img.device)
    std = _IMAGENET_STD.to(img.device)
    return img * std + mean


@torch.no_grad()
def _run_split(
    *,
    dataset_root: Path,
    split: str,
    gender: str,
    generator: PerturbationGenerator,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    output_root: Path,
) -> None:
    ds = build_gender_datasets(dataset_root, split, include_gender=False)[gender]

    # Copy COCO annotation JSON next to generated images (per variant) for eval convenience.
    ann_src = Path(ds.ann_file)
    for variant in ("original", "perturbed", "combined"):
        try:
            rel_ann = ann_src.relative_to(dataset_root)
        except ValueError:
            rel_ann = ann_src.name
        ann_dst = output_root / variant / rel_ann
        ann_dst.parent.mkdir(parents=True, exist_ok=True)
        if not ann_dst.exists():
            shutil.copy2(ann_src, ann_dst)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=utils.collate_fn,
    )

    for samples, targets in tqdm(loader, desc=f"{gender}-{split}"):
        batch = samples.tensors.to(device)
        delta = generator(batch)
        perturbed = clamp_normalized(batch + delta)

        for i, target in enumerate(targets):
            image_id = int(target["image_id"])
            info = ds.coco.loadImgs(image_id)[0]
            file_name = Path(info["file_name"]).name
            orig_file = ds.img_folder / file_name
            rel_path = orig_file.relative_to(dataset_root)

            # Save original by copying to keep pixels identical to annotation
            for variant in ("original", "perturbed", "combined"):
                out_path = output_root / variant / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)

            original_out = output_root / "original" / rel_path
            if not original_out.exists():
                shutil.copy2(orig_file, original_out)

            # Current (transformed) size and original size
            h_cur, w_cur = [int(x) for x in target["size"]]
            h_orig, w_orig = [int(x) for x in target["orig_size"]]

            # Remove padding added by collate_fn
            delta_crop = delta[i][:, :h_cur, :w_cur]
            combined_crop = perturbed[i][:, :h_cur, :w_cur]

            # Save noise only (visualized)
            # Collapse RGB delta into a single channel so the saved perturbation is grayscale
            noise_scalar = delta_crop.cpu().mean(dim=0, keepdim=True)
            noise_vis = noise_scalar / (2 * generator.epsilon) + 0.5
            noise_vis = noise_vis.clamp(0.0, 1.0)
            noise_pil = to_pil_image(noise_vis)
            if (noise_pil.width, noise_pil.height) != (w_orig, h_orig):
                noise_pil = noise_pil.resize((w_orig, h_orig), Image.BILINEAR)

            # Save combined (orig + delta)
            combined = _denormalize(combined_crop.cpu()).clamp(0.0, 1.0)
            if combined.ndim == 4:
                combined = combined.squeeze(0)
            combined_pil = to_pil_image(combined)
            if (combined_pil.width, combined_pil.height) != (w_orig, h_orig):
                combined_pil = combined_pil.resize((w_orig, h_orig), Image.BILINEAR)

            # Paths
            pert_out = output_root / "perturbed" / rel_path  # noise-only
            combined_out = output_root / "combined" / rel_path  # original + noise

            noise_pil.save(pert_out)
            combined_pil.save(combined_out)

            # (original already copied above)


def main() -> None:
    args = parse_args()
    ensure_detr_repo_on_path(DETR_REPO)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.split == "train":
        print(
            "[warn] train split uses random augmentation; saved images may not align 1:1 with original annotations. "
            "Prefer val/test when creating eval-ready perturbations.",
        )

    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)
    state = torch.load(args.generator_checkpoint, map_location=device)
    if "generator" in state:
        generator.load_state_dict(state["generator"])
    else:
        generator.load_state_dict(state)
    generator.eval()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    for gender in ("female", "male"):
        _run_split(
            dataset_root=dataset_root,
            split=args.split,
            gender=gender,
            generator=generator,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            output_root=output_root,
        )

    print(f"Done. Originals / perturbed / combined saved under {output_root}")


if __name__ == "__main__":
    main()
