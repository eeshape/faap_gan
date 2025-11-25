from pathlib import Path
from typing import Dict, List, Tuple

from .path_utils import DETR_REPO, ensure_detr_repo_on_path

# ensure DETR repo is importable before pulling datasets/util
ensure_detr_repo_on_path(DETR_REPO)

from torch.utils.data import (
    BatchSampler,
    ConcatDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)
from torch.utils.data.distributed import DistributedSampler

from datasets.coco import CocoDetection, make_coco_transforms
import util.misc as utils
# from util.misc import NestedTensor
# import torch

# Temporarily disabling gender-aware collate until needed.
# def collate_fn_with_gender(batch):
#     """Custom collate function that handles (image, target, gender) tuples."""
#     if len(batch[0]) == 3:
#         # (image, target, gender) format
#         images = [item[0] for item in batch]
#         targets = [item[1] for item in batch]
#         genders = [item[2] for item in batch]
#         # Use DETR's nested tensor for images
#         batch_images = utils.nested_tensor_from_tensor_list(images)
#         return batch_images, targets, genders
#     else:
#         # (image, target) format - fallback to standard collate
#         return utils.collate_fn(batch)


_GENDER_ALIASES = {
    "men": "male",
    "man": "male",
    "male": "male",
    "women": "female",
    "woman": "female",
    "female": "female",
}


def _canonical_gender(name: str) -> str:
    key = name.lower()
    if key not in _GENDER_ALIASES:
        raise ValueError(f"Unknown gender key: {name}")
    return _GENDER_ALIASES[key]


class GenderCocoDataset(CocoDetection):
    """
    COCO-style dataset that keeps track of the protected attribute label.

    The underlying annotations live under:
    {root}/{gender}_split/{split}/images...
    {root}/{gender}_split/gender_{gender}_{split}.json
    """

    def __init__(
        self,
        root: Path,
        split: str,
        gender_key: str,
        *,
        include_gender: bool = True,
        return_masks: bool = False,
    ) -> None:
        gender = _canonical_gender(gender_key)
        base_dir = Path(root) / f"{'women' if gender == 'female' else 'men'}_split"
        img_folder = base_dir / split
        ann_file = base_dir / f"gender_{'women' if gender == 'female' else 'men'}_{split}.json"
        if not img_folder.exists():
            raise FileNotFoundError(f"Image folder not found: {img_folder}")
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        transforms = make_coco_transforms("train" if split == "train" else "val")
        super().__init__(img_folder, ann_file, transforms=transforms, return_masks=return_masks, light_filter=None)
        self.gender = gender
        self.split = split
        self.include_gender = include_gender
        self.ann_file = ann_file
        self.img_folder = img_folder

    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        if self.include_gender:
            return image, target, self.gender
        return image, target


def inspect_faap_dataset(root: Path) -> Dict[Tuple[str, str], Dict[str, object]]:
    """
    Lightweight scan of the FAAP dataset layout to simplify debugging from the agent.
    Returns a dict keyed by (gender, split) with counts and file types.
    """
    info: Dict[str, Dict[str, object]] = {}
    for gender_key in ("women", "men"):
        for split in ("train", "val", "test"):
            base_dir = Path(root) / f"{gender_key}_split"
            img_dir = base_dir / split
            ann_path = base_dir / f"gender_{gender_key}_{split}.json"
            image_exts: Dict[str, int] = {}
            if img_dir.exists():
                for path in img_dir.iterdir():
                    if path.is_file():
                        image_exts[path.suffix] = image_exts.get(path.suffix, 0) + 1
            key = f"{gender_key}_{split}"
            info[key] = {
                "images_dir": str(img_dir),
                "annotation": str(ann_path),
                "annotation_exists": ann_path.exists(),
                "num_images": sum(image_exts.values()),
                "extensions": image_exts,
            }
    return info


def build_gender_datasets(root: Path, split: str, *, include_gender: bool = True, return_masks: bool = False):
    female_ds = GenderCocoDataset(root, split, "women", include_gender=include_gender, return_masks=return_masks)
    male_ds = GenderCocoDataset(root, split, "men", include_gender=include_gender, return_masks=return_masks)
    return {"female": female_ds, "male": male_ds}


def _balanced_sampler(female_len: int, male_len: int) -> WeightedRandomSampler:
    # keep genders roughly balanced by sampling with replacement
    weights: List[float] = [0.5 / female_len] * female_len + [0.5 / male_len] * male_len
    num_samples = 2 * max(male_len, female_len)
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


def build_faap_dataloader(
    root: Path,
    split: str,
    batch_size: int,
    *,
    include_gender: bool = True,
    balance_genders: bool = True,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, Dict[str, GenderCocoDataset]]:
    datasets = build_gender_datasets(root, split, include_gender=include_gender)
    combo = ConcatDataset([datasets["female"], datasets["male"]])

    # Choose collate function; gender-aware collate currently disabled
    # collate = collate_fn_with_gender if include_gender else utils.collate_fn
    collate = utils.collate_fn

    if split == "train":
        if distributed:
            sampler = DistributedSampler(combo, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
            loader = DataLoader(
                combo,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=True,
                collate_fn=collate,
                num_workers=num_workers,
                pin_memory=True,
            )
        elif balance_genders:
            sampler = _balanced_sampler(len(datasets["female"]), len(datasets["male"]))
            batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
            loader = DataLoader(
                combo,
                batch_sampler=batch_sampler,
                collate_fn=collate,
                num_workers=num_workers,
                pin_memory=True,
            )
        else:
            sampler = RandomSampler(combo)
            loader = DataLoader(
                combo,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=True,
                collate_fn=collate,
                num_workers=num_workers,
                pin_memory=True,
            )
    else:
        sampler = SequentialSampler(combo)
        loader = DataLoader(
            combo,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=False,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loader, datasets


def build_eval_loader(
    dataset: GenderCocoDataset, batch_size: int, num_workers: int = 4
) -> DataLoader:
    sampler = SequentialSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader
