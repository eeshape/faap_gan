"""
Build /workspace/faap_dataset layout expected by faap_gan/datasets.py.

Source data (FACET, lives under /workspace):
    annotations/annotations.csv     person-level demographic attributes
    annotations/coco_boxes.json     COCO-format boxes (image_id == sa_<id>.jpg)
    imgs_1, imgs_2, imgs_3          raw images (sa_<id>.jpg)

Output layout (consumed by GenderCocoDataset):
    {root}/women_split/{train,val,test}/sa_<id>.jpg   (symlinks)
    {root}/women_split/gender_women_{train,val,test}.json
    {root}/men_split/{train,val,test}/sa_<id>.jpg
    {root}/men_split/gender_men_{train,val,test}.json

A person becomes "women" if gender_presentation_fem == 1 and masc == 0,
"men" if masc == 1 and fem == 0. Non-binary/ambiguous rows are dropped.
Split is per image (so all people on one image stay together).
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def _build_image_index(img_dirs):
    """Map sa_<id>.jpg -> absolute path on disk."""
    index = {}
    for d in img_dirs:
        for p in Path(d).iterdir():
            if p.is_file():
                index[p.name] = p.resolve()
    return index


def _load_gender_by_person(csv_path):
    """Return {person_id: 'women'|'men'} for unambiguous rows."""
    gender_by_pid = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pid = int(row["person_id"])
                masc = int(row["gender_presentation_masc"])
                fem = int(row["gender_presentation_fem"])
            except (KeyError, ValueError):
                continue
            if fem == 1 and masc == 0:
                gender_by_pid[pid] = "women"
            elif masc == 1 and fem == 0:
                gender_by_pid[pid] = "men"
    return gender_by_pid


def _split_image_ids(image_ids, ratios, seed):
    rng = random.Random(seed)
    ids = sorted(image_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def _symlink_images(target_dir, image_records, file_index):
    target_dir.mkdir(parents=True, exist_ok=True)
    missing = 0
    for img in image_records:
        src = file_index.get(img["file_name"])
        if src is None:
            missing += 1
            continue
        dst = target_dir / img["file_name"]
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    return missing


def _dump_coco(out_path, info, categories, images, annotations):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "info": info,
                "categories": categories,
                "images": images,
                "annotations": annotations,
            },
            f,
        )


def main():
    parser = argparse.ArgumentParser(description="Build FAAP dataset layout from FACET source.")
    parser.add_argument("--source_root", default="/workspace",
                        help="Folder containing annotations/ and imgs_1/imgs_2/imgs_3/.")
    parser.add_argument("--dataset_root", default="/workspace/faap_dataset",
                        help="Destination root that matches faap_gan/datasets.py expectations.")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collapse_to_coco_person", action="store_true", default=True,
                        help="Collapse all FACET person sub-categories to COCO person (id=1) "
                             "so a COCO-pretrained DETR can be evaluated meaningfully.")
    parser.add_argument("--no_collapse_to_coco_person", dest="collapse_to_coco_person",
                        action="store_false")
    args = parser.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1.0, got {args.ratios}")

    source = Path(args.source_root)
    ann_dir = source / "annotations"
    img_dirs = [source / "imgs_1", source / "imgs_2", source / "imgs_3"]

    print(f"[1/5] Indexing image files under {[str(d) for d in img_dirs]} ...")
    file_index = _build_image_index(img_dirs)
    print(f"      indexed {len(file_index)} files")

    print(f"[2/5] Reading person->gender map from {ann_dir/'annotations.csv'} ...")
    gender_by_pid = _load_gender_by_person(ann_dir / "annotations.csv")
    print(f"      kept {len(gender_by_pid)} unambiguous person rows")

    print(f"[3/5] Loading {ann_dir/'coco_boxes.json'} ...")
    with open(ann_dir / "coco_boxes.json") as f:
        coco = json.load(f)
    info = coco.get("info", {})
    categories = coco.get("categories", [])
    images = coco["images"]
    annotations = coco["annotations"]
    images_by_id = {img["id"]: img for img in images}

    if args.collapse_to_coco_person:
        # COCO pretrained DETR expects category_id=1 == "person".
        categories = [{"id": 1, "name": "person", "supercategory": "person"}]
        for ann in annotations:
            ann["category_id"] = 1

    anns_per_gender = {"women": [], "men": []}
    image_ids_per_gender = {"women": set(), "men": set()}
    for ann in annotations:
        pid = ann.get("person_id", ann.get("id"))
        g = gender_by_pid.get(pid)
        if g is None:
            continue
        if ann["image_id"] not in images_by_id:
            continue
        anns_per_gender[g].append(ann)
        image_ids_per_gender[g].add(ann["image_id"])

    print(f"      women: {len(anns_per_gender['women'])} anns on "
          f"{len(image_ids_per_gender['women'])} images")
    print(f"      men:   {len(anns_per_gender['men'])} anns on "
          f"{len(image_ids_per_gender['men'])} images")

    print(f"[4/5] Splitting per-image with ratios={args.ratios} seed={args.seed} ...")
    splits = {g: _split_image_ids(ids, args.ratios, args.seed)
              for g, ids in image_ids_per_gender.items()}
    for g in ("women", "men"):
        print(f"      {g}: " + ", ".join(f"{k}={len(v)}" for k, v in splits[g].items()))

    print(f"[5/5] Writing layout under {args.dataset_root} ...")
    dataset_root = Path(args.dataset_root)
    anns_by_image = defaultdict(list)
    for g in ("women", "men"):
        for ann in anns_per_gender[g]:
            anns_by_image[(g, ann["image_id"])].append(ann)

    for g in ("women", "men"):
        for split, img_ids in splits[g].items():
            split_dir = dataset_root / f"{g}_split" / split
            split_images = [images_by_id[i] for i in img_ids]
            missing = _symlink_images(split_dir, split_images, file_index)
            split_anns = [a for i in img_ids for a in anns_by_image[(g, i)]]
            json_path = dataset_root / f"{g}_split" / f"gender_{g}_{split}.json"
            _dump_coco(json_path, info, categories, split_images, split_anns)
            print(f"      {g}/{split}: {len(split_images)} imgs, "
                  f"{len(split_anns)} anns, missing files: {missing} -> {json_path}")

    print("Done.")


if __name__ == "__main__":
    main()
