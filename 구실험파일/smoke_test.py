"""Tiny end-to-end smoke test: load dataset + DETR, run one inference batch."""
import sys
from pathlib import Path

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/detr")

import torch
from faap_gan.datasets import build_faap_dataloader
from faap_gan.models import FrozenDETR
from faap_gan.path_utils import DETR_REPO, default_detr_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[smoke] device={device}")

print("[smoke] building val dataloader (batch=2) ...")
loader, datasets = build_faap_dataloader(
    Path("/workspace/faap_dataset"),
    split="val",
    batch_size=2,
    num_workers=0,
    include_gender=True,
    balance_genders=False,
)
print(f"[smoke]   female={len(datasets['female'])} male={len(datasets['male'])}")

print(f"[smoke] loading frozen DETR from {default_detr_checkpoint()} ...")
detr = FrozenDETR(
    checkpoint_path=default_detr_checkpoint(),
    device=device,
    detr_repo=DETR_REPO,
)
print("[smoke] DETR loaded.")

print("[smoke] iterating one batch ...")
batch = next(iter(loader))
images, targets, genders = batch
images = images.to(device)
print(f"[smoke]   image batch tensor shape: {images.tensors.shape}, mask: {images.mask.shape}")
print(f"[smoke]   genders: {genders}")

with torch.no_grad():
    outputs = detr.model(images)
print(f"[smoke] DETR outputs keys: {list(outputs.keys())}")
print(f"[smoke]   pred_logits: {outputs['pred_logits'].shape}")
print(f"[smoke]   pred_boxes : {outputs['pred_boxes'].shape}")
print("[smoke] OK")
