"""
t-SNE Feature Visualization for fix11 Contrastive Experiments

사용법:
  # L2 있는 버전 분석
  python eval_tsne_features.py --checkpoint faap_outputs/.../epoch_0014.pth --title "fix11 w/ L2"

  # L2 없는 ablation 분석
  python eval_tsne_features.py --checkpoint faap_outputs/.../epoch_0014.pth --title "fix11 w/o L2 (ablation)"

  # 두 체크포인트 비교 (한 figure에 나란히)
  python eval_tsne_features.py \
      --checkpoint faap_outputs/.../epoch_0014.pth \
      --checkpoint2 faap_outputs/.../epoch_0014.pth \
      --title "w/ L2" --title2 "w/o L2 (ablation)"

출력:
  - t-SNE scatter plot (clean vs perturbed, gender별 색상)
  - displacement 수치 (여성/남성 이동량, cosine similarity 변화)
"""

import argparse
import json
from pathlib import Path
from typing import Optional

if __package__ is None or __package__ == "":
    import sys
    pkg_dir = Path(__file__).resolve().parent
    parent = pkg_dir.parent
    if str(parent) not in sys.path:
        sys.path.append(str(parent))
    if str(pkg_dir) not in sys.path:
        sys.path.append(str(pkg_dir))
    __package__ = "faap_gan"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

from faap_gan.datasets import build_faap_dataloader
from faap_gan.models import FrozenDETR, PerturbationGenerator, clamp_normalized
from faap_gan.path_utils import DETR_REPO, default_detr_checkpoint, ensure_detr_repo_on_path

ensure_detr_repo_on_path(DETR_REPO)
from util.misc import NestedTensor


# =========================================================================
# ProjectionHead (학습 스크립트와 동일)
# =========================================================================

class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256,
                 output_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        proj = self.net(pooled)
        return F.normalize(proj, dim=-1, p=2)


# =========================================================================
# Feature Extraction
# =========================================================================

@torch.no_grad()
def extract_features(
    detr: FrozenDETR,
    generator: PerturbationGenerator,
    proj_head: ProjectionHead,
    dataloader,
    device: torch.device,
    max_samples: int = 500,
) -> dict:
    """Clean/Perturbed feature를 성별별로 수집"""
    generator.eval()
    proj_head.eval()

    z_clean_f, z_clean_m = [], []
    z_pert_f, z_pert_m = [], []
    count_f, count_m = 0, 0
    per_gender = max_samples // 2  # 성별별 균등 수집

    for samples, targets, genders in dataloader:
        if count_f >= per_gender and count_m >= per_gender:
            break

        samples = samples.to(device)
        genders = [g.lower() for g in genders]

        female_idx = [i for i, g in enumerate(genders) if g == "female"]
        male_idx = [i for i, g in enumerate(genders) if g == "male"]

        # 이미 충분히 수집한 성별은 스킵
        if count_f >= per_gender:
            female_idx = []
        if count_m >= per_gender:
            male_idx = []

        if len(female_idx) == 0 and len(male_idx) == 0:
            continue

        # 실제 사용할 인덱스만 추출
        use_idx = sorted(female_idx + male_idx)
        sub_samples = NestedTensor(samples.tensors[use_idx], samples.mask[use_idx])

        # Clean features
        _, feat_clean = detr.forward_with_features(sub_samples)
        z_clean = proj_head(feat_clean)

        # Perturbed features
        tensors = sub_samples.tensors
        delta = generator(tensors)
        perturbed = NestedTensor(clamp_normalized(tensors + delta), sub_samples.mask)
        _, feat_pert = detr.forward_with_features(perturbed)
        z_pert = proj_head(feat_pert)

        # use_idx 기준으로 재매핑
        local_f = [i for i, orig in enumerate(use_idx) if orig in female_idx]
        local_m = [i for i, orig in enumerate(use_idx) if orig in male_idx]

        if local_f:
            z_clean_f.append(z_clean[local_f].cpu())
            z_pert_f.append(z_pert[local_f].cpu())
            count_f += len(local_f)
        if local_m:
            z_clean_m.append(z_clean[local_m].cpu())
            z_pert_m.append(z_pert[local_m].cpu())
            count_m += len(local_m)

    print(f"  Collected: {count_f} female, {count_m} male samples")

    return {
        "z_clean_f": torch.cat(z_clean_f) if z_clean_f else torch.empty(0),
        "z_clean_m": torch.cat(z_clean_m) if z_clean_m else torch.empty(0),
        "z_pert_f": torch.cat(z_pert_f) if z_pert_f else torch.empty(0),
        "z_pert_m": torch.cat(z_pert_m) if z_pert_m else torch.empty(0),
    }


# =========================================================================
# Displacement Metrics
# =========================================================================

def compute_metrics(feats: dict) -> dict:
    """Feature 이동 방향/크기 정량 분석"""
    z_cf = feats["z_clean_f"]
    z_cm = feats["z_clean_m"]
    z_pf = feats["z_pert_f"]
    z_pm = feats["z_pert_m"]

    # 각 성별 이동량 (L2 norm)
    disp_f = (z_pf - z_cf).norm(dim=-1).mean().item() if len(z_cf) > 0 else 0
    disp_m = (z_pm - z_cm).norm(dim=-1).mean().item() if len(z_cm) > 0 else 0

    # 성별 중심 간 거리
    center_cf = z_cf.mean(0) if len(z_cf) > 0 else torch.zeros(1)
    center_cm = z_cm.mean(0) if len(z_cm) > 0 else torch.zeros(1)
    center_pf = z_pf.mean(0) if len(z_pf) > 0 else torch.zeros(1)
    center_pm = z_pm.mean(0) if len(z_pm) > 0 else torch.zeros(1)

    dist_before = (center_cf - center_cm).norm().item()
    dist_after = (center_pf - center_pm).norm().item()

    # Cosine similarity (성별 중심 간)
    cos_before = F.cosine_similarity(center_cf.unsqueeze(0), center_cm.unsqueeze(0)).item()
    cos_after = F.cosine_similarity(center_pf.unsqueeze(0), center_pm.unsqueeze(0)).item()

    return {
        "disp_female": disp_f,
        "disp_male": disp_m,
        "disp_ratio": disp_f / max(disp_m, 1e-8),
        "center_dist_before": dist_before,
        "center_dist_after": dist_after,
        "center_dist_delta": dist_after - dist_before,
        "cos_sim_before": cos_before,
        "cos_sim_after": cos_after,
        "cos_sim_delta": cos_after - cos_before,
        "n_female": len(z_cf),
        "n_male": len(z_cm),
    }


# =========================================================================
# t-SNE Visualization
# =========================================================================

def plot_tsne_single(feats: dict, metrics: dict, title: str, ax):
    """하나의 체크포인트에 대한 t-SNE plot"""
    z_cf = feats["z_clean_f"].numpy()
    z_cm = feats["z_clean_m"].numpy()
    z_pf = feats["z_pert_f"].numpy()
    z_pm = feats["z_pert_m"].numpy()

    # Ensure all arrays are 2D (handle empty arrays)
    # First, infer the feature dimension from non-empty arrays
    feat_dim = None
    for arr in [z_cf, z_cm, z_pf, z_pm]:
        if arr.size > 0 and arr.ndim >= 2:
            feat_dim = arr.shape[-1]
            break
    if feat_dim is None:
        feat_dim = 128  # default projection head output dim
    
    # Reshape all arrays to be (N, feat_dim)
    z_cf = z_cf.reshape(-1, feat_dim) if z_cf.size > 0 else np.empty((0, feat_dim))
    z_cm = z_cm.reshape(-1, feat_dim) if z_cm.size > 0 else np.empty((0, feat_dim))
    z_pf = z_pf.reshape(-1, feat_dim) if z_pf.size > 0 else np.empty((0, feat_dim))
    z_pm = z_pm.reshape(-1, feat_dim) if z_pm.size > 0 else np.empty((0, feat_dim))

    # 전체 합쳐서 t-SNE
    all_feats = np.concatenate([z_cf, z_cm, z_pf, z_pm], axis=0)
    n_cf, n_cm, n_pf, n_pm = len(z_cf), len(z_cm), len(z_pf), len(z_pm)

    perplexity = min(30, len(all_feats) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedded = tsne.fit_transform(all_feats)

    idx = 0
    e_cf = embedded[idx:idx + n_cf]; idx += n_cf
    e_cm = embedded[idx:idx + n_cm]; idx += n_cm
    e_pf = embedded[idx:idx + n_pf]; idx += n_pf
    e_pm = embedded[idx:idx + n_pm]; idx += n_pm

    # Plot
    s = 15
    alpha = 0.6
    ax.scatter(e_cf[:, 0], e_cf[:, 1], c='#FF9999', marker='o', s=s, alpha=alpha, label='Clean Female')
    ax.scatter(e_cm[:, 0], e_cm[:, 1], c='#9999FF', marker='o', s=s, alpha=alpha, label='Clean Male')
    ax.scatter(e_pf[:, 0], e_pf[:, 1], c='#FF0000', marker='^', s=s, alpha=alpha, label='Pert Female')
    ax.scatter(e_pm[:, 0], e_pm[:, 1], c='#0000FF', marker='^', s=s, alpha=alpha, label='Pert Male')

    # 중심점 화살표 (clean -> pert 이동 방향)
    for e_clean, e_pert, color in [
        (e_cf, e_pf, '#FF0000'),
        (e_cm, e_pm, '#0000FF'),
    ]:
        if len(e_clean) > 0 and len(e_pert) > 0:
            c0 = e_clean.mean(axis=0)
            c1 = e_pert.mean(axis=0)
            ax.annotate('', xy=c1, xytext=c0,
                        arrowprops=dict(arrowstyle='->', color=color, lw=2.5))

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')

    # 수치 텍스트
    text = (
        f"disp_F={metrics['disp_female']:.4f}  disp_M={metrics['disp_male']:.4f}\n"
        f"ratio(F/M)={metrics['disp_ratio']:.2f}\n"
        f"cos: {metrics['cos_sim_before']:.4f} -> {metrics['cos_sim_after']:.4f} "
        f"({metrics['cos_sim_delta']:+.4f})"
    )
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=7,
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def load_model_from_checkpoint(ckpt_path: str, device: torch.device, detr_repo: Path):
    """체크포인트에서 generator + proj_head 로드"""
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {})

    detr_ckpt = saved_args.get("detr_checkpoint", str(default_detr_checkpoint()))
    detr_ckpt_path = Path(detr_ckpt)
    if not detr_ckpt_path.is_absolute():
        detr_ckpt_path = detr_repo / detr_ckpt_path

    detr = FrozenDETR(checkpoint_path=detr_ckpt_path, device=str(device), detr_repo=detr_repo)

    epsilon = saved_args.get("epsilon_final", saved_args.get("epsilon", 0.1))
    generator = PerturbationGenerator(epsilon=epsilon).to(device)
    generator.load_state_dict(ckpt["generator"])

    proj_dim = saved_args.get("proj_dim", 128)
    proj_dropout = saved_args.get("proj_dropout", 0.1)
    proj_head = ProjectionHead(
        input_dim=detr.hidden_dim,
        hidden_dim=detr.hidden_dim,
        output_dim=proj_dim,
        dropout=proj_dropout,
    ).to(device)
    proj_head.load_state_dict(ckpt["proj_head"])

    epoch = ckpt.get("epoch", "?")
    return detr, generator, proj_head, epoch


# =========================================================================
# Main
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser("t-SNE Feature Visualization")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="첫 번째 체크포인트 경로")
    parser.add_argument("--checkpoint2", type=str, default="",
                        help="두 번째 체크포인트 경로 (비교 모드)")
    parser.add_argument("--title", type=str, default="Experiment 1")
    parser.add_argument("--title2", type=str, default="Experiment 2")
    parser.add_argument("--dataset_root", type=str, default="/workspace/faap_dataset")
    parser.add_argument("--split", type=str, default="val",
                        help="val 또는 test")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=500,
                        help="t-SNE에 사용할 최대 샘플 수")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="",
                        help="저장할 figure 경로 (비우면 자동 생성)")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    detr_repo = ensure_detr_repo_on_path(Path(args.detr_repo))

    compare_mode = bool(args.checkpoint2)

    # DataLoader (한 번만 생성)
    dataloader, _ = build_faap_dataloader(
        Path(args.dataset_root),
        args.split,
        args.batch_size,
        include_gender=True,
        balance_genders=False,
        num_workers=args.num_workers,
    )

    # ---- Checkpoint 1 ----
    print(f"Loading checkpoint 1: {args.checkpoint}")
    detr1, gen1, proj1, epoch1 = load_model_from_checkpoint(
        args.checkpoint, device, detr_repo
    )
    print(f"  Epoch: {epoch1}")
    print("Extracting features...")
    feats1 = extract_features(detr1, gen1, proj1, dataloader, device, args.max_samples)
    metrics1 = compute_metrics(feats1)

    print(f"\n[{args.title}] Displacement Metrics:")
    for k, v in metrics1.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ---- Checkpoint 2 (optional) ----
    if compare_mode:
        print(f"\nLoading checkpoint 2: {args.checkpoint2}")
        detr2, gen2, proj2, epoch2 = load_model_from_checkpoint(
            args.checkpoint2, device, detr_repo
        )
        print(f"  Epoch: {epoch2}")
        print("Extracting features...")
        feats2 = extract_features(detr2, gen2, proj2, dataloader, device, args.max_samples)
        metrics2 = compute_metrics(feats2)

        print(f"\n[{args.title2}] Displacement Metrics:")
        for k, v in metrics2.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ---- Plot ----
    if compare_mode:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        plot_tsne_single(feats1, metrics1, f"{args.title} (epoch {epoch1})", ax1)
        plot_tsne_single(feats2, metrics2, f"{args.title2} (epoch {epoch2})", ax2)
        fig.suptitle("t-SNE Feature Comparison: Clean vs Perturbed", fontsize=14, fontweight='bold')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        plot_tsne_single(feats1, metrics1, f"{args.title} (epoch {epoch1})", ax)

    plt.tight_layout()

    # Save
    if args.output:
        save_path = Path(args.output)
    else:
        ckpt_dir = Path(args.checkpoint).parent.parent
        save_path = ckpt_dir / f"tsne_{args.split}_epoch{epoch1}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {save_path}")

    # Save metrics
    metrics_path = save_path.with_suffix('.json')
    metrics_out = {"checkpoint1": {"path": args.checkpoint, "epoch": epoch1, **metrics1}}
    if compare_mode:
        metrics_out["checkpoint2"] = {"path": args.checkpoint2, "epoch": epoch2, **metrics2}
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Metrics: {metrics_path}")

    plt.show()


if __name__ == "__main__":
    main()
