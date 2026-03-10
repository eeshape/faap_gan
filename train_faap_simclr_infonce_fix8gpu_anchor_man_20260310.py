"""
FAAP Training - Gender-Centered SupCon, Anchor=Male (fix8gpu_anchor_man)

성별 중심 anchor 설정: Male을 anchor로, 같은 quality bin의 Female을 positive로.
"Male feature가 같은 품질의 Female feature를 닮아라"

=============================================================================
fix7 vs fix8 실험 비교:
=============================================================================
fix7: 모든 샘플이 anchor, 성별 무관 (Quality SupCon)
fix8_women: Female만 anchor, Male이 target
fix8_man:   Male만 anchor, Female이 target (이 파일)

=============================================================================
Anchor 설정:
=============================================================================
Anchor:   Male 샘플 (quality bin Q)
Positive: Female 샘플 (같은 quality bin Q)
Negative: 모든 샘플 (다른 quality bin)
Female은 anchor로 사용하지 않음 → gradient가 Male 쪽에서 흐름

의도: anchor_women과 비교하여 anchor 방향이 결과에 미치는 영향 확인

=============================================================================
GPU 최적화 (A100 / RTX 5090)
=============================================================================
- BF16 AMP, TF32, torch.compile, channels_last, cudnn.benchmark
=============================================================================
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Sequence, Tuple

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


# =============================================================================
# Projection Head
# =============================================================================

class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, output_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        proj = self.net(pooled)
        return F.normalize(proj, dim=-1, p=2)


# =============================================================================
# Gender-Centered SupCon Loss (Anchor = Male)
# =============================================================================

class GenderAnchorSupConLoss(nn.Module):
    """
    Gender-Centered Supervised Contrastive Loss.

    Anchor: Male만 (anchor_gender="male")
    Positive: 같은 quality bin의 Female
    Negative: 다른 quality bin의 모든 샘플

    Male feature → 같은 품질의 Female feature 방향으로 당김.
    Female은 anchor가 아니므로 직접적인 contrastive gradient를 받지 않음.
    """

    ANCHOR_GENDER = "male"
    TARGET_GENDER = "female"

    def __init__(self, temperature: float = 0.1, n_bins: int = 3):
        super().__init__()
        self.temperature = temperature
        self.n_bins = n_bins

    def forward(
        self,
        projections: torch.Tensor,  # (N, D) L2-normalized
        scores: torch.Tensor,       # (N,) detection scores
        genders: list,
    ) -> Tuple[torch.Tensor, dict]:

        n = projections.size(0)
        anchor_idx = [i for i, g in enumerate(genders) if g == self.ANCHOR_GENDER]
        target_idx = [i for i, g in enumerate(genders) if g == self.TARGET_GENDER]

        if len(anchor_idx) < 2 or len(target_idx) < 1:
            return projections.new_tensor(0.0), {
                "n_anchor": len(anchor_idx), "n_target": len(target_idx),
                "score_gap": 0.0, "valid_anchors": 0,
            }

        # Quality bin 할당 (전체 batch 기준)
        n_bins = min(self.n_bins, n // 2)
        boundaries = torch.quantile(
            scores.detach(),
            torch.linspace(0, 1, n_bins + 1, device=scores.device)[1:-1],
        )
        labels = torch.bucketize(scores.detach(), boundaries)  # (N,)

        # Similarity: anchor(Male) vs 전체
        sim_all = torch.mm(projections, projections.t()) / self.temperature  # (N, N)
        mask_self = torch.eye(n, device=projections.device, dtype=torch.bool)

        # Vectorized computation (no Python for-loop over GPU tensors)
        anchor_t = torch.tensor(anchor_idx, device=projections.device, dtype=torch.long)
        target_t = torch.tensor(target_idx, device=projections.device, dtype=torch.long)

        # Positive mask: same quality bin between anchor and target
        pos_mask = labels[anchor_t].unsqueeze(1) == labels[target_t].unsqueeze(0)  # (n_a, n_t)
        has_pos = pos_mask.any(dim=1)  # (n_a,)

        if not has_pos.any():
            return projections.new_tensor(0.0), {
                "n_anchor": len(anchor_idx), "n_target": len(target_idx),
                "score_gap": 0.0, "valid_anchors": 0,
            }

        # Denominator: logsumexp over all non-self for each anchor
        sim_anchors = sim_all[anchor_t].masked_fill(mask_self[anchor_t], float('-inf'))
        log_denom = torch.logsumexp(sim_anchors, dim=1)  # (n_a,)

        # Numerator: mean of positive similarities per anchor
        sim_a2t = sim_all[anchor_t][:, target_t]  # (n_a, n_t)
        n_pos = pos_mask.float().sum(dim=1).clamp(min=1)
        mean_pos_sim = (sim_a2t * pos_mask.float()).sum(dim=1) / n_pos

        # SupCon loss
        per_anchor_loss = -(mean_pos_sim - log_denom)
        loss = per_anchor_loss[has_pos].mean()
        valid_anchors = has_pos.sum().item()

        # 로깅
        score_gap = 0.0
        if anchor_idx and target_idx:
            score_gap = (scores[target_idx].mean() - scores[anchor_idx].mean()).item()

        info = {
            "n_anchor": len(anchor_idx),
            "n_target": len(target_idx),
            "score_gap": score_gap,
            "score_f_mean": scores[anchor_idx].mean().item(),
            "score_m_mean": scores[target_idx].mean().item(),
            "valid_anchors": valid_anchors,
            "n_bins_used": len(labels.unique()),
        }

        return loss, info


# =============================================================================
# Hungarian Matcher Score
# =============================================================================

def _hungarian_matcher_score(
    detr: FrozenDETR, outputs: dict, targets: Sequence[dict],
) -> torch.Tensor:
    if len(targets) == 0:
        return outputs["pred_logits"].new_zeros(0)
    indices = detr.criterion.matcher(outputs, targets)
    probs = outputs["pred_logits"].softmax(dim=-1)
    image_scores = []
    for b, (src_idx, tgt_idx) in enumerate(indices):
        if len(src_idx) == 0:
            max_probs = probs[b, :, :-1].max(dim=-1).values
            topk = min(5, max_probs.size(0))
            image_scores.append(max_probs.topk(topk).values.mean())
        else:
            tgt_labels = targets[b]["labels"][tgt_idx]
            matched_scores = probs[b, src_idx, tgt_labels]
            image_scores.append(matched_scores.mean())
    return torch.stack(image_scores)


def _image_level_detection_score(outputs: dict, top_k: int = 10) -> torch.Tensor:
    probs = outputs["pred_logits"].softmax(dim=-1)[..., :-1]
    max_probs = probs.max(dim=-1).values
    if top_k > 0 and top_k < max_probs.size(1):
        topk_probs = max_probs.topk(top_k, dim=1).values
        return topk_probs.mean(dim=1)
    return max_probs.mean(dim=1)


# =============================================================================
# Wasserstein Loss (단방향)
# =============================================================================

def _resize_sorted(sorted_vals: torch.Tensor, k: int) -> torch.Tensor:
    if sorted_vals.numel() == k:
        return sorted_vals
    idx = torch.linspace(0, sorted_vals.numel() - 1, k, device=sorted_vals.device)
    idx_low, idx_high = idx.floor().long(), idx.ceil().long()
    weight = idx - idx_low.float()
    return sorted_vals[idx_low] * (1 - weight) + sorted_vals[idx_high] * weight


def _wasserstein_1d_unidirectional(female_scores: torch.Tensor, male_scores: torch.Tensor) -> torch.Tensor:
    if female_scores.numel() == 0 or male_scores.numel() == 0:
        return female_scores.new_tensor(0.0)
    sorted_f = female_scores.sort().values
    sorted_m = male_scores.detach().sort().values
    k = max(sorted_f.numel(), sorted_m.numel())
    sorted_f = _resize_sorted(sorted_f, k)
    sorted_m = _resize_sorted(sorted_m, k)
    return F.relu(sorted_m - sorted_f).mean()


# =============================================================================
# Scheduling & Utilities
# =============================================================================

def _scheduled_epsilon(epoch, eps_start, eps_peak, eps_min, warmup_epochs, hold_epochs, cooldown_epochs):
    warmup_end = warmup_epochs
    hold_end = warmup_end + hold_epochs
    if epoch < warmup_end:
        return eps_start + (eps_peak - eps_start) * epoch / max(1, warmup_end)
    elif epoch < hold_end:
        return eps_peak
    else:
        progress = min((epoch - hold_end) / max(1, cooldown_epochs), 1.0)
        return eps_peak + (eps_min - eps_peak) * progress

def _scheduled_beta(epoch, total_epochs, beta_start, beta_final):
    if total_epochs <= 1:
        return beta_start
    return beta_start + (beta_final - beta_start) * min(epoch / max(1, total_epochs - 1), 1.0)

def _contrastive_warmup_weight(epoch, warmup_epochs):
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return 1.0
    return epoch / warmup_epochs

def _default_output_dir(script_path):
    stem = script_path.stem
    for prefix in ("train_faap_simclr_", "train_faap_", "train_"):
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix):]
            break
    return str(Path("faap_outputs") / f"faap_outputs_{stem.lower()}")

def _unwrap_ddp(module):
    return module.module if isinstance(module, DDP) else module


class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)
    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}
    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


def parse_args():
    parser = argparse.ArgumentParser("FAAP fix8 anchor=man")

    parser.add_argument("--dataset_root", type=str, default="/workspace/faap_dataset")
    parser.add_argument("--detr_repo", type=str, default=str(DETR_REPO))
    parser.add_argument("--detr_checkpoint", type=str, default=str(default_detr_checkpoint()))
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(Path(__file__)))

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr_g", type=float, default=5e-5)
    parser.add_argument("--lr_warmup_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--epsilon_final", type=float, default=0.10)
    parser.add_argument("--epsilon_min", type=float, default=0.09)
    parser.add_argument("--epsilon_warmup_epochs", type=int, default=8)
    parser.add_argument("--epsilon_hold_epochs", type=int, default=6)
    parser.add_argument("--epsilon_cooldown_epochs", type=int, default=10)

    parser.add_argument("--lambda_supcon", type=float, default=1.0)
    parser.add_argument("--lambda_wass", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--beta_final", type=float, default=0.6)

    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--n_bins", type=int, default=3)
    parser.add_argument("--contrastive_warmup_epochs", type=int, default=3)

    parser.add_argument("--use_hungarian_score", action="store_true", default=False)
    parser.add_argument("--no_hungarian_score", dest="use_hungarian_score", action="store_false")
    parser.add_argument("--score_top_k", type=int, default=10)

    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--proj_dropout", type=float, default=0.1)

    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--no_ema", action="store_true", default=False)
    parser.add_argument("--no_compile", action="store_true", default=True)

    parser.add_argument("--max_norm", type=float, default=0.1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--dist_url", default="env://")

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    utils.init_distributed_mode(args)
    if not hasattr(args, "gpu"):
        args.gpu = None

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False  # 가변 크기 입력 → benchmark 끄기 (VRAM 안정화)
    torch.set_float32_matmul_precision('high')

    detr_repo = ensure_detr_repo_on_path(Path(args.detr_repo))
    ckpt_path = Path(args.detr_checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = detr_repo / ckpt_path

    if args.distributed:
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
        with (output_dir / "config.json").open("w") as f:
            json.dump(vars(args), f, indent=2)
    if args.distributed:
        dist.barrier()

    if utils.is_main_process():
        dataset_info = inspect_faap_dataset(Path(args.dataset_root))
        with (output_dir / "dataset_layout.json").open("w") as f:
            json.dump(dataset_info, f, indent=2)
        print("=" * 70)
        print("Gender-Centered SupCon — Anchor=MALE (fix8gpu_anchor_man)")
        print("=" * 70)
        print("[Anchor 설정]")
        print("  Anchor:   Male (변해야 하는 쪽)")
        print("  Positive: 같은 quality bin의 Female (목표)")
        print("  Negative: 다른 quality bin의 모든 샘플")
        print("  → Male feature가 같은 품질의 Female feature를 닮아가도록 학습")
        print("-" * 70)
        print(f"Temperature: {args.temperature}, Bins: {args.n_bins}")
        print(f"LR: {args.lr_g} (warmup: {args.lr_warmup_epochs} epochs)")
        print(f"Epsilon: {args.epsilon} → {args.epsilon_final} → {args.epsilon_min}")
        print(f"Batch size: {args.batch_size}")
        print("=" * 70)

    # Models
    detr = FrozenDETR(checkpoint_path=ckpt_path, device=str(device), detr_repo=detr_repo)
    generator = PerturbationGenerator(epsilon=args.epsilon).to(device)
    proj_head = ProjectionHead(detr.hidden_dim, detr.hidden_dim, args.proj_dim, args.proj_dropout).to(device)
    supcon_loss_fn = GenderAnchorSupConLoss(args.temperature, args.n_bins).to(device)

    if not args.no_compile:
        generator = torch.compile(generator, mode="reduce-overhead")
        proj_head = torch.compile(proj_head, mode="reduce-overhead")

    if args.distributed:
        generator = DDP(generator, device_ids=[args.gpu] if args.gpu is not None else None)
        proj_head = DDP(proj_head, device_ids=[args.gpu] if args.gpu is not None else None)

    params = list(_unwrap_ddp(generator).parameters()) + list(_unwrap_ddp(proj_head).parameters())
    opt_g = torch.optim.AdamW(params, lr=args.lr_g, weight_decay=0.01)

    def lr_lambda(epoch):
        if epoch < args.lr_warmup_epochs:
            return (epoch + 1) / args.lr_warmup_epochs
        progress = (epoch - args.lr_warmup_epochs) / max(1, args.epochs - args.lr_warmup_epochs)
        return 0.1 + 0.9 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item()) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)

    ema_gen, ema_proj = None, None
    if not args.no_ema:
        ema_gen = EMAModel(_unwrap_ddp(generator), args.ema_decay)
        ema_proj = EMAModel(_unwrap_ddp(proj_head), args.ema_decay)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if "generator" in ckpt: _unwrap_ddp(generator).load_state_dict(ckpt["generator"])
        if "proj_head" in ckpt: _unwrap_ddp(proj_head).load_state_dict(ckpt["proj_head"])
        if "opt_g" in ckpt: opt_g.load_state_dict(ckpt["opt_g"])
        if "epoch" in ckpt: start_epoch = ckpt["epoch"] + 1
        if "ema_gen" in ckpt and ema_gen: ema_gen.load_state_dict(ckpt["ema_gen"])
        if "ema_proj" in ckpt and ema_proj: ema_proj.load_state_dict(ckpt["ema_proj"])

    train_loader, _ = build_faap_dataloader(
        Path(args.dataset_root), "train", args.batch_size,
        include_gender=True, balance_genders=False,
        num_workers=args.num_workers, distributed=args.distributed,
        rank=args.rank, world_size=args.world_size,
    )

    log_path = output_dir / "train_log.jsonl"
    best_score_gap = float('inf')
    best_epoch = -1

    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        metrics_logger = utils.MetricLogger(delimiter="  ")
        generator.train()
        proj_head.train()
        if args.distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        current_beta = _scheduled_beta(epoch, args.epochs, args.beta, args.beta_final)
        current_epsilon = _scheduled_epsilon(
            epoch, args.epsilon, args.epsilon_final, args.epsilon_min,
            args.epsilon_warmup_epochs, args.epsilon_hold_epochs, args.epsilon_cooldown_epochs)
        contrastive_weight = _contrastive_warmup_weight(epoch, args.contrastive_warmup_epochs)
        current_lr = scheduler.get_last_lr()[0]
        _unwrap_ddp(generator).epsilon = current_epsilon

        for samples, targets, genders in metrics_logger.log_every(train_loader, args.log_every, f"Epoch {epoch}"):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            genders = [g.lower() for g in genders]
            female_idx = [i for i, g in enumerate(genders) if g == "female"]
            male_idx = [i for i, g in enumerate(genders) if g == "male"]

            opt_g.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                tensors = samples.tensors
                delta = generator(tensors)
                perturbed_tensors = clamp_normalized(tensors + delta)
                perturbed = NestedTensor(perturbed_tensors, samples.mask)
                outputs, features = detr.forward_with_features(perturbed)

                if args.use_hungarian_score:
                    image_scores = _hungarian_matcher_score(detr, outputs, targets)
                else:
                    image_scores = _image_level_detection_score(outputs, top_k=args.score_top_k)

                projections = proj_head(features)
                loss_supcon, supcon_info = supcon_loss_fn(projections, image_scores, genders)

                loss_wasserstein = perturbed_tensors.new_tensor(0.0)
                if len(female_idx) > 0 and len(male_idx) > 0:
                    loss_wasserstein = _wasserstein_1d_unidirectional(
                        image_scores[female_idx], image_scores[male_idx])

                loss_det, _ = detr.detection_loss(outputs, targets)

                total_g = (
                    args.lambda_supcon * contrastive_weight * loss_supcon
                    + args.lambda_wass * loss_wasserstein
                    + current_beta * loss_det
                )

            with torch.no_grad():
                dv = perturbed_tensors - tensors
                delta_linf = dv.abs().amax(dim=(1, 2, 3)).mean()
                delta_l2 = dv.flatten(1).norm(p=2, dim=1).mean()

            total_g.backward()
            if args.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(generator.parameters()) + list(proj_head.parameters()), args.max_norm)
            opt_g.step()

            if ema_gen:
                ema_gen.update(_unwrap_ddp(generator))
                ema_proj.update(_unwrap_ddp(proj_head))

            metrics_logger.update(
                loss_supcon=loss_supcon.item(), loss_wasserstein=loss_wasserstein.item(),
                loss_det=loss_det.item(), total_g=total_g.item(),
                beta=current_beta, epsilon=current_epsilon, lr=current_lr,
                delta_linf=delta_linf.item(), delta_l2=delta_l2.item(),
                score_gap=supcon_info.get("score_gap", 0.0),
                score_f=supcon_info.get("score_f_mean", 0.0),
                score_m=supcon_info.get("score_m_mean", 0.0),
                n_bins_used=supcon_info.get("n_bins_used", 0),
                valid_anchors=supcon_info.get("valid_anchors", 0),
                n_f=len(female_idx), n_m=len(male_idx),
            )

        scheduler.step()
        metrics_logger.synchronize_between_processes()

        if utils.is_main_process():
            avg_gap = abs(metrics_logger.meters["score_gap"].global_avg)
            log_entry = {
                "epoch": epoch,
                "loss_supcon": metrics_logger.meters["loss_supcon"].global_avg,
                "loss_wasserstein": metrics_logger.meters["loss_wasserstein"].global_avg,
                "loss_det": metrics_logger.meters["loss_det"].global_avg,
                "total_g": metrics_logger.meters["total_g"].global_avg,
                "beta": current_beta, "epsilon": current_epsilon, "lr": current_lr,
                "delta_linf": metrics_logger.meters["delta_linf"].global_avg,
                "delta_l2": metrics_logger.meters["delta_l2"].global_avg,
                "score_f": metrics_logger.meters["score_f"].global_avg,
                "score_m": metrics_logger.meters["score_m"].global_avg,
                "score_gap": metrics_logger.meters["score_gap"].global_avg,
                "abs_score_gap": avg_gap,
                "valid_anchors": metrics_logger.meters["valid_anchors"].global_avg,
                "n_bins_used": metrics_logger.meters["n_bins_used"].global_avg,
            }
            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            is_best = avg_gap < best_score_gap
            if is_best:
                best_score_gap = avg_gap
                best_epoch = epoch

            print(f"\n[Epoch {epoch}] Anchor=MALE")
            print(f"  SupCon: {log_entry['loss_supcon']:.4f} (valid_anchors: {log_entry['valid_anchors']:.0f})")
            print(f"  Wasserstein: {log_entry['loss_wasserstein']:.4f}")
            print(f"  Detection: {log_entry['loss_det']:.4f} | Total: {log_entry['total_g']:.4f}")
            print(f"  Score (F/M): {log_entry['score_f']:.4f} / {log_entry['score_m']:.4f}")
            print(f"  Score Gap: {log_entry['score_gap']:.4f}")
            print(f"  Epsilon: {current_epsilon:.4f}, LR: {current_lr:.6f}")
            if is_best:
                print(f"  *** NEW BEST (|gap|={avg_gap:.4f}) ***")

            if (epoch + 1) % args.save_every == 0:
                sd = {"epoch": epoch, "generator": _unwrap_ddp(generator).state_dict(),
                      "proj_head": _unwrap_ddp(proj_head).state_dict(),
                      "opt_g": opt_g.state_dict(), "scheduler": scheduler.state_dict(), "args": vars(args)}
                if ema_gen:
                    sd["ema_gen"] = ema_gen.state_dict()
                    sd["ema_proj"] = ema_proj.state_dict()
                torch.save(sd, output_dir / "checkpoints" / f"epoch_{epoch:04d}.pth")
                if is_best:
                    best_sd = copy.deepcopy(sd)
                    if ema_gen:
                        best_sd["generator"] = ema_gen.state_dict()
                        best_sd["proj_head"] = ema_proj.state_dict()
                    torch.save(best_sd, output_dir / "checkpoints" / "best_model.pth")
                    print(f"  Best model saved")

        if args.distributed:
            dist.barrier()

    if utils.is_main_process():
        print(f"\n{'='*70}")
        print(f"Training Complete — Anchor=MALE (fix8gpu_anchor_man)")
        print(f"Best epoch: {best_epoch} (|gap|={best_score_gap:.4f})")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
