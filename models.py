import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .path_utils import DETR_REPO, default_detr_checkpoint, ensure_detr_repo_on_path

# ensure DETR repo is importable before loading its modules
ensure_detr_repo_on_path(DETR_REPO)

from main import get_args_parser
from models import build_model
from util.misc import NestedTensor, nested_tensor_from_tensor_list


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def clamp_normalized(tensor: torch.Tensor) -> torch.Tensor:
    """
    Clamp normalized tensors back to the valid pixel range (after ToTensor+Normalize).
    """
    mean = _IMAGENET_MEAN.to(tensor.device)
    std = _IMAGENET_STD.to(tensor.device)
    min_val = (0.0 - mean) / std
    max_val = (1.0 - mean) / std
    return torch.max(torch.min(tensor, max_val), min_val)


class FrozenDETR(nn.Module):
    """
    Wrapper around the existing DETR implementation that:
    - loads pretrained weights
    - freezes all parameters
    - exposes transformer features for the discriminator
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path],
        device: str = "cuda",
        args: Optional[argparse.Namespace] = None,
        detr_repo: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.detr_repo = Path(detr_repo) if detr_repo is not None else DETR_REPO
        ensure_detr_repo_on_path(self.detr_repo)
        if args is None:
            parser = get_args_parser()
            args = parser.parse_args([])
        args.device = device
        self.args = args

        model, criterion, postprocessors = build_model(args)
        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors

        self.device = torch.device(device)
        self.to(self.device)

        ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else default_detr_checkpoint(self.detr_repo)
        if ckpt_path is not None and ckpt_path.exists():
            state_dict = torch.load(str(ckpt_path), map_location=self.device)
            if isinstance(state_dict, dict) and "model" in state_dict:
                self.model.load_state_dict(state_dict["model"], strict=False)
            else:
                self.model.load_state_dict(state_dict, strict=False)

        self.freeze()

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

    @property
    def hidden_dim(self) -> int:
        return getattr(self.model.transformer, "d_model", 256)

    def forward(self, samples: NestedTensor):
        outputs, _ = self.forward_with_features(samples)
        return outputs

    def forward_with_features(self, samples) -> Tuple[dict, torch.Tensor]:
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        samples = samples.to(self.device)

        features, pos = self.model.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.model.transformer(self.model.input_proj(src), mask, self.model.query_embed.weight, pos[-1])[0]

        outputs_class = self.model.class_embed(hs)
        outputs_coord = self.model.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if getattr(self.model, "aux_loss", False):
            out["aux_outputs"] = self.model._set_aux_loss(outputs_class, outputs_coord)
        return out, hs[-1]

    def detection_loss(self, outputs: dict, targets) -> Tuple[torch.Tensor, dict]:
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        total = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        return total, loss_dict


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        x = self.conv(x)
        return F.relu(self.bn(x), inplace=True)


class PerturbationGenerator(nn.Module):
    """
    Lightweight U-Net style generator that emits a bounded perturbation.
    """

    def __init__(self, base_channels: int = 32, epsilon: float = 0.05) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.down1 = ConvBlock(3, base_channels, stride=1)
        self.down2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.down3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4, stride=1)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels)
        self.out_conv = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.down1(x)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        h3 = self.bottleneck(h3)
        u2 = self.up2(h3, h2.shape[-2:]) + h2
        u1 = self.up1(u2, h1.shape[-2:]) + h1
        delta = torch.tanh(self.out_conv(u1))
        return self.epsilon * delta


class GenderDiscriminator(nn.Module):
    """
    Simple discriminator operating on DETR decoder features.
    """

    def __init__(self, feature_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        # hs: (batch, num_queries, feature_dim)
        pooled = hs.mean(dim=1)
        return self.net(pooled)
