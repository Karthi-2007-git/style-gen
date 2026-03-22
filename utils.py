"""
utils.py — Shared image utilities for style_gen.

Used by inference.py (and optionally training/train.py).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

__all__ = [
    "load_style_image",
    "tensor_to_pil",
    "save_generated",
]

# ── the two sizes your pipeline uses ──────────────────────────────────────────
GENERATOR_SIZE = (128, 512)   # what the UNet generates
STYLE_ENC_SIZE = (64, 256)    # what Style_Encoder expects


def load_style_image(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load a handwriting image from disk and return two tensors:

        generator_img  (1, 1, 128, 512) — for the diffusion target / reference
        style_enc_img  (1, 1,  64, 256) — downsampled input for Style_Encoder

    Both are normalised to [-1, 1].
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(GENERATOR_SIZE),
        transforms.ToTensor(),                   # → [0, 1]
        transforms.Normalize((0.5,), (0.5,)),    # → [-1, 1]
    ])

    img = Image.open(path).convert("L")
    generator_img = transform(img).unsqueeze(0).to(device)  # (1,1,128,512)

    style_enc_img = F.interpolate(
        generator_img,
        size=STYLE_ENC_SIZE,
        mode="bilinear",
        align_corners=False,
    )                                                        # (1,1,64,256)

    return generator_img, style_enc_img


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a (1, 1, H, W) or (1, H, W) float tensor in [-1, 1] to a PIL image.
    """
    t = tensor.squeeze().cpu().float()
    t = (t.clamp(-1.0, 1.0) + 1.0) / 2.0   # → [0, 1]
    t = (t * 255).byte()
    return Image.fromarray(t.numpy(), mode="L")


def save_generated(tensor: torch.Tensor, path: str | Path) -> None:
    """
    Save a (1, 1, H, W) float tensor in [-1, 1] as a PNG image.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # save_image expects [0, 1]
    img = (tensor.clamp(-1.0, 1.0) + 1.0) / 2.0
    save_image(img, str(path))
    print(f"  ✓ saved → {path}")