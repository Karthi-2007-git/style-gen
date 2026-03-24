"""Shared image and layout utilities for style_gen."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.utils import save_image

__all__ = [
    "GENERATOR_SIZE",
    "STYLE_ENC_SIZE",
    "denormalize_tensor",
    "normalize_tensor",
    "resize_for_style_encoder",
    "load_style_image",
    "compose_handwriting_page",
    "tensor_to_pil",
    "save_generated",
]

GENERATOR_SIZE = (128, 512)
STYLE_ENC_SIZE = (64, 256)


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Map image tensors from [0, 1] to [-1, 1]."""
    return tensor.mul(2.0).sub(1.0)


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Map image tensors from [-1, 1] to [0, 1]."""
    return tensor.clamp(-1.0, 1.0).add(1.0).div(2.0)


def resize_for_style_encoder(images: torch.Tensor) -> torch.Tensor:
    """
    Resize handwriting images for the style encoder.

    Accepts `(B, 1, H, W)` or `(1, H, W)` tensors in any numeric range.
    Returns a `(B, 1, 64, 256)` tensor.
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(
            f"Expected image tensor of shape (B, 1, H, W) or (1, H, W), got {tuple(images.shape)}"
        )
    return F.interpolate(images, size=STYLE_ENC_SIZE, mode="bilinear", align_corners=False)


def load_style_image(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load a handwriting sample and return generator-sized and style-encoder-sized tensors.
    """
    image = Image.open(path).convert("L")
    image = ImageOps.exif_transpose(image)
    transform = transforms.Compose(
        [
            transforms.Resize(GENERATOR_SIZE),
            transforms.ToTensor(),
        ]
    )
    generator_image = normalize_tensor(transform(image)).unsqueeze(0).to(device)
    style_image = resize_for_style_encoder(generator_image)
    return generator_image, style_image


def _to_line_tensor(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("Expected a single image tensor when composing a page.")
        image = image.squeeze(0)
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError(f"Expected line image shape (1, H, W), got {tuple(image.shape)}")
    return image.detach().cpu().float()


def compose_handwriting_page(
    line_images: Iterable[torch.Tensor],
    line_spacing: int = 24,
    page_margin: int = 24,
    blank_line_height: int | None = None,
    background_value: float = 1.0,
) -> torch.Tensor:
    """
    Stack generated line images into a single page tensor in [-1, 1].

    Empty lines should be passed in as white line tensors if they should take full line height.
    """
    if line_spacing < 0 or page_margin < 0:
        raise ValueError("line_spacing and page_margin must be non-negative")

    lines = [_to_line_tensor(image) for image in line_images]
    if not lines:
        raise ValueError("line_images must contain at least one image")

    widths = [line.shape[-1] for line in lines]
    line_height = blank_line_height or max(line.shape[-2] for line in lines)
    total_height = page_margin * 2 + sum(max(line.shape[-2], line_height) for line in lines)
    total_height += line_spacing * max(len(lines) - 1, 0)
    total_width = page_margin * 2 + max(widths)

    canvas = torch.full((1, total_height, total_width), background_value, dtype=lines[0].dtype)
    y = page_margin
    for line in lines:
        height, width = line.shape[-2:]
        canvas[:, y : y + height, page_margin : page_margin + width] = line
        y += max(height, line_height) + line_spacing

    return canvas


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized handwriting tensor to a grayscale PIL image."""
    image = denormalize_tensor(_to_line_tensor(tensor)).mul(255).byte().squeeze(0)
    return Image.fromarray(image.numpy(), mode="L")


def save_generated(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a normalized image tensor to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError(f"Expected tensor shape (B, 1, H, W), got {tuple(tensor.shape)}")

    save_image(denormalize_tensor(tensor), str(output_path), nrow=1, padding=0)
