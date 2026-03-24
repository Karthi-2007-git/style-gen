"""Inference entrypoint for generating handwriting from a style sample and text."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import torch
from diffusers import DDIMScheduler, DDPMScheduler
from tqdm.auto import tqdm

from models.generator import StyleTextGenerator
from models.style_encoder import Style_Encoder
from models.text_encoder import PAD_TOKEN, Transformer_Text_Encoder, build_vocab, encode_text
from utils import (
    GENERATOR_SIZE,
    compose_handwriting_page,
    load_style_image,
    resize_for_style_encoder,
    save_generated,
)


def _cfg_value(config: dict | object | None, key: str, default):
    if isinstance(config, dict):
        return config.get(key, default)
    if config is not None and hasattr(config, key):
        return getattr(config, key)
    return default


def load_models(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[Style_Encoder, Transformer_Text_Encoder, StyleTextGenerator, dict[str, int], dict | None]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_config = checkpoint.get("config")
    char2idx = checkpoint.get("char2idx")
    if not char2idx:
        char2idx, _ = build_vocab()

    pad_idx = char2idx[PAD_TOKEN]
    vocab_size = len(char2idx)
    style_dim = _cfg_value(saved_config, "style_dim", 512)
    text_dim = _cfg_value(saved_config, "text_dim", 256)
    fusion_dim = _cfg_value(saved_config, "fusion_dim", 256)
    max_len = _cfg_value(saved_config, "max_len", 256)

    style_encoder = Style_Encoder(output_dim=style_dim).to(device)
    text_encoder = Transformer_Text_Encoder(
        vocab_size=vocab_size,
        embed_dim=text_dim,
        max_len=max_len,
        pad_idx=pad_idx,
    ).to(device)
    generator = StyleTextGenerator(
        style_dim=style_dim,
        text_dim=text_dim,
        fusion_dim=fusion_dim,
    ).to(device)

    style_encoder.load_state_dict(checkpoint["style_encoder"])
    text_encoder.load_state_dict(checkpoint["text_encoder"])
    generator.load_state_dict(checkpoint["generator"])

    style_encoder.eval()
    text_encoder.eval()
    generator.eval()

    return style_encoder, text_encoder, generator, char2idx, saved_config


@lru_cache(maxsize=2)
def _load_models_cached(
    checkpoint_path: str,
    device_name: str,
) -> tuple[Style_Encoder, Transformer_Text_Encoder, StyleTextGenerator, dict[str, int], dict | None]:
    return load_models(checkpoint_path=checkpoint_path, device=torch.device(device_name))


def parse_text_lines(text_args: list[str] | None, text_file: str | None) -> list[str]:
    lines: list[str] = []

    if text_file:
        file_text = Path(text_file).read_text(encoding="utf-8")
        file_lines = file_text.splitlines()
        if not file_lines and file_text == "":
            file_lines = [""]
        lines.extend(file_lines)

    if text_args:
        for item in text_args:
            item_lines = item.splitlines() or [item]
            lines.extend(item_lines)

    if not lines:
        raise ValueError("Provide at least one text line via --text or --text-file.")

    return lines


def chunked(items: Iterable[tuple[int, str]], chunk_size: int) -> Iterable[list[tuple[int, str]]]:
    batch: list[tuple[int, str]] = []
    for item in items:
        batch.append(item)
        if len(batch) == chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_scheduler(sampler: str, steps: int):
    if sampler == "ddim":
        scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    else:
        scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(steps)
    return scheduler


@torch.no_grad()
def sample_batch(
    generator: StyleTextGenerator,
    style_emb: torch.Tensor,
    text_emb: torch.Tensor,
    sampler: str,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    scheduler = build_scheduler(sampler=sampler, steps=steps)
    images = torch.randn(style_emb.shape[0], 1, *GENERATOR_SIZE, device=device)

    for timestep in tqdm(scheduler.timesteps, desc=f"{sampler.upper()} sampling", leave=False):
        timestep_value = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        timesteps = torch.full(
            (style_emb.shape[0],),
            timestep_value,
            device=device,
            dtype=torch.long,
        )
        noise_pred = generator(images, timesteps, style_emb, text_emb)
        images = scheduler.step(noise_pred, timestep, images).prev_sample

    return images.clamp(-1.0, 1.0)


@torch.no_grad()
def generate_images_from_tensors(
    style_images: torch.Tensor,
    tokens: torch.Tensor,
    style_encoder: Style_Encoder,
    text_encoder: Transformer_Text_Encoder,
    generator: StyleTextGenerator,
    sampler: str,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    if style_images.ndim == 3:
        style_images = style_images.unsqueeze(0)
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)

    style_images = style_images.to(device)
    tokens = tokens.to(device)

    if style_images.shape[-2:] != (64, 256):
        style_images = resize_for_style_encoder(style_images)

    style_embeddings = style_encoder(style_images)
    text_embeddings = text_encoder(tokens)
    return sample_batch(
        generator=generator,
        style_emb=style_embeddings,
        text_emb=text_embeddings,
        sampler=sampler,
        steps=steps,
        device=device,
    )


@torch.no_grad()
def generate_lines(
    style_image_path: str,
    text_lines: list[str],
    style_encoder: Style_Encoder,
    text_encoder: Transformer_Text_Encoder,
    generator: StyleTextGenerator,
    char2idx: dict[str, int],
    device: torch.device,
    sampler: str,
    steps: int,
    max_len: int,
    batch_size: int,
) -> list[torch.Tensor]:
    _, style_image = load_style_image(style_image_path, device=device)
    style_embedding = style_encoder(style_image)

    line_images: list[torch.Tensor | None] = [None] * len(text_lines)
    non_empty_lines = [(index, line) for index, line in enumerate(text_lines) if line != ""]

    for batch in chunked(non_empty_lines, batch_size):
        tokens = torch.stack([encode_text(line, char2idx, max_len) for _, line in batch]).to(device)
        text_embeddings = text_encoder(tokens)
        repeated_style = style_embedding.expand(len(batch), -1)
        generated = sample_batch(
            generator=generator,
            style_emb=repeated_style,
            text_emb=text_embeddings,
            sampler=sampler,
            steps=steps,
            device=device,
        )

        for (index, _), image in zip(batch, generated, strict=True):
            line_images[index] = image.detach().cpu()

    for index, line in enumerate(text_lines):
        if line_images[index] is None:
            line_images[index] = torch.ones(1, *GENERATOR_SIZE)

    return [image for image in line_images if image is not None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate handwriting with style_gen.")
    parser.add_argument("--checkpoint", required=True, help="Path to a training checkpoint.")
    parser.add_argument("--style-image", required=True, help="Path to a handwriting sample image.")
    parser.add_argument("--text", nargs="*", default=None, help="One or more text lines to generate.")
    parser.add_argument("--text-file", default=None, help="UTF-8 file containing text to render.")
    parser.add_argument("--output", default="outputs/generated_page.png", help="Where to save the composed page.")
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--steps", type=int, default=None, help="Sampling steps. Defaults: ddpm=1000, ddim=50.")
    parser.add_argument("--max-len", type=int, default=None, help="Override checkpoint max token length.")
    parser.add_argument("--batch-size", type=int, default=4, help="How many lines to generate at once.")
    parser.add_argument("--line-spacing", type=int, default=24, help="Pixels between rendered lines.")
    parser.add_argument("--page-margin", type=int, default=24, help="Pixels around the composed page.")
    parser.add_argument("--device", default=None, help="Force a device, for example `cuda` or `cpu`.")
    return parser.parse_args()


def resolve_device(device_name: str | None) -> torch.device:
    return torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))


def generate_handwriting_page(
    checkpoint_path: str,
    style_image_path: str,
    text: str | list[str],
    output_path: str = "outputs/generated_page.png",
    sampler: str = "ddim",
    steps: int | None = None,
    max_len: int | None = None,
    batch_size: int = 4,
    line_spacing: int = 24,
    page_margin: int = 24,
    device_name: str | None = None,
) -> str:
    device = resolve_device(device_name)
    style_encoder, text_encoder, generator, char2idx, saved_config = _load_models_cached(
        checkpoint_path=checkpoint_path,
        device_name=str(device),
    )

    resolved_max_len = max_len or _cfg_value(saved_config, "max_len", 256)
    resolved_steps = steps or (1000 if sampler == "ddpm" else 50)
    text_lines = text if isinstance(text, list) else parse_text_lines([text], None)

    line_images = generate_lines(
        style_image_path=style_image_path,
        text_lines=text_lines,
        style_encoder=style_encoder,
        text_encoder=text_encoder,
        generator=generator,
        char2idx=char2idx,
        device=device,
        sampler=sampler,
        steps=resolved_steps,
        max_len=resolved_max_len,
        batch_size=max(batch_size, 1),
    )
    page = compose_handwriting_page(
        line_images=line_images,
        line_spacing=line_spacing,
        page_margin=page_margin,
    )
    save_generated(page, output_path)
    return output_path


def main() -> None:
    args = parse_args()
    text_lines = parse_text_lines(args.text, args.text_file)
    output_path = generate_handwriting_page(
        checkpoint_path=args.checkpoint,
        style_image_path=args.style_image,
        text=text_lines,
        output_path=args.output,
        sampler=args.sampler,
        steps=args.steps,
        max_len=args.max_len,
        batch_size=args.batch_size,
        line_spacing=args.line_spacing,
        page_margin=args.page_margin,
        device_name=args.device,
    )
    print(f"Saved generated handwriting page to {output_path}")


if __name__ == "__main__":
    main()
