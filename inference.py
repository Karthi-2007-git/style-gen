"""
inference.py — Generate handwriting images from a trained style_gen checkpoint.

How it works
────────────
  Training   : real image → add noise → model learns to predict that noise
  Inference  : pure noise → model removes noise step-by-step → handwriting

Usage
─────
  # Full 1000-step DDPM (highest quality)
  python inference.py \\
      --checkpoint checkpoints/style_gen_final.pt \\
      --style_image samples/my_handwriting.png \\
      --text "Hello world" \\
      --output outputs/generated.png

  # Fast 50-step DDIM (good quality, ~20× faster)
  python inference.py \\
      --checkpoint checkpoints/style_gen_final.pt \\
      --style_image samples/my_handwriting.png \\
      --text "Hello world" \\
      --output outputs/generated.png \\
      --sampler ddim \\
      --steps 50

  # Generate a batch with different texts, same style
  python inference.py \\
      --checkpoint checkpoints/style_gen_final.pt \\
      --style_image samples/my_handwriting.png \\
      --text "First line" "Second line" "Third line" \\
      --output outputs/batch.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from torchvision.utils import save_image
from tqdm import tqdm

from models.generator import StyleTextGenerator
from models.style_encoder import Style_Encoder
from models.text_encoder import Transformer_Text_Encoder, build_vocab, encode_text
from utils import load_style_image, save_generated


# ──────────────────────────────────────────────────────────────────────────────
# Load checkpoint
# ──────────────────────────────────────────────────────────────────────────────

def load_models(
    checkpoint_path: str,
    device: torch.device,
    vocab_size: int,
) -> tuple[Style_Encoder, Transformer_Text_Encoder, StyleTextGenerator]:

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt.get("config", None)

    # Pull dims from saved config if available, else use defaults
    style_dim  = getattr(cfg, "style_dim",  512)
    text_dim   = getattr(cfg, "text_dim",   256)
    fusion_dim = getattr(cfg, "fusion_dim", 256)

    style_encoder = Style_Encoder().to(device)
    text_encoder  = Transformer_Text_Encoder(
        vocab_size=vocab_size, embed_dim=text_dim
    ).to(device)
    generator = StyleTextGenerator(
        style_dim=style_dim, text_dim=text_dim, fusion_dim=fusion_dim
    ).to(device)

    style_encoder.load_state_dict(ckpt["style_encoder"])
    text_encoder.load_state_dict(ckpt["text_encoder"])
    generator.load_state_dict(ckpt["generator"])

    style_encoder.eval()
    text_encoder.eval()
    generator.eval()

    print(f"  ✓ loaded checkpoint  →  {checkpoint_path}")
    if cfg:
        epoch = ckpt.get("epoch", "?")
        step  = ckpt.get("global_step", "?")
        print(f"  ✓ trained for {epoch + 1} epochs  ({step} steps)")

    return style_encoder, text_encoder, generator


# ──────────────────────────────────────────────────────────────────────────────
# Encode inputs
# ──────────────────────────────────────────────────────────────────────────────

def encode_inputs(
    style_img_path: str,
    texts: list[str],
    style_encoder: Style_Encoder,
    text_encoder: Transformer_Text_Encoder,
    char2idx: dict,
    device: torch.device,
    max_len: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        style_emb  (B, 512)
        text_emb   (B, 256)
    where B = len(texts).
    """
    B = len(texts)

    # ── style: load once, expand to batch size ──
    _, style_enc_img = load_style_image(style_img_path, device=device)
    style_enc_img = style_enc_img.expand(B, -1, -1, -1)     # (B, 1, 64, 256)
    style_emb = style_encoder(style_enc_img)                 # (B, 512)

    # ── text: tokenise each string ──
    tokens = torch.stack([
        encode_text(t, char2idx, max_len) for t in texts
    ]).to(device)                                            # (B, max_len)
    text_emb = text_encoder(tokens)                          # (B, 256)

    return style_emb, text_emb


# ──────────────────────────────────────────────────────────────────────────────
# DDPM reverse loop  (1000 steps, highest quality)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ddpm_sample(
    generator: StyleTextGenerator,
    style_emb: torch.Tensor,
    text_emb: torch.Tensor,
    num_steps: int = 1000,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Standard DDPM reverse diffusion.
    Starts from pure Gaussian noise and denoises step-by-step.

    Returns: (B, 1, 128, 512) float tensor in [-1, 1]
    """
    B = style_emb.shape[0]
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_steps)

    # Start from pure noise
    x = torch.randn(B, 1, 128, 512, device=device)

    for t in tqdm(scheduler.timesteps, desc="DDPM sampling"):
        t_batch = t.unsqueeze(0).expand(B).to(device)
        noise_pred = generator(x, t_batch, style_emb, text_emb)
        x = scheduler.step(noise_pred, t, x).prev_sample

    return x                                                 # (B, 1, 128, 512)


# ──────────────────────────────────────────────────────────────────────────────
# DDIM reverse loop  (50 steps, ~20× faster, good quality)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_sample(
    generator: StyleTextGenerator,
    style_emb: torch.Tensor,
    text_emb: torch.Tensor,
    num_steps: int = 50,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    DDIM reverse diffusion — deterministic, much fewer steps.
    Produces slightly softer results than full DDPM but is 20× faster.

    Returns: (B, 1, 128, 512) float tensor in [-1, 1]
    """
    B = style_emb.shape[0]
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
    )
    scheduler.set_timesteps(num_steps)

    x = torch.randn(B, 1, 128, 512, device=device)

    for t in tqdm(scheduler.timesteps, desc=f"DDIM sampling ({num_steps} steps)"):
        t_batch = t.unsqueeze(0).expand(B).to(device)
        noise_pred = generator(x, t_batch, style_emb, text_emb)
        x = scheduler.step(noise_pred, t, x).prev_sample

    return x                                                 # (B, 1, 128, 512)


# ──────────────────────────────────────────────────────────────────────────────
# Save batch output
# ──────────────────────────────────────────────────────────────────────────────

def save_output(images: torch.Tensor, output_path: str, texts: list[str]) -> None:
    """
    If single image  → save as-is.
    If batch         → save as a vertical grid (one row per text).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalise [-1,1] → [0,1] for torchvision
    imgs_01 = (images.clamp(-1.0, 1.0) + 1.0) / 2.0

    if imgs_01.shape[0] == 1:
        save_image(imgs_01, str(path))
    else:
        # nrow=1 → stack vertically
        save_image(imgs_01, str(path), nrow=1, padding=4, pad_value=1.0)

    print(f"  ✓ saved {len(texts)} image(s) → {path}")
    for i, t in enumerate(texts):
        print(f"    [{i}] \"{t}\"")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate handwriting with style_gen",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--checkpoint",   required=True,  help="Path to .pt checkpoint")
    p.add_argument("--style_image",  required=True,  help="Path to a handwriting sample image")
    p.add_argument("--text",         required=True,  nargs="+", help="Text string(s) to generate")
    p.add_argument("--output",       default="outputs/generated.png", help="Output image path")
    p.add_argument("--sampler",      default="ddpm", choices=["ddpm", "ddim"],
                   help="ddpm = 1000 steps (best quality) | ddim = fast (default 50 steps)")
    p.add_argument("--steps",        type=int, default=None,
                   help="Override number of denoising steps (default: 1000 for ddpm, 50 for ddim)")
    p.add_argument("--max_len",      type=int, default=256, help="Max token length")
    p.add_argument("--device",       default=None,
                   help="cuda / cpu (auto-detected if not set)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── device ──
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── vocab ──
    char2idx, _ = build_vocab()
    vocab_size   = len(char2idx)            # 98

    # ── load models ──
    style_encoder, text_encoder, generator = load_models(
        args.checkpoint, device, vocab_size
    )

    # ── encode style + text ──
    with torch.no_grad():
        style_emb, text_emb = encode_inputs(
            style_img_path=args.style_image,
            texts=args.text,
            style_encoder=style_encoder,
            text_encoder=text_encoder,
            char2idx=char2idx,
            device=device,
            max_len=args.max_len,
        )

    # ── sample ──
    with torch.no_grad():
        if args.sampler == "ddim":
            steps  = args.steps or 50
            images = ddim_sample(generator, style_emb, text_emb, steps, device)
        else:
            steps  = args.steps or 1000
            images = ddpm_sample(generator, style_emb, text_emb, steps, device)

    # ── save ──
    save_output(images, args.output, args.text)


if __name__ == "__main__":
    main()