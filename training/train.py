"""Training entrypoint for the style_gen diffusion pipeline."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from models.dataset import get_dataloader
from models.generator import StyleTextGenerator
from models.style_encoder import Style_Encoder
from models.text_encoder import PAD_TOKEN, Transformer_Text_Encoder, build_vocab
from utils import resize_for_style_encoder


@dataclass
class TrainConfig:
    train_split: str = "train"
    val_split: str = "validation"
    batch_size: int = 4
    val_batch_size: int = 4
    grad_accum_steps: int = 1
    max_len: int = 256
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    warmup_steps: int = 500
    min_lr_scale: float = 0.1
    num_train_timesteps: int = 1000
    num_workers: int = 0
    fallback_writer_group_size: int = 20
    mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    enable_attention_slicing: bool = True
    log_every: int = 25
    save_every: int = 1
    output_dir: str = "training/checkpoints"
    resume_from: str | None = None
    seed: int = 42
    device: str | None = None
    cache_dir: str | None = None
    text_dim: int = 256
    style_dim: int = 512
    fusion_dim: int = 256


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def unpack_batch(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        images = batch["image"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        style_images = batch.get("style_image")
        if style_images is None:
            style_images = resize_for_style_encoder(images)
        else:
            style_images = style_images.to(device, non_blocking=True)
    elif isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Expected at least image and token tensors in the batch.")
        images = batch[0].to(device, non_blocking=True)
        tokens = batch[1].to(device, non_blocking=True)
        if len(batch) >= 4 and torch.is_tensor(batch[3]):
            style_images = batch[3].to(device, non_blocking=True)
        else:
            style_images = resize_for_style_encoder(images)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)!r}")

    if style_images.shape[-2:] != (64, 256):
        style_images = resize_for_style_encoder(style_images)
    return images, tokens, style_images


def compute_diffusion_loss(
    batch: Any,
    style_encoder: Style_Encoder,
    text_encoder: Transformer_Text_Encoder,
    generator: StyleTextGenerator,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    images, tokens, style_images = unpack_batch(batch, device)
    batch_size = images.shape[0]
    noise = torch.randn_like(images)
    timesteps = torch.randint(
        low=0,
        high=noise_scheduler.config.num_train_timesteps,
        size=(batch_size,),
        device=device,
        dtype=torch.long,
    )

    noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

    with autocast_context(device, use_amp):
        style_emb = style_encoder(style_images)
        text_emb = text_encoder(tokens)
        noise_pred = generator(noisy_images, timesteps, style_emb, text_emb)
        loss = F.mse_loss(noise_pred.float(), noise.float())

    return loss


def build_lr_scheduler(optimizer: AdamW, total_steps: int, warmup_steps: int, min_lr_scale: float) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return max(step + 1, 1) / warmup_steps

        cosine_steps = max(total_steps - warmup_steps, 1)
        progress = min(max(step - warmup_steps, 0) / cosine_steps, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def configure_generator_for_memory(generator: StyleTextGenerator, config: TrainConfig) -> None:
    if config.enable_gradient_checkpointing and hasattr(generator.unet, "enable_gradient_checkpointing"):
        generator.unet.enable_gradient_checkpointing()
    if config.enable_attention_slicing and hasattr(generator.unet, "set_attention_slice"):
        generator.unet.set_attention_slice("auto")


@torch.no_grad()
def run_validation(
    dataloader: Any,
    style_encoder: Style_Encoder,
    text_encoder: Transformer_Text_Encoder,
    generator: StyleTextGenerator,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    use_amp: bool,
) -> float:
    style_encoder.eval()
    text_encoder.eval()
    generator.eval()

    losses = []
    for batch in tqdm(dataloader, desc="Validation", leave=False):
        loss = compute_diffusion_loss(
            batch=batch,
            style_encoder=style_encoder,
            text_encoder=text_encoder,
            generator=generator,
            noise_scheduler=noise_scheduler,
            device=device,
            use_amp=use_amp,
        )
        losses.append(loss.item())

    style_encoder.train()
    text_encoder.train()
    generator.train()
    return sum(losses) / max(len(losses), 1)


def save_checkpoint(
    checkpoint_path: Path,
    config: TrainConfig,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    style_encoder: Style_Encoder,
    text_encoder: Transformer_Text_Encoder,
    generator: StyleTextGenerator,
    optimizer: AdamW,
    lr_scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    char2idx: dict[str, int],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "char2idx": char2idx,
            "style_encoder": style_encoder.state_dict(),
            "text_encoder": text_encoder.state_dict(),
            "generator": generator.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    style_encoder: Style_Encoder,
    text_encoder: Transformer_Text_Encoder,
    generator: StyleTextGenerator,
    optimizer: AdamW,
    lr_scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
) -> tuple[int, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    style_encoder.load_state_dict(checkpoint["style_encoder"])
    text_encoder.load_state_dict(checkpoint["text_encoder"])
    generator.load_state_dict(checkpoint["generator"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    return start_epoch, global_step, best_val_loss


def train(config: TrainConfig | None = None) -> dict[str, float]:
    config = config or TrainConfig()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    use_amp = config.mixed_precision and device.type == "cuda"

    char2idx, _ = build_vocab()
    pad_idx = char2idx[PAD_TOKEN]
    vocab_size = len(char2idx)

    train_loader = get_dataloader(
        split=config.train_split,
        batch_size=config.batch_size,
        shuffle=True,
        max_len=config.max_len,
        num_workers=config.num_workers,
        cache_dir=config.cache_dir,
        drop_last=True,
        fallback_writer_group_size=config.fallback_writer_group_size,
    )
    val_loader = get_dataloader(
        split=config.val_split,
        batch_size=config.val_batch_size,
        shuffle=False,
        max_len=config.max_len,
        num_workers=config.num_workers,
        cache_dir=config.cache_dir,
        drop_last=False,
        fallback_writer_group_size=config.fallback_writer_group_size,
    )

    style_encoder = Style_Encoder(output_dim=config.style_dim).to(device)
    text_encoder = Transformer_Text_Encoder(
        vocab_size=vocab_size,
        embed_dim=config.text_dim,
        max_len=config.max_len,
        pad_idx=pad_idx,
    ).to(device)
    generator = StyleTextGenerator(
        style_dim=config.style_dim,
        text_dim=config.text_dim,
        fusion_dim=config.fusion_dim,
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)
    optimizer = AdamW(
        list(style_encoder.parameters())
        + list(text_encoder.parameters())
        + list(generator.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    configure_generator_for_memory(generator, config)

    total_steps = math.ceil(max(len(train_loader), 1) / max(config.grad_accum_steps, 1)) * config.epochs
    lr_scheduler = build_lr_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=config.warmup_steps,
        min_lr_scale=config.min_lr_scale,
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    if config.resume_from:
        start_epoch, global_step, best_val_loss = load_checkpoint(
            checkpoint_path=config.resume_from,
            device=device,
            style_encoder=style_encoder,
            text_encoder=text_encoder,
            generator=generator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
        )

    history: list[dict[str, float]] = []
    print(f"Training on {device} with {vocab_size} text tokens")

    for epoch in range(start_epoch, config.epochs):
        style_encoder.train()
        text_encoder.train()
        generator.train()

        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(progress, start=1):
            loss = compute_diffusion_loss(
                batch=batch,
                style_encoder=style_encoder,
                text_encoder=text_encoder,
                generator=generator,
                noise_scheduler=noise_scheduler,
                device=device,
                use_amp=use_amp,
            )
            running_loss += loss.item()

            scaled_loss = loss / max(config.grad_accum_steps, 1)
            scaler.scale(scaled_loss).backward()

            should_step = (
                batch_index % max(config.grad_accum_steps, 1) == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(style_encoder.parameters())
                    + list(text_encoder.parameters())
                    + list(generator.parameters()),
                    max_norm=config.grad_clip_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

            global_step += 1

            if batch_index % config.log_every == 0 or batch_index == len(train_loader):
                avg_loss = running_loss / batch_index
                current_lr = optimizer.param_groups[0]["lr"]
                progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}")

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss = run_validation(
            dataloader=val_loader,
            style_encoder=style_encoder,
            text_encoder=text_encoder,
            generator=generator,
            noise_scheduler=noise_scheduler,
            device=device,
            use_amp=use_amp,
        )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                checkpoint_path=output_dir / "style_gen_best.pt",
                config=config,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                style_encoder=style_encoder,
                text_encoder=text_encoder,
                generator=generator,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                char2idx=char2idx,
            )

        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                checkpoint_path=output_dir / f"style_gen_epoch_{epoch + 1:03d}.pt",
                config=config,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                style_encoder=style_encoder,
                text_encoder=text_encoder,
                generator=generator,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                char2idx=char2idx,
            )

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | best_val_loss={best_val_loss:.4f}"
        )

    save_checkpoint(
        checkpoint_path=output_dir / "style_gen_final.pt",
        config=config,
        epoch=config.epochs - 1,
        global_step=global_step,
        best_val_loss=best_val_loss,
        style_encoder=style_encoder,
        text_encoder=text_encoder,
        generator=generator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        char2idx=char2idx,
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {
        "train_loss": history[-1]["train_loss"] if history else float("nan"),
        "val_loss": history[-1]["val_loss"] if history else float("nan"),
        "best_val_loss": best_val_loss,
    }


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the style_gen handwriting diffusion model.")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--val-batch-size", type=int, default=TrainConfig.val_batch_size)
    parser.add_argument("--grad-accum-steps", type=int, default=TrainConfig.grad_accum_steps)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--output-dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--fallback-writer-group-size", type=int, default=TrainConfig.fallback_writer_group_size)
    args = parser.parse_args()

    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        device=args.device,
        cache_dir=args.cache_dir,
        seed=args.seed,
        fallback_writer_group_size=args.fallback_writer_group_size,
    )


if __name__ == "__main__":
    train(parse_args())
