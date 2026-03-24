"""Evaluate a trained style_gen checkpoint on the IAM validation or test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision.utils import save_image
from tqdm.auto import tqdm

from evaluation.metrics import MetricsSummary, MetricsTracker
from inference import generate_images_from_tensors, load_models, resolve_device
from models.dataset import get_dataloader
from training.train import TrainConfig, unpack_batch
from utils import denormalize_tensor


def _summary_to_dict(summary: MetricsSummary) -> dict[str, float]:
    return {
        "psnr": float(summary.psnr),
        "ssim": float(summary.ssim),
        "fid": float(summary.fid),
        "cer": float(summary.cer),
    }


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str,
    split: str = "validation",
    batch_size: int = 4,
    num_batches: int | None = 10,
    sampler: str = "ddim",
    steps: int | None = None,
    max_len: int | None = None,
    num_workers: int = 0,
    cache_dir: str | None = None,
    fallback_writer_group_size: int = 20,
    device_name: str | None = None,
    save_preview_path: str | None = "outputs/eval_preview.png",
) -> MetricsSummary:
    device = resolve_device(device_name)
    style_encoder, text_encoder, generator, _, saved_config = load_models(checkpoint_path, device)

    resolved_max_len = max_len or int(
        (saved_config or {}).get("max_len", TrainConfig.max_len)
        if isinstance(saved_config, dict)
        else getattr(saved_config, "max_len", TrainConfig.max_len)
    )
    resolved_steps = steps or (50 if sampler == "ddim" else 1000)

    dataloader = get_dataloader(
        split=split,
        batch_size=batch_size,
        shuffle=False,
        max_len=resolved_max_len,
        num_workers=num_workers,
        drop_last=False,
        cache_dir=cache_dir,
        fallback_writer_group_size=fallback_writer_group_size,
    )

    tracker = MetricsTracker(device=device)
    preview_saved = False
    total_batches = len(dataloader) if num_batches is None else min(len(dataloader), num_batches)

    for batch_index, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split}"), start=1):
        real_images, tokens, style_images = unpack_batch(batch, device)
        generated_images = generate_images_from_tensors(
            style_images=style_images,
            tokens=tokens,
            style_encoder=style_encoder,
            text_encoder=text_encoder,
            generator=generator,
            sampler=sampler,
            steps=resolved_steps,
            device=device,
        )
        tracker.update(real_images=real_images, fake_images=generated_images)

        if save_preview_path and not preview_saved:
            preview_path = Path(save_preview_path)
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            preview = torch.cat([denormalize_tensor(real_images.cpu()), denormalize_tensor(generated_images.cpu())], dim=0)
            save_image(preview, str(preview_path), nrow=real_images.shape[0], padding=4)
            preview_saved = True

        if num_batches is not None and batch_index >= num_batches:
            break

    summary = tracker.compute()
    print(f"Evaluated {min(total_batches, batch_index)} batch(es) on {split}")
    print(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a style_gen checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a training checkpoint.")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=10, help="Limit batches for faster evaluation.")
    parser.add_argument("--sampler", default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--fallback-writer-group-size", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-preview", type=str, default="outputs/eval_preview.png")
    parser.add_argument("--save-json", type=str, default="outputs/eval_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        sampler=args.sampler,
        steps=args.steps,
        max_len=args.max_len,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        fallback_writer_group_size=args.fallback_writer_group_size,
        device_name=args.device,
        save_preview_path=args.save_preview,
    )

    output_path = Path(args.save_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_summary_to_dict(summary), indent=2), encoding="utf-8")
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
