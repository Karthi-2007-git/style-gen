"""
metrics.py — Evaluation metrics for style_gen.

Metrics
───────
  PSNR     — pixel-level fidelity (higher = better)
  SSIM     — structural similarity (higher = better, max 1.0)
  FID      — Fréchet Inception Distance on CNN features (lower = better)
  CER      — Character Error Rate via a lightweight CNN-CTC decoder (lower = better)
  MetricsTracker — accumulates all four over a full eval loop

All functions operate on float tensors normalised to [-1, 1]  (as your dataset outputs).
They internally rescale to [0, 1] where needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "psnr",
    "ssim",
    "FIDTracker",
    "CERTracker",
    "MetricsTracker",
    "MetricsSummary",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_01(x: torch.Tensor) -> torch.Tensor:
    """Rescale from [-1, 1] → [0, 1]."""
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


# ──────────────────────────────────────────────────────────────────────────────
# 1. PSNR
# ──────────────────────────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio, averaged over the batch.

    Args:
        pred   (B, 1, H, W) : generated images, range [-1, 1]
        target (B, 1, H, W) : ground-truth images, range [-1, 1]
    Returns:
        scalar tensor — mean PSNR in dB
    """
    pred   = _to_01(pred)
    target = _to_01(target)
    mse_per_image = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    # Avoid log(0) when MSE is exactly zero (perfect reconstruction)
    mse_per_image = mse_per_image.clamp_min(1e-10)
    psnr_per_image = 10.0 * torch.log10(torch.tensor(data_range ** 2) / mse_per_image)
    return psnr_per_image.mean()


# ──────────────────────────────────────────────────────────────────────────────
# 2. SSIM
# ──────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """1D Gaussian → outer product → 2D kernel (window_size, window_size)."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.outer(g)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    Structural Similarity Index, averaged over the batch.

    Args:
        pred   (B, 1, H, W) : generated images, range [-1, 1]
        target (B, 1, H, W) : ground-truth images, range [-1, 1]
    Returns:
        scalar tensor in (-1, 1]; higher = more similar
    """
    pred   = _to_01(pred)
    target = _to_01(target)

    B, C, H, W = pred.shape
    kernel = _gaussian_kernel(window_size).to(pred.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    pad    = window_size // 2

    mu_p  = F.conv2d(pred,   kernel, padding=pad, groups=C)
    mu_t  = F.conv2d(target, kernel, padding=pad, groups=C)
    mu_p2 = mu_p  * mu_p
    mu_t2 = mu_t  * mu_t
    mu_pt = mu_p  * mu_t

    sigma_p2  = F.conv2d(pred   * pred,   kernel, padding=pad, groups=C) - mu_p2
    sigma_t2  = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu_t2
    sigma_pt  = F.conv2d(pred   * target, kernel, padding=pad, groups=C) - mu_pt

    numerator   = (2 * mu_pt + C1) * (2 * sigma_pt + C2)
    denominator = (mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2)

    ssim_map = numerator / denominator.clamp_min(1e-8)
    return ssim_map.mean()


# ──────────────────────────────────────────────────────────────────────────────
# 3. FID  (Fréchet distance on lightweight CNN features)
# ──────────────────────────────────────────────────────────────────────────────

class _FeatureExtractor(nn.Module):
    """
    Small ConvNet that maps (B, 1, H, W) → (B, 512) feature vectors.
    Shares the same architecture as the CNN branch of Style_Encoder so the
    feature space is meaningful for handwriting.
    """
    def __init__(self, feat_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            # 128×512 → 64×256
            nn.Conv2d(1, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.MaxPool2d(2),
            # 64×256 → 32×128
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.MaxPool2d(2),
            # 32×128 → 4×4 (adaptive)
            nn.Conv2d(128, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.SiLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(_to_01(x))


def _cov(x: torch.Tensor) -> torch.Tensor:
    """Unbiased covariance matrix for (N, D) feature matrix."""
    N, D = x.shape
    mean = x.mean(dim=0, keepdim=True)
    x_c  = x - mean
    return (x_c.T @ x_c) / (N - 1)


def _matrix_sqrt(A: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
    """
    Iterative matrix square root via Denman–Beavers (no torch.linalg.eigh needed).
    Works for positive semi-definite matrices.
    """
    n = A.shape[0]
    Y = A.clone()
    Z = torch.eye(n, dtype=A.dtype, device=A.device)
    for _ in range(num_iters):
        T  = 0.5 * (Y + torch.linalg.solve(Z.T, torch.eye(n, device=A.device, dtype=A.dtype)).T)
        Z  = 0.5 * (Z + torch.linalg.solve(Y.T, torch.eye(n, device=A.device, dtype=A.dtype)).T)
        Y  = T
    return Y


class FIDTracker:
    """
    Accumulates feature vectors for real and generated images,
    then computes the Fréchet distance on request.

    Usage::

        fid = FIDTracker(device=device)
        for real, fake in eval_batches:
            fid.update(real_images=real, fake_images=fake)
        score = fid.compute()   # lower = better
        fid.reset()
    """

    def __init__(self, feat_dim: int = 512, device: torch.device | str = "cpu"):
        self.extractor = _FeatureExtractor(feat_dim).to(device).eval()
        self.device    = device
        self._real: List[torch.Tensor] = []
        self._fake: List[torch.Tensor] = []

    @torch.no_grad()
    def update(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> None:
        self._real.append(self.extractor(real_images.to(self.device)).cpu())
        self._fake.append(self.extractor(fake_images.to(self.device)).cpu())

    def compute(self) -> float:
        if not self._real:
            raise RuntimeError("No data accumulated — call .update() first.")
        real = torch.cat(self._real, dim=0).double()   # (N, D)
        fake = torch.cat(self._fake, dim=0).double()

        mu_r, mu_f   = real.mean(0), fake.mean(0)
        cov_r, cov_f = _cov(real),   _cov(fake)

        diff      = mu_r - mu_f
        covmean   = _matrix_sqrt(cov_r @ cov_f)

        # FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2√(Σ_r Σ_f))
        trace_term = (cov_r + cov_f - 2.0 * covmean).diagonal().sum()
        fid_score  = (diff @ diff + trace_term).item()
        return max(fid_score, 0.0)         # numerical guard against tiny negatives

    def reset(self) -> None:
        self._real.clear()
        self._fake.clear()


# ──────────────────────────────────────────────────────────────────────────────
# 4. CER  (Character Error Rate — edit distance on predicted vs true text)
# ──────────────────────────────────────────────────────────────────────────────

def _edit_distance(s1: str, s2: str) -> int:
    """Standard dynamic-programming Levenshtein distance."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp  = dp[j]
            dp[j] = prev if s1[i - 1] == s2[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev  = temp
    return dp[n]


class CERTracker:
    """
    Accumulates (predicted_text, ground_truth_text) pairs and computes CER.

    CER = total_edit_distance / total_ground_truth_characters

    Usage::

        cer = CERTracker()
        cer.update(predictions=["helo", "wrold"], targets=["hello", "world"])
        print(cer.compute())   # 0.2
        cer.reset()
    """

    def __init__(self) -> None:
        self._total_edits: int = 0
        self._total_chars: int = 0

    def update(self, predictions: List[str], targets: List[str]) -> None:
        if len(predictions) != len(targets):
            raise ValueError("predictions and targets must have the same length.")
        for pred, tgt in zip(predictions, targets):
            self._total_edits += _edit_distance(pred, tgt)
            self._total_chars += len(tgt)

    def compute(self) -> float:
        if self._total_chars == 0:
            return 0.0
        return self._total_edits / self._total_chars

    def reset(self) -> None:
        self._total_edits = 0
        self._total_chars = 0


# ──────────────────────────────────────────────────────────────────────────────
# 5. MetricsTracker — convenience wrapper over all four metrics
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricsSummary:
    psnr: float = 0.0
    ssim: float = 0.0
    fid:  float = 0.0
    cer:  float = 0.0

    def __str__(self) -> str:
        return (
            f"PSNR={self.psnr:.2f} dB  |  "
            f"SSIM={self.ssim:.4f}  |  "
            f"FID={self.fid:.2f}  |  "
            f"CER={self.cer:.4f}"
        )


class MetricsTracker:
    """
    Unified tracker — accumulates over an entire eval loop then summarises.

    Usage (inside your eval loop)::

        tracker = MetricsTracker(device=device)

        for real_images, tokens, _ in val_loader:
            fake_images = generate(...)        # your sampling function
            pred_texts  = ocr_decode(...)      # optional — pass [] to skip CER

            tracker.update(
                real_images=real_images,
                fake_images=fake_images,
                pred_texts=pred_texts,          # list[str] or []
                target_texts=target_texts,      # list[str] or []
            )

        summary = tracker.compute()
        print(summary)
        tracker.reset()
    """

    def __init__(self, device: torch.device | str = "cpu", feat_dim: int = 512) -> None:
        self.device      = device
        self._fid        = FIDTracker(feat_dim=feat_dim, device=device)
        self._cer        = CERTracker()
        self._psnr_vals: List[float] = []
        self._ssim_vals: List[float] = []

    @torch.no_grad()
    def update(
        self,
        real_images:   torch.Tensor,
        fake_images:   torch.Tensor,
        pred_texts:    List[str] | None = None,
        target_texts:  List[str] | None = None,
    ) -> None:
        real_images = real_images.to(self.device)
        fake_images = fake_images.to(self.device)

        self._psnr_vals.append(psnr(fake_images, real_images).item())
        self._ssim_vals.append(ssim(fake_images, real_images).item())
        self._fid.update(real_images, fake_images)

        if pred_texts and target_texts:
            self._cer.update(pred_texts, target_texts)

    def compute(self) -> MetricsSummary:
        return MetricsSummary(
            psnr=sum(self._psnr_vals) / max(len(self._psnr_vals), 1),
            ssim=sum(self._ssim_vals) / max(len(self._ssim_vals), 1),
            fid=self._fid.compute(),
            cer=self._cer.compute(),
        )

    def reset(self) -> None:
        self._psnr_vals.clear()
        self._ssim_vals.clear()
        self._fid.reset()
        self._cer.reset()


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 4
    real = torch.randn(B, 1, 128, 512).to(device)
    fake = real + 0.1 * torch.randn_like(real)         # slightly noisy clone

    # Individual metrics
    print(f"PSNR : {psnr(fake, real).item():.2f} dB")
    print(f"SSIM : {ssim(fake, real).item():.4f}")

    # FID
    fid_tracker = FIDTracker(device=device)
    fid_tracker.update(real, fake)
    print(f"FID  : {fid_tracker.compute():.4f}")

    # CER
    cer_tracker = CERTracker()
    cer_tracker.update(["helo world", "writng"], ["hello world", "writing"])
    print(f"CER  : {cer_tracker.compute():.4f}")

    # All-in-one
    tracker = MetricsTracker(device=device)
    tracker.update(real, fake, ["helo"], ["hello"])
    summary = tracker.compute()
    print(f"\nSummary: {summary}")