from typing import Tuple

import torch
import torch.nn as nn

__all__ = [
	"CNN_Style_Encoder",
	"Patch_Embedding",
	"VIT_Embeddings",
	"Transformer_Block",
	"VIT_Style_Encoder",
	"Style_Encoder",
]


class CNN_Style_Encoder(nn.Module):
	"""
	Scans handwriting with filters to extract stroke features
	Input: (B,1,64,256) grayscale image of a word
	Output: (B, 256) one feature vector per image
	"""

	def __init__(self, in_channels: int = 1, embed_dim: int = 256):
		super().__init__()
		if in_channels <= 0:
			raise ValueError("in_channels must be a positive integer")
		if embed_dim <= 0:
			raise ValueError("embed_dim must be a positive integer")

		#Learn the edges and structures of the handwriting
		# 64x256 -> 32x128
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
			nn.GroupNorm(8, 64),
			nn.SiLU(),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.GroupNorm(8, 64),
			nn.SiLU(),
			nn.MaxPool2d(2),
		)

		#Learn letters shapes
		# 32x128 -> 16x64
		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.GroupNorm(8, 128),
			nn.SiLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.GroupNorm(8, 128),
			nn.SiLU(),
			nn.MaxPool2d(2),
		)

		#Learn writing stlyle and patterns
		#16x64 -> 4x16
		self.conv3 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.GroupNorm(8, 256),
			nn.SiLU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.GroupNorm(8, 256),
			nn.SiLU(),
			nn.AdaptiveAvgPool2d(4),
		)
		self.flatten = nn.Sequential(
			nn.Flatten(),
			nn.Linear(256 * 4 * 4, embed_dim),
			nn.LayerNorm(embed_dim),
		)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		return x


class Patch_Embedding(nn.Module):
	"""
	Seperates images into patches and converts to it to an embedding vector
	Input : (B,1,64,256)
	Output : (B,256, 256)
	"""

	def __init__(self, patch_size: int = 8, embed_dim: int = 256, in_channels: int = 1):
		super().__init__()
		if patch_size <= 0:
			raise ValueError("patch_size must be a positive integer")
		if embed_dim <= 0:
			raise ValueError("embed_dim must be a positive integer")
		if in_channels <= 0:
			raise ValueError("in_channels must be a positive integer")
		self.patch_size = patch_size
		self.in_channels = in_channels

		# Each patch is 8x8 pix x1 channel = 64 numbers
		# Linear projects 64 numbers -> 256 numbers (rich embedding)
		self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# cuts images into 8x8 patches
		if x.ndim != 4:
			raise ValueError(f"Expected input tensor of shape (B, C, H, W), got {tuple(x.shape)}")
		B, C, H, W = x.shape
		if C != self.in_channels:
			raise ValueError(f"Expected {self.in_channels} input channels, got {C}")
		p = self.patch_size
		if H % p != 0 or W % p != 0:
			raise ValueError(f"Input size {(H, W)} must be divisible by patch_size {p}")
		# Reshape it into patch grid
		x = x.reshape(B, C, H//p, p, W//p, p)  # (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
		# Reorder Dimensions to get patches as a sequence
		x = x.permute(0, 2, 4, 3, 5, 1)
		# flatten patch pixels
		x = x.reshape(B, (H // p) * (W // p), p * p * C) # (B, num_patches, patch_size* patch_size * C)
		return self.proj(x) # -> (B, num_patches, embed_dim)



class VIT_Embeddings(nn.Module):
	"""
	Adds CLS token and position info to patch embeddings
	Input: (B,256,256) patch embeddings
	Output: (B, 257, 256) patch embeddings + CLS token, with positions
	"""

	def __init__(self, num_patches: int = 256, embed_dim: int = 256):
		super().__init__()
		if num_patches <= 0:
			raise ValueError("num_patches must be a positive integer")
		if embed_dim <= 0:
			raise ValueError("embed_dim must be a positive integer")
		# CLS token - a learnable summary vector
		# starts random and learns to represent the whole image
		self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

		# Position embeddings - learnable vectors for each patch position
		# helps model understand the order of patches
		# 256 patches + 1 CLS token = 257 total tokens
		self.pos_embed = nn.Parameter(
			torch.randn(1, num_patches + 1, embed_dim) * 0.02
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		B = x.shape[0]

		# Expand CLS token for whole size
		cls = self.cls_token.expand(B, -1, -1)    # (1,1,256) → (B,1,256)

		# Prepare CLS to patch sequence
		x = torch.cat([cls, x], dim=1)           # (B,256,256) → (B,257,256)

		# Add position info to every token
		if x.size(1) != self.pos_embed.size(1):
			raise ValueError(
				f"Token count mismatch: got {x.size(1)} tokens, expected {self.pos_embed.size(1)}"
			)
		x = x + self.pos_embed                    # (B,257,256)
		return x


class Transformer_Block(nn.Module):
	"""
	One transformer block - patches are attended to each other and learn relationships
	Input: (B, 257, 256) patch embeddings + CLS token, with positions
	Output: (B, 257, 256) same shape but with learned relationships
	"""

	def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
		super().__init__()
		if embed_dim <= 0:
			raise ValueError("embed_dim must be a positive integer")
		if num_heads <= 0:
			raise ValueError("num_heads must be a positive integer")
		if embed_dim % num_heads != 0:
			raise ValueError("embed_dim must be divisible by num_heads")
		if not 0.0 <= dropout < 1.0:
			raise ValueError("dropout must be in [0, 1)")
		# Normalization before attention
		self.norm1 = nn.LayerNorm(embed_dim)

		# Multi-head self attention - learns relationships between patches
		# num_heads = 8 we have 8 parallel attention layers to capture different relationships
		self.attn = nn.MultiheadAttention(
			embed_dim,
			num_heads,
			dropout=dropout,
			batch_first=True,
		)
		# Normalization before MLP
		self.norm2 = nn.LayerNorm(embed_dim)

		# MLP - tokens processed independently to learn complex features
		self.mlp = nn.Sequential(
			nn.Linear(embed_dim, embed_dim * 4),  # Expand to higher dimension
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(embed_dim * 4, embed_dim),
			nn.Dropout(dropout),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Attention with residual connection
		x_norm = self.norm1(x)
		attn_out, _ = self.attn(x_norm, x_norm, x_norm)
		x = x + attn_out

		# MLP with residual connection
		x_norm = self.norm2(x)
		attn_out = self.mlp(x_norm)
		x = x + attn_out
		return x


class VIT_Style_Encoder(nn.Module):
	"""
	Full Vision Transformer for handwriting style extraction.
	Understanding global letter flow and connections.
	Input: (B,1,64,256) - greyscale image
	Output: (B, 256) - style vector from CLS token
	"""

	def __init__(
		self,
		patch_size: int = 8,
		embed_dim: int = 256,
		depth: int = 6,
		num_heads: int = 8,
		dropout: float = 0.1,
		image_size: Tuple[int, int] = (64, 256),
		in_channels: int = 1,
	):
		super().__init__()
		if depth <= 0:
			raise ValueError("depth must be a positive integer")
		height, width = image_size
		if height <= 0 or width <= 0:
			raise ValueError("image_size must contain positive values")
		if height % patch_size != 0 or width % patch_size != 0:
			raise ValueError("image_size must be divisible by patch_size")
		num_patches = (height // patch_size) * (width // patch_size)

		# cut images to patches and embed them
		self.patch_embed = Patch_Embedding(
			patch_size=patch_size,
			embed_dim=embed_dim,
			in_channels=in_channels,
		)

		# add CLS token and position info
		self.vit_embed = VIT_Embeddings(num_patches=num_patches, embed_dim=embed_dim)

		# stack of transformer blocks to learn relationships between patches
		self.blocks = nn.Sequential(
			*[
				Transformer_Block(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
				for _ in range(depth)
			]
		)

		# Normalize final output
		self.norm = nn.LayerNorm(embed_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# patches and embed
		x = self.patch_embed(x)  # (B, 256, 256)

		# Add CLS token and position
		x = self.vit_embed(x)  # (B, 257, 256)

		# Run through transformer blocks
		x = self.blocks(x)  # (B, 257, 256)

		# Normalize
		x = self.norm(x)  # (B, 257, 256)

		# Return CLS token as style vector
		return x[:, 0, :]  # (B, 256)


class Style_Encoder(nn.Module):
	"""
	Full style extractor combines CNN local features and VIT global features
	Input: (B, 1, 64, 256) - greyscale handwriting word
	Output: (B, 512) - final style vector
	"""

	def __init__(
		self,
		cnn_embed_dim: int = 256,
		vit_embed_dim: int = 256,
		output_dim: int = 512,
		image_size: Tuple[int, int] = (64, 256),
		in_channels: int = 1,
	):
		super().__init__()
		if output_dim <= 0:
			raise ValueError("output_dim must be a positive integer")

		# CNN extracts local stroke features
		self.cnn = CNN_Style_Encoder(in_channels=in_channels, embed_dim=cnn_embed_dim)

		# ViT captures global style and letter flow
		self.vit = VIT_Style_Encoder(
			embed_dim=vit_embed_dim,
			image_size=image_size,
			in_channels=in_channels,
		)

		# Fusion layer to combine CNN and ViT features
		fusion_in = cnn_embed_dim + vit_embed_dim
		self.fusion = nn.Sequential(
			nn.Linear(fusion_in, output_dim),
			nn.LayerNorm(output_dim),
			nn.SiLU(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		cnn_features = self.cnn(x)
		vit_features = self.vit(x)

		combined = torch.cat([cnn_features, vit_features], dim=1)  # (B, 512)
		style = self.fusion(combined)

		return style
