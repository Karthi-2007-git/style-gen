import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = [
	"build_vocab",
	"encode_text",
	"Text_Encoder",
	"Learned_Positional_Encoding",
	"Sinusoidal_Positional_Encoding",
	"Transformer_Text_Encoder",
	"count_params",
]

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


def build_vocab(
	special_tokens: Sequence[str] = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN),
	printable_start: int = 32,
	printable_end: int = 126,
) -> Tuple[Dict[str, int], Dict[int, str]]:
	"""
	Builds a character -> integer lookup table.
	Returns:
		char2idx : dict — maps character to integer index
		idx2char : dict — maps integer index to character
	"""
	if printable_start < 0 or printable_end <= printable_start:
		raise ValueError("Invalid printable range")
	if len(set(special_tokens)) != len(special_tokens):
		raise ValueError("special_tokens must be unique")
	charset = tuple(chr(i) for i in range(printable_start, printable_end))
	char2idx = {ch: idx for idx, ch in enumerate(special_tokens)}
	for i, ch in enumerate(charset):
		char2idx[ch] = i + len(special_tokens)
	idx2char = {idx: ch for ch, idx in char2idx.items()}
	return char2idx, idx2char


def encode_text(text: str, char2idx: Dict[str, int], max_len: int = 256) -> torch.LongTensor:
	"""
	Converts a text string into a padded integer tensor.
	Args:
		text     (str)  : input string
		char2idx (dict) : character to index lookup table
		max_len  (int)  : fixed sequence length including BOS and EOS
	Returns:
		tensor (torch.LongTensor): shape (max_len,)
	"""
	if max_len < 2:
		raise ValueError("max_len must be at least 2 to fit <BOS> and <EOS>")
	required = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
	missing = [token for token in required if token not in char2idx]
	if missing:
		raise KeyError(f"char2idx is missing required tokens: {missing}")
	tokens = [char2idx.get(c, char2idx[UNK_TOKEN]) for c in text]
	tokens = tokens[:max_len - 2]
	tokens = [char2idx[BOS_TOKEN]] + tokens + [char2idx[EOS_TOKEN]]
	pad_len = max_len - len(tokens)
	tokens = tokens + [char2idx[PAD_TOKEN]] * pad_len
	return torch.tensor(tokens, dtype=torch.long)


class Text_Encoder(nn.Module):
	"""
	Encodes a tokenized text sequence into a sequence of embeddings.

	Args:
		vocab_size (int): number of characters in the vocabulary
		embed_dim  (int): dimensionality of the output embeddings
		pad_idx   (int): index used for padding tokens (default: 0)
	Returns:
		tensor (torch.FloatTensor): shape (B, embed_dim)
	"""

	def __init__(self, vocab_size: int, embed_dim: int = 256, pad_idx: int = 0):
		super().__init__()
		if vocab_size <= 0:
			raise ValueError("vocab_size must be a positive integer")
		if embed_dim <= 0:
			raise ValueError("embed_dim must be a positive integer")
		if pad_idx < 0 or pad_idx >= vocab_size:
			raise ValueError("pad_idx must be in [0, vocab_size)")
		self.embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=embed_dim,
			padding_idx=pad_idx,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 2:
			raise ValueError(f"Expected token tensor of shape (B, T), got {tuple(x.shape)}")
		if x.dtype != torch.long:
			x = x.long()
		# token_ids shape: (B, max_len)
		token_ids = x
		# Embed each token
		embedded = self.embedding(token_ids)  # (B, max_len, embed_dim)

		# Build mask to ignore padding tokens
		mask = (token_ids != self.embedding.padding_idx)  # (B, max_len)

		# zero out embeddings at padding positions
		mask_expanded = mask.unsqueeze(-1).float()
		masked = embedded * mask_expanded

		# Average of real token embeddings (ignore padding)
		counts = mask.sum(dim=1, keepdim=True).float().clamp_min(1.0)  # (B, 1)
		average = masked.sum(dim=1) / counts  # (B, embed_dim)

		return average


class Learned_Positional_Encoding(nn.Module):
	"""
	Learns a vector for each position during training.
	Args:
		max_len   (int) : maximum sequence length
		embed_dim (int) : embedding dimensionality
	"""

	def __init__(self, max_len: int = 256, embed_dim: int = 256):
		super().__init__()
		if max_len <= 0:
			raise ValueError("max_len must be a positive integer")
		if embed_dim <= 0:
			raise ValueError("embed_dim must be a positive integer")
		self.pos_embedding = nn.Embedding(max_len, embed_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		B, T, _ = x.shape
		if T > self.pos_embedding.num_embeddings:
			raise ValueError(
				f"Sequence length {T} exceeds max_len {self.pos_embedding.num_embeddings}"
			)
		positions = torch.arange(T, device=x.device)
		pos_emb = self.pos_embedding(positions)
		return x + pos_emb


class Sinusoidal_Positional_Encoding(nn.Module):
	"""
	Fixed sin/cos positional encoding. No learnable parameters.
	Args:
		max_len   (int) : maximum sequence length
		embed_dim (int) : embedding dimensionality
	"""

	def __init__(self, max_len: int = 256, embed_dim: int = 256):
		super().__init__()
		if max_len <= 0:
			raise ValueError("max_len must be a positive integer")
		if embed_dim <= 0:
			raise ValueError("embed_dim must be a positive integer")
		pe = torch.zeros(max_len, embed_dim)
		position = torch.arange(0, max_len).unsqueeze(1).float()
		div_term = torch.exp(
			torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
		)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		self.register_buffer("pe", pe.unsqueeze(0))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.size(1) > self.pe.size(1):
			raise ValueError(f"Sequence length {x.size(1)} exceeds max_len {self.pe.size(1)}")
		return x + self.pe[:, :x.size(1), :]


class Transformer_Text_Encoder(nn.Module):
	"""
	Full text encoder: embed → positional encoding → transformer → masked average.
	Args:
		vocab_size   (int) : number of characters in vocabulary
		embed_dim    (int) : embedding dimensionality
		num_heads    (int) : number of attention heads
		ff_dim       (int) : feedforward layer size inside transformer
		num_layers   (int) : number of stacked transformer layers
		max_len      (int) : maximum sequence length
		pad_idx      (int) : padding token index
		pos_encoding (str) : learned or sinusoidal
	Returns:
		tensor (torch.FloatTensor): shape (B, embed_dim)
	"""

	def __init__(self, vocab_size: int, embed_dim: int = 256, num_heads: int = 8,
				 ff_dim: int = 512, num_layers: int = 2, max_len: int = 256,
				 pad_idx: int = 0, pos_encoding: str = "sinusoidal"):
		super().__init__()
		if vocab_size <= 0:
			raise ValueError("vocab_size must be a positive integer")
		if embed_dim <= 0:
			raise ValueError("embed_dim must be a positive integer")
		if num_heads <= 0 or embed_dim % num_heads != 0:
			raise ValueError("num_heads must be positive and divide embed_dim")
		if ff_dim <= 0 or num_layers <= 0:
			raise ValueError("ff_dim and num_layers must be positive integers")
		if max_len <= 0:
			raise ValueError("max_len must be a positive integer")
		if pad_idx < 0 or pad_idx >= vocab_size:
			raise ValueError("pad_idx must be in [0, vocab_size)")
		self.pad_idx = pad_idx
		self.embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=embed_dim,
			padding_idx=pad_idx
		)
		if pos_encoding == "learned":
			self.pos_enc = Learned_Positional_Encoding(max_len=max_len, embed_dim=embed_dim)
		elif pos_encoding == "sinusoidal":
			self.pos_enc = Sinusoidal_Positional_Encoding(max_len=max_len, embed_dim=embed_dim)
		else:
			raise ValueError("pos_encoding must be 'learned' or 'sinusoidal'")
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=embed_dim,
			nhead=num_heads,
			dim_feedforward=ff_dim,
			batch_first=True,
			dropout=0.1,
			activation="relu"
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

	def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
		if token_ids.ndim != 2:
			raise ValueError(f"Expected token_ids shape (B, T), got {tuple(token_ids.shape)}")
		if token_ids.dtype != torch.long:
			token_ids = token_ids.long()
		x = self.embedding(token_ids)
		x = self.pos_enc(x)
		pad_mask = (token_ids == self.pad_idx)
		x = self.transformer(x, src_key_padding_mask=pad_mask)
		mask = (token_ids != self.pad_idx).unsqueeze(-1).float()
		x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
		return x


def count_params(model: nn.Module, print_summary: bool = True) -> Tuple[int, int]:
	total = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	if print_summary:
		print(f"Total params:     {total:,}")
		print(f"Trainable params: {trainable:,}")
	return total, trainable
