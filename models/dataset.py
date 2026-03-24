from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.text_encoder import build_vocab, encode_text
from utils import GENERATOR_SIZE, resize_for_style_encoder

try:
    from datasets import Dataset as HFDataset
    from datasets import load_dataset
except ImportError:  # pragma: no cover - dependency availability is environment-specific
    HFDataset = None
    load_dataset = None


class IAM_Dataset(Dataset):
    """
    IAM handwriting line dataset with a paired style reference image.

    Each sample returns a dictionary with:
        image       : target handwriting line, shape (1, 128, 512)
        style_image : style reference image, shape (1, 64, 256)
        tokens      : tokenized text, shape (max_len,)
        text        : raw text string
        writer_id   : normalized integer writer id
    """

    def __init__(
        self,
        split: str = "train",
        max_len: int = 256,
        img_width: int = GENERATOR_SIZE[1],
        img_height: int = GENERATOR_SIZE[0],
        dataset_name: str = "Teklia/IAM-line",
        cache_dir: str | None = None,
        fallback_writer_group_size: int = 20,
    ) -> None:
        if load_dataset is None:
            raise ImportError(
                "The 'datasets' package is required for IAM_Dataset. Install it with `pip install datasets`."
            )

        resolved_split = self._resolve_split_name(split)
        self.data = self._load_split_dataset(
            dataset_name=dataset_name,
            split=resolved_split,
            cache_dir=cache_dir,
        )
        self.max_len = max_len
        if fallback_writer_group_size <= 0:
            raise ValueError("fallback_writer_group_size must be a positive integer")
        self.fallback_writer_group_size = fallback_writer_group_size
        self.char2idx, self.idx2char = build_vocab()
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_height, img_width)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        self.writer_ids = self._build_writer_ids()
        self.writer_to_indices = defaultdict(list)
        for index, writer_id in enumerate(self.writer_ids):
            self.writer_to_indices[writer_id].append(index)

        self.index_to_position = {}
        for writer_id, indices in self.writer_to_indices.items():
            for position, index in enumerate(indices):
                self.index_to_position[index] = position

    @staticmethod
    def _cached_arrow_path(dataset_name: str, split: str) -> Path | None:
        if dataset_name != "Teklia/IAM-line":
            return None

        cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "Teklia___iam-line"
        pattern = f"default/*/*/iam-line-{split}.arrow"
        matches = sorted(cache_root.glob(pattern))
        return matches[-1] if matches else None

    @classmethod
    def _load_split_dataset(
        cls,
        dataset_name: str,
        split: str,
        cache_dir: str | None,
    ):
        cached_arrow = cls._cached_arrow_path(dataset_name, split)
        if cache_dir is None and cached_arrow is not None and HFDataset is not None:
            return HFDataset.from_file(str(cached_arrow))

        try:
            return load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        except OSError as exc:
            if cached_arrow is None or HFDataset is None:
                raise
            if "Read-only file system" not in str(exc):
                raise
            return HFDataset.from_file(str(cached_arrow))

    @staticmethod
    def _resolve_split_name(split: str) -> str:
        aliases = {
            "val": "validation",
            "valid": "validation",
            "dev": "validation",
        }
        return aliases.get(split, split)

    def _infer_writer_key(self) -> str | None:
        if len(self.data) == 0:
            return None
        sample = self.data[0]
        for key in ("writer_id", "writer", "author", "writer_idx"):
            if key in sample:
                return key
        return None

    def _raw_writer_id(self, sample: dict[str, Any], index: int, writer_key: str | None) -> Any:
        if writer_key is not None:
            return sample[writer_key]

        sample_id = sample.get("id")
        if isinstance(sample_id, str) and sample_id:
            return sample_id.split("-")[0]

        # Teklia/IAM-line exposes only image and text, so we preserve the original
        # project behavior and approximate writers with local contiguous groups.
        return index // self.fallback_writer_group_size

    def _build_writer_ids(self) -> list[int]:
        writer_key = self._infer_writer_key()
        writer_lookup: dict[Any, int] = {}
        writer_ids: list[int] = []

        for index in range(len(self.data)):
            raw_writer_id = self._raw_writer_id(self.data[index], index, writer_key)
            if raw_writer_id not in writer_lookup:
                writer_lookup[raw_writer_id] = len(writer_lookup)
            writer_ids.append(writer_lookup[raw_writer_id])

        return writer_ids

    def _load_image(self, sample: dict[str, Any]) -> torch.Tensor:
        image = sample["image"].convert("L")
        return self.transform(image)

    def _select_style_reference(self, index: int, writer_id: int) -> int:
        candidates = self.writer_to_indices[writer_id]
        if len(candidates) == 1:
            return index

        position = self.index_to_position[index]
        return candidates[(position + 1) % len(candidates)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.data[index]
        writer_id = self.writer_ids[index]
        style_index = self._select_style_reference(index, writer_id)
        style_sample = self.data[style_index]

        image = self._load_image(sample)
        style_image = resize_for_style_encoder(self._load_image(style_sample))
        text = sample["text"]
        tokens = encode_text(text, self.char2idx, self.max_len)

        return {
            "image": image,
            "style_image": style_image.squeeze(0),
            "tokens": tokens,
            "text": text,
            "writer_id": writer_id,
        }


def get_dataloader(
    split: str = "train",
    batch_size: int = 16,
    shuffle: bool | None = None,
    max_len: int = 256,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    drop_last: bool | None = None,
    cache_dir: str | None = None,
    fallback_writer_group_size: int = 20,
) -> DataLoader:
    """
    Return a DataLoader for the IAM dataset.
    """
    dataset = IAM_Dataset(
        split=split,
        max_len=max_len,
        cache_dir=cache_dir,
        fallback_writer_group_size=fallback_writer_group_size,
    )
    is_train = IAM_Dataset._resolve_split_name(split) == "train"
    if shuffle is None:
        shuffle = is_train
    if drop_last is None:
        drop_last = is_train
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
