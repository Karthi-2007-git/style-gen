import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from models.text_encoder import build_vocab, encode_text


class IAM_Dataset(Dataset):
    """
    PyTorch Dataset for IAM handwriting lines.

    Each sample returns:
        image     (torch.FloatTensor) : shape (1, 128, 512)
        tokens    (torch.LongTensor)  : shape (max_len,)
        writer_id (int)               : derived from sample index
    """
    def __init__(self, split="train", max_len=256, img_width=512, img_height=128):
        self.data = load_dataset("Teklia/IAM-line")[split]
        self.max_len = max_len
        self.char2idx, self.idx2char = build_vocab()
        self.transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample["image"].convert("L")
        image = self.transform(image)
        tokens = encode_text(sample["text"], self.char2idx, self.max_len)
        writer_id = idx // 20
        return image, tokens, writer_id


def get_dataloader(split="train", batch_size=16, shuffle=True):
    """
    Returns a DataLoader for the IAM dataset.

    Args:
        split      (str)  : train, validation, or test
        batch_size (int)  : number of samples per batch
        shuffle    (bool) : shuffle the dataset
    Returns:
        DataLoader
    """
    dataset = IAM_Dataset(split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
