"""
Sequence dataset for the Transformer stage.

Instead of returning single epochs, returns a sequence of N consecutive
epochs so the Transformer can learn temporal context (sleep cycles).

Each item: (X, y) where
  X: (seq_len, 1, 3000)  — sequence of raw EEG epochs
  y: (seq_len,)           — label for each epoch in the sequence
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SplitType = Literal["train", "val", "test"]


class SleepSequenceDataset(Dataset):
    """
    Wraps preprocessed .npz files and returns sliding-window sequences.

    Args:
        processed_dir:  path to data/processed/
        split:          'train' | 'val' | 'test'
        seq_len:        number of consecutive epochs per sequence (default 20 = 10 min)
        stride:         step between sequence start points (default 1 = fully overlapping)
        train_frac:     fraction of subjects for training
        val_frac:       fraction of subjects for validation
        seed:           random seed for subject shuffle
    """

    LABEL_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

    def __init__(
        self,
        processed_dir: str | Path,
        split: SplitType = "train",
        seq_len: int = 20,
        stride: int = 1,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        processed_dir = Path(processed_dir)
        all_files = sorted(processed_dir.glob("subject_*.npz"))
        if not all_files:
            raise FileNotFoundError(f"No subject .npz files in {processed_dir}")

        rng   = np.random.default_rng(seed)
        files = rng.permutation(all_files).tolist()
        n     = len(files)
        n_tr  = int(n * train_frac)
        n_val = int(n * val_frac)

        split_files = {
            "train": files[:n_tr],
            "val":   files[n_tr : n_tr + n_val],
            "test":  files[n_tr + n_val :],
        }[split]

        # Build index: list of (subject_X, subject_y, start_idx)
        # Each entry represents one sequence window
        self.sequences: list[tuple[np.ndarray, np.ndarray, int]] = []

        for f in split_files:
            data = np.load(f)
            X = data["X"]  # (N_epochs, 3000)
            y = data["y"]  # (N_epochs,)

            # Slide a window of seq_len across this subject's epochs
            for start in range(0, len(y) - seq_len + 1, stride):
                self.sequences.append((X, y, start))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        X, y, start = self.sequences[idx]
        x_seq = X[start : start + self.seq_len]          # (seq_len, 3000)
        y_seq = y[start : start + self.seq_len]          # (seq_len,)

        # Add channel dim: (seq_len, 1, 3000)
        x_tensor = torch.from_numpy(x_seq).unsqueeze(1)
        y_tensor = torch.from_numpy(y_seq)
        return x_tensor, y_tensor

    def class_weights(self) -> torch.Tensor:
        all_labels = torch.cat([
            torch.from_numpy(y[start : start + self.seq_len])
            for _, y, start in self.sequences
        ])
        counts  = torch.bincount(all_labels, minlength=5).float()
        weights = 1.0 / (counts + 1e-8)
        return weights / weights.sum() * 5


def make_sequence_dataloaders(
    processed_dir: str | Path,
    batch_size: int = 32,
    seq_len: int = 20,
    stride: int = 1,
    num_workers: int = 4,
    **dataset_kwargs,
) -> dict[SplitType, DataLoader]:
    loaders = {}
    for split in ("train", "val", "test"):
        ds = SleepSequenceDataset(
            processed_dir,
            split=split,
            seq_len=seq_len,
            stride=stride,
            **dataset_kwargs,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=False,
        )
    return loaders
