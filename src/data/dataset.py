"""
PyTorch Dataset for preprocessed Sleep-EDF epochs.

Splits are done by SUBJECT (not by epoch) to prevent data leakage.
Each subject's data stays entirely in one split.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


SplitType = Literal["train", "val", "test"]


class SleepEDFDataset(Dataset):
    """
    Loads preprocessed .npz files (one per subject) and exposes individual
    30-second epochs as (X, y) pairs.

    Args:
        processed_dir:  path to data/processed/
        split:          'train' | 'val' | 'test'
        train_frac:     fraction of subjects used for training
        val_frac:       fraction used for validation (rest → test)
        seed:           random seed for subject shuffle
    """

    LABEL_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

    def __init__(
        self,
        processed_dir: str | Path,
        split: SplitType = "train",
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
    ) -> None:
        super().__init__()
        processed_dir = Path(processed_dir)
        all_files = sorted(processed_dir.glob("subject_*.npz"))

        if not all_files:
            raise FileNotFoundError(
                f"No subject .npz files found in {processed_dir}. "
                "Run preprocess.py first."
            )

        # Shuffle subjects deterministically, then split
        rng   = np.random.default_rng(seed)
        files = rng.permutation(all_files).tolist()
        n     = len(files)
        n_tr  = int(n * train_frac)
        n_val = int(n * val_frac)

        split_files: dict[SplitType, list[Path]] = {
            "train": files[:n_tr],
            "val":   files[n_tr : n_tr + n_val],
            "test":  files[n_tr + n_val :],
        }

        # Load all epochs for this split into memory
        # (Sleep-EDF is small enough — ~20 subjects ≈ 150 MB)
        xs, ys = [], []
        for f in split_files[split]:
            data = np.load(f)
            xs.append(data["X"])
            ys.append(data["y"])

        if not xs:
            raise ValueError(f"No files assigned to '{split}' split.")

        X_all = np.concatenate(xs, axis=0)  # (N_total, 3000)
        y_all = np.concatenate(ys, axis=0)  # (N_total,)

        # Add channel dim for Conv1d: (N, 1, 3000)
        self.X = torch.from_numpy(X_all).unsqueeze(1)
        self.y = torch.from_numpy(y_all)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency weights for CrossEntropyLoss to handle class imbalance.
        N2 dominates; N1 is rare — this balances the loss.
        """
        counts = torch.bincount(self.y, minlength=5).float()
        weights = 1.0 / (counts + 1e-8)
        return weights / weights.sum() * len(counts)   # normalise


def make_dataloaders(
    processed_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    **dataset_kwargs,
) -> dict[SplitType, DataLoader]:
    loaders = {}
    for split in ("train", "val", "test"):
        ds = SleepEDFDataset(processed_dir, split=split, **dataset_kwargs)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders
