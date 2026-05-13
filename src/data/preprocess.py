"""
Preprocessing pipeline: raw EDF recordings → clean .npy epoch arrays.

Steps:
  1. Bandpass filter (0.5–35 Hz)
  2. Slice into 30-second epochs
  3. Z-score normalise per recording
  4. Map AASM annotations → integer labels
  5. Save per-subject arrays to data/processed/

Usage: python -m src.data.preprocess --raw_dir data/raw --out_dir data/processed
"""

import argparse
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# AASM stage → integer label
LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,   # N3 and N4 merged → deep sleep
    "Sleep stage R": 4,
}
LABEL_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

SFREQ      = 100          # resample target (Hz)
EPOCH_SEC  = 30           # standard PSG epoch
EPOCH_SAMP = SFREQ * EPOCH_SEC   # 3000 samples
FILT_LOW   = 0.5
FILT_HIGH  = 35.0
EEG_CHAN   = "EEG Fpz-Cz"  # primary channel used in Sleep-EDF


def preprocess_dataset(raw_dir: str | Path, out_dir: str | Path) -> None:
    try:
        import mne
        from braindecode.datasets import SleepPhysionet
    except ImportError:
        raise ImportError("Run: pip install mne braindecode")

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all PSG files in raw_dir
    psg_files = sorted(raw_dir.rglob("*PSG.edf"))
    ann_files = sorted(raw_dir.rglob("*Hypnogram.edf"))

    if not psg_files:
        raise FileNotFoundError(
            f"No PSG files found in {raw_dir}. Run download.py first."
        )

    log.info(f"Found {len(psg_files)} recordings.")

    for subj_idx, (psg_path, ann_path) in enumerate(zip(psg_files, ann_files)):
        out_path = out_dir / f"subject_{subj_idx:03d}.npz"
        if out_path.exists():
            log.info(f"  [skip] {out_path.name} already exists.")
            continue

        log.info(f"  Processing subject {subj_idx}: {psg_path.name}")

        # --- Load raw EEG ---
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        raw.pick_channels([EEG_CHAN])
        raw.resample(SFREQ, verbose=False)
        raw.filter(FILT_LOW, FILT_HIGH, verbose=False)

        # --- Load annotations ---
        ann = mne.read_annotations(ann_path)
        raw.set_annotations(ann, emit_warning=False)
        events, event_id = mne.events_from_annotations(
            raw, event_id=LABEL_MAP, verbose=False
        )

        # --- Epoch ---
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=0.0,
            tmax=EPOCH_SEC - 1.0 / SFREQ,
            baseline=None,
            preload=True,
            verbose=False,
        )

        X = epochs.get_data(units="uV").squeeze(1).astype(np.float32)  # (N, 3000)
        y = epochs.events[:, -1].astype(np.int64)                       # (N,)

        # --- Per-recording z-score normalisation ---
        mean = X.mean(axis=(0, 1), keepdims=True)
        std  = X.std(axis=(0, 1), keepdims=True) + 1e-8
        X    = (X - mean) / std

        np.savez_compressed(out_path, X=X, y=y)
        log.info(f"    Saved {X.shape[0]} epochs → {out_path.name}")

    log.info("Preprocessing complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()
    preprocess_dataset(args.raw_dir, args.out_dir)
