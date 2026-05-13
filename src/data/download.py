"""
Download Sleep-EDF Cassette recordings from PhysioNet via braindecode.
Usage: python -m src.data.download --n_subjects 20 --data_dir data/raw
"""

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def download_sleep_edf(data_dir: str | Path, n_subjects: int = 20) -> None:
    """
    Download n_subjects from the Sleep-EDF Cassette dataset.
    braindecode handles the PhysioNet API and caching automatically.
    """
    try:
        from braindecode.datasets import SleepPhysionet
    except ImportError:
        raise ImportError("Run: pip install braindecode")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = list(range(n_subjects))
    log.info(f"Downloading {n_subjects} subjects to {data_dir} ...")

    # braindecode downloads to a cache dir; we record what was fetched
    dataset = SleepPhysionet(
        subject_ids=subject_ids,
        recording_ids=[1],          # night 1 only — keeps things balanced
        crop_wake_mins=30,           # trim excessive wake at start/end
        path=str(data_dir),
    )

    log.info(f"Downloaded {len(dataset.datasets)} recordings.")
    return dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_subjects", type=int, default=20)
    parser.add_argument("--data_dir", type=str, default="data/raw")
    args = parser.parse_args()
    download_sleep_edf(args.data_dir, args.n_subjects)
