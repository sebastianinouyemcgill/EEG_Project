"""
Band-power baseline using sklearn — useful as a sanity check before CNN training.

A Random Forest on 10 spectral features should get ~70% accuracy.
If your CNN can't beat this, something is wrong.

Usage:
  python -m src.models.baseline
"""

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def run_baseline(processed_dir: str = "data/processed") -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    from src.features.spectral import extract_features, feature_names

    processed_dir = Path(processed_dir)
    files = sorted(processed_dir.glob("subject_*.npz"))
    if not files:
        raise FileNotFoundError("No processed files found. Run preprocess.py first.")

    n = len(files)
    n_tr = int(n * 0.7)
    n_val = int(n * 0.15)

    def load_features(flist):
        Xs, ys = [], []
        for f in flist:
            data = np.load(f)
            feats = extract_features(data["X"])
            # Concatenate absolute and relative band power
            X_feat = np.concatenate([feats.band_power, feats.rel_band_power], axis=1)
            Xs.append(X_feat)
            ys.append(data["y"])
        return np.concatenate(Xs), np.concatenate(ys)

    log.info("Loading features ...")
    X_tr, y_tr = load_features(files[:n_tr])
    X_te, y_te = load_features(files[n_tr + int(n * 0.15):])

    log.info(f"Train: {X_tr.shape}, Test: {X_te.shape}")
    log.info("Training Random Forest ...")
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    print(classification_report(y_te, y_pred, target_names=["Wake", "N1", "N2", "N3", "REM"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_baseline()
