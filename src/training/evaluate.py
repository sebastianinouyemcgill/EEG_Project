"""
Evaluation script — runs the best checkpoint on the test set and produces:
  1. Classification report (per-class precision / recall / F1)
  2. Confusion matrix (saved as PNG)
  3. Hypnogram plot for one test subject (predicted vs ground truth)

Usage:
  python -m src.training.evaluate --config configs/cnn.yaml
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix

log = logging.getLogger(__name__)
LABEL_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(5), yticks=range(5),
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        ylabel="True label", xlabel="Predicted label",
        title="Confusion Matrix — Test Set",
    )
    # Annotate cells
    thresh = cm.max() / 2
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Confusion matrix saved → {out_path}")


def plot_hypnogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    n_epochs: int = 900,   # ~7.5 hours
) -> None:
    """
    Plot predicted vs ground-truth sleep stages over a full night.
    Standard clinical convention: deeper sleep stages at the bottom.
    """
    stage_order = [0, 4, 1, 2, 3]   # Wake, REM, N1, N2, N3  (top → bottom)
    y_t = np.array([stage_order.index(s) for s in y_true[:n_epochs]])
    y_p = np.array([stage_order.index(s) for s in y_pred[:n_epochs]])
    t   = np.arange(len(y_t)) * 0.5  # epochs → hours (0.5 min each)

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    for ax, y, title in zip(axes, [y_t, y_p], ["Ground Truth", "Predicted"]):
        ax.step(t, y, where="post", lw=1.2, color="#2563eb")
        ax.set_yticks(range(5))
        ax.set_yticklabels(["Wake", "REM", "N1", "N2", "N3"])
        ax.set_title(title, fontsize=10, loc="left")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    axes[-1].set_xlabel("Time (minutes)")
    fig.suptitle("Hypnogram — Test Subject", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Hypnogram saved → {out_path}")


def evaluate(cfg):
    device = get_device()
    out_dir = Path(cfg["training"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "best_model.pt"

    model_name = cfg["model"].get("name", "SleepCNN")
    is_transformer = model_name == "SleepTransformer"

    if is_transformer:
        from src.data.sequence_dataset import make_sequence_dataloaders
        from src.models.transformer import SleepTransformer
        model = SleepTransformer(
            d_model=cfg["model"]["d_model"],
            n_heads=cfg["model"]["n_heads"],
            n_layers=cfg["model"]["n_layers"],
            d_ff=cfg["model"]["d_ff"],
            dropout=0.0,
            n_classes=cfg["model"]["n_classes"],
        ).to(device)
        loaders = make_sequence_dataloaders(
            cfg["data"]["processed_dir"],
            batch_size=cfg["training"]["batch_size"],
            seq_len=cfg["model"]["seq_len"],
            num_workers=cfg["training"].get("num_workers", 4),
        )
    else:
        from src.data.dataset import make_dataloaders
        from src.models.cnn import SleepCNN
        model = SleepCNN(
            n_classes=cfg["model"]["n_classes"],
            dropout=0.0,
        ).to(device)
        loaders = make_dataloaders(
            cfg["data"]["processed_dir"],
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["training"].get("num_workers", 4),
        )

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loaders["test"]:
            X = X.to(device)
            logits = model(X)
            if is_transformer:
                preds = logits.view(-1, logits.size(-1)).argmax(1).cpu()
                all_labels.extend(y.view(-1).tolist())
            else:
                preds = logits.argmax(1).cpu()
                all_labels.extend(y.tolist())
            all_preds.extend(preds.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0)
    print("\\n" + "=" * 50)
    print("Classification Report — Test Set")
    print("=" * 50)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, out_dir / "confusion_matrix.png")
    plot_hypnogram(y_true, y_pred, out_dir / "hypnogram.png")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cnn.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)
