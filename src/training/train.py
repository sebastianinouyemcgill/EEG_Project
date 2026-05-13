"""
Training loop for SleepCNN (and later, the Transformer).

Features:
  - Class-weighted CrossEntropyLoss (handles N1/N3 imbalance)
  - Early stopping on validation macro-F1
  - MLflow experiment tracking (metrics, params, model checkpoint)
  - Config loaded from configs/*.yaml

Usage:
  python -m src.training.train --config configs/cnn.yaml
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    import mlflow
    from sklearn.metrics import f1_score

    from src.data.dataset import make_dataloaders
    from src.models.cnn import SleepCNN

    device = get_device()
    log.info(f"Using device: {device}")

    # --- Data ---
    loaders = make_dataloaders(
        cfg["data"]["processed_dir"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"].get("num_workers", 4),
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]

    # Class weights from training set
    class_weights = train_loader.dataset.class_weights().to(device)

    # --- Model ---
    model = SleepCNN(
        n_classes=cfg["model"]["n_classes"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    # --- Loss / optimiser / scheduler ---
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(
        optimiser,
        T_max=cfg["training"]["epochs"],
    )

    # --- MLflow ---
    mlflow.set_experiment(cfg.get("experiment_name", "sleep-eeg"))
    with mlflow.start_run(run_name=cfg.get("run_name", "cnn-baseline")):
        mlflow.log_params({
            "model":       "SleepCNN",
            "lr":          cfg["training"]["lr"],
            "batch_size":  cfg["training"]["batch_size"],
            "epochs":      cfg["training"]["epochs"],
            "dropout":     cfg["model"]["dropout"],
        })

        best_f1     = 0.0
        patience    = cfg["training"].get("patience", 10)
        no_improve  = 0
        ckpt_path   = Path(cfg["training"]["checkpoint_dir"]) / "best_model.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            # --- Train ---
            model.train()
            total_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimiser.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimiser.step()
                total_loss += loss.item()

            scheduler.step()
            avg_train_loss = total_loss / len(train_loader)

            # --- Validate ---
            model.eval()
            all_preds, all_labels = [], []
            val_loss = 0.0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    logits = model(X)
                    val_loss += criterion(logits, y).item()
                    all_preds.extend(logits.argmax(1).cpu().tolist())
                    all_labels.extend(y.cpu().tolist())

            avg_val_loss = val_loss / len(val_loader)
            macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss":   avg_val_loss,
                "val_macro_f1": macro_f1,
            }, step=epoch)

            log.info(
                f"Epoch {epoch:03d} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={avg_val_loss:.4f} | "
                f"macro_f1={macro_f1:.4f}"
            )

            # --- Early stopping ---
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
                mlflow.log_artifact(str(ckpt_path))
                log.info(f"  ✓ New best F1={best_f1:.4f} — checkpoint saved.")
            else:
                no_improve += 1
                if no_improve >= patience:
                    log.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                    break

        mlflow.log_metric("best_val_macro_f1", best_f1)
        log.info(f"Training complete. Best macro F1: {best_f1:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cnn.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
