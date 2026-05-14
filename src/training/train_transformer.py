"""
Training loop for SleepTransformer.

Usage:
  python -m src.training.train_transformer --config configs/transformer.yaml
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


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(cfg: dict) -> None:
    import mlflow
    from sklearn.metrics import f1_score

    from src.data.sequence_dataset import make_sequence_dataloaders
    from src.models.transformer import SleepTransformer

    device = get_device()
    log.info(f"Using device: {device}")

    # Data
    loaders = make_sequence_dataloaders(
        cfg["data"]["processed_dir"],
        batch_size=cfg["training"]["batch_size"],
        seq_len=cfg["model"]["seq_len"],
        stride=cfg["model"].get("stride", 1),
        num_workers=cfg["training"].get("num_workers", 4),
    )

    class_weights = loaders["train"].dataset.class_weights().to(device)

    # Model
    model = SleepTransformer(
        cnn_checkpoint=cfg["model"].get("cnn_checkpoint"),
        freeze_cnn=cfg["model"].get("freeze_cnn", False),
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        d_ff=cfg["model"]["d_ff"],
        dropout=cfg["model"]["dropout"],
        n_classes=cfg["model"]["n_classes"],
    ).to(device)

    # Loss / optimiser
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=cfg["training"]["epochs"])

    # MLflow
    mlflow.set_experiment(cfg.get("experiment_name", "sleep-eeg"))
    with mlflow.start_run(run_name=cfg.get("run_name", "transformer")):
        mlflow.log_params({
            "model":      "SleepTransformer",
            "seq_len":    cfg["model"]["seq_len"],
            "n_layers":   cfg["model"]["n_layers"],
            "n_heads":    cfg["model"]["n_heads"],
            "lr":         cfg["training"]["lr"],
            "freeze_cnn": cfg["model"].get("freeze_cnn", False),
        })

        best_f1    = 0.0
        patience   = cfg["training"].get("patience", 10)
        no_improve = 0
        ckpt_path  = Path(cfg["training"]["checkpoint_dir"]) / "best_model.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            # Train
            model.train()
            total_loss = 0.0
            for X, y in loaders["train"]:
                X, y = X.to(device), y.to(device)
                optimiser.zero_grad()

                logits = model(X)                          # (B, seq_len, 5)
                # Reshape for CrossEntropyLoss: (B*seq_len, 5) and (B*seq_len,)
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()
                total_loss += loss.item()

            scheduler.step()
            avg_train_loss = total_loss / len(loaders["train"])

            # Validate
            model.eval()
            all_preds, all_labels = [], []
            val_loss = 0.0
            with torch.no_grad():
                for X, y in loaders["val"]:
                    X, y = X.to(device), y.to(device)
                    logits = model(X)
                    val_loss += criterion(logits.view(-1, logits.size(-1)), y.view(-1)).item()
                    all_preds.extend(logits.view(-1, logits.size(-1)).argmax(1).cpu().tolist())
                    all_labels.extend(y.view(-1).cpu().tolist())

            avg_val_loss = val_loss / len(loaders["val"])
            macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

            mlflow.log_metrics({
                "train_loss":   avg_train_loss,
                "val_loss":     avg_val_loss,
                "val_macro_f1": macro_f1,
            }, step=epoch)

            log.info(
                f"Epoch {epoch:03d} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={avg_val_loss:.4f} | "
                f"macro_f1={macro_f1:.4f}"
            )

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
                mlflow.log_artifact(str(ckpt_path))
                log.info(f"  ✓ New best F1={best_f1:.4f} — checkpoint saved.")
            else:
                no_improve += 1
                if no_improve >= patience:
                    log.info(f"Early stopping at epoch {epoch}.")
                    break

        mlflow.log_metric("best_val_macro_f1", best_f1)
        log.info(f"Training complete. Best macro F1: {best_f1:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/transformer.yaml")
    args = parser.parse_args()
    train(load_config(args.config))
