"""
MLflow utilities — thin wrappers so train.py stays clean.

Why MLflow instead of a hand-rolled database:
  - Local by default (no server needed, just `mlflow ui` to browse)
  - Logs params, metrics, artifacts, and model checkpoints in one call
  - Industry-standard tool worth knowing
  - Zero extra infrastructure for a solo project

To view the UI:
  mlflow ui --port 5000
  then open http://localhost:5000
"""

import mlflow


def log_config(cfg: dict, prefix: str = "") -> None:
    """Flatten a nested config dict and log all leaves as MLflow params."""
    flat = _flatten(cfg, prefix)
    mlflow.log_params(flat)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out
