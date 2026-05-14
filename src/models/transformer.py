"""
Sleep Transformer — sequence model for sleep stage classification.

Architecture (TinySleepNet-style):
  1. CNN encoder:     each epoch (1, 3000) → embedding (256,)
                      reuses SleepCNN.encode() — can be frozen or fine-tuned
  2. Positional enc:  adds position information to the sequence
  3. Transformer:     attends across the sequence of epoch embeddings
  4. Classifier:      per-epoch FC head → 5 class logits

Input:  (batch, seq_len, 1, 3000)
Output: (batch, seq_len, 5)         — one prediction per epoch in sequence
"""

import math

import torch
import torch.nn as nn

from src.models.cnn import SleepCNN


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


class SleepTransformer(nn.Module):
    """
    Two-stage model:
      - CNN encodes each epoch independently into a fixed embedding
      - Transformer attends across the sequence of embeddings

    Args:
        cnn_checkpoint:  optional path to pretrained SleepCNN weights
        freeze_cnn:      if True, CNN weights are frozen during training
        d_model:         embedding dimension (must match CNN output = 256)
        n_heads:         number of attention heads
        n_layers:        number of Transformer encoder layers
        d_ff:            feedforward hidden dim inside Transformer
        dropout:         dropout rate
        n_classes:       number of sleep stages
    """

    def __init__(
        self,
        cnn_checkpoint: str | None = None,
        freeze_cnn: bool = False,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        n_classes: int = 5,
    ) -> None:
        super().__init__()

        # ── CNN epoch encoder ──────────────────────────────────────────────
        self.cnn = SleepCNN(n_classes=n_classes, dropout=0.0)
        if cnn_checkpoint:
            state = torch.load(cnn_checkpoint, map_location="cpu", weights_only=True)
            self.cnn.load_state_dict(state)

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        # ── Positional encoding ────────────────────────────────────────────
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # ── Transformer encoder ────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,   # (batch, seq, features)
            norm_first=True,    # Pre-LN — more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── Per-epoch classifier head ──────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, 1, 3000)
        returns: (batch, seq_len, n_classes)
        """
        batch, seq_len, C, T = x.shape

        # Encode each epoch independently with CNN
        x_flat = x.view(batch * seq_len, C, T)           # (B*S, 1, 3000)
        embeddings = self.cnn.encode(x_flat)              # (B*S, 256)
        embeddings = embeddings.view(batch, seq_len, -1)  # (B, S, 256)

        # Add positional encoding
        embeddings = self.pos_enc(embeddings)             # (B, S, 256)

        # Transformer — attends across the sequence
        context = self.transformer(embeddings)            # (B, S, 256)

        # Classify each epoch
        logits = self.classifier(context)                 # (B, S, 5)
        return logits


if __name__ == "__main__":
    model = SleepTransformer()
    x = torch.randn(4, 20, 1, 3000)   # batch=4, seq_len=20
    out = model(x)
    print(f"Output shape: {out.shape}")   # (4, 20, 5)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}")
