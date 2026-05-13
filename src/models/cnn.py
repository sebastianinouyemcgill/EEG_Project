"""
1D CNN for single-epoch sleep stage classification.

Architecture:
  4 conv blocks (Conv1d → BN → GELU → MaxPool)  — extracts local temporal features
  Global average pool                              — collapses time dimension
  Classifier head (FC → Dropout → FC)

Input:  (batch, 1, 3000)   — single 30-second EEG epoch at 100 Hz
Output: (batch, 5)         — logits over {Wake, N1, N2, N3, REM}
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int = 1,
        pool: int = 2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.MaxPool1d(pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SleepCNN(nn.Module):
    """
    Baseline CNN — fast to train, interpretable, good starting point.

    Block design note:
      Block 1 uses a wide kernel (50 samples = 500 ms) to capture slow waves.
      Later blocks use smaller kernels to pick up faster oscillations (spindles).
    """

    def __init__(self, n_classes: int = 5, dropout: float = 0.5) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(1,   32, kernel=50, pool=4),   # (B, 32, 375)
            ConvBlock(32,  64, kernel=10, pool=4),   # (B, 64, 93)
            ConvBlock(64, 128, kernel=5,  pool=2),   # (B, 128, 46)
            ConvBlock(128, 256, kernel=3,  pool=2),  # (B, 256, 11)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)           # (B, 256, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),                             # (B, 256)
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.pool(x)
        return self.classifier(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 256-dim embedding (for use in the Transformer stage)."""
        x = self.encoder(x)
        return self.pool(x).squeeze(-1)   # (B, 256)


if __name__ == "__main__":
    # Quick shape check
    model = SleepCNN()
    x = torch.randn(8, 1, 3000)
    logits = model(x)
    embed  = model.encode(x)
    print(f"Logits:    {logits.shape}")   # (8, 5)
    print(f"Embedding: {embed.shape}")    # (8, 256)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params:    {n_params:,}")
