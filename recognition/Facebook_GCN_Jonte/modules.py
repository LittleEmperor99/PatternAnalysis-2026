"""PyTorch model components for the Facebook node-classification project."""

import torch
from torch import nn


class MLP(nn.Module):
    """Feature-only baseline: linear -> ReLU -> dropout -> linear logits.

    Each row is classified independently. No edges or labels enter forward().
    Return unnormalised logits because CrossEntropyLoss applies log-softmax.
    """

    def __init__(
        self, in_features: int = 128, hidden_features: int = 64,
        num_classes: int = 4, dropout: float = 0.5,
    ):
        super().__init__()
        if min(in_features, hidden_features) <= 0 or num_classes < 2:
            raise ValueError("Feature dimensions must be positive; need at least two classes")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map an [N, in_features] feature tensor to [N, num_classes] logits."""
        return self.layers(features)
