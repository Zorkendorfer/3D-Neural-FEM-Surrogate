"""MLP surrogate: 5 geometry params → log(max_stress), log(max_displacement)."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 10,   # 5 geometry params + 5 physics-derived features
        out_dim: int = 2,
        hidden_dims: tuple = (256, 256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
