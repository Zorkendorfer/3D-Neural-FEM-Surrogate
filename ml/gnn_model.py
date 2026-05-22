"""MeshGraphNet: encoder → N message-passing steps → decoder.

Per-node prediction: (ux, uy, uz, von_mises) from node/edge features.
No torch_cluster or torch_sparse required — scatter is done with
torch.Tensor.scatter_add_ which ships with base PyTorch.
"""

import torch
import torch.nn as nn


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """Two-layer MLP: in → out → out with GELU and final LayerNorm."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim), nn.GELU(),
        nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim),
    )


class _GNNLayer(nn.Module):
    """One interaction step (both with residual connections):
      1. Edge update:  e_ij' = MLP(h_i ∥ h_j ∥ e_ij) + e_ij
      2. Node update:  h_i'  = MLP(h_i ∥ Σ_{j→i} e_ij') + h_i
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.edge_mlp = _mlp(3 * latent_dim, latent_dim)
        self.node_mlp = _mlp(2 * latent_dim, latent_dim)

    def forward(
        self,
        x:          torch.Tensor,   # (N, D)
        edge_index: torch.Tensor,   # (2, E)  — row=src, col=dst
        edge_attr:  torch.Tensor,   # (E, D)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index

        new_edge  = self.edge_mlp(torch.cat([x[src], x[dst], edge_attr], dim=-1))
        edge_attr = edge_attr + new_edge

        # Sum incoming edge embeddings at each destination node
        agg = x.new_zeros(x.size(0), edge_attr.size(1))
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_attr), edge_attr)

        x = x + self.node_mlp(torch.cat([x, agg], dim=-1))
        return x, edge_attr


class MeshGraphNet(nn.Module):
    def __init__(
        self,
        in_node_dim: int = 12,   # 3 pos + 2 BC flags + 2 hole proximity + 5 geometry params
        in_edge_dim: int = 4,    # Δx, Δy, Δz, ‖Δ‖  (all normalised)
        out_dim:     int = 3,    # ux, uz, von_mises
        latent_dim:  int = 128,
        n_mp_steps:  int = 10,
    ):
        super().__init__()
        self.node_encoder = _mlp(in_node_dim, latent_dim)
        self.edge_encoder = _mlp(in_edge_dim, latent_dim)
        self.layers = nn.ModuleList([_GNNLayer(latent_dim) for _ in range(n_mp_steps)])
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, out_dim),
        )

    def forward(self, data) -> torch.Tensor:
        x = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)
        for layer in self.layers:
            x, e = layer(x, data.edge_index, e)
        return self.decoder(x)
