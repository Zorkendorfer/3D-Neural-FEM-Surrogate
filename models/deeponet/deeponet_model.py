import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class FEMDeepONet(L.LightningModule):
    """DeepONet for spatial field prediction.

    Branch network: geometry parameters (and loads).
    Trunk network:  spatial coordinates [x, y, z].
    Output:         a scalar field value (e.g. stress) at each coordinate.
    """

    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int = 3,
        hidden_dim: int = 128,
        p_dim: int = 64,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Branch network: processes geometry parameters [length, width, ...].
        self.branch_net = nn.Sequential(
            nn.Linear(branch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, p_dim),
        )

        # Trunk network: processes spatial coordinates [x, y, z].
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, p_dim),
        )

        self.bias = nn.Parameter(torch.zeros(1))
        self.lr = lr

    def forward(self, x_branch, x_trunk):
        # x_branch: [batch, branch_dim]
        # x_trunk:  [batch, num_points, 3]
        b_out = self.branch_net(x_branch)  # [batch, p_dim]
        t_out = self.trunk_net(x_trunk)    # [batch, num_points, p_dim]

        # Dot product between branch and trunk outputs across the latent dim.
        res = torch.einsum("bp,bnp->bn", b_out, t_out)
        return res + self.bias

    def training_step(self, batch, batch_idx):
        branch_in, trunk_in, targets = batch
        y_hat = self(branch_in, trunk_in)
        loss = F.mse_loss(y_hat, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        branch_in, trunk_in, targets = batch
        y_hat = self(branch_in, trunk_in)
        val_loss = F.mse_loss(y_hat, targets)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
