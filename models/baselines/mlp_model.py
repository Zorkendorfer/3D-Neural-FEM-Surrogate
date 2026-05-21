import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class FEMBaselineMLP(L.LightningModule):
    """Baseline MLP that predicts scalar FEA targets from geometry parameters.

    Input:  [length, width, height, fillet, hole_diameter, ...]
    Output: [max_stress, max_displacement]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # Output: [max_stress, max_disp]
        )
        self.lr = lr

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)

        # Relative error (%) for interpretability; eps guards against zero targets.
        rel_error = torch.mean(torch.abs((y - y_hat) / (y + 1e-8))) * 100

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_rel_error_pct", rel_error, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
