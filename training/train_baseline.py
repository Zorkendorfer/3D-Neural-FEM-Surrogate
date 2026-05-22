import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
from loguru import logger

class ScalarSurrogateMLP(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) # [Max Stress, Max Displacement]
        )
        
    def forward(self, x):
        return self.net(x)

def train():
    # 1. Load Data
    ds_path = "datasets/processed/dataset.h5"
    with h5py.File(ds_path, "r") as f:
        X = torch.tensor(f["inputs"][:], dtype=torch.float32)
        # Stack targets: Max Stress (index 0) and Max Displacement (index 1)
        y_stress = torch.tensor(f["max_stress"][:], dtype=torch.float32).unsqueeze(1)
        y_disp = torch.tensor(f["max_disp"][:], dtype=torch.float32).unsqueeze(1)
        Y = torch.cat([y_stress, y_disp], dim=1)

    # 2. Simple Normalization
    X_mean, X_std = X.mean(0), X.std(0)
    X = (X - X_mean) / (X_std + 1e-6)
    
    # 3. Split
    train_size = int(0.8 * len(X))
    train_ds = TensorDataset(X[:train_size], Y[:train_size])
    val_ds = TensorDataset(X[train_size:], Y[train_size:])
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    # 4. Model Setup
    model = ScalarSurrogateMLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    logger.info("Starting Baseline Training...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = sum(criterion(model(bx), by).item() for bx, by in val_loader) / len(val_loader)
            logger.info(f"Epoch {epoch} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

    # 5. Save Artifacts
    torch.save({
        'model_state_dict': model.state_dict(),
        'x_mean': X_mean,
        'x_std': X_std
    }, "models/baselines/mlp_v1.pt")
    logger.success("Baseline training complete. Model saved to models/baselines/mlp_v1.pt")

if __name__ == "__main__":
    train()