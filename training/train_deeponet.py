"""Phase 6 - DeepONet training.

Trains the DeepONet that predicts a spatial stress field from geometry
parameters and sampled coordinates. Runs on a Mac (uses Metal/MPS automatically).
"""

import sys
from pathlib import Path

import yaml
import h5py
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader, random_split
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# parents[1]: training/train_deeponet.py -> repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.deeponet.deeponet_model import FEMDeepONet


class DeepONetDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            # Global geometry parameters: [length, width, ...].
            self.inputs = torch.tensor(f["inputs"][:], dtype=torch.float32)
            # Coordinates: [N, 2048, 3].
            self.coords = torch.tensor(f["coords"][:], dtype=torch.float32)
            # Stress fields: [N, 2048].
            self.stress = torch.tensor(f["stress"][:], dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.coords[idx], self.stress[idx]


def train():
    config_path = ROOT / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    L.seed_everything(42)

    ds_path = ROOT / "datasets" / "processed" / "dataset.h5"
    if not ds_path.exists():
        print(f"Dataset not found at {ds_path}. Run preprocessing first!")
        return

    dataset = DeepONetDataset(ds_path)
    train_ds, val_ds = random_split(dataset, [0.85, 0.15])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    input_dim = len(config["geometry"]["parameters"])
    model = FEMDeepONet(branch_dim=input_dim)

    wandb_logger = WandbLogger(
        project=config["project_name"], name="deeponet_field_predictor"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=ROOT / "models" / "checkpoints",
        filename="deeponet-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=20)

    trainer = L.Trainer(
        max_epochs=300,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop],
        accelerator="auto",  # Uses Metal (MPS) on Mac automatically.
        devices=1,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
