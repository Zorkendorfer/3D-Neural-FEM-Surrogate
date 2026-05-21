"""Phase 5 - Baseline model training.

Trains the MLP baseline that predicts [max_stress, max_displacement] from
geometry parameters. Runs on a Mac (uses Metal/MPS automatically).
"""

import sys
from pathlib import Path

import yaml
import h5py
import torch
import numpy as np
import lightning as L
from torch.utils.data import Dataset, DataLoader, random_split
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# parents[1]: training/train_baseline.py -> repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.baselines.mlp_model import FEMBaselineMLP


class FEMDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            self.inputs = torch.tensor(f["inputs"][:], dtype=torch.float32)
            # Stack the scalar targets into a single [max_stress, max_disp] vector.
            max_s = f["max_stress"][:].reshape(-1, 1)
            max_d = f["max_disp"][:].reshape(-1, 1)
            self.targets = torch.tensor(
                np.hstack([max_s, max_d]), dtype=torch.float32
            )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def train():
    config_path = ROOT / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    L.seed_everything(42)

    ds_path = ROOT / "datasets" / "processed" / "dataset.h5"
    if not ds_path.exists():
        print(f"Dataset not found at {ds_path}. Run preprocessing first!")
        return

    full_dataset = FEMDataset(ds_path)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    input_dim = len(config["geometry"]["parameters"])
    model = FEMBaselineMLP(input_dim=input_dim)

    wandb_logger = WandbLogger(project=config["project_name"], name="baseline_mlp")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ROOT / "models" / "checkpoints",
        filename="baseline-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=20)

    trainer = L.Trainer(
        max_epochs=200,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop],
        accelerator="auto",  # Uses Metal (MPS) on Mac automatically.
        devices=1,
    )
    trainer.fit(model, train_loader, val_loader)
    # trainer.test evaluation on test_ds can follow here.


if __name__ == "__main__":
    train()
