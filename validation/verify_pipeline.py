"""Pipeline self-check.

Verifies the repo structure, config, HDF5 dataset, and model forward passes
without needing Windows, Inventor, or Nastran. Safe to run on a Mac.
"""

import sys
from pathlib import Path

import yaml
import h5py
import torch
from loguru import logger

# parents[1]: validation/verify_pipeline.py -> repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.baselines.mlp_model import FEMBaselineMLP
from models.deeponet.deeponet_model import FEMDeepONet


def verify_config():
    config_path = ROOT / "configs" / "config.yaml"
    if not config_path.exists():
        logger.error(f"config.yaml not found at {config_path}!")
        return None

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.success("config.yaml loaded successfully.")
    return config


def verify_dataset_integrity():
    ds_path = ROOT / "datasets" / "processed" / "dataset.h5"
    if not ds_path.exists():
        logger.warning("dataset.h5 not found. Skipping HDF5 structure check.")
        return

    with h5py.File(ds_path, "r") as f:
        for key in f.keys():
            logger.info(
                f"Dataset key: {key} | Shape: {f[key].shape} | Dtype: {f[key].dtype}"
            )
    logger.success("HDF5 dataset structure verified.")


def verify_models(config):
    input_dim = len(config["geometry"]["parameters"])

    # Baseline MLP.
    mlp = FEMBaselineMLP(input_dim=input_dim)
    try:
        out = mlp(torch.randn(8, input_dim))
        assert out.shape == (8, 2), f"Expected (8, 2), got {out.shape}"
        logger.success("Baseline MLP forward pass verified.")
    except Exception as e:
        logger.error(f"Baseline MLP failed: {e}")

    # DeepONet.
    deeponet = FEMDeepONet(branch_dim=input_dim)
    try:
        out = deeponet(torch.randn(4, input_dim), torch.randn(4, 2048, 3))
        assert out.shape == (4, 2048), f"Expected (4, 2048), got {out.shape}"
        logger.success("DeepONet forward pass verified.")
    except Exception as e:
        logger.error(f"DeepONet failed: {e}")


def check_directories():
    required_dirs = [
        "cad/templates",
        "cad/generated",
        "fea/exports",
        "datasets/processed",
        "logs",
        "models/checkpoints",
    ]
    for d in required_dirs:
        p = ROOT / d
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created missing directory: {d}")
    logger.success("Directory structure verified.")


if __name__ == "__main__":
    logger.info("Starting pipeline verification...")
    check_directories()
    cfg = verify_config()
    if cfg:
        verify_dataset_integrity()
        verify_models(cfg)
    logger.info("Verification complete.")
