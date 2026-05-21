"""Generate a dummy HDF5 dataset.

Lets the ML pipeline (training, validation, visualization) be developed and
tested on a Mac without access to Windows, Inventor, or Nastran. The data is
random noise shaped exactly like a real processed dataset.
"""

from pathlib import Path

import yaml
import h5py
import numpy as np
from loguru import logger

# parents[2]: datasets/scripts/generate_dummy_h5.py -> repo root
ROOT = Path(__file__).resolve().parents[2]


def generate_dummy_dataset(
    config_path: str,
    num_samples: int = 100,
    num_points_per_sample: int = 2048,
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    processed_dir = ROOT / "datasets" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    ds_path = processed_dir / "dataset.h5"

    input_dim = len(config["geometry"]["parameters"])
    logger.info(f"Generating dummy HDF5 dataset with {num_samples} samples...")

    rng = np.random.default_rng(42)
    # Scaled to roughly realistic magnitudes (mm, MPa).
    inputs = rng.random((num_samples, input_dim), dtype=np.float32) * 100
    coords = rng.random((num_samples, num_points_per_sample, 3), dtype=np.float32) * 200
    stress = rng.random((num_samples, num_points_per_sample), dtype=np.float32) * 300
    max_stress = np.max(stress, axis=1).astype(np.float32)
    max_disp = rng.random(num_samples, dtype=np.float32) * 5

    with h5py.File(ds_path, "w") as hf:
        hf.create_dataset("inputs", data=inputs)
        hf.create_dataset("coords", data=coords)
        hf.create_dataset("stress", data=stress)
        hf.create_dataset("max_stress", data=max_stress)
        hf.create_dataset("max_disp", data=max_disp)

    logger.success(f"Dummy dataset saved to {ds_path}")
    logger.info("You can now run the training, validation, and visualization scripts.")


if __name__ == "__main__":
    config_file = ROOT / "configs" / "config.yaml"
    if not config_file.exists():
        logger.error(f"Config not found at {config_file}")
    else:
        generate_dummy_dataset(str(config_file))
