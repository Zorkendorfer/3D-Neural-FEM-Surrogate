"""Phase 4 - Preprocessing.

Converts raw FEA exports (CSV) into an ML-ready HDF5 dataset of sampled point
clouds. Each sample stores geometry parameters, 2048 sampled points with their
stress values, and the global scalar targets.
"""

import json
from pathlib import Path

import yaml
import h5py
import numpy as np
import pandas as pd
from loguru import logger

# parents[2]: datasets/scripts/build_pointcloud_dataset.py -> repo root
ROOT = Path(__file__).resolve().parents[2]

# Points sampled per geometry (see PLAN.md, Phase 4).
N_TARGET_POINTS = 2048


def process_simulation_results(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = ROOT / config["paths"]["fea_output_dir"]
    cad_dir = ROOT / config["paths"]["cad_output_dir"]
    processed_path = ROOT / "datasets" / "processed" / "dataset.h5"
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    all_inputs = []
    all_points = []
    all_stresses = []
    all_max_stresses = []
    all_max_displacements = []

    for sample_folder in sorted(raw_dir.glob("beam_*")):
        try:
            # Geometry parameters (input features). The current scikit-fem
            # solver writes params.json alongside the FEA exports; the legacy
            # Inventor flow wrote metadata.json under cad/generated/. Accept either.
            params_path = sample_folder / "params.json"
            if not params_path.exists():
                params_path = cad_dir / sample_folder.name / "metadata.json"
            with open(params_path, "r") as f:
                params = list(json.load(f).values())

            # Result CSVs.
            nodes = pd.read_csv(sample_folder / "nodes.csv")
            stress = pd.read_csv(sample_folder / "stress.csv")
            # displacement.csv is exported too; not used for the point cloud yet.

            # Global scalar targets.
            with open(sample_folder / "summary.json", "r") as f:
                summary = json.load(f)

            # Merge nodes + stress, then sample a fixed-size point cloud.
            # If the FEM mesh is coarser than N_TARGET_POINTS (typical for
            # small-cross-section beams) sample with replacement so the small
            # beams contribute redundant but valid points instead of being lost.
            data = pd.concat([nodes, stress], axis=1)
            replace = len(data) < N_TARGET_POINTS
            sampled_data = data.sample(N_TARGET_POINTS, replace=replace, random_state=42)

            all_inputs.append(params)
            all_points.append(sampled_data[["x", "y", "z"]].values)
            all_stresses.append(sampled_data["von_mises"].values)
            all_max_stresses.append(summary["max_stress_mpa"])
            all_max_displacements.append(summary["max_displacement_mm"])

        except Exception as e:
            logger.warning(f"Skipping {sample_folder.name}: {e}")

    if not all_inputs:
        logger.error(f"No usable samples found under {raw_dir}")
        return

    with h5py.File(processed_path, "w") as hf:
        hf.create_dataset("inputs", data=np.array(all_inputs, dtype=np.float32))
        hf.create_dataset("coords", data=np.array(all_points, dtype=np.float32))
        hf.create_dataset("stress", data=np.array(all_stresses, dtype=np.float32))
        hf.create_dataset("max_stress", data=np.array(all_max_stresses, dtype=np.float32))
        hf.create_dataset("max_disp", data=np.array(all_max_displacements, dtype=np.float32))

    logger.success(f"Saved {len(all_inputs)} samples to {processed_path}")


if __name__ == "__main__":
    config_file = ROOT / "configs" / "config.yaml"
    if not config_file.exists():
        logger.error(f"Config not found at {config_file}")
    else:
        process_simulation_results(str(config_file))
