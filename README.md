# 3D Neural FEM Surrogate

A machine-learning surrogate for 3D structural finite element analysis (FEA).
The pipeline generates parameterized CAD geometries, runs FEA simulations,
builds ML datasets, and trains neural surrogate models that predict structural
response (stress and displacement) far faster than a solver.

See [PLAN.md](PLAN.md) for the full phased implementation plan.

## Architecture

```text
CAD Generator  ->  FEA Automation  ->  Result Export
      ->  Dataset Builder  ->  ML Training  ->  Validation + Visualization
```

The first geometry is a cantilever beam parameterized by length, width,
height, fillet radius, and hole diameter, with a fixed base and a tip force.

## Repository Structure

```text
cad/          Inventor templates, generated parts, and the CAD script
fea/          FEA runs, exported results, and the simulation script
datasets/     raw / processed / metadata data, and dataset-building scripts
models/       model definitions (baselines, deeponet, graph_models)
training/     training entry points
validation/   pipeline self-checks
visualization/ result plotting
configs/      config.yaml (single source of truth for paths and parameters)
requirements/ per-platform dependency lists
notebooks/    exploratory notebooks
docs/         documentation
```

## Platform Split

- **Windows** runs the CAD/FEA half (Autodesk Inventor + Inventor Nastran):
  `generate_geometry.py`, `run_simulation.py`, `generate_dataset.py`.
- **Mac** runs the ML half: preprocessing, training, validation, visualization.

## Installation

Use Python 3.11.

```bash
# Windows (CAD + FEA)
pip install -r requirements/windows.txt

# Mac (ML)
pip install -r requirements/mac.txt
```

## Usage

All scripts read `configs/config.yaml` and resolve paths relative to the repo
root, so run them from the repository root.

### Dataset generation (Windows)

```bash
python datasets/scripts/generate_dataset.py     # CAD + FEA, resumable
python datasets/scripts/build_pointcloud_dataset.py  # raw exports -> dataset.h5
```

### Dummy dataset (Mac, no Windows needed)

```bash
python datasets/scripts/generate_dummy_h5.py    # writes datasets/processed/dataset.h5
```

### Training (Mac)

```bash
python training/train_baseline.py   # MLP: geometry params -> max stress/displacement
python training/train_deeponet.py   # DeepONet: spatial stress field
```

### Validation

```bash
python validation/verify_pipeline.py        # checks structure, config, models
python visualization/visualize_results.py   # 3D stress point cloud
```

## Dataset Format

Processed data is stored as HDF5 at `datasets/processed/dataset.h5`:

| Key          | Shape              | Description                        |
|--------------|--------------------|------------------------------------|
| `inputs`     | `[N, 5]`           | geometry parameters                |
| `coords`     | `[N, 2048, 3]`     | sampled point-cloud coordinates    |
| `stress`     | `[N, 2048]`        | von Mises stress per point         |
| `max_stress` | `[N]`              | global max stress (MPa)            |
| `max_disp`   | `[N]`              | global max displacement (mm)       |

Train / validation / test split is 70 / 15 / 15.
