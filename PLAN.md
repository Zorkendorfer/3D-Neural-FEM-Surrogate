# 3D Neural FEM Surrogate — Implementation Plan

# Project Goal

Build a machine learning surrogate model for 3D structural finite element analysis (FEA) using:

- Autodesk Inventor
- Autodesk Nastran In-CAD / Inventor Nastran
- Python
- PyTorch

The system should:
1. Automatically generate parameterized CAD geometries
2. Automatically run FEA simulations
3. Export simulation results
4. Build ML datasets
5. Train neural surrogate models
6. Validate predictions against real FEA solutions

---

# High-Level Architecture

```text
CAD Generator
    ↓
FEA Automation
    ↓
Result Export
    ↓
Dataset Builder
    ↓
ML Training
    ↓
Validation + Visualization
```

---

# IMPORTANT ENGINEERING RULES

## DO NOT:
- Start with graph neural networks
- Start with arbitrary meshes
- Start with complex CAD assemblies
- Start with full stress field prediction

## START WITH:
- Simple parametric geometries
- Scalar prediction targets
- Point-cloud or voxel representations
- Fully automated pipelines

---

# PHASE 0 — ENVIRONMENT SETUP

# Objective

Create stable cross-platform development environment.

---

# Windows Responsibilities

Use Windows ONLY for:
- Autodesk Inventor
- Nastran solving
- CAD automation
- Dataset generation

---

# Mac Responsibilities

Use Mac for:
- ML training
- Visualization
- Experimentation
- Research
- Dataset preprocessing

---

# Directory Structure

```text
neural-fem-3d/
│
├── cad/
│   ├── templates/
│   ├── generated/
│   └── scripts/
│
├── fea/
│   ├── runs/
│   ├── exports/
│   └── scripts/
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── models/
│   ├── baselines/
│   ├── deeponet/
│   └── graph_models/
│
├── training/
│
├── validation/
│
├── visualization/
│
├── configs/
│
├── notebooks/
│
├── docs/
│
├── requirements/
│
└── README.md
```

---

# Python Version

Use:
- Python 3.11

DO NOT use bleeding edge versions.

---

# Core Dependencies

## Windows

```bash
pip install:
pywin32
numpy
pandas
h5py
pyyaml
loguru
tqdm
```

## Mac + ML

```bash
pip install:
torch
torchvision
lightning
wandb
numpy
scipy
polars
h5py
matplotlib
plotly
open3d
trimesh
scikit-learn
```

---

# PHASE 1 — SIMPLE GEOMETRY PIPELINE

# Objective

Generate parameterized CAD geometries automatically.

---

# FIRST GEOMETRY

Use ONLY:
- cantilever beam

Parameters:
- length
- width
- height
- fillet radius
- hole diameter

Load:
- tip force

Constraint:
- fixed base

---

# Inventor Template Strategy

Create:
- single Inventor template file

Use named parameters:
- LENGTH
- WIDTH
- HEIGHT
- FILLET
- HOLE_DIAMETER

---

# Required Deliverables

## Script:
`generate_geometry.py`

Responsibilities:
- open Inventor template
- modify parameters
- regenerate geometry
- save generated part

---

# Output Structure

```text
cad/generated/
    beam_000001/
        geometry.ipt
        metadata.json
```

---

# Metadata Format

```json
{
  "length": 120.0,
  "width": 20.0,
  "height": 10.0,
  "fillet": 2.0,
  "hole_diameter": 5.0
}
```

---

# PHASE 2 — AUTOMATED FEA PIPELINE

# Objective

Automate Inventor Nastran simulations.

---

# Required Outputs

For each simulation export:
- nodal coordinates
- displacement
- von Mises stress
- max displacement
- max stress

---

# Deliverable Script

`run_simulation.py`

Responsibilities:
- load geometry
- apply material
- apply mesh settings
- apply constraints
- apply load
- run solver
- export results

---

# IMPORTANT

Use SINGLE material initially.

Recommended:
- Structural steel

---

# Mesh Settings

Use:
- tetrahedral mesh
- fixed mesh density

DO NOT vary mesh density initially.

---

# Output Structure

```text
fea/exports/
    beam_000001/
        nodes.csv
        displacement.csv
        stress.csv
        summary.json
```

---

# summary.json Example

```json
{
  "max_stress_mpa": 142.3,
  "max_displacement_mm": 1.82,
  "solver_status": "success"
}
```

---

# PHASE 3 — DATASET GENERATION

# Objective

Generate large supervised dataset.

---

# Initial Dataset Size

Generate:
- 1000 simulations

NOT more.

---

# Parameter Sampling

Use:
- Latin Hypercube Sampling (LHS)

DO NOT use purely random sampling.

---

# Deliverable Script

`generate_dataset.py`

Responsibilities:
- sample parameters
- generate CAD
- run simulation
- export results
- log failures

---

# Required Features

## Resume Capability

The generator MUST:
- skip completed samples
- continue after crashes

---

# Logging

Create:
`dataset_generation.log`

Track:
- failures
- invalid geometry
- solver crashes

---

# PHASE 4 — PREPROCESSING

# Objective

Convert raw FEM outputs into ML-ready tensors.

---

# IMPORTANT DESIGN CHOICE

DO NOT start with raw meshes.

Use:
- sampled point clouds

---

# Point Cloud Strategy

Sample:
- 2048 points

For each point store:
- x
- y
- z
- stress
- displacement magnitude

---

# Deliverable Script

`build_pointcloud_dataset.py`

---

# Final Tensor Format

## Inputs

```python
geometry_parameters
load_parameters
boundary_condition_parameters
```

## Outputs

Initially:
```python
max_stress
max_displacement
```

Later:
```python
pointwise_fields
```

---

# Dataset Format

Use:
- HDF5

NOT CSV for ML datasets.

---

# Example Structure

```text
datasets/processed/
    train.h5
    val.h5
    test.h5
```

---

# PHASE 5 — BASELINE MODEL

# Objective

Build FIRST working ML baseline.

---

# IMPORTANT

DO NOT start with DeepONet.

START WITH:
- simple MLP

---

# Baseline Input

```python
[length, width, height, fillet, hole_diameter, load]
```

---

# Baseline Output

```python
[max_stress, max_displacement]
```

---

# Deliverable

`train_baseline.py`

---

# Metrics

Track:
- MAE
- RMSE
- relative error %

---

# Validation

Split dataset:
- 70% train
- 15% validation
- 15% test

---

# PHASE 6 — DEEPONET IMPLEMENTATION

# Objective

Move from scalar prediction to field prediction.

---

# DeepONet Inputs

## Branch Network
Geometry parameters + loads

## Trunk Network
Spatial coordinates

---

# DeepONet Outputs

Predict:
- stress at spatial location
OR
- displacement at spatial location

---

# Deliverable

`train_deeponet.py`

---

# DeepONet Training Strategy

Sample:
- random spatial coordinates

DO NOT predict full volumetric grids initially.

---

# PHASE 7 — VALIDATION

# Objective

Compare neural predictions against Nastran.

---

# Required Validation Cases

Use:
- unseen geometries
- unseen load magnitudes
- unseen dimensions

---

# Deliverables

## Visualization
- stress contours
- displacement contours
- error heatmaps

## Metrics
- mean relative error
- max relative error

---

# PHASE 8 — ADVANCED MODELS

# ONLY AFTER EVERYTHING WORKS

---

# Future Architectures

## MeshGraphNet
Use for:
- arbitrary meshes

## Fourier Neural Operator
Use for:
- voxelized domains

## Point Transformer
Use for:
- complex geometry embeddings

---

# PHASE 9 — RESEARCH EXTENSIONS

Possible future directions:

- topology optimization surrogate
- thermal-structural coupling
- transient dynamics
- nonlinear materials
- fatigue prediction
- generative CAD optimization

---

# REQUIRED ENGINEERING PRACTICES

# Experiment Tracking

Use:
- Weights & Biases

Track:
- hyperparameters
- losses
- validation metrics
- dataset versions

---

# Configuration Management

Use:
- Hydra
OR
- YAML configs

DO NOT hardcode paths.

---

# Reproducibility

Set:
```python
torch.manual_seed(42)
numpy.random.seed(42)
```

---

# Data Versioning

Store:
- dataset metadata
- parameter ranges
- mesh settings
- material definitions

---

# REQUIRED README TASKS

The README MUST contain:
- installation
- architecture overview
- dataset structure
- training instructions
- validation instructions

---

# FIRST CODING TASKS

# PRIORITY ORDER

## 1
Create repo structure

## 2
Create Inventor parameterized beam template

## 3
Implement:
`generate_geometry.py`

## 4
Implement:
`run_simulation.py`

## 5
Generate first 10 simulations manually

## 6
Verify exported data integrity

## 7
Automate dataset generation

## 8
Train first MLP baseline

---

# SUCCESS CRITERIA

# Phase 1 Success
Can generate CAD automatically.

# Phase 2 Success
Can run automated FEA.

# Phase 3 Success
Can generate 1000 successful simulations.

# Phase 4 Success
Can train MLP predicting max stress within <10% error.

# Phase 5 Success
DeepONet predicts spatial stress field reasonably.

---

# CRITICAL WARNING

The HARDEST part is NOT neural networks.

The hardest part is:
- stable automation
- dataset consistency
- geometry parameterization
- mesh/result export

Focus engineering effort there first.

---

# FINAL RECOMMENDATION

Build this like industrial software:
- robust logging
- resumable pipelines
- config-driven architecture
- reproducible datasets

Do NOT optimize prematurely.

First goal:
```text
ONE FULLY AUTOMATED END-TO-END PIPELINE
```

Everything else comes later.