"""Dataset loading and preprocessing for the beam FEM surrogate."""

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import qmc
from torch.utils.data import Dataset, DataLoader
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FEATURE_NAMES = ["LENGTH", "WIDTH", "HEIGHT", "FILLET", "HOLE_DIAMETER"]
TARGET_NAMES  = ["log_max_stress_mpa", "log_max_displacement_mm"]

_E = 200_000.0   # MPa — structural steel (must match run_simulation.py)
_F = 1_000.0     # N   — load magnitude (must match config)


def _physics_features(params: dict) -> list[float]:
    """Five derived features that encode the dominant beam physics.

    These give the model the analytical solution as a prior so it only needs
    to learn the correction factor (stress concentration, hole, fillet effects).
    All features are log-transformed to compress the wide dynamic range.
    """
    L  = params["LENGTH"]
    W  = params["WIDTH"]
    H  = params["HEIGHT"]
    D  = params["HOLE_DIAMETER"]
    I  = W * H**3 / 12.0          # second moment of area [mm⁴]

    analytical_disp   = _F * L**3 / (3.0 * _E * I)   # Euler-Bernoulli tip deflection
    analytical_stress = _F * L * (H / 2.0) / I        # bending stress at fixed face

    return [
        np.log(analytical_disp),     # log-scale so range matches log targets
        np.log(analytical_stress),
        D / H,                       # hole-to-height ratio  (controls stress concentration)
        D / W,                       # hole-to-width ratio
        L / H,                       # slenderness ratio
    ]


def _lhs_params(root_dir: Path) -> list[dict]:
    """Reproduce the exact deterministic LHS sampling used during data generation."""
    with open(root_dir / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)

    s_cfg = config["sampling"]
    param_names = list(s_cfg["ranges"].keys())
    ranges = s_cfg["ranges"]
    n = s_cfg["n_samples"]

    scaled = qmc.scale(
        qmc.LatinHypercube(d=len(param_names), seed=42).random(n=n),
        [ranges[p][0] for p in param_names],
        [ranges[p][1] for p in param_names],
    )

    def feasible(s):
        hd = s["HOLE_DIAMETER"]
        return hd < s["LENGTH"] and hd < s["WIDTH"] and hd < s["HEIGHT"]

    rng = np.random.default_rng(seed=99)
    samples = []
    for i in range(n):
        s = {param_names[j]: float(scaled[i, j]) for j in range(len(param_names))}
        if feasible(s):
            samples.append(s)
        else:
            while True:
                r = {p: float(rng.uniform(ranges[p][0], ranges[p][1])) for p in param_names}
                if feasible(r):
                    samples.append(r)
                    break
    return samples


def load_data(root_dir: str | Path):
    """Load all completed samples.

    Returns
    -------
    X : (N, 5) float32  — raw geometry parameters [mm]
    y : (N, 2) float32  — log(max_stress_mpa), log(max_displacement_mm)
    ids : list[str]     — sample IDs in the same order
    """
    root_dir = Path(root_dir)
    exports = root_dir / "fea" / "exports"
    all_params = _lhs_params(root_dir)

    X_rows, y_rows, ids = [], [], []
    for i, params in enumerate(all_params):
        sample_id = f"{i + 1:06d}"
        sample_dir = exports / f"beam_{sample_id}"

        # Newer runs save params.json; fall back to LHS reconstruction for older data.
        params_file = sample_dir / "params.json"
        if params_file.exists():
            with open(params_file) as f:
                params = json.load(f)

        summary_file = sample_dir / "summary.json"
        if not summary_file.exists():
            continue

        with open(summary_file) as f:
            s = json.load(f)

        phys = _physics_features(params)           # [log_disp_theory, log_stress_theory, D/H, D/W, L/H]
        X_rows.append([params[k] for k in FEATURE_NAMES] + phys)

        log_stress_theory = phys[1]
        log_disp_theory   = phys[0]
        # Residual targets: log(FEA / analytical) — the stress-concentration correction.
        # Much lower variance than raw log(FEA), easier to learn.
        y_rows.append([
            np.log(s["max_stress_mpa"])      - log_stress_theory,
            np.log(s["max_displacement_mm"]) - log_disp_theory,
        ])
        ids.append(sample_id)

    return (
        np.array(X_rows, dtype=np.float32),
        np.array(y_rows, dtype=np.float32),
        ids,
    )


class BeamDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(
    root_dir: str | Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    batch_size: int = 64,
    seed: int = 0,
):
    """Split data, fit normalisation on train, return DataLoaders + stats.

    Inputs are min-max scaled to [0, 1] using the training set bounds.
    Log-targets are z-scored using the training set mean and std.

    Returns
    -------
    loaders  : dict with keys 'train', 'val', 'test'
    norm     : dict with x_min, x_max, y_mean, y_std  (all numpy arrays)
    split_idx: dict with train/val/test index arrays into the original data
    """
    X_raw, y_log, ids = load_data(root_dir)
    n = len(X_raw)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    test_idx  = perm[:n_test]
    val_idx   = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    # Normalisation stats from training set only
    x_min  = X_raw[train_idx].min(axis=0)
    x_max  = X_raw[train_idx].max(axis=0)
    y_mean = y_log[train_idx].mean(axis=0)
    y_std  = y_log[train_idx].std(axis=0)

    X_norm = (X_raw - x_min) / (x_max - x_min + 1e-8)
    y_norm = (y_log - y_mean) / y_std

    def ds(idx):
        return BeamDataset(X_norm[idx], y_norm[idx])

    loaders = {
        "train": DataLoader(ds(train_idx), batch_size=batch_size, shuffle=True),
        "val":   DataLoader(ds(val_idx),   batch_size=batch_size),
        "test":  DataLoader(ds(test_idx),  batch_size=batch_size),
    }
    norm = {"x_min": x_min, "x_max": x_max, "y_mean": y_mean, "y_std": y_std}
    split_idx = {"train": train_idx, "val": val_idx, "test": test_idx}
    return loaders, norm, split_idx
