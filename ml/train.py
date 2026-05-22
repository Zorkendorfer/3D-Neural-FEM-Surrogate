"""Train the MLP beam FEM surrogate and report test metrics."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.dataset import make_loaders
from ml.model import MLP


def _eval(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for X, y in loader:
            total += criterion(model(X.to(device)), y.to(device)).item() * len(X)
    return total / len(loader.dataset)


def train(
    root_dir: str | Path = ROOT,
    epochs: int = 500,
    patience: int = 60,
    lr: float = 1e-3,
    batch_size: int = 64,
):
    device = torch.device("cpu")   # MLP on 1 k samples is faster on CPU than GPU
    ml_dir = ROOT / "ml"

    loaders, norm, split_idx = make_loaders(root_dir, batch_size=batch_size)
    n_train = len(split_idx["train"])
    n_val   = len(split_idx["val"])
    n_test  = len(split_idx["test"])
    logger.info(f"Split — train: {n_train}  val: {n_val}  test: {n_test}")

    model     = MLP().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val   = float("inf")
    best_epoch = 0
    stale      = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for X, y in loaders["train"]:
            optimizer.zero_grad()
            criterion(model(X.to(device)), y.to(device)).backward()
            optimizer.step()
        scheduler.step()

        val_loss = _eval(model, loaders["val"], criterion, device)

        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), ml_dir / "best_model.pt")
            stale = 0
        else:
            stale += 1

        if epoch % 50 == 0 or epoch == 1:
            tr = _eval(model, loaders["train"], criterion, device)
            logger.info(f"Epoch {epoch:4d} | train {tr:.4f}  val {val_loss:.4f}  best {best_val:.4f} @ {best_epoch}")

        if stale >= patience:
            logger.info(f"Early stop at epoch {epoch} ({patience} epochs without improvement).")
            break

    # ── Test evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ml_dir / "best_model.pt", weights_only=True))
    test_mse = _eval(model, loaders["test"], criterion, device)

    all_pred, all_true = [], []
    model.eval()
    with torch.no_grad():
        for X, y in loaders["test"]:
            all_pred.append(model(X.to(device)).cpu())
            all_true.append(y)

    # Un-standardise residual targets → residuals in log space
    y_mean = norm["y_mean"]
    y_std  = norm["y_std"]
    pred_resid = torch.cat(all_pred).numpy() * y_std + y_mean
    true_resid = torch.cat(all_true).numpy() * y_std + y_mean

    # Reconstruct full log(FEA) = residual + log(analytical)
    from ml.dataset import load_data, _physics_features, _lhs_params, ROOT as DS_ROOT
    _, _, ids = load_data(root_dir)
    test_ids = [ids[i] for i in split_idx["test"]]
    all_params_list = _lhs_params(Path(root_dir))
    id_to_params = {f"{i+1:06d}": p for i, p in enumerate(all_params_list)}

    log_analytical = np.array([
        [_physics_features(id_to_params[sid])[0],   # log_disp_theory
         _physics_features(id_to_params[sid])[1]]   # log_stress_theory
        for sid in test_ids
    ])                              # shape (n_test, 2) — note: col 0=disp, col 1=stress
    # Reorder to match target order [stress, disp]
    log_analytical = log_analytical[:, [1, 0]]

    pred_log = pred_resid + log_analytical
    true_log = true_resid + log_analytical

    pred = np.exp(pred_log)
    true = np.exp(true_log)

    # R² on log scale
    ss_res = ((true_log - pred_log) ** 2).sum(axis=0)
    ss_tot = ((true_log - true_log.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / ss_tot

    # Mean absolute relative error on original scale
    mare = np.mean(np.abs(pred - true) / true, axis=0)

    logger.info("── Test results ──────────────────────────────────")
    logger.info(f"MSE (log-scale):          {test_mse:.4f}")
    logger.info(f"R²  [stress, disp]:       {r2.round(4)}")
    logger.info(f"MARE [stress, disp]:      {(mare * 100).round(2)} %")
    logger.info(f"Best epoch:               {best_epoch}")

    results = {
        "test_mse_log":            round(float(test_mse), 6),
        "r2_stress":               round(float(r2[0]), 4),
        "r2_displacement":         round(float(r2[1]), 4),
        "mare_stress_pct":         round(float(mare[0] * 100), 2),
        "mare_displacement_pct":   round(float(mare[1] * 100), 2),
        "best_epoch":              best_epoch,
    }
    with open(ml_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Normalisation stats for inference
    np.savez(
        ml_dir / "norm_stats.npz",
        x_min=norm["x_min"], x_max=norm["x_max"],
        y_mean=norm["y_mean"], y_std=norm["y_std"],
    )
    logger.info("Saved ml/best_model.pt  ml/results.json  ml/norm_stats.npz")


if __name__ == "__main__":
    train()
