"""Train the MeshGraphNet FEM surrogate and report per-field test metrics."""

import json
import os
import sys
from pathlib import Path

# Lift the MPS memory high-watermark so PyTorch can use the full unified
# memory on Apple Silicon (default caps allocations to a fraction). Must be
# set before `import torch` so the MPS module reads it on initialisation.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.graph_dataset import load_graphs, make_graph_splits
from ml.gnn_model import MeshGraphNet


def _normalize_graphs(graphs, y_mean, y_std):
    y_mean_t = torch.tensor(y_mean)
    y_std_t  = torch.tensor(y_std)
    out = []
    for g in graphs:
        g = g.clone()
        g.y = (g.y - y_mean_t) / y_std_t
        out.append(g)
    return out


_AMP_DTYPE = torch.bfloat16   # BF16: no GradScaler needed, stable on Ada Lovelace


def _eval(model, loader, device, criterion):
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with torch.autocast(device_type=device.type, dtype=_AMP_DTYPE, enabled=device.type in ("cuda", "mps")):
                pred = model(batch)
                loss = criterion(pred, batch.y)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches


def train(
    root_dir:   str | Path = ROOT,
    epochs:     int   = 300,
    patience:   int   = 50,
    lr:         float = 1e-3,
    batch_size: int   = 4,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")
    ml_dir = ROOT / "ml"

    logger.info("Loading graphs...")
    graphs, _ = load_graphs(root_dir)
    logger.info(f"Loaded {len(graphs)} graphs")

    splits, norm, _ = make_graph_splits(graphs)
    y_mean, y_std   = norm["y_mean"], norm["y_std"]
    logger.info(
        f"Split — train: {len(splits['train'])}  "
        f"val: {len(splits['val'])}  test: {len(splits['test'])}"
    )

    loaders = {
        split: DataLoader(
            _normalize_graphs(gs, y_mean, y_std),
            batch_size=batch_size,
            shuffle=(split == "train"),
        )
        for split, gs in splits.items()
    }

    model    = MeshGraphNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # HuberLoss is more robust to stress concentrations than MSE
    # which often causes GNNs to plateau early.
    criterion = nn.HuberLoss(delta=1.0)
    mse_metric = nn.MSELoss()

    best_val, best_epoch, stale = float("inf"), 0, 0

    pbar = tqdm(range(1, epochs + 1), desc="training", unit="epoch", position=0)
    for epoch in pbar:
        model.train()
        train_loss_sum, n_train = 0.0, 0
        # Inner bar shows live per-step progress within this epoch; leave=False
        # so it disappears at epoch end and the scrollback only keeps the
        # one-line per-epoch summary written further down.
        inner = tqdm(
            loaders["train"],
            desc=f"  epoch {epoch:3d}/{epochs}",
            unit="batch",
            leave=False,
            position=1,
        )
        for step, batch in enumerate(inner, 1):
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=_AMP_DTYPE, enabled=device.type in ("cuda", "mps")):
                loss = criterion(model(batch), batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            n_train += 1
            inner.set_postfix(
                step=f"{step}/{len(loaders['train'])}",
                loss=f"{loss.item():.4f}",
                avg=f"{train_loss_sum / n_train:.4f}",
            )
        inner.close()
        train_loss = train_loss_sum / max(n_train, 1)

        val_loss = _eval(model, loaders["val"], device, criterion)
        scheduler.step(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val, best_epoch, stale = val_loss, epoch, 0
            torch.save(model.state_dict(), ml_dir / "best_gnn.pt")
        else:
            stale += 1

        lr_now = optimizer.param_groups[0]["lr"]
        # Persistent per-epoch line above the live progress bar; "*" marks a new best.
        pbar.write(
            f"epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  "
            f"best={best_val:.4f}  stale={stale:2d}  lr={lr_now:.1e}"
            + ("  *" if improved else "")
        )
        pbar.set_postfix(
            train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
            best=f"{best_val:.4f}", stale=stale, lr=f"{lr_now:.1e}",
        )

        if stale >= patience:
            logger.info(f"Early stop at epoch {epoch}.")
            break

    # ── Test evaluation ──────────────────────────────────────────────────
    model.load_state_dict(torch.load(ml_dir / "best_gnn.pt", weights_only=True))
    test_mse = _eval(model, loaders["test"], device, mse_metric)

    all_pred, all_true = [], []
    model.eval()
    with torch.no_grad():
        for batch in loaders["test"]:
            batch = batch.to(device)
            all_pred.append(model(batch).cpu())
            all_true.append(batch.y.cpu())

    pred_norm = torch.cat(all_pred).numpy()
    true_norm = torch.cat(all_true).numpy()
    pred = pred_norm * y_std + y_mean
    true = true_norm * y_std + y_mean

    names  = ["ux", "uz", "von_mises"]
    ss_res = ((true - pred) ** 2).sum(axis=0)
    ss_tot = ((true - true.mean(axis=0)) ** 2).sum(axis=0)
    r2   = 1 - ss_res / ss_tot
    rmse = np.sqrt(((true - pred) ** 2).mean(axis=0))

    # MARE only for von_mises — displacement has near-zero values at the fixed face
    mare_vm = float(np.mean(np.abs(pred[:, 2] - true[:, 2]) / (np.abs(true[:, 2]) + 1e-10)))

    logger.info("── Test results ─────────────────────────────────────")
    logger.info(f"MSE (normalised): {test_mse:.4f}")
    for i, name in enumerate(names):
        logger.info(f"  {name:12s}: R²={r2[i]:.4f}  RMSE={rmse[i]:.6f}")
    logger.info(f"von_mises MARE:   {mare_vm * 100:.2f}%")
    logger.info(f"Best epoch:       {best_epoch}")

    results = {
        "test_mse_norm":       round(float(test_mse), 6),
        "best_epoch":          best_epoch,
        "mare_von_mises_pct":  round(mare_vm * 100, 2),
    }
    for i, name in enumerate(names):
        results[f"r2_{name}"]   = round(float(r2[i]), 4)
        results[f"rmse_{name}"] = round(float(rmse[i]), 6)

    with open(ml_dir / "gnn_results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savez(ml_dir / "gnn_norm_stats.npz", y_mean=y_mean, y_std=y_std)
    logger.info("Saved ml/best_gnn.pt  ml/gnn_results.json  ml/gnn_norm_stats.npz")


if __name__ == "__main__":
    train(batch_size=8)
