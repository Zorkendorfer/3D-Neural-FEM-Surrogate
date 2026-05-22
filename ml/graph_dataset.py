"""Build PyG graph objects for each FEA sample (per-node field prediction)."""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.dataset import _lhs_params

_PARAM_RANGES = {
    "LENGTH":        (100.0, 250.0),
    "WIDTH":         (20.0,  50.0),
    "HEIGHT":        (5.0,   20.0),
    "FILLET":        (1.0,   5.0),
    "HOLE_DIAMETER": (5.0,   15.0),
}


def _fast_csv(path: Path) -> np.ndarray:
    """Read a single-header CSV into a float32 array without np.loadtxt overhead."""
    with open(path) as f:
        lines = f.readlines()
    rows = [l.rstrip("\n\r").split(",") for l in lines[1:] if l.strip()]
    return np.array(rows, dtype=np.float32)


def _knn_edges(pos: torch.Tensor, k: int) -> torch.Tensor:
    """Undirected kNN edge_index from normalised 3-D positions."""
    n = pos.size(0)
    dists = torch.cdist(pos, pos)
    dists.fill_diagonal_(float("inf"))
    _, knn_idx = dists.topk(k, largest=False, dim=1)

    src = torch.arange(n).repeat_interleave(k)
    dst = knn_idx.reshape(-1)

    all_src = torch.cat([src, dst])
    all_dst = torch.cat([dst, src])
    edges = torch.stack([all_src, all_dst], dim=1)
    edges = torch.unique(edges, dim=0, sorted=True)
    return edges.T.contiguous()


def _process_sample(
    sample_dir: Path, k: int, lhs_lookup: dict
) -> tuple:
    """Build one PyG Data object from a single FEA sample directory.

    Returns (Data, sample_id) or (None, None) if the sample should be skipped.
    """
    if not (sample_dir / "summary.json").exists():
        return None, None

    sid         = sample_dir.name.split("_")[1]
    params_file = sample_dir / "params.json"

    if params_file.exists():
        with open(params_file) as f:
            params = json.load(f)
    elif sid in lhs_lookup:
        params = lhs_lookup[sid]
    else:
        return None, None

    nodes  = _fast_csv(sample_dir / "nodes.csv")
    disp   = _fast_csv(sample_dir / "displacement.csv")
    stress = _fast_csv(sample_dir / "stress.csv")   # (n, 1)

    L, W, H = params["LENGTH"], params["WIDTH"], params["HEIGHT"]
    D   = params["HOLE_DIAMETER"]
    eps = 1e-6
    n_nodes = nodes.shape[0]

    pos_norm  = (nodes / np.array([L, W, H])).astype(np.float32)
    is_fixed  = (nodes[:, 0] < eps).astype(np.float32).reshape(-1, 1)
    is_loaded = (nodes[:, 0] > L - eps).astype(np.float32).reshape(-1, 1)

    # Hole proximity — hole axis at (L/2, W/2) running along Z
    dx      = nodes[:, 0] - L / 2
    dy      = nodes[:, 1] - W / 2
    r_norm  = np.sqrt(dx**2 + dy**2) / (D / 2)          # 1.0 at hole surface
    r_norm  = r_norm.astype(np.float32).reshape(-1, 1)
    d_surf  = np.maximum(0.0, r_norm - 1.0)             # 0 at surface, grows outward

    param_feat = np.array(
        [(params[p] - lo) / (hi - lo) for p, (lo, hi) in _PARAM_RANGES.items()],
        dtype=np.float32,
    )
    x = np.concatenate(
        [pos_norm, is_fixed, is_loaded, r_norm, d_surf, np.tile(param_feat, (n_nodes, 1))],
        axis=1,
    )  # 12 features total

    # Targets: ux, uz, von_mises  (uy ≈ 0 for symmetric -Z load, excluded)
    y = np.concatenate([disp[:, [0, 2]], stress], axis=1).astype(np.float32)

    pos_t      = torch.tensor(pos_norm, dtype=torch.float)
    edge_index = _knn_edges(pos_t, k=k)
    src, dst   = edge_index
    delta      = pos_t[dst] - pos_t[src]
    edge_attr  = torch.cat([delta, delta.norm(dim=1, keepdim=True)], dim=1)

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(y, dtype=torch.float),
        pos=pos_t,
    ), sid


def load_graphs(root_dir: str | Path, k: int = 12) -> tuple[list[Data], list[str]]:
    """Load all completed FEA samples as PyG Data objects.

    Builds graphs in parallel using a thread pool (numpy/torch release the GIL
    for C-level ops; file I/O is also GIL-free).
    Caches results to ml/graph_cache_k{k}.pt for fast subsequent loads.

    Returns
    -------
    graphs     : list[Data]
    sample_ids : list[str]  (parallel, same order)
    """
    root_dir   = Path(root_dir)
    cache_file = root_dir / "ml" / f"graph_cache_k{k}_v2.pt"

    if cache_file.exists():
        logger.info(f"Loading graph cache from {cache_file.name}")
        graphs, sample_ids = torch.load(cache_file, weights_only=False)
        if graphs:
            return graphs, sample_ids
        logger.warning("Cached graph list is empty — rebuilding.")
        cache_file.unlink()

    exports    = root_dir / "fea" / "exports"
    all_lhs    = _lhs_params(root_dir)
    lhs_lookup = {f"{i + 1:06d}": p for i, p in enumerate(all_lhs)}

    candidates = sorted(exports.glob("beam_*"))
    n_workers  = min(os.cpu_count() or 4, 8)
    logger.info(f"Building {len(candidates)} graphs with {n_workers} threads...")

    results: dict[str, Data] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_process_sample, d, k, lhs_lookup): d.name
            for d in candidates
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="graphs"):
            data, sid = fut.result()
            if data is not None:
                results[sid] = data

    sample_ids = sorted(results)
    graphs     = [results[s] for s in sample_ids]

    logger.info(f"Built {len(graphs)} graphs. Saving cache → {cache_file.name}")
    torch.save((graphs, sample_ids), cache_file)
    return graphs, sample_ids


def make_graph_splits(
    graphs:     list[Data],
    val_ratio:  float = 0.1,
    test_ratio: float = 0.1,
    seed:       int   = 0,
) -> tuple[dict, dict, dict]:
    """Shuffle-split and compute z-score stats from the training set.

    Returns
    -------
    splits    : dict['train'|'val'|'test'] → list[Data]  (raw, unnormalised y)
    norm      : dict with float32 arrays y_mean, y_std   (shape (4,))
    split_idx : dict with integer index arrays into `graphs`
    """
    n    = len(graphs)
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_test   = int(n * test_ratio)
    n_val    = int(n * val_ratio)
    test_idx  = perm[:n_test]
    val_idx   = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    train_y = np.concatenate([graphs[i].y.numpy() for i in train_idx], axis=0)
    y_mean  = train_y.mean(axis=0).astype(np.float32)
    y_std   = np.maximum(train_y.std(axis=0), 1e-8).astype(np.float32)

    splits = {
        "train": [graphs[i] for i in train_idx],
        "val":   [graphs[i] for i in val_idx],
        "test":  [graphs[i] for i in test_idx],
    }
    return splits, {"y_mean": y_mean, "y_std": y_std}, {
        "train": train_idx, "val": val_idx, "test": test_idx,
    }
