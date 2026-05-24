# Training on Kaggle

Free GPU training for the MeshGraphNet, off your Mac.

## 1. Upload the graph cache as a Kaggle Dataset

1. Go to <https://www.kaggle.com/datasets> → **New Dataset**.
2. Upload `ml/graph_cache_k12_v2.pt` (~994 MB; built locally the first
   time `ml/train_gnn.py` ran).
3. Give it a title — e.g. **fem-surrogate-graphs**. Set visibility to Private.
4. After it's created, note the slug shown in the URL,
   e.g. `your-username/fem-surrogate-graphs`.

## 2. Create the notebook on Kaggle

1. <https://www.kaggle.com/code> → **New Notebook**.
2. **File → Upload Notebook** → pick `kaggle/train_gnn_kaggle.ipynb` from this repo.
3. Right sidebar → **Add Data** → search your dataset, add it.
4. Right sidebar → **Settings**:
   - **Accelerator: GPU T4 ×2** (P100 looks attractive on paper - 2.3× memory
     bandwidth - but Kaggle's PyTorch build dropped Pascal `sm_60` support,
     so P100 fails with "CUDA capability sm_60 is not compatible". T4 is
     Turing `sm_75` and works.)
   - **Internet: On** (needed once for the `pip install` cell)
5. In **cell 3** (`CACHE_PATH = Path(...)`) make sure the path matches your dataset's
   slug, i.e. `/kaggle/input/<your-dataset-name>/graph_cache_k12_v2.pt`.

## 3. Run

**Save Version → Save & Run All (Commit)**. Training:

- ~300 epochs cap, early-stopping after 50 stale validation epochs.
- One persistent line per epoch in scrollback; live per-batch progress bar.
- Estimated wall time on a single T4: ~10–30 min depending on early stop
  (with `num_workers=2` and fp16 autocast — without those the GPU sits idle
  waiting on CPU collation and a run can take 4+ hours).

Output appears in `/kaggle/working/`:

- `best_gnn.pt` — model weights
- `gnn_results.json` — test-set metrics (R², RMSE, MARE per channel)
- `gnn_norm_stats.npz` — `y_mean`, `y_std` for inference

## 4. Pull the trained model back

When the run finishes, **right sidebar → Output → Download**.
Drop the three files into your local `ml/` directory:

```
ml/best_gnn.pt
ml/gnn_results.json
ml/gnn_norm_stats.npz
```

Now any local inference / evaluation script that reads `ml/best_gnn.pt`
just works.

## Notes

- **Single-device only**: T4 ×2 gives you two T4 cards but the script uses
  one (`torch.device("cuda")` → GPU 0). Using both would need a
  DistributedDataParallel rewrite — overkill for a 1000-graph dataset.
- **Batch size**: notebook defaults to 32. T4's 16 GB can take 64 easily
  if you want a small further speedup — edit `BATCH_SIZE` in cell 8.
- **bf16 autocast**: enabled. T4's Tensor Cores are fp16-only (no native
  bf16) so the gain is modest, but the cast is cheap and PyTorch falls
  back to fp16/fp32 paths gracefully.
- **Why not P100**: Kaggle ships a CUDA build that requires `sm_70+`; the
  P100 is `sm_60` (Pascal) and is rejected at runtime. T4 is the only
  Kaggle GPU that actually runs.
