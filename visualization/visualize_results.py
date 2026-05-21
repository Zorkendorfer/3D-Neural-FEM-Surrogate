"""Phase 7 - Visualization.

Renders a 3D point cloud of a sample's von Mises stress field from the
processed HDF5 dataset using Plotly. Runs on a Mac.
"""

from pathlib import Path

import h5py
import plotly.graph_objects as go

# parents[1]: visualization/visualize_results.py -> repo root
ROOT = Path(__file__).resolve().parents[1]


def plot_sample(h5_path, sample_idx=0):
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][sample_idx]
        stress = f["stress"][sample_idx]
        params = f["inputs"][sample_idx]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=stress,
                    colorscale="Jet",
                    colorbar=dict(title="Von Mises Stress (MPa)"),
                    opacity=0.8,
                ),
            )
        ]
    )
    fig.update_layout(
        title=f"Sample {sample_idx} FEA Results | Params: {params}",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
        ),
    )
    fig.show()


if __name__ == "__main__":
    ds_path = ROOT / "datasets" / "processed" / "dataset.h5"
    if not ds_path.exists():
        print(f"Dataset not found at {ds_path}. Run generate_dummy_h5.py first.")
    else:
        plot_sample(ds_path, sample_idx=0)
