"""Phase 3 - Dataset generation.

Samples geometry parameters with Latin Hypercube Sampling, then drives the CAD
and FEA pipelines for each sample. Resumable: completed samples are skipped.
Windows-only (requires Autodesk Inventor + Nastran).
"""

import sys
from pathlib import Path

import yaml
from scipy.stats import qmc
from loguru import logger
from tqdm import tqdm

# Make the repo root importable regardless of the working directory.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cad.scripts.generate_geometry import InventorAutomation
from fea.scripts.run_simulation import NastranAutomation


class DatasetOrchestrator:
    def __init__(self, config_path: str):
        config_path = Path(config_path).resolve()
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # config lives in configs/, so the repo root is two levels up.
        self.root_dir = config_path.parents[1]
        self.cad_gen = InventorAutomation(config_path, self.root_dir)
        self.fea_gen = NastranAutomation(config_path, self.root_dir)

        self.output_dir = self.root_dir / self.config["paths"]["cad_output_dir"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log_dir = self.root_dir / self.config["paths"]["logs_dir"]
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(log_dir / "dataset_generation.log", rotation="10 MB")

    def generate_lhs_samples(self):
        """Generate scaled Latin Hypercube Samples based on the config ranges."""
        sampling_cfg = self.config["sampling"]
        param_names = list(sampling_cfg["ranges"].keys())
        n_samples = sampling_cfg["n_samples"]

        sampler = qmc.LatinHypercube(d=len(param_names), seed=42)
        sample_points = sampler.random(n=n_samples)

        l_bounds = [sampling_cfg["ranges"][p][0] for p in param_names]
        u_bounds = [sampling_cfg["ranges"][p][1] for p in param_names]
        scaled_samples = qmc.scale(sample_points, l_bounds, u_bounds)

        return [
            {param_names[j]: float(scaled_samples[i, j]) for j in range(len(param_names))}
            for i in range(n_samples)
        ]

    def run(self):
        """Main dataset-generation loop with resume capability."""
        samples = self.generate_lhs_samples()
        logger.info(f"Planned samples: {len(samples)}")

        for i, params in enumerate(tqdm(samples, desc="Generating dataset")):
            sample_id = f"{i + 1:06d}"
            sample_dir = self.output_dir / f"beam_{sample_id}"
            metadata_path = sample_dir / "metadata.json"

            # Resume: skip samples that already have metadata.
            if metadata_path.exists():
                continue

            try:
                if not self.cad_gen.generate_sample(params, sample_id):
                    logger.error(f"CAD generation failed for sample {sample_id}")
                    continue

                ipt_path = str(sample_dir / "geometry.ipt")
                if not self.fea_gen.run_simulation(sample_id, ipt_path):
                    logger.error(f"FEA failed for sample {sample_id}")
                    continue

                logger.success(f"Completed sample {sample_id}")

            except Exception as e:
                # Continue to the next sample even if one fails.
                logger.exception(f"Critical error on sample {sample_id}: {e}")
                continue


if __name__ == "__main__":
    config_file = ROOT / "configs" / "config.yaml"

    if not config_file.exists():
        logger.error(f"Config not found at {config_file}")
    else:
        DatasetOrchestrator(str(config_file)).run()
