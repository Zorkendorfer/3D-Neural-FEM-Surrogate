"""Phase 3 - Dataset generation.

Samples geometry parameters with Latin Hypercube Sampling, then runs scikit-fem
FEA for each sample in parallel using ProcessPoolExecutor. Resumable: completed
samples (those with an existing summary.json) are skipped.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import qmc
from loguru import logger
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_simulation import SkfemSolver


def _run_sample(config_path: str, root_dir: str, sample_id: str, params: dict):
    """Worker function: one solver instance per call, safe for multiprocessing."""
    solver = SkfemSolver(config_path, root_dir)
    success = solver.run_simulation(sample_id, params)
    return sample_id, success


class DatasetOrchestrator:
    def __init__(self, config_path: str):
        config_path = Path(config_path).resolve()
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.config_path = str(config_path)
        self.root_dir = config_path.parents[1]
        self.root_dir_str = str(self.root_dir)

        log_dir = self.root_dir / self.config["paths"]["logs_dir"]
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(log_dir / "dataset_generation.log", rotation="10 MB")

    @staticmethod
    def _is_feasible(params: dict) -> bool:
        hd = params["HOLE_DIAMETER"]
        return hd < params["LENGTH"] and hd < params["WIDTH"] and hd < params["HEIGHT"]

    def _sample_valid_replacement(self, rng, param_names, ranges):
        while True:
            s = {p: float(rng.uniform(ranges[p][0], ranges[p][1])) for p in param_names}
            if self._is_feasible(s):
                return s

    def generate_lhs_samples(self):
        """LHS sampling with rejection replacement for infeasible geometries."""
        sampling_cfg = self.config["sampling"]
        param_names = list(sampling_cfg["ranges"].keys())
        n_samples = sampling_cfg["n_samples"]
        ranges = sampling_cfg["ranges"]

        sampler = qmc.LatinHypercube(d=len(param_names), seed=42)
        scaled = qmc.scale(
            sampler.random(n=n_samples),
            [ranges[p][0] for p in param_names],
            [ranges[p][1] for p in param_names],
        )

        rng = np.random.default_rng(seed=99)
        samples, n_replaced = [], 0

        for i in range(n_samples):
            s = {param_names[j]: float(scaled[i, j]) for j in range(len(param_names))}
            if self._is_feasible(s):
                samples.append(s)
            else:
                samples.append(self._sample_valid_replacement(rng, param_names, ranges))
                n_replaced += 1

        if n_replaced:
            logger.warning(f"Replaced {n_replaced}/{n_samples} infeasible LHS samples.")
        else:
            logger.info("All LHS samples are geometrically feasible.")

        return samples

    def run(self, max_workers: int | None = None):
        """Parallel dataset-generation loop with resume capability."""
        samples = self.generate_lhs_samples()
        logger.info(f"Planned samples: {len(samples)}")

        fea_output_dir = self.root_dir / self.config["paths"]["fea_output_dir"]

        # Build work list, skipping already-completed samples
        work = []
        for i, params in enumerate(samples):
            sample_id = f"{i + 1:06d}"
            if (fea_output_dir / f"beam_{sample_id}" / "summary.json").exists():
                continue
            work.append((sample_id, params))

        n_skip = len(samples) - len(work)
        if n_skip:
            logger.info(f"Skipping {n_skip} already-completed samples.")
        logger.info(f"Submitting {len(work)} samples to {max_workers or os.cpu_count()} workers.")

        n_ok = n_skip
        n_err = 0

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_run_sample, self.config_path, self.root_dir_str, sid, params): sid
                for sid, params in work
            }
            with tqdm(total=len(samples), initial=n_skip, desc="Generating dataset") as pbar:
                for future in as_completed(futures):
                    sample_id = futures[future]
                    try:
                        _, success = future.result()
                        if success:
                            n_ok += 1
                        else:
                            n_err += 1
                            logger.error(f"Simulation failed for sample {sample_id}")
                    except Exception as e:
                        n_err += 1
                        logger.exception(f"Worker crashed for sample {sample_id}: {e}")
                    pbar.update(1)

        logger.info(f"Done. {n_ok} succeeded, {n_err} failed out of {len(samples)} total.")


if __name__ == "__main__":
    config_file = ROOT / "configs" / "config.yaml"

    if not config_file.exists():
        logger.error(f"Config not found at {config_file}")
    else:
        DatasetOrchestrator(str(config_file)).run()
