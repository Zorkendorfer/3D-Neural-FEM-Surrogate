import os
import yaml
import json
import numpy as np
from pathlib import Path
from scipy.stats import qmc
from loguru import logger
from tqdm import tqdm

# Import existing automation
from generate_geometry import InventorAutomation

class DatasetOrchestrator:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.cad_gen = InventorAutomation(config_path)
        self.output_dir = Path(self.config['paths']['cad_output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Logging
        log_path = Path(self.config['paths']['logs_dir']) / "dataset_generation.log"
        logger.add(log_path, rotation="10 MB")

    def generate_lhs_samples(self):
        """Generates scaled Latin Hypercube Samples based on config ranges."""
        sampling_cfg = self.config['sampling']
        param_names = list(sampling_cfg['ranges'].keys())
        n_samples = sampling_cfg['n_samples']
        
        sampler = qmc.LatinHypercube(d=len(param_names), seed=42)
        sample_points = sampler.random(n=n_samples)
        
        l_bounds = [sampling_cfg['ranges'][p][0] for p in param_names]
        u_bounds = [sampling_cfg['ranges'][p][1] for p in param_names]
        
        scaled_samples = qmc.scale(sample_points, l_bounds, u_bounds)
        
        samples = []
        for i in range(n_samples):
            sample_dict = {
                param_names[j]: float(scaled_samples[i, j]) 
                for j in range(len(param_names))
            }
            samples.append(sample_dict)
            
        return samples

    def run(self):
        """Main loop for dataset generation with resume capability."""
        samples = self.generate_lhs_samples()
        logger.info(f"Planned samples: {len(samples)}")

        for i, params in enumerate(tqdm(samples, desc="Generating Dataset")):
            sample_id = f"{i+1:06d}"
            sample_dir = self.output_dir / f"beam_{sample_id}"
            metadata_path = sample_dir / "metadata.json"

            # Resume Capability: Skip if metadata already exists
            if metadata_path.exists():
                continue

            try:
                # 1. Generate CAD
                success = self.cad_gen.generate_sample(params, sample_id)
                
                if not success:
                    logger.error(f"CAD generation failed for sample {sample_id}")
                    continue

                # 2. Run FEA (To be implemented in run_simulation.py)
                # This is where we will call NastranAutomation.run(...)
                
                logger.success(f"Completed sample {sample_id}")

            except Exception as e:
                logger.exception(f"Critical error on sample {sample_id}: {e}")
                # We continue to next sample even if one fails
                continue

if __name__ == "__main__":
    # Correct pathing based on root location
    config_file = Path(__file__).parent / "config.yaml"
    
    if not config_file.exists():
        logger.error(f"Config not found at {config_file}")
    else:
        orchestrator = DatasetOrchestrator(str(config_file))
        orchestrator.run()