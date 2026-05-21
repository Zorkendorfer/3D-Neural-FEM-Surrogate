import os
import json
from pathlib import Path
import win32com.client
from loguru import logger
import yaml

class InventorAutomation:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.app = None
        self._connect_to_inventor()

    def _connect_to_inventor(self):
        """Connects to a running instance of Inventor or starts a new one."""
        try:
            # Try to get an active instance
            self.app = win32com.client.GetActiveObject("Inventor.Application")
            logger.info("Connected to existing Inventor instance.")
        except Exception:
            try:
                # Start a new instance
                self.app = win32com.client.Dispatch("Inventor.Application")
                self.app.Visible = True
                logger.info("Started new Inventor instance.")
            except Exception as e:
                logger.error(f"Failed to connect to Inventor: {e}")
                raise

    def generate_sample(self, params: dict, sample_id: str):
        """
        Opens template, updates parameters, saves IPT and metadata.
        """
        template_path = Path(self.config['paths']['cad_template']).absolute()
        output_root = Path(self.config['paths']['cad_output_dir'])
        sample_dir = output_root / f"beam_{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        part_output_path = sample_dir / "geometry.ipt"
        metadata_path = sample_dir / "metadata.json"

        doc = None
        try:
            # Open the template
            doc = self.app.Documents.Open(str(template_path))
            comp_def = doc.ComponentDefinition
            inv_params = comp_def.Parameters

            # Update parameters
            for key, value in params.items():
                try:
                    # Inventor parameters are case-sensitive and require unit strings
                    # We assume 'mm' based on plan, but could be parameterized
                    inv_params.Item(key).Expression = f"{value} mm"
                except Exception as e:
                    logger.warning(f"Could not set parameter {key}: {e}")

            # Recompute geometry
            doc.Update()
            
            # Save the generated part
            doc.SaveAs(str(part_output_path), False)
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(params, f, indent=2)

            logger.info(f"Successfully generated sample {sample_id}")
            return True

        except Exception as e:
            logger.error(f"Error generating sample {sample_id}: {e}")
            return False
        finally:
            if doc:
                doc.Close(True)

if __name__ == "__main__":
    # Example usage for verification (Phase 1, Task 5)
    config_file = Path(__file__).parents[2] / "configs" / "config.yaml"
    
    # Ensure logs directory exists
    os.makedirs(Path(__file__).parents[2] / "logs", exist_ok=True)
    logger.add(Path(__file__).parents[2] / "logs" / "generation.log")

    generator = InventorAutomation(config_file)

    # Test Sample
    test_params = {
        "LENGTH": 120.0,
        "WIDTH": 20.0,
        "HEIGHT": 10.0,
        "FILLET": 2.0,
        "HOLE_DIAMETER": 5.0
    }

    generator.generate_sample(test_params, "000001")