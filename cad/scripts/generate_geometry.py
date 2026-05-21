"""Phase 1 - CAD generation.

Opens a parameterized Inventor template, updates named parameters, and saves
the regenerated part plus its metadata. Windows-only (requires Autodesk Inventor).
"""

import os
import json
from pathlib import Path

import yaml
import win32com.client
from loguru import logger


class InventorAutomation:
    def __init__(self, config_path, root_dir):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.root_dir = Path(root_dir)

        self.app = None
        self._connect_to_inventor()

    def _connect_to_inventor(self):
        """Connect to a running Inventor instance, or start a new one."""
        try:
            self.app = win32com.client.GetActiveObject("Inventor.Application")
            logger.info("Connected to existing Inventor instance.")
        except Exception:
            try:
                self.app = win32com.client.Dispatch("Inventor.Application")
                self.app.Visible = True
                logger.info("Started new Inventor instance.")
            except Exception as e:
                logger.error(f"Failed to connect to Inventor: {e}")
                raise

    def generate_sample(self, params: dict, sample_id: str):
        """Open the template, update parameters, and save the IPT and metadata."""
        template_path = self.root_dir / self.config["paths"]["cad_template"]
        output_root = self.root_dir / self.config["paths"]["cad_output_dir"]
        sample_dir = output_root / f"beam_{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        part_output_path = sample_dir / "geometry.ipt"
        metadata_path = sample_dir / "metadata.json"

        doc = None
        try:
            doc = self.app.Documents.Open(str(template_path))
            comp_def = doc.ComponentDefinition
            inv_params = comp_def.Parameters

            # Inventor parameters are case-sensitive and require unit strings.
            for key, value in params.items():
                try:
                    inv_params.Item(key).Expression = f"{value} mm"
                except Exception as e:
                    logger.warning(f"Could not set parameter {key}: {e}")

            doc.Update()  # Recompute geometry.
            doc.SaveAs(str(part_output_path), False)

            with open(metadata_path, "w") as f:
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
    # parents[2]: cad/scripts/generate_geometry.py -> repo root
    root_dir = Path(__file__).resolve().parents[2]
    config_file = root_dir / "configs" / "config.yaml"

    os.makedirs(root_dir / "logs", exist_ok=True)
    logger.add(root_dir / "logs" / "generation.log")

    generator = InventorAutomation(config_file, root_dir)

    test_params = {
        "LENGTH": 120.0,
        "WIDTH": 20.0,
        "HEIGHT": 10.0,
        "FILLET": 2.0,
        "HOLE_DIAMETER": 5.0,
    }
    generator.generate_sample(test_params, "000001")
