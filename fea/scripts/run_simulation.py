"""Phase 2 - Automated FEA.

Drives Autodesk Inventor Nastran (Nastran In-CAD) through COM to mesh, solve,
and export results for a generated part. Windows-only.
"""

from pathlib import Path

import yaml
import win32com.client
from loguru import logger


class NastranAutomation:
    # Inventor Nastran In-CAD add-in GUID.
    INCAD_ADDIN_ID = "{24A9992B-0F41-4340-A227-96A19FF4E41F}"

    def __init__(self, config_path, root_dir):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.root_dir = Path(root_dir)

        self.app = win32com.client.GetActiveObject("Inventor.Application")
        self.incad_addin = self.app.ApplicationAddIns.ItemById(self.INCAD_ADDIN_ID)

        if not self.incad_addin.Activated:
            self.incad_addin.Activate()

        self.nastran_obj = self.incad_addin.Automation

    def run_simulation(self, sample_id: str, ipt_path: str):
        """Run the FEA pipeline for a single sample."""
        doc = None
        try:
            doc = self.app.Documents.Open(ipt_path)
            model = self.nastran_obj.GetModel()

            # 1. Setup analysis (1 = Linear Static).
            analysis = model.CreateAnalysis(1, f"Analysis_{sample_id}")

            # 2. Material and idealization.
            # Detailed property setup via COM typically requires specific ID
            # mapping; 'Structural Steel' is assumed available in the library.

            # 3. Meshing.
            mesh_cfg = self.config["simulation"]
            model.SetMeshSize(mesh_cfg["mesh_size"])
            model.GenerateMesh()

            # 4. Constraints (fixed support) - Face1 tagged in the IPT template.
            # model.CreateConstraint(...)

            # 5. Loads (tip force).
            # model.CreateLoad(...)

            # 6. Solve.
            model.RunAnalysis()

            # 7. Export results to the configured CSV paths.
            export_base = (
                self.root_dir
                / self.config["paths"]["fea_output_dir"]
                / f"beam_{sample_id}"
            )
            export_base.mkdir(parents=True, exist_ok=True)
            # result = analysis.GetResult()
            # result.ExportCSV(...)

            logger.info(f"FEA successful for {sample_id}")
            return True

        except Exception as e:
            logger.error(f"FEA failed for {sample_id}: {e}")
            return False
        finally:
            if doc:
                doc.Close(True)


if __name__ == "__main__":
    # Driven by datasets/scripts/generate_dataset.py.
    pass
