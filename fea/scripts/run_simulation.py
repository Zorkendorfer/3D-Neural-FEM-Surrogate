"""Phase 2 - Automated FEA via Inventor Nastran iLogic automation.

Architecture
------------
  1. Python reads fea/ilogic/nastran_rule.vb and substitutes ALLCAPS placeholders.
  2. The customised rule is written to a temp .vb file and executed via the
     Inventor iLogic COM add-in.  The rule:
       a. Enters the Nastran In-CAD environment.
       b. Discovers named-face geometry IDs with GetContextID /
          GetReferenceKey / GetNastranEntityParams.
       c. Creates analysis, material, constraint, and load via XML Run() calls.
       d. Runs the solver.
       e. Writes the IPT directory to a signal file so Python can find outputs.
  3. Python locates the resulting .op2 (or .f06) files and parses them with
     pyNastran, then writes nodes.csv, displacement.csv, stress.csv, summary.json.

References
----------
  iLogic Command Reference (2020-2022):
    help.autodesk.com/view/NINCAD/<year>/ENU/?guid=GUID-8DAC709F-E247-464F-...
  Tutorial B7 - iLogic Linear Static Analysis:
    help.autodesk.com/cloudhelp/2023/ENU/NINCAD-Tutorials/files/GUID-B4853552...
"""

import csv
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import yaml
import win32com.client
from loguru import logger

_ILOGIC_ADDIN_GUID = "{3BDD8D79-2179-4B11-8A5A-257B1C0263AC}"
_INVENTOR_2024_EXE = r"C:\Program Files\Autodesk\Inventor 2024\Bin\Inventor.exe"

# Path to the VB.NET rule template, relative to this file.
_RULE_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "ilogic" / "nastran_rule.vb"


class NastranAutomation:
    def __init__(self, config_path, root_dir):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.root_dir = Path(root_dir)

        self.app = self._get_inventor()

        self.ilogic_addin = self.app.ApplicationAddIns.ItemById(_ILOGIC_ADDIN_GUID)
        if not self.ilogic_addin.Activated:
            self.ilogic_addin.Activate()
        self.ilogic = self.ilogic_addin.Automation

        logger.info(f"iLogic automation object acquired (Inventor {getattr(self.app, 'SoftwareVersion', '?')}).")

    @staticmethod
    def _get_inventor():
        """Connect to a running Inventor 2024 instance, or launch one."""
        try:
            app = win32com.client.GetActiveObject("Inventor.Application")
            logger.info(f"Connected to running Inventor (version: {getattr(app, 'SoftwareVersion', '?')}).")
            return app
        except Exception:
            logger.info("No running Inventor instance found. Launching Inventor 2024...")
            subprocess.Popen([_INVENTOR_2024_EXE])
            for _ in range(30):
                time.sleep(2)
                try:
                    app = win32com.client.GetActiveObject("Inventor.Application")
                    app.Visible = True
                    logger.info("Inventor 2024 started and connected.")
                    return app
                except Exception:
                    pass
            raise RuntimeError("Inventor 2024 did not become available within 60 seconds.")

    # ------------------------------------------------------------------
    # Rule generation
    # ------------------------------------------------------------------

    def _build_rule(self, sim_cfg: dict, signal_file: str) -> str:
        template = _RULE_TEMPLATE_PATH.read_text(encoding="utf-8")
        # Use plain str.replace so no Python format-string conflicts with VB.NET syntax.
        return (
            template
            .replace("FIXED_FACE_NAME", sim_cfg["fixed_face_name"])
            .replace("LOAD_FACE_NAME", sim_cfg["load_face_name"])
            .replace("MATERIAL_NAME", sim_cfg["material"])
            .replace("LOAD_MAGNITUDE", str(float(sim_cfg["load_magnitude"])))
            .replace("MESH_SIZE_MM", str(float(sim_cfg["mesh_size"])))
            .replace("SIGNAL_FILE_PATH", signal_file.replace("\\", "\\\\"))
        )

    # ------------------------------------------------------------------
    # Output file discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _find_output_file(ipt_path: str, extension: str, timeout_s: int = 120) -> Path | None:
        """Wait up to timeout_s seconds for a Nastran output file to appear.

        Nastran In-CAD may place output files in the IPT directory, a sub-folder
        named after the part, or a system temp location.
        """
        ipt = Path(ipt_path)
        part_stem = ipt.stem

        search_roots = [
            ipt.parent,
            ipt.parent / part_stem,
            Path(os.environ.get("TEMP", "")) / "InventorNastran",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Autodesk" / "Inventor Nastran",
        ]

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            for root in search_roots:
                if root.exists():
                    matches = sorted(
                        root.rglob(f"*{extension}"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if matches:
                        return matches[0]
            time.sleep(2)

        return None

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    def _parse_and_export(self, op2_path: Path, export_dir: Path) -> dict:
        try:
            from pyNastran.op2.op2 import OP2
        except ImportError:
            logger.error("pyNastran not installed. Run: pip install pyNastran. Falling back to .f06.")
            f06_path = op2_path.with_suffix(".f06")
            if f06_path.exists():
                return self._parse_f06(f06_path, export_dir)
            raise RuntimeError("No parseable output file found.")

        op2 = OP2(debug=False)
        op2.read_op2(str(op2_path))

        disp_obj = op2.displacements[1]
        node_ids = disp_obj.node_gridtype[:, 0].tolist()
        disp_data = disp_obj.data[0]

        try:
            coords = {nid: op2.nodes[nid].xyz for nid in node_ids}
        except Exception:
            coords = {nid: [0.0, 0.0, 0.0] for nid in node_ids}

        nodes_rows, disp_rows = [], []
        for idx, nid in enumerate(node_ids):
            x, y, z = coords.get(nid, [0.0, 0.0, 0.0])
            nodes_rows.append([nid, x, y, z])
            ux, uy, uz = float(disp_data[idx, 0]), float(disp_data[idx, 1]), float(disp_data[idx, 2])
            mag = (ux**2 + uy**2 + uz**2) ** 0.5
            disp_rows.append([nid, ux, uy, uz, mag])

        stress_rows = self._element_stress_to_nodal(op2, node_ids)
        return self._write_outputs(export_dir, nodes_rows, disp_rows, stress_rows)

    @staticmethod
    def _element_stress_to_nodal(op2, node_ids: list) -> list:
        node_vm_sum = {nid: 0.0 for nid in node_ids}
        node_vm_cnt = {nid: 0 for nid in node_ids}

        for key in ("ctetra_stress", "chexa_stress", "cpenta_stress"):
            obj = getattr(op2, key, None)
            if not obj:
                continue
            stress_obj = obj.get(1)
            if stress_obj is None:
                continue
            try:
                for eid_idx in range(stress_obj.data.shape[1]):
                    vm = float(stress_obj.data[0, eid_idx, -1])
                    nid = int(stress_obj.element_node[eid_idx, 1])
                    if nid in node_vm_sum:
                        node_vm_sum[nid] += vm
                        node_vm_cnt[nid] += 1
            except Exception:
                pass

        return [
            [nid, node_vm_sum[nid] / node_vm_cnt[nid] if node_vm_cnt[nid] > 0 else 0.0]
            for nid in node_ids
        ]

    # ------------------------------------------------------------------
    # Fallback: minimal .f06 text parser (displacement only)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_f06(f06_path: Path, export_dir: Path) -> dict:
        import re

        nodes_rows, disp_rows, stress_rows = [], [], []
        in_disp = False
        disp_pat = re.compile(
            r"^\s+(\d+)\s+G\s+([+-]?\d+\.\d+E[+-]?\d+)\s+([+-]?\d+\.\d+E[+-]?\d+)\s+([+-]?\d+\.\d+E[+-]?\d+)"
        )

        with open(f06_path, "r") as fh:
            for line in fh:
                if "D I S P L A C E M E N T   V E C T O R" in line:
                    in_disp = True
                    continue
                if in_disp:
                    m = disp_pat.match(line)
                    if m:
                        nid = int(m.group(1))
                        ux, uy, uz = float(m.group(2)), float(m.group(3)), float(m.group(4))
                        mag = (ux**2 + uy**2 + uz**2) ** 0.5
                        nodes_rows.append([nid, 0.0, 0.0, 0.0])
                        disp_rows.append([nid, ux, uy, uz, mag])
                        stress_rows.append([nid, 0.0])
                    elif line.strip() == "" and disp_rows:
                        in_disp = False

        if not disp_rows:
            raise RuntimeError(f"No displacement data found in {f06_path}")

        return NastranAutomation._write_outputs(export_dir, nodes_rows, disp_rows, stress_rows)

    # ------------------------------------------------------------------
    # CSV / JSON writers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_outputs(export_dir: Path, nodes_rows, disp_rows, stress_rows) -> dict:
        def write_csv(path, header, rows):
            with open(path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                writer.writerows(rows)

        write_csv(export_dir / "nodes.csv", ["node_id", "x", "y", "z"], nodes_rows)
        write_csv(
            export_dir / "displacement.csv",
            ["node_id", "ux", "uy", "uz", "magnitude"],
            disp_rows,
        )
        write_csv(export_dir / "stress.csv", ["node_id", "von_mises"], stress_rows)

        max_stress = max((r[1] for r in stress_rows), default=0.0)
        max_disp = max((r[4] for r in disp_rows), default=0.0)

        summary = {
            "max_stress_mpa": round(max_stress, 4),
            "max_displacement_mm": round(max_disp, 4),
            "solver_status": "success",
            "node_count": len(nodes_rows),
        }
        with open(export_dir / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)

        return summary

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_simulation(self, sample_id: str, ipt_path: str) -> bool:
        doc = None
        signal_file = str(
            self.root_dir / self.config["paths"]["fea_output_dir"] / f"beam_{sample_id}_workdir.txt"
        )
        rule_file = None

        try:
            doc = self.app.Documents.Open(ipt_path)
            sim_cfg = self.config["simulation"]

            rule_code = self._build_rule(sim_cfg, signal_file)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".vb", delete=False, encoding="utf-8"
            ) as tf:
                tf.write(rule_code)
                rule_file = tf.name

            logger.debug(f"iLogic rule written to: {rule_file}")
            self.ilogic.RunExternalRule(doc, rule_file)
            logger.info(f"iLogic rule completed for {sample_id}")

            op2_path = self._find_output_file(ipt_path, ".op2")
            if op2_path is None:
                op2_path = self._find_output_file(ipt_path, ".f06", timeout_s=10)
            if op2_path is None:
                raise FileNotFoundError(
                    "Nastran output files (.op2 / .f06) not found. "
                    "Check the Nastran working directory in Inventor Options."
                )

            export_dir = (
                self.root_dir
                / self.config["paths"]["fea_output_dir"]
                / f"beam_{sample_id}"
            )
            export_dir.mkdir(parents=True, exist_ok=True)

            if op2_path.suffix == ".op2":
                summary = self._parse_and_export(op2_path, export_dir)
            else:
                summary = self._parse_f06(op2_path, export_dir)

            logger.success(
                f"Sample {sample_id}: "
                f"max_stress={summary['max_stress_mpa']} MPa, "
                f"max_disp={summary['max_displacement_mm']} mm"
            )
            return True

        except Exception as e:
            logger.exception(f"FEA failed for {sample_id}: {e}")
            return False
        finally:
            if doc:
                doc.Close(True)
            if rule_file and Path(rule_file).exists():
                Path(rule_file).unlink(missing_ok=True)


if __name__ == "__main__":
    pass
