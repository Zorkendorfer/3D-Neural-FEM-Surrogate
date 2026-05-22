"""FEA via scikit-fem + gmsh: 3D cantilever beam for surrogate dataset generation.

Mesh: linear P1 tetrahedra via gmsh OCC kernel.
Physics: 3D linear elasticity (Navier-Cauchy).
  - Fixed (all DOFs = 0) at x = 0 face.
  - Uniform traction in -Z direction at x = L face.
"""

import json
import tempfile
from pathlib import Path

import meshio
import numpy as np
import yaml
import gmsh
from loguru import logger
from skfem import *
from skfem.io.meshio import from_meshio
from skfem.models.elasticity import linear_elasticity


class SkfemSolver:
    def __init__(self, config_path, root_dir=None):
        config_path = Path(config_path).resolve()
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        # config lives in configs/, repo root is one level up
        self.root_dir = Path(root_dir) if root_dir else config_path.parent.parent

    def _build_mesh(self, params, with_fillet: bool):
        """Inner mesh builder — called by create_mesh with fillet fallback support."""
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Verbosity", 0)
            gmsh.option.setNumber("Mesh.Algorithm3D", 4)   # Frontal: handles periodic/curved surfaces
            gmsh.model.add("beam")

            L = params["LENGTH"]
            W = params["WIDTH"]
            H = params["HEIGHT"]
            D = params.get("HOLE_DIAMETER", 0.0)
            # Clamp fillet to 45 % of the smaller cross-section dimension so OCC can always compute it.
            R = min(params.get("FILLET", 0.0), 0.45 * min(W, H)) if with_fillet else 0.0
            eps = 1e-6

            box_tag = gmsh.model.occ.addBox(0, 0, 0, L, W, H)

            if D > 0:
                # Two half-cylinders fused together — a full addCylinder() creates a
                # parametrically periodic surface that some gmsh algorithms cannot mesh.
                h1 = gmsh.model.occ.addCylinder(L / 2, W / 2, 0, 0, 0, H, D / 2, angle=np.pi)
                h2 = gmsh.model.occ.addCylinder(L / 2, W / 2, 0, 0, 0, H, D / 2, angle=np.pi)
                gmsh.model.occ.rotate([(3, h2)], L / 2, W / 2, 0, 0, 0, 1, np.pi)
                full_cyl, _ = gmsh.model.occ.fuse([(3, h1)], [(3, h2)])
                out, _ = gmsh.model.occ.cut([(3, box_tag)], full_cyl)
                box_tag = out[0][1]

            gmsh.model.occ.synchronize()

            if R > 0:
                # Fillet all edges at the fixed face (x ≈ 0)
                edges_at_base = gmsh.model.occ.getEntitiesInBoundingBox(
                    -eps, -eps, -eps, eps, W + eps, H + eps, dim=1
                )
                edge_tags = [tag for _, tag in edges_at_base]
                if edge_tags:
                    gmsh.model.occ.fillet([box_tag], edge_tags, [R] * len(edge_tags))
                    gmsh.model.occ.synchronize()

            # Adaptive mesh size: target ~5000 elements so large beams don't blow up
            # memory during parallel assembly. config mesh_size is the floor (min resolution).
            base_size = float(self.config["simulation"]["mesh_size"])
            target_elements = 5000
            # Empirical: n_elements ≈ 4.4 * volume / h³ for Frontal algorithm on these geometries.
            adaptive_size = (4.4 * L * W * H / target_elements) ** (1 / 3)
            mesh_size = max(base_size, adaptive_size)
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
            gmsh.model.mesh.generate(3)

            with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
                tmp_path = tmp.name
            gmsh.write(tmp_path)
        finally:
            gmsh.finalize()

        mesh = from_meshio(meshio.read(tmp_path))
        Path(tmp_path).unlink(missing_ok=True)
        return mesh, L

    def create_mesh(self, params):
        """Generate mesh, falling back to no-fillet if gmsh fails."""
        if params.get("FILLET", 0.0) > 0:
            try:
                return self._build_mesh(params, with_fillet=True)
            except Exception as e:
                logger.warning(f"Mesh with fillet failed ({e}), retrying without fillet.")
        return self._build_mesh(params, with_fillet=False)

    # ------------------------------------------------------------------

    def _von_mises_nodal(self, mesh, u, basis, lmbda, mu):
        """Vectorized nodal von Mises stress via element averaging.

        For P1 tets the strain is constant per element, so element averaging
        is exact — no L2 projection needed.
        """
        n_nodes = mesh.p.shape[1]

        # Component-wise nodal displacements
        ux = u[basis.nodal_dofs[0]]
        uy = u[basis.nodal_dofs[1]]
        uz = u[basis.nodal_dofs[2]]

        # Build shape-function gradient matrices for all elements at once.
        # mesh.p[:, mesh.t] → (3, 4, n_elems); transpose → (n_elems, 4, 3)
        coords = mesh.p[:, mesh.t].transpose(2, 1, 0)
        ones = np.ones((coords.shape[0], 4, 1))
        A = np.concatenate([ones, coords], axis=2)   # (n_elems, 4, 4)
        Ainv = np.linalg.inv(A)
        grad_N = Ainv[:, 1:, :]                       # (n_elems, 3, 4)

        # Element-level displacements: (n_elems, 4)
        ux_e = ux[mesh.t.T]
        uy_e = uy[mesh.t.T]
        uz_e = uz[mesh.t.T]

        # Constant strain per element: (n_elems, 3)
        dux = np.einsum("ijk,ik->ij", grad_N, ux_e)
        duy = np.einsum("ijk,ik->ij", grad_N, uy_e)
        duz = np.einsum("ijk,ik->ij", grad_N, uz_e)

        eps_xx, eps_yy, eps_zz = dux[:, 0], duy[:, 1], duz[:, 2]
        eps_xy = 0.5 * (dux[:, 1] + duy[:, 0])
        eps_yz = 0.5 * (duy[:, 2] + duz[:, 1])
        eps_xz = 0.5 * (dux[:, 2] + duz[:, 0])

        tr = eps_xx + eps_yy + eps_zz
        s_xx = lmbda * tr + 2 * mu * eps_xx
        s_yy = lmbda * tr + 2 * mu * eps_yy
        s_zz = lmbda * tr + 2 * mu * eps_zz
        s_xy = 2 * mu * eps_xy
        s_yz = 2 * mu * eps_yz
        s_xz = 2 * mu * eps_xz

        vm_elem = np.sqrt(
            0.5 * (
                (s_xx - s_yy) ** 2 + (s_yy - s_zz) ** 2 + (s_zz - s_xx) ** 2
                + 6 * (s_xy ** 2 + s_yz ** 2 + s_xz ** 2)
            )
        )

        # Average element values to nodes
        vm_sum = np.zeros(n_nodes)
        vm_cnt = np.zeros(n_nodes)
        for i in range(4):
            np.add.at(vm_sum, mesh.t[i], vm_elem)
            np.add.at(vm_cnt, mesh.t[i], 1)
        return vm_sum / np.maximum(vm_cnt, 1)

    def solve_elasticity(self, mesh, L, W, H, load_val):
        """3D linear elasticity: cantilever with uniform tip traction in -Z."""
        E = 200e3   # MPa — structural steel
        nu = 0.3
        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        element = ElementVector(ElementTetP1())
        basis = Basis(mesh, element)

        K = asm(linear_elasticity(lmbda, mu), basis)

        # Neumann BC: uniform traction in -Z at the tip face (x ≈ L).
        # load_val is total force [N]; divide by tip face area to get traction [N/mm²].
        tip_area = W * H
        traction_val = load_val / tip_area

        eps = 1e-10
        tip_facets = mesh.facets_satisfying(lambda x: x[0] > L - eps)
        f_basis = FacetBasis(mesh, element, facets=tip_facets)

        @LinearForm
        def traction(v, w):
            return -traction_val * v[2]

        f = asm(traction, f_basis)

        # Dirichlet BC: fix all DOFs at the base face (x ≈ 0)
        fixed_facets = mesh.facets_satisfying(lambda x: x[0] < eps)
        D = basis.get_dofs(fixed_facets)
        u = solve(*condense(K, f, D=D))

        vm_stress = self._von_mises_nodal(mesh, u, basis, lmbda, mu)
        return u, vm_stress, basis

    def run_simulation(self, sample_id: str, params: dict) -> bool:
        try:
            logger.info(f"Skfem simulation started: {sample_id}")

            mesh, L = self.create_mesh(params)
            load_val = float(self.config["simulation"]["load_magnitude"])
            W, H = params["WIDTH"], params["HEIGHT"]
            u, vm_stress, basis = self.solve_elasticity(mesh, L, W, H, load_val)

            # Nodal displacements: (n_nodes, 3)
            u_nodal = np.column_stack([
                u[basis.nodal_dofs[0]],
                u[basis.nodal_dofs[1]],
                u[basis.nodal_dofs[2]],
            ])
            max_disp = float(np.max(np.linalg.norm(u_nodal, axis=1)))
            max_stress = float(np.max(vm_stress))

            export_dir = (
                self.root_dir
                / self.config["paths"]["fea_output_dir"]
                / f"beam_{sample_id}"
            )
            export_dir.mkdir(parents=True, exist_ok=True)

            with open(export_dir / "params.json", "w") as fh:
                json.dump(params, fh, indent=2)

            np.savetxt(
                export_dir / "nodes.csv", mesh.p.T,
                delimiter=",", header="x,y,z", comments=""
            )
            np.savetxt(
                export_dir / "displacement.csv", u_nodal,
                delimiter=",", header="ux,uy,uz", comments=""
            )
            np.savetxt(
                export_dir / "stress.csv", vm_stress[:, None],
                delimiter=",", header="von_mises", comments=""
            )

            summary = {
                "max_stress_mpa": round(max_stress, 4),
                "max_displacement_mm": round(max_disp, 4),
                "solver_status": "success",
                "node_count": mesh.p.shape[1],
            }
            with open(export_dir / "summary.json", "w") as fh:
                json.dump(summary, fh, indent=2)

            logger.success(
                f"Sample {sample_id}: max_stress={max_stress:.2f} MPa, "
                f"max_disp={max_disp:.4f} mm, nodes={mesh.p.shape[1]}"
            )
            return True

        except Exception as e:
            logger.exception(f"FEA failed for {sample_id}: {e}")
            return False


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    solver = SkfemSolver(root_dir / "configs" / "config.yaml", root_dir)
    test_params = {
        "LENGTH": 100.0,
        "WIDTH": 20.0,
        "HEIGHT": 10.0,
        "FILLET": 2.0,
        "HOLE_DIAMETER": 5.0,
    }
    solver.run_simulation("test_001", test_params)
