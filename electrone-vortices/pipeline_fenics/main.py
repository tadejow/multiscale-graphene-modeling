"""
Main orchestration script for the dolfinx Hydrodynamic Pipeline.
"""
import logging
import yaml
from pathlib import Path
from mpi4py import MPI
from dolfinx import io, fem
import basix.ufl

from geometry import GrapheneDeviceMesh
from solver import GurzhiSolver

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    setup_logging()
    logging.info("Initializing FEniCSx (dolfinx) Hydrodynamic Pipeline...")

    # 1. Konfiguracja
    cfg = load_config("../configs/fenics/gurzhi_config.yaml")
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    geo_cfg = cfg["geometry"]
    phys_cfg = cfg["physics"]

    # 2. Generowanie geometrii i siatki (Gmsh -> dolfinx)
    domain = GrapheneDeviceMesh(
        length=geo_cfg["channel_length"],
        width=geo_cfg["channel_width"],
        radius=geo_cfg["disk_radius"],
        cx=geo_cfg["disk_center_x"],
        cy=geo_cfg["disk_center_y"],
        resolution=geo_cfg["mesh_resolution"]
    )
    mesh = domain.get_mesh()

    # 3. Rozwiązywanie układu PDE
    solver = GurzhiSolver(
        domain=mesh,
        D_nu=phys_cfg["gurzhi_length"],
        inflow_J=phys_cfg["inflow_current_density"],
        L=geo_cfg["channel_length"],
        W=geo_cfg["channel_width"]
    )

    J_field, phi_field = solver.solve()

    # 4. Interpolacja P2 -> P1 dla zapisu XDMF
    # Pola wektorowe Taylor-Hooda mają stopień 2 (P2), a siatka ma stopień 1 (P1).
    # Rzutujemy prędkość do przestrzeni liniowej przed eksportem.
    logging.info("Interpolating high-order functions to linear space for XDMF export...")
    cell_name = mesh.topology.cell_name()
    v_cg1 = basix.ufl.element("CG", cell_name, 1, shape=(mesh.geometry.dim,))
    V_p1 = fem.functionspace(mesh, v_cg1)

    J_out = fem.Function(V_p1)
    J_out.interpolate(J_field)
    J_out.name = "Current_Density"

    phi_field.name = "Potential"  # Cisnienie (phi) jest z definicji P1, więc jest gotowe do zapisu

    # 5. Zapis wyników do XDMF (zoptymalizowane pod ParaView)
    logging.info(f"Exporting solutions to {out_dir}...")

    with io.XDMFFile(MPI.COMM_WORLD, str(out_dir / "current_density.xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(J_out)

    with io.XDMFFile(MPI.COMM_WORLD, str(out_dir / "potential.xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(phi_field)

    logging.info("Simulation completed! Open the .xdmf files in ParaView to visualize the Gurzhi whirlpools.")

if __name__ == "__main__":
    main()