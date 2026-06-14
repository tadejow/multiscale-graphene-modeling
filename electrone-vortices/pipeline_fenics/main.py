"""
Main orchestration script for the FEniCS Hydrodynamic Pipeline.
"""
import logging
import yaml
from pathlib import Path
from dolfin import File

from geometry import GrapheneDeviceMesh
from solver import GurzhiSolver


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    setup_logging()
    logging.info("Initializing FEniCS Hydrodynamic Pipeline...")

    # 1. Konfiguracja
    cfg = load_config("../configs/fenics/gurzhi_config.yaml")
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    geo_cfg = cfg["geometry"]
    phys_cfg = cfg["physics"]

    # 2. Generowanie geometrii i siatki
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
        mesh=mesh,
        D_nu=phys_cfg["gurzhi_length"],
        inflow_J=phys_cfg["inflow_current_density"],
        L=geo_cfg["channel_length"],
        W=geo_cfg["channel_width"]
    )

    J_field, phi_field = solver.solve()

    # 4. Zapis wyników do VTK (dla ParaView)
    logging.info(f"Exporting solutions to {out_dir}...")
    J_field.rename("Current_Density", "vector")
    phi_field.rename("Potential", "scalar")

    File(str(out_dir / "current_density.pvd")) << J_field
    File(str(out_dir / "potential.pvd")) << phi_field

    logging.info("Simulation completed successfully! Open .pvd files in ParaView to see the whirlpools.")


if __name__ == "__main__":
    main()