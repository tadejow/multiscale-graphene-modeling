# Pipeline: Electron Hydrodynamics (FEniCSx / dolfinx)

This directory contains the macroscopic finite element method (FEM) pipeline designed to simulate the steady-state viscous flow of the electron fluid in graphene. By solving the linearized Navier-Stokes (Gurzhi) equations, this pipeline numerically reproduces the stationary current whirlpools (vortices) observed in room-temperature scanning magnetometry experiments at ETH Zurich.

The implementation is built on the modern **FEniCSx (`dolfinx`)** platform, utilising **Gmsh** for CAD-like geometry creation, and **PETSc/MUMPS** for native, high-performance sparse matrix solver operations on high-performance computing (HPC) clusters.

---

## ??? Module Architecture

### `geometry.py`
**Responsibility:** Computational Domain and Mesh Generation.
Utilizes the Python API of **Gmsh** to build the 2D device geometry from the ETHZ paper (main channel + circular side pocket) via Constructive Solid Geometry (CSG) and Boolean operations:
* Defines a rectangular main channel and a circular side pocket.
* Executes a Boolean union (`fuse`) of both shapes and synchronizes the CAD kernel.
* Sets up physical groups for boundary marker mapping.
* Generates a 2D unstructured Delaunay triangulation mesh with a globally constrained mesh size, translating it directly into a `dolfinx.mesh.Mesh` object.

### `solver.py`
**Responsibility:** Variational Formulation and PDE Solving.
Sets up and solves the coupled, linear saddle-point problem representing the Gurzhi equations:
* **Function Space:** Instantiates a mixed finite element space using **Taylor-Hood ($P_2 - P_1$)** elements. Continuous piecewise quadratics ($P_2$) are assigned to the current density $\mathbf{J}$, while continuous piecewise linears ($P_1$) are assigned to the potential $\phi$. This choice rigorously satisfies the Ladyzhenskaya-Babuška-Brezzi (LBB) inf-sup stability condition.
* **Boundary Conditions:** Extracts boundary facets topologically. Applies a uniform Dirichlet inflow velocity at the inlet, a grounded reference potential ($\phi=0$) at the outlet, and strict **no-slip** ($\mathbf{J}=\mathbf{0}$) boundary conditions on all solid device walls.
* **PETSc Assembly:** Manages native matrix and vector assembly (`assemble_matrix`, `assemble_vector`, and boundary lifting) to bypass fragile wrapper APIs, ensuring robust compatibility across minor FEniCSx releases.
* **Solver:** Configures the **MUMPS** direct sparse solver with LU factorization via `petsc4py` to robustly invert the indefinite block matrix.

### `main.py`
**Responsibility:** Orchestration and I/O.
Coordinates the entire macroscopic pipeline:
* Loads geometry and physics parameters from `../configs/fenics/gurzhi_config.yaml`.
* Drives mesh generation and solver execution.
* Handles post-processing: because XDMF does not natively support writing quadratic functions ($P_2$) on linear meshes ($P_1$), it performs an **interpolation (projection) from $P_2 \to P_1$** on the current density vector field.
* Exports the velocity ($\mathbf{J}$) and potential ($\phi$) fields to ParaView-compliant `.xdmf` files.

---

## ?? Execution Workflow

FEniCSx requires a highly specific compiled C++/MPI backend. It is strongly recommended to run this pipeline in a configured Conda environment or on a cluster node managed via Slurm.

### Step 1: Environment Setup
Ensure FEniCSx and Gmsh are loaded. On a cluster, activate your conda environment:
```bash
conda activate fenicsx_env
```

### Step 2: Parameter Configuration
Adjust your physical properties (such as the Gurzhi length $D_\nu$ or inlet current) and device dimensions in `../configs/fenics/gurzhi_config.yaml`.

### Step 3: Run the Solver
Execute the orchestrator locally or submit it as a Slurm batch job (`sbatch`):
```bash
python main.py
```
* **Output:** Generates `current_density.xdmf` (and associated `.h5` files) and `potential.xdmf` in the designated output folder (e.g., `../data/data_fenics/`).

### Step 4: Visualization in ParaView
To visualize the physical whirlpools and velocity profiles:
1. Open **ParaView** on your local machine.
2. Load `current_density.xdmf`.
3. Apply the **Glyph** filter (to plot current vectors) or the **Stream Tracer** filter (to plot continuous stream lines of the flow).
4. Observe the characteristic **Poiseuille parabolic profile** in the main channel ($J_x$) and the alternating sign of the vertical current ($J_y$) inside the circular pocket, proving the existence of a stationary, viscous **current vortex**.