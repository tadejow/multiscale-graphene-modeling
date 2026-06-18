# Pipeline: Fireball (Quantum Mechanics)

This directory contains automated pipelines for executing Tight-Binding Density Functional Theory (TB-DFT) simulations using the `fireball.x` solver. The project follows ISO software engineering standards, separating grid generation from physics computation.

## Directory Structure

* `ml_dataset/` - Generates a large dataset of variably strained graphene structures. Extracts total energy, Fermi energy, and $F_{\max}$.
* `band_structure/` - Computes the 1D energy dispersion relation.
* `dirac_cone/` - Computes a high-resolution 2D energy map around the $\mathbf{K}$ point.
* `density_of_states/` - Integrates the Hamiltonian over the entire Brillouin Zone to extract the DOS profile.
* `visualizations/` - Destination folder for Python plotting scripts and output images.

## Execution Guide

### 1. Generating K-points meshes
Before running physics computations, you must generate the reciprocal space grids. Navigate to the desired module and execute its generator script:
```bash
cd band_structure/
bash generate_path.sh
```
*Note: Generator scripts rely on `awk` and force `LC_NUMERIC=C` to ensure floating-point precision compatibility across different OS locales.*

### 2. Computing Physics (The Two-Step SCF Protocol)
For modules relying on band extraction (`band_structure` and `dirac_cone`), the computation is strictly divided into two phases within the compute script:
1. **SCF Phase:** Unfrozen charge (`ifixcharge = 0`) on a sparse grid to converge the density matrix and generate the physical `CHARGES` file.
2. **Diagonalization Phase:** Frozen charge (`ifixcharge = 1`) on the target path/grid to extract eigenvalues without altering the converged physical state.

To execute the solver:
```bash
bash compute_band.sh
```

### Outputs
* **`eigen.dat`** - Contains raw energy bands.
* **`dens_TOT.dat`** - Contains the DOS profile.
* **`out_*.log`** - Execution logs, from which the script automatically extracts and prints the resulting $F_{\max}$ (maximum atomic force).