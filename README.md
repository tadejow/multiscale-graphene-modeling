# Multiscale Simulation of Electron Hydrodynamics in Graphene

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Abstract
This repository contains a multiscale computational framework designed to simulate electron hydrodynamics in strained graphene. The project bridges three distinct domains of physics and computer science: 
1. **Quantum Mechanics:** Tight-binding Density Functional Theory (TB-DFT) via Fireball.
2. **Computational Statistics:** Machine Learning surrogate models (MLP) trained via PyTorch.
3. **Fluid Dynamics:** Macroscopic continuous flows via FEniCS / Lattice Boltzmann Method.

## Repository Structure
Based on standard separation of concerns (ISO/IEC 25010):
- `project-graphene/configs/`: YAML configuration files separating hyperparameters from code logic.
- `project-graphene/pipeline_fireball/`: Scripts for local quantum geometry perturbation and TB-DFT eigenvalue extraction.
- `project-graphene/pipeline_ml/`: Object-Oriented PyTorch implementation of the MLP mapping function $f_{\text{ML}}: \mathcal{X} \to \mathcal{Y}$.
- `project-graphene/pipeline_fenics/`: Finite Element Method solvers for the electronic Navier-Stokes (Gurzhi) equations.
- `workshops/`: Archival codes and exercises from the IFD course.

## Installation
Ensure you have Python 3.12+ installed. 
```bash
# Clone the repository
git clone https://github.com/YOUR_GITHUB_USERNAME/puw-ifd-workshop.git
cd puw-ifd-workshop

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows

# Install dependencies (requirements.txt to be added)
pip install -r requirements.txt