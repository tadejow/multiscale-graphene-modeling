"""
Module for data extraction, preprocessing, and dataset generation.
Builds the PyTorch Dataset for the Machine Learning model.
"""
import tarfile
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List
from torch.utils.data import Dataset


class GrapheneFireballDataset(Dataset):
    """
    PyTorch Dataset implementation for the Graphene Hydrodynamics ML model.
    Extracts the TAR archive and prepares feature vectors and targets.
    """

    def __init__(self, archive_path: str, extract_dir: str):
        """
        Args:
            archive_path (str): Path to the RESULTS.tgz file.
            extract_dir (str): Directory where files will be extracted.
        """
        self.archive_path = Path(archive_path)
        self.extract_dir = Path(extract_dir)
        self.features: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []

        self._prepare_data()

    def _prepare_data(self) -> None:
        """Extracts the archive and builds the dataset."""
        self._extract_archive()
        self._parse_files()

    def _extract_archive(self) -> None:
        """Extracts the .tgz archive if not already extracted."""
        if not self.extract_dir.exists():
            self.extract_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Extracting {self.archive_path} to {self.extract_dir}...")
            with tarfile.open(self.archive_path, "r:gz") as tar:
                tar.extractall(path=self.extract_dir)
        else:
            logging.info(f"Data already extracted at {self.extract_dir}")

    def _parse_files(self) -> None:
        """
        Parses Fireball output files to construct the input feature space X
        and target space Y as defined in the theoretical framework.
        """
        # TODO: Implement actual file parsing logic here.
        # This is a placeholder demonstrating the data structure defined in Eq. (41).
        # x = [d1, d2, d3, v_F, D(E_F)]
        # y = [kinematic_viscosity_nu]

        logging.info("Parsing Fireball outputs to construct X and Y spaces...")

        # MOCK DATA GENERATION FOR PIPELINE TESTING
        # To be replaced with actual physical parsing of eigen.dat and dens_TOT.dat
        num_samples = 100
        for _ in range(num_samples):
            # Mock features: d1, d2, d3 (angstroms), v_F (m/s), D(E_F) (states/eV)
            d1, d2, d3 = np.random.normal(1.42, 0.05, 3)
            v_f = np.random.normal(1e6, 0.1e6)
            d_ef = np.random.normal(0.02, 0.005)

            x_tensor = torch.tensor([d1, d2, d3, v_f, d_ef], dtype=torch.float32)

            # Mock target: Kinematic viscosity nu
            nu = self._calculate_theoretical_viscosity(v_f, d_ef)
            y_tensor = torch.tensor([nu], dtype=torch.float32)

            self.features.append(x_tensor)
            self.targets.append(y_tensor)

        logging.info(f"Successfully constructed dataset with {len(self.features)} samples.")

    @staticmethod
    def _calculate_theoretical_viscosity(v_f: float, d_ef: float) -> float:
        """
        Physical bridge: computes the target viscosity from quantum features
        using Chapman-Enskog expansion principles.
        """
        # Placeholder for physical calculation
        return (v_f ** 2) * d_ef * 1e-12

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]
