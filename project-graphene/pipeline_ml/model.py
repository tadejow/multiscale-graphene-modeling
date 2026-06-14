"""
Module containing the Neural Network architectures.
Follows the Open/Closed Principle - easy to extend with new models.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List


class BaseSurrogateModel(nn.Module, ABC):
    """Abstract Base Class for all ML surrogate models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class ViscosityMLP(BaseSurrogateModel):
    """
    Multi-Layer Perceptron (MLP) for predicting kinematic viscosity.
    Maps physical features X to macroscopic parameter Y.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super(ViscosityMLP, self).__init__()

        layers = []
        current_dim = input_dim

        # Build hidden layers dynamically
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())  # Non-linear activation (Eq. 42)
            current_dim = h_dim

        # Final output layer (affine transformation)
        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Applies Xavier/Glorot initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation through the network.

        Args:
            x (torch.Tensor): Input feature vector.

        Returns:
            torch.Tensor: Predicted kinematic viscosity.
        """
        return self.network(x)
    