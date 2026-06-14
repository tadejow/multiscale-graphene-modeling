"""
Module responsible for the Empirical Risk Minimization (ERM) loop.
Handles GPU placement, loss computation, and metric tracking.
"""
import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from pathlib import Path


class MLPTrainer:
    """Class encapsulating the training and evaluation loop."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 device: torch.device, save_path: str):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = nn.MSELoss()  # L2 Tikhonov is handled via optimizer weight_decay
        self.device = device
        self.save_path = Path(save_path)

        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def train(self, train_loader: DataLoader, epochs: int) -> None:
        """Trains the model over the specified number of epochs."""
        self.model.train()
        logging.info(f"Starting training loop on device: {self.device}...")

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                avg_loss = total_loss / len(train_loader)
                logging.info(f"Epoch [{epoch + 1}/{epochs}] - MSE Loss: {avg_loss:.6f}")

        self._save_model()

    def evaluate(self, test_loader: DataLoader) -> None:
        """Evaluates the model on the test set and calculates R^2 score."""
        self.model.eval()
        all_targets = []
        all_preds = []
        total_mse = 0.0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_x)

                loss = self.criterion(predictions, batch_y)
                total_mse += loss.item()

                all_targets.extend(batch_y.cpu().numpy())
                all_preds.extend(predictions.cpu().numpy())

        avg_mse = total_mse / len(test_loader)
        r2 = r2_score(all_targets, all_preds)

        logging.info("--- Evaluation Results ---")
        logging.info(f"Test MSE: {avg_mse:.6f}")
        logging.info(f"Test R^2 Score: {r2:.4f}")

    def _save_model(self) -> None:
        """Saves the trained model parameters to disk."""
        torch.save(self.model.state_dict(), self.save_path)
        logging.info(f"Model successfully saved to {self.save_path}")