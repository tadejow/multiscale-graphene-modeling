"""
Module responsible for the Empirical Risk Minimization (ERM) loop.
Handles GPU placement, loss computation, and saving logs/metrics.
"""
import torch
import torch.nn as nn
import logging
import csv
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from pathlib import Path

class MLPTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 device: torch.device, out_dir: str):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = nn.MSELoss()
        self.device = device

        self.out_dir = Path(out_dir)
        self.model_save_path = self.out_dir / "viscosity_mlp.pth"
        self.metrics_path = self.out_dir / "training_metrics.csv"

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> None:
        logging.info(f"Starting ERM loop on {self.device}...")
        loss_history = []

        for epoch in range(epochs):
            # --- FAZA TRENINGU ---
            self.model.train()
            total_train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)

                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # --- FAZA WALIDACJI (EWALUACJI) W KAZDEJ EPOCE ---
            self.model.eval()
            total_val_loss = 0.0
            all_targets, all_preds = [], []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    val_preds = self.model(batch_x)
                    val_loss = self.criterion(val_preds, batch_y)

                    total_val_loss += val_loss.item()
                    all_targets.extend(batch_y.cpu().numpy())
                    all_preds.extend(val_preds.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            val_r2 = r2_score(all_targets, all_preds)

            # Zapis historii
            loss_history.append({
                'epoch': epoch + 1,
                'train_mse': avg_train_loss,
                'val_mse': avg_val_loss,
                'val_r2': val_r2
            })

            # Logowanie co 50 epok
            if (epoch + 1) % 50 == 0 or epoch == 0:
                logging.info(f"Epoch [{epoch+1:3d}/{epochs}] | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f} | Val R^2: {val_r2:.4f}")

        self._save_metrics(loss_history)
        self._save_model()

    def evaluate(self, test_loader: DataLoader) -> None:
        """Final evaluation on test set."""
        self.model.eval()
        all_targets, all_preds = [], []
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

        logging.info("--- Final Evaluation Results ---")
        logging.info(f"Test MSE: {avg_mse:.6f}")
        logging.info(f"Test R^2 Score: {r2:.4f}")

    def _save_metrics(self, history: list) -> None:
        with open(self.metrics_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'train_mse', 'val_mse', 'val_r2'])
            writer.writeheader()
            writer.writerows(history)
        logging.info(f"Training metrics saved to {self.metrics_path}")

    def _save_model(self) -> None:
        torch.save(self.model.state_dict(), self.model_save_path)
        logging.info(f"Model weights saved to {self.model_save_path}")