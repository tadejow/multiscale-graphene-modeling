"""
Main orchestration script.
Initializes configuration, prepares data, builds the model, and executes training.
"""
import logging
import torch
from torch.utils.data import random_split, DataLoader

from config_parser import ConfigParser
from data_loader import GrapheneFireballDataset
from model import ViscosityMLP
from trainer import MLPTrainer


def setup_logging():
    """Configures the ISO standard application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    setup_logging()
    logging.info("Initializing Multiscale Graphene ML Pipeline...")

    # 1. Load Configuration
    config_loader = ConfigParser("../configs/ml_model/mlp_config.yaml")
    cfg = config_loader.load()

    # 2. Hardware configuration (GPU Setup)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Compute device allocated: {device}")

    # 3. Data Extraction and Loader Preparation
    dataset = GrapheneFireballDataset(
        archive_path=cfg['paths']['data_archive'],
        extract_dir=cfg['paths']['extract_dir']
    )

    # Split dataset (Train / Test)
    test_size = int(cfg['training']['test_split'] * len(dataset))
    train_size = len(dataset) - test_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(cfg['training']['random_seed'])
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)

    # 4. Model Instantiation
    model = ViscosityMLP(
        input_dim=cfg['model']['input_dim'],
        hidden_dims=cfg['model']['hidden_dims'],
        output_dim=cfg['model']['output_dim']
    )

    # 5. Optimizer formulation (Adam with L2 Regularization / weight_decay)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )

    # 6. Empirical Risk Minimization (Training)
    trainer = MLPTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        save_path=cfg['paths']['model_save_path']
    )

    trainer.train(train_loader=train_loader, epochs=cfg['training']['epochs'])

    # 7. Model Evaluation
    trainer.evaluate(test_loader=test_loader)


if __name__ == "__main__":
    main()
