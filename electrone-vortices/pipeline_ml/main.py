"""
Main orchestration script.
Initializes configuration, prepares data, builds the model, and executes training.
"""
import logging
import torch
from pathlib import Path
from torch.utils.data import random_split, DataLoader

from config_parser import ConfigParser
from data_loader import GrapheneFireballDataset
from model import ViscosityMLP
from trainer import MLPTrainer

def setup_logging(out_dir: str):
    """Configures logging to both terminal and a file in data_learning."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(out_dir) / "pipeline_execution.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True  # Ratuje logowanie przed konfliktem z ustawieniami domyślnymi
    )

def main():
    # 1. Load Configuration First
    config_loader = ConfigParser("../configs/ml_model/mlp_config.yaml")
    cfg = config_loader.load()

    # 2. Setup Logging
    out_dir = cfg['paths']['learning_output_dir']
    setup_logging(out_dir)
    logging.info("Initializing Multiscale Graphene ML Pipeline...")

    # 3. Hardware configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Compute device allocated: {device}")

    # 4. Data Extraction & Scaling
    dataset = GrapheneFireballDataset(
        archive_path=cfg['paths']['data_archive'],
        extract_dir=cfg['paths']['extract_dir']
    )

    # Split dataset
    test_size = int(cfg['training']['test_split'] * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(cfg['training']['random_seed'])
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)

    # 5. Model Setup
    model = ViscosityMLP(
        input_dim=cfg['model']['input_dim'],
        hidden_dims=cfg['model']['hidden_dims'],
        output_dim=cfg['model']['output_dim']
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )

    # 6. Training & Evaluation
    trainer = MLPTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        out_dir=out_dir
    )

    trainer.train(train_loader=train_loader, val_loader=test_loader, epochs=cfg['training']['epochs'])
    trainer.evaluate(test_loader=test_loader)


if __name__ == "__main__":
    main()
