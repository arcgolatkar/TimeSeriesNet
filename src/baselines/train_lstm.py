"""
Training script for LSTM baseline.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
import argparse
from pathlib import Path
import sys
sys.path.append('..')

from baselines.lstm_model import LSTMLightningModule
from utils.data_loader import create_dataloaders


def train_lstm(config_path: str):
    """Train LSTM baseline model."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*50)
    print("Training LSTM Baseline")
    print("="*50)
    
    # Set random seeds
    pl.seed_everything(config['reproducibility']['seed'])
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_dir=config['data']['data_dir'],
        sequence_length=config['data']['sequence_length'],
        forecast_horizon=config['data']['forecast_horizon'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        train_years=tuple(config['data']['train_years']),
        val_years=tuple(config['data']['val_years']),
        test_years=tuple(config['data']['test_years']),
        hierarchical=True  # Use hierarchical data but only predict facility level
    )
    
    # Create model
    model = LSTMLightningModule(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        bidirectional=config['model']['bidirectional'],
        forecast_horizon=config['data']['forecast_horizon'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['logging']['checkpoint_dir'],
        filename='lstm-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['training']['early_stopping_patience'],
        mode='min'
    )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config['logging']['log_dir'],
        name=config['logging']['experiment_name']
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        gradient_clip_val=config['training']['gradient_clip_val']
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, dataloaders['train'], dataloaders['val'])
    
    # Test
    print("\nEvaluating on test set...")
    trainer.test(model, dataloaders['test'])
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Train LSTM Baseline')
    parser.add_argument('--config', type=str, default='configs/lstm_config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    train_lstm(args.config)


if __name__ == "__main__":
    main()

