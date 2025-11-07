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
import subprocess

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.lstm_model import LSTMLightningModule
from utils.data_loader import create_dataloaders


def train_lstm(config_path: str):
    """Train LSTM baseline model."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("Training LSTM Baseline")
    print("="*60)
    
    # Set random seeds
    pl.seed_everything(config['reproducibility']['seed'])
    
    print(f"\nData configuration:")
    print(f"  Data directory: {config['data']['data_dir']}")
    print(f"  Sequence length: {config['data']['sequence_length']}")
    print(f"  Forecast horizon: {config['data']['forecast_horizon']}")
    print(f"  Batch size: {config['data']['batch_size']}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
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
    
    print(f"\nModel configuration:")
    print(f"  Hidden size: {config['model']['hidden_size']}")
    print(f"  Num layers: {config['model']['num_layers']}")
    print(f"  Dropout: {config['model']['dropout']}")
    print(f"  Bidirectional: {config['model']['bidirectional']}")
    
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
    test_results = trainer.test(model, dataloaders['test'])
    
    # Save test metrics
    if test_results:
        import json
        from pathlib import Path
        
        results_dir = Path(config['logging']['checkpoint_dir']).parent / 'metrics'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_path = results_dir / 'lstm_test_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'test_metrics': test_results[0],
                'best_model_path': str(checkpoint_callback.best_model_path),
                'config': config
            }, f, indent=2)
        
        print(f"\nTest metrics saved to: {metrics_path}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print("="*60)
    
    # Run detailed evaluation
    print("\n" + "="*60)
    print("Running Detailed Evaluation (Facility/Sector/National)")
    print("="*60)
    
    try:
        result = subprocess.run(
            [sys.executable, 'src/baselines/evaluate_lstm_detailed.py'],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print("Warning: Detailed evaluation had some issues:")
            print(result.stderr)
    except Exception as e:
        print(f"Warning: Could not run detailed evaluation: {e}")
        print("You can run it manually with: python src/baselines/evaluate_lstm_detailed.py")


def main():
    parser = argparse.ArgumentParser(description='Train LSTM Baseline')
    parser.add_argument('--config', type=str, default='configs/lstm_config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    train_lstm(args.config)


if __name__ == "__main__":
    main()

