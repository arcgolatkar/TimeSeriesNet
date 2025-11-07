"""
Training script for Multi-Scale Transformer.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
import argparse
from pathlib import Path
import sys
sys.path.append('..')

from multi_scale_transformer import MultiScaleTransformer
from data_loader import create_dataloaders
from metrics import compute_all_metrics


class MSTLightningModule(pl.LightningModule):
    """PyTorch Lightning module for Multi-Scale Transformer."""
    
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 downsample_factor: int = 3,
                 forecast_horizon: int = 1,
                 learning_rate: float = 0.0001,
                 weight_decay: float = 0.01,
                 loss_weights: dict = None):
        """Initialize Lightning module."""
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = MultiScaleTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            downsample_factor=downsample_factor,
            forecast_horizon=forecast_horizon
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Loss weights for multi-scale learning
        if loss_weights is None:
            loss_weights = {
                'fine_level': 1.0,
                'coarse_level': 0.5,
                'hierarchical_consistency': 0.3
            }
        self.loss_weights = loss_weights
        
        # For tracking outputs
        self.validation_outputs = []
        self.test_outputs = []
    
    def forward(self, x: torch.Tensor) -> dict:
        return self.model(x)
    
    def compute_loss(self, predictions: dict, batch: dict) -> dict:
        """
        Compute multi-scale loss.
        
        Returns:
            Dictionary with total loss and component losses
        """
        # Facility-level loss (fine-grained)
        facility_loss = self.criterion(
            predictions['facility'],
            batch['facility_target']
        )
        
        # Sector-level loss (coarse-grained)
        sector_loss = self.criterion(
            predictions['sector'],
            batch['sector_target']
        )
        
        # National-level loss
        national_loss = self.criterion(
            predictions['national'],
            batch['national_target']
        )
        
        # Hierarchical consistency loss
        # Ensure predictions are consistent across levels
        # (This is a simplified version - can be enhanced)
        consistency_loss = torch.abs(
            predictions['facility'].mean() - predictions['sector'].mean()
        ) + torch.abs(
            predictions['sector'].mean() - predictions['national'].mean()
        )
        
        # Weighted total loss
        total_loss = (
            self.loss_weights['fine_level'] * facility_loss +
            self.loss_weights['coarse_level'] * (sector_loss + national_loss) / 2 +
            self.loss_weights['hierarchical_consistency'] * consistency_loss
        )
        
        return {
            'total_loss': total_loss,
            'facility_loss': facility_loss,
            'sector_loss': sector_loss,
            'national_loss': national_loss,
            'consistency_loss': consistency_loss
        }
    
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        predictions = self(batch['facility_input'])
        losses = self.compute_loss(predictions, batch)
        
        # Log losses
        self.log('train_loss', losses['total_loss'], prog_bar=True)
        self.log('train_facility_loss', losses['facility_loss'])
        self.log('train_sector_loss', losses['sector_loss'])
        self.log('train_national_loss', losses['national_loss'])
        self.log('train_consistency_loss', losses['consistency_loss'])
        
        return losses['total_loss']
    
    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """Validation step."""
        predictions = self(batch['facility_input'])
        losses = self.compute_loss(predictions, batch)
        
        # Log losses
        self.log('val_loss', losses['total_loss'], prog_bar=True)
        self.log('val_facility_loss', losses['facility_loss'])
        self.log('val_sector_loss', losses['sector_loss'])
        self.log('val_national_loss', losses['national_loss'])
        
        # Store predictions for metrics
        self.validation_outputs.append({
            'facility_pred': predictions['facility'].detach().cpu(),
            'facility_true': batch['facility_target'].detach().cpu(),
            'sector_pred': predictions['sector'].detach().cpu(),
            'sector_true': batch['sector_target'].detach().cpu(),
            'national_pred': predictions['national'].detach().cpu(),
            'national_true': batch['national_target'].detach().cpu(),
        })
        
        return losses
    
    def on_validation_epoch_end(self):
        """Compute metrics at end of validation."""
        if not self.validation_outputs:
            return
        
        # Concatenate predictions for each level
        facility_pred = torch.cat([x['facility_pred'] for x in self.validation_outputs])
        facility_true = torch.cat([x['facility_true'] for x in self.validation_outputs])
        sector_pred = torch.cat([x['sector_pred'] for x in self.validation_outputs])
        sector_true = torch.cat([x['sector_true'] for x in self.validation_outputs])
        national_pred = torch.cat([x['national_pred'] for x in self.validation_outputs])
        national_true = torch.cat([x['national_true'] for x in self.validation_outputs])
        
        # Compute metrics for each level
        facility_metrics = compute_all_metrics(facility_true, facility_pred)
        sector_metrics = compute_all_metrics(sector_true, sector_pred)
        national_metrics = compute_all_metrics(national_true, national_pred)
        
        # Log facility metrics
        for name, value in facility_metrics.items():
            self.log(f'val_facility_{name}', value, prog_bar=(name == 'MAE'))
        
        # Log sector metrics
        for name, value in sector_metrics.items():
            self.log(f'val_sector_{name}', value)
        
        # Log national metrics
        for name, value in national_metrics.items():
            self.log(f'val_national_{name}', value)
        
        self.validation_outputs.clear()
    
    def test_step(self, batch: dict, batch_idx: int) -> dict:
        """Test step."""
        predictions = self(batch['facility_input'])
        losses = self.compute_loss(predictions, batch)
        
        self.test_outputs.append({
            'facility_pred': predictions['facility'].detach().cpu(),
            'facility_true': batch['facility_target'].detach().cpu(),
            'sector_pred': predictions['sector'].detach().cpu(),
            'sector_true': batch['sector_target'].detach().cpu(),
            'national_pred': predictions['national'].detach().cpu(),
            'national_true': batch['national_target'].detach().cpu(),
        })
        
        return losses
    
    def on_test_epoch_end(self):
        """Compute final test metrics."""
        if not self.test_outputs:
            return
        
        facility_pred = torch.cat([x['facility_pred'] for x in self.test_outputs])
        facility_true = torch.cat([x['facility_true'] for x in self.test_outputs])
        sector_pred = torch.cat([x['sector_pred'] for x in self.test_outputs])
        sector_true = torch.cat([x['sector_true'] for x in self.test_outputs])
        national_pred = torch.cat([x['national_pred'] for x in self.test_outputs])
        national_true = torch.cat([x['national_true'] for x in self.test_outputs])
        
        facility_metrics = compute_all_metrics(facility_true, facility_pred)
        sector_metrics = compute_all_metrics(sector_true, sector_pred)
        national_metrics = compute_all_metrics(national_true, national_pred)
        
        for name, value in facility_metrics.items():
            self.log(f'test_facility_{name}', value)
        for name, value in sector_metrics.items():
            self.log(f'test_sector_{name}', value)
        for name, value in national_metrics.items():
            self.log(f'test_national_{name}', value)
        
        self.test_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.learning_rate * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


def train_mst(config_path: str):
    """Train Multi-Scale Transformer."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*50)
    print("Training Multi-Scale Transformer")
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
        hierarchical=True
    )
    
    # Create model
    model = MSTLightningModule(
        d_model=config['model']['fine_encoder']['d_model'],
        nhead=config['model']['fine_encoder']['nhead'],
        num_encoder_layers=config['model']['fine_encoder']['num_layers'],
        num_decoder_layers=config['model']['decoder']['num_layers'],
        dim_feedforward=config['model']['fine_encoder']['dim_feedforward'],
        dropout=config['model']['fine_encoder']['dropout'],
        downsample_factor=config['model']['coarse_encoder']['downsample_factor'],
        forecast_horizon=config['data']['forecast_horizon'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        loss_weights=config['training']['loss_weights']
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['logging']['checkpoint_dir'],
        filename='mst-{epoch:02d}-{val_loss:.4f}',
        monitor=config['logging']['monitor'],
        mode=config['logging']['mode'],
        save_top_k=config['logging']['save_top_k']
    )
    
    early_stopping = EarlyStopping(
        monitor=config['logging']['monitor'],
        patience=config['training']['early_stopping_patience'],
        mode=config['logging']['mode']
    )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config['logging']['log_dir'],
        name=config['logging']['experiment_name']
    )
    
    # Trainer
    # trainer = pl.Trainer(
    #     max_epochs=config['training']['max_epochs'],
    #     accelerator='auto',
    #     devices=1,
    #     callbacks=[checkpoint_callback, early_stopping],
    #     logger=logger,
    #     gradient_clip_val=config['training']['gradient_clip_val'],
    #     deterministic=config['reproducibility']['deterministic']
    # )
    num_gpus = torch.cuda.device_count()

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu' if num_gpus > 0 else 'cpu',
        devices=num_gpus if num_gpus > 0 else 1,   # e.g., 1 or all GPUs
        precision='16-mixed' if num_gpus > 0 else '32-true',  # AMP on GPU
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        gradient_clip_val=config['training']['gradient_clip_val'],
        deterministic=config['reproducibility']['deterministic']
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
    parser = argparse.ArgumentParser(description='Train Multi-Scale Transformer')
    parser.add_argument('--config', type=str, default='configs/mst_config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    train_mst(args.config)


if __name__ == "__main__":
    main()

