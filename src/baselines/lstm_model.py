"""
LSTM baseline model for time series forecasting.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Tuple
import sys
sys.path.append('..')
from utils.metrics import compute_all_metrics


class LSTMForecaster(nn.Module):
    """LSTM model for time series forecasting."""
    
    def __init__(self,
                 input_size: int = 1,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 bidirectional: bool = False,
                 forecast_horizon: int = 1):
        """
        Initialize LSTM forecaster.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            forecast_horizon: Number of steps to forecast
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.forecast_horizon = forecast_horizon
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output projection
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, forecast_horizon * input_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, input_size]
            
        Returns:
            Predictions [batch, forecast_horizon, input_size]
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_final = h_n[-1]
        
        # Dropout and projection
        h_final = self.dropout(h_final)
        output = self.fc(h_final)
        
        # Reshape to [batch, forecast_horizon, input_size]
        output = output.view(-1, self.forecast_horizon, 1)
        
        return output


class LSTMLightningModule(pl.LightningModule):
    """PyTorch Lightning module for LSTM training."""
    
    def __init__(self,
                 input_size: int = 1,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 bidirectional: bool = False,
                 forecast_horizon: int = 1,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0001):
        """Initialize Lightning module."""
        super().__init__()
        self.save_hyperparameters()
        
        self.model = LSTMForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            forecast_horizon=forecast_horizon
        )
        
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # For tracking predictions
        self.validation_outputs = []
        self.test_outputs = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Handle both simple and hierarchical datasets
        if 'input' in batch:
            x = batch['input']
            y = batch['target']
        else:
            x = batch['facility_input']
            y = batch['facility_target']
        
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Validation step."""
        if 'input' in batch:
            x = batch['input']
            y = batch['target']
        else:
            x = batch['facility_input']
            y = batch['facility_target']
        
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store predictions for epoch-end metrics
        self.validation_outputs.append({
            'y_true': y.detach().cpu(),
            'y_pred': y_pred.detach().cpu()
        })
        
        return {'val_loss': loss}
    
    def on_validation_epoch_end(self):
        """Compute metrics at end of validation epoch."""
        if not self.validation_outputs:
            return
        
        # Concatenate all predictions
        y_true = torch.cat([x['y_true'] for x in self.validation_outputs])
        y_pred = torch.cat([x['y_pred'] for x in self.validation_outputs])
        
        # Compute metrics
        metrics = compute_all_metrics(y_true, y_pred)
        
        for name, value in metrics.items():
            self.log(f'val_{name}', value, prog_bar=(name in ['MAE', 'sMAPE']))
        
        self.validation_outputs.clear()
    
    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Test step."""
        if 'input' in batch:
            x = batch['input']
            y = batch['target']
        else:
            x = batch['facility_input']
            y = batch['facility_target']
        
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        
        self.test_outputs.append({
            'y_true': y.detach().cpu(),
            'y_pred': y_pred.detach().cpu()
        })
        
        return {'test_loss': loss}
    
    def on_test_epoch_end(self):
        """Compute metrics at end of test epoch."""
        if not self.test_outputs:
            return
        
        y_true = torch.cat([x['y_true'] for x in self.test_outputs])
        y_pred = torch.cat([x['y_pred'] for x in self.test_outputs])
        
        metrics = compute_all_metrics(y_true, y_pred)
        
        for name, value in metrics.items():
            self.log(f'test_{name}', value)
        
        self.test_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }


if __name__ == "__main__":
    # Test model
    model = LSTMForecaster(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        forecast_horizon=1
    )
    
    # Test forward pass
    x = torch.randn(32, 10, 1)  # [batch, seq_len, features]
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

