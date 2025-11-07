"""
Multi-Scale Transformer for time series forecasting.

Implements a Transformer that learns both fine-grained and coarse-grained
patterns through hierarchical attention and cross-scale fusion.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
from typing import Dict, Tuple, Optional
import sys
sys.path.append('..')
from metrics import compute_all_metrics, hierarchical_consistency_error


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


class FineScaleEncoder(nn.Module):
    """Fine-scale encoder for short-term patterns."""
    
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for fine-scale encoding.
        
        Args:
            x: Input tensor [batch, seq_len, 1]
            
        Returns:
            Encoded tensor [batch, seq_len, d_model]
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        x = self.dropout(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # [batch, seq_len, d_model]
        
        return x


class CoarseScaleEncoder(nn.Module):
    """Coarse-scale encoder for long-term patterns."""
    
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 downsample_factor: int = 3):
        super().__init__()
        
        self.d_model = d_model
        self.downsample_factor = downsample_factor
        
        # Input projection
        self.input_projection = nn.Linear(1, d_model)
        
        # Downsampling layer (average pooling)
        self.downsample = nn.AvgPool1d(
            kernel_size=downsample_factor,
            stride=downsample_factor
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for coarse-scale encoding.
        
        Args:
            x: Input tensor [batch, seq_len, 1]
            
        Returns:
            Encoded tensor [batch, downsampled_len, d_model]
        """
        # Downsample input
        x_down = self.downsample(x.transpose(1, 2)).transpose(1, 2)  # [batch, seq_len//factor, 1]
        
        # Project to d_model
        x_down = self.input_projection(x_down)
        x_down = self.dropout(x_down)
        
        # Add positional encoding
        x_down = self.pos_encoder(x_down)
        
        # Transformer encoding
        x_down = self.transformer_encoder(x_down)
        
        return x_down


class CrossScaleFusion(nn.Module):
    """Cross-scale fusion layer using cross-attention."""
    
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        # Cross-attention from fine to coarse
        self.fine_to_coarse_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention from coarse to fine
        self.coarse_to_fine_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norms
        self.norm_fine = nn.LayerNorm(d_model)
        self.norm_coarse = nn.LayerNorm(d_model)
        
        # Feed-forward networks
        self.ffn_fine = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.ffn_coarse = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm_ffn_fine = nn.LayerNorm(d_model)
        self.norm_ffn_coarse = nn.LayerNorm(d_model)
    
    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-scale fusion.
        
        Args:
            fine: Fine-scale features [batch, fine_len, d_model]
            coarse: Coarse-scale features [batch, coarse_len, d_model]
            
        Returns:
            Tuple of (fused_fine, fused_coarse)
        """
        # Fine to coarse attention (coarse queries fine)
        coarse_attn, _ = self.coarse_to_fine_attention(
            query=coarse,
            key=fine,
            value=fine
        )
        coarse_fused = self.norm_coarse(coarse + coarse_attn)
        coarse_fused = self.norm_ffn_coarse(coarse_fused + self.ffn_coarse(coarse_fused))
        
        # Coarse to fine attention (fine queries coarse)
        fine_attn, _ = self.fine_to_coarse_attention(
            query=fine,
            key=coarse,
            value=coarse
        )
        fine_fused = self.norm_fine(fine + fine_attn)
        fine_fused = self.norm_ffn_fine(fine_fused + self.ffn_fine(fine_fused))
        
        return fine_fused, coarse_fused


class HierarchicalDecoder(nn.Module):
    """Hierarchical decoder for multi-level predictions."""
    
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 forecast_horizon: int = 1):
        super().__init__()
        
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        
        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Prediction heads for each hierarchical level
        self.facility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, forecast_horizon)
        )
        
        self.sector_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, forecast_horizon)
        )
        
        self.national_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, forecast_horizon)
        )
        
        # Learnable query embeddings for decoder
        self.query_embed = nn.Parameter(torch.randn(1, forecast_horizon, d_model))
    
    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode and predict at multiple hierarchical levels.
        
        Args:
            fine: Fine-scale features [batch, fine_len, d_model]
            coarse: Coarse-scale features [batch, coarse_len, d_model]
            
        Returns:
            Dictionary with predictions for each level
        """
        batch_size = fine.size(0)
        
        # Expand query embeddings
        query = self.query_embed.expand(batch_size, -1, -1)
        
        # Concatenate fine and coarse as memory
        memory = torch.cat([fine, coarse], dim=1)  # [batch, fine_len + coarse_len, d_model]
        
        # Decode
        decoded = self.transformer_decoder(query, memory)  # [batch, forecast_horizon, d_model]
        
        # Aggregate over horizon dimension for prediction
        decoded_pooled = decoded.mean(dim=1)  # [batch, d_model]
        
        # Predict at each level
        facility_pred = self.facility_head(decoded_pooled).unsqueeze(-1)  # [batch, horizon, 1]
        sector_pred = self.sector_head(decoded_pooled).unsqueeze(-1)
        national_pred = self.national_head(decoded_pooled).unsqueeze(-1)
        
        return {
            'facility': facility_pred,
            'sector': sector_pred,
            'national': national_pred
        }


class MultiScaleTransformer(nn.Module):
    """Complete Multi-Scale Transformer model."""
    
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 downsample_factor: int = 3,
                 forecast_horizon: int = 1):
        super().__init__()
        
        # Encoders
        self.fine_encoder = FineScaleEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.coarse_encoder = CoarseScaleEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            downsample_factor=downsample_factor
        )
        
        # Cross-scale fusion
        self.fusion = CrossScaleFusion(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout
        )
        
        # Hierarchical decoder
        self.decoder = HierarchicalDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            forecast_horizon=forecast_horizon
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, 1]
            
        Returns:
            Dictionary with predictions at each hierarchical level
        """
        # Encode at multiple scales
        fine_features = self.fine_encoder(x)
        coarse_features = self.coarse_encoder(x)
        
        # Cross-scale fusion
        fine_fused, coarse_fused = self.fusion(fine_features, coarse_features)
        
        # Decode and predict
        predictions = self.decoder(fine_fused, coarse_fused)
        
        return predictions


if __name__ == "__main__":
    # Test Multi-Scale Transformer
    model = MultiScaleTransformer(
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        downsample_factor=3,
        forecast_horizon=1
    )
    
    # Test forward pass
    x = torch.randn(8, 12, 1)  # [batch, seq_len, features]
    predictions = model(x)
    
    print("Multi-Scale Transformer Test")
    print("="*50)
    print(f"Input shape: {x.shape}")
    for level, pred in predictions.items():
        print(f"{level} prediction shape: {pred.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nNumber of parameters: {num_params:,}")

