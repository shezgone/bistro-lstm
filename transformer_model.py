"""
Task-Specific Transformer for Korean CPI Forecasting
=====================================================
LSTM과 동일한 Variable Fusion + Temporal Decoder 구조에서
LSTM Encoder만 Transformer Encoder로 교체.
파라미터 수를 LSTM과 유사하게 맞춤.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

# LSTM 모델에서 공유 컴포넌트 임포트
from lstm_model import VariableEmbedding, VariableFusionAttention, TemporalAttentionDecoder


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class AttentionTransformerForecaster(nn.Module):
    """
    Task-Specific Transformer for CPI Forecasting.

    LSTM과 동일한 구조에서 encoder만 Transformer로 교체:
        1. Variable Embedding (공유)
        2. Variable Fusion Attention (공유)
        3. Transformer Encoder (NEW - replaces LSTM)
        4. Temporal Attention Decoder (공유)
        5. Output Head (동일)
    """

    def __init__(
        self,
        n_vars: int,
        d_model: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        pred_len: int = 12,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len

        # 1. Variable Embedding (LSTM과 동일)
        self.var_embedding = VariableEmbedding(n_vars, d_model)

        # 2. Variable Fusion Attention (LSTM과 동일)
        self.var_fusion = VariableFusionAttention(d_model, n_heads, dropout)

        # 3. Transformer Encoder (LSTM 대체)
        self.pos_encoding = PositionalEncoding(d_model, max_len=512, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        # Projection: d_model → hidden_dim (for temporal decoder)
        self.proj = nn.Linear(d_model, hidden_dim)

        # 4. Temporal Attention Decoder (LSTM과 동일)
        self.temporal_decoder = TemporalAttentionDecoder(
            hidden_dim, pred_len, n_heads, dropout
        )

        # 5. Output Head (LSTM과 동일)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, 1)
        self.fc_log_sigma = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # 1. Variable Embedding
        embedded = self.var_embedding(x)  # (B, T, N, D)

        # 2. Variable Fusion
        fused = self.var_fusion(embedded)  # (B, T, D)

        # 3. Transformer Encoder
        fused = self.pos_encoding(fused)

        # Causal mask (forecasting은 미래 정보 사용 불가)
        seq_len = fused.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(fused.device)

        encoded = self.transformer_encoder(fused, mask=causal_mask)  # (B, T, D)
        encoded = self.proj(encoded)  # (B, T, H)

        # 4. Temporal Attention Decoding
        decoded = self.temporal_decoder(encoded)  # (B, pred_len, H)

        # 5. Output Head
        h = F.relu(self.fc_hidden(self.dropout(decoded)))
        mu = self.fc_mu(h).squeeze(-1)
        log_sigma = self.fc_log_sigma(h).squeeze(-1)

        result = {"mu": mu, "log_sigma": log_sigma}

        if return_attention:
            var_attn = self.var_fusion.get_attention_weights()
            temp_attn = self.temporal_decoder.get_attention_weights()
            if var_attn is not None:
                result["variable_attention"] = var_attn
            if temp_attn is not None:
                result["temporal_attention"] = temp_attn

        return result

    def predict(
        self,
        x: torch.Tensor,
        n_samples: int = 100,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            out = self.forward(x, return_attention=True)
            mu = out["mu"]
            sigma = torch.exp(out["log_sigma"]).clamp(min=1e-6)

            dist = torch.distributions.Normal(mu.unsqueeze(1), sigma.unsqueeze(1))
            samples = dist.sample((n_samples,)).squeeze(2).permute(1, 0, 2)

            ci_lo = torch.quantile(samples, 0.05, dim=1)
            ci_hi = torch.quantile(samples, 0.95, dim=1)

        result = {"mu": mu, "sigma": sigma, "samples": samples, "ci_lo": ci_lo, "ci_hi": ci_hi}
        for key in ["variable_attention", "temporal_attention"]:
            if key in out:
                result[key] = out[key]
        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config) -> "AttentionTransformerForecaster":
        return cls(
            n_vars=config.n_variates,
            d_model=config.d_model,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            pred_len=config.pred_len,
            dropout=config.dropout,
        )
