"""
BISTRO-LSTM Model — Attention-Augmented Stacked LSTM
====================================================
한국 CPI YoY 예측을 위한 LSTM 아키텍처.
BISTRO Transformer(91M params)와의 공정 비교를 위해
cross-variable attention과 temporal attention을 포함.

Architecture (~2M params):
    Variable Embedding → Variable Fusion Attention → Stacked LSTM → Temporal Attention Decoder → Output Head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class VariableEmbedding(nn.Module):
    """
    Per-variable linear projection.
    각 변수를 독립적인 learned embedding으로 변환.

    Input:  (batch, seq_len, n_vars)
    Output: (batch, seq_len, n_vars, d_model)
    """

    def __init__(self, n_vars: int, d_model: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_vars)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_vars)
        parts = []
        for i, proj in enumerate(self.projections):
            # (batch, seq_len, 1) → (batch, seq_len, d_model)
            parts.append(proj(x[:, :, i:i+1]))
        # (batch, seq_len, n_vars, d_model)
        return torch.stack(parts, dim=2)


class VariableFusionAttention(nn.Module):
    """
    Cross-variable attention per timestep.
    각 시점에서 변수 간 상호작용을 학습.

    Input:  (batch, seq_len, n_vars, d_model)
    Output: (batch, seq_len, d_model)

    Stores attention weights for interpretability.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # 저장용: forward에서 업데이트
        self._attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_vars, d_model)
        B, T, N, D = x.shape

        # 각 시점별로 변수 간 attention 수행
        # Reshape to (B*T, N, D) for efficient batched attention
        x_flat = x.reshape(B * T, N, D)

        Q = self.W_q(x_flat)  # (B*T, N, D)
        K = self.W_k(x_flat)
        V = self.W_v(x_flat)

        # Multi-head reshape: (B*T, N, D) → (B*T, n_heads, N, head_dim)
        Q = Q.view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B*T, n_heads, N, N)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 저장 (head 평균, 시점 평균)
        # (B*T, n_heads, N, N) → (B, T, n_heads, N, N) → (B, N, N)
        self._attn_weights = (
            attn_weights
            .detach()
            .view(B, T, self.n_heads, N, N)
            .mean(dim=(1, 2))  # 시점·head 평균
        )

        # Apply attention
        context = torch.matmul(attn_weights, V)  # (B*T, n_heads, N, head_dim)
        context = context.transpose(1, 2).contiguous().view(B * T, N, D)
        output = self.W_o(context)  # (B*T, N, D)

        # Residual + LayerNorm, then aggregate across variables
        output = self.layer_norm(output + x_flat)
        output = output.view(B, T, N, D)

        # 변수 차원 aggregation: attention-weighted mean
        # 타겟 변수(idx 0)의 attended representation을 주로 사용하되,
        # 전체 변수 평균도 결합
        target_repr = output[:, :, 0, :]   # (B, T, D) — 타겟 시점
        mean_repr = output.mean(dim=2)      # (B, T, D) — 전체 평균

        # 타겟 중심 + 전체 정보 결합
        fused = target_repr + mean_repr     # (B, T, D)

        return fused

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """최근 forward의 attention weights 반환. (B, N, N)"""
        return self._attn_weights


class TemporalAttentionDecoder(nn.Module):
    """
    Temporal Attention Decoder.
    Learnable forecast queries가 LSTM hidden states에 attend.

    Input:  LSTM hidden states (batch, seq_len, hidden_dim)
    Output: (batch, pred_len, hidden_dim)

    Stores temporal attention weights for interpretability.
    """

    def __init__(
        self,
        hidden_dim: int,
        pred_len: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0

        # Learnable forecast queries (하나의 query per forecast month)
        self.forecast_queries = nn.Parameter(
            torch.randn(pred_len, hidden_dim) * 0.02
        )

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._attn_weights: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, T, H)
        B, T, H = hidden_states.shape

        # Expand queries to batch: (pred_len, H) → (B, pred_len, H)
        queries = self.forecast_queries.unsqueeze(0).expand(B, -1, -1)

        Q = self.W_q(queries)        # (B, pred_len, H)
        K = self.W_k(hidden_states)  # (B, T, H)
        V = self.W_v(hidden_states)  # (B, T, H)

        # Multi-head reshape
        Q = Q.view(B, self.pred_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, n_heads, pred_len, T)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 저장: head 평균
        self._attn_weights = attn_weights.detach().mean(dim=1)  # (B, pred_len, T)

        context = torch.matmul(attn_weights, V)  # (B, n_heads, pred_len, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, self.pred_len, H)
        output = self.W_o(context)
        output = self.layer_norm(output + queries)

        return output  # (B, pred_len, H)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """최근 forward의 temporal attention weights. (B, pred_len, seq_len)"""
        return self._attn_weights


class AttentionLSTMForecaster(nn.Module):
    """
    Attention-Augmented Stacked LSTM for Macroeconomic Forecasting.

    Architecture:
        1. Variable Embedding: per-variable linear projection
        2. Variable Fusion Attention: cross-variable attention per timestep
        3. Stacked LSTM Encoder: temporal feature extraction
        4. Temporal Attention Decoder: forecast query → past hidden states
        5. Output Head: point forecast + uncertainty (Gaussian NLL)

    Parameters
    ----------
    n_vars    : 입력 변수 개수
    d_model   : 변수 임베딩 차원 (default 128)
    hidden_dim: LSTM hidden dimension (default 256)
    n_layers  : LSTM 레이어 수 (default 3)
    n_heads   : Attention head 수 (default 4)
    pred_len  : 예측 기간 (default 12)
    dropout   : Dropout rate (default 0.2)
    """

    def __init__(
        self,
        n_vars: int,
        d_model: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        pred_len: int = 12,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.pred_len = pred_len

        # 1. Variable Embedding
        self.var_embedding = VariableEmbedding(n_vars, d_model)

        # 2. Variable Fusion Attention
        self.var_fusion = VariableFusionAttention(d_model, n_heads, dropout)

        # 3. Stacked LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # 4. Temporal Attention Decoder
        self.temporal_decoder = TemporalAttentionDecoder(
            hidden_dim, pred_len, n_heads, dropout
        )

        # 5. Output Head
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, 1)        # Point forecast
        self.fc_log_sigma = nn.Linear(hidden_dim // 2, 1)  # Uncertainty

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : (batch, seq_len, n_vars) 입력 시퀀스
        return_attention : attention weights를 결과에 포함할지

        Returns
        -------
        dict with keys:
            mu       : (batch, pred_len) point forecast
            log_sigma: (batch, pred_len) log standard deviation
            variable_attention  : (batch, n_vars, n_vars) if return_attention
            temporal_attention  : (batch, pred_len, seq_len) if return_attention
        """
        # 1. Variable Embedding
        embedded = self.var_embedding(x)  # (B, T, N, D)

        # 2. Variable Fusion
        fused = self.var_fusion(embedded)  # (B, T, D)

        # 3. LSTM Encoding
        lstm_out, _ = self.lstm(fused)  # (B, T, H)

        # 4. Temporal Attention Decoding
        decoded = self.temporal_decoder(lstm_out)  # (B, pred_len, H)

        # 5. Output Head
        h = F.relu(self.fc_hidden(self.dropout(decoded)))
        mu = self.fc_mu(h).squeeze(-1)              # (B, pred_len)
        log_sigma = self.fc_log_sigma(h).squeeze(-1)  # (B, pred_len)

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
        """
        예측 + 불확실성 추정.

        Returns
        -------
        dict with keys:
            mu       : (batch, pred_len)
            sigma    : (batch, pred_len)
            samples  : (batch, n_samples, pred_len) sampled forecasts
            ci_lo    : (batch, pred_len) 5th percentile
            ci_hi    : (batch, pred_len) 95th percentile
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x, return_attention=True)
            mu = out["mu"]
            sigma = torch.exp(out["log_sigma"]).clamp(min=1e-6)

            # Sample from Gaussian
            dist = torch.distributions.Normal(mu.unsqueeze(1), sigma.unsqueeze(1))
            samples = dist.sample((n_samples,)).squeeze(2).permute(1, 0, 2)
            # samples: (batch, n_samples, pred_len)

            ci_lo = torch.quantile(samples, 0.05, dim=1)
            ci_hi = torch.quantile(samples, 0.95, dim=1)

        result = {
            "mu": mu,
            "sigma": sigma,
            "samples": samples,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }

        # Attention weights
        for key in ["variable_attention", "temporal_attention"]:
            if key in out:
                result[key] = out[key]

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config) -> "AttentionLSTMForecaster":
        """LSTMConfig에서 모델 생성."""
        return cls(
            n_vars=config.n_variates,
            d_model=config.d_model,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            pred_len=config.pred_len,
            dropout=config.dropout,
        )
