"""
BISTRO-LSTM Core — Domain Classes
==================================
bistro-xai의 bistro_core.py를 미러링하되 LSTM에 맞게 적응.
Streamlit 앱과 리포트가 공유하는 도메인 계층.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Variable Presets (bistro-xai와 동일)
# ============================================================

PRESETS: Dict[str, List[str]] = {
    "Korean CPI 6-var": [
        "CPI_KR_YoY", "Oil_WTI", "USD_KRW", "Oil_Brent", "Copper", "Wheat"
    ],
    "Korean CPI 10-var": [
        "CPI_KR_YoY", "Oil_WTI", "USD_KRW", "Oil_Brent", "Copper", "Wheat",
        "FedFunds", "US_Unemp", "US_ConsConf", "KR_Exports"
    ],
    "Full 29-var": [
        "CPI_KR_YoY",
        # Tier 1: 환율, 원자재
        "CNY_USD", "JPY_USD", "USD_KRW", "DXY_Broad",
        "Oil_Brent", "Oil_WTI", "NatGas_HH", "Gold", "Copper", "Wheat", "Corn",
        # Tier 2: 수요 측 / 실물경제
        "KR_Exports", "KR_Imports", "KR_Unemp", "US_Unemp", "US_ConsConf",
        # Tier 3: 통화/금융
        "Rate_KR", "Rate_ECB", "FedFunds",
        "KR_Interbank3M", "KR_LongRate", "JP_Interbank3M",
        "US_M2", "US_YieldSpread", "VIX",
        # Tier 4: 글로벌 물가
        "CPI_US_YoY", "CPI_XM_YoY", "China_CPI",
    ],
    "Custom": [],
}

TIER_LABELS: Dict[str, str] = {
    # Tier 1: 직접 가격 전달 (환율, 원자재, 물가지수)
    "CPI_KR_YoY": "T1",
    "Oil_WTI": "T1", "Oil_Brent": "T1", "NatGas_HH": "T1",
    "USD_KRW": "T1", "CNY_USD": "T1", "JPY_USD": "T1", "DXY_Broad": "T1",
    "Copper": "T1", "Wheat": "T1", "Corn": "T1", "Gold": "T1",
    # Tier 2: 수요 측 / 실물경제
    "KR_Exports": "T2", "KR_Imports": "T2", "KR_Unemp": "T2",
    "US_Unemp": "T2", "US_ConsConf": "T2",
    # Tier 3: 통화/금융
    "Rate_KR": "T3", "Rate_ECB": "T3", "FedFunds": "T3",
    "KR_Interbank3M": "T3", "KR_LongRate": "T3", "JP_Interbank3M": "T3",
    "US_M2": "T3", "US_YieldSpread": "T3", "VIX": "T3",
    # Tier 4: 글로벌 물가/교역
    "CPI_US_YoY": "T4", "CPI_XM_YoY": "T4", "China_CPI": "T4",
    "US_CoreCPI_idx": "T4", "US_PPI": "T4",
    # Extended variables (optimal18 model)
    "AUD_USD": "T1", "THB_USD": "T1", "JP_REER": "T1",
    "CN_Interbank3M": "T3", "US_Mortgage15Y": "T3",
    "UK_10Y_Bond": "T3", "US_DepInstCredit": "T3", "KC_FSI": "T3",
    "US_UnempRate": "T2", "US_NFP": "T2", "US_TradeTransEmp": "T2",
    "KR_MfgProd": "T2",
    "JP_CoreCPI": "T4", "CN_PPI": "T4", "US_ExportPI": "T4",
    "Pork": "T1", "PPI_CopperNickel": "T1",
}

VARIABLE_FREQ: Dict[str, str] = {
    # Daily
    "USD_KRW": "daily", "JPY_USD": "daily", "CNY_USD": "daily", "DXY_Broad": "daily",
    "Oil_Brent": "daily", "NatGas_HH": "daily", "Gold": "daily",
    "US_YieldSpread": "daily", "VIX": "daily", "Oil_WTI": "daily",
    # Monthly
    "CPI_KR_YoY": "monthly", "Rate_KR": "monthly", "Rate_ECB": "monthly",
    "CPI_US_YoY": "monthly", "CPI_XM_YoY": "monthly",
    "Copper": "monthly", "Wheat": "monthly", "Corn": "monthly",
    "FedFunds": "monthly", "US_CoreCPI_idx": "monthly",
    "US_PPI": "monthly", "US_Unemp": "monthly", "US_ConsConf": "monthly",
    "US_M2": "monthly",
    "KR_Interbank3M": "monthly", "KR_Unemp": "monthly", "KR_LongRate": "monthly",
    "JP_Interbank3M": "monthly", "China_CPI": "monthly",
    "KR_Imports": "monthly", "KR_Exports": "monthly",
}


# ============================================================
# Shared Constants
# ============================================================

SEQ_LEN = 120          # 컨텍스트 길이 (개월, ~10년)
PRED_LEN = 12          # 예측 구간 (개월)
TARGET_COL = "CPI_KR_YoY"
FORECAST_START = "2023-01"
TRAIN_END = "2022-12"


# ============================================================
# LSTMConfig
# ============================================================

class LSTMConfig:
    """
    LSTM 모델 설정.

    Parameters
    ----------
    variates    : 변수 이름 리스트 (순서 = 입력 순서, target first)
    target_idx  : 예측 타겟 인덱스 (default 0)
    seq_len     : 입력 시퀀스 길이 (개월, default 120)
    pred_len    : 예측 기간 (개월, default 12)
    hidden_dim  : LSTM hidden dimension (default 256)
    n_layers    : LSTM 레이어 수 (default 3)
    n_heads     : Attention head 수 (default 4)
    d_model     : 변수 임베딩 차원 (default 128)
    dropout     : Dropout rate (default 0.2)
    """

    def __init__(
        self,
        variates: List[str],
        target_idx: int = 0,
        seq_len: int = SEQ_LEN,
        pred_len: int = PRED_LEN,
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        d_model: int = 128,
        dropout: float = 0.2,
    ):
        self.variates = variates
        self.target_idx = target_idx
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout

    @property
    def n_variates(self) -> int:
        return len(self.variates)

    @property
    def target_name(self) -> str:
        return self.variates[self.target_idx]

    def to_dict(self) -> dict:
        return {
            "variates": self.variates,
            "target_idx": self.target_idx,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LSTMConfig":
        return cls(**d)


# ============================================================
# ImportanceAnalyzer
# ============================================================

class ImportanceAnalyzer:
    """
    LSTM의 Variable Fusion Attention 및 Temporal Attention에서
    cross-variate, temporal 분석 수행.
    """

    def __init__(self, config: LSTMConfig):
        self.config = config

    def cross_variate_matrix(
        self,
        variable_attention: np.ndarray,
    ) -> pd.DataFrame:
        """
        Variable Fusion Attention weights → N×N cross-variate matrix.

        Parameters
        ----------
        variable_attention : (n_vars, n_vars) or (seq_len, n_vars, n_vars)
            If 3D, average across timesteps.

        Returns
        -------
        pd.DataFrame (n_vars × n_vars), 각 행의 합 ≈ 1.0
        """
        if variable_attention.ndim == 3:
            attn = variable_attention.mean(axis=0)
        else:
            attn = variable_attention
        return pd.DataFrame(
            attn,
            index=self.config.variates,
            columns=self.config.variates,
        )

    def target_importance(
        self,
        variable_attention: np.ndarray,
    ) -> pd.Series:
        """타겟 변수의 각 변수에 대한 attention 비율."""
        cross = self.cross_variate_matrix(variable_attention)
        return cross.iloc[self.config.target_idx]

    def temporal_importance(
        self,
        temporal_attention: np.ndarray,
    ) -> pd.DataFrame:
        """
        Temporal Attention Decoder weights → (pred_len, seq_len) matrix.

        Parameters
        ----------
        temporal_attention : (pred_len, seq_len)

        Returns
        -------
        pd.DataFrame with forecast months as rows, past months as columns.
        """
        return pd.DataFrame(
            temporal_attention,
            index=[f"M+{i+1}" for i in range(temporal_attention.shape[0])],
            columns=[f"t-{temporal_attention.shape[1]-i}" for i in range(temporal_attention.shape[1])],
        )


# ============================================================
# Results Loaders
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _results_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


def results_available(filename: str = "lstm_inference_results.npz") -> bool:
    return os.path.exists(_results_path(filename))


def load_inference_results(filename: str = "lstm_inference_results.npz") -> dict:
    """LSTM 추론 결과 로딩."""
    data = np.load(_results_path(filename), allow_pickle=True)

    def _safe(key):
        return data[key] if key in data else None

    result = {
        "variates": [str(v) for v in data["variates"]],
        "n_variates": int(data["n_variates"]),
        "forecast_date": [str(d) for d in data["forecast_date"]],
        "forecast_med": data["forecast_med"],
        "forecast_ci_lo": data["forecast_ci_lo"],
        "forecast_ci_hi": data["forecast_ci_hi"],
        "forecast_ar1": data["forecast_ar1"],
        "forecast_actual": data["forecast_actual"],
        "history_date": [str(d) for d in data["history_date"]],
        "history_cpi": data["history_cpi"],
    }

    # LSTM-specific fields
    for key in ["variable_attention", "temporal_attention",
                "gradient_importance", "permutation_importance"]:
        result[key] = _safe(key)

    return result


def load_stage1_screening() -> dict:
    """Stage 1 전체 변수 스크리닝 결과 로딩."""
    data = np.load(_results_path("lstm_stage1_screening.npz"), allow_pickle=True)

    def _safe(key):
        return data[key] if key in data else None

    return {
        "variates": [str(v) for v in data["variates"]],
        "n_variates": int(data["n_variates"]),
        "ranking_vars": [str(v) for v in data["ranking_vars"]] if "ranking_vars" in data else None,
        "ranking_scores": _safe("ranking_scores"),
        "variable_attention": _safe("variable_attention"),
        "gradient_importance": _safe("gradient_importance"),
        "permutation_importance": _safe("permutation_importance"),
    }


def load_narrative_results() -> Optional[dict]:
    """경제 서사 분석 결과 로딩."""
    path = _results_path("lstm_narrative_results.npz")
    if not os.path.exists(path):
        return None
    from causal_narrative import load_narrative_results as _load
    return _load()


def load_ablation_results() -> dict:
    """Ablation 실험 결과 로딩."""
    data = np.load(_results_path("lstm_ablation_results.npz"), allow_pickle=True)
    return {
        "baseline_rmse": float(data["baseline_rmse"]),
        "baseline_mae": float(data["baseline_mae"]),
        "attn_ranking": [str(v) for v in data["attn_ranking"]],
        "attn_values": data["attn_values"].astype(float),
        "abl_vars": [str(v) for v in data["abl_vars"]],
        "abl_rmse": data["abl_rmse"].astype(float),
        "abl_mae": data["abl_mae"].astype(float),
        "abl_delta_rmse": data["abl_delta_rmse"].astype(float),
        "inc_labels": [str(v) for v in data["inc_labels"]],
        "inc_n_vars": data["inc_n_vars"].astype(int),
        "inc_rmse": data["inc_rmse"].astype(float),
        "inc_mae": data["inc_mae"].astype(float),
    }
