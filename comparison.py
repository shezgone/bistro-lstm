"""
BISTRO vs LSTM Comparison Framework
=====================================
양쪽 프로젝트 결과를 로드하여 직접 비교.

Usage:
    .venv/bin/python3 comparison.py
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional


BISTRO_XAI_DIR = os.path.join(os.path.dirname(__file__), "..", "bistro-xai")
LSTM_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_bistro_results(variant: str = "optimal18") -> Optional[Dict]:
    """
    bistro-xai의 inference 결과 로딩.

    Parameters
    ----------
    variant : "10var" (real_inference_results) 또는 "optimal18" (forecast_optimal18)
    """
    if variant == "optimal18":
        path = os.path.join(BISTRO_XAI_DIR, "data", "forecast_optimal18.npz")
        # optimal18은 forecast_actual이 없으므로 real_inference_results에서 가져옴
        base_path = os.path.join(BISTRO_XAI_DIR, "data", "real_inference_results.npz")
    else:
        path = os.path.join(BISTRO_XAI_DIR, "data", "real_inference_results.npz")
        base_path = path

    if not os.path.exists(path):
        print(f"BISTRO results not found: {path}")
        return None

    data = np.load(path, allow_pickle=True)

    def _safe(key):
        return data[key] if key in data else None

    result = {
        "forecast_med": data["forecast_med"],
        "forecast_ci_lo": _safe("forecast_ci_lo"),
        "forecast_ci_hi": _safe("forecast_ci_hi"),
        "variant": variant,
    }

    # covariates 정보
    if "covariates" in data:
        result["variates"] = [str(v) for v in data["covariates"]]
        result["n_variates"] = int(data["n_variates"]) + 1  # covariates + target
    elif "variates" in data:
        result["variates"] = [str(v) for v in data["variates"]]
        result["n_variates"] = int(data["n_variates"])

    # base에서 actual, ar1, history 가져오기
    if os.path.exists(base_path):
        base = np.load(base_path, allow_pickle=True)
        result["forecast_date"] = [str(d) for d in base["forecast_date"]]
        result["forecast_actual"] = base["forecast_actual"]
        result["forecast_ar1"] = base["forecast_ar1"]
        result["history_date"] = [str(d) for d in base["history_date"]]
        result["history_cpi"] = base["history_cpi"]

    return result


def load_lstm_results() -> Optional[Dict]:
    """LSTM inference 결과 로딩."""
    path = os.path.join(LSTM_DATA_DIR, "lstm_inference_results.npz")
    if not os.path.exists(path):
        print(f"LSTM results not found: {path}")
        return None

    data = np.load(path, allow_pickle=True)

    def _safe(key):
        return data[key] if key in data else None

    return {
        "variates": [str(v) for v in data["variates"]],
        "n_variates": int(data["n_variates"]),
        "forecast_date": [str(d) for d in data["forecast_date"]],
        "forecast_med": data["forecast_med"],
        "forecast_ci_lo": _safe("forecast_ci_lo"),
        "forecast_ci_hi": _safe("forecast_ci_hi"),
        "forecast_ar1": data["forecast_ar1"],
        "forecast_actual": data["forecast_actual"],
        "history_date": [str(d) for d in data["history_date"]],
        "history_cpi": data["history_cpi"],
        "variable_attention": _safe("variable_attention"),
        "temporal_attention": _safe("temporal_attention"),
    }


def load_bistro_ablation() -> Optional[Dict]:
    """bistro-xai의 ablation 결과 로딩."""
    path = os.path.join(BISTRO_XAI_DIR, "data", "ablation_results.npz")
    if not os.path.exists(path):
        return None

    data = np.load(path, allow_pickle=True)
    return {
        "baseline_rmse": float(data["baseline_rmse"]),
        "attn_ranking": [str(v) for v in data["attn_ranking"]],
        "abl_vars": [str(v) for v in data["abl_vars"]],
        "abl_delta_rmse": data["abl_delta_rmse"].astype(float),
    }


def load_lstm_ablation() -> Optional[Dict]:
    """LSTM ablation 결과 로딩."""
    path = os.path.join(LSTM_DATA_DIR, "lstm_ablation_results.npz")
    if not os.path.exists(path):
        return None

    data = np.load(path, allow_pickle=True)
    return {
        "baseline_rmse": float(data["baseline_rmse"]),
        "attn_ranking": [str(v) for v in data["attn_ranking"]],
        "abl_vars": [str(v) for v in data["abl_vars"]],
        "abl_delta_rmse": data["abl_delta_rmse"].astype(float),
    }


def compute_metrics(preds: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """RMSE, MAE, MAPE 계산."""
    rmse = np.sqrt(np.mean((preds - actual) ** 2))
    mae = np.mean(np.abs(preds - actual))
    mape = np.mean(np.abs((preds - actual) / (actual + 1e-8))) * 100
    return {"rmse": rmse, "mae": mae, "mape": mape}


def compare_forecasts(bistro: Dict, lstm: Dict) -> pd.DataFrame:
    """예측 성능 비교 테이블 생성."""
    actual = bistro["forecast_actual"]

    rows = []

    # BISTRO
    m = compute_metrics(bistro["forecast_med"], actual)
    rows.append({"Model": "BISTRO (Transformer)", "RMSE": m["rmse"], "MAE": m["mae"], "MAPE": m["mape"]})

    # LSTM
    m = compute_metrics(lstm["forecast_med"], actual)
    rows.append({"Model": "LSTM (Ours)", "RMSE": m["rmse"], "MAE": m["mae"], "MAPE": m["mape"]})

    # AR(1)
    m = compute_metrics(bistro["forecast_ar1"], actual)
    rows.append({"Model": "AR(1) Baseline", "RMSE": m["rmse"], "MAE": m["mae"], "MAPE": m["mape"]})

    df = pd.DataFrame(rows)
    df = df.set_index("Model")
    return df


def compare_variable_rankings(
    bistro_abl: Dict,
    lstm_abl: Dict,
) -> pd.DataFrame:
    """
    변수 중요도 랭킹 비교.
    공통 변수에 대해 BISTRO ablation ΔRMSE vs LSTM ablation ΔRMSE.
    """
    common_vars = set(bistro_abl["abl_vars"]) & set(lstm_abl["abl_vars"])

    rows = []
    for var in sorted(common_vars):
        b_idx = list(bistro_abl["abl_vars"]).index(var)
        l_idx = list(lstm_abl["abl_vars"]).index(var)

        rows.append({
            "Variable": var,
            "BISTRO_ΔRMSE": bistro_abl["abl_delta_rmse"][b_idx],
            "LSTM_ΔRMSE": lstm_abl["abl_delta_rmse"][l_idx],
        })

    df = pd.DataFrame(rows).set_index("Variable")
    df["BISTRO_Rank"] = df["BISTRO_ΔRMSE"].rank(ascending=False).astype(int)
    df["LSTM_Rank"] = df["LSTM_ΔRMSE"].rank(ascending=False).astype(int)

    return df.sort_values("LSTM_ΔRMSE", ascending=False)


def ranking_correlation(bistro_abl: Dict, lstm_abl: Dict) -> Dict[str, float]:
    """Spearman/Kendall 상관계수로 랭킹 일치도 측정."""
    from scipy import stats

    common_vars = sorted(set(bistro_abl["abl_vars"]) & set(lstm_abl["abl_vars"]))
    if len(common_vars) < 3:
        return {"spearman": float("nan"), "kendall": float("nan")}

    b_deltas = []
    l_deltas = []
    for var in common_vars:
        b_idx = list(bistro_abl["abl_vars"]).index(var)
        l_idx = list(lstm_abl["abl_vars"]).index(var)
        b_deltas.append(bistro_abl["abl_delta_rmse"][b_idx])
        l_deltas.append(lstm_abl["abl_delta_rmse"][l_idx])

    spearman, sp_p = stats.spearmanr(b_deltas, l_deltas)
    kendall, kt_p = stats.kendalltau(b_deltas, l_deltas)

    return {
        "spearman": spearman, "spearman_p": sp_p,
        "kendall": kendall, "kendall_p": kt_p,
    }


def main():
    print("=" * 60)
    print("BISTRO vs LSTM Comparison")
    print("=" * 60)

    bistro = load_bistro_results()
    lstm = load_lstm_results()

    if bistro is None or lstm is None:
        print("\nBoth results needed for comparison.")
        if bistro is None:
            print("  Missing: bistro-xai/data/real_inference_results.npz")
        if lstm is None:
            print("  Missing: data/lstm_inference_results.npz")
        return

    # Forecast comparison
    print("\n▶ Forecast Performance (2023 OOS)")
    print("-" * 50)
    comp_df = compare_forecasts(bistro, lstm)
    print(comp_df.to_string(float_format="%.4f"))

    # Winner
    bistro_rmse = comp_df.loc["BISTRO (Transformer)", "RMSE"]
    lstm_rmse = comp_df.loc["LSTM (Ours)", "RMSE"]
    if lstm_rmse < bistro_rmse:
        improvement = (1 - lstm_rmse / bistro_rmse) * 100
        print(f"\n★ LSTM wins! {improvement:.1f}% lower RMSE than BISTRO")
    else:
        degradation = (lstm_rmse / bistro_rmse - 1) * 100
        print(f"\n  BISTRO wins by {degradation:.1f}% RMSE margin")

    # Ablation comparison
    bistro_abl = load_bistro_ablation()
    lstm_abl = load_lstm_ablation()

    if bistro_abl is not None and lstm_abl is not None:
        print("\n▶ Variable Importance Ranking Comparison")
        print("-" * 50)
        rank_df = compare_variable_rankings(bistro_abl, lstm_abl)
        print(rank_df.to_string(float_format="%.4f"))

        corr = ranking_correlation(bistro_abl, lstm_abl)
        print(f"\nRanking Correlation:")
        print(f"  Spearman ρ = {corr['spearman']:.4f} (p = {corr['spearman_p']:.4f})")
        print(f"  Kendall  τ = {corr['kendall']:.4f} (p = {corr['kendall_p']:.4f})")


if __name__ == "__main__":
    main()
