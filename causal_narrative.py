"""
Economic Narrative Analysis for BISTRO-LSTM
============================================
어텐션을 넘어선 경제적 서사 분석.

4가지 분석:
1. Counterfactual Intervention — 변수 충격 → CPI 반응 (방향, 크기, 비대칭)
2. Jacobian Lag Sensitivity — 시차별 영향도 (∂CPI/∂x per timestep)
3. Pathway Decomposition — 경로 분해 (직접 vs 상관 변수 경유 간접)
4. Economic Narrative — 자동 서사 생성

Usage (standalone):
    .venv/bin/python3 causal_narrative.py

Usage (from code):
    from causal_narrative import run_full_analysis
    results = run_full_analysis(model, normalizer, train_df, config)
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional

from lstm_model import AttentionLSTMForecaster
from lstm_core import (
    LSTMConfig, TIER_LABELS, TARGET_COL, SEQ_LEN, PRED_LEN,
    FORECAST_START, TRAIN_END,
)
from preprocessing_util import load_macro_panel, split_train_test, ZScoreNormalizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MACRO_CSV = os.path.join(DATA_DIR, "macro_panel.csv")


# ============================================================
# Mediator Channel Definitions
# ============================================================

MEDIATOR_CHANNELS = {
    "commodity": {
        "vars": [
            # 표준 변수명
            "Oil_WTI", "Oil_Brent", "NatGas_HH", "Copper", "Wheat", "Corn", "Gold",
            # 확장 변수명
            "Pork", "PPI_CopperNickel",
        ],
        "label_kr": "원자재",
        "label_en": "Commodity",
    },
    "exchange_rate": {
        "vars": [
            "USD_KRW", "CNY_USD", "JPY_USD", "DXY_Broad",
            "AUD_USD", "THB_USD", "JP_REER",
        ],
        "label_kr": "환율",
        "label_en": "Exchange Rate",
    },
    "monetary": {
        "vars": [
            "FedFunds", "Rate_KR", "Rate_ECB", "KR_Interbank3M",
            "KR_LongRate", "US_YieldSpread", "VIX", "US_M2",
            "CN_Interbank3M", "JP_Interbank3M",
            "US_Mortgage15Y", "UK_10Y_Bond", "US_DepInstCredit", "KC_FSI",
        ],
        "label_kr": "통화/금융",
        "label_en": "Monetary/Financial",
    },
    "demand": {
        "vars": [
            "KR_Exports", "KR_Imports", "KR_Unemp", "US_Unemp", "US_ConsConf",
            "US_UnempRate", "US_NFP", "US_TradeTransEmp", "KR_MfgProd",
        ],
        "label_kr": "수요/실물",
        "label_en": "Demand/Real Economy",
    },
    "global_price": {
        "vars": [
            "CPI_US_YoY", "CPI_XM_YoY", "China_CPI", "US_CoreCPI_idx", "US_PPI",
            "JP_CoreCPI", "CN_PPI", "US_ExportPI",
        ],
        "label_kr": "글로벌 물가",
        "label_en": "Global Prices",
    },
}

CHANNEL_ORDER = ["commodity", "exchange_rate", "monetary", "demand", "global_price"]


def get_variable_channel(var: str) -> Optional[str]:
    """변수가 속한 채널 반환."""
    for ch_name, ch_info in MEDIATOR_CHANNELS.items():
        if var in ch_info["vars"]:
            return ch_name
    return None


def get_active_channels(variates: List[str]) -> Dict[str, List[str]]:
    """모델에 포함된 변수 기준으로 활성 채널 반환."""
    active = {}
    for ch_name in CHANNEL_ORDER:
        present = [v for v in MEDIATOR_CHANNELS[ch_name]["vars"] if v in variates]
        if present:
            active[ch_name] = present
    return active


# ============================================================
# 1. Counterfactual Intervention Analysis
# ============================================================

def counterfactual_analysis(
    model: AttentionLSTMForecaster,
    input_tensor: torch.Tensor,
    variates: List[str],
    normalizer: ZScoreNormalizer,
    device: torch.device,
    perturb_months: int = 12,
) -> Dict:
    """
    각 공변량을 ±1σ 충격 → CPI 예측 변화.

    Returns
    -------
    dict with:
        covariate_names: List[str]
        impact_plus:  (n_cov, pred_len) — CPI change in pp when var +1σ
        impact_minus: (n_cov, pred_len) — CPI change in pp when var -1σ
        total_impact: (n_cov,) — mean absolute impact per covariate
        direction:    (n_cov,) — +1 if positive, -1 if negative
        asymmetry:    (n_cov,) — asymmetry score [-1, 1]
    """
    model = model.to(device)
    model.eval()

    # Baseline
    with torch.no_grad():
        baseline_norm = model(input_tensor.to(device))["mu"].cpu().numpy()[0]
    baseline_raw = normalizer.inverse_transform_target(baseline_norm.reshape(1, -1))[0]

    covariates = [v for v in variates if v != TARGET_COL]
    n_cov = len(covariates)
    pred_len = baseline_norm.shape[0]

    impact_plus = np.zeros((n_cov, pred_len))
    impact_minus = np.zeros((n_cov, pred_len))

    for i, var in enumerate(covariates):
        var_idx = variates.index(var)

        for sign, storage in [(1.0, impact_plus), (-1.0, impact_minus)]:
            perturbed = input_tensor.clone().to(device)
            perturbed[0, -perturb_months:, var_idx] += sign * 1.0  # 1σ in z-score space
            with torch.no_grad():
                pred_norm = model(perturbed)["mu"].cpu().numpy()[0]
            pred_raw = normalizer.inverse_transform_target(pred_norm.reshape(1, -1))[0]
            storage[i] = pred_raw - baseline_raw

    # Derived metrics
    mean_plus = impact_plus.mean(axis=1)
    mean_minus = impact_minus.mean(axis=1)
    abs_plus = np.abs(mean_plus)
    abs_minus = np.abs(mean_minus)

    total_impact = (abs_plus + abs_minus) / 2.0
    direction = np.sign(mean_plus)
    asymmetry = (abs_plus - abs_minus) / np.maximum(abs_plus + abs_minus, 1e-8)

    return {
        "covariate_names": covariates,
        "impact_plus": impact_plus,
        "impact_minus": impact_minus,
        "total_impact": total_impact,
        "direction": direction,
        "asymmetry": asymmetry,
        "baseline_raw": baseline_raw,
    }


# ============================================================
# 2. Jacobian Lag Sensitivity Analysis
# ============================================================

def jacobian_lag_analysis(
    model: AttentionLSTMForecaster,
    input_tensor: torch.Tensor,
    variates: List[str],
    normalizer: ZScoreNormalizer,
    device: torch.device,
) -> Dict:
    """
    Jacobian 기반 시차별 민감도: ∂CPI_pred[h] / ∂x[t, var].

    Returns
    -------
    dict with:
        jacobian_raw: (pred_len, n_vars, seq_len) — raw-scale ∂CPI_pp / ∂var_1σ
        jacobian_avg: (n_cov, seq_len) — pred_len 평균, covariates only
        peak_lags:    (n_cov,) — peak sensitivity lag (months ago)
        covariate_names: List[str]
    """
    model = model.to(device)
    model.eval()

    x = input_tensor.clone().to(device).requires_grad_(True)
    out = model(x)

    pred_len = out["mu"].shape[1]
    seq_len = x.shape[1]
    n_vars = x.shape[2]

    sigma_cpi = float(normalizer.std_[TARGET_COL])

    jacobian_raw = np.zeros((pred_len, n_vars, seq_len))

    for h in range(pred_len):
        if x.grad is not None:
            x.grad.zero_()
        out["mu"][0, h].backward(retain_graph=True)
        # grad: (1, seq_len, n_vars)
        grad = x.grad[0].cpu().numpy()  # (seq_len, n_vars)
        # Scale: ∂CPI_pp per 1σ shock = grad × σ_CPI
        jacobian_raw[h] = (grad * sigma_cpi).T  # (n_vars, seq_len)

    # Extract covariates and compute averages
    covariates = [v for v in variates if v != TARGET_COL]
    cov_indices = [variates.index(v) for v in covariates]

    # Average across forecast months
    jacobian_avg = jacobian_raw.mean(axis=0)[cov_indices]  # (n_cov, seq_len)

    # Peak lag per covariate (months ago = seq_len - argmax)
    peak_lags = np.zeros(len(covariates), dtype=int)
    for i in range(len(covariates)):
        abs_jac = np.abs(jacobian_avg[i])
        peak_idx = np.argmax(abs_jac)
        peak_lags[i] = seq_len - peak_idx  # months ago

    return {
        "jacobian_raw": jacobian_raw,
        "jacobian_avg": jacobian_avg,
        "peak_lags": peak_lags,
        "covariate_names": covariates,
    }


# ============================================================
# 3. Pathway Decomposition
# ============================================================

def pathway_decomposition(
    model: AttentionLSTMForecaster,
    input_tensor: torch.Tensor,
    variates: List[str],
    normalizer: ZScoreNormalizer,
    train_df: pd.DataFrame,
    device: torch.device,
    perturb_months: int = 12,
    corr_threshold: float = 0.1,
) -> Dict:
    """
    경로 분해: 변수 X의 CPI 영향을 직접/간접으로 분해.

    방법:
    - Isolated: X만 1σ 충격 → 직접 효과
    - Realistic: X 1σ + 상관된 모든 변수도 상관계수만큼 조정 → 총 효과
    - Per-channel: X 1σ + 특정 채널 변수만 상관계수만큼 조정 → 채널 기여

    Returns
    -------
    dict with:
        covariate_names: List[str]
        isolated_effect:  (n_cov,) pp — 직접 효과
        realistic_effect: (n_cov,) pp — 상관 변수 포함 총 효과
        channel_effects:  (n_cov, n_channels) pp — 채널별 추가 효과
        channel_names:    List[str]
    """
    model = model.to(device)
    model.eval()

    # Correlation matrix from normalized training data
    norm_train = normalizer.transform(train_df)
    corr_matrix = norm_train.corr()

    # Baseline
    with torch.no_grad():
        baseline_norm = model(input_tensor.to(device))["mu"].cpu().numpy()[0]
    baseline_raw = normalizer.inverse_transform_target(baseline_norm.reshape(1, -1))[0]

    covariates = [v for v in variates if v != TARGET_COL]
    active_channels = get_active_channels(variates)
    channel_names = [ch for ch in CHANNEL_ORDER if ch in active_channels]
    n_cov = len(covariates)
    n_ch = len(channel_names)

    isolated_effect = np.zeros(n_cov)
    realistic_effect = np.zeros(n_cov)
    channel_effects = np.zeros((n_cov, n_ch))

    for i, var in enumerate(covariates):
        var_idx = variates.index(var)

        # --- Isolated: shock only var ---
        iso = input_tensor.clone().to(device)
        iso[0, -perturb_months:, var_idx] += 1.0
        with torch.no_grad():
            iso_pred = model(iso)["mu"].cpu().numpy()[0]
        iso_raw = normalizer.inverse_transform_target(iso_pred.reshape(1, -1))[0]
        isolated_effect[i] = float(np.mean(iso_raw - baseline_raw))

        # --- Realistic: shock var + adjust all correlated covariates ---
        real = input_tensor.clone().to(device)
        real[0, -perturb_months:, var_idx] += 1.0
        for other_var in covariates:
            if other_var == var:
                continue
            if var not in corr_matrix.columns or other_var not in corr_matrix.columns:
                continue
            beta = float(corr_matrix.loc[var, other_var])
            if abs(beta) > corr_threshold:
                other_idx = variates.index(other_var)
                real[0, -perturb_months:, other_idx] += beta * 1.0
        with torch.no_grad():
            real_pred = model(real)["mu"].cpu().numpy()[0]
        real_raw = normalizer.inverse_transform_target(real_pred.reshape(1, -1))[0]
        realistic_effect[i] = float(np.mean(real_raw - baseline_raw))

        # --- Per-channel: shock var + adjust only channel vars ---
        for j, ch_name in enumerate(channel_names):
            ch_vars = active_channels[ch_name]
            ch_input = input_tensor.clone().to(device)
            ch_input[0, -perturb_months:, var_idx] += 1.0

            has_corr = False
            for ch_var in ch_vars:
                if ch_var == var or ch_var == TARGET_COL:
                    continue
                if var not in corr_matrix.columns or ch_var not in corr_matrix.columns:
                    continue
                beta = float(corr_matrix.loc[var, ch_var])
                if abs(beta) > corr_threshold:
                    ch_idx = variates.index(ch_var)
                    ch_input[0, -perturb_months:, ch_idx] += beta * 1.0
                    has_corr = True

            if has_corr:
                with torch.no_grad():
                    ch_pred = model(ch_input)["mu"].cpu().numpy()[0]
                ch_raw = normalizer.inverse_transform_target(ch_pred.reshape(1, -1))[0]
                ch_effect = float(np.mean(ch_raw - baseline_raw))
                # Channel contribution = channel effect - isolated effect
                channel_effects[i, j] = ch_effect - isolated_effect[i]

    return {
        "covariate_names": covariates,
        "isolated_effect": isolated_effect,
        "realistic_effect": realistic_effect,
        "channel_effects": channel_effects,
        "channel_names": channel_names,
    }


# ============================================================
# 4. Economic Narrative Generation
# ============================================================

def generate_narrative(
    var: str,
    cf_result: Dict,
    lag_result: Dict,
    pw_result: Dict,
    normalizer: ZScoreNormalizer,
) -> str:
    """
    단일 변수에 대한 경제 서사 생성.

    Returns
    -------
    str: 마크다운 형식의 경제 서사
    """
    cov_names = cf_result["covariate_names"]
    if var not in cov_names:
        return f"{var}: 분석 데이터 없음"

    idx = cov_names.index(var)
    tier = TIER_LABELS.get(var, "?")
    channel = get_variable_channel(var)
    ch_label = MEDIATOR_CHANNELS[channel]["label_kr"] if channel else "기타"

    # --- Counterfactual ---
    total_pp = cf_result["total_impact"][idx]
    direction = cf_result["direction"][idx]
    asym = cf_result["asymmetry"][idx]
    plus_mean = cf_result["impact_plus"][idx].mean()
    minus_mean = cf_result["impact_minus"][idx].mean()

    sigma_raw = float(normalizer.std_.get(var, 1.0))
    direction_text = "상승" if direction > 0 else "하락"
    direction_inv = "하락" if direction > 0 else "상승"

    # Magnitude description
    if total_pp > 0.05:
        mag_text = "강한"
    elif total_pp > 0.02:
        mag_text = "중간 수준의"
    else:
        mag_text = "미약한"

    # --- Lag ---
    lag_idx = lag_result["covariate_names"].index(var) if var in lag_result["covariate_names"] else None
    peak_lag = int(lag_result["peak_lags"][lag_idx]) if lag_idx is not None else 0

    if peak_lag <= 2:
        lag_text = "거의 즉각적으로 반영"
    elif peak_lag <= 6:
        lag_text = f"약 {peak_lag}개월의 단기 시차"
    elif peak_lag <= 12:
        lag_text = f"약 {peak_lag}개월의 중기 시차"
    else:
        lag_text = f"{peak_lag}개월 이상의 장기 시차"

    # --- Pathway ---
    pw_idx = pw_result["covariate_names"].index(var) if var in pw_result["covariate_names"] else None
    iso_eff = pw_result["isolated_effect"][pw_idx] if pw_idx is not None else 0
    real_eff = pw_result["realistic_effect"][pw_idx] if pw_idx is not None else 0
    indirect = real_eff - iso_eff

    pathway_lines = []
    if pw_idx is not None and abs(real_eff) > 1e-6:
        iso_pct = abs(iso_eff) / max(abs(real_eff), 1e-8) * 100
        pathway_lines.append(f"  직접 효과: {iso_eff:+.4f}pp ({min(iso_pct, 100):.0f}%)")

        ch_effects = pw_result["channel_effects"][pw_idx]
        for j, ch_name in enumerate(pw_result["channel_names"]):
            if abs(ch_effects[j]) > 1e-5:
                ch_pct = abs(ch_effects[j]) / max(abs(real_eff), 1e-8) * 100
                ch_label_kr = MEDIATOR_CHANNELS[ch_name]["label_kr"]
                pathway_lines.append(f"  {ch_label_kr} 경유: {ch_effects[j]:+.4f}pp ({ch_pct:.0f}%)")

    # --- Asymmetry text ---
    if abs(asym) < 0.1:
        asym_text = "대칭적 반응 (상승/하락 시 영향도 유사)"
    elif asym > 0:
        asym_text = f"비대칭: 상승 충격이 하락 충격보다 {abs(asym):.0%} 더 강함"
    else:
        asym_text = f"비대칭: 하락 충격이 상승 충격보다 {abs(asym):.0%} 더 강함"

    # --- Economic interpretation ---
    interp_parts = []
    if channel == "commodity":
        if direction > 0:
            interp_parts.append("원자재 가격 상승이 수입물가를 거쳐 소비자물가로 전달되는 비용 전가(cost-push) 경로")
        else:
            interp_parts.append("원자재 가격 하락이 물가 하방 압력으로 작용")
    elif channel == "exchange_rate":
        if direction > 0:
            interp_parts.append("환율 상승(원화 약세)이 수입물가 경로를 통해 CPI 상승 압력으로 작용")
        else:
            interp_parts.append("환율 변동이 수입물가 경로를 통해 CPI에 전달")
    elif channel == "monetary":
        if direction > 0:
            interp_parts.append("금리/통화 변수가 자산가격·대출경로를 통해 물가에 양의 영향")
        else:
            interp_parts.append("긴축적 통화정책이 수요 억제를 통해 물가 하방 압력으로 작용")
    elif channel == "demand":
        interp_parts.append("실물경제 변수가 총수요 경로를 통해 물가에 전달")
    elif channel == "global_price":
        interp_parts.append("글로벌 물가 동조화를 통한 전이 효과(spillover)")

    if peak_lag > 6:
        interp_parts.append(f"전달 시차가 {peak_lag}개월로 긴 편이며, 간접적/다단계 전달 경로를 시사")
    elif peak_lag <= 2:
        interp_parts.append("즉각적 반응으로, 시장 기대 또는 직접 가격 연동 가능성")

    # Compose narrative
    lines = [
        f"### {var} &rarr; Korean CPI 영향 분석",
        f"*Tier: {tier} | 채널: {ch_label}*",
        "",
        "**총 영향도**",
        f"- +1σ (원본 단위 약 {sigma_raw:.2f}) 충격 시 CPI {direction_text} **{total_pp:.4f}pp** (12개월 평균)",
        f"- 영향 강도: {mag_text} 영향",
        "",
        "**영향 방향**",
        f"- {var} 상승(+1σ) &rarr; CPI {direction_text} ({plus_mean:+.4f}pp)",
        f"- {var} 하락(-1σ) &rarr; CPI {direction_inv} ({minus_mean:+.4f}pp)",
        f"- {asym_text}",
        "",
        "**피크 시차**",
        f"- {lag_text} (과거 t-{peak_lag}개월 변동에 최대 반응)",
        "",
        "**경로 분해**",
        f"- 총 효과 (상관 변수 연동): {real_eff:+.4f}pp",
    ]
    lines.extend(pathway_lines)
    lines.append("")
    lines.append("*Note: 채널 기여도는 비선형성으로 인해 합이 100%를 초과할 수 있음*")
    lines.append("")
    lines.append("**경제적 해석**")
    for part in interp_parts:
        lines.append(f"- {part}")

    return "\n".join(lines)


# ============================================================
# Orchestrator
# ============================================================

def run_full_analysis(
    model: AttentionLSTMForecaster,
    normalizer: ZScoreNormalizer,
    train_df: pd.DataFrame,
    config: LSTMConfig,
    device: torch.device = None,
    perturb_months: int = 12,
) -> Dict:
    """
    전체 경제 서사 분석 실행.

    Returns
    -------
    dict with all analysis results + narratives
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Prepare input tensor (use config.seq_len, not global SEQ_LEN)
    train_normed = normalizer.transform(train_df).values
    seq_len = config.seq_len
    input_seq = train_normed[-seq_len:]
    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0)

    variates = config.variates

    print("=" * 60)
    print("Economic Narrative Analysis")
    print("=" * 60)

    # 1. Counterfactual
    print("\n[1/4] Counterfactual Intervention Analysis...")
    cf = counterfactual_analysis(model, input_tensor, variates, normalizer, device, perturb_months)
    print(f"  Done — {len(cf['covariate_names'])} covariates analyzed")

    # 2. Jacobian Lag Sensitivity
    print("\n[2/4] Jacobian Lag Sensitivity Analysis...")
    lag = jacobian_lag_analysis(model, input_tensor, variates, normalizer, device)
    print(f"  Done — Jacobian shape: {lag['jacobian_raw'].shape}")

    # 3. Pathway Decomposition
    print("\n[3/4] Pathway Decomposition...")
    pw = pathway_decomposition(model, input_tensor, variates, normalizer, train_df, device, perturb_months)
    print(f"  Done — {len(pw['channel_names'])} active channels: {pw['channel_names']}")

    # 4. Narrative Generation
    print("\n[4/4] Generating Economic Narratives...")
    narratives = {}
    for var in cf["covariate_names"]:
        narratives[var] = generate_narrative(var, cf, lag, pw, normalizer)
    print(f"  Done — {len(narratives)} narratives generated")

    # Print top-5 by impact
    sorted_idx = np.argsort(-cf["total_impact"])
    print(f"\n{'='*60}")
    print("Top-5 Variables by Counterfactual Impact:")
    for rank, idx in enumerate(sorted_idx[:5]):
        var = cf["covariate_names"][idx]
        imp = cf["total_impact"][idx]
        d = "↑" if cf["direction"][idx] > 0 else "↓"
        lag_i = lag["covariate_names"].index(var)
        pk = lag["peak_lags"][lag_i]
        print(f"  {rank+1}. {var}: {imp:.4f}pp ({d}), peak lag t-{pk}")

    return {
        "counterfactual": cf,
        "lag_sensitivity": lag,
        "pathway": pw,
        "narratives": narratives,
        "variates": variates,
    }


# ============================================================
# Save / Load
# ============================================================

def save_narrative_results(results: Dict, filename: str = "lstm_narrative_results.npz"):
    """분석 결과를 npz로 저장."""
    path = os.path.join(DATA_DIR, filename)
    os.makedirs(DATA_DIR, exist_ok=True)

    cf = results["counterfactual"]
    lag = results["lag_sensitivity"]
    pw = results["pathway"]

    save_dict = {
        # Metadata
        "variates": np.array(results["variates"], dtype=object),
        "covariate_names": np.array(cf["covariate_names"], dtype=object),
        # Counterfactual
        "cf_impact_plus": cf["impact_plus"],
        "cf_impact_minus": cf["impact_minus"],
        "cf_total_impact": cf["total_impact"],
        "cf_direction": cf["direction"],
        "cf_asymmetry": cf["asymmetry"],
        "cf_baseline_raw": cf["baseline_raw"],
        # Jacobian
        "jac_raw": lag["jacobian_raw"],
        "jac_avg": lag["jacobian_avg"],
        "jac_peak_lags": lag["peak_lags"],
        # Pathway
        "pw_isolated": pw["isolated_effect"],
        "pw_realistic": pw["realistic_effect"],
        "pw_channel_effects": pw["channel_effects"],
        "pw_channel_names": np.array(pw["channel_names"], dtype=object),
        # Narratives
        "narrative_vars": np.array(list(results["narratives"].keys()), dtype=object),
        "narrative_texts": np.array(list(results["narratives"].values()), dtype=object),
    }

    np.savez_compressed(path, **save_dict)
    print(f"\nSaved narrative results to {path}")


def load_narrative_results(filename: str = "lstm_narrative_results.npz") -> Optional[Dict]:
    """npz에서 분석 결과 로딩."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None

    data = np.load(path, allow_pickle=True)

    def _safe(key):
        return data[key] if key in data else None

    covariate_names = [str(v) for v in data["covariate_names"]]

    # Reconstruct narratives dict
    narr_vars = [str(v) for v in data["narrative_vars"]] if "narrative_vars" in data else []
    narr_texts = [str(t) for t in data["narrative_texts"]] if "narrative_texts" in data else []
    narratives = dict(zip(narr_vars, narr_texts))

    return {
        "variates": [str(v) for v in data["variates"]],
        "covariate_names": covariate_names,
        "counterfactual": {
            "covariate_names": covariate_names,
            "impact_plus": data["cf_impact_plus"],
            "impact_minus": data["cf_impact_minus"],
            "total_impact": data["cf_total_impact"],
            "direction": data["cf_direction"],
            "asymmetry": data["cf_asymmetry"],
            "baseline_raw": data["cf_baseline_raw"],
        },
        "lag_sensitivity": {
            "covariate_names": covariate_names,
            "jacobian_raw": data["jac_raw"],
            "jacobian_avg": data["jac_avg"],
            "peak_lags": data["jac_peak_lags"],
        },
        "pathway": {
            "covariate_names": covariate_names,
            "isolated_effect": data["pw_isolated"],
            "realistic_effect": data["pw_realistic"],
            "channel_effects": data["pw_channel_effects"],
            "channel_names": [str(n) for n in data["pw_channel_names"]],
        },
        "narratives": narratives,
    }


# ============================================================
# Standalone Execution
# ============================================================

def main():
    """모델 체크포인트에서 로딩하여 분석 실행."""
    import argparse

    parser = argparse.ArgumentParser(description="BISTRO-LSTM Economic Narrative Analysis")
    parser.add_argument("--perturb-months", type=int, default=12, help="Perturbation window (months)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model + config from checkpoint
    ckpt_path = os.path.join(DATA_DIR, "lstm_model_best.pt")
    if not os.path.exists(ckpt_path):
        print("ERROR: lstm_model_best.pt not found. Run training first.")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Config can be in checkpoint or in npz
    if isinstance(ckpt, dict) and "config" in ckpt:
        config_dict = ckpt["config"]
        state_dict = ckpt["model_state_dict"]
    else:
        # Fallback: reconstruct from npz
        npz_path = os.path.join(DATA_DIR, "lstm_inference_results.npz")
        if not os.path.exists(npz_path):
            print("ERROR: Cannot find config. Need lstm_model_best.pt with config or lstm_inference_results.npz.")
            return
        npz = np.load(npz_path, allow_pickle=True)
        variates_arr = [str(v) for v in npz["variates"]]
        config_dict = {"variates": variates_arr}
        state_dict = ckpt

    config = LSTMConfig.from_dict(config_dict)
    variates = config.variates
    print(f"Config loaded: {config.n_variates} variables, seq_len={config.seq_len}")

    model = AttentionLSTMForecaster.from_config(config)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded: {model.count_parameters():,} parameters")

    # Load and prepare data — try optimal18 first, fall back to standard
    optimal_csv = os.path.join(DATA_DIR, "macro_panel_optimal18.csv")
    csv_path = optimal_csv if os.path.exists(optimal_csv) else MACRO_CSV
    df = load_macro_panel(csv_path, TARGET_COL, variates)
    train_df, _ = split_train_test(df, TRAIN_END, FORECAST_START, "2023-12")

    normalizer = ZScoreNormalizer()
    normalizer.fit_transform(train_df)

    # Run analysis
    results = run_full_analysis(
        model, normalizer, train_df, config,
        device=device,
        perturb_months=args.perturb_months,
    )

    # Save
    save_narrative_results(results)

    print("\nDone! View results in the Streamlit dashboard (Economic Narrative tab).")


if __name__ == "__main__":
    main()
