"""Backtest 4 statistical regime detectors on Korean CPI 2003-2025.

Goal: identify months where simple baselines (Trend12 / AR(1)) are about to fail
       — i.e. the "shock regime" that warrants escalating to HCX-style LLM forecast.

Detectors evaluated (all vintage-safe — score at month t uses only data through t):
  A. Rolling σ_12m of CPI YoY
  B. Two-sided CUSUM with rolling-window reference + std normalization
  C. MAD around 12-month MA (deviation in MAD units)
  D. Multi-signal: weighted z-score of (σ_CPI, σ_CSI, |ΔCPI/3m|)

Ground truth: shock(t) = |Trend12 forecast error at t+1| > 0.5pp
            (the baseline we'd otherwise rely on actually failed)

Outputs:
  - AUC, optimal threshold (Youden's J), TPR / FPR at that threshold
  - Coverage of known historical shock periods (GFC, COVID, 2022 surge, 2023 disinflation)
  - False-positive rate during known stable periods (2014-15, 2018-19, 2024H2, 2025)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "macro_panel_full.csv"

SHOCK_THRESHOLD = 0.5  # pp; |Trend12 error| above this counts as a shock month

# Historical period labels (for sanity-checking detector coverage)
KNOWN_SHOCKS = {
    "2008 GFC era":         ("2008-01", "2009-12"),
    "2010 commodity surge": ("2010-08", "2011-12"),
    "COVID drop":           ("2020-02", "2020-07"),
    "2022 inflation surge": ("2021-10", "2022-12"),
    "2023 disinflation":    ("2023-01", "2023-12"),
}
KNOWN_STABLE = {
    "2014-15 stable":  ("2014-01", "2015-12"),
    "2018-19 stable":  ("2018-01", "2019-12"),
    "2024 H2 stable":  ("2024-07", "2024-12"),
    "2025 stable":     ("2025-01", "2025-12"),
}


def trend12_forecast(hist_values: np.ndarray) -> float:
    last12 = hist_values[-12:]
    coef = np.polyfit(np.arange(12), last12, 1)
    return float(np.polyval(coef, 12))


def compute_trend12_errors(y: pd.Series) -> pd.Series:
    """For each month t (with ≥12 history and a t+1 observation), compute
    |Trend12_t→t+1 - actual_t+1|. Returns series indexed by t (the origin)."""
    errs = {}
    arr = y.values
    for i, t in enumerate(y.index):
        if i < 12 or i + 1 >= len(arr):
            continue
        pred = trend12_forecast(arr[: i + 1])
        actual = arr[i + 1]
        errs[t] = abs(pred - actual)
    return pd.Series(errs)


# === Detectors (each returns a per-month score, vintage-safe) ===

def rolling_sigma(y: pd.Series, window: int = 12) -> pd.Series:
    return y.rolling(window).std()


def cusum_two_sided(y: pd.Series, window: int = 24, k: float = 0.5) -> pd.Series:
    """Two-sided CUSUM with rolling-window reference mean + rolling std normalization.
    Score = max(positive accumulator, |negative accumulator|).
    k = slack (in std units) — values close to mean don't accumulate."""
    ref = y.rolling(window).mean()
    sd = y.rolling(window).std().clip(lower=0.05)
    pos = np.zeros(len(y)); neg = np.zeros(len(y))
    for i in range(1, len(y)):
        if pd.isna(ref.iloc[i]) or pd.isna(sd.iloc[i]):
            continue
        z = (y.iloc[i] - ref.iloc[i]) / sd.iloc[i]
        pos[i] = max(0.0, pos[i - 1] + z - k)
        neg[i] = min(0.0, neg[i - 1] + z + k)
    return pd.Series(np.maximum(pos, -neg), index=y.index)


def mad_around_ma(y: pd.Series, ma_window: int = 12, mad_window: int = 24) -> pd.Series:
    ma = y.rolling(ma_window).mean()
    dev = y - ma
    mad = dev.rolling(mad_window).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    ).clip(lower=0.05)
    return (dev.abs() / mad).clip(0, 50)


def multi_signal(y: pd.Series, csi: pd.Series | None = None,
                 weights=(0.5, 0.3, 0.2)) -> pd.Series:
    """Weighted z-score of three signals; each z-scored against its own
    expanding-window mean/std (vintage-safe)."""
    sig_y = y.rolling(12).std()
    abs_dy = y.diff(3).abs()
    if csi is not None and not csi.isna().all():
        sig_csi = csi.rolling(12).std()
    else:
        sig_csi = pd.Series(0.0, index=y.index)

    def expanding_z(s: pd.Series, min_periods: int = 24) -> pd.Series:
        mu = s.expanding(min_periods=min_periods).mean()
        sd = s.expanding(min_periods=min_periods).std().clip(lower=1e-6)
        return ((s - mu) / sd).fillna(0)

    z1, z2, z3 = expanding_z(sig_y), expanding_z(sig_csi), expanding_z(abs_dy)
    return weights[0] * z1 + weights[1] * z2 + weights[2] * z3


def evaluate_detector(score: pd.Series, err: pd.Series, threshold: float) -> dict:
    common = err.index.intersection(score.dropna().index)
    s = score.loc[common]
    e = err.loc[common]
    y_true = (e > threshold).astype(int).values
    y_score = s.values
    if y_true.sum() < 5 or (1 - y_true).sum() < 5:
        return {"error": "insufficient class balance"}

    auc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    j_idx = int(np.argmax(j))
    return {
        "auc": auc,
        "n_total": int(len(y_true)),
        "n_shock": int(y_true.sum()),
        "n_stable": int((1 - y_true).sum()),
        "optimal_threshold": float(thr[j_idx]),
        "tpr_at_opt": float(tpr[j_idx]),
        "fpr_at_opt": float(fpr[j_idx]),
        "j_at_opt": float(j[j_idx]),
    }


def coverage_in_periods(score: pd.Series, threshold: float, periods: dict) -> dict:
    flagged = score[score > threshold].index
    out = {}
    for label, (start, end) in periods.items():
        ps, pe = pd.Period(start, "M"), pd.Period(end, "M")
        period_months = pd.period_range(ps, pe, freq="M")
        in_period = [m for m in flagged if ps <= m <= pe]
        out[label] = {
            "period_len": int(len(period_months)),
            "flagged": int(len(in_period)),
            "rate": float(len(in_period) / len(period_months)) if len(period_months) else 0.0,
            "first_flag": str(in_period[0]) if in_period else None,
        }
    return out


def main() -> None:
    df = pd.read_csv(DATA, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="M")
    y = df["CPI_KR_YoY"].astype(float)
    csi = df["BoK_CSI"].astype(float) if "BoK_CSI" in df.columns else None

    err = compute_trend12_errors(y)
    n_total = len(err)
    n_shock = int((err > SHOCK_THRESHOLD).sum())
    print(f"Period: {y.index[0]} .. {y.index[-1]}  ({len(y)} months)")
    print(f"Trend12 baseline error series: n={n_total}")
    print(f"Shock months (|err| > {SHOCK_THRESHOLD}pp): {n_shock}/{n_total} "
          f"({100*n_shock/n_total:.1f}%)")
    print(f"Mean |err|: {err.mean():.3f}pp,  median: {err.median():.3f}pp,  "
          f"max: {err.max():.3f}pp at {err.idxmax()}")

    detectors = {
        "A_RollingSigma_12m":   rolling_sigma(y, window=12),
        "B_CUSUM_2sided":       cusum_two_sided(y, window=24, k=0.5),
        "C_MAD_around_MA12":    mad_around_ma(y, ma_window=12, mad_window=24),
        "D_MultiSignal_zsum":   multi_signal(y, csi),
    }

    print(f"\n{'='*78}")
    print(f"{'detector':24s} {'AUC':>6s} {'TPR':>6s} {'FPR':>6s} {'J':>6s} {'thresh':>10s}")
    print("-" * 78)
    summary = {}
    for name, score in detectors.items():
        ev = evaluate_detector(score, err, SHOCK_THRESHOLD)
        summary[name] = ev
        if "error" in ev:
            print(f"{name:24s} (insufficient data)")
            continue
        print(f"{name:24s} {ev['auc']:6.3f} {ev['tpr_at_opt']:6.3f} "
              f"{ev['fpr_at_opt']:6.3f} {ev['j_at_opt']:6.3f} "
              f"{ev['optimal_threshold']:10.4f}")

    print(f"\n{'='*78}\nKnown-shock period coverage (using each detector's optimal threshold):")
    coverage_summary = {}
    for name, score in detectors.items():
        if "error" in summary[name]: continue
        thr = summary[name]["optimal_threshold"]
        cov = coverage_in_periods(score, thr, KNOWN_SHOCKS)
        coverage_summary[name] = {"shock_coverage": cov}
        print(f"\n  {name} (threshold={thr:.3f}):")
        for label, c in cov.items():
            print(f"    {label:25s}: {c['flagged']}/{c['period_len']} months "
                  f"({100*c['rate']:.0f}%)  first_flag={c['first_flag']}")

    print(f"\n{'='*78}\nKnown-stable period false-positive (lower = better):")
    for name, score in detectors.items():
        if "error" in summary[name]: continue
        thr = summary[name]["optimal_threshold"]
        cov = coverage_in_periods(score, thr, KNOWN_STABLE)
        coverage_summary[name]["stable_fp"] = cov
        print(f"\n  {name} (threshold={thr:.3f}):")
        for label, c in cov.items():
            print(f"    {label:25s}: {c['flagged']}/{c['period_len']} months FP "
                  f"({100*c['rate']:.0f}%)")

    # Score per period summary: shock_recall - stable_fp_rate (the operational j-statistic)
    print(f"\n{'='*78}\nOperational score (mean shock_recall − mean stable_FP):")
    op_scores = []
    for name in detectors:
        if "error" in summary[name]: continue
        sr = np.mean([c["rate"] for c in coverage_summary[name]["shock_coverage"].values()])
        fp = np.mean([c["rate"] for c in coverage_summary[name]["stable_fp"].values()])
        op = sr - fp
        op_scores.append((name, sr, fp, op))
        print(f"  {name:24s} shock_recall={sr:.3f}  stable_fp={fp:.3f}  "
              f"op_score={op:+.3f}")

    op_scores.sort(key=lambda x: -x[3])
    print(f"\nWinner by operational score: {op_scores[0][0]}")

    out = ROOT / "data" / "regime_detector_backtest.json"
    out.write_text(json.dumps({
        "shock_threshold_pp": SHOCK_THRESHOLD,
        "n_total_months": n_total,
        "n_shock_months": n_shock,
        "detector_summary": summary,
        "coverage": coverage_summary,
        "operational_ranking": [
            {"name": n, "shock_recall": sr, "stable_fp": fp, "op_score": op}
            for n, sr, fp, op in op_scores
        ],
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
