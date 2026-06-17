"""Quantify HCX's forecast value when D MultiSignal detector flags shock months.

Pulls all 1-step-rolling HCX forced-CoT forecasts available:
  - rolling_2025_csi_forced_results.json  (8 origins, 2025-04..2025-11)
  - experiments.db Round 0                (8 origins, 2021-12..2022-07; stress_2022_h1)

For each origin, compares:
  hcx_err     = |HCX seed-mean forecast - actual|
  trend12_err = |Trend12 forecast       - actual|

Stratifies by D MultiSignal score at origin (threshold 0.362 from backtest).
Reports mean errors per stratum, paired difference, and Welch's t-test.

Bonus: 12-step BLINDED forecasts for 2023 + 2024 from
       stress_2023_2024_blinded_results.json — shown separately (different protocol).
"""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "macro_panel_full.csv"

# ---- Detector replication (same as regime_detector_backtest.py) ----
def trend12_forecast(hist_values: np.ndarray) -> float:
    last12 = hist_values[-12:]
    coef = np.polyfit(np.arange(12), last12, 1)
    return float(np.polyval(coef, 12))


def multi_signal(y: pd.Series, csi: pd.Series, weights=(0.5, 0.3, 0.2)) -> pd.Series:
    sig_y = y.rolling(12).std()
    abs_dy = y.diff(3).abs()
    sig_csi = csi.rolling(12).std()
    def expanding_z(s, min_periods=24):
        mu = s.expanding(min_periods=min_periods).mean()
        sd = s.expanding(min_periods=min_periods).std().clip(lower=1e-6)
        return ((s - mu) / sd).fillna(0)
    return (weights[0] * expanding_z(sig_y)
            + weights[1] * expanding_z(sig_csi)
            + weights[2] * expanding_z(abs_dy))


# ---- Load HCX forecasts ----
def load_2025_forced_cot(panel: pd.DataFrame) -> list[dict]:
    p = ROOT / "data" / "rolling_2025_csi_forced_results.json"
    d = json.loads(p.read_text())
    rows = []
    for tgt in d["targets"]:
        fcs = d["forecasts"].get(tgt, [])
        if not fcs: continue
        actual = float(d["actuals"][tgt])
        origin = pd.Period(tgt, "M") - 1
        rows.append({
            "source": "rolling_2025_csi_forced",
            "origin": origin, "target": pd.Period(tgt, "M"),
            "actual": actual,
            "hcx_forecasts": list(map(float, fcs)),
            "hcx_mean": float(np.mean(fcs)),
        })
    return rows


def load_2022_h1_round0(panel: pd.DataFrame) -> list[dict]:
    db = ROOT / "data" / "experiments.db"
    if not db.exists(): return []
    conn = sqlite3.connect(db); conn.row_factory = sqlite3.Row
    rows_db = conn.execute("""
        SELECT c.id, m.per_target_json FROM candidates c
        JOIN metrics m ON c.id = m.candidate_id
        WHERE c.round_idx = 0
    """).fetchall()
    out = []
    for r in rows_db:
        per_target = json.loads(r["per_target_json"])
        for tgt, info in per_target.items():
            fcs = info.get("forecasts") or []
            if not fcs or info.get("mean") is None: continue
            origin = pd.Period(tgt, "M") - 1
            out.append({
                "source": "stress_2022_h1_round0",
                "origin": origin, "target": pd.Period(tgt, "M"),
                "actual": float(info["actual"]),
                "hcx_forecasts": list(map(float, fcs)),
                "hcx_mean": float(info["mean"]),
            })
    return out


def load_2023_2024_blinded(panel: pd.DataFrame) -> list[dict]:
    p = ROOT / "data" / "stress_2023_2024_blinded_results.json"
    if not p.exists(): return []
    d = json.loads(p.read_text())
    out = []
    # forecasts[year] = list of n_seeds, each a 12-elem list of monthly forecasts
    # actuals[year] = 12-elem list
    for year, fc_seeds in d["forecasts"].items():
        actuals = d["actuals"][year]
        # Stack seeds: shape (n_seeds, 12)
        arr = np.array(fc_seeds, dtype=float)
        for m in range(12):
            target = pd.Period(f"{year}-{m+1:02d}", "M")
            origin = target - 1
            seed_fcs = arr[:, m].tolist()
            out.append({
                "source": f"stress_{year}_blinded_step{m+1}",
                "origin": origin, "target": target,
                "actual": float(actuals[m]),
                "hcx_forecasts": seed_fcs,
                "hcx_mean": float(np.mean(seed_fcs)),
                "horizon": m + 1,  # 1..12-step ahead from year-start origin
            })
    return out


def annotate_baselines(rows: list[dict], y: pd.Series, d_score: pd.Series) -> pd.DataFrame:
    yy = y.values
    out = []
    for r in rows:
        origin = r["origin"]
        if origin not in y.index: continue
        idx = y.index.get_loc(origin)
        if idx < 12: continue
        # Trend12 1-step ahead from origin
        t12 = trend12_forecast(yy[: idx + 1])
        target_idx = y.index.get_loc(r["target"])
        actual = r["actual"]
        hcx_err = abs(r["hcx_mean"] - actual)
        t12_err = abs(t12 - actual)
        ds = float(d_score.loc[origin]) if origin in d_score.index else np.nan
        out.append({
            **r,
            "trend12_pred": t12,
            "trend12_err": t12_err,
            "hcx_err": hcx_err,
            "d_score": ds,
            "delta_err": hcx_err - t12_err,  # negative = HCX better
        })
    return pd.DataFrame(out)


def stratified_summary(df: pd.DataFrame, label: str, d_threshold: float = 0.362) -> dict:
    if df.empty: return {}
    df = df.dropna(subset=["d_score"]).copy()
    df["flagged"] = df["d_score"] > d_threshold
    print(f"\n=== {label} (n={len(df)}, threshold τ_D={d_threshold}) ===")
    print(f"{'stratum':18s} {'n':>3s} {'HCX RMSE':>10s} {'Trend12 RMSE':>14s} "
          f"{'mean Δerr':>10s} {'paired t':>10s} {'p':>8s}")
    print("-" * 78)
    out = {}
    for flag, sub in df.groupby("flagged"):
        if len(sub) == 0: continue
        hcx_rmse = float(np.sqrt(np.mean(sub["hcx_err"] ** 2)))
        t12_rmse = float(np.sqrt(np.mean(sub["trend12_err"] ** 2)))
        mean_delta = float(sub["delta_err"].mean())
        if len(sub) >= 3:
            t, p = stats.ttest_rel(sub["hcx_err"].values, sub["trend12_err"].values)
            t = float(t); p = float(p)
        else:
            t, p = float("nan"), float("nan")
        name = "shock-flagged" if flag else "non-flagged  "
        print(f"{name:18s} {len(sub):>3d} {hcx_rmse:>10.4f} {t12_rmse:>14.4f} "
              f"{mean_delta:>+10.4f} {t:>+10.3f} {p:>8.4f}")
        out[name.strip()] = {
            "n": int(len(sub)),
            "hcx_rmse": hcx_rmse,
            "trend12_rmse": t12_rmse,
            "mean_delta_err": mean_delta,
            "paired_t": t, "p": p,
        }
    return out


def main() -> None:
    df = pd.read_csv(DATA, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="M")
    y = df["CPI_KR_YoY"].astype(float)
    csi = df["BoK_CSI"].astype(float)
    d_score = multi_signal(y, csi)

    # Threshold from regime_detector_backtest.py optimal Youden's J
    bk = json.loads((ROOT / "data" / "regime_detector_backtest.json").read_text())
    tau_d = bk["detector_summary"]["D_MultiSignal_zsum"]["optimal_threshold"]
    print(f"D MultiSignal threshold (from backtest): τ_D = {tau_d:.4f}")
    print(f"  → flagged months in 2003-2025 = {int((d_score > tau_d).sum())}/{len(d_score)}")

    # === Main analysis: 1-step rolling forced CoT ===
    main_rows = (load_2025_forced_cot(df) + load_2022_h1_round0(df))
    main_df = annotate_baselines(main_rows, y, d_score)
    main_summary = stratified_summary(main_df, "MAIN: 1-step rolling forced CoT (2022H1 + 2025)", tau_d)

    # Per-target dump for the main set
    print(f"\n--- per-target detail ({len(main_df)}) ---")
    print(f"{'origin':10s} {'target':10s} {'actual':>7s} {'HCX':>7s} {'T12':>7s} "
          f"{'D':>6s} {'flag':>5s} {'src':30s}")
    for r in main_df.sort_values("origin").itertuples():
        flag = "🔴" if r.d_score > tau_d else "—"
        print(f"{str(r.origin):10s} {str(r.target):10s} "
              f"{r.actual:>7.3f} {r.hcx_mean:>7.3f} {r.trend12_pred:>7.3f} "
              f"{r.d_score:>6.2f} {flag:>5s} {r.source:30s}")

    # === Bonus: 12-step blinded 2023 + 2024 ===
    blind_rows = load_2023_2024_blinded(df)
    blind_df = annotate_baselines(blind_rows, y, d_score)
    if not blind_df.empty:
        # Restrict to short-horizon steps (1-3) to keep apples-to-apples-ish
        for h_max in (3, 6, 12):
            sub = blind_df[blind_df["horizon"] <= h_max]
            stratified_summary(sub, f"BONUS: blinded 2023+2024 horizon≤{h_max}", tau_d)

    # === Save ===
    out = ROOT / "data" / "hcx_value_by_regime.json"
    payload = {
        "tau_d": tau_d,
        "main_summary": main_summary,
        "main_per_target": main_df.assign(
            origin=main_df["origin"].astype(str),
            target=main_df["target"].astype(str),
        ).drop(columns=["hcx_forecasts"]).to_dict(orient="records"),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
