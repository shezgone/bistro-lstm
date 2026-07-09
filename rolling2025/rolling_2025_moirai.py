"""Moirai zero-shot rolling 1-step CPI forecast WITH BoK_CSI as past covariate.

Compares Moirai-1.0-R family (zero-shot, multivariate native) on:
  - With CSI (past_feat_dynamic_real=['BoK_CSI'])  — uses lesson 7's leading indicator
  - Without CSI (univariate baseline)

vs reference (same 8 origins, rolling 2025-04..2025-11):
  Trend12               0.258
  HCX forced CoT + CSI  0.250
  Chronos-bolt-tiny     0.281 (univariate, no CSI)
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "macro_panel_full.csv"
OUT_JSON = ROOT / "data" / "rolling_2025_moirai_results.json"

ROLL_START = pd.Period("2025-04", "M")
ROLL_END = pd.Period("2025-11", "M")
TARGET_COL = "CPI_KR_YoY"
CSI_COL = "BoK_CSI"

MODELS = [
    ("Salesforce/moirai-1.0-R-small", 14),  # ~14M params
    ("Salesforce/moirai-1.0-R-base", 91),
    ("Salesforce/moirai-1.0-R-large", 311),
]
CTX_VARIANTS = [36, None]  # None = full history (clipped to CSI start)
CSI_VARIANTS = [True, False]

# CSI is dummy=100 before 2020-01; only meaningful from 2020-01 onward.
CSI_VALID_FROM = pd.Period("2020-01", "M")


def pick_device() -> str:
    # MPS path triggers float64 conversion deep in GluonTS batchify; CPU is reliable
    # and fast enough for 8 inferences on these model sizes.
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df[[TARGET_COL, CSI_COL]].astype(float)


def origins() -> list[pd.Period]:
    return list(pd.period_range(ROLL_START, ROLL_END, freq="M"))


def build_predictor(model_id: str, ctx_len: int, use_csi: bool, device: str):
    module = MoiraiModule.from_pretrained(model_id)
    model = MoiraiForecast(
        module=module,
        prediction_length=1,
        context_length=ctx_len,
        patch_size="auto",
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=1 if use_csi else 0,
    )
    return model.create_predictor(batch_size=1, device=device)


def forecast_one(predictor, hist_df: pd.DataFrame, use_csi: bool) -> tuple[float, float, float, float]:
    """Run predictor on history; return (mean, q10, q50, q90) for the next month."""
    # Convert PeriodIndex → DatetimeIndex (gluonts needs timestamps)
    df_dt = hist_df.astype(np.float32).copy()
    df_dt.index = df_dt.index.to_timestamp()
    ds_kwargs = dict(target=TARGET_COL, freq="M", dtype=np.float32)
    if use_csi:
        ds_kwargs["past_feat_dynamic_real"] = [CSI_COL]
    ds = PandasDataset(df_dt, **ds_kwargs)
    forecasts = list(predictor.predict(ds))
    f = forecasts[0]
    samples = f.samples  # (num_samples, prediction_length)
    s = samples[:, 0]
    mean = float(s.mean())
    q10 = float(np.quantile(s, 0.1))
    q50 = float(np.quantile(s, 0.5))
    q90 = float(np.quantile(s, 0.9))
    return mean, q10, q50, q90


def evaluate(predictor, panel: pd.DataFrame, ctx_len: int, use_csi: bool, label: str) -> dict:
    rows = []
    for origin in origins():
        target = origin + 1
        if target not in panel.index:
            print(f"  skip {origin}: target {target} not in data")
            continue
        # Clip history to origin; if using CSI, also clip to CSI_VALID_FROM
        if use_csi:
            hist = panel.loc[CSI_VALID_FROM:origin]
        else:
            hist = panel[[TARGET_COL]].loc[:origin]
        # Apply ctx window
        if len(hist) < ctx_len:
            print(f"  skip {origin}: history too short ({len(hist)} < {ctx_len})")
            continue
        hist = hist.iloc[-max(ctx_len, 36):]  # always at least ctx_len
        actual = float(panel.loc[target, TARGET_COL])
        try:
            mean, q10, q50, q90 = forecast_one(predictor, hist, use_csi)
        except Exception as e:
            print(f"  ERROR {origin}: {e}")
            continue
        err = mean - actual
        rows.append({
            "origin": str(origin), "target": str(target),
            "actual": actual,
            "pred_mean": mean, "pred_q10": q10, "pred_q50": q50, "pred_q90": q90,
            "err": err, "ctx_used": int(len(hist)),
        })
        print(f"  {origin} → {target}: pred={mean:.3f} (q50={q50:.3f}) actual={actual:.3f} err={err:+.3f}")
    if not rows: return {"label": label, "n": 0}
    actuals = np.array([r["actual"] for r in rows])
    means = np.array([r["pred_mean"] for r in rows])
    medians = np.array([r["pred_q50"] for r in rows])
    rmse_mean = float(np.sqrt(np.mean((means - actuals) ** 2)))
    rmse_median = float(np.sqrt(np.mean((medians - actuals) ** 2)))
    mae_mean = float(np.mean(np.abs(means - actuals)))
    print(f"\n  {label}: RMSE(mean)={rmse_mean:.4f}  RMSE(median)={rmse_median:.4f}  "
          f"MAE(mean)={mae_mean:.4f}  n={len(rows)}")
    return {
        "label": label,
        "rmse_mean": rmse_mean, "rmse_median": rmse_median, "mae_mean": mae_mean,
        "n": len(rows), "rows": rows,
    }


def main() -> None:
    panel = load_panel()
    print(f"Loaded panel: {panel.index[0]} → {panel.index[-1]} ({len(panel)} obs)")
    print(f"  CSI valid from: {CSI_VALID_FROM} (history before is dummy=100)")
    device = pick_device()
    print(f"Device: {device}")

    out = {"protocol": "rolling 1-step ahead, origins 2025-04..2025-11 (8)", "results": []}

    for model_id, n_params in MODELS:
        for ctx in CTX_VARIANTS:
            for use_csi in CSI_VARIANTS:
                label = f"{model_id.split('/')[-1]}_ctx{ctx if ctx else 'ALL'}_{'CSI' if use_csi else 'uni'}"
                print(f"\n=== {label} ({n_params}M params) ===")
                t0 = time.time()
                # ctxALL means use all available history (clipped at CSI_VALID_FROM if use_csi)
                ctx_actual = ctx if ctx else 64  # full ~64 months from 2020-01 to 2025-04
                try:
                    predictor = build_predictor(model_id, ctx_actual, use_csi, device)
                except Exception as e:
                    print(f"  build failed on {device}, retry on cpu: {e}")
                    predictor = build_predictor(model_id, ctx_actual, use_csi, "cpu")
                print(f"  predictor built in {time.time() - t0:.1f}s")
                res = evaluate(predictor, panel, ctx_actual, use_csi, label)
                res["model"] = model_id
                res["ctx_len"] = ctx_actual
                res["use_csi"] = use_csi
                res["n_params_M"] = n_params
                out["results"].append(res)
                del predictor
                if device == "mps": torch.mps.empty_cache()

    # Leaderboard
    print("\n" + "=" * 90)
    print(f"{'variant':55s} {'RMSE_mean':>10s} {'RMSE_med':>10s} {'MAE':>8s} {'n':>4s}")
    print("-" * 90)
    valid = [r for r in out["results"] if r.get("n", 0) > 0]
    for r in sorted(valid, key=lambda x: x["rmse_mean"]):
        print(f"{r['label']:55s} {r['rmse_mean']:10.4f} {r['rmse_median']:10.4f} "
              f"{r['mae_mean']:8.4f} {r['n']:4d}")

    print("\nReference (same 8 origins):")
    print("  Trend12               RMSE 0.258")
    print("  HCX forced CoT + CSI  RMSE 0.250  ← target to beat")
    print("  Chronos-bolt-tiny     RMSE 0.281")

    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
