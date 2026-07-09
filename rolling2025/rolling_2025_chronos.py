"""Chronos zero-shot rolling 1-step CPI forecast — same protocol as rolling_2025_torch.py.

Compares Chronos-Bolt family (zero-shot, no in-domain training) against:
  - Trend12 (0.258 RMSE on this set)
  - HCX forced CoT + CSI (0.250)
  - In-domain LSTM (0.484), TFM (0.773)

Univariate input only (CPI_KR_YoY) — Chronos is univariate; CSI/exo not used here.
Two context-length variants:
  - ctx=36  (apples-to-apples with LSTM/TFM)
  - ctx=ALL (full history up to origin; favors Chronos since it's zero-shot)
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "macro_panel_full.csv"
OUT_JSON = ROOT / "data" / "rolling_2025_chronos_results.json"

ROLL_START = pd.Period("2025-04", "M")
ROLL_END = pd.Period("2025-11", "M")
TARGET_COL = "CPI_KR_YoY"

MODELS = [
    "amazon/chronos-bolt-tiny",
    "amazon/chronos-bolt-mini",
    "amazon/chronos-bolt-small",
    "amazon/chronos-bolt-base",
]
CTX_VARIANTS = [36, None]  # None = use full history up to origin


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_series() -> pd.Series:
    df = pd.read_csv(DATA_CSV, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df[TARGET_COL].astype(float)


def origins() -> list[pd.Period]:
    return list(pd.period_range(ROLL_START, ROLL_END, freq="M"))


def predict_one(pipeline, ctx_values: np.ndarray) -> tuple[float, float, float, float]:
    """Predict 1-step ahead. Returns (mean, q10, q50, q90)."""
    ctx = torch.tensor(ctx_values, dtype=torch.float32)
    quantiles, mean = pipeline.predict_quantiles(
        context=ctx,
        prediction_length=1,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    # quantiles shape: (1, 1, 3) ; mean shape: (1, 1)
    q = quantiles[0, 0].cpu().numpy()
    m = float(mean[0, 0].cpu().numpy())
    return m, float(q[0]), float(q[1]), float(q[2])


def evaluate(pipeline, series: pd.Series, ctx_len: int | None, label: str) -> dict:
    rows = []
    for origin in origins():
        target = origin + 1
        if target not in series.index:
            print(f"  skip {origin}: target {target} not in data")
            continue
        # Context = values up to and including origin
        hist = series.loc[:origin].values
        if ctx_len is not None:
            if len(hist) < ctx_len:
                print(f"  skip {origin}: not enough history (need {ctx_len}, have {len(hist)})")
                continue
            ctx_values = hist[-ctx_len:]
        else:
            ctx_values = hist
        actual = float(series.loc[target])
        mean, q10, q50, q90 = predict_one(pipeline, ctx_values)
        err = mean - actual
        rows.append({
            "origin": str(origin), "target": str(target),
            "actual": actual,
            "pred_mean": mean, "pred_q10": q10, "pred_q50": q50, "pred_q90": q90,
            "err": err, "ctx_used": int(len(ctx_values)),
        })
        print(f"  {origin} → {target}: pred={mean:.3f} (q50={q50:.3f}) actual={actual:.3f} err={err:+.3f}")
    actuals = np.array([r["actual"] for r in rows])
    preds = np.array([r["pred_mean"] for r in rows])
    medians = np.array([r["pred_q50"] for r in rows])
    rmse_mean = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    rmse_median = float(np.sqrt(np.mean((medians - actuals) ** 2)))
    mae_mean = float(np.mean(np.abs(preds - actuals)))
    print(f"\n  {label}: RMSE(mean)={rmse_mean:.4f}  RMSE(median)={rmse_median:.4f}  "
          f"MAE(mean)={mae_mean:.4f}  n={len(rows)}")
    return {
        "label": label,
        "rmse_mean": rmse_mean, "rmse_median": rmse_median, "mae_mean": mae_mean,
        "n": len(rows), "rows": rows,
    }


def main() -> None:
    series = load_series()
    print(f"Loaded {TARGET_COL}: {series.index[0]} → {series.index[-1]} ({len(series)} obs)")
    device = pick_device()
    print(f"Device: {device}")

    out = {"protocol": "rolling 1-step ahead, origins 2025-04..2025-11 (8)", "results": []}

    for model_id in MODELS:
        print(f"\n=== {model_id} ===")
        t0 = time.time()
        try:
            pipeline = BaseChronosPipeline.from_pretrained(
                model_id, device_map=device, torch_dtype=torch.float32,
            )
        except Exception as e:
            print(f"  load failed on {device}, retry on cpu: {e}")
            pipeline = BaseChronosPipeline.from_pretrained(
                model_id, device_map="cpu", torch_dtype=torch.float32,
            )
        print(f"  loaded in {time.time() - t0:.1f}s")

        for ctx in CTX_VARIANTS:
            label = f"{model_id.split('/')[-1]}_ctx{ctx if ctx else 'ALL'}"
            print(f"\n  --- {label} ---")
            res = evaluate(pipeline, series, ctx, label)
            res["model"] = model_id
            res["ctx_len"] = ctx
            out["results"].append(res)

        # free memory between models
        del pipeline
        if device == "mps":
            torch.mps.empty_cache()

    # Leaderboard
    print("\n" + "=" * 80)
    print(f"{'variant':45s} {'RMSE_mean':>10s} {'RMSE_med':>10s} {'MAE':>8s} {'n':>4s}")
    print("-" * 80)
    for r in sorted(out["results"], key=lambda x: x["rmse_mean"]):
        print(f"{r['label']:45s} {r['rmse_mean']:10.4f} {r['rmse_median']:10.4f} "
              f"{r['mae_mean']:8.4f} {r['n']:4d}")

    print("\nReference (same 8 origins):")
    print("  Trend12               RMSE 0.258")
    print("  HCX forced CoT + CSI  RMSE 0.250")
    print("  Ensemble HCX+Trend12  RMSE 0.235")

    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
