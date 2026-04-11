"""
Foundation Model Inference for Korean CPI Forecasting
=====================================================
Chronos-2, TimesFM, Sundial을 zero-shot으로 추론하여
LSTM/BISTRO와 비교.

Usage:
    .venv/bin/python3 run_foundation_models.py
"""

import os
import time
import numpy as np
import pandas as pd
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TARGET_COL = "CPI_KR_YoY"
PRED_LEN = 12


def load_data():
    """CPI 데이터 로딩 (monthly)."""
    df = pd.read_csv(
        os.path.join(DATA_DIR, "macro_panel_optimal18.csv"),
        index_col=0, parse_dates=True,
    )
    df = df.ffill().bfill()

    train = df[df.index <= "2022-12-31"]
    test = df[(df.index >= "2023-01-01") & (df.index <= "2023-12-31")]
    actual = test[TARGET_COL].values[:PRED_LEN]

    return df, train, test, actual


def run_chronos(train: pd.DataFrame, actual: np.ndarray) -> dict:
    """Chronos-2 zero-shot inference."""
    print("\n" + "=" * 60)
    print("CHRONOS-2 (Amazon, 205M params)")
    print("=" * 60)

    from chronos import ChronosPipeline

    t0 = time.time()
    print("[0s] Loading model...")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    print(f"[{time.time()-t0:.0f}s] Model loaded.")

    # CPI 시계열 준비
    cpi_values = train[TARGET_COL].values.astype(np.float32)
    context = torch.tensor(cpi_values).unsqueeze(0)

    print(f"[{time.time()-t0:.0f}s] Running inference (context={len(cpi_values)}, horizon={PRED_LEN})...")
    forecast = pipeline.predict(
        context,
        prediction_length=PRED_LEN,
        num_samples=200,
    )
    # forecast shape: (1, n_samples, pred_len)
    samples = forecast.numpy()[0]  # (n_samples, pred_len)

    med = np.median(samples, axis=0)
    ci_lo = np.percentile(samples, 5, axis=0)
    ci_hi = np.percentile(samples, 95, axis=0)

    rmse = np.sqrt(np.mean((med - actual) ** 2))
    print(f"[{time.time()-t0:.0f}s] Done! RMSE: {rmse:.4f}pp")
    print(f"  Pred: {med}")

    return {
        "name": "Chronos-2",
        "forecast_med": med,
        "forecast_ci_lo": ci_lo,
        "forecast_ci_hi": ci_hi,
        "rmse": rmse,
        "params": "205M",
    }


def run_timesfm(train: pd.DataFrame, actual: np.ndarray) -> dict:
    """TimesFM zero-shot inference."""
    print("\n" + "=" * 60)
    print("TimesFM (Google, 200M params)")
    print("=" * 60)

    import timesfm

    t0 = time.time()
    print("[0s] Loading model...")
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=1,
            horizon_len=PRED_LEN,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
        ),
    )
    print(f"[{time.time()-t0:.0f}s] Model loaded.")

    cpi_values = train[TARGET_COL].values.astype(np.float32)

    print(f"[{time.time()-t0:.0f}s] Running inference (context={len(cpi_values)}, horizon={PRED_LEN})...")
    point_forecast, experimental_quantile_forecast = tfm.forecast(
        [cpi_values],
        freq=[0],  # 0 = monthly
    )

    med = point_forecast[0][:PRED_LEN]

    # Quantile forecasts for CI
    if experimental_quantile_forecast is not None and len(experimental_quantile_forecast) > 0:
        q_forecast = experimental_quantile_forecast[0]  # (pred_len, n_quantiles)
        ci_lo = q_forecast[:PRED_LEN, 0]  # first quantile
        ci_hi = q_forecast[:PRED_LEN, -1]  # last quantile
    else:
        ci_lo = med - 0.5
        ci_hi = med + 0.5

    rmse = np.sqrt(np.mean((med - actual) ** 2))
    print(f"[{time.time()-t0:.0f}s] Done! RMSE: {rmse:.4f}pp")
    print(f"  Pred: {med}")

    return {
        "name": "TimesFM",
        "forecast_med": med,
        "forecast_ci_lo": ci_lo,
        "forecast_ci_hi": ci_hi,
        "rmse": rmse,
        "params": "200M",
    }


def run_sundial(train: pd.DataFrame, actual: np.ndarray) -> dict:
    """Sundial zero-shot inference via HuggingFace transformers."""
    print("\n" + "=" * 60)
    print("Sundial (Tsinghua, 128M params)")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoConfig
    import torch

    t0 = time.time()
    print("[0s] Loading model...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            "thuml/sundial-base-128m",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        print(f"[{time.time()-t0:.0f}s] Model loaded.")
    except Exception as e:
        print(f"  Model load failed: {e}")
        print("  Trying alternative approach...")

        # Fallback: use pipeline
        from transformers import pipeline as hf_pipeline
        try:
            pipe = hf_pipeline(
                "time-series-forecasting",
                model="thuml/sundial-base-128m",
                trust_remote_code=True,
                device="cpu",
            )
            cpi_values = train[TARGET_COL].values.astype(np.float32)
            result = pipe(cpi_values.tolist(), prediction_length=PRED_LEN)
            med = np.array(result["mean"])[:PRED_LEN]
            rmse = np.sqrt(np.mean((med - actual) ** 2))
            print(f"[{time.time()-t0:.0f}s] Done! RMSE: {rmse:.4f}pp")
            return {
                "name": "Sundial",
                "forecast_med": med,
                "forecast_ci_lo": med - 0.5,
                "forecast_ci_hi": med + 0.5,
                "rmse": rmse,
                "params": "128M",
            }
        except Exception as e2:
            print(f"  Pipeline also failed: {e2}")
            return None

    # Sundial inference
    cpi_values = train[TARGET_COL].values.astype(np.float32)
    context = torch.tensor(cpi_values, dtype=torch.float32).unsqueeze(0)

    print(f"[{time.time()-t0:.0f}s] Running inference...")
    model.eval()
    with torch.no_grad():
        try:
            output = model.generate(
                context,
                max_new_tokens=PRED_LEN,
                num_samples=200,
            )
            if isinstance(output, dict):
                samples = output.get("samples", output.get("sequences", None))
                if samples is not None:
                    samples = samples.numpy()
                    if samples.ndim == 3:
                        samples = samples[0]
                    med = np.median(samples, axis=0)[:PRED_LEN]
                    ci_lo = np.percentile(samples, 5, axis=0)[:PRED_LEN]
                    ci_hi = np.percentile(samples, 95, axis=0)[:PRED_LEN]
                else:
                    med = output.numpy().flatten()[:PRED_LEN]
                    ci_lo = med - 0.5
                    ci_hi = med + 0.5
            elif isinstance(output, torch.Tensor):
                if output.ndim == 3:
                    samples = output.numpy()[0]
                    med = np.median(samples, axis=0)[:PRED_LEN]
                    ci_lo = np.percentile(samples, 5, axis=0)[:PRED_LEN]
                    ci_hi = np.percentile(samples, 95, axis=0)[:PRED_LEN]
                else:
                    med = output.numpy().flatten()[:PRED_LEN]
                    ci_lo = med - 0.5
                    ci_hi = med + 0.5
            else:
                med = np.array(output).flatten()[:PRED_LEN]
                ci_lo = med - 0.5
                ci_hi = med + 0.5
        except Exception as e:
            print(f"  generate() failed: {e}, trying forward pass...")
            try:
                output = model(context)
                if hasattr(output, 'logits'):
                    med = output.logits.numpy().flatten()[:PRED_LEN]
                else:
                    med = output[0].numpy().flatten()[:PRED_LEN]
                ci_lo = med - 0.5
                ci_hi = med + 0.5
            except Exception as e3:
                print(f"  Forward pass also failed: {e3}")
                return None

    rmse = np.sqrt(np.mean((med - actual) ** 2))
    print(f"[{time.time()-t0:.0f}s] Done! RMSE: {rmse:.4f}pp")
    print(f"  Pred: {med}")

    return {
        "name": "Sundial",
        "forecast_med": med,
        "forecast_ci_lo": ci_lo,
        "forecast_ci_hi": ci_hi,
        "rmse": rmse,
        "params": "128M",
    }


def main():
    df, train, test, actual = load_data()
    dates = [f"2023-{m:02d}" for m in range(1, 13)]

    print(f"Data: {len(train)} train months, forecast 2023 (12 months)")
    print(f"Actual CPI: {actual}")

    # BISTRO baseline
    bistro18 = np.load("../bistro-xai/data/forecast_optimal18.npz")["forecast_med"]
    b_rmse = np.sqrt(np.mean((bistro18 - actual) ** 2))
    print(f"BISTRO RMSE: {b_rmse:.4f}pp")

    # LSTM baseline
    lstm_data = np.load(os.path.join(DATA_DIR, "lstm_inference_results.npz"))
    lstm_pred = lstm_data["forecast_med"]
    l_rmse = np.sqrt(np.mean((lstm_pred - actual) ** 2))
    print(f"LSTM RMSE: {l_rmse:.4f}pp")

    results = {}

    # 1. Chronos-2
    try:
        r = run_chronos(train, actual)
        if r:
            results["chronos"] = r
    except Exception as e:
        print(f"Chronos failed: {e}")

    # 2. TimesFM
    try:
        r = run_timesfm(train, actual)
        if r:
            results["timesfm"] = r
    except Exception as e:
        print(f"TimesFM failed: {e}")

    # 3. Sundial
    try:
        r = run_sundial(train, actual)
        if r:
            results["sundial"] = r
    except Exception as e:
        print(f"Sundial failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':>20s} {'RMSE':>10s} {'Params':>10s}")
    print(f"{'LSTM (Ours)':>20s} {l_rmse:10.4f} {'0.33M':>10s}")
    print(f"{'BISTRO':>20s} {b_rmse:10.4f} {'91M':>10s}")
    for k, r in results.items():
        print(f"{r['name']:>20s} {r['rmse']:10.4f} {r['params']:>10s}")

    # Save
    save_dict = {}
    for k, r in results.items():
        save_dict[f"{k}_med"] = r["forecast_med"]
        save_dict[f"{k}_ci_lo"] = r["forecast_ci_lo"]
        save_dict[f"{k}_ci_hi"] = r["forecast_ci_hi"]
        save_dict[f"{k}_rmse"] = np.array(r["rmse"])
        save_dict[f"{k}_name"] = np.array(r["name"])
        save_dict[f"{k}_params"] = np.array(r["params"])

    save_dict["forecast_date"] = np.array(dates)
    save_dict["forecast_actual"] = actual
    save_dict["model_names"] = np.array(list(results.keys()))

    np.savez_compressed(
        os.path.join(DATA_DIR, "foundation_model_results.npz"),
        **save_dict,
    )
    print(f"\nResults saved to data/foundation_model_results.npz")


if __name__ == "__main__":
    main()
