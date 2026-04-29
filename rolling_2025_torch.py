"""Rolling 1-step-ahead forecast with LSTM and Transformer (same arch as report).

Protocol:
- Train both models on data 2003-01 to 2024-12 (full 2025 held out)
- Rolling inference: for each origin in 2025-04..2025-11, feed last 36 months,
  predict 12-step output, take first month (= t+1)
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing_util import load_macro_panel, split_train_test, ZScoreNormalizer, create_sequences, prepare_walk_forward_splits
from lstm_model import AttentionLSTMForecaster
from transformer_model import AttentionTransformerForecaster
from lstm_trainer import train_model, set_seed
from lstm_core import LSTMConfig, TARGET_COL

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV = os.path.join(DATA_DIR, "macro_panel_optimal18.csv")
SEQ_LEN = 36
PRED_LEN = 12
TRAIN_END = "2024-12"
ROLL_START = pd.Period("2025-04", "M")
ROLL_END = pd.Period("2025-11", "M")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# Load data
df = load_macro_panel(CSV, TARGET_COL, None)
variates = list(df.columns)
print(f"Variables ({len(variates)}): {variates}", flush=True)

train_df, test_df = split_train_test(df, TRAIN_END, "2025-01", "2025-12")
print(f"Train: {train_df.index[0]} → {train_df.index[-1]} ({len(train_df)} months)")
print(f"Test:  {test_df.index[0]} → {test_df.index[-1]} ({len(test_df)} months)")

# Normalize
normalizer = ZScoreNormalizer()
train_normed = normalizer.fit_transform(train_df).values
print(f"Train shape (normed): {train_normed.shape}", flush=True)

# Walk-forward splits (for validation during training)
splits = prepare_walk_forward_splits(
    train_df, ZScoreNormalizer, SEQ_LEN, PRED_LEN, target_idx=0
)
val_X = splits[-1]["val_X"]
val_y = splits[-1]["val_y"]

# Create train sequences
train_X, train_y = create_sequences(train_normed, SEQ_LEN, PRED_LEN, target_idx=0)
print(f"Train sequences: {train_X.shape}", flush=True)

config = LSTMConfig(
    variates=variates, seq_len=SEQ_LEN, pred_len=PRED_LEN,
    hidden_dim=128, n_layers=2, d_model=64, n_heads=4, dropout=0.2,
)


def train_and_save(model_cls, name: str, seed: int = 42):
    set_seed(seed)
    model = model_cls.from_config(config)
    n_params = model.count_parameters()
    print(f"\n=== {name} (seed={seed}, params={n_params:,}) ===", flush=True)
    t0 = time.time()
    result = train_model(
        model=model,
        train_X=train_X, train_y=train_y,
        val_X=val_X, val_y=val_y,
        lr=1e-3, weight_decay=1e-5,
        batch_size=32, max_epochs=300, patience=20,
        device=device, verbose=True,
    )
    model.load_state_dict(result["best_model_state"])
    print(f"  best val RMSE: {result['best_val_rmse']:.4f}, train time: {time.time()-t0:.1f}s", flush=True)
    return model, n_params


def rolling_inference(model, name: str):
    """Rolling 1-step-ahead inference for origins 2025-04 to 2025-11."""
    model.eval()
    full_normed = normalizer.transform(df).values  # (T, N) normalized
    target_col_idx = list(df.columns).index(TARGET_COL)

    results = []
    for origin in pd.period_range(ROLL_START, ROLL_END, freq="M"):
        # Find index in df.index for this origin
        origin_idx = list(df.index).index(origin)
        # Use last SEQ_LEN months ending at origin (inclusive)
        ctx_end = origin_idx + 1  # exclusive end
        ctx_start = ctx_end - SEQ_LEN
        if ctx_start < 0:
            print(f"  {origin}: not enough history"); continue
        input_seq = full_normed[ctx_start:ctx_end]  # (SEQ_LEN, N)
        x = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
        pred = model.predict(x, n_samples=200)
        # Take first step of multi-step output
        mu_n = float(pred["mu"].cpu().numpy()[0, 0])
        ci_lo_n = float(pred["ci_lo"].cpu().numpy()[0, 0])
        ci_hi_n = float(pred["ci_hi"].cpu().numpy()[0, 0])
        # Inverse transform target
        mu_raw = float(normalizer.inverse_transform_target(np.array([mu_n]))[0])
        ci_lo_raw = float(normalizer.inverse_transform_target(np.array([ci_lo_n]))[0])
        ci_hi_raw = float(normalizer.inverse_transform_target(np.array([ci_hi_n]))[0])
        target = origin + 1
        actual = float(df.loc[target, TARGET_COL])
        err = mu_raw - actual
        results.append({
            "origin": str(origin), "target": str(target),
            "actual": actual, "pred": mu_raw,
            "ci_lo": ci_lo_raw, "ci_hi": ci_hi_raw, "err": err,
        })
        print(f"  {origin} → {target}: pred={mu_raw:.3f} actual={actual:.3f} err={err:+.3f}")
    return results


def summarize(results, name: str, n_params: int):
    actuals = np.array([r["actual"] for r in results])
    preds = np.array([r["pred"] for r in results])
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    mae = float(np.mean(np.abs(preds - actuals)))
    print(f"\n{name}: RMSE={rmse:.4f}, MAE={mae:.4f}, n={len(results)}, params={n_params:,}")
    return {"name": name, "rmse": rmse, "mae": mae, "n_params": n_params, "rows": results}


# Train both
out = {}
lstm_model, lstm_p = train_and_save(AttentionLSTMForecaster, "LSTM")
lstm_results = rolling_inference(lstm_model, "LSTM")
out["LSTM"] = summarize(lstm_results, "LSTM", lstm_p)

tfm_model, tfm_p = train_and_save(AttentionTransformerForecaster, "Transformer")
tfm_results = rolling_inference(tfm_model, "Transformer")
out["Transformer"] = summarize(tfm_results, "Transformer", tfm_p)

# Save
with open(os.path.join(DATA_DIR, "rolling_2025_torch_results.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved data/rolling_2025_torch_results.json")

# Comparison summary
print("\n" + "=" * 80)
print(f"{'Method':30s} {'RMSE':>8s} {'MAE':>8s} {'params':>10s}")
print("-" * 80)
for name, res in out.items():
    print(f"{name:30s} {res['rmse']:8.4f} {res['mae']:8.4f} {res['n_params']:10,}")
