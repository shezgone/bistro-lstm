"""LSTM/Transformer rolling 2025 with vs without BoK CSI sentiment column.

4 models trained: LSTM-base, LSTM-csi, TFM-base, TFM-csi.
Each on 2003-01 to 2024-12, then rolling 1-step-ahead inference for
2025-04 to 2025-11 origins (predicting 2025-05 to 2025-12).
"""
import os, sys, json, time
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing_util import (load_macro_panel, split_train_test, ZScoreNormalizer,
                                 create_sequences, prepare_walk_forward_splits)
from lstm_model import AttentionLSTMForecaster
from transformer_model import AttentionTransformerForecaster
from lstm_trainer import train_model, set_seed
from lstm_core import LSTMConfig, TARGET_COL

DATA_DIR = "data"
SEQ_LEN, PRED_LEN = 36, 12
TRAIN_END = "2024-12"
ROLL_START, ROLL_END = pd.Period("2025-04", "M"), pd.Period("2025-11", "M")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# Load both panels
df_base = pd.read_csv(f"{DATA_DIR}/macro_panel_optimal18.csv", index_col=0)
df_base.index = pd.PeriodIndex(df_base.index, freq="M")

df_full_raw = pd.read_csv(f"{DATA_DIR}/macro_panel_full.csv", index_col=0)
df_full_raw.index = pd.PeriodIndex(df_full_raw.index, freq="M")
gt_cols = [c for c in df_full_raw.columns if c.startswith("GT_")]
df_csi = df_full_raw.drop(columns=gt_cols)  # 18 + BoK_CSI = 19 cov + CPI = 20

print(f"Base: {df_base.shape}, cols={len(df_base.columns)}")
print(f"CSI:  {df_csi.shape}, cols={len(df_csi.columns)}", flush=True)


def run_one(df, model_cls, name: str, seed: int = 42):
    print(f"\n{'='*70}\n{name}  (seed={seed})\n{'='*70}", flush=True)
    variates = list(df.columns)
    train_df, _ = split_train_test(df, TRAIN_END, "2025-01", "2025-12")
    print(f"Train: {train_df.index[0]} → {train_df.index[-1]} ({len(train_df)} mo)")

    norm = ZScoreNormalizer()
    train_normed = norm.fit_transform(train_df).values

    splits = prepare_walk_forward_splits(train_df, ZScoreNormalizer, SEQ_LEN, PRED_LEN, target_idx=0)
    val_X = splits[-1]["val_X"]; val_y = splits[-1]["val_y"]
    train_X, train_y = create_sequences(train_normed, SEQ_LEN, PRED_LEN, target_idx=0)
    print(f"Train seqs: {train_X.shape}", flush=True)

    config = LSTMConfig(variates=variates, seq_len=SEQ_LEN, pred_len=PRED_LEN,
                        hidden_dim=128, n_layers=2, d_model=64, n_heads=4, dropout=0.2)
    set_seed(seed)
    model = model_cls.from_config(config)
    n_params = model.count_parameters()
    print(f"Params: {n_params:,}")

    t0 = time.time()
    res = train_model(model=model, train_X=train_X, train_y=train_y,
                      val_X=val_X, val_y=val_y,
                      lr=1e-3, weight_decay=1e-5, batch_size=32,
                      max_epochs=300, patience=20, device=device, verbose=False)
    model.load_state_dict(res["best_model_state"])
    print(f"Best val RMSE: {res['best_val_rmse']:.4f}, time: {time.time()-t0:.1f}s", flush=True)

    # Rolling inference
    full_normed = norm.transform(df).values
    target_idx = list(df.columns).index(TARGET_COL)

    rows = []
    for origin in pd.period_range(ROLL_START, ROLL_END, freq="M"):
        oi = list(df.index).index(origin)
        ctx = full_normed[oi - SEQ_LEN + 1 : oi + 1]
        x = torch.FloatTensor(ctx).unsqueeze(0).to(device)
        pred = model.predict(x, n_samples=200)
        mu_n = float(pred["mu"].cpu().numpy()[0, 0])
        mu_raw = float(norm.inverse_transform_target(np.array([mu_n]))[0])
        target = origin + 1
        actual = float(df.loc[target, TARGET_COL])
        err = mu_raw - actual
        rows.append({"origin": str(origin), "target": str(target),
                     "actual": actual, "pred": mu_raw, "err": err})
        print(f"  {origin}→{target}: pred={mu_raw:.3f} actual={actual:.3f} err={err:+.3f}")

    actuals = np.array([r["actual"] for r in rows])
    preds = np.array([r["pred"] for r in rows])
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    mae = float(np.mean(np.abs(preds - actuals)))
    print(f"\n{name} → RMSE={rmse:.4f}, MAE={mae:.4f}, n_params={n_params:,}", flush=True)
    return {"name": name, "rmse": rmse, "mae": mae, "n_params": n_params, "rows": rows}


out = {}
out["LSTM-base (18)"]      = run_one(df_base, AttentionLSTMForecaster, "LSTM-base (18)")
out["LSTM-csi (18+CSI)"]   = run_one(df_csi,  AttentionLSTMForecaster, "LSTM-csi (18+CSI)")
out["TFM-base (18)"]       = run_one(df_base, AttentionTransformerForecaster, "TFM-base (18)")
out["TFM-csi (18+CSI)"]    = run_one(df_csi,  AttentionTransformerForecaster, "TFM-csi (18+CSI)")

with open(f"{DATA_DIR}/rolling_2025_torch_csi_results.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n" + "=" * 80)
print(f"{'Method':30s} {'RMSE':>8s} {'MAE':>8s} {'params':>10s} {'Δ vs base':>10s}")
print("-" * 80)
print(f"{'LSTM-base (18)':30s} {out['LSTM-base (18)']['rmse']:8.4f} {out['LSTM-base (18)']['mae']:8.4f} {out['LSTM-base (18)']['n_params']:10,}")
delta_l = out['LSTM-csi (18+CSI)']['rmse'] - out['LSTM-base (18)']['rmse']
print(f"{'LSTM-csi (18+CSI)':30s} {out['LSTM-csi (18+CSI)']['rmse']:8.4f} {out['LSTM-csi (18+CSI)']['mae']:8.4f} {out['LSTM-csi (18+CSI)']['n_params']:10,} {delta_l:+10.4f}")
print(f"{'TFM-base (18)':30s} {out['TFM-base (18)']['rmse']:8.4f} {out['TFM-base (18)']['mae']:8.4f} {out['TFM-base (18)']['n_params']:10,}")
delta_t = out['TFM-csi (18+CSI)']['rmse'] - out['TFM-base (18)']['rmse']
print(f"{'TFM-csi (18+CSI)':30s} {out['TFM-csi (18+CSI)']['rmse']:8.4f} {out['TFM-csi (18+CSI)']['mae']:8.4f} {out['TFM-csi (18+CSI)']['n_params']:10,} {delta_t:+10.4f}")
print(f"\nSentiment effect: LSTM Δ = {delta_l:+.4f}, TFM Δ = {delta_t:+.4f}")
print("(negative Δ = sentiment helped)")
