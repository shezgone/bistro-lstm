"""
Ablation Study for BISTRO-LSTM
===============================
bistro-xai의 ablation_study.py를 미러링.

1. Leave-one-out: 변수 하나씩 제거 후 재학습 → ΔRMSE 측정
2. Incremental: attention 순위대로 변수 추가 → RMSE 변화 곡선

Usage:
    .venv/bin/python3 ablation_study.py [--epochs 200]
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch

from lstm_core import (
    LSTMConfig, SEQ_LEN, PRED_LEN, TARGET_COL,
    FORECAST_START, TRAIN_END,
    results_available, load_inference_results, load_stage1_screening,
)
from lstm_model import AttentionLSTMForecaster
from lstm_trainer import set_seed, train_model
from preprocessing_util import (
    load_macro_panel, split_train_test, ZScoreNormalizer,
    create_sequences, prepare_walk_forward_splits,
)


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MACRO_CSV = os.path.join(DATA_DIR, "macro_panel.csv")


def train_and_evaluate_subset(
    variates: list,
    max_epochs: int = 200,
    patience: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    device: torch.device = None,
    seed: int = 42,
) -> dict:
    """
    주어진 변수 서브셋으로 학습 후 2023 OOS RMSE/MAE 반환.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    df = load_macro_panel(MACRO_CSV, TARGET_COL, variates)
    actual_vars = list(df.columns)

    train_df, test_df = split_train_test(df, TRAIN_END, FORECAST_START, "2023-12")

    normalizer = ZScoreNormalizer()
    train_normed = normalizer.fit_transform(train_df).values

    train_X, train_y = create_sequences(train_normed, SEQ_LEN, PRED_LEN, target_idx=0)

    # Validation: 마지막 fold
    splits = prepare_walk_forward_splits(
        train_df, ZScoreNormalizer, SEQ_LEN, PRED_LEN, target_idx=0,
        val_years=[2022],
    )
    val_X = splits[0]["val_X"] if splits else train_X[-3:]
    val_y = splits[0]["val_y"] if splits else train_y[-3:]

    config = LSTMConfig(variates=actual_vars)
    model = AttentionLSTMForecaster.from_config(config)

    result = train_model(
        model=model,
        train_X=train_X, train_y=train_y,
        val_X=val_X, val_y=val_y,
        lr=lr, weight_decay=weight_decay,
        batch_size=batch_size, max_epochs=max_epochs,
        patience=patience, device=device, verbose=False,
    )

    model.load_state_dict(result["best_model_state"])
    model = model.to(device)
    model.eval()

    # 2023 예측
    input_seq = train_normed[-SEQ_LEN:]
    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)["mu"].cpu().numpy()[0]

    pred_raw = normalizer.inverse_transform_target(pred)
    actual = test_df[TARGET_COL].values[:PRED_LEN]

    rmse = np.sqrt(np.mean((pred_raw - actual) ** 2))
    mae = np.mean(np.abs(pred_raw - actual))

    return {"rmse": rmse, "mae": mae, "pred": pred_raw}


def run_ablation(
    base_vars: list,
    max_epochs: int = 200,
    patience: int = 15,
    device: torch.device = None,
) -> dict:
    """
    Leave-one-out ablation study.

    Parameters
    ----------
    base_vars : Stage 2에서 사용된 변수 리스트 (target 포함)

    Returns
    -------
    dict with baseline_rmse/mae, abl_vars, abl_rmse/mae/delta_rmse
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("ABLATION STUDY: Leave-One-Out")
    print("=" * 60)
    print(f"Base variables ({len(base_vars)}): {base_vars}")

    # Baseline: 모든 변수 포함
    print("\n▶ Baseline (all variables)...")
    baseline = train_and_evaluate_subset(
        base_vars, max_epochs=max_epochs, patience=patience, device=device
    )
    baseline_rmse = baseline["rmse"]
    baseline_mae = baseline["mae"]
    print(f"  Baseline RMSE: {baseline_rmse:.4f}pp")

    # Leave-one-out
    covariates = [v for v in base_vars if v != TARGET_COL]
    abl_rmse = []
    abl_mae = []
    abl_delta_rmse = []

    for i, var in enumerate(covariates):
        print(f"\n▶ Removing {var} ({i+1}/{len(covariates)})...")
        remaining = [v for v in base_vars if v != var]

        result = train_and_evaluate_subset(
            remaining, max_epochs=max_epochs, patience=patience, device=device
        )

        delta = result["rmse"] - baseline_rmse
        abl_rmse.append(result["rmse"])
        abl_mae.append(result["mae"])
        abl_delta_rmse.append(delta)

        sign = "+" if delta > 0 else ""
        print(f"  RMSE: {result['rmse']:.4f}pp (Δ{sign}{delta:.4f})")

    return {
        "baseline_rmse": baseline_rmse,
        "baseline_mae": baseline_mae,
        "abl_vars": covariates,
        "abl_rmse": np.array(abl_rmse),
        "abl_mae": np.array(abl_mae),
        "abl_delta_rmse": np.array(abl_delta_rmse),
    }


def run_incremental(
    ranking_vars: list,
    ranking_values: np.ndarray = None,
    max_epochs: int = 200,
    patience: int = 15,
    device: torch.device = None,
) -> dict:
    """
    Incremental addition: 중요도 순으로 변수를 하나씩 추가.

    Returns
    -------
    dict with inc_labels, inc_n_vars, inc_rmse, inc_mae
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("INCREMENTAL ADDITION")
    print("=" * 60)

    inc_labels = []
    inc_n_vars = []
    inc_rmse = []
    inc_mae = []

    # Target only (univariate)
    print("\n▶ Target only (univariate)...")
    result = train_and_evaluate_subset(
        [TARGET_COL], max_epochs=max_epochs, patience=patience, device=device
    )
    inc_labels.append("Target only")
    inc_n_vars.append(1)
    inc_rmse.append(result["rmse"])
    inc_mae.append(result["mae"])
    print(f"  RMSE: {result['rmse']:.4f}pp")

    # 변수 추가
    current_vars = [TARGET_COL]
    for i, var in enumerate(ranking_vars):
        current_vars.append(var)
        print(f"\n▶ Adding {var} ({i+1}/{len(ranking_vars)})...")

        result = train_and_evaluate_subset(
            current_vars.copy(), max_epochs=max_epochs, patience=patience, device=device
        )

        inc_labels.append(f"+{var}")
        inc_n_vars.append(len(current_vars))
        inc_rmse.append(result["rmse"])
        inc_mae.append(result["mae"])
        print(f"  RMSE: {result['rmse']:.4f}pp (n_vars={len(current_vars)})")

    return {
        "inc_labels": inc_labels,
        "inc_n_vars": np.array(inc_n_vars),
        "inc_rmse": np.array(inc_rmse),
        "inc_mae": np.array(inc_mae),
    }


def main():
    parser = argparse.ArgumentParser(description="BISTRO-LSTM Ablation Study")
    parser.add_argument("--epochs", type=int, default=200, help="Max epochs per run")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--skip-incremental", action="store_true", help="Skip incremental study")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stage 2 결과에서 사용된 변수 가져오기
    if results_available("lstm_inference_results.npz"):
        inf_data = load_inference_results()
        base_vars = inf_data["variates"]
    else:
        print("Warning: inference results not found. Using full 29 variables.")
        df = load_macro_panel(MACRO_CSV, TARGET_COL)
        base_vars = list(df.columns)

    # Stage 1 결과에서 attention 랭킹 가져오기
    attn_ranking = None
    attn_values = None
    if results_available("lstm_stage1_screening.npz"):
        s1 = load_stage1_screening()
        attn_ranking = s1["ranking_vars"]
        attn_values = s1["ranking_scores"]

    # Ablation
    abl_result = run_ablation(
        base_vars, max_epochs=args.epochs, patience=args.patience, device=device
    )

    # Incremental
    inc_result = None
    if not args.skip_incremental and attn_ranking is not None:
        # Stage 2에 포함된 변수만으로 incremental
        covariates = [v for v in base_vars if v != TARGET_COL]
        inc_vars = [v for v in attn_ranking if v in covariates]
        inc_result = run_incremental(
            inc_vars, max_epochs=args.epochs, patience=args.patience, device=device
        )

    # 저장
    save_dict = {
        "baseline_rmse": abl_result["baseline_rmse"],
        "baseline_mae": abl_result["baseline_mae"],
        "attn_ranking": np.array(attn_ranking or abl_result["abl_vars"]),
        "attn_values": np.array(attn_values if attn_values is not None else np.zeros(len(abl_result["abl_vars"]))),
        "abl_vars": np.array(abl_result["abl_vars"]),
        "abl_rmse": abl_result["abl_rmse"],
        "abl_mae": abl_result["abl_mae"],
        "abl_delta_rmse": abl_result["abl_delta_rmse"],
    }

    if inc_result is not None:
        save_dict.update({
            "inc_labels": np.array(inc_result["inc_labels"]),
            "inc_n_vars": inc_result["inc_n_vars"],
            "inc_rmse": inc_result["inc_rmse"],
            "inc_mae": inc_result["inc_mae"],
        })

    filepath = os.path.join(DATA_DIR, "lstm_ablation_results.npz")
    np.savez_compressed(filepath, **save_dict)
    print(f"\nAblation results saved to {filepath}")

    # Summary
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline RMSE: {abl_result['baseline_rmse']:.4f}pp")
    print(f"\nVariable Impact (ΔRMSE when removed):")
    sorted_idx = np.argsort(-abl_result["abl_delta_rmse"])
    for idx in sorted_idx:
        var = abl_result["abl_vars"][idx]
        delta = abl_result["abl_delta_rmse"][idx]
        sign = "+" if delta > 0 else ""
        impact = "helps" if delta > 0 else "hurts"
        print(f"  {var:20s} Δ{sign}{delta:.4f}pp ({impact})")


if __name__ == "__main__":
    main()
