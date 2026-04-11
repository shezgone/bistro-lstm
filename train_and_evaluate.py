"""
BISTRO-LSTM: Main Entry Point
==============================
학습 → 평가 → 결과 저장 파이프라인.

Usage:
    .venv/bin/python3 train_and_evaluate.py [--tune] [--top-k 10] [--epochs 300]
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch

from lstm_core import LSTMConfig, SEQ_LEN, PRED_LEN, TARGET_COL, FORECAST_START, TRAIN_END
from lstm_model import AttentionLSTMForecaster
from lstm_trainer import set_seed, train_model, walk_forward_cv, run_optuna_tuning
from preprocessing_util import (
    load_macro_panel, split_train_test, ZScoreNormalizer,
    create_sequences, prepare_walk_forward_splits,
)
from inference_util import ar1_forecast


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MACRO_CSV = os.path.join(DATA_DIR, "macro_panel.csv")


def run_training(
    selected_vars: list = None,
    tune: bool = False,
    n_trials: int = 50,
    max_epochs: int = 300,
    patience: int = 20,
    batch_size: int = 32,
    seed: int = 42,
    device: torch.device = None,
    verbose: bool = True,
) -> dict:
    """
    전체 학습 + 평가 파이프라인.

    Parameters
    ----------
    selected_vars : 사용할 변수 리스트 (None이면 전체)
    tune : Optuna 튜닝 수행 여부
    n_trials : Optuna trial 수
    max_epochs : 최대 에폭
    patience : Early stopping patience
    batch_size : 배치 크기
    seed : 랜덤 시드
    device : PyTorch device
    verbose : 진행 상황 출력

    Returns
    -------
    dict with model, config, forecast, metrics, normalizer
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    print(f"Device: {device}")

    # ── 1. 데이터 로딩 ──
    df = load_macro_panel(MACRO_CSV, TARGET_COL, selected_vars)
    variates = list(df.columns)
    n_vars = len(variates)
    print(f"Variables ({n_vars}): {variates[:5]}{'...' if n_vars > 5 else ''}")

    train_df, test_df = split_train_test(df, TRAIN_END, FORECAST_START, "2023-12")
    print(f"Train: {train_df.index[0]} → {train_df.index[-1]} ({len(train_df)} months)")
    print(f"Test:  {test_df.index[0]} → {test_df.index[-1]} ({len(test_df)} months)")

    # ── 2. Walk-Forward CV 데이터 준비 ──
    splits = prepare_walk_forward_splits(
        train_df, ZScoreNormalizer, SEQ_LEN, PRED_LEN, target_idx=0
    )
    print(f"Walk-forward folds: {len(splits)}")

    # ── 3. 하이퍼파라미터 결정 ──
    if tune:
        print("\n▶ Optuna 튜닝 시작...")
        tune_result = run_optuna_tuning(
            splits, variates, n_trials=n_trials, device=device
        )
        best_params = tune_result["best_params"]
        config = LSTMConfig(
            variates=variates,
            hidden_dim=best_params["hidden_dim"],
            n_layers=best_params["n_layers"],
            d_model=best_params["d_model"],
            n_heads=best_params["n_heads"],
            dropout=best_params["dropout"],
        )
        lr = best_params["lr"]
        weight_decay = best_params["weight_decay"]
    else:
        config = LSTMConfig(variates=variates)
        lr = 1e-3
        weight_decay = 1e-5

    print(f"\nModel config: hidden={config.hidden_dim}, layers={config.n_layers}, "
          f"d_model={config.d_model}, heads={config.n_heads}, dropout={config.dropout}")

    # ── 4. Walk-Forward CV ──
    print("\n▶ Walk-Forward Cross-Validation...")
    cv_result = walk_forward_cv(
        splits=splits,
        config=config,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        verbose=verbose,
    )

    # ── 5. 최종 모델 학습 (전체 학습 데이터) ──
    print("\n▶ 최종 모델 학습 (Train 2003-2022)...")
    set_seed(seed)

    normalizer = ZScoreNormalizer()
    train_normed = normalizer.fit_transform(train_df).values
    train_X, train_y = create_sequences(train_normed, SEQ_LEN, PRED_LEN, target_idx=0)

    # 검증용: 마지막 fold의 val 데이터 사용
    val_X = splits[-1]["val_X"]
    val_y = splits[-1]["val_y"]

    model = AttentionLSTMForecaster.from_config(config)
    print(f"Model parameters: {model.count_parameters():,}")

    final_result = train_model(
        model=model,
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        verbose=verbose,
    )

    model.load_state_dict(final_result["best_model_state"])

    # ── 6. 2023 OOS 예측 ──
    print("\n▶ 2023 Out-of-Sample 예측...")
    model = model.to(device)

    # 마지막 SEQ_LEN개월로 입력 시퀀스 구성
    test_normed = normalizer.transform(test_df).values
    full_normed = np.vstack([train_normed, test_normed])

    # 예측 시작점: 학습 데이터 끝에서 SEQ_LEN개월
    input_seq = train_normed[-SEQ_LEN:]
    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)

    prediction = model.predict(input_tensor, n_samples=200)

    # 역변환 (z-score → 원래 스케일)
    mu_raw = normalizer.inverse_transform_target(prediction["mu"].cpu().numpy()[0])
    ci_lo_raw = normalizer.inverse_transform_target(prediction["ci_lo"].cpu().numpy()[0])
    ci_hi_raw = normalizer.inverse_transform_target(prediction["ci_hi"].cpu().numpy()[0])

    # Actual values
    actual = test_df[TARGET_COL].values[:PRED_LEN]

    # AR(1) baseline
    train_cpi = train_df[TARGET_COL]
    pred_index = pd.PeriodIndex(
        [pd.Period(f"2023-{m:02d}", freq="M") for m in range(1, 13)],
        freq="M",
    )
    ar1_preds = ar1_forecast(train_cpi, pred_index).values

    # Metrics
    rmse = np.sqrt(np.mean((mu_raw - actual) ** 2))
    mae = np.mean(np.abs(mu_raw - actual))
    ar1_rmse = np.sqrt(np.mean((ar1_preds - actual) ** 2))

    print(f"\n{'='*50}")
    print(f"2023 OOS Results:")
    print(f"  LSTM RMSE: {rmse:.4f}pp")
    print(f"  LSTM MAE:  {mae:.4f}pp")
    print(f"  AR(1) RMSE: {ar1_rmse:.4f}pp")
    print(f"  Improvement: {(1 - rmse/ar1_rmse)*100:.1f}% over AR(1)")

    # ── 7. Attention 추출 ──
    var_attn = prediction.get("variable_attention")
    temp_attn = prediction.get("temporal_attention")

    if var_attn is not None:
        var_attn = var_attn.cpu().numpy()[0]  # (n_vars, n_vars)
    if temp_attn is not None:
        temp_attn = temp_attn.cpu().numpy()[0]  # (pred_len, seq_len)

    # ── 8. History data (대시보드용) ──
    history_months = 60
    history_df = train_df.iloc[-history_months:]
    history_date = [str(p) for p in history_df.index]
    history_cpi = history_df[TARGET_COL].values

    # ── 결과 ──
    forecast = {
        "date": [str(p) for p in pred_index],
        "med": mu_raw,
        "ci_lo": ci_lo_raw,
        "ci_hi": ci_hi_raw,
        "ar1": ar1_preds,
        "actual": actual,
        "history_date": history_date,
        "history_cpi": history_cpi,
        "variable_attention": var_attn,
        "temporal_attention": temp_attn,
    }

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "ar1_rmse": ar1_rmse,
        "cv_mean_rmse": cv_result["mean_val_rmse"],
        "cv_fold_rmses": cv_result["val_rmses"],
    }

    return {
        "model": model,
        "config": config,
        "forecast": forecast,
        "metrics": metrics,
        "normalizer": normalizer,
    }


def save_results(result: dict, filename: str = "lstm_inference_results.npz"):
    """결과를 .npz로 저장."""
    forecast = result["forecast"]
    config = result["config"]

    save_dict = {
        "variates": np.array(config.variates),
        "n_variates": np.array(config.n_variates),
        "forecast_date": np.array(forecast["date"]),
        "forecast_med": forecast["med"],
        "forecast_ci_lo": forecast["ci_lo"],
        "forecast_ci_hi": forecast["ci_hi"],
        "forecast_ar1": forecast["ar1"],
        "forecast_actual": forecast["actual"],
        "history_date": np.array(forecast["history_date"]),
        "history_cpi": forecast["history_cpi"],
    }

    if forecast.get("variable_attention") is not None:
        save_dict["variable_attention"] = forecast["variable_attention"]
    if forecast.get("temporal_attention") is not None:
        save_dict["temporal_attention"] = forecast["temporal_attention"]

    # Model config
    save_dict["config_dict"] = np.array([config.to_dict()])

    filepath = os.path.join(DATA_DIR, filename)
    np.savez_compressed(filepath, **save_dict)
    print(f"Results saved to {filepath}")

    # Model checkpoint
    model_path = os.path.join(DATA_DIR, "lstm_model_best.pt")
    torch.save({
        "model_state_dict": result["model"].state_dict(),
        "config": config.to_dict(),
        "metrics": result["metrics"],
    }, model_path)
    print(f"Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser(description="BISTRO-LSTM Training & Evaluation")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--top-k", type=int, default=None, help="Use top-K variables (None=all)")
    parser.add_argument("--epochs", type=int, default=300, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # 변수 선택
    selected_vars = None
    if args.top_k is not None:
        # Stage 1 결과가 있으면 사용, 없으면 전체
        from lstm_core import results_available, load_stage1_screening
        if results_available("lstm_stage1_screening.npz"):
            s1 = load_stage1_screening()
            selected_vars = [TARGET_COL] + s1["ranking_vars"][:args.top_k]
            print(f"Using top-{args.top_k} from Stage 1: {selected_vars}")
        else:
            print(f"Stage 1 results not found. Using all variables.")

    result = run_training(
        selected_vars=selected_vars,
        tune=args.tune,
        n_trials=args.n_trials,
        max_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=True,
    )

    save_results(result)

    print("\n✓ Done!")
    print(f"  LSTM RMSE: {result['metrics']['rmse']:.4f}pp")
    print(f"  AR(1) RMSE: {result['metrics']['ar1_rmse']:.4f}pp")
    print(f"  CV Mean RMSE: {result['metrics']['cv_mean_rmse']:.4f}")


if __name__ == "__main__":
    main()
