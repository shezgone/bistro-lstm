"""
BISTRO-LSTM Runner — 2-Stage Inference Pipeline
================================================
bistro-xai의 bistro_runner_30var.py를 미러링.

Stage 1: 전체 29개 변수로 학습 → 중요도 기반 랭킹
Stage 2: Top-K 변수로 재학습 → 최종 예측 + CI

Usage:
    .venv/bin/python3 lstm_runner.py [--top-k 10] [--tune]
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch

from lstm_core import (
    LSTMConfig, SEQ_LEN, PRED_LEN, TARGET_COL,
    FORECAST_START, TRAIN_END,
)
from lstm_model import AttentionLSTMForecaster
from lstm_trainer import set_seed, train_model
from preprocessing_util import (
    load_macro_panel, split_train_test, ZScoreNormalizer,
    create_sequences, prepare_walk_forward_splits,
)
from inference_util import ar1_forecast
from feature_importance import compute_all_importance


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MACRO_CSV = os.path.join(DATA_DIR, "macro_panel.csv")


def run_stage1(
    config: LSTMConfig = None,
    max_epochs: int = 200,
    patience: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    device: torch.device = None,
) -> dict:
    """
    Stage 1: 전체 변수 스크리닝.

    Returns
    -------
    dict with ranking_vars, ranking_scores, variable_attention,
         gradient_importance, permutation_importance, model, normalizer
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("STAGE 1: Full Variable Screening (29 vars)")
    print("=" * 60)

    # 데이터 로딩
    df = load_macro_panel(MACRO_CSV, TARGET_COL)
    variates = list(df.columns)
    n_vars = len(variates)
    print(f"Variables ({n_vars}): {variates}")

    train_df, test_df = split_train_test(df, TRAIN_END, FORECAST_START, "2023-12")

    if config is None:
        config = LSTMConfig(variates=variates)

    # 정규화 + 시퀀스 생성
    normalizer = ZScoreNormalizer()
    train_normed = normalizer.fit_transform(train_df).values
    train_X, train_y = create_sequences(train_normed, SEQ_LEN, PRED_LEN, target_idx=0)

    # Walk-forward splits for validation
    splits = prepare_walk_forward_splits(
        train_df, ZScoreNormalizer, SEQ_LEN, PRED_LEN, target_idx=0
    )
    val_X = splits[-1]["val_X"] if splits else train_X[-5:]
    val_y = splits[-1]["val_y"] if splits else train_y[-5:]

    # 학습
    set_seed(42)
    model = AttentionLSTMForecaster.from_config(config)
    print(f"Model parameters: {model.count_parameters():,}")

    result = train_model(
        model=model,
        train_X=train_X, train_y=train_y,
        val_X=val_X, val_y=val_y,
        lr=lr, weight_decay=weight_decay,
        batch_size=batch_size, max_epochs=max_epochs,
        patience=patience, device=device, verbose=True,
    )

    model.load_state_dict(result["best_model_state"])
    model = model.to(device)

    # 중요도 계산
    print("\n▶ Computing feature importance...")
    input_seq = train_normed[-SEQ_LEN:]
    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)

    importance = compute_all_importance(
        model, input_tensor, variates,
        val_X=val_X, val_y=val_y,
        normalizer=normalizer,
        device=device,
    )

    # 종합 랭킹 (attention + gradient + permutation 평균)
    scores = {}
    covariates = [v for v in variates if v != TARGET_COL]

    for var in covariates:
        var_idx = variates.index(var)
        attn_score = importance["variable_attention"][0, var_idx] if importance["variable_attention"] is not None else 0
        grad_score = importance["gradient_importance"][var_idx] if importance["gradient_importance"] is not None else 0
        perm_score = importance["permutation_importance"].get(var, 0) if importance["permutation_importance"] is not None else 0

        # 정규화된 종합 점수
        scores[var] = attn_score + grad_score + perm_score

    # 정렬
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranking_vars = [v for v, s in ranking]
    ranking_scores = np.array([s for v, s in ranking])

    # 정규화
    if ranking_scores.sum() > 0:
        ranking_scores = ranking_scores / ranking_scores.sum()

    print(f"\nVariable Ranking (top 10):")
    for i, (var, score) in enumerate(ranking[:10]):
        print(f"  {i+1:2d}. {var:20s} {score:.4f}")

    # 저장
    save_dict = {
        "variates": np.array(variates),
        "n_variates": np.array(n_vars),
        "ranking_vars": np.array(ranking_vars),
        "ranking_scores": ranking_scores,
    }
    if importance["variable_attention"] is not None:
        save_dict["variable_attention"] = importance["variable_attention"]
    if importance["gradient_importance"] is not None:
        save_dict["gradient_importance"] = importance["gradient_importance"]
    if importance["permutation_importance"] is not None:
        perm_arr = np.array([importance["permutation_importance"].get(v, 0) for v in variates])
        save_dict["permutation_importance"] = perm_arr

    filepath = os.path.join(DATA_DIR, "lstm_stage1_screening.npz")
    np.savez_compressed(filepath, **save_dict)
    print(f"\nStage 1 results saved to {filepath}")

    return {
        "ranking_vars": ranking_vars,
        "ranking_scores": ranking_scores,
        "importance": importance,
        "model": model,
        "normalizer": normalizer,
        "config": config,
    }


def run_stage2(
    selected_vars: list,
    max_epochs: int = 300,
    patience: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    device: torch.device = None,
) -> dict:
    """
    Stage 2: 선택된 변수로 최종 학습 + 예측.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_vars = [TARGET_COL] + [v for v in selected_vars if v != TARGET_COL]

    print("\n" + "=" * 60)
    print(f"STAGE 2: Refined Inference ({len(all_vars)} vars)")
    print("=" * 60)
    print(f"Variables: {all_vars}")

    # train_and_evaluate의 run_training을 재사용
    from train_and_evaluate import run_training, save_results

    result = run_training(
        selected_vars=all_vars,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        device=device,
        verbose=True,
    )

    save_results(result, "lstm_inference_results.npz")

    return result


def run_counterfactual(
    model: AttentionLSTMForecaster,
    normalizer: ZScoreNormalizer,
    train_df: pd.DataFrame,
    config: LSTMConfig,
    device: torch.device = None,
    sigma_mult: float = 1.0,
) -> dict:
    """
    Counterfactual 분석: 각 공변량을 ±1σ perturbation.

    Returns
    -------
    dict with cf_variates, cf_impacts, cf_preds_plus, cf_preds_minus
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n▶ Counterfactual Analysis...")
    model = model.to(device)
    model.eval()

    train_normed = normalizer.transform(train_df).values
    input_seq = train_normed[-SEQ_LEN:]
    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)

    # Baseline prediction
    with torch.no_grad():
        baseline = model(input_tensor)["mu"].cpu().numpy()[0]

    covariates = [v for v in config.variates if v != TARGET_COL]
    cf_impacts = []
    cf_preds_plus = []
    cf_preds_minus = []

    for var in covariates:
        var_idx = config.variates.index(var)

        for sign, storage in [(1.0, cf_preds_plus), (-1.0, cf_preds_minus)]:
            perturbed = input_tensor.clone()
            # 마지막 12개월에 perturbation 적용
            perturbed[0, -12:, var_idx] += sign * sigma_mult
            with torch.no_grad():
                pred = model(perturbed)["mu"].cpu().numpy()[0]
            storage.append(pred)

        # Impact: RMSE(perturbed) - RMSE(baseline)
        impact = np.mean(np.abs(cf_preds_plus[-1] - baseline) + np.abs(cf_preds_minus[-1] - baseline)) / 2
        cf_impacts.append(impact)

    return {
        "cf_variates": covariates,
        "cf_impacts": np.array(cf_impacts),
        "cf_preds_plus": np.array(cf_preds_plus),
        "cf_preds_minus": np.array(cf_preds_minus),
    }


def run_narrative(
    model: AttentionLSTMForecaster,
    normalizer: ZScoreNormalizer,
    train_df: pd.DataFrame,
    config: LSTMConfig,
    device: torch.device = None,
) -> dict:
    """
    Stage 3: Economic Narrative Analysis.
    Counterfactual + Jacobian lag + Pathway decomposition + narrative generation.
    """
    from causal_narrative import run_full_analysis, save_narrative_results

    results = run_full_analysis(model, normalizer, train_df, config, device=device)
    save_narrative_results(results)
    return results


def main():
    parser = argparse.ArgumentParser(description="BISTRO-LSTM 2-Stage Pipeline")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K variables for Stage 2")
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning")
    parser.add_argument("--epochs", type=int, default=300, help="Max epochs")
    parser.add_argument("--skip-stage1", action="store_true", help="Skip Stage 1 if results exist")
    parser.add_argument("--narrative", action="store_true", help="Run narrative analysis after Stage 2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stage 1
    s1_path = os.path.join(DATA_DIR, "lstm_stage1_screening.npz")
    if args.skip_stage1 and os.path.exists(s1_path):
        print("Stage 1 results found, skipping...")
        s1_data = np.load(s1_path, allow_pickle=True)
        ranking_vars = [str(v) for v in s1_data["ranking_vars"]]
    else:
        s1_result = run_stage1(device=device)
        ranking_vars = s1_result["ranking_vars"]

    # Stage 2
    top_k_vars = ranking_vars[:args.top_k]
    print(f"\nTop-{args.top_k} variables: {top_k_vars}")

    s2_result = run_stage2(
        selected_vars=top_k_vars,
        max_epochs=args.epochs,
        device=device,
    )

    print(f"\n{'='*60}")
    print(f"Pipeline Complete!")
    print(f"  LSTM RMSE: {s2_result['metrics']['rmse']:.4f}pp")
    print(f"  AR(1) RMSE: {s2_result['metrics']['ar1_rmse']:.4f}pp")

    # Stage 3: Narrative Analysis (optional)
    if args.narrative and "model" in s2_result and "normalizer" in s2_result:
        run_narrative(
            s2_result["model"], s2_result["normalizer"],
            s2_result["train_df"], s2_result["config"],
            device=device,
        )


if __name__ == "__main__":
    main()
