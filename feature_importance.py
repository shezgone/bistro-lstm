"""
Feature Importance Methods for LSTM
====================================
1. Variable Fusion Attention weights (cross-variate)
2. Gradient-based importance (Integrated Gradients)
3. Permutation importance

bistro-xai의 attention 분석과 직접 비교 가능한 형태로 출력.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from lstm_model import AttentionLSTMForecaster
from preprocessing_util import ZScoreNormalizer


def extract_attention_importance(
    model: AttentionLSTMForecaster,
    input_tensor: torch.Tensor,
) -> Optional[np.ndarray]:
    """
    Variable Fusion Attention weights 추출.

    Parameters
    ----------
    model : 학습된 모델
    input_tensor : (1, seq_len, n_vars)

    Returns
    -------
    (n_vars, n_vars) attention matrix or None
    """
    model.eval()
    with torch.no_grad():
        model(input_tensor, return_attention=True)
        attn = model.var_fusion.get_attention_weights()

    if attn is not None:
        return attn.cpu().numpy()  # (1, n_vars, n_vars)
    return None


def compute_gradient_importance(
    model: AttentionLSTMForecaster,
    input_tensor: torch.Tensor,
    target_idx: int = 0,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Integrated Gradients 기반 변수 중요도.

    각 변수에 대해 baseline(0)에서 실제 값까지의 경로를 따라
    gradient를 적분하여 기여도를 산출.

    Parameters
    ----------
    model : 학습된 모델
    input_tensor : (1, seq_len, n_vars)
    target_idx : 타겟 출력 인덱스
    n_steps : 적분 단계 수

    Returns
    -------
    (n_vars,) importance scores (non-negative)
    """
    model.eval()
    baseline = torch.zeros_like(input_tensor)
    diff = input_tensor - baseline

    grads_sum = torch.zeros_like(input_tensor)

    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        interpolated = baseline + alpha * diff
        interpolated.requires_grad_(True)

        out = model(interpolated)
        # 전체 pred_len에 대한 평균 예측을 타겟으로
        target_output = out["mu"].mean()

        model.zero_grad()
        target_output.backward()

        if interpolated.grad is not None:
            grads_sum += interpolated.grad.detach()
            interpolated.grad.zero_()

    # Integrated gradients = (input - baseline) * avg_gradient
    ig = (diff * grads_sum / n_steps).squeeze(0)  # (seq_len, n_vars)

    # Per-variable importance: 시간 축으로 절대값 합산
    importance = ig.abs().sum(dim=0).cpu().numpy()  # (n_vars,)

    # 정규화
    total = importance.sum()
    if total > 0:
        importance = importance / total

    return importance


def compute_permutation_importance(
    model: AttentionLSTMForecaster,
    val_X: np.ndarray,
    val_y: np.ndarray,
    variates: List[str],
    normalizer: ZScoreNormalizer,
    target_col: str = "CPI_KR_YoY",
    n_repeats: int = 10,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Permutation importance.
    각 변수의 시간 순서를 셔플하여 RMSE 증가량 측정.

    Parameters
    ----------
    model : 학습된 모델
    val_X : (n_samples, seq_len, n_vars) 검증 데이터
    val_y : (n_samples, pred_len) 검증 타겟
    variates : 변수 이름 리스트
    normalizer : 역변환용 정규화기
    target_col : 타겟 변수명
    n_repeats : 셔플 반복 횟수
    device : PyTorch device

    Returns
    -------
    dict: {변수명: delta_RMSE}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model = model.to(device)

    # Baseline RMSE
    with torch.no_grad():
        X_tensor = torch.FloatTensor(val_X).to(device)
        baseline_pred = model(X_tensor)["mu"].cpu().numpy()
        # 역변환
        baseline_pred_raw = normalizer.inverse_transform_target(baseline_pred)
        val_y_raw = normalizer.inverse_transform_target(val_y)
        baseline_rmse = np.sqrt(np.mean((baseline_pred_raw - val_y_raw) ** 2))

    importance = {}
    target_idx = variates.index(target_col) if target_col in variates else 0

    for var_idx, var_name in enumerate(variates):
        if var_idx == target_idx:
            continue  # 타겟은 제외

        delta_rmses = []
        for _ in range(n_repeats):
            X_perm = val_X.copy()
            # 시간 축 셔플 (변수별 독립)
            perm_idx = np.random.permutation(X_perm.shape[1])
            X_perm[:, :, var_idx] = X_perm[:, perm_idx, var_idx]

            with torch.no_grad():
                X_perm_tensor = torch.FloatTensor(X_perm).to(device)
                perm_pred = model(X_perm_tensor)["mu"].cpu().numpy()
                perm_pred_raw = normalizer.inverse_transform_target(perm_pred)
                perm_rmse = np.sqrt(np.mean((perm_pred_raw - val_y_raw) ** 2))

            delta_rmses.append(perm_rmse - baseline_rmse)

        importance[var_name] = np.mean(delta_rmses)

    return importance


def compute_all_importance(
    model: AttentionLSTMForecaster,
    input_tensor: torch.Tensor,
    variates: List[str],
    val_X: np.ndarray = None,
    val_y: np.ndarray = None,
    normalizer: ZScoreNormalizer = None,
    device: torch.device = None,
) -> Dict:
    """
    모든 중요도 메트릭 계산.

    Returns
    -------
    dict with variable_attention, gradient_importance, permutation_importance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # 1. Attention
    var_attn = extract_attention_importance(model, input_tensor)

    # 2. Gradient
    grad_imp = compute_gradient_importance(model, input_tensor)

    # 3. Permutation (검증 데이터 필요)
    perm_imp = None
    if val_X is not None and val_y is not None and normalizer is not None:
        perm_imp = compute_permutation_importance(
            model, val_X, val_y, variates, normalizer, device=device
        )

    return {
        "variable_attention": var_attn,
        "gradient_importance": grad_imp,
        "permutation_importance": perm_imp,
    }
