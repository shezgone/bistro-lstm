"""
Utility: AR(1) baseline forecast.
bistro-xai의 inference_util.py와 동일.
"""
import numpy as np
import pandas as pd


def ar1_forecast(
    train_y: pd.Series,
    pred_index: pd.PeriodIndex,
    method: str = "ols",
    trend: str = "c",
) -> pd.Series:
    """
    Simple AR(1) forecast: y_t = c + phi * y_{t-1}

    Parameters
    ----------
    train_y : training series (PeriodIndex)
    pred_index : periods to forecast
    method : fitting method (only "ols" supported)
    trend : "c" for constant, "n" for no constant

    Returns
    -------
    pd.Series with pred_index as index
    """
    y = train_y.dropna().values.astype(float)
    if len(y) < 3:
        return pd.Series(np.nan, index=pred_index)

    # OLS: y[t] = c + phi * y[t-1]
    y_lag = y[:-1]
    y_cur = y[1:]

    if trend == "c":
        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        beta = np.linalg.lstsq(X, y_cur, rcond=None)[0]
        c, phi = beta[0], beta[1]
    else:
        phi = np.dot(y_lag, y_cur) / np.dot(y_lag, y_lag)
        c = 0.0

    # Recursive forecast
    n_pred = len(pred_index)
    preds = np.empty(n_pred)
    last_val = y[-1]
    for i in range(n_pred):
        preds[i] = c + phi * last_val
        last_val = preds[i]

    return pd.Series(preds, index=pred_index)
