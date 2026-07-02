"""Phase B 공통 하네스: 우리 모델 예측을 그들과 동일 잣대로 채점.
평가 그리드/타깃/DFM baseline은 제공 아티팩트에서 로드. flash w[-19,-1] 평균 RMSE.
"""
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
REF = REPO / "output/csv/_reference_shipped_20260123/summary"
PRED_CSV = REF / "predictions_all_model_comparison_11_20260123_maxlag00.csv"
DFM_ROOT = REPO / "output/model/DFM/11"
LSTM_ROOT = REPO / "output/model/LSTM/39"
DFM_CSV = lambda tq: sorted((DFM_ROOT / tq).glob("*.csv"))


def load_grid():
    """평가 그리드: (tq, vintage, week_idx, flash) unique + 참조 xgboost 예측."""
    df = pd.read_csv(PRED_CSV, dtype={"tq": str})
    df["vintage"] = pd.to_datetime(df["vintage"]).dt.strftime("%Y-%m-%d")
    grid = df[["tq", "vintage", "week_idx", "flash", "provisional"]].drop_duplicates()
    return grid.reset_index(drop=True), df


def load_rtf(path, vintage, cols):
    p = Path(path)
    if not p.exists():
        return None
    rtf = pd.read_pickle(p)
    if not isinstance(rtf, pd.DataFrame) or rtf.empty:
        return None
    rtf = rtf.copy()
    rtf.index = pd.to_datetime(rtf.index).strftime("%Y-%m-%d")
    if vintage not in rtf.index:
        return None
    for c in cols:
        if c in rtf.columns and pd.notna(rtf.loc[vintage, c]):
            return float(rtf.loc[vintage, c])
    num = rtf.loc[vintage].apply(pd.to_numeric, errors="coerce").dropna()
    return float(num.iloc[0]) if not num.empty else None


def load_baseline(grid, which="dfm"):
    """DFM/LSTM baseline 예측을 그리드에 매핑."""
    root = DFM_ROOT if which == "dfm" else LSTM_ROOT
    cols = ("dfm", "mean") if which == "dfm" else ("mean", "lstm", "median")
    out = []
    for r in grid.itertuples(index=False):
        yp = load_rtf(root / r.tq / "rtf.pkl", r.vintage, cols)
        out.append({"tq": r.tq, "vintage": r.vintage, "week_idx": r.week_idx,
                    "flash": r.flash, "model_name": which, "y_pred": yp})
    return pd.DataFrame(out)


def score(pred_df, wmin=-19, wmax=-1, target="flash"):
    """flash w[wmin,wmax] 평균 RMSE (모형별). 주차별 RMSE(분기 평균)의 평균."""
    d = pred_df.dropna(subset=["y_pred", target]).copy()
    d = d[(d["week_idx"] >= wmin) & (d["week_idx"] <= wmax)]
    d["se"] = (d["y_pred"] - d[target]) ** 2
    wk = d.groupby(["model_name", "week_idx"])["se"].mean().pow(0.5).rename("rmse")
    return wk.groupby("model_name").mean().sort_values().rename("mean_weekly_rmse")


def ensemble_with_dfm(pred_df, dfm_df, suffix=None):
    """(dfm + model)/2 앙상블 예측 생성."""
    dfm_map = dfm_df.set_index(["tq", "vintage", "week_idx"])["y_pred"]
    rows = []
    for r in pred_df.itertuples(index=False):
        key = (r.tq, r.vintage, r.week_idx)
        if key not in dfm_map.index:
            continue
        d = dfm_map.loc[key]
        if pd.isna(d) or pd.isna(r.y_pred):
            continue
        rows.append({"tq": r.tq, "vintage": r.vintage, "week_idx": r.week_idx,
                     "flash": r.flash,
                     "model_name": f"ensemble_dfm_{suffix or r.model_name}",
                     "y_pred": (d + r.y_pred) / 2.0})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    grid, refdf = load_grid()
    print(f"grid: {len(grid)} rows | quarters={grid['tq'].nunique()} | "
          f"week_idx {grid['week_idx'].min()}..{grid['week_idx'].max()}")
    dfm = load_baseline(grid, "dfm")
    lstm = load_baseline(grid, "lstm")
    # sanity: DFM, LSTM, XGBoost 재계산
    xgb = refdf[refdf.model_name == "xgboost"][["tq", "vintage", "week_idx", "flash", "y_pred", "model_name"]]
    print("\n=== 하네스 sanity (기대: dfm 0.865, lstm 0.922, xgboost 0.937) ===")
    print(score(pd.concat([dfm, lstm, xgb], ignore_index=True)).to_string())
