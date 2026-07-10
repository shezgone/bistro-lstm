"""T1: Hopp nowcast_lstm (JOS 2022, UNCTAD) 공정 주입.

- 입력: XGBoost 하네스와 동일한 DFM 스냅샷 패널 (output/model/DFM/11/<tq>/<vintage>.csv,
  월별·결측완성, N_gdp = 월별화 타깃) → 정보 조건 동일한 공정 비교
- 학습: 분기당 1회 (최초 빈티지, quarter_batch 'fvintage'와 동일), 직전 144개월,
  n_timesteps=12 시퀀스, n_models 시드 앙상블 (Hopp 권장 구조)
- 예측: 각 주간 빈티지 패널로 predict → target 분기말 월의 값
"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from pandas.tseries.offsets import QuarterEnd
from nowcast_lstm.LSTM import LSTM
import phase_b_harness as H

N_MODELS = int(os.environ.get("NM", "5"))
EPOCHS = int(os.environ.get("EP", "150"))
SAMPLE_MONTHS = 144

grid, _ = H.load_grid(); KEY = ["tq", "vintage", "week_idx"]
g = grid.copy(); g["vintage"] = pd.to_datetime(g["vintage"]).dt.strftime("%Y-%m-%d")
QSUB = os.environ.get("QSUB")
quarters = QSUB.split(",") if QSUB else sorted(g.tq.unique(), key=lambda x: pd.Period(x, "Q"))

def load_panel(tq, vintage):
    """분기 디렉토리에서 해당(또는 직전) 빈티지 CSV 로드 → date 컬럼 형식."""
    files = sorted(glob.glob(f"output/model/DFM/11/{tq}/*.csv"))
    cands = [f for f in files if os.path.basename(f)[:-4] <= vintage]
    if not cands: return None
    df = pd.read_csv(cands[-1])
    dcol = df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol])
    df = df.rename(columns={dcol: "date"})
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[["date"] + num]

rows = []
for tq in quarters:
    sub = g[g.tq == tq].drop_duplicates(["vintage", "week_idx"]).sort_values("week_idx")
    if sub.empty: continue
    tv = sub.vintage.iloc[0]                       # 학습 빈티지 = 최초
    panel = load_panel(tq, tv)
    if panel is None: continue
    cutoff = pd.Period(tq, "Q").to_timestamp() - QuarterEnd(2)
    train = panel[panel.date <= cutoff].tail(SAMPLE_MONTHS).reset_index(drop=True)
    try:
        m = LSTM(data=train, target_variable="N_gdp", n_timesteps=12,
                 n_models=N_MODELS, train_episodes=EPOCHS, n_hidden=32, n_layers=2,
                 dropout=0.0, seeds=list(range(N_MODELS)))
        m.train(quiet=True)
    except Exception as e:
        print(f"[{tq}] train FAILED: {e}", flush=True); continue
    qend = pd.Period(tq, "Q").end_time.normalize().replace(day=1) + pd.offsets.MonthEnd(0)
    for r in sub.itertuples(index=False):
        pv = load_panel(tq, r.vintage)
        if pv is None: continue
        try:
            pred = m.predict(pv, only_actuals_obs=False)
            pcol = [c for c in pred.columns if "pred" in c.lower()][0]
            hit = pred[pred.date == qend]
            yp = float(hit[pcol].iloc[0]) if len(hit) else np.nan
        except Exception:
            yp = np.nan
        rows.append({"tq": tq, "vintage": r.vintage, "week_idx": r.week_idx,
                     "flash": r.flash, "model_name": "our_hopp", "y_pred": yp})
    done = [x for x in rows if x["tq"] == tq]
    print(f"[{tq}] preds={len(done)} (예: {done[-1]['y_pred'] if done else None})", flush=True)

pred = pd.DataFrame(rows)
pred.to_csv("output/csv/_phase_b_hopp_predictions.csv", index=False)
dfm = H.load_baseline(grid, "dfm")
dfm["vintage"] = pd.to_datetime(dfm["vintage"]).dt.strftime("%Y-%m-%d")
ens = H.ensemble_with_dfm(pred.dropna(subset=["y_pred"]), dfm, suffix="hopp")
sc = H.score(pd.concat([pred, ens], ignore_index=True))
print("\n=== T1 Hopp nowcast_lstm (공정 주입) flash w[-19,-1] avg RMSE ===")
print(sc.to_string())
print("[기준] DFM 0.865 / DFM+XGB 0.765 / DFM+TabPFN 0.815 / DFM+our_mlp 0.820")
print("PHASE_B_HOPP_DONE")
