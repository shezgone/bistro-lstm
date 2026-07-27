"""Chronos-2 (Amazon, 2025.10) 공변량 zero-shot 나우캐스트 — 신형 TSFM 재도전.

우리가 기각한 것은 구세대 Chronos-Bolt(단변량 zero-shot, 1.35). Chronos-2는
①공변량(past_covariates) 지원 ②다변량 정보 공유 ③분위수 출력 — 세 가지가 다름.
프로토콜은 phase_b_ttm.py와 동일(공정 비교): DFM 스냅샷 패널을 빈티지 월까지 자르고
분기말 월의 N_gdp를 외삽. 공변량 = 스냅샷 내 월별 지표 10종.
분위수(0.1~0.9)도 저장 → Track B(fan chart) 재료.
"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch
from chronos import Chronos2Pipeline
import phase_b_harness as H

COVARS = ["I_m", "I_s", "M_s", "S_es", "S_cb", "S_mo", "B_bx", "B_bi", "R_s", "M_fi"]
PLEN_MAX = 6
QUANTS = [0.1, 0.25, 0.5, 0.75, 0.9]

grid, _ = H.load_grid()
g = grid.copy(); g["vintage"] = pd.to_datetime(g["vintage"]).dt.strftime("%Y-%m-%d")
QSUB = os.environ.get("QSUB")
quarters = QSUB.split(",") if QSUB else sorted(g.tq.unique(), key=lambda x: pd.Period(x, "Q"))

pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
print("[chronos-2] loaded", flush=True)

def load_panel(tq, vintage):
    files = sorted(glob.glob(f"output/model/DFM/11/{tq}/*.csv"))
    cands = [f for f in files if os.path.basename(f)[:-4] <= vintage]
    if not cands: return None
    df = pd.read_csv(cands[-1]); dcol = df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol])
    return df.rename(columns={dcol: "date"}).set_index("date").astype(np.float32)

rows = []
for tq in quarters:
    sub = g[g.tq == tq].drop_duplicates(["vintage", "week_idx"]).sort_values("week_idx")
    if sub.empty: continue
    qend = pd.Period(tq, "Q").end_time.normalize().replace(day=1) + pd.offsets.MonthEnd(0)
    for r in sub.itertuples(index=False):
        panel = load_panel(tq, r.vintage)
        yp = np.nan; qv = {q: np.nan for q in QUANTS}
        if panel is not None:
            edge = pd.Timestamp(r.vintage) + pd.offsets.MonthEnd(0)
            hist = panel[panel.index <= edge]
            h = int((qend.to_period("M") - hist.index[-1].to_period("M")).n) if len(hist) else 99
            if h < 1:
                yp = float(panel.loc[qend, "N_gdp"]) if qend in panel.index else np.nan
            elif h <= PLEN_MAX and len(hist) >= 60:
                inp = [{"target": hist["N_gdp"].to_numpy(np.float32),
                        "past_covariates": {c: hist[c].to_numpy(np.float32) for c in COVARS if c in hist.columns}}]
                try:
                    qt, mean = pipe.predict_quantiles(inp, prediction_length=h, quantile_levels=QUANTS)
                    arr = np.asarray(qt[0])          # (1, h, n_q) — 배치 차원 포함
                    if arr.ndim == 3: arr = arr[0]   # → (h, n_q)
                    for qi_, q in enumerate(QUANTS): qv[q] = float(arr[h - 1, qi_])
                    yp = qv[0.5]
                except Exception as e:
                    print(f"  predict err {tq} {r.vintage}: {type(e).__name__}: {e}", flush=True)
        row = {"tq": tq, "vintage": r.vintage, "week_idx": r.week_idx,
               "flash": r.flash, "model_name": "our_chronos2", "y_pred": yp}
        row.update({f"q{int(q*100)}": v for q, v in qv.items()})
        rows.append(row)
    print(f"[{tq}] done", flush=True)

pred = pd.DataFrame(rows)
pred.to_csv("output/csv/_phase_b_chronos2_predictions.csv", index=False)
dfm = H.load_baseline(grid, "dfm"); dfm["vintage"] = pd.to_datetime(dfm["vintage"]).dt.strftime("%Y-%m-%d")
core = pred[["tq","vintage","week_idx","flash","model_name","y_pred"]].dropna(subset=["y_pred"])
ens = H.ensemble_with_dfm(core, dfm, suffix="chronos2")
print("\n=== Chronos-2 (공변량 zero-shot) flash w[-19,-1] avg RMSE ===")
print(H.score(pd.concat([core, ens], ignore_index=True)).to_string())
print("[기준] 구세대 Chronos-Bolt 1.351 / TTM_ft 0.854 / DFM 0.865 / DFM+XGB 0.765")
print("PHASE_B_CHRONOS2_DONE")
