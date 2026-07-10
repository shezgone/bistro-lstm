"""T2: IBM TTM (Tiny Time Mixers, NeurIPS 2024) — zero-shot / few-shot 나우캐스트.

- 패널: DFM 스냅샷(월별·결측완성). 단, 빈티지 월 이후 행은 잘라서 TTM이 직접
  외삽하게 함 (DFM 외삽을 베끼지 않도록 — DL 기여분 분리)
- zero-shot: N_gdp 채널 단변량 예측 (TTM 사전학습만으로)
- --finetune: 분기별 확장창 few-shot (선형 head+decoder 미세조정, 다채널)
- 예측치 = target 분기말 월의 N_gdp 예측
"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch
from tsfm_public.toolkit.get_model import get_model
import phase_b_harness as H

CTX = int(os.environ.get("CTX", "90"))
PLEN = 6
FT = "--finetune" in sys.argv
CHANNELS = ["N_gdp"] if not FT else ["N_gdp", "new_coin", "I_m", "M_p", "I_s", "R_s",
                                     "B_gx", "new_esi", "S_cb", "M_ic"]

grid, _ = H.load_grid()
g = grid.copy(); g["vintage"] = pd.to_datetime(g["vintage"]).dt.strftime("%Y-%m-%d")
QSUB = os.environ.get("QSUB")
quarters = QSUB.split(",") if QSUB else sorted(g.tq.unique(), key=lambda x: pd.Period(x, "Q"))

model = get_model("ibm-granite/granite-timeseries-ttm-r2",
                  context_length=CTX, prediction_length=PLEN)
model.eval()
print(f"[TTM] ctx={CTX} plen={PLEN} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
      f"mode={'few-shot' if FT else 'zero-shot'}", flush=True)

def load_panel(tq, vintage):
    files = sorted(glob.glob(f"output/model/DFM/11/{tq}/*.csv"))
    cands = [f for f in files if os.path.basename(f)[:-4] <= vintage]
    if not cands: return None
    df = pd.read_csv(cands[-1])
    dcol = df.columns[0]; df[dcol] = pd.to_datetime(df[dcol])
    df = df.rename(columns={dcol: "date"}).set_index("date")
    cols = [c for c in CHANNELS if c in df.columns]
    return df[cols].astype(np.float32)

def forecast_qend(panel, vintage, qend, mdl):
    """빈티지 월까지 자른 뒤 TTM으로 분기말 월까지 외삽."""
    edge = pd.Timestamp(vintage) + pd.offsets.MonthEnd(0)
    hist = panel[panel.index <= edge].tail(CTX)
    if len(hist) < CTX: return np.nan
    h = int((qend.to_period("M") - hist.index[-1].to_period("M")).n)
    if h < 1:                       # 분기말 월이 이미 관측창 안 → 마지막 값
        return float(panel.loc[qend, "N_gdp"]) if qend in panel.index else np.nan
    if h > PLEN: return np.nan
    x = torch.tensor(hist.values[None])         # (1, CTX, C)
    with torch.no_grad():
        out = mdl(past_values=x, freq_token=torch.zeros(1, dtype=torch.long))
    return float(out.prediction_outputs[0, h - 1, 0])   # N_gdp 채널

def finetune(tq, panel_train):
    import copy
    mdl = copy.deepcopy(model); mdl.train()
    opt = torch.optim.Adam([p for p in mdl.parameters() if p.requires_grad], lr=1e-4)
    V = panel_train.values
    xs, ys = [], []
    for e in range(CTX, len(V) - PLEN):
        xs.append(V[e - CTX:e]); ys.append(V[e:e + PLEN])
    if len(xs) < 20: return model
    X = torch.tensor(np.stack(xs)); Y = torch.tensor(np.stack(ys))
    for ep in range(int(os.environ.get("FT_EP", "10"))):
        idx = torch.randperm(len(X))[:64]
        opt.zero_grad()
        out = mdl(past_values=X[idx], future_values=Y[idx], freq_token=torch.zeros(len(idx), dtype=torch.long))
        out.loss.backward(); opt.step()
    mdl.eval(); return mdl

rows = []
for tq in quarters:
    sub = g[g.tq == tq].drop_duplicates(["vintage", "week_idx"]).sort_values("week_idx")
    if sub.empty: continue
    qend = pd.Period(tq, "Q").end_time.normalize().replace(day=1) + pd.offsets.MonthEnd(0)
    mdl = model
    if FT:
        p0 = load_panel(tq, sub.vintage.iloc[0])
        cutoff = pd.Period(tq, "Q").to_timestamp() - pd.offsets.QuarterEnd(2)
        mdl = finetune(tq, p0[p0.index <= cutoff])
    for r in sub.itertuples(index=False):
        panel = load_panel(tq, r.vintage)
        yp = np.nan if panel is None else forecast_qend(panel, r.vintage, qend, mdl)
        rows.append({"tq": tq, "vintage": r.vintage, "week_idx": r.week_idx,
                     "flash": r.flash, "model_name": "our_ttm" + ("_ft" if FT else ""), "y_pred": yp})
    print(f"[{tq}] done", flush=True)

pred = pd.DataFrame(rows)
name = "our_ttm_ft" if FT else "our_ttm"
pred.to_csv(f"output/csv/_phase_b_{name}_predictions.csv", index=False)
dfm = H.load_baseline(grid, "dfm"); dfm["vintage"] = pd.to_datetime(dfm["vintage"]).dt.strftime("%Y-%m-%d")
ens = H.ensemble_with_dfm(pred.dropna(subset=["y_pred"]), dfm, suffix=name)
print(f"\n=== T2 TTM ({'few-shot' if FT else 'zero-shot'}) flash w[-19,-1] avg RMSE ===")
print(H.score(pd.concat([pred, ens], ignore_index=True)).to_string())
print("[기준] DFM 0.865 / DFM+XGB 0.765 / DFM+TabPFN 0.815")
print("PHASE_B_TTM_DONE")
