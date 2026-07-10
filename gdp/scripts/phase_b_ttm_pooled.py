"""T4: 다국가 풀링 사전학습 TTM (Montero-Manso–Hyndman 2021 풀링 원리 + TTM few-shot).

- 코퍼스: FRED OECD 38개국 분기 GDP q/q 성장률 → 월별 선형보간 경로 (우리 월별화
  N_gdp와 동일한 통계적 객체). 사전학습은 **2017-12까지만** (평가구간 정보 차단).
- 절차: TTM(r2) → 풀링 코퍼스 계속-사전학습 → 분기별 한국 few-shot(기존 T2와 동일)
- 백테스트·채점: phase_b_ttm.py --finetune 과 동일 조건
"""
import os, sys, glob, pickle, copy, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch
from tsfm_public.toolkit.get_model import get_model
import phase_b_harness as H

CTX, PLEN = 90, 6
POOL_EPOCHS = int(os.environ.get("POOL_EP", "20"))
FT_EP = int(os.environ.get("FT_EP", "15"))
CUT = pd.Timestamp("2017-12-31")

# ---------- 풀링 코퍼스 (월별 보간 경로) ----------
with open("data/pooled_gdp/oecd_gdp_qoq.pkl", "rb") as f:
    POOL = pickle.load(f)
def monthly_path(df):
    s = df.set_index("date")["g"]
    s.index = s.index + pd.offsets.QuarterEnd(0) + pd.offsets.MonthEnd(0)
    m = s.resample("ME").interpolate("linear")
    return m.clip(-6, 6).astype(np.float32)          # 극단치 완화 (스케일 안정)
Xs, Ys = [], []
for cc, df in POOL.items():
    m = monthly_path(df[df.date <= CUT])
    v = m.values
    for e in range(CTX, len(v) - PLEN):
        Xs.append(v[e - CTX:e, None]); Ys.append(v[e:e + PLEN, None])
X = torch.tensor(np.stack(Xs)); Y = torch.tensor(np.stack(Ys))
print(f"[pool] {len(POOL)}개국, 샘플 {len(X):,} (ctx{CTX}/plen{PLEN}, ≤2017-12)", flush=True)

# ---------- 계속-사전학습 ----------
torch.manual_seed(0)
model = get_model("ibm-granite/granite-timeseries-ttm-r2", context_length=CTX, prediction_length=PLEN)
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-5)
model.train()
for ep in range(POOL_EPOCHS):
    perm = torch.randperm(len(X)); tot = 0.0; nb = 0
    for i in range(0, len(X), 256):
        idx = perm[i:i + 256]
        opt.zero_grad()
        out = model(past_values=X[idx], future_values=Y[idx],
                    freq_token=torch.zeros(len(idx), dtype=torch.long))
        out.loss.backward(); opt.step()
        tot += float(out.loss); nb += 1
    if ep % 5 == 0: print(f"[pool] epoch {ep} loss {tot/nb:.4f}", flush=True)
model.eval()
torch.save(model.state_dict(), "output/ttm_pooled.pt")
print("[pool] 사전학습 완료", flush=True)

# ---------- 한국 백테스트 (T2 few-shot과 동일 절차) ----------
grid, _ = H.load_grid()
g = grid.copy(); g["vintage"] = pd.to_datetime(g["vintage"]).dt.strftime("%Y-%m-%d")
QSUB = os.environ.get("QSUB")
quarters = QSUB.split(",") if QSUB else sorted(g.tq.unique(), key=lambda x: pd.Period(x, "Q"))

def load_panel(tq, vintage):
    files = sorted(glob.glob(f"output/model/DFM/11/{tq}/*.csv"))
    cands = [f for f in files if os.path.basename(f)[:-4] <= vintage]
    if not cands: return None
    df = pd.read_csv(cands[-1]); dcol = df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol])
    return df.rename(columns={dcol: "date"}).set_index("date")[["N_gdp"]].astype(np.float32)

def finetune_kr(panel_train):
    mdl = copy.deepcopy(model); mdl.train()
    opt = torch.optim.Adam([p for p in mdl.parameters() if p.requires_grad], lr=1e-4)
    V = panel_train.values
    xs, ys = [], []
    for e in range(CTX, len(V) - PLEN):
        xs.append(V[e - CTX:e]); ys.append(V[e:e + PLEN])
    if len(xs) < 20: return model
    Xk = torch.tensor(np.stack(xs)); Yk = torch.tensor(np.stack(ys))
    for ep in range(FT_EP):
        idx = torch.randperm(len(Xk))[:64]
        opt.zero_grad()
        out = mdl(past_values=Xk[idx], future_values=Yk[idx],
                  freq_token=torch.zeros(len(idx), dtype=torch.long))
        out.loss.backward(); opt.step()
    mdl.eval(); return mdl

rows = []
for tq in quarters:
    sub = g[g.tq == tq].drop_duplicates(["vintage", "week_idx"]).sort_values("week_idx")
    if sub.empty: continue
    qend = pd.Period(tq, "Q").end_time.normalize().replace(day=1) + pd.offsets.MonthEnd(0)
    p0 = load_panel(tq, sub.vintage.iloc[0])
    cutoff = pd.Period(tq, "Q").to_timestamp() - pd.offsets.QuarterEnd(2)
    mdl = finetune_kr(p0[p0.index <= cutoff])
    for r in sub.itertuples(index=False):
        panel = load_panel(tq, r.vintage)
        yp = np.nan
        if panel is not None:
            edge = pd.Timestamp(r.vintage) + pd.offsets.MonthEnd(0)
            hist = panel[panel.index <= edge].tail(CTX)
            if len(hist) >= CTX:
                h = int((qend.to_period("M") - hist.index[-1].to_period("M")).n)
                if h < 1:
                    yp = float(panel.loc[qend, "N_gdp"]) if qend in panel.index else np.nan
                elif h <= PLEN:
                    with torch.no_grad():
                        out = mdl(past_values=torch.tensor(hist.values[None]),
                                  freq_token=torch.zeros(1, dtype=torch.long))
                    yp = float(out.prediction_outputs[0, h - 1, 0])
        rows.append({"tq": tq, "vintage": r.vintage, "week_idx": r.week_idx,
                     "flash": r.flash, "model_name": "our_ttm_pooled", "y_pred": yp})
    print(f"[{tq}] done", flush=True)

pred = pd.DataFrame(rows)
pred.to_csv("output/csv/_phase_b_ttm_pooled_predictions.csv", index=False)
dfm = H.load_baseline(grid, "dfm"); dfm["vintage"] = pd.to_datetime(dfm["vintage"]).dt.strftime("%Y-%m-%d")
ens = H.ensemble_with_dfm(pred.dropna(subset=["y_pred"]), dfm, suffix="ttm_pooled")
print("\n=== T4 TTM 풀링(38개국) + 한국 few-shot — flash w[-19,-1] avg RMSE ===")
print(H.score(pd.concat([pred, ens], ignore_index=True)).to_string())
print("[기준] TTM_ft(풀링 없음) 0.854/ens 0.846 | DFM 0.865 | DFM+XGB 0.765")
print("PHASE_B_TTM_POOLED_DONE")
