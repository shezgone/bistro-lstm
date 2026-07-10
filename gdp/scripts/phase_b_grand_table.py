"""전 실험 통합 리더보드 + soft gate v3(+TTM) 검증.

soft_v3 = 완전 연속 게이트: ŷ = DFM + g·0.5·(TTM−DFM) + (1−g)·0.5·(p·XGB+(1−p)·RF − DFM)
  p = σ(vol z) 충격 가중(연속), g = σ(−flash(q−1)/τ)·[심리 저점통과] 반등 강도(연속)
  (g,p를 0/1로 계단화하면 v3a와 동일 — v3a의 연속 일반화)
"""
import sys, math, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy import stats
import phase_b_harness as H

grid, refdf = H.load_grid(); KEY = ["tq", "vintage", "week_idx"]
def norm(d):
    d = d.copy(); d["vintage"] = pd.to_datetime(d["vintage"]).dt.strftime("%Y-%m-%d"); return d
b = norm(H.load_baseline(grid, "dfm"))[KEY + ["y_pred", "flash"]].rename(columns={"y_pred": "dfm"})
for m in ["xgboost", "rf"]:
    b = b.merge(norm(refdf[refdf.model_name == m])[KEY + ["y_pred"]].rename(columns={"y_pred": m}), on=KEY, how="left")

SRC = {  # 파일: (모델명필터, 컬럼명)
    "_phase_b_our_ttm_ft_predictions.csv": ("our_ttm_ft", "ttm"),
    "_phase_b_ttm_pooled_predictions.csv": ("our_ttm_pooled", "ttm_pool"),
    "_phase_b_tabpfn_predictions.csv": ("our_tabpfn", "tabpfn"),
    "_phase_b_fair_predictions.csv": ("our_mlp", "mlp"),
    "_phase_b_seq_predictions.csv": ("our_seq", "seq"),
    "_phase_b_attnlstm.csv": ("our_attnlstm", "attnlstm"),
    "_phase_b_ncde_predictions.csv": ("our_ncde", "ncde"),
    "_phase_b_hopp_predictions.csv": ("our_hopp", "hopp"),
    "_phase_b_d2fm_predictions.csv": ("our_d2fm", "d2fm"),
    "_phase_b_chronos.csv": ("our_chronos", "chronos"),
    "_phase_b_moirai.csv": ("our_moirai", "moirai"),
}
for f, (mn, col) in SRC.items():
    try:
        d = norm(pd.read_csv(f"output/csv/{f}", dtype={"tq": str}))
        d = d[d.model_name == mn][KEY + ["y_pred"]].rename(columns={"y_pred": col})
        b = b.merge(d, on=KEY, how="left")
    except FileNotFoundError:
        print("skip:", f)
b = b.merge(norm(pd.read_csv("output/csv/_phase_b_regime_gated.csv", dtype={"tq": str}))[KEY + ["gated"]], on=KEY)
b = b.merge(norm(pd.read_csv("output/csv/_phase_b_regime_gated_v2.csv", dtype={"tq": str}))[KEY + ["gated_v2"]], on=KEY)
b = b.dropna(subset=["dfm", "xgboost", "rf", "flash"]).reset_index(drop=True)

# ---- 국면/신호 ----
REB = {'2018Q1','2019Q2','2020Q3','2023Q1','2024Q3','2025Q2'}
targets = sorted(b.tq.unique(), key=lambda x: pd.Period(x, "Q"))
rel = pd.read_pickle("data/GDP_releases.pkl"); fl = rel["flash"].dropna()
qs_all = [str(p) for p in pd.period_range("2017Q1", "2025Q4", freq="Q")]
flq = fl.reindex([q for q in qs_all if q in fl.index])
x = pd.read_excel("data/vintages/2026-03-04.xlsx")
if "date" in x.columns: x = x.rename(columns={"date": "Date"})
x["Date"] = pd.to_datetime(x["Date"]); esi = x.set_index("Date")["new_esi"].dropna()
def vol_before(q, K=4):
    past = [k for k in flq.index if pd.Period(k, "Q") < pd.Period(q, "Q")]
    return float(flq.loc[past[-K:]].std()) if len(past) >= 2 else np.nan
vols = {q: vol_before(q) for q in targets}
sig = lambda z: 1 / (1 + math.exp(-z))
def p_soft(q):
    i = targets.index(q)
    pv = np.array([vols[targets[j]] for j in range(i) if not np.isnan(vols[targets[j]])])
    if np.isnan(vols[q]) or len(pv) < 2: return 1.0
    med = np.median(pv); mad = np.median(np.abs(pv - med)) or pv.std() or 1.0
    return sig((vols[q] - med) / mad)
def trough(q, win=6):
    e = esi[esi.index < pd.Period(q, "Q").start_time].tail(win)
    return len(e) >= 3 and e.iloc[-1] > e.min() and e.idxmin() < e.index[-1]
def g_soft(q, tau=0.2):
    i = qs_all.index(q); prev = qs_all[i - 1]
    if prev not in flq.index or not trough(q): return 0.0
    return sig(-float(flq[prev]) / tau) if flq[prev] < 0 else 0.0
P = b.tq.map({q: p_soft(q) for q in targets})
G = b.tq.map({q: g_soft(q) for q in targets})
reg_reb = b.tq.isin(REB)

ttm_f = b.ttm.fillna(b.dfm)
# hard v3a / soft 변형들
b["v3a"] = np.where(reg_reb, (b.dfm + ttm_f) / 2, b.gated_v2)
mlmix = P * b.xgboost + (1 - P) * b.rf
b["v3_softp"] = np.where(reg_reb, (b.dfm + ttm_f) / 2, b.dfm + 0.5 * (mlmix - b.dfm))       # p만 연속
b["v3_soft"] = b.dfm + G * 0.5 * (ttm_f - b.dfm) + (1 - G) * 0.5 * (mlmix - b.dfm)          # 완전 soft

def sc(col, sub=None):
    d = b if sub is None else b[sub]
    t = pd.DataFrame({"model_name": col, "tq": d.tq, "vintage": d.vintage,
                      "week_idx": d.week_idx, "flash": d.flash, "y_pred": d[col]})
    s = H.score(t.dropna(subset=["y_pred"])); return float(s.iloc[0]) if len(s) else np.nan

# 앙상블 컬럼 생성
for col in ["ttm", "ttm_pool", "tabpfn", "mlp", "seq", "attnlstm", "ncde", "hopp", "d2fm", "chronos", "moirai"]:
    if col in b.columns:
        b[f"dfm_{col}"] = (b.dfm + b[col]) / 2

ROWS = [
    ("게이트 v3_soft (완전연속+TTM)", "v3_soft"), ("게이트 v3_softp (p연속+TTM)", "v3_softp"),
    ("게이트 v3a (hard+TTM)", "v3a"), ("게이트 v2 (hard)", "gated_v2"), ("게이트 v1", "gated"),
    ("DFM+XGBoost (기존최고)", "dfm_xgb_tmp"), ("DFM+TTM", "dfm_ttm"), ("DFM+TabPFN", "dfm_tabpfn"),
    ("DFM+MLP", "dfm_mlp"), ("DFM+Hopp", "dfm_hopp"), ("DFM+D2FM", "dfm_d2fm"),
    ("DFM+AttnLSTM(seq)", "dfm_seq"), ("DFM+NCDE", "dfm_ncde"), ("DFM (기준선)", "dfm"),
    ("TTM 단독", "ttm"), ("TTM풀링 단독", "ttm_pool"), ("TabPFN 단독", "tabpfn"),
    ("XGBoost 단독", "xgboost"), ("MLP 단독", "mlp"), ("Hopp LSTM 단독", "hopp"),
    ("D2FM 단독", "d2fm"), ("NCDE 단독", "ncde"), ("AttnLSTM(seq) 단독", "seq"),
    ("AttnLSTM(구버전) 단독", "attnlstm"), ("BISTRO(Moirai) zero-shot", "moirai"),
    ("Chronos zero-shot", "chronos"),
]
b["dfm_xgb_tmp"] = (b.dfm + b.xgboost) / 2
reb = b.tq.isin(REB)
res = []
for name, col in ROWS:
    if col not in b.columns: continue
    res.append((name, sc(col), sc(col, reb)))
res.sort(key=lambda r: r[1])
print(f"{'순위':4s} {'모형':30s} {'전체32Q':>8s} {'반등6Q':>8s}")
for i, (n, a, r) in enumerate(res, 1):
    print(f"{i:<4d} {n:30s} {a:8.4f} {r:8.4f}")

def dm(colA, colB):
    la = b.groupby("tq").apply(lambda d: np.nanmean((d[colA] - d.flash) ** 2))
    lb = b.groupby("tq").apply(lambda d: np.nanmean((d[colB] - d.flash) ** 2))
    d = (la - lb).reindex(targets).dropna().values; n = len(d)
    dbar = d.mean(); g0 = np.mean((d - dbar) ** 2); g1 = np.mean((d[1:] - dbar) * (d[:-1] - dbar))
    var = max((g0 + 2 * g1) / n, g0 / n / 10)
    t = dbar / np.sqrt(var) * np.sqrt((n - 1) / n)
    return t, 2 * (1 - stats.t.cdf(abs(t), df=n - 1))
print("\nDM: ", end="")
for a, r in [("v3_soft", "dfm_xgb_tmp"), ("v3_softp", "dfm_xgb_tmp"), ("v3a", "dfm_xgb_tmp"), ("v3_soft", "gated_v2")]:
    t, p = dm(a, r); print(f"{a} vs {r.replace('_tmp','')}: p={p:.3f} | ", end="")
print()
