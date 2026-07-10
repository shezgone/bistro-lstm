"""T5: DL 실험 승자(TTM)를 게이트에 통합 — v3 후보 채점 + DM 검정.

v3 후보:
  v3a: v2에서 반등 arm만 교체 → (DFM+TTM_ft)/2   (반등 6Q에서 DFM 단독을 제친 유일 후보)
  v3b: v3a + calm arm을 (DFM+RF+TTM)/3            (탐색용)
  v3c: soft gate(연속 λ) 위에 반등 arm=(DFM+TTM)/2
DM: 후보 vs v2 / DFM+XGB (분기 손실차, NW lag1 + HLN 소표본 보정)
"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy import stats
import phase_b_harness as H

grid, refdf = H.load_grid(); KEY = ["tq", "vintage", "week_idx"]
def norm(d):
    d = d.copy(); d["vintage"] = pd.to_datetime(d["vintage"]).dt.strftime("%Y-%m-%d"); return d
b = norm(H.load_baseline(grid, "dfm"))[KEY + ["y_pred", "flash"]].rename(columns={"y_pred": "dfm"})
for m in ["xgboost", "rf"]:
    b = b.merge(norm(refdf[refdf.model_name == m])[KEY + ["y_pred"]].rename(columns={"y_pred": m}), on=KEY, how="left")
ttm = norm(pd.read_csv("output/csv/_phase_b_our_ttm_ft_predictions.csv", dtype={"tq": str}))
b = b.merge(ttm[KEY + ["y_pred"]].rename(columns={"y_pred": "ttm"}), on=KEY, how="left")
b = b.merge(norm(pd.read_csv("output/csv/_phase_b_regime_gated.csv", dtype={"tq": str}))[KEY + ["gated"]], on=KEY)
b = b.merge(norm(pd.read_csv("output/csv/_phase_b_regime_gated_v2.csv", dtype={"tq": str}))[KEY + ["gated_v2"]], on=KEY)
b = b.dropna(subset=["dfm", "xgboost", "rf", "flash"]).reset_index(drop=True)
b["dfm_xgb"] = (b.dfm + b.xgboost) / 2; b["dfm_rf"] = (b.dfm + b.rf) / 2
b["dfm_ttm"] = (b.dfm + b.ttm) / 2

REB = {'2018Q1','2019Q2','2020Q3','2023Q1','2024Q3','2025Q2'}
targets = sorted(b.tq.unique(), key=lambda x: pd.Period(x, "Q"))
rel = pd.read_pickle("data/GDP_releases.pkl"); fl = rel["flash"].dropna()
qs_all = [str(p) for p in pd.period_range("2017Q1", "2025Q4", freq="Q")]
flq = fl.reindex([q for q in qs_all if q in fl.index])
def vol_before(q, K=4):
    past = [k for k in flq.index if pd.Period(k, "Q") < pd.Period(q, "Q")]
    return float(flq.loc[past[-K:]].std()) if len(past) >= 2 else np.nan
vols = {q: vol_before(q) for q in targets}
def regime(q):
    if q in REB: return "REB"
    i = targets.index(q); pv = [vols[targets[j]] for j in range(i) if not np.isnan(vols[targets[j]])]
    return "SHOCK" if (np.isnan(vols[q]) or len(pv) < 2 or vols[q] > np.median(pv)) else "CALM"
b["reg"] = b.tq.map(regime)

reb_m = b.reg == "REB"
b["v3a"] = np.where(reb_m, b.dfm_ttm.fillna(b.dfm), b.gated_v2)
b["v3b"] = np.where(reb_m, b.dfm_ttm.fillna(b.dfm),
            np.where(b.reg == "CALM", (b.dfm + b.rf + b.ttm.fillna(b.dfm)) / 3, b.dfm_xgb))
# v3c: soft λ (vol z 시그모이드) + 반등 arm TTM
import math
sig = lambda z: 1 / (1 + math.exp(-z))
def p_rule(q):
    i = targets.index(q)
    pv = np.array([vols[targets[j]] for j in range(i) if not np.isnan(vols[targets[j]])])
    if np.isnan(vols[q]) or len(pv) < 2: return 1.0
    med = np.median(pv); mad = np.median(np.abs(pv - med)) or pv.std() or 1.0
    return sig((vols[q] - med) / mad)
P = b.tq.map({q: p_rule(q) for q in targets})
ml_mix = P * b.xgboost + (1 - P) * b.rf
b["v3c"] = np.where(reb_m, b.dfm_ttm.fillna(b.dfm), b.dfm + 0.5 * (ml_mix - b.dfm))

def sc(col, sub=None):
    d = b if sub is None else b[sub]
    t = pd.DataFrame({"model_name": col, "tq": d.tq, "vintage": d.vintage,
                      "week_idx": d.week_idx, "flash": d.flash, "y_pred": d[col]})
    s = H.score(t.dropna(subset=["y_pred"])); return float(s.iloc[0]) if len(s) else np.nan
reb = b.tq.isin(REB); rec = b.tq.isin([str(p) for p in pd.period_range("2023Q1","2025Q4",freq="Q")])
print(f"{'구성':14s} {'전체32Q':>8s} {'반등6Q':>8s} {'최근12Q':>8s}")
for c in ["dfm_xgb", "gated", "gated_v2", "v3a", "v3b", "v3c"]:
    print(f"{c:14s} {sc(c):8.4f} {sc(c,reb):8.4f} {sc(c,rec):8.4f}")

def dm(colA, colB):
    la = b.groupby("tq").apply(lambda d: np.nanmean((d[colA] - d.flash) ** 2))
    lb = b.groupby("tq").apply(lambda d: np.nanmean((d[colB] - d.flash) ** 2))
    d = (la - lb).reindex(targets).dropna().values; n = len(d)
    dbar = d.mean()
    g0 = np.mean((d - dbar) ** 2); g1 = np.mean((d[1:] - dbar) * (d[:-1] - dbar))
    var = max((g0 + 2 * g1) / n, g0 / n / 10)
    t = dbar / np.sqrt(var) * np.sqrt((n + 1 - 2) / n)
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1))
    return t, p
print("\n=== DM 검정 (음수 t = 앞이 더 정확) ===")
for a, ref in [("v3a","gated_v2"), ("v3a","dfm_xgb"), ("v3c","dfm_xgb"), ("gated_v2","dfm_xgb")]:
    t, p = dm(a, ref)
    print(f"  {a} vs {ref}: t={t:.2f}, p={p:.3f}")
