"""regime-gated 앙상블: 실시간(vintage-safe) 충격 탐지기로 DFM+XGB(shock)/DFM+RF(calm) 전환.
탐지기: 각 target 분기 직전 실현 GDP(flash) 최근 K분기 변동성 > 적응형 임계(과거 vol 중앙값) → shock.
"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,".")
import numpy as np, pandas as pd
import phase_b_harness as H

COVID=[str(p) for p in pd.period_range("2020Q1","2022Q4",freq="Q")]
RECENT=[str(p) for p in pd.period_range("2023Q1","2025Q4",freq="Q")]
K=4  # 변동성 윈도우(분기)
grid,refdf=H.load_grid(); KEY=["tq","vintage","week_idx"]
def norm(d): d=d.copy(); d["vintage"]=pd.to_datetime(d["vintage"]).dt.strftime("%Y-%m-%d"); return d
b=norm(H.load_baseline(grid,"dfm"))[KEY+["y_pred","flash"]].rename(columns={"y_pred":"dfm"})
for m in ["xgboost","rf"]:
    b=b.merge(norm(refdf[refdf.model_name==m])[KEY+["y_pred"]].rename(columns={"y_pred":m}),on=KEY,how="left")
b=b.dropna().reset_index(drop=True)

# ---- 실시간 충격 탐지 (vintage-safe): 분기 단위 ----
rel=pd.read_pickle("data/GDP_releases.pkl"); rel.index=rel.index.astype(str)
gdp=rel["flash"].dropna()  # 실현 GDP growth (분기)
qorder=[str(p) for p in pd.period_range("2017Q1","2025Q4",freq="Q")]
gdp=gdp.reindex([q for q in qorder if q in gdp.index])

def vol_before(q):
    """q 직전 K분기 실현 GDP 변동성 (q 이전 정보만)."""
    past=[x for x in gdp.index if pd.Period(x,"Q")<pd.Period(q,"Q")]
    if len(past)<2: return np.nan
    recent=gdp.loc[past[-K:]]
    return float(recent.std())

targets=sorted(b.tq.unique(), key=lambda x: pd.Period(x,"Q"))
vols={q: vol_before(q) for q in targets}
# 적응형 임계: 해당 분기 이전까지의 vol 중앙값 (vintage-safe, 확장창)
shock={}
for i,q in enumerate(targets):
    past_vols=[vols[targets[j]] for j in range(i) if not np.isnan(vols[targets[j]])]
    v=vols[q]
    if np.isnan(v) or len(past_vols)<2:
        shock[q]=True  # 정보부족 초기엔 보수적으로 DFM+XGB
    else:
        shock[q]= v > np.median(past_vols)

flagged=[q for q in targets if shock[q]]
print(f"shock 판정 분기 {len(flagged)}/{len(targets)}: {flagged}")
print(f"  COVID(12) 중 shock: {sum(q in flagged for q in COVID)} | 최근(12) 중 shock: {sum(q in flagged for q in RECENT)}")

# ---- 게이트 적용 ----
b["gated"]=np.where(b.tq.map(shock), (b.dfm+b.xgboost)/2, (b.dfm+b.rf)/2)
def rmse(col,sub=None):
    d=b if sub is None else b[b.tq.isin(sub)]
    t=pd.DataFrame({"model_name":"c","tq":d.tq,"vintage":d.vintage,"week_idx":d.week_idx,"flash":d.flash,"y_pred":d[col]})
    s=H.score(t); return float(s.iloc[0]) if len(s) else np.nan
b["dfm_xgb"]=(b.dfm+b.xgboost)/2
b["dfm_rf"]=(b.dfm+b.rf)/2
print("\n=== regime-gated 결과 (flash w[-19,-1] avg RMSE) ===")
for name,col in [("regime-gated (실시간)","gated"),("DFM+XGBoost","dfm_xgb"),("DFM+RF","dfm_rf"),("DFM","dfm")]:
    print(f"  {name:24s} 전체 {rmse(col):.4f} | COVID {rmse(col,COVID):.4f} | 최근 {rmse(col,RECENT):.4f}")
# 사후라벨 상한
b["oracle"]=np.where(b.tq.isin(COVID),(b.dfm+b.xgboost)/2,(b.dfm+b.rf)/2)
print(f"  {'oracle(사후라벨 상한)':24s} 전체 {rmse('oracle'):.4f}")
b[KEY+["gated","flash"]].to_csv("output/csv/_phase_b_regime_gated.csv",index=False)
print("REGIME_GATED_DONE")
