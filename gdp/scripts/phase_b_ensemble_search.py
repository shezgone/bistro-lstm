"""DFM+XGBoost(0.765)를 넘는 앙상블 탐색. 이미 가진 base 예측을 결합(재학습 없음).
base: dfm, xgboost, rf, gbm, our_mlp. 조합: 부분집합 단순평균 + inverse-RMSE 가중 + walk-forward 스택.
"""
import sys, warnings, itertools; warnings.filterwarnings("ignore"); sys.path.insert(0,".")
import numpy as np, pandas as pd
import phase_b_harness as H

COVID=[str(p) for p in pd.period_range("2020Q1","2022Q4",freq="Q")]
RECENT=[str(p) for p in pd.period_range("2023Q1","2025Q4",freq="Q")]
grid,refdf=H.load_grid()
KEY=["tq","vintage","week_idx"]

def norm(df):
    df=df.copy(); df["vintage"]=pd.to_datetime(df["vintage"]).dt.strftime("%Y-%m-%d"); return df

# base 예측들 → wide
dfm=norm(H.load_baseline(grid,"dfm"))[KEY+["y_pred","flash"]].rename(columns={"y_pred":"dfm"})
base=dfm.copy()
for m in ["xgboost","rf","gbm"]:
    d=norm(refdf[refdf.model_name==m])[KEY+["y_pred"]].rename(columns={"y_pred":m})
    base=base.merge(d,on=KEY,how="left")
mlp=norm(pd.read_csv("output/csv/_phase_b_fair_predictions.csv",dtype={"tq":str}))
mlp=mlp[mlp.model_name=="our_mlp"][KEY+["y_pred"]].rename(columns={"y_pred":"our_mlp"})
base=base.merge(mlp,on=KEY,how="left")
base=base.dropna(subset=["dfm","xgboost","rf","gbm","our_mlp","flash"]).reset_index(drop=True)
print(f"정합된 행: {len(base)} | 분기 {base.tq.nunique()}")

def rmse_of(w_pred, sub=None):
    d=base if sub is None else base[base.tq.isin(sub)]
    idx=d.index
    p=w_pred.loc[idx]
    tmp=pd.DataFrame({"model_name":"c","tq":d.tq,"vintage":d.vintage,"week_idx":d.week_idx,"flash":d.flash,"y_pred":p})
    s=H.score(tmp); return float(s.iloc[0]) if len(s) else np.nan

MODELS=["dfm","xgboost","rf","gbm","our_mlp"]
results=[]
# 1) DFM + 부분집합 단순평균 (DFM 항상 포함)
others=["xgboost","rf","gbm","our_mlp"]
for r in range(1,len(others)+1):
    for combo in itertools.combinations(others,r):
        cols=["dfm"]+list(combo)
        pred=base[cols].mean(axis=1)
        results.append(("DFM+mean("+"+".join(combo)+")", rmse_of(pred), rmse_of(pred,COVID), rmse_of(pred,RECENT)))
# 2) inverse-RMSE 가중 (전체구간 성과 기반 — 참고용, 약한 과적합)
def inv_rmse_w(cols):
    per={c: rmse_of(base[c]) for c in cols}
    w=np.array([1/per[c] for c in cols]); w=w/w.sum()
    return (base[cols]*w).sum(axis=1)
for cols in [["dfm","xgboost","our_mlp"],["dfm","xgboost","rf","gbm","our_mlp"],["dfm","xgboost","gbm","our_mlp"]]:
    pred=inv_rmse_w(cols)
    results.append(("wINV("+"+".join(cols)+")", rmse_of(pred), rmse_of(pred,COVID), rmse_of(pred,RECENT)))

res=pd.DataFrame(results,columns=["ensemble","전체","COVID","최근"]).sort_values("전체")
pd.set_option("display.float_format",lambda x:f"{x:.4f}")
print("\n=== 앙상블 탐색 (전체 오름차순, 상위 15) ===")
print(res.head(15).to_string(index=False))
print(f"\n기준선: DFM 0.865 | DFM+XGBoost {rmse_of((base['dfm']+base['xgboost'])/2):.4f}")
best=res.iloc[0]
print(f"\n>>> 최상 조합: {best['ensemble']}  전체 {best['전체']:.4f}  (DFM+XGB 0.765 대비 {best['전체']-0.765:+.4f})")
res.to_csv("output/csv/_phase_b_ensemble_search.csv",index=False)
print("ENS_SEARCH_DONE")
