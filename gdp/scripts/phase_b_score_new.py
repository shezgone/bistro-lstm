"""TabPFN·NCDE 결과 통합 채점: 단독 + DFM 앙상블 + 반등 6분기, 기존 기준선과 비교."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
import phase_b_harness as H

grid, refdf = H.load_grid(); KEY = ["tq", "vintage", "week_idx"]
def norm(d):
    d = d.copy(); d["vintage"] = pd.to_datetime(d["vintage"]).dt.strftime("%Y-%m-%d"); return d
dfm = norm(H.load_baseline(grid, "dfm"))
REB = {'2018Q1','2019Q2','2020Q3','2023Q1','2024Q3','2025Q2'}

parts = [dfm.assign(model_name="dfm")[KEY+["flash","model_name","y_pred"]]]
xgb = norm(refdf[refdf.model_name=="xgboost"])[KEY+["flash","y_pred","model_name"]]
parts += [xgb, H.ensemble_with_dfm(xgb, dfm, suffix="xgboost")]
for f, name in [("output/csv/_phase_b_tabpfn_predictions.csv","our_tabpfn"),
                ("output/csv/_phase_b_ncde_predictions.csv","our_ncde")]:
    try:
        p = norm(pd.read_csv(f, dtype={"tq":str}))
        p = p[p.model_name==name][KEY+["flash","y_pred","model_name"]].dropna()
        parts += [p, H.ensemble_with_dfm(p, dfm, suffix=name)]
        print(f"{name}: {p.tq.nunique()}분기 {len(p)}행 로드")
    except FileNotFoundError:
        print(f"{name}: 파일 없음 (아직 실행 중)")
allp = pd.concat(parts, ignore_index=True)
print("\n=== 전체 32Q (flash w[-19,-1] avg RMSE) ===")
print(H.score(allp).to_string())
print("\n=== 반등 6분기 한정 ===")
print(H.score(allp[allp.tq.isin(REB)]).to_string())
print("\n=== 최근 12분기 (2023Q1~2025Q4) ===")
rec = [str(p) for p in pd.period_range("2023Q1","2025Q4",freq="Q")]
print(H.score(allp[allp.tq.isin(rec)]).to_string())
