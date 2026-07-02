"""Phase B 종합 비교: 우리 모델(단독/앙상블) vs DFM 0.865 / DFM+XGBoost 0.765. 동일 잣대·국면별."""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
import phase_b_harness as H

COVID = [str(p) for p in pd.period_range("2020Q1", "2022Q4", freq="Q")]
RECENT = [str(p) for p in pd.period_range("2023Q1", "2025Q4", freq="Q")]

def score_sub(pred, quarters=None):
    d = pred if quarters is None else pred[pred.tq.isin(quarters)]
    return H.score(d)

grid, refdf = H.load_grid()
dfm = H.load_baseline(grid, "dfm")
xgb = refdf[refdf.model_name == "xgboost"][["tq","vintage","week_idx","flash","y_pred","model_name"]].copy()
xgb["vintage"] = pd.to_datetime(xgb["vintage"]).dt.strftime("%Y-%m-%d")

attn = pd.read_csv("output/csv/_phase_b_attnlstm.csv", dtype={"tq":str})
chro = pd.read_csv("output/csv/_phase_b_chronos.csv", dtype={"tq":str})
moir = pd.read_csv("output/csv/_phase_b_moirai.csv", dtype={"tq":str})
for d in (attn,chro,moir): d["vintage"]=pd.to_datetime(d["vintage"]).dt.strftime("%Y-%m-%d")

# 앙상블
ens_xgb = H.ensemble_with_dfm(xgb, dfm, "xgboost")
ens_attn = H.ensemble_with_dfm(attn, dfm, "attnlstm")

def one(df, q=None):
    s = score_sub(df, q)
    return float(s.iloc[0]) if len(s) else float("nan")

# 우리 attnlstm이 예측한 분기 집합 (전체 32), foundation은 26 subset
found_q = sorted(chro[chro.y_pred.notna()].tq.unique())

print("="*66)
print("Phase B 종합 — flash w[-19,-1] 평균 RMSE (낮을수록 좋음)")
print("="*66)
print("\n[1] 전체 32분기 기준")
print(f"  DFM (baseline)            {one(dfm):.4f}")
print(f"  XGBoost                   {one(xgb):.4f}")
print(f"  DFM+XGBoost (그들 최고)    {one(ens_xgb):.4f}")
print(f"  ── 우리 모델 ──")
print(f"  our AttnLSTM (단독)        {one(attn):.4f}")
print(f"  DFM+AttnLSTM (앙상블)      {one(ens_attn):.4f}")

print(f"\n[2] Foundation 비교 (동일 {len(found_q)}분기 subset, 공정)")
print(f"  DFM (same subset)         {one(dfm[dfm.tq.isin(found_q)]):.4f}")
print(f"  our AttnLSTM (subset)     {one(attn[attn.tq.isin(found_q)]):.4f}")
print(f"  our Chronos               {one(chro):.4f}")
print(f"  our Moirai                {one(moir):.4f}")

print("\n[3] 국면별 (우리 모델 vs DFM)")
for lbl,qs in [("COVID 20Q1-22Q4",COVID),("최근 23Q1-25Q4",RECENT)]:
    print(f"  {lbl}: DFM {one(dfm,qs):.4f} | AttnLSTM {one(attn,qs):.4f} | "
          f"DFM+AttnLSTM {one(ens_attn,qs):.4f} | DFM+XGB {one(ens_xgb,qs):.4f}")

# 저장
summ = pd.DataFrame({
 "model":["dfm","xgboost","ensemble_dfm_xgboost","our_attnlstm","ensemble_dfm_attnlstm","our_chronos","our_moirai"],
 "rmse_full":[one(dfm),one(xgb),one(ens_xgb),one(attn),one(ens_attn),float("nan"),float("nan")],
})
summ.to_csv("output/csv/_phase_b_summary.csv", index=False)
print("\nsaved -> output/csv/_phase_b_summary.csv")
print("PHASE_B_COMPARE_DONE")
