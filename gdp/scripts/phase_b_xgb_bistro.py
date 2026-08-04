"""XGB(신, schema v2) × BISTRO(AttnLSTM) 결합 검증 — 기각 (2026-08-04).

배경: 단위버그 수정 후 XGB(신) 단독이 신 기준선(0.7499)이 되자,
사내 BISTRO 어텐션 LSTM(core/lstm_model.py 계열, GDP 이식판 = phase_b_attnlstm)
과의 결합 가치를 재확인.

결과 (공통표본 26분기·509행 — BISTRO는 2018Q1~2019Q2 예측 없음):
  XGB(신) 단독          0.8000  (반등 0.8502)   ← 공통표본 기준
  BISTRO 단독           1.2681  (반등 2.4009)
  (XGB+BISTRO)/2        0.9968  (반등 1.6007)
  w(XGB)=0.9 최소 혼합   0.8300  — 어느 가중에서도 단독보다 악화 (단조 열화)
  (XGB+GBM+BISTRO)/3    0.9206 / (XGB+BISTRO+C2f)/3 0.9223 — 모두 열세
판정: 기각. 오차 상관 0.847(높음) + 정확도 큰 격차 → 결합 이득의 두 조건
(낮은 상관 또는 대등한 정확도) 모두 불충족. 7/9의 "AttnLSTM 기각"이
신 체계(XGB 신예측) 기준으로도 유지됨.

실행: gdp-nowcasting-renewal 루트에서 .venv-gdp/bin/python phase_b_xgb_bistro.py
"""
import sys, glob, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy import stats
import phase_b_harness as H

KEY = ["tq", "vintage", "week_idx"]
def norm(d):
    d = d.copy(); d["vintage"] = pd.to_datetime(d["vintage"]).dt.strftime("%Y-%m-%d"); return d

files = sorted(glob.glob("output/csv/all_model_comparison_11_20260123_maxlag00/*/"
                         "all_model_comparison_predictions_with_all_ensembles_*.csv"))
new = pd.concat([pd.read_csv(f, dtype={"tq": str}) for f in files], ignore_index=True)
new["vintage"] = pd.to_datetime(new["vintage"]).dt.strftime("%Y-%m-%d")
def pick(mn, col):
    return new[new.model_name == mn][KEY + ["y_pred"]].rename(columns={"y_pred": col}).drop_duplicates(KEY)

b = pick("xgboost", "xgb").merge(pick("gbm", "gbm"), on=KEY)
b = b.merge(new[new.model_name == "dfm"][KEY + ["flash"]].drop_duplicates(KEY), on=KEY)
bi = norm(pd.read_csv("output/csv/_phase_b_attnlstm.csv", dtype={"tq": str}))
b = b.merge(bi[KEY + ["y_pred"]].rename(columns={"y_pred": "bistro"}), on=KEY, how="left")
c2 = norm(pd.read_csv("output/csv/_phase_b_our_chronos2f_predictions.csv", dtype={"tq": str}))
b = b.merge(c2[c2.model_name == "our_chronos2f"][KEY + ["y_pred"]].rename(columns={"y_pred": "c2f"}), on=KEY, how="left")
b = b.dropna(subset=["xgb", "bistro", "flash"]).reset_index(drop=True)

targets = sorted(b.tq.unique(), key=lambda x: pd.Period(x, "Q"))
REB = {"2018Q1", "2019Q2", "2020Q3", "2023Q1", "2024Q3", "2025Q2"}
reb = b.tq.isin(REB); covid = b.tq.isin(["2020Q1", "2020Q2", "2020Q3"])
def sc(v, sub=None):
    d = b if sub is None else b[sub]; vv = v if sub is None else v[sub]
    t = pd.DataFrame({"model_name": "x", "tq": d.tq, "vintage": d.vintage,
                      "week_idx": d.week_idx, "flash": d.flash, "y_pred": vv})
    return float(H.score(t).iloc[0])

print(f"공통 그리드: {len(b)}행 {b.tq.nunique()}분기 (BISTRO 결측 분기: 2018Q1~2019Q2)")
print("오차 상관 XGB-BISTRO: %.3f" % (b.xgb - b.flash).corr(b.bistro - b.flash))
C = {
    "XGB(신) 단독 [기준]": b.xgb,
    "BISTRO 단독": b.bistro,
    "(XGB+BISTRO)/2": (b.xgb + b.bistro) / 2,
    "(XGB+GBM+BISTRO)/3": (b.xgb + b.gbm + b.bistro) / 3,
    "(XGB+BISTRO+C2f)/3": (b.xgb + b.bistro + b.c2f) / 3,
}
print(f"{'구성':22s} {'전체':>8s} {'exCOVID':>8s} {'반등':>8s}")
for k, v in C.items():
    print(f"{k:22s} {sc(v):8.4f} {sc(v, ~covid):8.4f} {sc(v, reb):8.4f}")
print("\nw·XGB+(1-w)·BISTRO:", {w: round(sc(w * b.xgb + (1 - w) * b.bistro), 4) for w in [0.9, 0.8, 0.7, 0.6, 0.5]})
