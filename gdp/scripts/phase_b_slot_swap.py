"""부품 교체 실험 — 주차별 결합의 조기 슬롯에서 Chronos-2f ↔ BISTRO 교체 비교 (2026-08-12).

설계: 검증된 주차별 결합(early=(GBM+부품)/2, 이후 XGBoost)에서 조기 슬롯 부품만 교체.
공통표본 26분기 (BISTRO 예측 없는 2018Q1~2019Q2 제외 — 32Q 리더보드 수치와 비교 불가).

결과:
  XGB 단독 [기준]                 0.8000 (반등 0.8502)
  early=(GBM+Chronos-2f)/2       0.7889 (반등 0.7891)   ← -1.4% 개선
  early=(GBM+BISTRO)/2           0.8259 (반등 1.0055)   ← +3.2% 악화
  참고: early=GBM만               0.7916 / early=BISTRO만 0.8800
원인 (조기 구간 -19~-14 단독 RMSE): GBM 0.997 ≈ C2f 0.999 < XGB 1.024 ≪ BISTRO 1.277
  — 결합 규칙의 공이 아니라 부품의 조기 구간 역량이 개선의 원천.
  오차 상관은 BISTRO가 오히려 낮음(GBM과 0.904 vs C2f 0.959)에도 정확도 격차가 압도.
유의: BISTRO = 당사 GDP 이식 재구현판(phase_b_attnlstm) 기준 — 원 연구 판정 아님.
산출물: docs/부품비교_Chronos2f_vs_BISTRO_1p_2026-08-12.pptx/pdf

실행: gdp-nowcasting-renewal 루트에서 .venv-gdp/bin/python phase_b_slot_swap.py
"""
import sys, glob, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
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
c2 = norm(pd.read_csv("output/csv/_phase_b_our_chronos2f_predictions.csv", dtype={"tq": str}))
b = b.merge(c2[c2.model_name == "our_chronos2f"][KEY + ["y_pred"]].rename(columns={"y_pred": "c2f"}), on=KEY)
bi = norm(pd.read_csv("output/csv/_phase_b_attnlstm.csv", dtype={"tq": str}))
b = b.merge(bi[KEY + ["y_pred"]].rename(columns={"y_pred": "bis"}), on=KEY, how="left")
b = b.dropna().reset_index(drop=True)
early = (b.week_idx <= -14).values
REB = {"2018Q1", "2019Q2", "2020Q3", "2023Q1", "2024Q3", "2025Q2"}
reb = b.tq.isin(REB)

def sc(v, sub=None):
    d = b if sub is None else b[sub]
    vv = pd.Series(v) if sub is None else pd.Series(v)[sub if isinstance(sub, np.ndarray) else sub.values]
    t = pd.DataFrame({"model_name": "x", "tq": d.tq, "vintage": d.vintage,
                      "week_idx": d.week_idx, "flash": d.flash, "y_pred": vv.values})
    return float(H.score(t).iloc[0])

X, G, C, B = b.xgb.values, b.gbm.values, b.c2f.values, b.bis.values
print(f"공통표본 {b.tq.nunique()}분기 {len(b)}행")
print("[조기 구간 단독]", {n: round(sc(v, early), 4) for n, v in
                            [("XGB", X), ("GBM", G), ("C2f", C), ("BISTRO", B)]})
print(f"{'구성':30s} {'전체':>8s} {'반등':>8s}")
for k, v in {"XGB 단독 [기준]": X,
             "early=(GBM+C2f)/2 → XGB": np.where(early, (G + C) / 2, X),
             "early=(GBM+BISTRO)/2 → XGB": np.where(early, (G + B) / 2, X)}.items():
    print(f"{k:30s} {sc(v):8.4f} {sc(v, reb):8.4f}")
