"""schema v2 재순위 — 고객(한은) ML 단위버그 수정 반영 전면 재채점 (2026-08-04).

배경 (docs/code_changes_from_previous_results_2026-07-30.md, 한은 클론):
  구 ML 예측치는 target 표준화(z-score) 단위 그대로 y_pred에 저장되어 있었음
  → 원 단위 DFM과 섞은 앙상블·채점은 단위 혼합으로 "경제적 의미 자체가 불성립" (고객 판정).
  수정: 원 단위 역변환 + inner 튜닝 누수 제거 + pooled RMSE + min_train 60.
  구 XGB(평균 -0.15, std 0.63) vs 신 XGB(평균 +0.59, std 0.48), 상관 0.983
  — 같은 신호의 위치·스케일 오류였고, 구 (DFM+XGB)/2=0.765는 우연한 수축 효과였음.

무효화된 수치 (전부 구 XGB 기반):
  - 구 현행최고 (DFM+XGB)/2 = 0.765  →  신 XGB로는 0.7866
  - 우리 구기록 (DFM+XGB+C2f)/3 = 0.7457  →  신 XGB로는 0.7863
  - 게이트 v1~v3 (calm arm이 구 DFM+XGB) — 재계산 전까지 인용 금지

신 순위 (동일 규약: 실시간 빈티지, flash, w[-19,-1] 주차RMSE 평균, 32Q):
  1) XGBoost(신) 단독      0.7499   ← 신 기준선
  2) (XGB+GBM)/2           0.7518
  3) (XGB+GBM+C2f)/3       0.7541
  4) GBM(신) 단독          0.7566
  5) (XGB+C2f)/2           0.7593
  6) (DFM+XGB+GBM+C2f)/4   0.7733
  7) (DFM+XGB)/2           0.7866 ≈ (DFM+XGB+C2f)/3 0.7863  ← DFM 혼합이 이제 해악
  8) C2f 단독 (불변)        0.8075   — 반등 6Q 0.5813은 여전히 전체 최강
     DFM 단독 (불변)        0.8646 / LSTM 0.9219
  결합 개선은 전 구성 비유의(DM p>0.08) — XGB 단독 대비 강건한 개선 없음.
  반등 6Q에서는 C2f 혼합이 뚜렷(XGB 0.7169 → +C2f 0.62~0.64) — 국면 조건부 가치 유지.

실행: gdp-nowcasting-renewal 루트에서 .venv-gdp/bin/python phase_b_rank_v2.py
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
assert (new.prediction_schema_version.dropna() >= 2).all(), "schema v2 아님 — 구 산출물 혼입"

def pick(mn, col):
    d = new[new.model_name == mn][KEY + ["y_pred"]].rename(columns={"y_pred": col})
    return d.drop_duplicates(KEY)

b = pick("dfm", "dfm").merge(pick("xgboost", "xgb"), on=KEY).merge(pick("gbm", "gbm"), on=KEY)
b = b.merge(new[new.model_name == "dfm"][KEY + ["flash"]].drop_duplicates(KEY), on=KEY)
c2 = norm(pd.read_csv("output/csv/_phase_b_our_chronos2f_predictions.csv", dtype={"tq": str}))
b = b.merge(c2[c2.model_name == "our_chronos2f"][KEY + ["y_pred"]].rename(columns={"y_pred": "c2f"}), on=KEY)
b = b.dropna().reset_index(drop=True)

targets = sorted(b.tq.unique(), key=lambda x: pd.Period(x, "Q"))
REB = {"2018Q1", "2019Q2", "2020Q3", "2023Q1", "2024Q3", "2025Q2"}
reb = b.tq.isin(REB); covid = b.tq.isin(["2020Q1", "2020Q2", "2020Q3"])

def sc(v, sub=None):
    d = b if sub is None else b[sub]; vv = v if sub is None else v[sub]
    t = pd.DataFrame({"model_name": "x", "tq": d.tq, "vintage": d.vintage,
                      "week_idx": d.week_idx, "flash": d.flash, "y_pred": vv})
    return float(H.score(t).iloc[0])

def dm(vA, vB):
    la = pd.Series((vA - b.flash) ** 2).groupby(b.tq).mean()
    lb = pd.Series((vB - b.flash) ** 2).groupby(b.tq).mean()
    d = (la - lb).reindex(targets).dropna().values; n = len(d); db = d.mean()
    g0 = np.mean((d - db) ** 2); g1 = np.mean((d[1:] - db) * (d[:-1] - db))
    var = max((g0 + 2 * g1) / n, g0 / n / 10)
    t = db / np.sqrt(var) * np.sqrt((n - 1) / n)
    return 2 * (1 - stats.t.cdf(abs(t), df=n - 1))

C = {
    "XGB(신) 단독":             b.xgb,
    "(XGB+GBM)/2":              (b.xgb + b.gbm) / 2,
    "(XGB+GBM+C2f)/3":          (b.xgb + b.gbm + b.c2f) / 3,
    "GBM(신) 단독":             b.gbm,
    "(XGB+C2f)/2":              (b.xgb + b.c2f) / 2,
    "(DFM+XGB)/2 [구 현행]":    (b.dfm + b.xgb) / 2,
    "(DFM+XGB+C2f)/3 [구기록]": (b.dfm + b.xgb + b.c2f) / 3,
    "C2f 단독 (불변)":           b.c2f,
    "DFM 단독 (불변)":           b.dfm,
}
print(f"공통 그리드: {len(b)}행 {b.tq.nunique()}분기")
print(f"{'구성':26s} {'전체32Q':>8s} {'exCOVID':>8s} {'반등6Q':>8s} {'DM vs XGB신':>10s}")
for k, v in C.items():
    p = "-" if k.startswith("XGB(신)") else f"{dm(v, b.xgb):.3f}"
    print(f"{k:26s} {sc(v):8.4f} {sc(v, ~covid):8.4f} {sc(v, reb):8.4f} {p:>10s}")
