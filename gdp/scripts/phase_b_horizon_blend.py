"""전망시계(주차) 조건화 하이브리드 — XGB(신) 단독을 수치상 이기는 유일한 구조 (2026-08-04).

배경: schema v2 이후 XGB 단독(0.7499)이 신 기준선. 국면 게이트는 기각됐지만
"전망시계 조건화"(주차별 다른 구성)는 fan chart에서 이미 쓴 표준 관행 — 정치적으로 안전.

진단 (주차 구간별 RMSE, 32Q):
              XGB     GBM     C2f
  early(-19~-14) 0.8877  0.8694  0.8686   ← 조기엔 XGB가 최강이 아님
  mid(-13~-8)    0.7112  0.7536  0.8378   ← 중반부터 XGB 독주
  late(-7~-1)    0.6151  0.6154  0.6818
메커니즘: 조기 주차 = 월별 공식지표 정보 빈곤기 → 사전학습(C2f)·완만한 GBM 우위.
중·후반 = 지표 축적 → 튜닝된 XGB 독주. 버킷 경계(-14)는 fan chart에서 선등록(7/27).

결과 (기준 XGB 0.7499):
  H2: early=(GBM+C2f)/2, 이후 XGB      = 0.7399 (-1.3%), 반등 0.6659, 17/32승, DM p=0.41
  H1: early=(X+G+C2f)/3, 이후 XGB      = 0.7422, 반등 0.6823, 18/32승
  H3: early=C2f, 이후 XGB              = 0.7429, 반등 0.6498
  M1: median(X,G,C2f) 전주차           = 0.7569 (기각 — 중후반에서 XGB 희석)

정직한 한계: ①7개 구성 중 최선 선택 → 선택편의 (버킷 경계만 선등록, 구성은 사후)
②p=0.41 비유의 — 32분기 검정력으론 -1.3% 입증 불가 ③보고 수위는 "비열등+개선 방향".
가치는 배포보다 진단: "XGB의 왕좌는 중·후반 주차의 것, 조기 주차는 여전히 열린 시장".

경계 민감도 (2026-08-13 추가 실측 — "6주는 어떻게 정했나" 대응):
  조기 4주 0.7405 / 5주 0.7402 / 6주(채택) 0.7399 / 7주 0.7409 / 8주 0.7439 / 9주 0.7492 / 10주 0.7545
  → 4~7주 평탄(경계 비의존), 8주+에선 XGB 우위 구간 침범으로 이득 소멸 — 메커니즘 방증.
  경계 -14 자체는 fan chart 구간 보정(7/27)에서 19주 3등분으로 선정의된 값(사후 튜닝 아님).

실행: gdp-nowcasting-renewal 루트에서 .venv-gdp/bin/python phase_b_horizon_blend.py
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
c2 = norm(pd.read_csv("output/csv/_phase_b_our_chronos2f_predictions.csv", dtype={"tq": str}))
b = b.merge(c2[c2.model_name == "our_chronos2f"][KEY + ["y_pred"]].rename(columns={"y_pred": "c2f"}), on=KEY)
b = b.dropna().reset_index(drop=True)
early = (b.week_idx <= -14).values

targets = sorted(b.tq.unique(), key=lambda x: pd.Period(x, "Q"))
REB = {"2018Q1", "2019Q2", "2020Q3", "2023Q1", "2024Q3", "2025Q2"}
reb = b.tq.isin(REB); covid = b.tq.isin(["2020Q1", "2020Q2", "2020Q3"])
def sc(v, sub=None):
    d = b if sub is None else b[sub]; vv = pd.Series(v) if sub is None else pd.Series(v)[sub.values]
    t = pd.DataFrame({"model_name": "x", "tq": d.tq, "vintage": d.vintage,
                      "week_idx": d.week_idx, "flash": d.flash, "y_pred": vv.values})
    return float(H.score(t).iloc[0])
def dm(vA, vB):
    la = pd.Series((vA - b.flash) ** 2).groupby(b.tq).mean()
    lb = pd.Series((vB - b.flash) ** 2).groupby(b.tq).mean()
    d = (la - lb).reindex(targets).dropna().values; n = len(d); db = d.mean()
    g0 = np.mean((d - db) ** 2); g1 = np.mean((d[1:] - db) * (d[:-1] - db))
    var = max((g0 + 2 * g1) / n, g0 / n / 10)
    t = db / np.sqrt(var) * np.sqrt((n - 1) / n)
    return 2 * (1 - stats.t.cdf(abs(t), df=n - 1))

X, G, C = b.xgb.values, b.gbm.values, b.c2f.values
CFG = {
    "XGB 단독 [기준]":                 X,
    "H2: early=(G+C2f)/2, 이후 X":     np.where(early, (G + C) / 2, X),
    "H1: early=(X+G+C2f)/3, 이후 X":   np.where(early, (X + G + C) / 3, X),
    "H3: early=C2f, 이후 X":           np.where(early, C, X),
    "M1: median(X,G,C2f) 전주차":      np.median(np.vstack([X, G, C]), axis=0),
}
print(f"{'구성':34s} {'전체32Q':>8s} {'exCOVID':>8s} {'반등6Q':>8s} {'DM':>6s}")
for k, v in CFG.items():
    p = "-" if "기준" in k else f"{dm(pd.Series(v), pd.Series(X)):.3f}"
    print(f"{k:34s} {sc(v):8.4f} {sc(v, ~covid):8.4f} {sc(v, reb):8.4f} {p:>6s}")
