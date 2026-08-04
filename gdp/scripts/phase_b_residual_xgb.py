"""XGB 잔차 encompassing 보정 (XGB + λ·g) — 기각 (2026-08-04).

구조: 최종 = XGB + ridge(gap, gap×early, kospi3m, krw3m, esi_mom) — walk-forward,
release-safe(q-2), 표준화 후 ridge. 킬 기준 = 같은 표본에서 H2 고정 하이브리드 미달 시 폐기.

결과 (walk-forward 공통표본 23분기, 2020Q2~):
  XGB 단독            0.7349 (반등 0.8502)
  H2 고정 하이브리드    0.7236 (반등 0.7891)   ← 킬 기준
  R1 ridge α=10       0.7875 (반등 0.6879)   ← 기각: XGB 단독보다도 악화
  R2 gap만            0.8292 / R1s α=1 0.7992 / R1w α=100 0.7420
  수축 단조성: α↑(보정→0)일수록 개선 → 학습된 보정은 표본외에서 노이즈.

encompassing 판정: 잔차에 신호의 '방향'은 있음 (평균 계수: esi_mom +0.29,
gap_early +0.13 — 심리 모멘텀 상승기에 XGB가 과소예측, 조기 주차에서 C2f 견해가 유익)
— 그러나 32분기 유효표본에서 그 크기를 '학습'하면 표본외 전패. 반등 6Q는 개선
(0.85→0.69)되나 평시 손실이 더 큼. 결론: 고정 구조(H2)만 생존 — "가중치 학습
전패, 적응 가중 전패, 잔차학습 전패"로 소표본 3연속 동일 교훈.

실행: gdp-nowcasting-renewal 루트에서 .venv-gdp/bin/python phase_b_residual_xgb.py
"""
import sys, glob, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import phase_b_harness as H
import fast_signals as FS

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

sig = {}
for v in b.vintage.unique():
    mi = pd.date_range(end=pd.Timestamp(v) + pd.offsets.MonthEnd(0), periods=6, freq="ME")
    cov = FS.monthly_covariates(mi, v)
    sig[v] = dict(kospi3=float(cov.kospi_mret.iloc[-3:].sum()),
                  krw3=float(cov.krw_mret.iloc[-3:].sum()),
                  esimom=float(cov.esi_mom.iloc[-1]))
b = b.join(pd.DataFrame(sig).T, on="vintage")
b["early"] = (b.week_idx <= -14).astype(float)
b["gap"] = b.c2f - b.xgb
b["gap_early"] = b.gap * b.early
b["e"] = b.flash - b.xgb

targets = sorted(b.tq.unique(), key=lambda x: pd.Period(x, "Q")); qi = {q: i for i, q in enumerate(targets)}
FEATS = ["gap", "gap_early", "kospi3", "krw3", "esimom"]
MIN_Q = 8

def walkforward(feats, alpha):
    pred = np.full(len(b), np.nan)
    for q in targets:
        tr_qs = set(targets[:max(0, qi[q] - 1)])
        if len(tr_qs) < MIN_Q: continue
        tr = b.tq.isin(tr_qs); te = (b.tq == q)
        scX = StandardScaler().fit(b.loc[tr, feats])
        m = Ridge(alpha=alpha).fit(scX.transform(b.loc[tr, feats]), b.loc[tr, "e"])
        pred[te.values] = b.loc[te, "xgb"].values + m.predict(scX.transform(b.loc[te, feats]))
    return pred

covered = (b.tq.map(qi) >= MIN_Q + 1).values
REB = {"2018Q1", "2019Q2", "2020Q3", "2023Q1", "2024Q3", "2025Q2"}
reb = (b.tq.isin(REB).values & covered)
def sc(v, mask):
    d = b[mask]
    t = pd.DataFrame({"model_name": "x", "tq": d.tq, "vintage": d.vintage,
                      "week_idx": d.week_idx, "flash": d.flash, "y_pred": pd.Series(v)[mask].values})
    return float(H.score(t).iloc[0])

X, G, C = b.xgb.values, b.gbm.values, b.c2f.values
H2 = np.where(b.early.values == 1, (G + C) / 2, X)
print(f"walk-forward 공통표본: {b[covered].tq.nunique()}분기")
print(f"{'구성':26s} {'전체':>8s} {'반등':>8s}")
for name, v in [("XGB 단독", X), ("H2 고정 하이브리드", H2)]:
    print(f"{name:26s} {sc(v, covered):8.4f} {sc(v, reb):8.4f}")
for a in [1, 10, 100]:
    p = walkforward(FEATS, a)
    print(f"{'R ridge α=' + str(a):26s} {sc(p, covered):8.4f} {sc(p, reb):8.4f}")
