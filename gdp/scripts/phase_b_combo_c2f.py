"""무게이트 상시 앙상블 — 기존최고(DFM+XGB) × Chronos-2f(빠른신호) 결합 검증.

배경: 7/22 무게이트 앙상블 검증((DFM+XGB+TTM)/3=0.7638≈0.765)은 Chronos-2 등장 전.
Chronos-2f(0.8075)는 정보원이 다름(일별 KOSPI·원달러 + ESI 원지수 공변량 zero-shot)
→ 결합 다양성 가치가 있어 유일하게 남은 미검증 조합이었음.

결과 (2026-07-29):
  - (DFM+XGB+C2f)/3 균등가중 = 0.7457 (기존최고 0.7650 대비 −2.5%) — 게이트 없이 첫 돌파
  - 가중치 w(INC)∈[0.3,0.7] 전 구간 0.746~0.764 < 0.765 (가중 선택 비의존, 강건)
  - 반등 6Q 0.7150 (<0.8153), exCOVID 0.5359 (<0.5813), 분기별 20/32 승
  - DM vs INC p=0.367 — 비유의 (32분기 검정력 한계와 일관, 방향은 개선)
  - 4자((+TTM)/4=0.7592)·5자(0.7673)는 후퇴 — 약한 부품 추가는 희석
  - 적응 가중(확장창 역MSE) 0.7629 — 균등가중보다 못함 (소표본 가중 학습 한계 재확인)

실행: gdp-nowcasting-renewal 루트에서
  .venv-gdp/bin/python phase_b_combo_c2f.py
선행조건: output/csv/_phase_b_our_chronos2f_predictions.csv (phase_b_chronos2.py FAST_COV=1)
"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy import stats
import phase_b_harness as H

grid, refdf = H.load_grid(); KEY = ["tq", "vintage", "week_idx"]
def norm(d):
    d = d.copy(); d["vintage"] = pd.to_datetime(d["vintage"]).dt.strftime("%Y-%m-%d"); return d
b = norm(H.load_baseline(grid, "dfm"))[KEY + ["y_pred", "flash"]].rename(columns={"y_pred": "dfm"})
b = b.merge(norm(refdf[refdf.model_name == "xgboost"])[KEY + ["y_pred"]].rename(columns={"y_pred": "xgb"}), on=KEY)
c2 = norm(pd.read_csv("output/csv/_phase_b_our_chronos2f_predictions.csv", dtype={"tq": str}))
b = b.merge(c2[c2.model_name == "our_chronos2f"][KEY + ["y_pred"]].rename(columns={"y_pred": "c2f"}), on=KEY)
b = b.dropna().reset_index(drop=True)
b["inc"] = (b.dfm + b.xgb) / 2

targets = sorted(b.tq.unique(), key=lambda x: pd.Period(x, "Q"))
REB = {"2018Q1", "2019Q2", "2020Q3", "2023Q1", "2024Q3", "2025Q2"}
reb = b.tq.isin(REB)
covid = b.tq.isin(["2020Q1", "2020Q2", "2020Q3"])

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

print(f"기준: DFM+XGB(INC) {sc(b.inc):.4f} | Chronos-2f 단독 {sc(b.c2f):.4f}")
print(f"\n{'구성':16s} {'전체32Q':>8s} {'exCOVID':>8s} {'반등6Q':>8s} {'DM vs INC':>9s}")
for w in [0.7, 0.6, 0.5, 0.4, 0.3]:
    v = w * b.inc + (1 - w) * b.c2f
    print(f"w(INC)={w:<9.1f} {sc(v):8.4f} {sc(v, ~covid):8.4f} {sc(v, reb):8.4f} {dm(v, b.inc):9.3f}")
tri = (b.dfm + b.xgb + b.c2f) / 3
print(f"{'(D+X+C2f)/3':16s} {sc(tri):8.4f} {sc(tri, ~covid):8.4f} {sc(tri, reb):8.4f} {dm(tri, b.inc):9.3f}")

b["tri"] = tri
q = b.groupby("tq").apply(lambda d: pd.Series({
    "inc": np.sqrt(np.mean((d.inc - d.flash) ** 2)),
    "tri": np.sqrt(np.mean((d.tri - d.flash) ** 2))}))
wins = (q.tri < q.inc).sum()
print(f"\n분기별 승패: 3-way 개선 {wins}/{len(q)}분기 (중앙값 {float((q.tri - q.inc).median()):+.4f})")
