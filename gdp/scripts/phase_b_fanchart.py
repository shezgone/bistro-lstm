"""Fan Chart 상품화 엔진 — 예측 불변, 컨포멀 보정 + 전망시계 하이브리드.

구성:
  - 중심선: 현직(DFM+XGB) 점예측 그대로 (불변)
  - 구간: 컨포멀 (과거 오차 경험분위수, 확장창·주차버킷별, release-safe q−2)
  - 하이브리드: 조기 주차(w≤−14)는 Chronos-2f(빠른신호) 중위수+컨포멀 구간이
    분포적으로 우수 → 조기 구간만 C2f-컨포멀 사용 ("전망시계 조건화" — 국면 스킴 아님)
출력:
  - 검증표 (coverage/폭/Winkler, 전체·ex-COVID)
  - 데모 분기 fan chart 시계열 CSV (시각화 재료): output/csv/_fanchart_demo.csv
"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
import phase_b_harness as H

QL = [0.10, 0.25, 0.75, 0.90]
MIN_TRAIN_Q = 8
DEMO_QS = ["2025Q2", "2025Q3"]

grid, refdf = H.load_grid(); KEY = ["tq", "vintage", "week_idx"]
def norm(d):
    d = d.copy(); d["vintage"] = pd.to_datetime(d["vintage"]).dt.strftime("%Y-%m-%d"); return d
b = norm(H.load_baseline(grid, "dfm"))[KEY + ["y_pred", "flash"]].rename(columns={"y_pred": "dfm"})
b = b.merge(norm(refdf[refdf.model_name == "xgboost"])[KEY + ["y_pred"]].rename(columns={"y_pred": "xgb"}), on=KEY)
b = b.dropna().reset_index(drop=True)
b["inc"] = (b.dfm + b.xgb) / 2
c2 = norm(pd.read_csv("output/csv/_phase_b_our_chronos2f_predictions.csv", dtype={"tq": str}))
b = b.merge(c2[KEY + ["y_pred"]].rename(columns={"y_pred": "c2f"}), on=KEY, how="left")

targets = sorted(b.tq.unique(), key=lambda x: pd.Period(x, "Q"))
qi = {q: i for i, q in enumerate(targets)}
b["bucket"] = np.where(b.week_idx <= -14, "early", np.where(b.week_idx <= -8, "mid", "late"))

def conformal(base_col, prefix):
    e = b.flash - b[base_col]
    for q in QL: b[f"{prefix}{int(q*100)}"] = np.nan
    for tq in targets:
        hs = set(targets[:max(0, qi[tq] - 1)])
        if len(hs) < MIN_TRAIN_Q: continue
        for bk in ["early", "mid", "late"]:
            tr = b.tq.isin(hs) & (b.bucket == bk) & b[base_col].notna()
            if tr.sum() < 30: tr = b.tq.isin(hs) & b[base_col].notna()
            errs = e[tr].dropna().values
            if len(errs) < 30: continue
            m = ((b.tq == tq) & (b.bucket == bk)).values
            for q in QL:
                b.loc[m, f"{prefix}{int(q*100)}"] = b.loc[m, base_col].values + np.quantile(errs, q)
conformal("inc", "i")
conformal("c2f", "c")

# 하이브리드: 조기=C2f-컨포멀(가용 시), 그 외=현직-컨포멀. 중심선은 항상 현직(불변).
use_c2 = (b.bucket == "early") & b.c50 .notna() if "c50" in b else None
b["hy10"] = np.where((b.bucket == "early") & b.c10.notna(), b.c10, b.i10)
b["hy25"] = np.where((b.bucket == "early") & b.c25.notna(), b.c25, b.i25)
b["hy75"] = np.where((b.bucket == "early") & b.c75.notna(), b.c75, b.i75)
b["hy90"] = np.where((b.bucket == "early") & b.c90.notna(), b.c90, b.i90)

def winkler(lo, hi, y, a=0.2):
    w = hi - lo
    return w + np.where(y < lo, 2/a*(lo-y), np.where(y > hi, 2/a*(y-hi), 0.0))
ev = b.dropna(subset=["i10", "hy10"]).copy()
covid = ev.tq.isin(["2020Q1", "2020Q2", "2020Q3"])
def rep(d, lab):
    for name, (l, h25, h75, u) in {"현직+컨포멀": ("i10","i25","i75","i90"),
                                    "하이브리드(조기=C2f)": ("hy10","hy25","hy75","hy90")}.items():
        c80 = ((d.flash >= d[l]) & (d.flash <= d[u])).mean()
        c50 = ((d.flash >= d[h25]) & (d.flash <= d[h75])).mean()
        wk = winkler(d[l].values, d[u].values, d.flash.values).mean()
        print(f"  [{lab}] {name:18s} cov80 {c80:.0%} cov50 {c50:.0%} 폭 {(d[u]-d[l]).mean():.2f} Winkler {wk:.3f}")
print(f"발동 표본: {len(ev)}행 {ev.tq.nunique()}분기")
rep(ev, "전체"); rep(ev[~covid], "exCOVID")

demo = b[b.tq.isin(DEMO_QS)][KEY + ["flash", "inc", "hy10", "hy25", "hy75", "hy90"]].sort_values(["tq", "week_idx"])
demo.to_csv("output/csv/_fanchart_demo.csv", index=False)
print("saved: output/csv/_fanchart_demo.csv (데모:", DEMO_QS, ")")
