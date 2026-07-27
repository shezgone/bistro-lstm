"""E2: 확률예측 축 맞대결 — 분포/구간 예측에서 무엇이 이기는가 (점예측 불변).

후보:
  CONF-INC : 현직(DFM+XGB) 점예측 + 컨포멀 구간 (과거 오차 경험분위수, 확장창·주차버킷별)
  QREG-INC : 현직 점예측 + 조건부 분위수회귀 구간 (오차 ~ 주차 + vol z + ESI mom, 확장창)
  C2-RAW   : Chronos-2 자체 분위수 (보정 없음 — 과소산포 진단용)
  C2-CONF  : Chronos-2 중위수 + 컨포멀 구간 (DL 점 + 통계 보정)
평가: 80%/50% 커버리지, 평균 구간폭, Winkler-80, pinball(가용 분위수 평균).
모든 학습/보정은 target 분기 q−2까지 (release-safe 확장창). 발동 분기만 공통 비교.
"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
import phase_b_harness as H

QL = [0.10, 0.25, 0.75, 0.90]
MIN_TRAIN_Q = 8

grid, refdf = H.load_grid(); KEY = ["tq", "vintage", "week_idx"]
def norm(d):
    d = d.copy(); d["vintage"] = pd.to_datetime(d["vintage"]).dt.strftime("%Y-%m-%d"); return d
b = norm(H.load_baseline(grid, "dfm"))[KEY + ["y_pred", "flash"]].rename(columns={"y_pred": "dfm"})
b = b.merge(norm(refdf[refdf.model_name == "xgboost"])[KEY + ["y_pred"]].rename(columns={"y_pred": "xgb"}), on=KEY)
b = b.dropna().reset_index(drop=True)
b["inc"] = (b.dfm + b.xgb) / 2
c2 = norm(pd.read_csv("output/csv/_phase_b_chronos2_predictions.csv", dtype={"tq": str}))
b = b.merge(c2[KEY + ["y_pred", "q10", "q25", "q75", "q90"]].rename(
    columns={"y_pred": "c2", "q10": "c2q10", "q25": "c2q25", "q75": "c2q75", "q90": "c2q90"}), on=KEY, how="left")

targets = sorted(b.tq.unique(), key=lambda x: pd.Period(x, "Q"))
qi = {q: i for i, q in enumerate(targets)}
b["bucket"] = np.where(b.week_idx <= -14, "early", np.where(b.week_idx <= -8, "mid", "late"))

# 조건 변수 (QREG용): vol z + ESI mom — unused_cols의 피처 빌더 재사용
import fast_signals
rel = pd.read_pickle("data/GDP_releases.pkl"); fl = rel["flash"].dropna()
qs_all = [str(p) for p in pd.period_range("2017Q1", "2025Q4", freq="Q")]
flq = fl.reindex([q for q in qs_all if q in fl.index])
def vol_z(q):
    i = targets.index(q) if q in targets else None
    past = [k for k in flq.index if pd.Period(k, "Q") < pd.Period(q, "Q")]
    if len(past) < 6: return 0.0
    v = float(flq.loc[past[-4:]].std())
    hist = [float(flq.loc[past[max(0,j-4):j]].std()) for j in range(4, len(past))]
    med = np.median(hist); mad = np.median(np.abs(np.array(hist) - med)) or 1.0
    return float(np.clip((v - med) / mad, -3, 3))
b["volz"] = b.tq.map({q: vol_z(q) for q in targets})
esim = {}
for v in b.vintage.unique():
    raw = fast_signals.load_raw_vintage(v)
    s = raw["new_esi"].dropna() if raw is not None and "new_esi" in raw else None
    esim[v] = float((s.iloc[-1] - s.tail(4).mean())) if s is not None and len(s) >= 4 else 0.0
b["esim"] = b.vintage.map(esim)

def conformal_intervals(base_col):
    """확장창·버킷별 경험 오차 분위수 → 구간 컬럼 4개 추가."""
    e = b.flash - b[base_col]
    out = {q: np.full(len(b), np.nan) for q in QL}
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
                out[q][m] = b.loc[m, base_col].values + np.quantile(errs, q)
    for q in QL: b[f"{base_col}_cf{int(q*100)}"] = out[q]

conformal_intervals("inc")
conformal_intervals("c2")

# QREG: 오차 ~ [1, wk, volz, esim] 분위수회귀 (확장창)
try:
    import statsmodels.api as sm
    for q in QL: b[f"inc_qr{int(q*100)}"] = np.nan
    feats = ["week_idx", "volz", "esim"]
    for tq in targets:
        hs = set(targets[:max(0, qi[tq] - 1)])
        if len(hs) < MIN_TRAIN_Q: continue
        tr = b.tq.isin(hs)
        Xt = sm.add_constant(b.loc[tr, feats]); yt = (b.flash - b.inc)[tr]
        m = (b.tq == tq).values
        Xp = sm.add_constant(b.loc[m, feats], has_constant="add")
        for q in QL:
            try:
                r = sm.QuantReg(yt, Xt).fit(q=q, max_iter=200)
                b.loc[m, f"inc_qr{int(q*100)}"] = b.loc[m, "inc"].values + r.predict(Xp).values
            except Exception: pass
except ImportError:
    print("statsmodels 없음 — QREG 생략")

# ---- 평가 ----
def winkler(lo, hi, y, alpha=0.2):
    w = hi - lo
    pen = np.where(y < lo, 2/alpha*(lo-y), np.where(y > hi, 2/alpha*(y-hi), 0.0))
    return w + pen
def pinball(pred_q, y, q):
    d = y - pred_q
    return np.maximum(q*d, (q-1)*d)

METHODS = {
    "CONF-INC (현직+컨포멀)": ("inc_cf10","inc_cf25","inc_cf75","inc_cf90"),
    "QREG-INC (현직+분위수회귀)": ("inc_qr10","inc_qr25","inc_qr75","inc_qr90"),
    "C2-RAW  (Chronos2 원시)": ("c2q10","c2q25","c2q75","c2q90"),
    "C2-CONF (Chronos2+컨포멀)": ("c2_cf10","c2_cf25","c2_cf75","c2_cf90"),
}
common = np.ones(len(b), bool)
for cols in METHODS.values():
    for c in cols:
        common &= b[c].notna().values if c in b.columns else False
ev = b[common].copy()
print(f"공통 평가 표본: {len(ev)}행, {ev.tq.nunique()}분기 ({sorted(ev.tq.unique())[0]}~)")
covid = ev.tq.isin(["2020Q1","2020Q2","2020Q3"])
def report(d, label):
    print(f"\n[{label}] n={len(d)}")
    print(f"{'방법':26s} {'cov80':>6s} {'cov50':>6s} {'폭80':>6s} {'Winkler80':>9s} {'pinball':>8s}")
    for name,(l10,l25,l75,l90) in METHODS.items():
        c80 = ((d.flash >= d[l10]) & (d.flash <= d[l90])).mean()
        c50 = ((d.flash >= d[l25]) & (d.flash <= d[l75])).mean()
        w80 = (d[l90] - d[l10]).mean()
        wk = winkler(d[l10].values, d[l90].values, d.flash.values).mean()
        pb = np.mean([pinball(d[c].values, d.flash.values, q).mean()
                      for c, q in zip((l10,l25,l75,l90), QL)])
        print(f"{name:26s} {c80:6.0%} {c50:6.0%} {w80:6.2f} {wk:9.3f} {pb:8.4f}")
report(ev, "전체 (발동 분기)")
report(ev[~covid], "ex-COVID")
for bk in ["early","mid","late"]:
    report(ev[ev.bucket==bk], f"bucket={bk}")
ev.to_csv("output/csv/_phase_b_prob_axis.csv", index=False)
print("\nsaved: output/csv/_phase_b_prob_axis.csv")
