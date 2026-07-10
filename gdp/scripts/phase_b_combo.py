"""GDP 예측 개선 후보 일괄 검증 — 예측결합 문헌 × soft gate 합성.

모든 방법은 vintage-safe: 가중치/보정에 쓰는 오차 이력은 target 분기 q 기준
q−2 분기까지만 (q−1 flash는 q 중반에야 발표되므로 보수적으로 제외), 확장창.

방법 (문헌 근거):
  M1 debias    : DFM + 과거 평균오차 보정 (Clements–Hendry 절편 보정)
  M2 bg        : 역-MSE 가중 결합 {DFM, DFM+XGB, DFM+RF, DFM+TabPFN} (Bates–Granger 1969)
  M3 bg_regime : M2를 v2 국면(REBOUND/SHOCK/CALM)별 이력으로 (Aiolfi–Timmermann 조건부 결합)
  M4 lam_week  : 주차 구간별 보정강도 λ(bucket) 학습 — soft gate의 horizon 축 확장
  M5 stack     : ridge 스태킹 (Granger–Ramanathan 1984), 주차 구간별·확장창
  M6 median    : median(DFM, DFM+XGB, DFM+RF) (Stock–Watson 2004 단순결합 강건성)
  M7 v2_tabpfn : v2 게이트에서 SHOCK arm을 (DFM+TabPFN)/2로 교체 (사후 구조 탐색 — 라벨 명시)
  M8 soft_x    : M3 결합 위에 M4 λ(bucket) 중첩 — "국면×전망시계 2축 soft gate"
승자는 DM 검정 (분기 손실차, HLN 소표본 보정, NW lag1).
"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
import phase_b_harness as H

# ---------- 데이터 ----------
grid, refdf = H.load_grid(); KEY = ["tq", "vintage", "week_idx"]
def norm(d):
    d = d.copy(); d["vintage"] = pd.to_datetime(d["vintage"]).dt.strftime("%Y-%m-%d"); return d
b = norm(H.load_baseline(grid, "dfm"))[KEY + ["y_pred", "flash"]].rename(columns={"y_pred": "dfm"})
for m in ["xgboost", "rf"]:
    b = b.merge(norm(refdf[refdf.model_name == m])[KEY + ["y_pred"]].rename(columns={"y_pred": m}), on=KEY, how="left")
tab = norm(pd.read_csv("output/csv/_phase_b_tabpfn_predictions.csv", dtype={"tq": str}))
tab = tab[tab.model_name == "our_tabpfn"][KEY + ["y_pred"]].rename(columns={"y_pred": "tabpfn"})
b = b.merge(tab, on=KEY, how="left")
b = b.merge(norm(pd.read_csv("output/csv/_phase_b_regime_gated.csv", dtype={"tq": str}))[KEY + ["gated"]], on=KEY)
b = b.merge(norm(pd.read_csv("output/csv/_phase_b_regime_gated_v2.csv", dtype={"tq": str}))[KEY + ["gated_v2"]], on=KEY)
b = b.dropna(subset=["dfm", "xgboost", "rf", "flash"]).reset_index(drop=True)
b["dfm_xgb"] = (b.dfm + b.xgboost) / 2; b["dfm_rf"] = (b.dfm + b.rf) / 2
b["dfm_tab"] = (b.dfm + b.tabpfn) / 2

targets = sorted(b.tq.unique(), key=lambda x: pd.Period(x, "Q"))
qi = {q: i for i, q in enumerate(targets)}
REB = {'2018Q1','2019Q2','2020Q3','2023Q1','2024Q3','2025Q2'}
rel = pd.read_pickle("data/GDP_releases.pkl"); fl = rel["flash"].dropna()
qs_all = [str(p) for p in pd.period_range("2017Q1", "2025Q4", freq="Q")]
flq = fl.reindex([q for q in qs_all if q in fl.index])
def vol_before(q, K=4):
    past = [k for k in flq.index if pd.Period(k, "Q") < pd.Period(q, "Q")]
    return float(flq.loc[past[-K:]].std()) if len(past) >= 2 else np.nan
vols = {q: vol_before(q) for q in targets}
def regime(q):
    if q in REB: return "REB"
    i = qi[q]; pv = [vols[targets[j]] for j in range(i) if not np.isnan(vols[targets[j]])]
    return "SHOCK" if (np.isnan(vols[q]) or len(pv) < 2 or vols[q] > np.median(pv)) else "CALM"
b["regime"] = b.tq.map(regime)
BUCKET = lambda w: "early" if w <= -14 else ("mid" if w <= -8 else "late")
b["bucket"] = b.week_idx.map(BUCKET)

CANDS = ["dfm", "dfm_xgb", "dfm_rf", "dfm_tab"]
# 분기별 후보 MSE (주차 풀링) — 가중치 학습용
qmse = b.groupby("tq").apply(lambda d: pd.Series({c: np.nanmean((d[c] - d.flash) ** 2) for c in CANDS}))

def hist_qs(q, same_regime=None):
    """q−2까지 분기 (release-safe). same_regime 지정 시 그 국면만."""
    hs = targets[:max(0, qi[q] - 1)]
    if same_regime: hs = [h for h in hs if regime(h) == same_regime]
    return hs

# ---------- M1 debias ----------
def m1(row_q):
    hs = hist_qs(row_q)
    if len(hs) < 4: return 0.0
    e = [float((b[b.tq == h].flash - b[b.tq == h].dfm).mean()) for h in hs]
    return float(np.mean(e))
bias = {q: m1(q) for q in targets}
b["M1_debias"] = b.dfm + b.tq.map(bias)

# ---------- M2/M3 Bates–Granger ----------
def bg_weights(hs):
    if len(hs) < 4: return None
    mse = qmse.loc[hs, CANDS].mean()
    if mse.isna().any(): mse = mse.fillna(mse.max() * 2)
    w = 1.0 / mse.clip(lower=1e-6); return (w / w.sum()).values
W2 = {q: bg_weights(hist_qs(q)) for q in targets}
W3 = {}
for q in targets:
    w = bg_weights(hist_qs(q, regime(q)))
    W3[q] = w if w is not None else W2[q]
def apply_w(Wmap, col):
    out = np.full(len(b), np.nan)
    P = b[CANDS].values
    for q in targets:
        m = (b.tq == q).values
        w = Wmap[q]
        out[m] = b.loc[m, "gated_v2"] if w is None else P[m] @ w
    b[col] = out
apply_w(W2, "M2_bg"); apply_w(W3, "M3_bg_regime")

# ---------- M4 horizon λ (v2 보정 강도의 주차별 스케일) ----------
GRIDW = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
def lam_for(q, bucket):
    hs = hist_qs(q)
    if len(hs) < 4: return 1.0
    h = b[b.tq.isin(hs) & (b.bucket == bucket)]
    corr = h.gated_v2 - h.dfm
    best, bw = np.inf, 1.0
    for w in GRIDW:
        r = np.nanmean((h.dfm + w * corr - h.flash) ** 2)
        if r < best: best, bw = r, w
    return bw
lam = {(q, bk): lam_for(q, bk) for q in targets for bk in ["early", "mid", "late"]}
b["M4_lam_week"] = b.dfm + b.apply(lambda r: lam[(r.tq, r.bucket)], axis=1) * (b.gated_v2 - b.dfm)

# ---------- M5 ridge 스태킹 (주차구간별·확장창) ----------
def stack_pred(q, bucket, rows):
    hs = hist_qs(q)
    if len(hs) < 8: return rows["gated_v2"].values
    h = b[b.tq.isin(hs) & (b.bucket == bucket)].dropna(subset=CANDS)
    if len(h) < 30: return rows["gated_v2"].values
    X = h[CANDS].values; y = h.flash.values
    Xa = np.c_[np.ones(len(X)), X]
    A = Xa.T @ Xa + 1.0 * np.eye(Xa.shape[1]); A[0, 0] -= 1.0
    beta = np.linalg.solve(A, Xa.T @ y)
    Xt = np.c_[np.ones(len(rows)), rows[CANDS].values]
    return Xt @ beta
b["M5_stack"] = np.nan
for q in targets:
    for bk in ["early", "mid", "late"]:
        m = (b.tq == q) & (b.bucket == bk)
        if m.any(): b.loc[m, "M5_stack"] = stack_pred(q, bk, b.loc[m])

# ---------- M6 median ----------
b["M6_median"] = b[["dfm", "dfm_xgb", "dfm_rf"]].median(axis=1)

# ---------- M7 v2 게이트 + TabPFN arm (사후 탐색 — 라벨용) ----------
b["M7_v2_tab"] = np.where(b.regime == "REB", b.dfm,
                  np.where(b.regime == "SHOCK", b.dfm_tab.fillna(b.dfm_xgb), b.dfm_rf))

# ---------- M8 국면×시계 2축 soft ----------
b["M8_soft_x"] = b.dfm + b.apply(lambda r: lam[(r.tq, r.bucket)], axis=1) * (b.M3_bg_regime - b.dfm)

# ---------- 채점 ----------
def sc(col, sub=None):
    d = b if sub is None else b[sub]
    t = pd.DataFrame({"model_name": col, "tq": d.tq, "vintage": d.vintage,
                      "week_idx": d.week_idx, "flash": d.flash, "y_pred": d[col]})
    s = H.score(t.dropna(subset=["y_pred"])); return float(s.iloc[0]) if len(s) else np.nan
METHODS = ["M1_debias","M2_bg","M3_bg_regime","M4_lam_week","M5_stack","M6_median","M7_v2_tab","M8_soft_x"]
reb = b.tq.isin(REB); rec = b.tq.isin([str(p) for p in pd.period_range("2023Q1","2025Q4",freq="Q")])
print(f"{'method':16s} {'전체32Q':>8s} {'반등6Q':>8s} {'최근12Q':>8s}")
for c in ["gated","gated_v2","dfm_xgb"] + METHODS:
    print(f"{c:16s} {sc(c):8.4f} {sc(c,reb):8.4f} {sc(c,rec):8.4f}")

# ---------- DM 검정 (분기 손실차, NW lag1 + HLN) ----------
def dm(colA, colB):
    la = b.groupby("tq").apply(lambda d: np.nanmean((d[colA]-d.flash)**2))
    lb = b.groupby("tq").apply(lambda d: np.nanmean((d[colB]-d.flash)**2))
    d = (la - lb).reindex(targets).dropna().values; n = len(d)
    dbar = d.mean()
    g0 = np.mean((d-dbar)**2); g1 = np.mean((d[1:]-dbar)*(d[:-1]-dbar))
    var = (g0 + 2*g1) / n
    if var <= 0: var = g0 / n
    t = dbar / np.sqrt(var)
    t *= np.sqrt((n + 1 - 2*1 + 1*(1-1)/n) / n)   # HLN h=1
    from scipy import stats
    p = 2 * (1 - stats.t.cdf(abs(t), df=n-1))
    return t, p, n
best = min(METHODS, key=lambda c: sc(c))
print(f"\n최저 RMSE 방법: {best} = {sc(best):.4f}")
for ref in ["gated_v2", "dfm_xgb", "gated"]:
    t, p, n = dm(best, ref)
    print(f"  DM({best} vs {ref}): t={t:.2f}, p={p:.3f} (n={n}분기)  {'유의' if p<0.05 else '비유의'}")
t, p, n = dm("gated_v2", "dfm_xgb")
print(f"  [참고] DM(v2 vs DFM+XGB): t={t:.2f}, p={p:.3f}")
