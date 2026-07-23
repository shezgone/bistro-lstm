"""미사용 39컬럼 정보가치 검정: 기존최고(DFM+XGB) 잔차를 원시 빈티지 피처로 보정.

변형:
  V0 기존최고 그대로 (0.765)
  V1 잔차보정 — 사용중 34컬럼 파생 피처만 (대조군: 잔차 모델링 효과 분리)
  V2 잔차보정 — 미사용 컬럼 피처만 (검정 대상)
  V3 잔차보정 — 둘 다
피처: 각 컬럼의 (a) 모멘텀 = 최근 3개월 평균 − 직전 3개월 평균 (z화)
              (b) 수준 z = 최근값 vs 직전 5년   + week_idx
학습: 분기 확장창 (release-safe: target q 기준 q−2까지), ridge (소표본 안전).
이력 부족 분기는 보정 0 → 전체 32Q가 V0과 직접 비교 가능.
"""
import sys, os, glob, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy import stats
import phase_b_harness as H

USED = ["S_es", "S_cb", "S_mo", "I_m", "I_s", "M_s", "B_bx", "R_s"]            # 사용중(스펙) 대표 8
UNUSED = ["new_usli", "new_chli", "new_coin", "new_esi", "B_gx", "B_si", "M_ic", "new_ksp"]  # 미사용 대표 8
MIN_TRAIN_Q = 8
RIDGE = 3.0

grid, refdf = H.load_grid(); KEY = ["tq", "vintage", "week_idx"]
def norm(d):
    d = d.copy(); d["vintage"] = pd.to_datetime(d["vintage"]).dt.strftime("%Y-%m-%d"); return d
b = norm(H.load_baseline(grid, "dfm"))[KEY + ["y_pred", "flash"]].rename(columns={"y_pred": "dfm"})
b = b.merge(norm(refdf[refdf.model_name == "xgboost"])[KEY + ["y_pred"]].rename(columns={"y_pred": "xgb"}), on=KEY)
b = b.dropna().reset_index(drop=True)
b["base"] = (b.dfm + b.xgb) / 2
b["resid"] = b.flash - b.base

# ---- 원시 빈티지 피처 (파일 캐시) ----
VFILES = sorted(glob.glob("data/vintages/*.xlsx"))
VDATES = [os.path.basename(f)[:-5] for f in VFILES]
_cache = {}
def load_raw(vintage):
    cands = [d for d in VDATES if d <= vintage]
    if not cands: return None
    f = cands[-1]
    if f not in _cache:
        x = pd.read_excel(f"data/vintages/{f}.xlsx")
        if "date" in x.columns: x = x.rename(columns={"date": "Date"})
        x["Date"] = pd.to_datetime(x["Date"])
        _cache[f] = x.set_index("Date")
    return _cache[f]

def feats(vintage, cols):
    x = load_raw(vintage)
    out = {}
    for c in cols:
        if x is None or c not in x.columns:
            out[f"{c}_mom"] = 0.0; out[f"{c}_lvl"] = 0.0; continue
        s = x[c].dropna()
        if len(s) < 66:
            out[f"{c}_mom"] = 0.0; out[f"{c}_lvl"] = 0.0; continue
        hist = s.tail(66)                      # 최근 5.5년
        mom = hist.tail(3).mean() - hist.iloc[-6:-3].mean()
        mom_hist = hist.rolling(3).mean().diff(3).dropna()
        msd = mom_hist.std() or 1.0
        lsd = hist.std() or 1.0
        out[f"{c}_mom"] = float(np.clip(mom / msd, -4, 4))
        out[f"{c}_lvl"] = float(np.clip((hist.iloc[-1] - hist.mean()) / lsd, -4, 4))
    return out

print(f"[build] 피처 계산: {b[['vintage']].drop_duplicates().shape[0]}개 빈티지 × {len(USED)+len(UNUSED)}컬럼", flush=True)
F = {}
for v in sorted(b.vintage.unique()):
    F[v] = feats(v, USED + UNUSED)
FD = pd.DataFrame([F[v] for v in b.vintage], index=b.index)
FD["wk"] = b.week_idx.astype(float) / 19.0

COLSETS = {"V1_used34파생": [f"{c}_{k}" for c in USED for k in ("mom", "lvl")] + ["wk"],
           "V2_unused":     [f"{c}_{k}" for c in UNUSED for k in ("mom", "lvl")] + ["wk"],
           "V3_both":       [f"{c}_{k}" for c in USED + UNUSED for k in ("mom", "lvl")] + ["wk"]}

targets = sorted(b.tq.unique(), key=lambda x: pd.Period(x, "Q"))
qi = {q: i for i, q in enumerate(targets)}
def ridge_fit(X, y, lam=RIDGE):
    Xa = np.c_[np.ones(len(X)), X]
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1]); A[0, 0] -= lam
    return np.linalg.solve(A, Xa.T @ y)

for name, cols in COLSETS.items():
    pred = np.zeros(len(b))
    for q in targets:
        hs = targets[:max(0, qi[q] - 1)]                    # q−2까지 (release-safe)
        if len(hs) < MIN_TRAIN_Q: continue
        tr = b.tq.isin(hs).values
        beta = ridge_fit(FD.loc[tr, cols].values, b.resid[tr].values)
        m = (b.tq == q).values
        pred[m] = np.c_[np.ones(m.sum()), FD.loc[m, cols].values] @ beta
    b[name] = b.base + np.clip(pred, -1.5, 1.5)

def sc(col, sub=None):
    d = b if sub is None else b[sub]
    t = pd.DataFrame({"model_name": col, "tq": d.tq, "vintage": d.vintage,
                      "week_idx": d.week_idx, "flash": d.flash, "y_pred": d[col]})
    return float(H.score(t).iloc[0])
def dm(colA, colB):
    la = b.groupby("tq").apply(lambda d: np.nanmean((d[colA] - d.flash) ** 2))
    lb = b.groupby("tq").apply(lambda d: np.nanmean((d[colB] - d.flash) ** 2))
    d = (la - lb).reindex(targets).dropna().values; n = len(d)
    db = d.mean(); g0 = np.mean((d - db) ** 2); g1 = np.mean((d[1:] - db) * (d[:-1] - db))
    var = max((g0 + 2 * g1) / n, g0 / n / 10)
    t = db / np.sqrt(var) * np.sqrt((n - 1) / n)
    return 2 * (1 - stats.t.cdf(abs(t), df=n - 1))

REB = {'2018Q1','2019Q2','2020Q3','2023Q1','2024Q3','2025Q2'}
reb = b.tq.isin(REB)
eval_q = [q for q in targets if qi[q] - 1 >= MIN_TRAIN_Q]     # 보정 발동 분기
ev = b.tq.isin(eval_q)
print(f"\n보정 발동: {len(eval_q)}분기 ({eval_q[0]}~) — 이전 분기는 보정 0")
print(f"{'변형':16s} {'전체32Q':>8s} {'발동분기만':>9s} {'반등6Q':>8s} {'DM vs V0':>9s}")
print(f"{'V0 기존최고':16s} {sc('base'):8.4f} {sc('base', ev):9.4f} {sc('base', reb):8.4f} {'-':>9s}")
for name in COLSETS:
    print(f"{name:16s} {sc(name):8.4f} {sc(name, ev):9.4f} {sc(name, reb):8.4f} {dm(name,'base'):9.3f}")
