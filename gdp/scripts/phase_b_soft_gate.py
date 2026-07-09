"""soft gate 타당성 프로토타입: (1) 규칙 신호 연속화, (2) 오토인코더 이상탐지 점수 게이트.

ŷ = DFM + 0.5·(1−g)·( p·XGB + (1−p)·RF − DFM )
  p: 충격 가중 ∈(0,1) — vol z(규칙) 또는 AE 재구성오차 z(딥러닝)의 시그모이드
  g: 반등 축소 ∈[0,1] — σ(−flash(q−1)/τ)·[심리 저점통과]
  (p,g → 0/1 계단화하면 v2와 동일 — v2의 연속 일반화)

AE: 월별 패널(~70지표)을 2017-12까지로만 학습(look-ahead 차단), 2018+ 재구성오차 = 충격점수.
분기 q의 점수 = q 시작 직전 3개월 평균. 데이터는 최신 빈티지(개정 반영) 사용 — 타당성 판단용 근사.
"""
import sys, math, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
import phase_b_harness as H

SENT_VINTAGE = "data/vintages/2026-03-04.xlsx"
TAU = 0.2          # 반등 강도 스케일 (flash(q-1) 크기)
SEEDS = [0, 1, 2]  # AE 시드

# ---------- 공통 로드 ----------
x = pd.read_excel(SENT_VINTAGE)
if "date" in x.columns: x = x.rename(columns={"date": "Date"})
x["Date"] = pd.to_datetime(x["Date"]); x = x.set_index("Date")
esi = x["new_esi"].dropna()
rel = pd.read_pickle("data/GDP_releases.pkl"); fl = rel["flash"].dropna()
qs = [str(p) for p in pd.period_range("2017Q1", "2025Q4", freq="Q")]
flq = fl.reindex([q for q in qs if q in fl.index])
targets = [str(p) for p in pd.period_range("2018Q1", "2025Q4", freq="Q")]

def vol_before(q, K=4):
    past = [k for k in flq.index if pd.Period(k, "Q") < pd.Period(q, "Q")]
    return float(flq.loc[past[-K:]].std()) if len(past) >= 2 else np.nan

def trough_passed(q, win=6):
    e = esi[esi.index < pd.Period(q, "Q").start_time].tail(win)
    return len(e) >= 3 and e.iloc[-1] > e.min() and e.idxmin() < e.index[-1]

sig = lambda z: 1.0 / (1.0 + math.exp(-z))
vols = {q: vol_before(q) for q in targets}

def p_rule(q):
    """vol 기반 충격 가중 (연속) — z = (vol−median)/MAD, 확장창."""
    i = targets.index(q)
    pv = np.array([vols[targets[j]] for j in range(i) if not np.isnan(vols[targets[j]])])
    if np.isnan(vols[q]) or len(pv) < 2: return 1.0        # 초기: 보수적으로 shock
    med = np.median(pv); mad = np.median(np.abs(pv - med)) or pv.std() or 1.0
    return sig((vols[q] - med) / mad)

def g_soft(q):
    i = qs.index(q); prev = qs[i-1]
    if prev not in flq.index or not trough_passed(q): return 0.0
    return sig(-float(flq[prev]) / TAU) if flq[prev] < 0 else 0.0

# ---------- 예측 그리드 ----------
grid, refdf = H.load_grid(); KEY = ["tq", "vintage", "week_idx"]
def norm(d):
    d = d.copy(); d["vintage"] = pd.to_datetime(d["vintage"]).dt.strftime("%Y-%m-%d"); return d
b = norm(H.load_baseline(grid, "dfm"))[KEY + ["y_pred", "flash"]].rename(columns={"y_pred": "dfm"})
for m in ["xgboost", "rf"]:
    b = b.merge(norm(refdf[refdf.model_name == m])[KEY + ["y_pred"]].rename(columns={"y_pred": m}), on=KEY, how="left")
gated = norm(pd.read_csv("output/csv/_phase_b_regime_gated.csv", dtype={"tq": str}))[KEY + ["gated"]]
v2 = norm(pd.read_csv("output/csv/_phase_b_regime_gated_v2.csv", dtype={"tq": str}))[KEY + ["gated_v2"]]
b = b.merge(gated, on=KEY).merge(v2, on=KEY).dropna(subset=["dfm", "xgboost", "rf"])

def rmse(col, sub=None):
    d = b if sub is None else b[sub]
    t = pd.DataFrame({"model_name": col, "tq": d.tq, "vintage": d.vintage,
                      "week_idx": d.week_idx, "flash": d.flash, "y_pred": d[col]})
    s = H.score(t); return float(s.iloc[0]) if len(s) else np.nan

def apply_gate(pmap, gmap, col):
    p = b.tq.map(pmap); g = b.tq.map(gmap)
    ml = p * b.xgboost + (1 - p) * b.rf
    b[col] = b.dfm + 0.5 * (1 - g) * (ml - b.dfm)

# ---------- ① 규칙 신호 soft gate ----------
P1 = {q: p_rule(q) for q in targets}; G1 = {q: g_soft(q) for q in targets}
apply_gate(P1, G1, "soft_rule")
apply_gate(P1, {q: (1.0 if G1[q] > 0 else 0.0) for q in targets}, "soft_p_hard_g")  # p만 연속

# ---------- ② 오토인코더 이상탐지 점수 ----------
import torch, torch.nn as nn
drop = [c for c in x.columns if c.startswith("N_")]
panel = x.drop(columns=drop)
panel = panel.loc["2001-01-01":]
TRAIN_END = "2017-12-31"
mu, sd = panel.loc[:TRAIN_END].mean(), panel.loc[:TRAIN_END].std().replace(0, 1)
z = ((panel - mu) / sd).clip(-6, 6)
mask = z.notna().astype(np.float32).values
zv = z.fillna(0.0).astype(np.float32).values
n_feat = zv.shape[1]
tr = z.index <= pd.Timestamp(TRAIN_END)

def run_ae(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    ae = nn.Sequential(nn.Linear(n_feat, 24), nn.ReLU(), nn.Linear(24, 4),
                       nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, n_feat))
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    X = torch.tensor(zv[tr]); M = torch.tensor(mask[tr])
    for _ in range(400):
        opt.zero_grad()
        loss = (((ae(X) - X) ** 2) * M).sum() / M.sum()
        loss.backward(); opt.step()
    with torch.no_grad():
        E = ((ae(torch.tensor(zv)) - torch.tensor(zv)) ** 2 * torch.tensor(mask)).sum(1) / torch.tensor(mask).sum(1)
    return pd.Series(E.numpy(), index=z.index)

def p_ae(err, q, win=36):
    """분기 q 시작 직전 3개월 평균 오차의 z. 기준을 최근 win개월로 국한(국소 z) —
    전체 이력 기준이면 학습기간(~2017) 대비 이후 전체가 '이상'으로 보여 p가 1로 포화됨."""
    start = pd.Period(q, "Q").start_time
    hist = err[err.index < start]
    if len(hist) < win: return 1.0
    ref = hist.tail(win); cur = hist.tail(3).mean()
    med = ref.median(); mad = np.median(np.abs(ref - med)) or ref.std()
    return sig((cur - med) / mad)

ae_scores = []
for s in SEEDS:
    err = run_ae(s)
    P2 = {q: p_ae(err, q) for q in targets}
    apply_gate(P2, G1, f"soft_ae_{s}")
    apply_gate(P2, G1, f"tmp{s}")
    ae_scores.append(P2)

# ---------- 결과 ----------
REB = {'2018Q1','2019Q2','2020Q3','2023Q1','2024Q3','2025Q2'}
reb = b.tq.isin(REB)
print("=== 전체 32Q RMSE (flash w[-19,-1] 평균) ===")
print(f"  {'v1 (2-arm hard)':26s} {rmse('gated'):.4f}")
print(f"  {'v2 (3-arm hard)':26s} {rmse('gated_v2'):.4f}")
print(f"  {'① soft (p연속·g연속)':24s} {rmse('soft_rule'):.4f}")
print(f"  {'① soft (p연속·g계단)':24s} {rmse('soft_p_hard_g'):.4f}")
for s in SEEDS:
    print(f"  ② soft-AE seed{s}            {rmse(f'soft_ae_{s}'):.4f}")
print(f"\n=== 반등 6Q 한정 ===")
print(f"  v2 {rmse('gated_v2',reb):.4f} | soft_rule {rmse('soft_rule',reb):.4f} | " +
      " ".join(f"AE{s} {rmse(f'soft_ae_{s}',reb):.4f}" for s in SEEDS))
print("\n=== 신호 값 샘플 (p=충격가중, g=반등축소) ===")
hdr = ["q", "p_rule"] + [f"p_ae{s}" for s in SEEDS] + ["g"]
print(" | ".join(f"{h:>7s}" for h in hdr))
for q in ["2019Q1","2020Q1","2020Q2","2020Q3","2022Q4","2023Q1","2024Q3","2025Q1","2025Q2","2025Q3"]:
    row = [q, f"{P1[q]:.2f}"] + [f"{ae_scores[i][q]:.2f}" for i in range(len(SEEDS))] + [f"{G1[q]:.2f}"]
    print(" | ".join(f"{v:>7s}" for v in row))
