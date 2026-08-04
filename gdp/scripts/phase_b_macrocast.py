"""E3: MACROCAST 레시피 (arXiv 2606.28670 변형) — 자기 DFM 합성 코퍼스로 TTM 미세조정.

아이디어: 실데이터 140행으로 미세조정하면 과적합(기존 few-shot 0.854가 상한).
학습창 패널에 경량 요인모형(표준화→PCA k요인→VAR(1)+AR(1) 개별성분)을 적합해
같은 상관·지속성 구조의 합성 패널을 대량 생성 → TTM을 합성으로 먼저 미세조정
→ 실데이터로 어닐링(마무리 소량 학습; 반대증거 2605.06032 "증강 유해" 대응).

프로토콜: phase_b_ttm.py --finetune과 동일 (분기별 확장창, cutoff Q-2 release-safe,
CTX=90, PLEN=6, 10채널, 분기말 월 N_gdp 외삽). 다른 것은 미세조정 데이터뿐.

env: QSUB=2020Q3,2023Q1 (부분 실행) / N_SYN=20 (합성 패널 수) / SYN_LEN=300
     EP_SYN=30 EP_REAL=10 (어닐링 스케줄) / SEED=7
실행: .venv-gdp/bin/python phase_b_macrocast.py

결과 (2026-08-05, 32분기 전체) — 기각 (무개선):
  MACROCAST 0.8531 (반등 0.6103) vs TTM few-shot 0.8542 (반등 0.6411) — 차이 0.001
  예측 상관 0.990, 평균절대차 0.045 → 합성 사전 미세조정이 도달점을 거의 안 바꿈.
  C2f 0.8075(반등 0.5813)에 미달 — H2 조기 주차 슬롯 교체 불가.
해석: PCA 3요인+VAR 합성은 실패널이 이미 가진 정보의 재표집일 뿐 — TTM의 병목은
  학습 표본량이 아니라 패널의 정보 상한. 원논문의 상태공간·빈티지 일관 합성이면
  다를 수 있으나(라이트 구현 한계), 상관 0.990은 어닐링 단계가 도달점을 지배함을
  시사 → 변형 반복의 기대값 낮음. 점추정 축 마지막 카드 소진.
"""
import os, sys, glob, copy, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch
from tsfm_public.toolkit.get_model import get_model
import phase_b_harness as H

CTX = 90; PLEN = 6
CHANNELS = ["N_gdp", "new_coin", "I_m", "M_p", "I_s", "R_s",
            "B_gx", "new_esi", "S_cb", "M_ic"]
N_SYN = int(os.environ.get("N_SYN", "20"))
SYN_LEN = int(os.environ.get("SYN_LEN", "300"))
EP_SYN = int(os.environ.get("EP_SYN", "30"))
EP_REAL = int(os.environ.get("EP_REAL", "10"))
K_FACTOR = 3
SEED = int(os.environ.get("SEED", "7"))

grid, _ = H.load_grid()
g = grid.copy(); g["vintage"] = pd.to_datetime(g["vintage"]).dt.strftime("%Y-%m-%d")
QSUB = os.environ.get("QSUB")
quarters = QSUB.split(",") if QSUB else sorted(g.tq.unique(), key=lambda x: pd.Period(x, "Q"))

model = get_model("ibm-granite/granite-timeseries-ttm-r2", context_length=CTX, prediction_length=PLEN)
model.eval()
print(f"[E3 MACROCAST] ctx={CTX} plen={PLEN} n_syn={N_SYN}x{SYN_LEN} ep={EP_SYN}+{EP_REAL}", flush=True)

def load_panel(tq, vintage):
    files = sorted(glob.glob(f"output/model/DFM/11/{tq}/*.csv"))
    cands = [f for f in files if os.path.basename(f)[:-4] <= vintage]
    if not cands: return None
    df = pd.read_csv(cands[-1])
    dcol = df.columns[0]; df[dcol] = pd.to_datetime(df[dcol])
    df = df.rename(columns={dcol: "date"}).set_index("date")
    return df[[c for c in CHANNELS if c in df.columns]].astype(np.float32)

def synth_corpus(train, rng):
    """학습창 패널 → 요인모형 적합 → 합성 패널 N_SYN개."""
    V = train.values
    mu, sd = V.mean(0), V.std(0) + 1e-6
    Z = (V - mu) / sd
    # PCA 요인 + 적재
    U, S, Wt = np.linalg.svd(Z, full_matrices=False)
    F = U[:, :K_FACTOR] * S[:K_FACTOR]          # (T, k)
    L = Wt[:K_FACTOR].T                          # (C, k)
    # VAR(1) on factors (spectral radius 클리핑)
    A = np.linalg.lstsq(F[:-1], F[1:], rcond=None)[0].T
    ev = np.max(np.abs(np.linalg.eigvals(A)))
    if ev > 0.98: A *= 0.98 / ev
    Ef = F[1:] - F[:-1] @ A.T; ef_std = Ef.std(0) + 1e-6
    # 개별성분 AR(1)
    Eid = Z - F @ L.T
    phi = np.clip(np.array([np.corrcoef(Eid[1:, c], Eid[:-1, c])[0, 1] if Eid[:, c].std() > 1e-8 else 0.0
                            for c in range(Z.shape[1])]), -0.95, 0.95)
    eid_std = (Eid[1:] - Eid[:-1] * phi).std(0) + 1e-6
    panels = []
    for _ in range(N_SYN):
        f = np.zeros((SYN_LEN, K_FACTOR), np.float32); u = np.zeros((SYN_LEN, Z.shape[1]), np.float32)
        for t in range(1, SYN_LEN):
            f[t] = f[t - 1] @ A.T + rng.normal(0, ef_std)
            u[t] = u[t - 1] * phi + rng.normal(0, eid_std)
        panels.append(((f @ L.T + u) * sd + mu).astype(np.float32))
    return panels

def windows(V):
    xs, ys = [], []
    for e in range(CTX, len(V) - PLEN):
        xs.append(V[e - CTX:e]); ys.append(V[e:e + PLEN])
    return xs, ys

def finetune_macrocast(train, rng):
    mdl = copy.deepcopy(model); mdl.train()
    opt = torch.optim.Adam([p for p in mdl.parameters() if p.requires_grad], lr=1e-4)
    sx, sy = [], []
    for P in synth_corpus(train, rng):
        a, b_ = windows(P); sx += a; sy += b_
    rx, ry = windows(train.values)
    if len(rx) < 20 or len(sx) < 100: return model
    SX, SY = torch.tensor(np.stack(sx)), torch.tensor(np.stack(sy))
    RX, RY = torch.tensor(np.stack(rx)), torch.tensor(np.stack(ry))
    for X, Y, eps in [(SX, SY, EP_SYN), (RX, RY, EP_REAL)]:   # 합성 → 실데이터 어닐링
        for ep in range(eps):
            idx = torch.randperm(len(X))[:64]
            opt.zero_grad()
            out = mdl(past_values=X[idx], future_values=Y[idx],
                      freq_token=torch.zeros(len(idx), dtype=torch.long))
            out.loss.backward(); opt.step()
    mdl.eval(); return mdl

def forecast_qend(panel, vintage, qend, mdl):
    edge = pd.Timestamp(vintage) + pd.offsets.MonthEnd(0)
    hist = panel[panel.index <= edge].tail(CTX)
    if len(hist) < CTX: return np.nan
    h = int((qend.to_period("M") - hist.index[-1].to_period("M")).n)
    if h < 1:
        return float(panel.loc[qend, "N_gdp"]) if qend in panel.index else np.nan
    if h > PLEN: return np.nan
    x = torch.tensor(hist.values[None])
    with torch.no_grad():
        out = mdl(past_values=x, freq_token=torch.zeros(1, dtype=torch.long))
    return float(out.prediction_outputs[0, h - 1, 0])

rows = []
for tq in quarters:
    sub = g[g.tq == tq].drop_duplicates(["vintage", "week_idx"]).sort_values("week_idx")
    if sub.empty: continue
    qend = pd.Period(tq, "Q").end_time.normalize().replace(day=1) + pd.offsets.MonthEnd(0)
    rng = np.random.default_rng(SEED + hash(tq) % 10000)
    torch.manual_seed(SEED)
    p0 = load_panel(tq, sub.vintage.iloc[0])
    cutoff = pd.Period(tq, "Q").to_timestamp() - pd.offsets.QuarterEnd(2)
    mdl = model if p0 is None else finetune_macrocast(p0[p0.index <= cutoff], rng)
    for r in sub.itertuples(index=False):
        panel = load_panel(tq, r.vintage)
        yp = np.nan if panel is None else forecast_qend(panel, r.vintage, qend, mdl)
        rows.append({"tq": tq, "vintage": r.vintage, "week_idx": r.week_idx,
                     "flash": r.flash, "model_name": "our_macrocast", "y_pred": yp})
    print(f"[{tq}] done", flush=True)

pred = pd.DataFrame(rows)
pred.to_csv("output/csv/_phase_b_macrocast_predictions.csv", index=False)
print("\n=== E3 MACROCAST flash w[-19,-1] avg RMSE ===")
print(H.score(pred).to_string())
print("[벤치] TTM few-shot 0.854 / C2f 0.8075 / XGB(신) 0.750")
print("PHASE_B_MACROCAST_DONE")
