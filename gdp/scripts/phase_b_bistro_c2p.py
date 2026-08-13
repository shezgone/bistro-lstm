"""BISTRO(어텐션 LSTM) — Chronos-2f와 '완전 동일 프로토콜' 재실험 (2026-08-13).

목적: 기존 이식판(phase_b_attnlstm)은 입력·프로토콜이 C2f와 달라 모델 비교가 오염됨.
본 스크립트는 phase_b_chronos2.py(FAST_COV=1)의 입력·과제·그리드를 그대로 두고
모델만 교체: 사전학습 Chronos-2 → 직접학습 어텐션 LSTM (BISTRO 백본 계열).

동일 조건:
  - 평가 그리드·채점 규약 동일 (H.load_grid, w[-19,-1] flash RMSE)
  - 입력 패널 동일: DFM 스냅샷(output/model/DFM/11/<tq>/<vintage>.csv) 빈티지 월까지
  - 채널 동일: N_gdp + 공변량 10종(I_m I_s M_s S_es S_cb S_mo B_bx B_bi R_s M_fi)
              + 빠른신호 4종(kospi_mret krw_mret esi_raw esi_mom — NSI 제외, C2f 기록과 동일)
  - 과제 동일: 분기말 월 N_gdp 외삽 (h=1..6), h<1이면 패널 실측값
차이 (모델 본질상 불가피):
  - C2f = 외부 사전학습·동결(zero-shot) / BISTRO = 분기별 확장창 직접학습
    (첫 빈티지 패널, cutoff Q-2 release-safe — TTM few-shot과 동일한 학습 규약)

env: QSUB=2024Q1 (부분) / CTX=36 PLEN=6 / EPOCHS=80 HID=48 / SEED=7
결과 (2026-08-13, 32분기 전체):
  단독 1.0191 (조기 1.2861 / 반등 0.5289) — 구 이식판 1.268(26Q) 대비 개선되나 여전히 격차.
  슬롯 교체: early=(GBM+LSTM)/2 → 0.7768 (기준 0.7499보다 악화 — 자격 미달)
  입력(빠른신호 포함)을 C2f와 동일하게 줘도 소표본 직접학습 한계 잔존.
  특이점: 반등 6Q 0.5289는 전 부품 최강 — 단 6분기 소표본이라 신뢰 유보(사후 확인용 기록).

실행: .venv-gdp/bin/python phase_b_bistro_c2p.py
"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch
import torch.nn as nn
import phase_b_harness as H
import fast_signals as FS

COVARS = ["I_m", "I_s", "M_s", "S_es", "S_cb", "S_mo", "B_bx", "B_bi", "R_s", "M_fi"]
FASTC = ["kospi_mret", "krw_mret", "esi_raw", "esi_mom"]
CTX = int(os.environ.get("CTX", "36")); PLEN = 6
EPOCHS = int(os.environ.get("EPOCHS", "80")); HID = int(os.environ.get("HID", "48"))
SEED = int(os.environ.get("SEED", "7"))
TAG = "our_bistro_c2p"

grid, _ = H.load_grid()
g = grid.copy(); g["vintage"] = pd.to_datetime(g["vintage"]).dt.strftime("%Y-%m-%d")
QSUB = os.environ.get("QSUB")
quarters = QSUB.split(",") if QSUB else sorted(g.tq.unique(), key=lambda x: pd.Period(x, "Q"))


class AttnLSTM(nn.Module):
    """BISTRO 백본 계열: LSTM 인코더 + 시점 어텐션 풀링 + 다단계 헤드."""
    def __init__(self, c_in, hid=HID, plen=PLEN):
        super().__init__()
        self.lstm = nn.LSTM(c_in, hid, num_layers=1, batch_first=True)
        self.attn = nn.Linear(hid, 1)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, plen))
    def forward(self, x):                    # x: (B, T, C)
        h, _ = self.lstm(x)                  # (B, T, H)
        w = torch.softmax(self.attn(h).squeeze(-1), dim=1)   # (B, T)
        z = (h * w.unsqueeze(-1)).sum(1)     # (B, H)
        return self.head(z)                  # (B, PLEN) — N_gdp 경로


def load_panel(tq, vintage):
    files = sorted(glob.glob(f"output/model/DFM/11/{tq}/*.csv"))
    cands = [f for f in files if os.path.basename(f)[:-4] <= vintage]
    if not cands: return None
    df = pd.read_csv(cands[-1])
    dcol = df.columns[0]; df[dcol] = pd.to_datetime(df[dcol])
    df = df.rename(columns={dcol: "date"}).set_index("date")
    cols = ["N_gdp"] + [c for c in COVARS if c in df.columns]
    return df[cols].astype(np.float32)


def with_fast(panel, vintage):
    """C2f와 동일: 빠른신호 4종을 빈티지 기준으로 채널 추가 (NSI 제외)."""
    fs = FS.monthly_covariates(panel.index, vintage)
    for c in FASTC:
        panel = panel.copy(); panel[c] = fs[c].values.astype(np.float32)
    return panel


def train_quarter(tq, first_vintage):
    p0 = load_panel(tq, first_vintage)
    if p0 is None: return None, None, None
    p0 = with_fast(p0, first_vintage)
    cutoff = pd.Period(tq, "Q").to_timestamp() - pd.offsets.QuarterEnd(2)
    tr = p0[p0.index <= cutoff]
    if len(tr) < CTX + PLEN + 20: return None, None, None
    mu, sd = tr.mean(), tr.std() + 1e-6
    Z = ((tr - mu) / sd).values.astype(np.float32)
    xs, ys = [], []
    for e in range(CTX, len(Z) - PLEN):
        xs.append(Z[e - CTX:e]); ys.append(Z[e:e + PLEN, 0])   # 타깃 = N_gdp(표준화)
    X = torch.tensor(np.stack(xs)); Y = torch.tensor(np.stack(ys))
    torch.manual_seed(SEED); np.random.seed(SEED)
    mdl = AttnLSTM(Z.shape[1])
    opt = torch.optim.Adam(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
    mdl.train()
    for ep in range(EPOCHS):
        idx = torch.randperm(len(X))[:64]
        opt.zero_grad()
        loss = nn.functional.mse_loss(mdl(X[idx]), Y[idx])
        loss.backward(); opt.step()
    mdl.eval(); return mdl, mu, sd


rows = []
for tq in quarters:
    sub = g[g.tq == tq].drop_duplicates(["vintage", "week_idx"]).sort_values("week_idx")
    if sub.empty: continue
    qend = pd.Period(tq, "Q").end_time.normalize().replace(day=1) + pd.offsets.MonthEnd(0)
    mdl, mu, sd = train_quarter(tq, sub.vintage.iloc[0])
    for r in sub.itertuples(index=False):
        yp = np.nan
        if mdl is not None:
            panel = load_panel(tq, r.vintage)
            if panel is not None:
                panel = with_fast(panel, r.vintage)
                edge = pd.Timestamp(r.vintage) + pd.offsets.MonthEnd(0)
                hist = panel[panel.index <= edge]
                h = int((qend.to_period("M") - hist.index[-1].to_period("M")).n) if len(hist) else 99
                if h < 1:
                    yp = float(panel.loc[qend, "N_gdp"]) if qend in panel.index else np.nan
                elif h <= PLEN and len(hist) >= CTX:
                    Zh = ((hist - mu) / sd).values.astype(np.float32)[-CTX:]
                    with torch.no_grad():
                        out = mdl(torch.tensor(Zh[None]))
                    yp = float(out[0, h - 1]) * float(sd.iloc[0]) + float(mu.iloc[0])
        rows.append({"tq": tq, "vintage": r.vintage, "week_idx": r.week_idx,
                     "flash": r.flash, "model_name": TAG, "y_pred": yp})
    print(f"[{tq}] done", flush=True)

pred = pd.DataFrame(rows)
pred.to_csv(f"output/csv/_phase_b_{TAG}_predictions.csv", index=False)
print(f"\n=== BISTRO(동일 프로토콜) flash w[-19,-1] avg RMSE ===")
print(H.score(pred).to_string())
print("[벤치] Chronos-2f 0.8075 / XGB(신) 0.750 / 구 이식판 1.268(26Q)")
print("PHASE_B_BISTRO_C2P_DONE")
