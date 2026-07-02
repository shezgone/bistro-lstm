"""Phase B — 우리 AttentionLSTMForecaster를 DFM 보정 월별 패널에 이식.
각 (target quarter q, vintage v): (q,v) DFM CSV의 월별 패널(34변수) 사용.
- 학습표본: 과거 분기 q'<q (flash 존재), 입력=q'말 직전 L개월 윈도우, 타깃=flash_q' (vintage-safe: v 시점 패널만 사용)
- 예측: q말 직전 L개월 윈도우 → flash_q
소표본이므로 소형 모델 + seed 앙상블.
"""
import sys, os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/user_1/vibe/bistro-lstm")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lstm_model import AttentionLSTMForecaster
import phase_b_harness as H

SMOKE = os.environ.get("SMOKE", "1") == "1"
L = 24            # 입력 월 수
SEEDS = 3
EPOCHS = 120
DFM_ROOT = H.DFM_ROOT

FEATCOLS = None  # 첫 CSV에서 결정 (Time, N_gdp 제외한 34→33? N_gdp 포함 여부 결정)

def q_last_month(tq):
    p = pd.Period(tq, freq="Q")
    return p.asfreq("M", "end").to_timestamp().normalize()

def load_panel(tq, vintage):
    csv = DFM_ROOT / tq / f"{vintage}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df["Time"] = pd.to_datetime(df["Time"]).dt.normalize()
    return df.set_index("Time").sort_index()

def make_window(panel, end_month, cols):
    """end_month에서 끝나는 L개월 윈도우 (features only)."""
    sub = panel.loc[:end_month]
    if len(sub) < L:
        return None
    return sub.iloc[-L:][cols].to_numpy(dtype=np.float32)

def train_predict(q, v, flash_by_q):
    panel = load_panel(q, v)
    if panel is None:
        return None
    global FEATCOLS
    if FEATCOLS is None:
        FEATCOLS = [c for c in panel.columns if c != "N_gdp"]  # 타깃 누수 방지: N_gdp 제외
    # 학습표본: 과거 분기
    Xtr, ytr = [], []
    past = [qq for qq in flash_by_q if pd.Period(qq, "Q") < pd.Period(q, "Q")]
    for qq in past:
        w = make_window(panel, q_last_month(qq), FEATCOLS)
        if w is not None and np.isfinite(w).all():
            Xtr.append(w); ytr.append(flash_by_q[qq])
    if len(Xtr) < 6:
        return None
    Xw = make_window(panel, q_last_month(q), FEATCOLS)
    if Xw is None or not np.isfinite(Xw).all():
        return None
    Xtr = np.stack(Xtr); ytr = np.array(ytr, np.float32)
    # 표준화 (학습통계)
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0); sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = ((Xw - mu) / sd)[None]
    ymu, ysd = ytr.mean(), ytr.std() + 1e-6
    ytr_n = (ytr - ymu) / ysd
    preds = []
    for s in range(SEEDS):
        torch.manual_seed(s); np.random.seed(s)
        m = AttentionLSTMForecaster(n_vars=len(FEATCOLS), d_model=16, hidden_dim=32,
                                    n_layers=1, n_heads=2, pred_len=1, dropout=0.3)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-3)
        xt = torch.tensor(Xtr); yt = torch.tensor(ytr_n).view(-1, 1)
        m.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            out = m(xt)["mu"]
            loss = nn.functional.mse_loss(out, yt)
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = m(torch.tensor(Xte))["mu"].item()
        preds.append(p * ysd + ymu)
    return float(np.mean(preds))


def main():
    grid, _ = H.load_grid()
    flash_by_q = {r.tq: r.flash for r in grid[["tq", "flash"]].drop_duplicates().itertuples(index=False)
                  if pd.notna(r.flash)}
    if SMOKE:
        tgt = ["2024Q1", "2024Q2", "2024Q3"]
        grid = grid[grid.tq.isin(tgt)]
        print(f"[SMOKE] target quarters={tgt}, grid rows={len(grid)}")
    rows = []
    for i, r in enumerate(grid.itertuples(index=False)):
        yp = train_predict(r.tq, r.vintage, flash_by_q)
        rows.append({"tq": r.tq, "vintage": r.vintage, "week_idx": r.week_idx,
                     "flash": r.flash, "model_name": "our_attnlstm", "y_pred": yp})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(grid)} done", flush=True)
    pred = pd.DataFrame(rows)
    out = "output/csv/_phase_b_attnlstm_smoke.csv" if SMOKE else "output/csv/_phase_b_attnlstm.csv"
    pred.to_csv(out, index=False)
    n_ok = pred["y_pred"].notna().sum()
    print(f"\n예측 완료: {n_ok}/{len(pred)} valid -> {out}")
    if n_ok:
        print("\n=== our_attnlstm 단독 (smoke: 부분 그리드라 참고용) ===")
        print(H.score(pred).to_string())
    print("PHASE_B_TF_DONE")

if __name__ == "__main__":
    main()
