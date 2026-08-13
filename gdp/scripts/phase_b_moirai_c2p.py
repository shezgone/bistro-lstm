"""Moirai(BISTRO FM 라인업) — Chronos-2f와 '완전 동일 프로토콜' zero-shot (2026-08-13).

배경: BISTRO 프레임워크의 FM 암 = Chronos·Moirai (온보딩 문서 기준).
기존 our_moirai(1.45)는 구세대 프로토콜(분기 flash 단변량 외삽)이라 비교 오염.
본 스크립트는 phase_b_chronos2.py(FAST_COV=1)와 입력·과제·그리드 동일, 모델만 교체:
  - 패널: DFM 스냅샷, 빈티지 월까지 / 타깃: 월별 N_gdp, 분기말 외삽 h=1..6
  - past 공변량: 10종 + 빠른신호 4종(NSI 제외) — past_feat_dynamic_real로 주입
  - 모델: Salesforce/moirai-1.0-R-small (BISTRO 라인업과 동일 체크포인트), zero-shot

env: QSUB=2024Q1 / CTX_MAX=200 / NSAMP=100
결과 (2026-08-13, 32분기 전체):
  단독 0.8731 (조기 1.0206 / 반등 0.7200) — 구세대 프로토콜 1.4533 대비 -40% (프로토콜 지배 확인)
  vs Chronos-2f 0.8075 (조기 0.9238 / 반등 0.5813) — 같은 zero-shot FM인데 격차
  슬롯 교체: early=(GBM+Moirai)/2 → 0.7536 (기준 XGB 0.7499보다 악화 — 자격 미달)
  FM 듀오 (C2f+Moirai)/2 슬롯: 0.7454 — C2f 단독 슬롯(0.7399) 미달 (오차상관 0.926 중복)
해석: 슬롯 자격 요건 = "조기 구간 GBM 대등(0.924)". Moirai-1.0-small은 1.021로 미달 —
  세대·공변량 처리 차이(Chronos-2는 공변량 네이티브)로 해석. "사전학습이면 된다" 아님.

실행: /Users/user/vibe/bistro-lstm/.venv-moirai/bin/python phase_b_moirai_c2p.py
"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from gluonts.dataset.common import ListDataset
import phase_b_harness as H
import fast_signals as FS

COVARS = ["I_m", "I_s", "M_s", "S_es", "S_cb", "S_mo", "B_bx", "B_bi", "R_s", "M_fi"]
FASTC = ["kospi_mret", "krw_mret", "esi_raw", "esi_mom"]
PLEN_MAX = 6
CTX_MAX = int(os.environ.get("CTX_MAX", "200"))
NSAMP = int(os.environ.get("NSAMP", "100"))
TAG = "our_moirai_c2p"

grid, _ = H.load_grid()
g = grid.copy(); g["vintage"] = pd.to_datetime(g["vintage"]).dt.strftime("%Y-%m-%d")
QSUB = os.environ.get("QSUB")
quarters = QSUB.split(",") if QSUB else sorted(g.tq.unique(), key=lambda x: pd.Period(x, "Q"))

module = MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small")
print("[moirai-1.0-R-small] loaded", flush=True)

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
    fs = FS.monthly_covariates(panel.index, vintage)
    panel = panel.copy()
    for c in FASTC:
        panel[c] = fs[c].values.astype(np.float32)
    return panel

def moirai_predict(hist, h):
    """hist: (T, 1+C) DataFrame [N_gdp + 공변량들], h: 예측 지평(월)."""
    hist = hist.tail(CTX_MAX)
    T = len(hist)
    covcols = [c for c in hist.columns if c != "N_gdp"]
    model = MoiraiForecast(module=module, prediction_length=h, context_length=T,
                           patch_size="auto", num_samples=NSAMP, target_dim=1,
                           feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=len(covcols))
    pred = model.create_predictor(batch_size=1, device="cpu")
    start = pd.Period(hist.index[0], freq="M")
    ds = ListDataset([{"target": hist["N_gdp"].values.astype(np.float32),
                       "start": start,
                       "past_feat_dynamic_real": hist[covcols].values.T.astype(np.float32)}],
                     freq="M", one_dim_target=True)
    fc = list(pred.predict(ds))[0]
    return float(np.median(fc.samples[:, h - 1]))

rows = []
for tq in quarters:
    sub = g[g.tq == tq].drop_duplicates(["vintage", "week_idx"]).sort_values("week_idx")
    if sub.empty: continue
    qend = pd.Period(tq, "Q").end_time.normalize().replace(day=1) + pd.offsets.MonthEnd(0)
    for r in sub.itertuples(index=False):
        yp = np.nan
        panel = load_panel(tq, r.vintage)
        if panel is not None:
            panel = with_fast(panel, r.vintage)
            edge = pd.Timestamp(r.vintage) + pd.offsets.MonthEnd(0)
            hist = panel[panel.index <= edge]
            h = int((qend.to_period("M") - hist.index[-1].to_period("M")).n) if len(hist) else 99
            if h < 1:
                yp = float(panel.loc[qend, "N_gdp"]) if qend in panel.index else np.nan
            elif h <= PLEN_MAX and len(hist) >= 60:
                try:
                    yp = moirai_predict(hist, h)
                except Exception as e:
                    print(f"  predict err {tq} {r.vintage}: {type(e).__name__}: {e}", flush=True)
        rows.append({"tq": tq, "vintage": r.vintage, "week_idx": r.week_idx,
                     "flash": r.flash, "model_name": TAG, "y_pred": yp})
    print(f"[{tq}] done", flush=True)

pred = pd.DataFrame(rows)
pred.to_csv(f"output/csv/_phase_b_{TAG}_predictions.csv", index=False)
print("\n=== Moirai(동일 프로토콜) flash w[-19,-1] avg RMSE ===")
print(H.score(pred).to_string())
print("[벤치] Chronos-2f 0.8075 / 구 Moirai(분기 단변량) 1.4533 / XGB(신) 0.750")
print("PHASE_B_MOIRAI_C2P_DONE")
