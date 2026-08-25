"""실험 A: Chronos-2 LoRA 적응 — 빈티지 경로 학습, walk-forward 평가 (2026-08-26).

플랜: docs/GDP_LoRA적응_실험플랜_2026-08-26.md (bistro-lstm). 사전 등록 설정(v1):
  finetune_mode="lora"(기본 lora_config), lr=1e-4, num_steps=500, batch_size=16, seed 고정
  연 1회 fresh 재적응 (2021~2025 = 5회), 학습분기 = 해당 연도 첫 빈티지 이전 flash 발표 완료 분기

학습 데이터(빈티지 경로 v1):
  분기 q당 시리즈 1개 = q의 마지막 주간 빈티지(w=-1) 스냅샷을 분기말 월까지 절단
  + 전 이력의 분기말 월 N_gdp를 실제 flash로 앵커링(라벨 월만 진실값, DFM 모사 최소화)
  + 채널 = C2f와 동일(N_gdp + 공변량 10종 + 빠른신호 4종) + est_flag(발표시차 근사 추정구간 마스크)
  ※ est_flag는 예측 시에도 동일 규칙으로 부여 — 학습/예측 분포 일치

평가: phase_b_chronos2.py와 동일 프로토콜/그리드, TAG=our_c2f_lora.
env: YEARS=2024 (부분) / NUM_STEPS / SEED / TAG
실행: .venv-gdp/bin/python phase_b_c2_lora.py
"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch
from chronos import Chronos2Pipeline
import phase_b_harness as H
import fast_signals as FS

COVARS = ["I_m", "I_s", "M_s", "S_es", "S_cb", "S_mo", "B_bx", "B_bi", "R_s", "M_fi"]
FASTC = ["kospi_mret", "krw_mret", "esi_raw", "esi_mom"]
PLEN = 6
NUM_STEPS = int(os.environ.get("NUM_STEPS", "500"))
LR = float(os.environ.get("LR", "1e-4"))
SEED = int(os.environ.get("SEED", "7"))
TAG = os.environ.get("TAG", "our_c2f_lora")
YEARS = [int(y) for y in os.environ.get("YEARS", "2021,2022,2023,2024,2025").split(",")]

grid, _ = H.load_grid()
g = grid.copy(); g["vintage"] = pd.to_datetime(g["vintage"]).dt.strftime("%Y-%m-%d")
FLASH = g[["tq", "flash"]].drop_duplicates().set_index("tq")["flash"].to_dict()
ALL_Q = sorted(g.tq.unique(), key=lambda x: pd.Period(x, "Q"))

def load_panel(tq, vintage):
    files = sorted(glob.glob(f"output/model/DFM/11/{tq}/*.csv"))
    cands = [f for f in files if os.path.basename(f)[:-4] <= vintage]
    if not cands: return None
    df = pd.read_csv(cands[-1])
    dcol = df.columns[0]; df[dcol] = pd.to_datetime(df[dcol])
    df = df.rename(columns={dcol: "date"}).set_index("date")
    return df[["N_gdp"] + [c for c in COVARS if c in df.columns]].astype(np.float32)

def with_fast_and_mask(panel, vintage):
    fs = FS.monthly_covariates(panel.index, vintage)
    panel = panel.copy()
    for c in FASTC:
        panel[c] = fs[c].values.astype(np.float32)
    vm = (pd.Timestamp(vintage) + pd.offsets.MonthEnd(0))
    est_cut = vm - pd.offsets.MonthEnd(1)          # 발표시차 1개월 근사: 그 이후 월은 DFM 추정 구간
    panel["est_flag"] = (panel.index > est_cut).astype(np.float32)
    return panel

def qend_of(tq):
    return pd.Period(tq, "Q").end_time.normalize().replace(day=1) + pd.offsets.MonthEnd(0)

def anchor_flashes(panel, upto_tq):
    """전 이력 분기말 월의 N_gdp를 실제 flash로 교체 (발표 완료 분기만)."""
    panel = panel.copy()
    for q in ALL_Q:
        if pd.Period(q, "Q") > pd.Period(upto_tq, "Q"): break
        qe = qend_of(q)
        if qe in panel.index and q in FLASH and pd.notna(FLASH[q]):
            panel.loc[qe, "N_gdp"] = np.float32(FLASH[q])
    return panel

def build_training_inputs(first_vintage_of_year):
    """해당 시점 이전에 flash가 발표 완료된 분기들의 경로 시리즈."""
    inputs = []
    fv = pd.Timestamp(first_vintage_of_year)
    for q in ALL_Q:
        # flash 발표 ≈ 분기말 +4주. 안전하게 +6주 이후에만 학습 사용 (release-safe)
        if qend_of(q) + pd.Timedelta(weeks=6) >= fv: continue
        sub = g[g.tq == q].sort_values("week_idx")
        if sub.empty: continue
        v_last = sub.vintage.iloc[-1]
        p = load_panel(q, v_last)
        if p is None: continue
        p = with_fast_and_mask(p, v_last)
        p = p[p.index <= qend_of(q)]
        p = anchor_flashes(p, q)
        if len(p) < 80: continue
        inputs.append({"target": p["N_gdp"].to_numpy(np.float32),
                       "past_covariates": {c: p[c].to_numpy(np.float32) for c in p.columns if c != "N_gdp"}})
    return inputs

def predict_year(pipe, year, rows):
    quarters = [q for q in ALL_Q if q.startswith(str(year))]
    for tq in quarters:
        sub = g[g.tq == tq].drop_duplicates(["vintage", "week_idx"]).sort_values("week_idx")
        qe = qend_of(tq)
        for r in sub.itertuples(index=False):
            yp = np.nan
            panel = load_panel(tq, r.vintage)
            if panel is not None:
                panel = with_fast_and_mask(panel, r.vintage)
                edge = pd.Timestamp(r.vintage) + pd.offsets.MonthEnd(0)
                hist = panel[panel.index <= edge]
                hist = anchor_flashes(hist, tq)          # 과거 발표분 앵커 (해당 분기 flash는 미발표라 미포함)
                h = int((qe.to_period("M") - hist.index[-1].to_period("M")).n) if len(hist) else 99
                if h < 1:
                    yp = float(panel.loc[qe, "N_gdp"]) if qe in panel.index else np.nan
                elif h <= PLEN and len(hist) >= 60:
                    try:
                        ctx = {"target": hist["N_gdp"].to_numpy(np.float32),
                               "past_covariates": {c: hist[c].to_numpy(np.float32) for c in hist.columns if c != "N_gdp"}}
                        qt, _ = pipe.predict_quantiles([ctx], prediction_length=h, quantile_levels=[0.5])
                        arr = np.asarray(qt[0]);  arr = arr[0] if arr.ndim == 3 else arr
                        yp = float(arr[h - 1, 0])
                    except Exception as e:
                        print(f"  predict err {tq} {r.vintage}: {type(e).__name__}: {e}", flush=True)
            rows.append({"tq": tq, "vintage": r.vintage, "week_idx": r.week_idx,
                         "flash": r.flash, "model_name": TAG, "y_pred": yp})
        print(f"[{tq}] done", flush=True)

rows = []
for year in YEARS:
    q1 = f"{year}Q1"
    sub = g[g.tq == q1].sort_values("week_idx")
    fv = sub.vintage.iloc[0]
    torch.manual_seed(SEED); np.random.seed(SEED)
    inputs = build_training_inputs(fv)
    print(f"\n=== {year}: 적응 시작 (학습 분기 {len(inputs)}개, 기준 빈티지 {fv}) ===", flush=True)
    base = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    tuned = base.fit(inputs, prediction_length=PLEN, finetune_mode="lora",
                     learning_rate=LR, num_steps=NUM_STEPS, batch_size=16,
                     output_dir=f"output/model/c2_lora/{TAG}_{year}")
    tuned.inner_model.eval()                          # Step 0 교훈: 드롭아웃 차단
    predict_year(tuned, year, rows)
    del base, tuned

pred = pd.DataFrame(rows)
pred.to_csv(f"output/csv/_phase_b_{TAG}_predictions.csv", index=False)
print(f"\n=== {TAG} flash w[-19,-1] avg RMSE (평가 연도: {YEARS}) ===")
print(H.score(pred).to_string())
print("PHASE_B_C2_LORA_DONE")
