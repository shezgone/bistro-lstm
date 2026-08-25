"""Step 0 스모크 — Chronos-2 fit(finetune_mode="lora") 왕복 검증 (2026-08-26).

목적: 기계 검증만 — 빈티지 경로 소량으로 ①LoRA 적응 실행 ②저장 ③재로딩 ④예측이
왕복되는지, 소요 시간은 어느 정도인지. 성능 판정 아님 (num_steps 극소).

구성: 2022~2023 8개 분기의 첫 빈티지 스냅샷(공변량 10종+빠른신호 4종, C2f와 동일 채널)
을 분기말 월까지 잘라 학습 입력으로 사용. prediction_length=6 (h 최대치와 동일).

실행: .venv-gdp/bin/python phase_b_lora_smoke.py
"""
import os, sys, glob, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch
from chronos import Chronos2Pipeline
import phase_b_harness as H
import fast_signals as FS

COVARS = ["I_m", "I_s", "M_s", "S_es", "S_cb", "S_mo", "B_bx", "B_bi", "R_s", "M_fi"]
FASTC = ["kospi_mret", "krw_mret", "esi_raw", "esi_mom"]
TRAIN_QS = ["2022Q1", "2022Q2", "2022Q3", "2022Q4", "2023Q1", "2023Q2", "2023Q3", "2023Q4"]
OUT_DIR = "output/model/lora_smoke"
NUM_STEPS = int(os.environ.get("NUM_STEPS", "30"))

grid, _ = H.load_grid()
g = grid.copy(); g["vintage"] = pd.to_datetime(g["vintage"]).dt.strftime("%Y-%m-%d")

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

# 학습 입력: 분기말 월까지의 전체 경로 (fit이 내부에서 윈도우 샘플링)
inputs = []
for tq in TRAIN_QS:
    sub = g[g.tq == tq].sort_values("week_idx")
    if sub.empty: continue
    v0 = sub.vintage.iloc[-1]                       # 해당 분기 마지막 빈티지 (분기말 월 관측 포함)
    qend = pd.Period(tq, "Q").end_time.normalize().replace(day=1) + pd.offsets.MonthEnd(0)
    p = load_panel(tq, v0)
    if p is None: continue
    p = with_fast(p, v0)
    p = p[p.index <= qend]
    inputs.append({"target": p["N_gdp"].to_numpy(np.float32),
                   "past_covariates": {c: p[c].to_numpy(np.float32) for c in p.columns if c != "N_gdp"}})
print(f"[스모크] 학습 입력 {len(inputs)}개 (길이 {[len(x['target']) for x in inputs][:3]}...)", flush=True)

pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
n0 = sum(p_.numel() for p_ in pipe.inner_model.parameters())

t0 = time.time()
tuned = pipe.fit(inputs, prediction_length=6, finetune_mode="lora",
                 num_steps=NUM_STEPS, batch_size=4, output_dir=OUT_DIR)
t1 = time.time()
print(f"[1/4] fit(lora) 완료 — {NUM_STEPS} steps, {t1-t0:.0f}초", flush=True)

# 예측 왕복 (빈티지 컷 컨텍스트)
test_ctx = {"target": inputs[-1]["target"][:-6],
            "past_covariates": {k: v[:-6] for k, v in inputs[-1]["past_covariates"].items()}}
q1, _ = tuned.predict_quantiles([test_ctx], prediction_length=6, quantile_levels=[0.5])
print(f"[2/4] 적응 모델 예측 OK — 출력 형상 {np.asarray(q1[0]).shape}", flush=True)

save_path = f"{OUT_DIR}/smoke-ckpt"
tuned.save_pretrained(save_path)
print(f"[3/4] 저장 OK → {save_path}", flush=True)

reloaded = Chronos2Pipeline.from_pretrained(save_path, device_map="cpu")
q2, _ = reloaded.predict_quantiles([test_ctx], prediction_length=6, quantile_levels=[0.5])
a1, a2 = np.asarray(q1[0]).ravel(), np.asarray(q2[0]).ravel()
print(f"[4/4] 재로딩 예측 OK — 저장 전후 예측 일치: {np.allclose(a1, a2, atol=1e-4)} (최대차 {np.abs(a1-a2).max():.2e})", flush=True)

# 베이스와 실제로 달라졌는지 (LoRA가 적용됐는지)
qb, _ = pipe.predict_quantiles([test_ctx], prediction_length=6, quantile_levels=[0.5])
diff = np.abs(np.asarray(qb[0]).ravel() - a1).max()
print(f"[검증] 베이스 대비 예측 변화 최대 {diff:.4f} ({'적응 반영됨' if diff > 1e-6 else '변화 없음 — 확인 필요'})", flush=True)
print(f"[정보] 베이스 파라미터 {n0/1e6:.1f}M | 적응 소요 {t1-t0:.0f}초/{NUM_STEPS}steps → 1000 steps 환산 ~{(t1-t0)/NUM_STEPS*1000/60:.0f}분")
print("PHASE_B_LORA_SMOKE_DONE")
