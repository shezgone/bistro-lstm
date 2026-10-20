# -*- coding: utf-8 -*-
"""Timestral(Mistral AI, 2026-10-15 출시) — Chronos-2f·Moirai와 '완전 동일 프로토콜' zero-shot (2026-10-20).

배경: GIFT-Eval 사전학습 zero-shot 1위(전체 8위) 신작. 동일 규약 실측으로
  ① 조기 슬롯 자격(조기 구간 GBM 0.924 대등 여부) ② 흡수율 ③ 절제(이웃 vs 힌트 스펙트럼 제3 설계점) 확인.
프로토콜 = phase_b_moirai_c2p.py와 입력·과제·그리드 동일, 모델 호출부만 교체:
  - 패널: DFM 스냅샷(빈티지 월까지) / 타깃: 월별 N_gdp 단변량, 분기말 외삽 h=1..6
  - past 공변량: 10종 + 빠른신호 4종(NSI 제외) — past_covariates (1,C,T)로 주입
  - 출력: 21분위수 중앙값 (q10/q90도 저장 — 독립 테스트에서 분위수 보정 불량 보고 있어 커버리지 확인용)

⚠️ 캐비앗: 학습 데이터 컷오프 미공개(기술 리포트 미발간) — 평가창 2018~2025 전체가
  사전학습에 포함됐을 수 있음. 2025 준청정 홀드아웃도 이 모델엔 무효. 레벨 RMSE는
  "오염 미상" 라벨로만 보고, 논문에는 행동 진단(흡수율·절제)만 사용 가능.

env: MODEL=mistralai/Timestral-209M | COVMODE=full/base/uni | CTX_MAX=200 | QSUB=2024Q1 | TAG
실행: /Users/user/vibe/bistro-lstm/.venv-timestral/bin/python phase_b_timestral_c2p.py

결과 (2026-10-20, 32분기 전체 — 레벨 수치는 전부 '오염 미상' 라벨):
  단독: 209M full 0.8344 / 28M full 0.8361 — C2f 0.8075 미달, Moirai-small 0.8731 상회
       (벤더 "28M이 Chronos-2 이김" 주장은 본 과제에서 불성립 — GIFT-Eval 순위와 과제 특성이 다름)
  절제(209M): uni 0.8930 → base(+10종) 0.8607(-3.6%, Moirai는 무이득이었음) → full(+빠른신호) 0.8344(-3.1%, C2 -3.4%급 '힌트형')
  조기(w<=-14): 28M 0.8754 < 209M 0.8948 < GBM 0.924 ≈ C2f 0.9238 — 조기 구간 신기록
  흡수율(조기→중반 RMSE 감소): 209M -1.3% / 28M +4.4% 역행 — BISTRO(-4.7%)보다도 평탄.
    절제는 힌트형인데 주차 흡수는 꼴찌 → 두 진단의 해리: 공변량을 레벨에는 쓰지만 분기 내 정보 흐름은 추적 못함.
    (조기 저오차 + 평탄 흡수 패턴은 사전학습 암기와도 합치 — 컷오프 공개 전까지 해석 유보)
  슬롯(동표본 32Q): XGB 0.7393 | C2f슬롯 0.7294 | 209M슬롯 0.7328 | 28M슬롯 0.7280 | 듀오 0.7279(조기 오차상관 0.83 — 중복)
    → 28M 슬롯이 -0.2% 미세 우위이나 선택편의+오염 미상으로 교체 권고 없음. 운영 구성 불변.
  분위수 커버리지: 명목 80% 구간 실측 47~55% — 커뮤니티 보정 불량 보고 재현 (C2 사전 conformal 49%와 동급)
"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch
from timestral import TimestralForecaster
import phase_b_harness as H
import fast_signals as FS

COVARS = ["I_m", "I_s", "M_s", "S_es", "S_cb", "S_mo", "B_bx", "B_bi", "R_s", "M_fi"]
FASTC = ["kospi_mret", "krw_mret", "esi_raw", "esi_mom"]
PLEN_MAX = 6
CTX_MAX = int(os.environ.get("CTX_MAX", "200"))
COVMODE = os.environ.get("COVMODE", "full")   # full=10종+빠른신호 | base=10종만 | uni=공변량 없음
MODEL_ID = os.environ.get("MODEL", "mistralai/Timestral-209M")
_size = MODEL_ID.rsplit("-", 1)[-1].lower()
TAG = os.environ.get("TAG", {"full": f"our_timestral{_size}_c2p", "base": f"our_timestral{_size}_base",
                             "uni": f"our_timestral{_size}_uni"}[COVMODE])

grid, _ = H.load_grid()
g = grid.copy(); g["vintage"] = pd.to_datetime(g["vintage"]).dt.strftime("%Y-%m-%d")
QSUB = os.environ.get("QSUB")
quarters = QSUB.split(",") if QSUB else sorted(g.tq.unique(), key=lambda x: pd.Period(x, "Q"))

torch.manual_seed(7)  # symmetric_averaging 외 확률 요소는 없다고 보나, 재현성 고정
model = TimestralForecaster.from_pretrained(MODEL_ID, device="cpu")
print(f"[{MODEL_ID}] loaded", flush=True)

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

def timestral_predict(hist, h):
    """hist: (T, 1+C) DataFrame [N_gdp + 공변량들], h: 예측 지평(월). 반환: (median, q10, q90)@h."""
    hist = hist.tail(CTX_MAX)
    covcols = [c for c in hist.columns if c != "N_gdp"]
    ctx = torch.from_numpy(hist["N_gdp"].values.astype(np.float32))[None, None, :]   # (1,1,T)
    pc = None
    if covcols:
        pc = torch.from_numpy(hist[covcols].values.T.astype(np.float32))[None, :, :]  # (1,C,T)
    with torch.no_grad():
        fc = model.forecast(ctx, prediction_length=h, past_covariates=pc)
    med = float(fc.median[0, 0, h - 1])
    q10 = float(fc.quantile(0.1)[0, 0, h - 1])
    q90 = float(fc.quantile(0.9)[0, 0, h - 1])
    return med, q10, q90

rows = []
for tq in quarters:
    sub = g[g.tq == tq].drop_duplicates(["vintage", "week_idx"]).sort_values("week_idx")
    if sub.empty: continue
    qend = pd.Period(tq, "Q").end_time.normalize().replace(day=1) + pd.offsets.MonthEnd(0)
    for r in sub.itertuples(index=False):
        yp, q10, q90 = np.nan, np.nan, np.nan
        panel = load_panel(tq, r.vintage)
        if panel is not None:
            if COVMODE == "uni":
                panel = panel[["N_gdp"]]
            elif COVMODE == "full":
                panel = with_fast(panel, r.vintage)
            edge = pd.Timestamp(r.vintage) + pd.offsets.MonthEnd(0)
            hist = panel[panel.index <= edge]
            h = int((qend.to_period("M") - hist.index[-1].to_period("M")).n) if len(hist) else 99
            if h < 1:
                yp = float(panel.loc[qend, "N_gdp"]) if qend in panel.index else np.nan
            elif h <= PLEN_MAX and len(hist) >= 60:
                try:
                    yp, q10, q90 = timestral_predict(hist, h)
                except Exception as e:
                    print(f"  predict err {tq} {r.vintage}: {type(e).__name__}: {e}", flush=True)
        rows.append({"tq": tq, "vintage": r.vintage, "week_idx": r.week_idx,
                     "flash": r.flash, "model_name": TAG, "y_pred": yp, "q10": q10, "q90": q90})
    print(f"[{tq}] done", flush=True)

pred = pd.DataFrame(rows)
pred.to_csv(f"output/csv/_phase_b_{TAG}_predictions.csv", index=False)
print(f"\n=== Timestral({MODEL_ID}, {COVMODE}) flash w[-19,-1] avg RMSE ===")
print(H.score(pred.drop(columns=["q10", "q90"])).to_string())
cov = pred.dropna(subset=["y_pred", "q10"])
if len(cov):
    inside = ((cov.flash >= cov.q10) & (cov.flash <= cov.q90)).mean()
    print(f"[분위수 커버리지] 명목 80% 구간 실측 커버 {inside*100:.1f}% (n={len(cov)})")
print("[벤치] Chronos-2f 0.8075(조기 0.9238) / Moirai-small 0.8731(조기 1.0206) / GBM 조기 0.924 / XGB(신) 0.750")
print("PHASE_B_TIMESTRAL_C2P_DONE")
