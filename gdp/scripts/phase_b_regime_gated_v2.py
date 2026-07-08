"""regime-gated v2 (3-arm): v1(shock/calm)에 반등(REBOUND) 국면 추가.

판정 (분기당 1번, 위에서부터 먼저 걸리는 국면):
  1) REBOUND: flash(q-1) < 0 AND 심리(ESI) 저점통과(마지막 관측월 > 최근 6개월 최저, 저점이 마지막 달 이전)
              -> DFM 단독 (ML 보정 OFF)
  2) SHOCK  : vol(q) > 과거 vol 중앙값 (v1과 동일)  -> (DFM+XGBoost)/2
  3) CALM   : 그 외                                  -> (DFM+RF)/2

심리 조건은 분기 시작 전월까지 관측분만 사용(vintage-safe). ESI/CSI x 윈도우 4~12개월
8개 변형 모두 동일 분류(실질 효과 = 2020Q2 연속추락기 제외).

--strict: flash(q-1) 발표(직전 분기 종료+28일 근사) 이후 주차부터만 REBOUND 발동.
          v1의 vol 판정도 같은 분기 단위 근사를 쓰므로 기본값은 v1과 동일 조건 비교.

결과 (flash w[-19,-1] 평균 RMSE, 2018Q1-2025Q4):
  v2 기본 0.7219 / v2 --strict 0.7378 / v1 0.7548 / DFM+XGBoost 0.7650
캐비엇: 반등 arm(DFM 단독)은 사후 선택 구조(선택편향), 반등 6분기 소표본(4승2패), DM 미검증.
"""
import sys, argparse, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
import phase_b_harness as H

SENT_VINTAGE = "data/vintages/2026-03-04.xlsx"  # 심리지표(서베이, 개정 거의 없음) 소스
SENT_COL, SENT_WIN = "new_esi", 6

ap = argparse.ArgumentParser()
ap.add_argument("--strict", action="store_true", help="flash(q-1) 발표 후 주차만 반등 arm 발동")
args = ap.parse_args()

# ---- 반등 분기 판정 ----
x = pd.read_excel(SENT_VINTAGE)
if "date" in x.columns: x = x.rename(columns={"date": "Date"})
x["Date"] = pd.to_datetime(x["Date"])
sent = x.set_index("Date")[SENT_COL].dropna()
rel = pd.read_pickle("data/GDP_releases.pkl"); fl = rel["flash"].dropna()
qs = [str(p) for p in pd.period_range("2017Q4", "2025Q4", freq="Q")]
flq = fl.reindex([q for q in qs if q in fl.index])

def sent_trough_passed(q):
    e = sent[sent.index < pd.Period(q, "Q").start_time].tail(SENT_WIN)
    return len(e) >= 3 and e.iloc[-1] > e.min() and e.idxmin() < e.index[-1]

REBOUND = {qs[i] for i in range(1, len(qs))
           if qs[i] >= "2018Q1" and qs[i-1] in flq.index
           and flq[qs[i-1]] < 0 and sent_trough_passed(qs[i])}
print(f"REBOUND 분기 {len(REBOUND)}개: {sorted(REBOUND)}")

# ---- v1 gated + base 예측 로드 ----
grid, refdf = H.load_grid(); KEY = ["tq", "vintage", "week_idx"]
def norm(d):
    d = d.copy(); d["vintage"] = pd.to_datetime(d["vintage"]).dt.strftime("%Y-%m-%d"); return d
b = norm(H.load_baseline(grid, "dfm"))[KEY + ["y_pred", "flash"]].rename(columns={"y_pred": "dfm"})
gated = norm(pd.read_csv("output/csv/_phase_b_regime_gated.csv", dtype={"tq": str}))[KEY + ["gated"]]
b = b.merge(gated, on=KEY, how="left").dropna(subset=["dfm", "gated"])

# ---- 게이트 적용 ----
mask = b.tq.isin(REBOUND)
if args.strict:
    def prev_flash_known(tq, vintage):
        return pd.Timestamp(vintage) >= (pd.Period(tq, "Q") - 1).end_time + pd.Timedelta(days=28)
    mask = mask & b.apply(lambda r: prev_flash_known(r.tq, r.vintage), axis=1)
b["gated_v2"] = np.where(mask, b.dfm, b.gated)

def rmse(col, sub=None):
    d = b if sub is None else b[sub]
    t = pd.DataFrame({"model_name": col, "tq": d.tq, "vintage": d.vintage,
                      "week_idx": d.week_idx, "flash": d.flash, "y_pred": d[col]})
    s = H.score(t); return float(s.iloc[0]) if len(s) else np.nan

reb = b.tq.isin(REBOUND)
print(f"\n=== v2 {'(strict)' if args.strict else '(기본)'} — flash w[-19,-1] avg RMSE ===")
print(f"  {'v2 (3-arm)':16s} 전체 {rmse('gated_v2'):.4f} | 반등분기 {rmse('gated_v2', reb):.4f}")
print(f"  {'v1 (2-arm)':16s} 전체 {rmse('gated'):.4f} | 반등분기 {rmse('gated', reb):.4f}")
out = f"output/csv/_phase_b_regime_gated_v2{'_strict' if args.strict else ''}.csv"
b[KEY + ["gated_v2", "flash"]].to_csv(out, index=False)
print("saved:", out)
