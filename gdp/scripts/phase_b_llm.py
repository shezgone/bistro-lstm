"""phase_b_llm: Claude LLM(Fable 5 / Opus 4.8) zero-shot GDP nowcast — 2025Q1·Q2 종전방식 비교.

각 (tq, vintage, week)마다 해당 빈티지 xlsx의 월별 지표(최근 96개월, ragged edge 그대로)와
그 빈티지에 기록된 분기 GDP 이력(N_gdp, vintage-safe)만 입력으로 주고 flash q/q 성장률 예측.
채점은 phase_b_harness.score (flash w[-19,-1] 평균 RMSE) 동일 잣대.

주의: Fable 5(cutoff 2026-01)·Opus 4.8은 2025년 발표치를 기억할 수 있음 → memorization 오염 캐비엇.
"""
import sys, os, json, warnings, argparse
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd
import anthropic
import phase_b_harness as H

QUARTERS = ["2025Q1", "2025Q2"]
MODELS = {"llm_fable5": "claude-fable-5", "llm_opus48": "claude-opus-4-8"}
N_MONTHS = 96          # 월별 지표 입력 창
MAX_WORKERS = 4
OUT_CSV = "output/csv/_phase_b_llm_predictions.csv"

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    kf = os.path.expanduser("~/.anthropic_key")
    if os.path.exists(kf):
        API_KEY = open(kf).read().strip()
if not API_KEY:
    sys.exit("ANTHROPIC_API_KEY missing (env 또는 ~/.anthropic_key)")

client = anthropic.Anthropic(api_key=API_KEY, max_retries=3)

SCHEMA = {"type": "object",
          "properties": {"gdp_qoq": {"type": "number"}},
          "required": ["gdp_qoq"], "additionalProperties": False}

SYSTEM = ("You are an expert macroeconomic nowcaster for the Korean economy. "
          "You will be given a real-time vintage of monthly indicators (some recent cells "
          "are missing because they were not yet released as of the vintage date) and the "
          "history of quarterly real GDP growth known at that date. "
          "Predict the flash (advance) estimate of Korea's real GDP growth for the target "
          "quarter, quarter-over-quarter, seasonally adjusted, in percent. "
          "Use only the data provided; do not rely on memorized knowledge of actual outcomes.")

_vintage_cache = {}

def load_vintage(vintage):
    if vintage not in _vintage_cache:
        x = pd.read_excel(f"data/vintages/{vintage}.xlsx")
        _vintage_cache[vintage] = x
    return _vintage_cache[vintage]

def build_prompt(tq, vintage):
    x = load_vintage(vintage)
    # 분기 GDP 이력: 빈티지 파일의 N_gdp (그 시점에 알려진 발표치만 존재 → vintage-safe)
    g = x[["Date", "N_gdp"]].dropna()
    g["q"] = pd.PeriodIndex(pd.to_datetime(g["Date"]), freq="Q").astype(str)
    gdp_hist = "\n".join(f"{r.q}\t{r.N_gdp:g}" for r in g.itertuples())
    # 월별 지표: 최근 N_MONTHS개월 (N_* 분기 계정 컬럼 제외), ragged edge 그대로
    m = x.drop(columns=[c for c in x.columns if c.startswith("N_")]).tail(N_MONTHS)
    tsv = m.to_csv(sep="\t", index=False, float_format="%.3f", na_rep="")
    user = (f"Vintage date: {vintage}\n"
            f"Target quarter: {tq}\n\n"
            f"Quarterly real GDP growth (q/q, %, flash releases known as of vintage date):\n"
            f"quarter\tgdp_qoq\n{gdp_hist}\n\n"
            f"Monthly indicators (last {N_MONTHS} months as of vintage date):\n{tsv}\n"
            f"Predict {tq} flash real GDP growth (q/q, %).")
    return user

def call_llm(model_id, tq, vintage):
    kwargs = dict(
        model=model_id, max_tokens=16000, system=SYSTEM,
        output_config={"effort": "medium",
                       "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": build_prompt(tq, vintage)}],
    )
    if model_id != "claude-fable-5":          # Fable 5: thinking 상시-on(파라미터 생략)
        kwargs["thinking"] = {"type": "adaptive"}
    r = client.messages.create(**kwargs)
    if r.stop_reason == "refusal":
        return np.nan, r.usage
    txt = next(b.text for b in r.content if b.type == "text")
    return float(json.loads(txt)["gdp_qoq"]), r.usage

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 call per model only")
    args = ap.parse_args()

    grid, refdf = H.load_grid()
    g = grid[grid.tq.isin(QUARTERS) & grid.week_idx.between(-19, -1)].copy()
    g["vintage"] = pd.to_datetime(g["vintage"]).dt.strftime("%Y-%m-%d")
    g = g.drop_duplicates(["tq", "vintage", "week_idx"]).reset_index(drop=True)
    if args.smoke:
        g = g.groupby("tq").head(1).head(1)
    jobs = [(mn, mid, r) for mn, mid in MODELS.items() for r in g.itertuples(index=False)]
    print(f"calls: {len(jobs)} ({len(g)} grid rows x {len(MODELS)} models)")

    rows, in_tok, out_tok = [], 0, 0
    def work(mn, mid, r):
        yp, usage = call_llm(mid, r.tq, r.vintage)
        return {"tq": r.tq, "vintage": r.vintage, "week_idx": r.week_idx,
                "flash": r.flash, "model_name": mn, "y_pred": yp}, usage
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(work, mn, mid, r): (mn, r) for mn, mid, r in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            mn, r = futs[f]
            try:
                row, usage = f.result()
                in_tok += usage.input_tokens; out_tok += usage.output_tokens
                rows.append(row)
                print(f"[{i}/{len(jobs)}] {mn} {r.tq} {r.vintage} w{r.week_idx} -> {row['y_pred']}")
            except Exception as e:
                print(f"[{i}/{len(jobs)}] {mn} {r.tq} {r.vintage} FAILED: {e}")
    pred = pd.DataFrame(rows)
    if not args.smoke:
        pred.to_csv(OUT_CSV, index=False)
        print("saved:", OUT_CSV)
    print(f"tokens: in={in_tok:,} out={out_tok:,}")

    # ---- 채점: 종전 방식과 동일 subset 비교 ----
    def norm(d):
        d = d.copy(); d["vintage"] = pd.to_datetime(d["vintage"]).dt.strftime("%Y-%m-%d"); return d
    dfm = norm(H.load_baseline(grid, "dfm")); dfm = dfm[dfm.tq.isin(QUARTERS)]
    xgb = norm(refdf[refdf.model_name == "xgboost"])[["tq","vintage","week_idx","flash","y_pred","model_name"]]
    xgb = xgb[xgb.tq.isin(QUARTERS)]
    ens_xgb = H.ensemble_with_dfm(xgb, dfm, suffix="xgboost")
    gated = pd.read_csv("output/csv/_phase_b_regime_gated.csv", dtype={"tq": str})
    gated = norm(gated).rename(columns={"gated": "y_pred"}); gated["model_name"] = "regime_gated"
    gated = gated[gated.tq.isin(QUARTERS)][["tq","vintage","week_idx","flash","model_name","y_pred"]]
    parts = [dfm, xgb, ens_xgb, gated, pred]
    for mn in pred.model_name.unique():   # DFM+LLM 앙상블도 참고용
        parts.append(H.ensemble_with_dfm(pred[pred.model_name == mn], dfm, suffix=mn))
    allp = pd.concat(parts, ignore_index=True)
    print("\n=== 2025Q1+Q2 flash w[-19,-1] avg RMSE ===")
    print(H.score(allp).to_string())

if __name__ == "__main__":
    main()
