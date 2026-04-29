"""HCX rolling 2025 OOS with few-shot examples + forced CoT.

For each origin t:
- 3 few-shot examples from (t-3, t-2, t-1), each with 12-month panel + actual
- Then the actual query at origin t with 36-month panel
- Combined with the forced 4-step CoT prompt (CSI focus)

Vintage-safe: examples only use data observable by origin t.
"""
import os, sys, json, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, numpy as np, pandas as pd

API_URL = "https://namc-aigw.io.naver.com/v1/chat/completions"
MODEL = "HyperCLOVAX-SEED-32B-Think-Text"
API_KEY = os.environ.get("HCX_API_KEY")
if not API_KEY: sys.exit("HCX_API_KEY missing")

CSV_FULL = "data/macro_panel_full.csv"
df_full = pd.read_csv(CSV_FULL, index_col=0); df_full.index = pd.PeriodIndex(df_full.index, freq="M")
gt_cols = [c for c in df_full.columns if c.startswith("GT_")]
df_csi = df_full.drop(columns=gt_cols)  # 18 + BoK_CSI

ORIGINS = pd.period_range("2025-04", "2025-11", freq="M")
N_SEEDS = 5
TEMP = 0.7
CTX_LEN = 36
EX_LEN = 12  # months per few-shot example
N_EX = 3


def panel_to_tsv(df_slice: pd.DataFrame) -> str:
    df_slice = df_slice.round(3).copy()
    df_slice.index = df_slice.index.astype(str)
    return df_slice.to_csv(sep="\t")


def build_messages(origin: pd.Period):
    target = origin + 1
    cols = list(df_csi.columns)

    # Few-shot examples: (origin - N_EX, ..., origin - 1) each predicting next month
    examples = []
    for k in range(N_EX, 0, -1):
        ex_origin = origin - k
        ex_target = ex_origin + 1
        ex_ctx_start = ex_origin - EX_LEN + 1
        ex_panel = df_csi.loc[str(ex_ctx_start):str(ex_origin), cols]
        ex_actual = float(df_csi.loc[ex_target, "CPI_KR_YoY"])
        examples.append({
            "origin": str(ex_origin), "target": str(ex_target),
            "panel_tsv": panel_to_tsv(ex_panel), "actual": ex_actual,
        })

    # Query panel (full 36 months ending at origin)
    q_ctx_start = origin - CTX_LEN + 1
    q_panel = df_csi.loc[str(q_ctx_start):str(origin), cols]
    q_panel_tsv = panel_to_tsv(q_panel)

    sys_msg = (
        "You are an expert macroeconomist forecasting Korean CPI YoY inflation. "
        "You will be shown 3 historical 1-step-ahead forecast EXAMPLES (with answer) "
        "followed by the actual question. Each example shows a 12-month macro panel "
        "ending at the example's origin and the actual CPI YoY for the next month. "
        "Use these examples to calibrate your forecasting approach for the query.\n\n"
        "CRITICAL — for the final query, you MUST follow this 4-step CoT in rationale:\n"
        "Step 1: BoK_CSI 트렌드 (직전 6개월 값 인용).\n"
        "Step 2: CSI 방향성 → 인플레 압력 해석.\n"
        "Step 3: 매크로 covariate cross-check (BoK 금리, 원자재, FX, base effect).\n"
        "Step 4: 종합 → 단일 forecast + 가중치 명시.\n\n"
        "BoK_CSI = Korean Consumer Sentiment Index (100=neutral, >100=optimistic, <100=pessimistic).\n\n"
        "Return ONLY a JSON object: "
        '{"forecast": v, "rationale": "..."} '
        "where v is a single decimal number like 2.3 (no % sign). "
        "Rationale must explicitly state BoK_CSI values used."
    )

    # Build user message: 3 examples + query
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(
            f"==== EXAMPLE {i} ====\n"
            f"Macro panel ending {ex['origin']} (12 months):\n"
            f"{ex['panel_tsv']}\n"
            f"Actual CPI_KR_YoY for {ex['target']}: {ex['actual']:.3f}\n"
        )
    parts.append(
        f"==== QUERY ====\n"
        f"Macro panel ending {origin} (36 months):\n"
        f"{q_panel_tsv}\n"
        f"Forecast CPI_KR_YoY for {target} (1-step-ahead).\n"
        f"Output only the JSON specified in the system message."
    )
    user_msg = "\n".join(parts)
    return sys_msg, user_msg


def hcx_call(origin, seed_idx):
    sys_msg, user_msg = build_messages(origin)
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": sys_msg},
                     {"role": "user", "content": user_msg}],
        "temperature": TEMP, "max_tokens": 8192,
    }
    for retry in range(3):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=240)
            if r.status_code != 200:
                if retry < 2: time.sleep(2 ** retry); continue
                return None, None, f"HTTP {r.status_code}"
            content = r.json()["choices"][0]["message"].get("content")
            if not content:
                if retry < 2: time.sleep(2 ** retry); continue
                return None, None, "empty"
            m = re.search(r"\{[\s\S]*\}", content)
            obj = json.loads(m.group(0))
            v = float(obj["forecast"]) if not isinstance(obj["forecast"], list) else float(obj["forecast"][0])
            return v, obj.get("rationale", ""), None
        except Exception as e:
            if retry < 2: time.sleep(2 ** retry); continue
            return None, None, str(e)[:100]
    return None, None, "max retries"


def main():
    targets = [str(o + 1) for o in ORIGINS]
    actuals = {t: float(df_csi.loc[t, "CPI_KR_YoY"]) for t in targets}
    actual_arr = np.array([actuals[t] for t in targets])

    # Print prompt size for first origin (sanity check)
    s, u = build_messages(ORIGINS[0])
    print(f"Sample prompt sizes for origin {ORIGINS[0]}: system={len(s)} chars, user={len(u)} chars\n", flush=True)

    jobs = [(o, s) for o in ORIGINS for s in range(N_SEEDS)]
    print(f"Total: {len(jobs)} ({len(ORIGINS)} origins × {N_SEEDS} seeds)\n", flush=True)

    forecasts = {}
    rationales = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(hcx_call, *j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            o, s = futures[fut]
            v, rat, err = fut.result()
            tgt = str(o + 1)
            if v is not None:
                forecasts.setdefault(tgt, []).append(v)
                rationales.setdefault(tgt, []).append(rat)
            done += 1
            print(f"[{done}/{len(jobs)}] {tgt} s{s}: {'%.3f' % v if v is not None else f'FAIL ({err})'}", flush=True)
    print(f"\nElapsed: {time.time()-t0:.1f}s")

    # Sample rationales
    print("\n" + "=" * 90)
    print("Sample rationales (first seed):")
    for tgt in targets:
        rats = rationales.get(tgt, [])
        if rats:
            print(f"\n[{tgt}] {rats[0][:400]}")

    # Stats
    print("\n" + "=" * 90)
    print(f"{'Target':10s} {'Actual':>8s} | {'fewshot μ':>10s} {'σ':>6s}")
    print("-" * 90)
    rows = []
    for tgt in targets:
        actual = actuals[tgt]
        fc = forecasts.get(tgt, [])
        m = np.mean(fc) if fc else float("nan")
        s = np.std(fc, ddof=1) if len(fc) > 1 else 0.0
        print(f"{tgt:10s} {actual:8.3f} | {m:10.3f} {s:6.3f}")
        rows.append({"target": tgt, "actual": actual, "fewshot": fc})

    fewshot_means = np.array([np.mean(r["fewshot"]) if r["fewshot"] else np.nan for r in rows])
    fewshot_rmse_mc = np.sqrt(np.nanmean((fewshot_means - actual_arr)**2))

    fewshot_per_seed = []
    for s in range(N_SEEDS):
        preds = [r["fewshot"][s] if s < len(r["fewshot"]) else np.nan for r in rows]
        preds = np.array(preds)
        if not np.any(np.isnan(preds)):
            fewshot_per_seed.append(np.sqrt(np.mean((preds - actual_arr)**2)))
    fa = np.array(fewshot_per_seed)

    print("\n" + "=" * 90)
    print(f"{'Method':30s} {'mean-curve RMSE':>16s} {'per-seed RMSE':>22s}")
    print("-" * 90)
    print(f"{'HCX cov_csi_fewshot+CoT':30s} {fewshot_rmse_mc:16.4f} "
          f"{f'{fa.mean():.4f}±{fa.std(ddof=1):.4f} (n={len(fa)})':>22s}")
    print(f"{'(prior) HCX forced CoT':30s} {0.2504:16.4f} {'0.2633±0.0044 (n=5)':>22s}")
    print(f"{'(prior) HCX cov_base':30s} {0.2955:16.4f} {'0.3001±0.0188 (n=5)':>22s}")
    print(f"{'(baseline) Trend12':30s} {0.2575:16.4f} {'(deterministic)':>22s}")

    # vs forced CoT (load prior data)
    try:
        with open("data/rolling_2025_csi_forced_results.json") as f:
            prior = json.load(f)
        prior_per_seed = prior.get("forced_per_seed_rmse", [])
        if prior_per_seed and len(fa) > 1:
            from scipy import stats
            t, p = stats.ttest_ind(fa, np.array(prior_per_seed), equal_var=False)
            print(f"\nWelch's t fewshot vs forced CoT: t={t:+.3f}, p={p:.4f}")
            print(f"Δ (fewshot - forced) = {fa.mean() - np.mean(prior_per_seed):+.4f}")
    except Exception as e:
        print(f"Could not compare to forced: {e}")

    out = {
        "targets": targets, "actuals": actuals,
        "forecasts": forecasts,
        "rationales_first_seed": {t: r[0] if r else "" for t, r in rationales.items()},
        "n_seeds": N_SEEDS, "temperature": TEMP, "ctx_len": CTX_LEN,
        "fewshot_per_seed_rmse": fa.tolist(),
    }
    with open("data/rolling_2025_fewshot_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved data/rolling_2025_fewshot_results.json")


if __name__ == "__main__":
    main()
