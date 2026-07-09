"""HCX rolling 2025 format ablation: minimal header changes vs current TSV.

Tests ONLY:
- Header first cell: '' → 'Date'
- Column headers: 'CPI_KR_YoY' → 'CPI_KR_YoY (YoY %)'
NOT changed (kept identical):
- TSV format (not markdown)
- No variable definitions in system msg
- Sentinels left visible (-0.999, 11.111)
- Same forced CoT 4-step prompt
- Same data, same model, same temp
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
df_csi = df_full.drop(columns=gt_cols)

ORIGINS = pd.period_range("2025-04", "2025-11", freq="M")
N_SEEDS = 5
TEMP = 0.7
CTX_LEN = 36

UNITS = {
    "CPI_KR_YoY": "YoY %", "AUD_USD": "FX", "CN_Interbank3M": "%",
    "US_UnempRate": "%", "JP_REER": "idx", "JP_Interbank3M": "%",
    "JP_CoreCPI": "YoY %", "KC_FSI": "idx", "KR_MfgProd": "YoY %",
    "Pork": "won/kg", "US_NFP": "k MoM", "US_TradeTransEmp": "k MoM",
    "THB_USD": "FX", "PPI_CopperNickel": "YoY %", "CN_PPI": "YoY %",
    "US_Mortgage15Y": "%", "UK_10Y_Bond": "%", "US_ExportPI": "YoY %",
    "US_DepInstCredit": "%", "BoK_CSI": "idx 100=neutral",
}


def panel_to_tsv_with_units(df_slice: pd.DataFrame) -> str:
    """TSV with 'Date' header + units in column names. No other changes."""
    df_slice = df_slice.round(3).copy()
    df_slice.index = df_slice.index.astype(str)
    # Rename columns to include units
    new_cols = [f"{c} ({UNITS.get(c, '?')})" for c in df_slice.columns]
    df_slice = df_slice.copy()
    df_slice.columns = new_cols
    df_slice.index.name = "Date"
    return df_slice.to_csv(sep="\t")


def build_messages(origin: pd.Period):
    target = origin + 1
    cols = list(df_csi.columns)
    ctx_start = origin - CTX_LEN + 1
    panel = df_csi.loc[str(ctx_start):str(origin), cols]
    table = panel_to_tsv_with_units(panel)

    sys_msg = (
        "You are an expert macroeconomist forecasting Korean CPI YoY inflation. "
        f"You have monthly data through {origin}. "
        f"Forecast Korean CPI YoY (%) for the SINGLE NEXT month: {target}.\n\n"
        "CRITICAL — you MUST use BoK_CSI as a primary leading indicator:\n"
        "- BoK_CSI = Bank of Korea's monthly survey of Korean households about future\n"
        "  economic outlook (consumer sentiment about income, spending, prices).\n"
        "- 100 = neutral, >100 = optimistic, <100 = pessimistic.\n"
        "- Survey-based sentiment captures forward-looking inflation expectations\n"
        "  that are NOT visible in contemporaneous macro time series.\n\n"
        "Required reasoning steps (your rationale MUST cover all 4):\n"
        "Step 1: Describe the recent BoK_CSI trend over the last 6 months\n"
        "  (cite specific values, e.g. '2025-04: 93.6 → 2025-08: 111.2').\n"
        "Step 2: Interpret this sentiment direction — does it imply rising or\n"
        "  falling demand-side inflation pressure? Why?\n"
        "Step 3: Cross-check with traditional macro covariates (BoK rate, commodity,\n"
        "  FX, base effect from one year ago).\n"
        "Step 4: Combine into a single point forecast and explain how Step 1-3 weighted.\n\n"
        "Return ONLY a JSON object:\n"
        '{"forecast": v, "rationale": "..."} '
        "where v is a single decimal number like 2.3 (no % sign). "
        "Rationale must explicitly state the BoK_CSI values used."
    )
    user_msg = (
        f"Macro panel (18 covariates + BoK CSI), {ctx_start} to {origin}, monthly TSV:\n"
        f"{table}\n\n"
        f"Forecast CPI_KR_YoY for {target} (single value, 1-step-ahead). "
        "Output only the JSON specified in the system message. "
        "Remember: rationale MUST cite specific BoK_CSI values from the data."
    )
    return sys_msg, user_msg


def hcx_call(origin, seed_idx):
    sys_msg, user_msg = build_messages(origin)
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL,
               "messages": [{"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg}],
               "temperature": TEMP, "max_tokens": 8192}
    for retry in range(3):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=240)
            if r.status_code != 200:
                if retry < 2: time.sleep(2 ** retry); continue
                return None, f"HTTP {r.status_code}"
            content = r.json()["choices"][0]["message"].get("content")
            if not content:
                if retry < 2: time.sleep(2 ** retry); continue
                return None, "empty"
            m = re.search(r"\{[\s\S]*\}", content)
            obj = json.loads(m.group(0))
            v = float(obj["forecast"]) if not isinstance(obj["forecast"], list) else float(obj["forecast"][0])
            return v, None
        except Exception as e:
            if retry < 2: time.sleep(2 ** retry); continue
            return None, str(e)[:100]
    return None, "max retries"


def main():
    targets = [str(o + 1) for o in ORIGINS]
    actuals = {t: float(df_csi.loc[t, "CPI_KR_YoY"]) for t in targets}
    actual_arr = np.array([actuals[t] for t in targets])

    sample_s, sample_u = build_messages(ORIGINS[0])
    print(f"--- Format change preview (origin {ORIGINS[0]}) ---")
    print(f"User msg (first 500 chars): {sample_u[:500]}\n", flush=True)

    jobs = [(o, s) for o in ORIGINS for s in range(N_SEEDS)]
    print(f"Total: {len(jobs)} ({len(ORIGINS)} origins × {N_SEEDS} seeds)\n", flush=True)

    forecasts = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(hcx_call, *j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            o, s = futures[fut]
            v, err = fut.result()
            tgt = str(o + 1)
            if v is not None:
                forecasts.setdefault(tgt, []).append(v)
            done += 1
            print(f"[{done}/{len(jobs)}] {tgt} s{s}: {'%.3f' % v if v is not None else f'FAIL ({err})'}", flush=True)
    print(f"\nElapsed: {time.time()-t0:.1f}s")

    # Stats
    print("\n" + "=" * 90)
    print(f"{'Target':10s} {'Actual':>8s} | {'fmt μ':>8s} {'σ':>6s}")
    print("-" * 90)
    rows = []
    for tgt in targets:
        actual = actuals[tgt]
        fc = forecasts.get(tgt, [])
        m = np.mean(fc) if fc else float("nan")
        s = np.std(fc, ddof=1) if len(fc) > 1 else 0.0
        print(f"{tgt:10s} {actual:8.3f} | {m:8.3f} {s:6.3f}")
        rows.append({"target": tgt, "actual": actual, "fmt": fc})

    fmt_means = np.array([np.mean(r["fmt"]) if r["fmt"] else np.nan for r in rows])
    fmt_rmse_mc = np.sqrt(np.nanmean((fmt_means - actual_arr)**2))

    fmt_per_seed = []
    for s in range(N_SEEDS):
        preds = [r["fmt"][s] if s < len(r["fmt"]) else np.nan for r in rows]
        preds = np.array(preds)
        if not np.any(np.isnan(preds)):
            fmt_per_seed.append(np.sqrt(np.mean((preds - actual_arr)**2)))
    fa = np.array(fmt_per_seed)

    print("\n" + "=" * 90)
    print(f"{'Method':40s} {'mean-curve RMSE':>16s} {'per-seed RMSE':>22s}")
    print("-" * 90)
    print(f"{'HCX cov_csi_forced (Date+units header)':40s} {fmt_rmse_mc:16.4f} "
          f"{f'{fa.mean():.4f}±{fa.std(ddof=1):.4f} (n={len(fa)})':>22s}")
    print(f"{'(prior) HCX forced CoT (default)':40s} {0.2504:16.4f} {'0.2633±0.0044 (n=5)':>22s}")
    print(f"{'(baseline) Trend12':40s} {0.2575:16.4f} {'(deterministic)':>22s}")

    # Compare to prior
    try:
        with open("data/rolling_2025_csi_forced_results.json") as f:
            prior = json.load(f)
        prior_per_seed = prior.get("forced_per_seed_rmse", [])
        if prior_per_seed and len(fa) > 1:
            from scipy import stats
            t, p = stats.ttest_ind(fa, np.array(prior_per_seed), equal_var=False)
            print(f"\nWelch's t fmt vs default: t={t:+.3f}, p={p:.4f}")
            print(f"Δ (fmt - default) = {fa.mean() - np.mean(prior_per_seed):+.4f}")
    except Exception as e:
        print(f"Could not compare: {e}")

    out = {"targets": targets, "actuals": actuals, "forecasts": forecasts,
           "n_seeds": N_SEEDS, "temperature": TEMP, "ctx_len": CTX_LEN,
           "fmt_per_seed_rmse": fa.tolist()}
    with open("data/rolling_2025_format_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved data/rolling_2025_format_results.json")


if __name__ == "__main__":
    main()
