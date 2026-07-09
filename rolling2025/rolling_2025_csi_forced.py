"""HCX rolling 2025 with CSI + FORCED attention prompt.

Re-uses cov_base results from rolling_2025_csi_results.json.
This script only adds: cov_csi_forced (CoT forcing on BoK_CSI).
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


def build_messages(origin: pd.Period):
    ctx_start = origin - CTX_LEN + 1
    target = origin + 1
    cols = list(df_csi.columns)
    ctx = df_csi.loc[str(ctx_start):str(origin), cols].round(3)
    ctx.index = ctx.index.astype(str)
    table = ctx.to_csv(sep="\t")

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
            rationale = obj.get("rationale", "")
            return v, rationale, None
        except Exception as e:
            if retry < 2: time.sleep(2 ** retry); continue
            return None, None, str(e)[:100]
    return None, None, "max retries"


def main():
    targets = [str(o + 1) for o in ORIGINS]
    actuals = {t: float(df_csi.loc[t, "CPI_KR_YoY"]) for t in targets}
    actual_arr = np.array([actuals[t] for t in targets])

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

    # Print one example rationale per target
    print("\n" + "=" * 90)
    print("Sample rationales (first seed per target):")
    print("-" * 90)
    for tgt in targets:
        rats = rationales.get(tgt, [])
        if rats:
            print(f"\n[{tgt}] {rats[0][:400]}")

    # Stats — load base from previous experiment
    base_results = {}
    try:
        with open("data/rolling_2025_csi_results.json") as f:
            prev = json.load(f)
        for k, v in prev["results"].items():
            tgt, mode = k.split("|")
            if mode == "cov_base":
                base_results[tgt] = v
    except Exception as e:
        print(f"Could not load base: {e}")

    print("\n" + "=" * 90)
    print(f"{'Target':10s} {'Actual':>8s} | {'base μ':>8s} {'σ':>6s} | {'forced μ':>10s} {'σ':>6s} | {'Δ forced-base':>14s}")
    print("-" * 90)
    rows = []
    for tgt in targets:
        actual = actuals[tgt]
        base = base_results.get(tgt, [])
        forced = forecasts.get(tgt, [])
        bm = np.mean(base) if base else float("nan")
        bs = np.std(base, ddof=1) if len(base) > 1 else 0.0
        fm = np.mean(forced) if forced else float("nan")
        fs = np.std(forced, ddof=1) if len(forced) > 1 else 0.0
        delta = fm - bm
        print(f"{tgt:10s} {actual:8.3f} | {bm:8.3f} {bs:6.3f} | {fm:10.3f} {fs:6.3f} | {delta:+14.3f}")
        rows.append({"target": tgt, "actual": actual, "base": base, "forced": forced})

    forced_per_seed = []
    for s in range(N_SEEDS):
        preds = []
        for r in rows:
            preds.append(r["forced"][s] if s < len(r["forced"]) else np.nan)
        preds = np.array(preds)
        if not np.any(np.isnan(preds)):
            forced_per_seed.append(np.sqrt(np.mean((preds - actual_arr)**2)))
    forced_arr = np.array(forced_per_seed)

    base_per_seed = []
    if base_results:
        for s in range(N_SEEDS):
            preds = []
            for r in rows:
                preds.append(r["base"][s] if s < len(r["base"]) else np.nan)
            preds = np.array(preds)
            if not np.any(np.isnan(preds)):
                base_per_seed.append(np.sqrt(np.mean((preds - actual_arr)**2)))
    base_arr = np.array(base_per_seed)

    forced_means = np.array([np.mean(r["forced"]) if r["forced"] else np.nan for r in rows])
    forced_rmse_mc = np.sqrt(np.nanmean((forced_means - actual_arr)**2))
    if base_results:
        base_means = np.array([np.mean(r["base"]) if r["base"] else np.nan for r in rows])
        base_rmse_mc = np.sqrt(np.nanmean((base_means - actual_arr)**2))

    print("\n" + "=" * 90)
    print(f"{'Method':35s} {'mean-curve RMSE':>16s} {'per-seed RMSE':>22s}")
    print("-" * 90)
    if base_results:
        print(f"{'HCX cov_base (18, mild prompt)':35s} {base_rmse_mc:16.4f} "
              f"{f'{base_arr.mean():.4f}±{base_arr.std(ddof=1):.4f} (n={len(base_arr)})':>22s}")
    print(f"{'HCX cov_csi_forced (18+CSI, CoT)':35s} {forced_rmse_mc:16.4f} "
          f"{f'{forced_arr.mean():.4f}±{forced_arr.std(ddof=1):.4f} (n={len(forced_arr)})':>22s}")

    if base_results and len(base_arr) > 1 and len(forced_arr) > 1:
        from scipy import stats
        t, p = stats.ttest_ind(base_arr, forced_arr, equal_var=False)
        u, pu = stats.mannwhitneyu(base_arr, forced_arr, alternative="two-sided")
        print(f"\nWelch's t base vs forced: t={t:+.3f}, p={p:.4f}")
        print(f"Mann-Whitney: U={u:.0f}, p={pu:.4f}")
        diff = forced_arr.mean() - base_arr.mean()
        print(f"Δ (forced - base) = {diff:+.4f}pp")

    # Save
    out = {
        "targets": targets, "actuals": actuals,
        "forecasts": forecasts, "rationales_first_seed": {t: r[0] if r else "" for t, r in rationales.items()},
        "n_seeds": N_SEEDS, "temperature": TEMP, "ctx_len": CTX_LEN,
        "forced_per_seed_rmse": forced_arr.tolist(),
        "base_per_seed_rmse": base_arr.tolist() if len(base_arr) else [],
    }
    with open("data/rolling_2025_csi_forced_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved data/rolling_2025_csi_forced_results.json")


if __name__ == "__main__":
    main()
