"""HCX rolling 2025 BLINDED evaluation — strip identifying info to test contamination.

Mode E: full strip
- Column names: CPI_KR_YoY, BoK_CSI, ... → var_1, var_2, ..., var_20
- Dates: 2025-04 → t=33 (relative timestep within context window)
- Domain hints removed: no "Korean CPI", "BoK", "macroeconomist", "inflation"
  Just "time series forecaster"
- CoT keeps structure but generic — model must identify leading indicators itself
- Numeric values kept as-is (additional Mode F could normalize)

Comparison: HCX cov_csi_forced (labeled, RMSE 0.250) vs blinded.
- If RMSE ≈ 0.25: HCX uses pure pattern matching, contamination evidence weakens further
- If RMSE >> 0.30: labels carried significant signal (could be recall OR domain knowledge)
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
# 컬럼 순서: CPI_KR_YoY (target = var_1), then 18 covariates, then BoK_CSI = var_20

ORIGINS = pd.period_range("2025-04", "2025-11", freq="M")
N_SEEDS = 5
TEMP = 0.7
CTX_LEN = 36


def panel_blinded_tsv(df_slice):
    """Convert to blinded TSV: var_N column names, t=N row labels."""
    df = df_slice.round(3).copy()
    n_cols = len(df.columns)
    new_cols = [f"var_{i+1}" for i in range(n_cols)]
    df.columns = new_cols
    # Replace dates with sequential timestep
    df.index = [f"t={i+1}" for i in range(len(df))]
    df.index.name = "step"
    return df.to_csv(sep="\t")


def build_messages(origin: pd.Period):
    target = origin + 1
    ctx_start = origin - CTX_LEN + 1
    panel = df_csi.loc[str(ctx_start):str(origin), list(df_csi.columns)]
    table = panel_blinded_tsv(panel)
    n_steps = len(panel)
    target_step = n_steps + 1  # T+1

    # System message — generic, no domain hints
    sys_msg = (
        "You are an expert time series forecaster. You will receive a multivariate "
        f"panel of 20 variables (var_1 to var_20) observed over {n_steps} sequential "
        f"timesteps (t=1 to t={n_steps}). Forecast the value of var_1 (the target) "
        f"for the next single timestep t={target_step}.\n\n"
        "var_2 through var_20 are correlated covariates that may carry leading or "
        "lagging signals. You must IDENTIFY which covariates are most informative "
        "by examining their patterns yourself — no metadata about what they represent.\n\n"
        "Required reasoning steps (your rationale MUST cover all):\n"
        f"Step 1: Describe the trajectory of var_1 over the last 6 timesteps (cite values from t={n_steps-5} to t={n_steps}).\n"
        "Step 2: Identify 2-3 covariates (var_X) that appear to lead, lag, or co-move with var_1.\n"
        "  Cite specific values you observe.\n"
        f"Step 3: Based on those covariates' recent direction (last 3-6 timesteps), what does it imply for var_1 at t={target_step}?\n"
        "Step 4: Combine into a single point forecast for var_1.\n\n"
        "Return ONLY a JSON object: "
        '{"forecast": v, "rationale": "..."} '
        "where v is a single decimal number. "
        "Rationale must cite specific var_X values and timesteps you used."
    )
    user_msg = (
        f"Multivariate time series panel (t=1 to t={n_steps}, TSV format):\n"
        f"{table}\n\n"
        f"Forecast var_1 at t={target_step} (1-step-ahead). "
        "Output only the JSON specified in the system message."
    )
    return sys_msg, user_msg


def hcx_call(origin, seed_idx):
    sys_msg, user_msg = build_messages(origin)
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL,
               "messages": [{"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg}],
               "temperature": TEMP, "max_tokens": 8192}
    for retry in range(4):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=240)
            if r.status_code != 200:
                if retry < 3: time.sleep(2 ** retry); continue
                return None, None, f"HTTP {r.status_code}"
            content = r.json()["choices"][0]["message"].get("content")
            if not content:
                if retry < 3: time.sleep(2 ** retry); continue
                return None, None, "empty"
            m = re.search(r"\{[\s\S]*\}", content)
            obj = json.loads(m.group(0))
            v = float(obj["forecast"]) if not isinstance(obj["forecast"], list) else float(obj["forecast"][0])
            return v, obj.get("rationale", ""), None
        except Exception as e:
            if retry < 3: time.sleep(2 ** retry); continue
            return None, None, str(e)[:100]
    return None, None, "max retries"


def main():
    targets = [str(o + 1) for o in ORIGINS]
    actuals = {t: float(df_csi.loc[t, "CPI_KR_YoY"]) for t in targets}
    actual_arr = np.array([actuals[t] for t in targets])

    # Sample prompt preview
    s, u = build_messages(ORIGINS[0])
    print(f"=== Blinded prompt preview (origin {ORIGINS[0]}) ===")
    print(f"System ({len(s)} chars):")
    print(s[:1200])
    print(f"\nUser msg first 500 chars:")
    print(u[:500])
    print(f"\nUser msg last 200 chars:")
    print(u[-200:])
    print()

    jobs = [(o, s) for o in ORIGINS for s in range(N_SEEDS)]
    print(f"Total: {len(jobs)} calls (8 origins × {N_SEEDS} seeds)\n", flush=True)

    forecasts = {}
    rationales = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
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
            print(f"[{done}/{len(jobs)}] {tgt} s{s}: "
                  f"{'%.3f' % v if v is not None else f'FAIL ({err})'}", flush=True)
    print(f"\nElapsed: {time.time()-t0:.1f}s")

    # Print sample rationales — does the model identify CSI-like signal?
    print("\n" + "=" * 90)
    print("Sample rationales (does model identify which var is CSI-like?):")
    for tgt in targets:
        rats = rationales.get(tgt, [])
        if rats:
            print(f"\n[{tgt}] {rats[0][:500]}")

    # Stats
    print("\n" + "=" * 90)
    print(f"{'Target':10s} {'Actual':>8s} | {'blinded μ':>10s} {'σ':>6s}")
    print("-" * 60)
    rows = []
    for tgt in targets:
        actual = actuals[tgt]
        fc = forecasts.get(tgt, [])
        m = np.mean(fc) if fc else float("nan")
        s = np.std(fc, ddof=1) if len(fc) > 1 else 0.0
        print(f"{tgt:10s} {actual:8.3f} | {m:10.3f} {s:6.3f}")
        rows.append({"target": tgt, "actual": actual, "blinded": fc})

    blinded_means = np.array([np.mean(r["blinded"]) if r["blinded"] else np.nan for r in rows])
    blinded_rmse_mc = np.sqrt(np.nanmean((blinded_means - actual_arr)**2))

    blinded_per_seed = []
    for s in range(N_SEEDS):
        preds = [r["blinded"][s] if s < len(r["blinded"]) else np.nan for r in rows]
        preds = np.array(preds)
        if not np.any(np.isnan(preds)):
            blinded_per_seed.append(np.sqrt(np.mean((preds - actual_arr)**2)))
    ba = np.array(blinded_per_seed)

    print("\n" + "=" * 90)
    print(f"{'Method':40s} {'mean-curve RMSE':>16s} {'per-seed':>22s}")
    print("-" * 90)
    print(f"{'HCX BLINDED (Mode E full strip)':40s} {blinded_rmse_mc:16.4f} "
          f"{f'{ba.mean():.4f}±{ba.std(ddof=1):.4f} (n={len(ba)})':>22s}")
    print(f"{'(prior) HCX forced CoT labeled':40s} {0.2504:16.4f} {'0.2633±0.0044 (n=5)':>22s}")
    print(f"{'(baseline) Trend12':40s} {0.2575:16.4f} {'(deterministic)':>22s}")
    print(f"{'(baseline) Random Walk':40s} {0.2690:16.4f} {'(deterministic)':>22s}")

    # vs labeled forced CoT
    try:
        from scipy import stats
        with open("data/rolling_2025_csi_forced_results.json") as f:
            prior = json.load(f)
        prior_per_seed = np.array(prior.get("forced_per_seed_rmse", []))
        if len(ba) > 1 and len(prior_per_seed) > 1:
            t, p = stats.ttest_ind(ba, prior_per_seed, equal_var=False)
            u, pu = stats.mannwhitneyu(ba, prior_per_seed, alternative="two-sided")
            d = ba.mean() - prior_per_seed.mean()
            print(f"\nWelch's t blinded vs labeled forced: t={t:+.3f}, p={p:.4f}")
            print(f"Mann-Whitney U={u:.0f}, p={pu:.4f}")
            print(f"Δ (blinded - labeled) = {d:+.4f}pp")
            if abs(d) < 0.02:
                print("→ NO label effect: HCX uses pure pattern recognition. Strong contamination-free evidence.")
            elif d < 0.05:
                print("→ Small label effect: labels carry some signal (recall or domain knowledge), but blinded still competitive.")
            else:
                print("→ Large label effect: labels carry significant signal — could be recall OR legit domain knowledge.")
    except Exception as e:
        print(f"could not test: {e}")

    out = {
        "targets": targets, "actuals": actuals,
        "forecasts": forecasts,
        "rationales_first_seed": {t: r[0] if r else "" for t, r in rationales.items()},
        "n_seeds": N_SEEDS, "temperature": TEMP, "ctx_len": CTX_LEN,
        "blinded_per_seed_rmse": ba.tolist(),
    }
    with open("data/rolling_2025_blinded_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved data/rolling_2025_blinded_results.json")


if __name__ == "__main__":
    main()
