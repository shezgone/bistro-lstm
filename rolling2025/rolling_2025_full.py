"""HCX rolling 2025 OOS: cov_base (18) vs cov_csi (18 + 1 BoK CSI).

Drop Google Trends (proven null effect) and isolate the BoK CSI contribution.
"""
import os, sys, json, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, numpy as np, pandas as pd

API_URL = "https://namc-aigw.io.naver.com/v1/chat/completions"
MODEL = "HyperCLOVAX-SEED-32B-Think-Text"
API_KEY = os.environ.get("HCX_API_KEY")
if not API_KEY: sys.exit("HCX_API_KEY missing")

CSV_BASE = "data/macro_panel_optimal18.csv"
CSV_FULL = "data/macro_panel_full.csv"   # 18 + 5 GT + 1 CSI = 24 covariates

df_base = pd.read_csv(CSV_BASE, index_col=0); df_base.index = pd.PeriodIndex(df_base.index, freq="M")
df_full = pd.read_csv(CSV_FULL, index_col=0); df_full.index = pd.PeriodIndex(df_full.index, freq="M")
# CSI-only mode: drop GT columns, keep 18 + BoK_CSI
gt_cols = [c for c in df_full.columns if c.startswith("GT_")]
df_csi = df_full.drop(columns=gt_cols)
print(f"CSI panel cols ({len(df_csi.columns)}): {list(df_csi.columns)}", flush=True)

ORIGINS = pd.period_range("2025-04", "2025-11", freq="M")
N_SEEDS = 5
TEMP = 0.7
CTX_LEN = 36


def build_messages(origin: pd.Period, mode: str):
    ctx_start = origin - CTX_LEN + 1
    target = origin + 1

    if mode == "cov_base":
        df = df_base
        label = "Macro panel (18 covariates)"
        sentiment_note = ""
    elif mode == "cov_csi":
        df = df_csi
        label = "Macro panel (18 covariates + BoK CSI)"
        sentiment_note = (
            "Note: BoK_CSI = Korean Consumer Sentiment Index (100=neutral, >100=optimistic, <100=pessimistic). "
            "This survey-based sentiment captures forward-looking inflation expectations not in the macro time series. "
        )
    else:
        raise ValueError(mode)

    cols = list(df.columns)
    ctx = df.loc[str(ctx_start):str(origin), cols].round(3)
    ctx.index = ctx.index.astype(str)
    table = ctx.to_csv(sep="\t")

    sys_msg = (
        "You are an expert macroeconomist forecasting Korean CPI YoY inflation. "
        f"You have monthly data through {origin}. "
        f"Forecast Korean CPI YoY (%) for the SINGLE NEXT month: {target}. "
        "Reason about base effects, BoK monetary lag, commodity/FX trends, "
        "global PPI signals, and immediate momentum. " + sentiment_note +
        "Return ONLY a JSON object: "
        '{"forecast": v, "rationale": "1-3 sentences"} '
        "where v is a single decimal number like 2.3 (no % sign)."
    )
    user_msg = (
        f"{label} ({ctx_start} to {origin}, monthly, TSV):\n{table}\n\n"
        f"Forecast CPI_KR_YoY for {target} (single value, 1-step-ahead). "
        "Output only the JSON specified in the system message."
    )
    return sys_msg, user_msg


def hcx_call(origin, mode, seed_idx):
    sys_msg, user_msg = build_messages(origin, mode)
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
    actuals = {t: float(df_base.loc[t, "CPI_KR_YoY"]) for t in targets}
    actual_arr = np.array([actuals[t] for t in targets])

    jobs = []
    for o in ORIGINS:
        for mode in ["cov_base", "cov_csi"]:
            for s in range(N_SEEDS):
                jobs.append((o, mode, s))
    print(f"Total: {len(jobs)} ({len(ORIGINS)} origins × 2 modes × {N_SEEDS} seeds)\n", flush=True)

    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(hcx_call, *j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            o, mode, s = futures[fut]
            v, err = fut.result()
            tgt = str(o + 1)
            if v is not None:
                results.setdefault((tgt, mode), []).append(v)
            done += 1
            print(f"[{done}/{len(jobs)}] {tgt} {mode} s{s}: "
                  f"{'%.3f' % v if v is not None else f'FAIL ({err})'}", flush=True)
    print(f"\nElapsed: {time.time()-t0:.1f}s")

    # Stats
    print("\n" + "=" * 90)
    print(f"{'Target':10s} {'Actual':>8s} | {'base μ':>8s} {'σ':>6s} | {'full μ':>8s} {'σ':>6s} | {'Δ full-base':>11s}")
    print("-" * 90)
    rows = []
    for tgt in targets:
        actual = actuals[tgt]
        base = results.get((tgt, "cov_base"), [])
        full = results.get((tgt, "cov_csi"), [])
        bm = np.mean(base) if base else float("nan")
        bs = np.std(base, ddof=1) if len(base) > 1 else 0.0
        fm = np.mean(full) if full else float("nan")
        fs = np.std(full, ddof=1) if len(full) > 1 else 0.0
        delta = fm - bm
        print(f"{tgt:10s} {actual:8.3f} | {bm:8.3f} {bs:6.3f} | {fm:8.3f} {fs:6.3f} | {delta:+11.3f}")
        rows.append({"target": tgt, "actual": actual, "base": base, "full": full})

    base_per_seed = []; full_per_seed = []
    for s in range(N_SEEDS):
        bp = []; fp = []
        for r in rows:
            bp.append(r["base"][s] if s < len(r["base"]) else np.nan)
            fp.append(r["full"][s] if s < len(r["full"]) else np.nan)
        bp = np.array(bp); fp = np.array(fp)
        if not np.any(np.isnan(bp)): base_per_seed.append(np.sqrt(np.mean((bp-actual_arr)**2)))
        if not np.any(np.isnan(fp)): full_per_seed.append(np.sqrt(np.mean((fp-actual_arr)**2)))
    base_arr = np.array(base_per_seed); full_arr = np.array(full_per_seed)

    base_means = np.array([np.mean(r["base"]) if r["base"] else np.nan for r in rows])
    full_means = np.array([np.mean(r["full"]) if r["full"] else np.nan for r in rows])
    base_rmse_mc = np.sqrt(np.nanmean((base_means - actual_arr)**2))
    full_rmse_mc = np.sqrt(np.nanmean((full_means - actual_arr)**2))

    print("\n" + "=" * 90)
    print(f"{'Method':30s} {'mean-curve RMSE':>16s} {'per-seed RMSE':>22s}")
    print("-" * 90)
    print(f"{'HCX cov_base (18)':30s} {base_rmse_mc:16.4f} "
          f"{f'{base_arr.mean():.4f}±{base_arr.std(ddof=1):.4f} (n={len(base_arr)})':>22s}")
    print(f"{'HCX cov_csi (18+1CSI)':30s} {full_rmse_mc:16.4f} "
          f"{f'{full_arr.mean():.4f}±{full_arr.std(ddof=1):.4f} (n={len(full_arr)})':>22s}")

    if len(base_arr) > 1 and len(full_arr) > 1:
        from scipy import stats
        t, p = stats.ttest_ind(base_arr, full_arr, equal_var=False)
        u, pu = stats.mannwhitneyu(base_arr, full_arr, alternative="two-sided")
        print(f"\nWelch's t-test base vs full per-seed RMSE: t={t:+.3f}, p={p:.4f}")
        print(f"Mann-Whitney U: U={u:.0f}, p={pu:.4f}")
        diff = full_arr.mean() - base_arr.mean()
        print(f"Δ (full - base) = {diff:+.4f}pp")

    out = {
        "targets": targets, "actuals": actuals,
        "results": {f"{k[0]}|{k[1]}": v for k, v in results.items()},
        "n_seeds": N_SEEDS, "temperature": TEMP, "ctx_len": CTX_LEN,
        "base_per_seed_rmse": base_arr.tolist(),
        "full_per_seed_rmse": full_arr.tolist(),
    }
    with open("data/rolling_2025_csi_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved data/rolling_2025_csi_results.json")


if __name__ == "__main__":
    main()
