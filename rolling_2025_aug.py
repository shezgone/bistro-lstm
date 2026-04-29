"""HCX rolling 2025 OOS: cov_base (18) vs cov_aug (18 + 5 Google Trends).

Direct ablation of "Korean macro sentiment proxy" effect on LLM forecast quality.
"""
import os, sys, json, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, numpy as np, pandas as pd

API_URL = "https://namc-aigw.io.naver.com/v1/chat/completions"
MODEL = "HyperCLOVAX-SEED-32B-Think-Text"
API_KEY = os.environ.get("HCX_API_KEY")
if not API_KEY: sys.exit("HCX_API_KEY missing")

CSV_BASE = "data/macro_panel_optimal18.csv"
CSV_AUG = "data/macro_panel_aug.csv"

df_base = pd.read_csv(CSV_BASE, index_col=0); df_base.index = pd.PeriodIndex(df_base.index, freq="M")
df_aug  = pd.read_csv(CSV_AUG,  index_col=0); df_aug.index  = pd.PeriodIndex(df_aug.index,  freq="M")

ORIGINS = pd.period_range("2025-04", "2025-11", freq="M")
N_SEEDS = 5
TEMP = 0.7
CTX_LEN = 36

TREND_COLS = ["GT_금리", "GT_물가", "GT_인플레이션", "GT_한국은행", "GT_디플레이션"]


def build_messages(origin: pd.Period, mode: str):
    ctx_start = origin - CTX_LEN + 1
    target = origin + 1

    if mode == "cov_base":
        df = df_base
        cols = list(df.columns)  # CPI + 18
        label = "Macro panel (18 covariates)"
    elif mode == "cov_aug":
        df = df_aug
        cols = list(df.columns)  # CPI + 18 + 5 trends
        label = "Macro panel (18 covariates + 5 Google Trends KR)"
    else:
        raise ValueError(mode)

    ctx = df.loc[str(ctx_start):str(origin), cols].round(3)
    ctx.index = ctx.index.astype(str)
    table = ctx.to_csv(sep="\t")

    sys_msg = (
        "You are an expert macroeconomist forecasting Korean CPI YoY inflation. "
        f"You have monthly data through {origin}. "
        f"Forecast Korean CPI YoY (%) for the SINGLE NEXT month: {target}. "
        "Reason about base effects, BoK monetary lag, commodity/FX trends, "
        "global PPI signals, and immediate momentum. "
    )
    if mode == "cov_aug":
        sys_msg += (
            "Note: GT_X columns are Google Trends Korean search interest (0-100 scale). "
            "Spikes indicate public attention to inflation/rate topics — useful as sentiment proxy. "
        )
    sys_msg += (
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
        for mode in ["cov_base", "cov_aug"]:
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
    print(f"{'Target':10s} {'Actual':>8s} | {'base μ':>8s} {'σ':>6s} | {'aug μ':>8s} {'σ':>6s} | {'Δ aug-base':>10s}")
    print("-" * 90)
    rows = []
    for tgt in targets:
        actual = actuals[tgt]
        base = results.get((tgt, "cov_base"), [])
        aug = results.get((tgt, "cov_aug"), [])
        bm = np.mean(base) if base else float("nan")
        bs = np.std(base, ddof=1) if len(base) > 1 else 0.0
        am = np.mean(aug) if aug else float("nan")
        asd = np.std(aug, ddof=1) if len(aug) > 1 else 0.0
        delta = am - bm
        print(f"{tgt:10s} {actual:8.3f} | {bm:8.3f} {bs:6.3f} | {am:8.3f} {asd:6.3f} | {delta:+10.3f}")
        rows.append({"target": tgt, "actual": actual, "base": base, "aug": aug})

    # Per-seed RMSE distribution
    base_per_seed_rmse = []
    aug_per_seed_rmse = []
    for s in range(N_SEEDS):
        base_preds = []; aug_preds = []
        for r in rows:
            base_preds.append(r["base"][s] if s < len(r["base"]) else np.nan)
            aug_preds.append(r["aug"][s] if s < len(r["aug"]) else np.nan)
        base_preds = np.array(base_preds); aug_preds = np.array(aug_preds)
        if not np.any(np.isnan(base_preds)):
            base_per_seed_rmse.append(np.sqrt(np.mean((base_preds - actual_arr)**2)))
        if not np.any(np.isnan(aug_preds)):
            aug_per_seed_rmse.append(np.sqrt(np.mean((aug_preds - actual_arr)**2)))

    base_arr = np.array(base_per_seed_rmse)
    aug_arr = np.array(aug_per_seed_rmse)

    # Seed-mean per target → RMSE
    base_means = np.array([np.mean(r["base"]) if r["base"] else np.nan for r in rows])
    aug_means = np.array([np.mean(r["aug"]) if r["aug"] else np.nan for r in rows])
    base_rmse_mc = np.sqrt(np.nanmean((base_means - actual_arr)**2))
    aug_rmse_mc = np.sqrt(np.nanmean((aug_means - actual_arr)**2))

    print("\n" + "=" * 90)
    print(f"{'Method':30s} {'mean-curve RMSE':>16s} {'per-seed RMSE':>22s}")
    print("-" * 90)
    print(f"{'HCX cov_base (18)':30s} {base_rmse_mc:16.4f} "
          f"{f'{base_arr.mean():.4f}±{base_arr.std(ddof=1):.4f} (n={len(base_arr)})':>22s}")
    print(f"{'HCX cov_aug (18+5GT)':30s} {aug_rmse_mc:16.4f} "
          f"{f'{aug_arr.mean():.4f}±{aug_arr.std(ddof=1):.4f} (n={len(aug_arr)})':>22s}")

    # Welch's t-test
    if len(base_arr) > 1 and len(aug_arr) > 1:
        from scipy import stats
        t, p = stats.ttest_ind(base_arr, aug_arr, equal_var=False)
        u, pu = stats.mannwhitneyu(base_arr, aug_arr, alternative="two-sided")
        print(f"\nWelch's t-test base vs aug per-seed RMSE: t={t:+.3f}, p={p:.4f}")
        print(f"Mann-Whitney U: U={u:.0f}, p={pu:.4f}")
        diff = aug_arr.mean() - base_arr.mean()
        print(f"Δ (aug - base) = {diff:+.4f}pp ({'aug 더 좋음' if diff < 0 else 'base 더 좋음' if diff > 0 else '동일'})")

    # Save
    out = {
        "targets": targets, "actuals": actuals,
        "results": {f"{k[0]}|{k[1]}": v for k, v in results.items()},
        "n_seeds": N_SEEDS, "temperature": TEMP, "ctx_len": CTX_LEN,
        "base_per_seed_rmse": base_arr.tolist(),
        "aug_per_seed_rmse": aug_arr.tolist(),
    }
    with open("data/rolling_2025_aug_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved data/rolling_2025_aug_results.json")


if __name__ == "__main__":
    main()
