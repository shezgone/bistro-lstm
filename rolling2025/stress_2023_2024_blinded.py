"""HCX blinded 12-step forecast for 2023 + 2024 (shock + transition periods).

Tests whether LLM "works in shock periods" claim survives blinding.

If HCX blinded 2023 RMSE ≈ HCX labeled 2023 (~0.66 N=8 mean) → genuine reasoning
If HCX blinded 2023 >> labeled (e.g., 0.9+) → labeled was contamination-aided
Either way, both should still vastly outperform Trend12 2023 (3.33 catastrophic) → regime-conditional value confirmed.
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
df_csi = df_full.drop(columns=gt_cols)  # CPI + 18 + BoK_CSI

# 12-step forecast configurations
CONFIGS = {
    2023: {"origin": pd.Period("2022-12", "M"), "ctx_len": 36, "pred_len": 12},
    2024: {"origin": pd.Period("2023-12", "M"), "ctx_len": 36, "pred_len": 12},
}
N_SEEDS = 5
TEMP = 0.7


def panel_blinded_tsv(df_slice):
    df = df_slice.round(3).copy()
    df.columns = [f"var_{i+1}" for i in range(len(df.columns))]
    df.index = [f"t={i+1}" for i in range(len(df))]
    df.index.name = "step"
    return df.to_csv(sep="\t")


def build_messages(year):
    cfg = CONFIGS[year]
    origin = cfg["origin"]
    ctx_len = cfg["ctx_len"]; pred_len = cfg["pred_len"]
    ctx_start = origin - ctx_len + 1
    panel = df_csi.loc[str(ctx_start):str(origin), list(df_csi.columns)]
    table = panel_blinded_tsv(panel)
    n_steps = len(panel)

    sys_msg = (
        "You are an expert time series forecaster. You will receive a multivariate "
        f"panel of 20 variables (var_1 to var_20) observed over {n_steps} sequential "
        f"timesteps (t=1 to t={n_steps}). Forecast the value of var_1 (the target) "
        f"for the NEXT {pred_len} timesteps (t={n_steps+1} to t={n_steps+pred_len}).\n\n"
        "var_2 through var_20 are correlated covariates that may carry leading or "
        "lagging signals. Identify which covariates are most informative by "
        "examining their patterns yourself — no metadata about what they represent.\n\n"
        "Required reasoning steps (rationale MUST cover):\n"
        f"Step 1: Describe var_1 trajectory over last 6-12 timesteps (cite values).\n"
        "Step 2: Identify 2-3 covariates with strongest leading/co-moving behavior.\n"
        f"Step 3: Project the {pred_len} future values considering momentum, base effects, and covariate signals.\n\n"
        "Return ONLY a JSON object: "
        '{"forecast": [v1, v2, ..., v12], "rationale": "..."} '
        f"with exactly {pred_len} decimal numbers (no % sign)."
    )
    user_msg = (
        f"Multivariate time series panel (t=1 to t={n_steps}, TSV):\n"
        f"{table}\n\n"
        f"Forecast var_1 for next {pred_len} timesteps (t={n_steps+1} to t={n_steps+pred_len}). "
        "Output JSON only."
    )
    return sys_msg, user_msg


def hcx_call(year, seed_idx):
    sys_msg, user_msg = build_messages(year)
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
            forecast = obj["forecast"]
            if not isinstance(forecast, list) or len(forecast) != 12:
                if retry < 3: time.sleep(2 ** retry); continue
                return None, None, f"bad shape: {len(forecast) if isinstance(forecast, list) else 'scalar'}"
            return [float(x) for x in forecast], obj.get("rationale", ""), None
        except Exception as e:
            if retry < 3: time.sleep(2 ** retry); continue
            return None, None, str(e)[:120]
    return None, None, "max retries"


def main():
    actuals = {y: df_csi.loc[f"{y}-01":f"{y}-12", "CPI_KR_YoY"].values for y in [2023, 2024]}

    jobs = [(y, s) for y in [2023, 2024] for s in range(N_SEEDS)]
    print(f"Total: {len(jobs)} calls (2 years × {N_SEEDS} seeds)\n", flush=True)

    forecasts = {2023: [], 2024: []}
    rationales = {2023: [], 2024: []}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(hcx_call, *j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            y, s = futures[fut]
            forecast, rat, err = fut.result()
            if forecast is not None:
                forecasts[y].append(forecast)
                rationales[y].append(rat)
            done += 1
            print(f"[{done}/{len(jobs)}] {y} s{s}: "
                  f"{'OK ' + str([round(x,2) for x in forecast]) if forecast else f'FAIL ({err})'}", flush=True)
    print(f"\nElapsed: {time.time()-t0:.1f}s")

    # Sample rationales
    print("\n" + "=" * 90)
    print("Sample rationale (first seed):")
    for y in [2023, 2024]:
        if rationales[y]:
            print(f"\n[{y}] {rationales[y][0][:600]}")

    # Per-month, per-year analysis
    print("\n" + "=" * 90)
    for y in [2023, 2024]:
        actual = actuals[y]
        fcs = np.array(forecasts[y])  # (n_seeds, 12)
        if len(fcs) == 0:
            print(f"\n=== {y} BLINDED — NO DATA ===")
            continue

        seed_mean = fcs.mean(axis=0)
        seed_std = fcs.std(axis=0, ddof=1) if len(fcs) > 1 else np.zeros(12)

        print(f"\n=== {y} BLINDED HCX (n={len(fcs)} seeds) ===")
        print(f"{'Month':6s} {'Actual':>8s} | {'pred μ':>8s} {'σ':>6s} | {'err':>7s}")
        print("-" * 55)
        for i in range(12):
            err = seed_mean[i] - actual[i]
            print(f"{i+1:>2d}     {actual[i]:8.3f} | {seed_mean[i]:8.3f} {seed_std[i]:6.3f} | {err:+7.3f}")

        # mean-curve and per-seed RMSE
        rmse_mc = float(np.sqrt(np.mean((seed_mean - actual) ** 2)))
        per_seed_rmse = [float(np.sqrt(np.mean((fcs[s] - actual)**2))) for s in range(len(fcs))]
        ps = np.array(per_seed_rmse)
        print(f"\n  mean-curve RMSE: {rmse_mc:.4f}")
        print(f"  per-seed RMSE: {ps.mean():.4f} ± {ps.std(ddof=1):.4f} [{ps.min():.4f}, {ps.max():.4f}]")

    # Comparison vs labeled (from prior N=8 ablation)
    print("\n" + "=" * 90)
    print("CONTAMINATION + REGIME TEST (labeled vs blinded, both 12-step)")
    print("-" * 90)
    print(f"{'Period':10s} {'Regime':18s} {'Labeled (N=8 mean)':>22s} {'Blinded':>14s} {'Trend12':>10s}")
    print("-" * 90)
    # Labeled N=8 means from earlier ablation:
    labeled_means = {2023: 0.79, 2024: 0.54}  # cov ON
    trend12_rmse = {2023: 3.33, 2024: 0.37}
    for y in [2023, 2024]:
        if not forecasts[y]: continue
        seed_mean = np.array(forecasts[y]).mean(axis=0)
        rmse_mc = float(np.sqrt(np.mean((seed_mean - actuals[y]) ** 2)))
        regime = "충격기 (5%→3%)" if y == 2023 else "단조 disinflation"
        print(f"{y:10d} {regime:18s} {labeled_means[y]:>22.3f} {rmse_mc:>14.3f} {trend12_rmse[y]:>10.3f}")

    out = {"forecasts": {str(y): forecasts[y] for y in [2023, 2024]},
           "actuals": {str(y): actuals[y].tolist() for y in [2023, 2024]},
           "rationales": {str(y): rationales[y] for y in [2023, 2024]},
           "n_seeds": N_SEEDS, "temperature": TEMP}
    with open("data/stress_2023_2024_blinded_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved data/stress_2023_2024_blinded_results.json")


if __name__ == "__main__":
    main()
