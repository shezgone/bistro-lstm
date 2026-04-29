"""Rolling 1-step-ahead forecast for 2025-05 to 2025-12 (post-LLM-cutoff OOS).

Models compared:
- HCX-32B-Think +18cov (n_seeds × 8 origins)
- HCX-32B-Think univar (n_seeds × 8 origins)
- AR(1), Random Walk, Trend (deterministic baselines)
"""
import os, sys, json, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, numpy as np, pandas as pd

API_URL = "https://namc-aigw.io.naver.com/v1/chat/completions"
MODEL = "HyperCLOVAX-SEED-32B-Think-Text"
API_KEY = os.environ.get("HCX_API_KEY")
if not API_KEY: sys.exit("HCX_API_KEY missing")

CSV = os.path.join(os.path.dirname(__file__), "data", "macro_panel_optimal18.csv")
df = pd.read_csv(CSV, index_col=0)
df.index = pd.PeriodIndex(df.index, freq="M")

# Forecast origins: 2025-04 through 2025-11 (8 origins, predict next month each)
ORIGINS = pd.period_range("2025-04", "2025-11", freq="M")
N_SEEDS = 5
TEMP = 0.7
MAX_TOKENS = 8192
CTX_LEN = 36  # months

def build_messages(origin: pd.Period, mode: str):
    """Build prompt for 1-step-ahead forecast given data through origin month."""
    ctx_start = origin - CTX_LEN + 1
    target = origin + 1
    cols = ["CPI_KR_YoY"] if mode == "univar" else list(df.columns)
    ctx = df.loc[str(ctx_start):str(origin), cols].round(3)
    ctx.index = ctx.index.astype(str)
    table = ctx.to_csv(sep="\t")

    sys_msg = (
        "You are an expert macroeconomist forecasting Korean CPI YoY inflation. "
        f"You have monthly data through {origin}. "
        f"Forecast Korean CPI YoY (%) for the SINGLE NEXT month: {target}. "
        "Reason about base effects, BoK monetary lag, commodity/FX trends, "
        "global PPI signals, and immediate momentum. Return ONLY a JSON object: "
        '{"forecast": v, "rationale": "1-3 sentences"} '
        "where v is a single decimal number like 2.3 (no % sign)."
    )
    label = "Macro panel" if mode == "cov" else "Korean CPI YoY history"
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
        "temperature": TEMP, "max_tokens": MAX_TOKENS,
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

def baseline_forecasts(origin):
    """Compute deterministic baseline forecasts for 1-step-ahead from origin."""
    target = origin + 1
    hist = df.loc[:str(origin), "CPI_KR_YoY"]
    last_val = hist.iloc[-1]

    # 1. Random walk
    rw = last_val

    # 2. AR(1): fit on full history, predict
    from sklearn.linear_model import LinearRegression
    y = hist.values
    X = y[:-1].reshape(-1, 1)
    Y = y[1:]
    ar1 = LinearRegression().fit(X, Y)
    ar1_pred = float(ar1.predict([[last_val]])[0])

    # 3. Trend extrap: linear fit on last 12 months, project 1 step
    last12 = hist.iloc[-12:].values
    coef = np.polyfit(np.arange(12), last12, 1)
    trend_pred = float(np.polyval(coef, 12))

    # 4. Linear toward 2% BoK target (slow reversion)
    bok_pred = last_val + 0.5 * (2.0 - last_val) / 12  # 1/12 of the gap closed per month

    # 5. 12-month moving average
    ma12 = float(hist.iloc[-12:].mean())

    return {"RW": rw, "AR(1)": ar1_pred, "Trend12": trend_pred,
            "BoK linear": bok_pred, "MA12": ma12}

def main():
    print(f"Origins: {[str(o) for o in ORIGINS]}")
    print(f"Targets: {[str(o+1) for o in ORIGINS]}")
    actuals = {str(o+1): float(df.loc[o+1, "CPI_KR_YoY"]) for o in ORIGINS}
    print(f"Actuals: {actuals}")
    print()

    # Deterministic baselines (one per origin)
    baseline_results = {}
    for o in ORIGINS:
        baseline_results[str(o+1)] = baseline_forecasts(o)
    print("Deterministic baselines computed.")
    print()

    # HCX rolling calls
    hcx_jobs = []
    for o in ORIGINS:
        for mode in ["cov", "univar"]:
            for s in range(N_SEEDS):
                hcx_jobs.append((o, mode, s))
    print(f"HCX calls: {len(hcx_jobs)} ({len(ORIGINS)} origins × 2 modes × {N_SEEDS} seeds)")

    hcx_results = {}  # key: (origin_str, mode) -> [forecasts]
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(hcx_call, *j): j for j in hcx_jobs}
        done = 0
        for fut in as_completed(futures):
            o, mode, s = futures[fut]
            v, err = fut.result()
            if v is not None:
                hcx_results.setdefault((str(o+1), mode), []).append(v)
            done += 1
            print(f"[{done}/{len(hcx_jobs)}] {o+1} {mode} seed{s}: "
                  f"{'%.3f' % v if v is not None else f'FAIL ({err})'}", flush=True)

    # Compute per-target stats
    print("\n" + "=" * 100)
    print(f"{'Target':10s} {'Actual':>8s} | "
          f"{'HCX cov μ':>11s} {'σ':>6s} | {'HCX univ μ':>12s} {'σ':>6s} | "
          f"{'RW':>7s} {'AR(1)':>7s} {'Trend':>7s} {'BoK':>7s} {'MA12':>7s}")
    print("-" * 100)
    rows = []
    for o in ORIGINS:
        tgt = str(o+1)
        actual = actuals[tgt]
        cov = hcx_results.get((tgt, "cov"), [])
        uni = hcx_results.get((tgt, "univar"), [])
        b = baseline_results[tgt]
        cov_mu = np.mean(cov) if cov else float("nan")
        cov_sd = np.std(cov, ddof=1) if len(cov) > 1 else 0.0
        uni_mu = np.mean(uni) if uni else float("nan")
        uni_sd = np.std(uni, ddof=1) if len(uni) > 1 else 0.0
        print(f"{tgt:10s} {actual:8.3f} | {cov_mu:11.3f} {cov_sd:6.3f} | "
              f"{uni_mu:12.3f} {uni_sd:6.3f} | "
              f"{b['RW']:7.3f} {b['AR(1)']:7.3f} {b['Trend12']:7.3f} {b['BoK linear']:7.3f} {b['MA12']:7.3f}")
        rows.append({"target": tgt, "actual": actual,
                     "hcx_cov": cov, "hcx_univar": uni, **b})

    # Compute RMSE & MAE per model across 8 targets
    print("\n" + "=" * 100)
    print(f"{'Model':22s} {'mean RMSE':>10s} {'mean MAE':>10s} {'Notes':30s}")
    print("-" * 100)

    actual_arr = np.array([r["actual"] for r in rows])

    # HCX with seed averaging (mean per target, then RMSE)
    cov_means = np.array([np.mean(r["hcx_cov"]) if r["hcx_cov"] else np.nan for r in rows])
    uni_means = np.array([np.mean(r["hcx_univar"]) if r["hcx_univar"] else np.nan for r in rows])
    print(f"{'HCX cov (seed-mean)':22s} {np.sqrt(np.mean((cov_means-actual_arr)**2)):10.4f} "
          f"{np.mean(np.abs(cov_means-actual_arr)):10.4f}   N={N_SEEDS} per origin")
    print(f"{'HCX univar (seed-mean)':22s} {np.sqrt(np.mean((uni_means-actual_arr)**2)):10.4f} "
          f"{np.mean(np.abs(uni_means-actual_arr)):10.4f}   N={N_SEEDS} per origin")

    # Per-seed HCX RMSE distribution (mean of per-seed RMSEs)
    cov_per_seed = np.array([
        [r["hcx_cov"][s] if s < len(r["hcx_cov"]) else np.nan for s in range(N_SEEDS)]
        for r in rows
    ])  # (8, N_SEEDS)
    uni_per_seed = np.array([
        [r["hcx_univar"][s] if s < len(r["hcx_univar"]) else np.nan for s in range(N_SEEDS)]
        for r in rows
    ])
    cov_seed_rmses = np.sqrt(np.nanmean((cov_per_seed - actual_arr[:, None])**2, axis=0))
    uni_seed_rmses = np.sqrt(np.nanmean((uni_per_seed - actual_arr[:, None])**2, axis=0))
    print(f"{'  per-seed RMSE cov':22s} mean={np.mean(cov_seed_rmses):.4f} "
          f"std={np.std(cov_seed_rmses, ddof=1):.4f} min={np.min(cov_seed_rmses):.4f} "
          f"max={np.max(cov_seed_rmses):.4f}")
    print(f"{'  per-seed RMSE univ':22s} mean={np.mean(uni_seed_rmses):.4f} "
          f"std={np.std(uni_seed_rmses, ddof=1):.4f} min={np.min(uni_seed_rmses):.4f} "
          f"max={np.max(uni_seed_rmses):.4f}")

    # Baselines
    for name in ["RW", "AR(1)", "Trend12", "BoK linear", "MA12"]:
        preds = np.array([r[name] for r in rows])
        rmse = np.sqrt(np.mean((preds - actual_arr) ** 2))
        mae = np.mean(np.abs(preds - actual_arr))
        print(f"{name:22s} {rmse:10.4f} {mae:10.4f}   deterministic")

    # Save raw
    out = {
        "origins": [str(o) for o in ORIGINS],
        "targets": [str(o+1) for o in ORIGINS],
        "actuals": actuals,
        "hcx_results": {f"{k[0]}|{k[1]}": v for k, v in hcx_results.items()},
        "baselines": baseline_results,
        "n_seeds": N_SEEDS, "temperature": TEMP, "ctx_len": CTX_LEN,
    }
    with open("data/rolling_2025_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved data/rolling_2025_results.json")

if __name__ == "__main__":
    main()
