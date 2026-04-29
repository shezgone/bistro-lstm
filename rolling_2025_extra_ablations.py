"""HCX rolling 2025 — extra ablations (literature-grounded).

Modes:
1. wrong_cpi: contamination probe (Golchin & Surdeanu ICLR 2024)
   Inject FAKE recent CPI into context. If HCX anchors to truth, it's contaminated.
   Real 2025-04 CPI = 2.079; we replace with 4.5. Track if forecast moves with the fake.

2. bias_feedback: in-prompt bias correction (TTA, Gibbs & Candes 2021)
   Append "Recent forecast errors" section to system msg using prior origin's mean forecasts.

3. narrative: sentiment as text (Thorsrud JBES 2020 spirit)
   Drop BoK_CSI numeric column; replace with natural language summary in system msg.
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

# Load prior forced CoT mean forecasts for bias_feedback
with open("data/rolling_2025_csi_forced_results.json") as f:
    prior = json.load(f)
prior_mean_forecasts = {tgt: float(np.mean(fcs)) for tgt, fcs in prior["forecasts"].items()}

ORIGINS = pd.period_range("2025-04", "2025-11", freq="M")
N_SEEDS = 5
TEMP = 0.7
CTX_LEN = 36

BASE_OPENING = (
    "You are an expert macroeconomist forecasting Korean CPI YoY inflation. "
    "You have monthly data through {origin}. "
    "Forecast Korean CPI YoY (%) for the SINGLE NEXT month: {target}.\n\n"
)

CSI_NOTE = (
    "BoK_CSI = Korean Consumer Sentiment Index (100=neutral, >100=optimistic). "
    "Survey-based forward-looking sentiment.\n\n"
)

BASE_STEPS = (
    "Required reasoning steps:\n"
    "Step 1: Describe BoK_CSI trend over last 6 months (cite values).\n"
    "Step 2: Interpret sentiment direction → inflation pressure.\n"
    "Step 3: Cross-check with macro covariates.\n"
    "Step 4: Combine into single point forecast.\n\n"
)

BASE_CLOSE = (
    "Return ONLY: {\"forecast\": v, \"rationale\": \"...\"} where v is decimal "
    "(no % sign). Rationale must cite BoK_CSI values."
)


def panel_tsv(df_slice):
    df_slice = df_slice.round(3).copy()
    df_slice.index = df_slice.index.astype(str)
    return df_slice.to_csv(sep="\t")


def build_messages(origin: pd.Period, mode: str):
    target = origin + 1
    ctx_start = origin - CTX_LEN + 1

    df = df_csi.copy()  # Will modify per mode
    panel = df.loc[str(ctx_start):str(origin), list(df.columns)]
    sys_extra = ""
    user_label = ""

    if mode == "wrong_cpi":
        # Inject fake high CPI for last 3 months: 4.5, 4.5, 4.5 (real values are ~2.0)
        # If HCX anchors to truth, ignores fake → it's contaminated/memorized
        # If HCX trusts data (not contaminated), forecast should follow fake → very high
        panel = panel.copy()
        for k in range(3, 0, -1):
            row_date = str(origin - k + 1)
            if row_date in panel.index.astype(str):
                idx = panel.index.astype(str).tolist().index(row_date)
                panel.iloc[idx, panel.columns.get_loc("CPI_KR_YoY")] = 4.5
        sys_extra = ""  # No mention of injection
        user_label = " (with recent CPI showing surge)"

    elif mode == "bias_feedback":
        # Append last 3 origins' forecast errors to system msg
        recent_errors = []
        for k in range(3, 0, -1):
            ex_origin = origin - k
            ex_target = ex_origin + 1
            tgt_str = str(ex_target)
            if tgt_str in prior_mean_forecasts:
                pred = prior_mean_forecasts[tgt_str]
                actual = float(df_csi.loc[ex_target, "CPI_KR_YoY"])
                err = pred - actual  # over-forecast = positive
                recent_errors.append(f"  Forecast for {tgt_str}: predicted {pred:.2f}, actual {actual:.2f}, error {err:+.2f}")
        if recent_errors:
            sys_extra = (
                "\nIMPORTANT — recent forecast track record (use for self-calibration):\n"
                + "\n".join(recent_errors)
                + "\n\nIf these show consistent over-forecasting, your forecast may need "
                  "downward bias adjustment; if under-forecasting, upward. Account for this "
                  "in your Step 4 final answer.\n"
            )

    elif mode == "narrative":
        # Drop BoK_CSI from numeric panel, add text summary
        csi_series = df_csi.loc[str(ctx_start):str(origin), "BoK_CSI"]
        # Generate a brief narrative
        recent6 = csi_series.iloc[-6:]
        first = recent6.iloc[0]; last = recent6.iloc[-1]; current = csi_series.iloc[-1]
        delta = last - first
        direction = "rose" if delta > 1 else ("fell" if delta < -1 else "held steady")
        level_descr = (
            "strongly optimistic" if current > 110 else
            "modestly optimistic" if current > 100 else
            "neutral" if current >= 95 else
            "modestly pessimistic" if current >= 85 else
            "strongly pessimistic"
        )
        sys_extra = (
            f"\nKorean Consumer Sentiment summary: Over the last 6 months (ending {origin}), "
            f"BoK CSI {direction} by {abs(delta):.1f} points (from {first:.0f} to {last:.0f}), "
            f"currently {level_descr} ({current:.0f}, where 100 = neutral).\n"
        )
        # Drop CSI column from numeric panel
        panel = panel.drop(columns=["BoK_CSI"])

    else:
        raise ValueError(mode)

    table = panel_tsv(panel)
    sys_msg = BASE_OPENING.format(origin=origin, target=target) + CSI_NOTE + BASE_STEPS + sys_extra + BASE_CLOSE
    user_msg = (
        f"Macro panel{user_label}, {ctx_start} to {origin}, monthly TSV:\n{table}\n\n"
        f"Forecast CPI_KR_YoY for {target}. Output JSON only."
    )
    return sys_msg, user_msg


def hcx_call(origin, mode, seed_idx):
    sys_msg, user_msg = build_messages(origin, mode)
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
                return None, f"HTTP {r.status_code}"
            content = r.json()["choices"][0]["message"].get("content")
            if not content:
                if retry < 3: time.sleep(2 ** retry); continue
                return None, "empty"
            m = re.search(r"\{[\s\S]*\}", content)
            obj = json.loads(m.group(0))
            v = float(obj["forecast"]) if not isinstance(obj["forecast"], list) else float(obj["forecast"][0])
            return v, None
        except Exception as e:
            if retry < 3: time.sleep(2 ** retry); continue
            return None, str(e)[:100]
    return None, "max retries"


def main():
    targets = [str(o + 1) for o in ORIGINS]
    actuals = {t: float(df_csi.loc[t, "CPI_KR_YoY"]) for t in targets}
    actual_arr = np.array([actuals[t] for t in targets])

    modes = ["wrong_cpi", "bias_feedback", "narrative"]
    jobs = [(o, m, s) for m in modes for o in ORIGINS for s in range(N_SEEDS)]
    print(f"Total: {len(jobs)} ({len(modes)} modes × {len(ORIGINS)} origins × {N_SEEDS} seeds)\n", flush=True)

    # Print sample prompts to verify
    for m in modes:
        s, u = build_messages(ORIGINS[0], m)
        print(f"--- mode={m} sample at origin {ORIGINS[0]} ---")
        print(f"System extra (last 400 chars): ...{s[-400:]}")
        if m == "wrong_cpi":
            print(f"User msg first 200 chars: {u[:200]}")
        print()

    results = {m: {} for m in modes}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(hcx_call, *j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            o, mode, s = futures[fut]
            v, err = fut.result()
            tgt = str(o + 1)
            if v is not None:
                results[mode].setdefault(tgt, []).append(v)
            done += 1
            print(f"[{done}/{len(jobs)}] {mode:14s} {tgt} s{s}: "
                  f"{'%.3f' % v if v is not None else f'FAIL ({err})'}", flush=True)
    print(f"\nElapsed: {time.time()-t0:.1f}s")

    # Per-target summary
    print("\n" + "=" * 110)
    header = f"{'Target':10s} {'Actual':>8s} | "
    for m in modes: header += f"{m:>14s} {'σ':>6s} | "
    print(header)
    print("-" * 110)
    for tgt in targets:
        line = f"{tgt:10s} {actuals[tgt]:8.3f} | "
        for m in modes:
            fc = results[m].get(tgt, [])
            mu = np.mean(fc) if fc else float("nan")
            sd = np.std(fc, ddof=1) if len(fc) > 1 else 0.0
            line += f"{mu:14.3f} {sd:6.3f} | "
        print(line)

    print("\n" + "=" * 110)
    print(f"{'Method':30s} {'mean-curve RMSE':>16s} {'per-seed RMSE':>30s}")
    print("-" * 110)

    out = {"actuals": actuals, "modes": modes, "targets": targets,
           "n_seeds": N_SEEDS, "results": {}}
    for m in modes:
        seed_means = []
        for tgt in targets:
            fc = results[m].get(tgt, [])
            seed_means.append(np.mean(fc) if fc else np.nan)
        seed_means = np.array(seed_means)
        rmse_mc = float(np.sqrt(np.nanmean((seed_means - actual_arr)**2)))
        per_seed = []
        for s in range(N_SEEDS):
            preds = [results[m].get(tgt, [np.nan]*N_SEEDS)[s] if s < len(results[m].get(tgt, [])) else np.nan
                     for tgt in targets]
            preds = np.array(preds, dtype=float)
            if not np.any(np.isnan(preds)):
                per_seed.append(float(np.sqrt(np.mean((preds - actual_arr)**2))))
        ps = np.array(per_seed)
        ps_str = f"{ps.mean():.4f}±{ps.std(ddof=1):.4f} (n={len(ps)})" if len(ps) > 1 else "n/a"
        print(f"{m:30s} {rmse_mc:16.4f} {ps_str:>30s}")
        out["results"][m] = {"forecasts": results[m], "rmse_mean_curve": rmse_mc,
                              "per_seed_rmse": ps.tolist()}

    print(f"\n{'(prior) HCX forced CoT default':30s} {0.2504:16.4f} {'0.2633±0.0044 (n=5)':>30s}")
    print(f"{'(post-hoc) Ensemble 50/50':30s} {0.2348:16.4f} {'0.2348±0.0062 (n=5)':>30s}")
    print(f"{'(baseline) Trend12':30s} {0.2575:16.4f} {'(deterministic)':>30s}")

    # Stat tests vs prior forced
    try:
        from scipy import stats
        prior_per_seed = np.array(prior.get("forced_per_seed_rmse", []))
        print(f"\nWelch's t vs prior forced CoT (0.263±0.004):")
        for m in modes:
            ps = np.array(out["results"][m]["per_seed_rmse"])
            if len(ps) > 1 and len(prior_per_seed) > 1:
                t, p = stats.ttest_ind(ps, prior_per_seed, equal_var=False)
                d = ps.mean() - prior_per_seed.mean()
                print(f"  {m:14s}: t={t:+.3f} p={p:.4f} | Δ={d:+.4f}")
    except Exception as e:
        print(f"could not test: {e}")

    with open("data/rolling_2025_extra_ablations.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved data/rolling_2025_extra_ablations.json")


if __name__ == "__main__":
    main()
