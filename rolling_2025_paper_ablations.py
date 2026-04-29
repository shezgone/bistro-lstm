"""HCX ablations grounded in literature.

Modes (system msg variants only — Minimal Prompt Principle compliant where possible):

1. stepback: Step-Back Prompting (Zheng et al, ICLR 2024 "Take a Step Back")
   Add Step 0: state general macro inflation principles before applying.

2. critique: Reflexion-style self-critique (Shinn et al, NeurIPS 2023)
   Add Step 5: critically question the forecast direction.

3. csilag: MIDAS-style lagged sentiment (Bok et al ARE 2018, Thorsrud JBES 2020)
   Add CSI_t-3 and CSI_t-6 as separate columns. Tests if phase info helps.

All compared to forced CoT default (0.250).
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

# CSI lag panel: add CSI_t-3 and CSI_t-6 as separate columns
df_csilag = df_csi.copy()
df_csilag["BoK_CSI_lag3"] = df_csilag["BoK_CSI"].shift(3).fillna(100.0)
df_csilag["BoK_CSI_lag6"] = df_csilag["BoK_CSI"].shift(6).fillna(100.0)

ORIGINS = pd.period_range("2025-04", "2025-11", freq="M")
N_SEEDS = 5
TEMP = 0.7
CTX_LEN = 36

# Common base CoT (forced CoT, default best config)
BASE_PROMPT_OPENING = (
    "You are an expert macroeconomist forecasting Korean CPI YoY inflation. "
    "You have monthly data through {origin}. "
    "Forecast Korean CPI YoY (%) for the SINGLE NEXT month: {target}.\n\n"
    "CRITICAL — you MUST use BoK_CSI as a primary leading indicator:\n"
    "- BoK_CSI = Bank of Korea's monthly survey of Korean households about future\n"
    "  economic outlook (consumer sentiment about income, spending, prices).\n"
    "- 100 = neutral, >100 = optimistic, <100 = pessimistic.\n"
    "- Survey-based sentiment captures forward-looking inflation expectations\n"
    "  that are NOT visible in contemporaneous macro time series.\n\n"
)

BASE_STEPS = (
    "Required reasoning steps (your rationale MUST cover all):\n"
    "Step 1: Describe the recent BoK_CSI trend over the last 6 months\n"
    "  (cite specific values, e.g. '2025-04: 93.6 → 2025-08: 111.2').\n"
    "Step 2: Interpret this sentiment direction — does it imply rising or\n"
    "  falling demand-side inflation pressure? Why?\n"
    "Step 3: Cross-check with traditional macro covariates (BoK rate, commodity,\n"
    "  FX, base effect from one year ago).\n"
    "Step 4: Combine into a single point forecast and explain how Step 1-3 weighted.\n\n"
)

BASE_CLOSE = (
    "Return ONLY a JSON object:\n"
    '{"forecast": v, "rationale": "..."} '
    "where v is a single decimal number like 2.3 (no % sign). "
    "Rationale must explicitly state the BoK_CSI values used."
)


def build_messages(origin: pd.Period, mode: str):
    target = origin + 1
    ctx_start = origin - CTX_LEN + 1

    # Choose panel
    if mode == "csilag":
        df = df_csilag
        label_extra = " + CSI lag-3 + CSI lag-6"
    else:
        df = df_csi
        label_extra = ""

    cols = list(df.columns)
    panel = df.loc[str(ctx_start):str(origin), cols]
    panel = panel.round(3).copy()
    panel.index = panel.index.astype(str)
    table = panel.to_csv(sep="\t")

    # Build system msg per mode
    opening = BASE_PROMPT_OPENING.format(origin=origin, target=target)

    if mode == "stepback":
        # Zheng et al ICLR 2024 — Step-Back Prompting
        steps = (
            "Required reasoning steps (your rationale MUST cover all):\n"
            "Step 0 (STEP-BACK): First, state 2-3 GENERAL PRINCIPLES of how Korean CPI YoY\n"
            "  evolves month-to-month — what historically drives next-month change?\n"
            "  (e.g., base effects, BoK lag, sentiment-demand link). DO NOT use numbers\n"
            "  from the panel here, just principles.\n"
            "Step 1: Describe the recent BoK_CSI trend over the last 6 months\n"
            "  (cite specific values).\n"
            "Step 2: Apply the principles from Step 0 to this specific data —\n"
            "  what does the CSI direction imply?\n"
            "Step 3: Cross-check with macro covariates.\n"
            "Step 4: Combine into a single point forecast.\n\n"
        )
    elif mode == "critique":
        # Shinn et al NeurIPS 2023 — Reflexion / self-critique
        steps = BASE_STEPS + (
            "Step 5 (SELF-CRITIQUE): Now critically question your Step 4 forecast.\n"
            "  - Could you be over-anchoring on CSI? On recent CPI?\n"
            "  - What's a plausible scenario where you'd be off by 0.3-0.5pp?\n"
            "  - If YES, adjust the forecast accordingly. State your final number\n"
            "    AFTER critique.\n\n"
        )
    elif mode == "csilag":
        # MIDAS / Bok et al lagged sentiment
        steps = (
            "Required reasoning steps:\n"
            "Step 1: Describe BoK_CSI levels at three points: current month,\n"
            "  3 months ago (BoK_CSI_lag3), 6 months ago (BoK_CSI_lag6).\n"
            "Step 2: Sentiment LAGS often dominate CPI — interpret what 3-6 month-old\n"
            "  consumer optimism implies for current/next-month inflation\n"
            "  (mechanism: sentiment → spending → demand → CPI takes time).\n"
            "Step 3: Cross-check with macro covariates.\n"
            "Step 4: Combine into a single point forecast.\n\n"
        )
    else:
        raise ValueError(mode)

    sys_msg = opening + steps + BASE_CLOSE

    user_msg = (
        f"Macro panel (18 covariates + BoK CSI{label_extra}), {ctx_start} to {origin}, monthly TSV:\n"
        f"{table}\n\n"
        f"Forecast CPI_KR_YoY for {target} (single value, 1-step-ahead). "
        "Output only the JSON specified in the system message. "
        "Remember: rationale MUST cite specific BoK_CSI values from the data."
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

    modes = ["stepback", "critique", "csilag"]
    jobs = [(o, m, s) for m in modes for o in ORIGINS for s in range(N_SEEDS)]
    print(f"Total: {len(jobs)} ({len(modes)} modes × {len(ORIGINS)} origins × {N_SEEDS} seeds)\n", flush=True)

    results = {m: {} for m in modes}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(hcx_call, *j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            o, mode, s = futures[fut]
            v, err = fut.result()
            tgt = str(o + 1)
            if v is not None:
                results[mode].setdefault(tgt, []).append(v)
            done += 1
            print(f"[{done}/{len(jobs)}] {mode:8s} {tgt} s{s}: "
                  f"{'%.3f' % v if v is not None else f'FAIL ({err})'}", flush=True)
    print(f"\nElapsed: {time.time()-t0:.1f}s")

    # Per-target summary
    print("\n" + "=" * 110)
    header = f"{'Target':10s} {'Actual':>8s} | "
    for m in modes:
        header += f"{m+'μ':>11s} {'σ':>6s} | "
    print(header)
    print("-" * 110)
    for tgt in targets:
        line = f"{tgt:10s} {actuals[tgt]:8.3f} | "
        for m in modes:
            fc = results[m].get(tgt, [])
            mu = np.mean(fc) if fc else float("nan")
            sd = np.std(fc, ddof=1) if len(fc) > 1 else 0.0
            line += f"{mu:11.3f} {sd:6.3f} | "
        print(line)

    # Final RMSE table
    print("\n" + "=" * 110)
    print(f"{'Method':30s} {'mean-curve RMSE':>16s} {'per-seed RMSE mean±std (n)':>30s}")
    print("-" * 110)

    out = {"actuals": actuals, "modes": modes, "targets": targets,
           "n_seeds": N_SEEDS, "temperature": TEMP, "results": {}}

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

        out["results"][m] = {
            "forecasts": results[m],
            "rmse_mean_curve": rmse_mc,
            "per_seed_rmse": ps.tolist(),
        }

    print(f"\n{'(prior) HCX forced CoT default':30s} {0.2504:16.4f} {'0.2633±0.0044 (n=5)':>30s}")
    print(f"{'(baseline) Trend12':30s} {0.2575:16.4f} {'(deterministic)':>30s}")

    # Welch's t-test vs prior forced
    try:
        from scipy import stats
        with open("data/rolling_2025_csi_forced_results.json") as f:
            prior = json.load(f)
        prior_per_seed = np.array(prior.get("forced_per_seed_rmse", []))
        print(f"\nWelch's t vs prior forced CoT:")
        for m in modes:
            ps = np.array(out["results"][m]["per_seed_rmse"])
            if len(ps) > 1 and len(prior_per_seed) > 1:
                t, p = stats.ttest_ind(ps, prior_per_seed, equal_var=False)
                u, pu = stats.mannwhitneyu(ps, prior_per_seed, alternative="two-sided")
                d = ps.mean() - prior_per_seed.mean()
                print(f"  {m:10s}: t={t:+.3f} p_t={p:.4f} | U={u:.0f} p_u={pu:.4f} | Δ={d:+.4f}")
    except Exception as e:
        print(f"could not test: {e}")

    with open("data/rolling_2025_paper_ablations.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved data/rolling_2025_paper_ablations.json")


if __name__ == "__main__":
    main()
