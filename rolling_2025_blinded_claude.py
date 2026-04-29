"""Claude (Opus 4.7 + Sonnet 4.5) BLINDED rolling 2025 — contamination evidence.

Key test: does Opus 4.7's spectacular labeled RMSE (0.168) drop to clean LLM
levels (~0.265) when labels and dates are stripped? If yes, that 0.10pp gap
was contamination/recall. If no, it's genuine reasoning.

Sonnet 4.5 serves as control: should stay near 0.265 (already clean).
"""
import os, sys, json, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd
import anthropic

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY: sys.exit("ANTHROPIC_API_KEY missing")

CSV_FULL = "data/macro_panel_full.csv"
df_full = pd.read_csv(CSV_FULL, index_col=0); df_full.index = pd.PeriodIndex(df_full.index, freq="M")
gt_cols = [c for c in df_full.columns if c.startswith("GT_")]
df_csi = df_full.drop(columns=gt_cols)

ORIGINS = pd.period_range("2025-04", "2025-11", freq="M")
N_SEEDS = 5
TEMP = 0.7
CTX_LEN = 36

MODELS = [
    ("claude-opus-4-7",   "Opus 4.7 (cutoff 2026-01)"),
    ("claude-sonnet-4-5", "Sonnet 4.5 (cutoff 2024-04, clean control)"),
]


def panel_blinded_tsv(df_slice):
    df = df_slice.round(3).copy()
    n_cols = len(df.columns)
    df.columns = [f"var_{i+1}" for i in range(n_cols)]
    df.index = [f"t={i+1}" for i in range(len(df))]
    df.index.name = "step"
    return df.to_csv(sep="\t")


def build_messages(origin: pd.Period):
    target = origin + 1
    ctx_start = origin - CTX_LEN + 1
    panel = df_csi.loc[str(ctx_start):str(origin), list(df_csi.columns)]
    table = panel_blinded_tsv(panel)
    n_steps = len(panel)
    target_step = n_steps + 1

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
        "Step 2: Identify 2-3 covariates (var_X) that appear to lead, lag, or co-move with var_1. "
        "Cite specific values you observe.\n"
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


client = anthropic.Anthropic(api_key=API_KEY, max_retries=2)


def claude_call(model_id, origin, seed_idx):
    sys_msg, user_msg = build_messages(origin)
    use_temp = "opus-4-7" not in model_id
    for retry in range(3):
        try:
            kwargs = {
                "model": model_id, "max_tokens": 2048,
                "system": sys_msg,
                "messages": [{"role": "user", "content": user_msg}],
            }
            if use_temp: kwargs["temperature"] = TEMP
            resp = client.messages.create(**kwargs)
            content = resp.content[0].text if resp.content else ""
            m = re.search(r"\{[\s\S]*\}", content)
            if not m:
                if retry < 2: time.sleep(2 ** retry); continue
                return None, None, "no JSON"
            obj = json.loads(m.group(0))
            v = float(obj["forecast"]) if not isinstance(obj["forecast"], list) else float(obj["forecast"][0])
            return v, obj.get("rationale", ""), None
        except Exception as e:
            if retry < 2: time.sleep(2 ** retry); continue
            return None, None, str(e)[:120]
    return None, None, "max retries"


def main():
    targets = [str(o + 1) for o in ORIGINS]
    actuals = {t: float(df_csi.loc[t, "CPI_KR_YoY"]) for t in targets}
    actual_arr = np.array([actuals[t] for t in targets])

    jobs = []
    for model_id, _ in MODELS:
        for o in ORIGINS:
            for s in range(N_SEEDS):
                jobs.append((model_id, o, s))
    print(f"Total: {len(jobs)} ({len(MODELS)} models × {len(ORIGINS)} origins × {N_SEEDS} seeds)\n", flush=True)

    results = {}  # (model_id, target) -> [forecasts]
    rationales = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(claude_call, *j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            model_id, o, s = futures[fut]
            v, rat, err = fut.result()
            tgt = str(o + 1)
            if v is not None:
                results.setdefault((model_id, tgt), []).append(v)
                if (model_id, tgt) not in rationales:
                    rationales[(model_id, tgt)] = rat
            done += 1
            short = "Opus47" if "opus" in model_id else "Sonnet45"
            print(f"[{done}/{len(jobs)}] {short} {tgt} s{s}: "
                  f"{'%.3f' % v if v is not None else f'FAIL ({err})'}", flush=True)
    print(f"\nElapsed: {time.time()-t0:.1f}s")

    # Per-model summary
    print("\n" + "=" * 100)
    out = {"actuals": actuals, "n_seeds": N_SEEDS, "results": {}}
    for model_id, model_label in MODELS:
        print(f"\n=== {model_label} ===")
        print(f"{'Target':10s} {'Actual':>8s} | {'blinded μ':>10s} {'σ':>6s}")
        print("-" * 60)
        rows = []
        for tgt in targets:
            actual = actuals[tgt]
            fc = results.get((model_id, tgt), [])
            m = np.mean(fc) if fc else float("nan")
            sd = np.std(fc, ddof=1) if len(fc) > 1 else 0.0
            print(f"{tgt:10s} {actual:8.3f} | {m:10.3f} {sd:6.3f}")
            rows.append({"target": tgt, "actual": actual, "blinded": fc})

        seed_means = np.array([np.mean(r["blinded"]) if r["blinded"] else np.nan for r in rows])
        rmse_mc = float(np.sqrt(np.nanmean((seed_means - actual_arr)**2)))

        per_seed = []
        for s in range(N_SEEDS):
            preds = [r["blinded"][s] if s < len(r["blinded"]) else np.nan for r in rows]
            preds = np.array(preds)
            if not np.any(np.isnan(preds)):
                per_seed.append(float(np.sqrt(np.mean((preds - actual_arr)**2))))
        ps = np.array(per_seed)
        ps_str = f"{ps.mean():.4f}±{ps.std(ddof=1):.4f} (n={len(ps)})" if len(ps) > 1 else "n/a"
        print(f"\n  mean-curve RMSE: {rmse_mc:.4f}  |  per-seed RMSE: {ps_str}")

        out["results"][model_id] = {
            "label": model_label,
            "forecasts": {tgt: results.get((model_id, tgt), []) for tgt in targets},
            "rationale_first_seed": {tgt: rationales.get((model_id, tgt), "") for tgt in targets},
            "rmse_mean_curve": rmse_mc,
            "per_seed_rmse": ps.tolist(),
        }

    # Comparison
    print("\n" + "=" * 100)
    print("CONTAMINATION TEST: blinded vs labeled (per model)")
    print("-" * 100)
    print(f"{'Model':35s} {'labeled RMSE':>14s} {'blinded RMSE':>14s} {'Δ (blinded-labeled)':>22s}")
    print("-" * 100)

    # Reference labeled RMSEs (from prior rolling_2025_claude_results.json)
    labeled_rmse = {
        "claude-opus-4-7":   0.184,  # cov mode mean-curve, prior result
        "claude-sonnet-4-5": 0.299,  # cov mode mean-curve
    }
    for model_id, model_label in MODELS:
        bl = out["results"][model_id]["rmse_mean_curve"]
        la = labeled_rmse.get(model_id, float("nan"))
        delta = bl - la
        sig = ""
        if delta > 0.05: sig = "→ labels carried significant signal (recall? or domain?)"
        elif delta > 0.02: sig = "→ small label effect"
        elif abs(delta) < 0.02: sig = "→ no label effect (pure pattern matching)"
        elif delta < -0.02: sig = "→ blinded BETTER (unusual!)"
        short = "Opus 4.7" if "opus" in model_id else "Sonnet 4.5 (control)"
        print(f"{short:35s} {la:14.4f} {bl:14.4f} {delta:+22.4f}  {sig}")

    # Sample rationales — does model identify what var_20 (CSI) is?
    print("\n" + "=" * 100)
    print("Sample rationales (does model identify which var is leading?):")
    for model_id, model_label in MODELS:
        print(f"\n--- {model_label} ---")
        for tgt in targets[:3]:
            rat = out["results"][model_id]["rationale_first_seed"].get(tgt, "")
            if rat:
                print(f"\n[{tgt}] {rat[:400]}")

    with open("data/rolling_2025_blinded_claude_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved data/rolling_2025_blinded_claude_results.json")


if __name__ == "__main__":
    main()
