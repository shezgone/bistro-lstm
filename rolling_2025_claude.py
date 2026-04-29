"""Rolling 1-step-ahead forecast for 2025-05 to 2025-12 with Claude models.

Identical protocol to rolling_2025.py (HCX) for direct comparison.
- claude-opus-4-7    : 2026-01 cutoff → 2025 data IS in training (contaminated)
- claude-3-5-sonnet  : 2024-04 cutoff → 2025 data NOT in training (clean OOS)
"""
import os, sys, json, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd
import anthropic

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY: sys.exit("ANTHROPIC_API_KEY missing")

CSV = os.path.join(os.path.dirname(__file__), "data", "macro_panel_optimal18.csv")
df = pd.read_csv(CSV, index_col=0)
df.index = pd.PeriodIndex(df.index, freq="M")

ORIGINS = pd.period_range("2025-04", "2025-11", freq="M")
N_SEEDS = 5
TEMP = 0.7
CTX_LEN = 36

MODELS = [
    ("claude-opus-4-7",   "Opus 4.7 (cutoff 2026-01, fully contaminated)"),
    ("claude-opus-4-5",   "Opus 4.5 (cutoff early 2025, partial OOS)"),
    ("claude-sonnet-4-5", "Sonnet 4.5 (cutoff 2024-04, FULLY CLEAN OOS)"),
]

def build_messages(origin: pd.Period, mode: str):
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

client = anthropic.Anthropic(api_key=API_KEY, max_retries=2)

def claude_call(model_id, origin, mode, seed_idx):
    sys_msg, user_msg = build_messages(origin, mode)
    # Newer Anthropic models (Opus 4.7+) deprecated temperature; use only on legacy models
    use_temp = "opus-4-7" not in model_id
    for retry in range(3):
        try:
            kwargs = {
                "model": model_id,
                "max_tokens": 2048,
                "system": sys_msg,
                "messages": [{"role": "user", "content": user_msg}],
            }
            if use_temp:
                kwargs["temperature"] = TEMP
            resp = client.messages.create(**kwargs)
            content = resp.content[0].text if resp.content else ""
            m = re.search(r"\{[\s\S]*\}", content)
            if not m:
                if retry < 2: time.sleep(2 ** retry); continue
                return None, "no JSON"
            obj = json.loads(m.group(0))
            v = float(obj["forecast"]) if not isinstance(obj["forecast"], list) else float(obj["forecast"][0])
            return v, None
        except Exception as e:
            if retry < 2: time.sleep(2 ** retry); continue
            return None, str(e)[:120]
    return None, "max retries"

def main():
    targets = [str(o + 1) for o in ORIGINS]
    actuals = {t: float(df.loc[t, "CPI_KR_YoY"]) for t in targets}
    actual_arr = np.array([actuals[t] for t in targets])
    print(f"Targets: {targets}")
    print(f"Actuals: {[round(a, 3) for a in actual_arr]}")

    jobs = []
    for model_id, _ in MODELS:
        for o in ORIGINS:
            for mode in ["cov", "univar"]:
                for s in range(N_SEEDS):
                    jobs.append((model_id, o, mode, s))
    print(f"Total calls: {len(jobs)} ({len(MODELS)} models × {len(ORIGINS)} origins × 2 modes × {N_SEEDS} seeds)\n", flush=True)

    results = {}  # (model, target_str, mode) -> [forecasts]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(claude_call, *j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            model_id, o, mode, s = futures[fut]
            v, err = fut.result()
            tgt = str(o + 1)
            if v is not None:
                results.setdefault((model_id, tgt, mode), []).append(v)
            done += 1
            short_model = {"claude-opus-4-7": "Opus47", "claude-opus-4-5": "Opus45",
                           "claude-sonnet-4-5": "Sonnet45"}.get(model_id, model_id[-10:])
            print(f"[{done}/{len(jobs)}] {short_model} {tgt} {mode} s{s}: "
                  f"{'%.3f' % v if v is not None else f'FAIL ({err})'}", flush=True)
    print(f"\nElapsed: {time.time()-t0:.1f}s")

    # Compute stats per model
    print("\n" + "=" * 110)
    for model_id, model_label in MODELS:
        print(f"\n{model_label} ({model_id})")
        print(f"{'Target':10s} {'Actual':>8s} | {'cov μ':>8s} {'σ':>6s} {'n':>3s} | {'univar μ':>10s} {'σ':>6s} {'n':>3s}")
        print("-" * 70)
        for tgt in targets:
            cov = results.get((model_id, tgt, "cov"), [])
            uni = results.get((model_id, tgt, "univar"), [])
            cov_mu = np.mean(cov) if cov else float("nan")
            cov_sd = np.std(cov, ddof=1) if len(cov) > 1 else 0.0
            uni_mu = np.mean(uni) if uni else float("nan")
            uni_sd = np.std(uni, ddof=1) if len(uni) > 1 else 0.0
            print(f"{tgt:10s} {actuals[tgt]:8.3f} | {cov_mu:8.3f} {cov_sd:6.3f} {len(cov):3d} | "
                  f"{uni_mu:10.3f} {uni_sd:6.3f} {len(uni):3d}")

    # Aggregate RMSE
    print("\n" + "=" * 110)
    print(f"{'Model':35s} {'Mode':8s} {'mean RMSE':>10s} {'mean MAE':>10s} {'per-seed RMSE mean±std':>26s}")
    print("-" * 110)
    for model_id, model_label in MODELS:
        for mode in ["cov", "univar"]:
            seed_means = []
            per_seed = []
            for tgt in targets:
                fcs = results.get((model_id, tgt, mode), [])
                if fcs: seed_means.append(np.mean(fcs))
                else: seed_means.append(np.nan)
            seed_means = np.array(seed_means)
            rmse_meancurve = float(np.sqrt(np.nanmean((seed_means - actual_arr) ** 2)))
            mae_meancurve = float(np.nanmean(np.abs(seed_means - actual_arr)))
            # per-seed RMSEs
            for s in range(N_SEEDS):
                preds = []
                for tgt in targets:
                    fcs = results.get((model_id, tgt, mode), [])
                    preds.append(fcs[s] if s < len(fcs) else np.nan)
                preds = np.array(preds)
                if not np.any(np.isnan(preds)):
                    per_seed.append(float(np.sqrt(np.mean((preds - actual_arr) ** 2))))
            per_seed = np.array(per_seed)
            ps_str = (f"{per_seed.mean():.4f}±{per_seed.std(ddof=1):.4f} (n={len(per_seed)})"
                      if len(per_seed) > 1 else "n/a")
            short = {"claude-opus-4-7": "Opus 4.7 (contaminated)",
                     "claude-opus-4-5": "Opus 4.5 (partial)",
                     "claude-sonnet-4-5": "Sonnet 4.5 (clean OOS)"}.get(model_id, model_id)
            print(f"{short:35s} {mode:8s} {rmse_meancurve:10.4f} {mae_meancurve:10.4f} {ps_str:>26s}")

    # Save raw
    out = {
        "targets": targets,
        "actuals": actuals,
        "results": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in results.items()},
        "n_seeds": N_SEEDS, "temperature": TEMP, "ctx_len": CTX_LEN,
    }
    with open("data/rolling_2025_claude_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved data/rolling_2025_claude_results.json")

if __name__ == "__main__":
    main()
