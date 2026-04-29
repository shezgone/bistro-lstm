"""Post-hoc analysis: aggregation methods on existing HCX forced CoT 5-seed data.

References:
- Wang et al. ICLR 2023 — Self-Consistency Improves CoT (median/majority voting)
- Trimmed mean — robust statistics, less sensitive to outlier seeds
- Bias correction — running residual subtraction (TTA literature)
"""
import json
import numpy as np

with open("data/rolling_2025_csi_forced_results.json") as f:
    d = json.load(f)

actuals = d["actuals"]
forecasts = d["forecasts"]   # {target: [seed0, seed1, ..., seed4]}
targets = sorted(forecasts.keys())
actual_arr = np.array([actuals[t] for t in targets])

print(f"=== Post-hoc aggregation on HCX forced CoT (5 seeds × 8 targets) ===\n")

# Show per-target seed distribution
print(f"{'Target':10s} {'Actual':>8s} | {'seeds':36s}")
print("-" * 75)
for tgt in targets:
    fcs = sorted(forecasts[tgt])
    print(f"{tgt:10s} {actuals[tgt]:8.3f} | {fcs}")
print()

def rmse(preds): return float(np.sqrt(np.mean((np.array(preds) - actual_arr)**2)))
def mae(preds):  return float(np.mean(np.abs(np.array(preds) - actual_arr)))

# 1. Mean aggregation (current default)
mean_preds = [np.mean(forecasts[t]) for t in targets]

# 2. Median (Wang ICLR 2023, robust to outliers)
median_preds = [np.median(forecasts[t]) for t in targets]

# 3. Trimmed mean (drop max+min, take mean of middle 3)
def trim_mean(xs, k=1):
    s = sorted(xs); return float(np.mean(s[k:-k]))
trimmed_preds = [trim_mean(forecasts[t], k=1) for t in targets]

# 4. Min (most optimistic seed)
min_preds = [min(forecasts[t]) for t in targets]
max_preds = [max(forecasts[t]) for t in targets]

# 5. Bias-corrected mean (subtract running mean error from prior 3 origins)
# For origin i (target t_{i+1}), bias = mean of (mean_pred[j] - actual[j]) for j in (i-3, i-2, i-1)
bias_corr_preds = []
for i in range(len(targets)):
    prior_errs = [(mean_preds[j] - actual_arr[j]) for j in range(max(0, i-3), i)]
    bias = float(np.mean(prior_errs)) if prior_errs else 0.0
    bias_corr_preds.append(mean_preds[i] - bias)

# 6. Mean-of-2-closest-to-median (consensus)
def consensus(xs):
    med = np.median(xs)
    s = sorted(xs, key=lambda x: abs(x - med))
    return float(np.mean(s[:3]))  # 3 closest to median
consensus_preds = [consensus(forecasts[t]) for t in targets]

# 7. Ensemble with Trend12 baseline
# Compute Trend12 for each origin
import pandas as pd
df = pd.read_csv('data/macro_panel_optimal18.csv', index_col=0)
df.index = pd.PeriodIndex(df.index, freq='M')
def trend12(origin_str):
    origin = pd.Period(origin_str, 'M')
    last12 = df.loc[:str(origin), 'CPI_KR_YoY'].iloc[-12:].values
    coef = np.polyfit(range(12), last12, 1)
    return float(np.polyval(coef, 12))

# Origin for target t is the prior month
trend12_preds = [trend12(str(pd.Period(t, 'M') - 1)) for t in targets]

# Ensemble: 50/50 HCX mean + Trend12
ensemble_50_preds = [0.5 * m + 0.5 * t for m, t in zip(mean_preds, trend12_preds)]
# Ensemble 70/30
ensemble_73_preds = [0.7 * m + 0.3 * t for m, t in zip(mean_preds, trend12_preds)]
# Ensemble 30/70 (more Trend12 weight)
ensemble_37_preds = [0.3 * m + 0.7 * t for m, t in zip(mean_preds, trend12_preds)]

# Print results table
print(f"\n{'Method':38s} {'RMSE':>8s} {'MAE':>8s}  {'Δ vs mean':>10s}")
print("-" * 75)
methods = [
    ("Mean (current default, baseline)", mean_preds),
    ("Median (Self-Consistency aggreg.)", median_preds),
    ("Trimmed mean (k=1, drop min+max)", trimmed_preds),
    ("Min seed (cherry-pick optimistic)", min_preds),
    ("Max seed (cherry-pick pessimistic)", max_preds),
    ("Consensus (3-closest-to-median)", consensus_preds),
    ("Bias-corrected mean (TTA-style)", bias_corr_preds),
    ("Trend12 baseline alone", trend12_preds),
    ("Ensemble HCX 50% + Trend12 50%", ensemble_50_preds),
    ("Ensemble HCX 70% + Trend12 30%", ensemble_73_preds),
    ("Ensemble HCX 30% + Trend12 70%", ensemble_37_preds),
]
ref_rmse = rmse(mean_preds)
for name, preds in methods:
    r = rmse(preds); m = mae(preds)
    d_mark = f"{r - ref_rmse:+.4f}" if name != "Mean (current default, baseline)" else "—"
    print(f"{name:38s} {r:8.4f} {m:8.4f}  {d_mark:>10s}")

# Save analysis
out = {
    "methods": {name: {"preds": list(p), "rmse": rmse(p), "mae": mae(p)}
                for name, p in methods},
    "actuals": dict(zip(targets, actual_arr.tolist())),
}
with open("data/posthoc_aggregation_results.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nSaved data/posthoc_aggregation_results.json")
