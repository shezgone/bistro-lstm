"""Per-seed ensemble analysis: each seed's HCX forecast averaged with Trend12.

Gives proper variance estimate for ensemble vs single methods.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

with open("data/rolling_2025_csi_forced_results.json") as f:
    d = json.load(f)
forecasts = d["forecasts"]
actuals = d["actuals"]
targets = sorted(forecasts.keys())
actual_arr = np.array([actuals[t] for t in targets])

# Compute Trend12 per origin
df = pd.read_csv('data/macro_panel_optimal18.csv', index_col=0)
df.index = pd.PeriodIndex(df.index, freq='M')
def trend12(origin_str):
    origin = pd.Period(origin_str, 'M')
    last12 = df.loc[:str(origin), 'CPI_KR_YoY'].iloc[-12:].values
    coef = np.polyfit(range(12), last12, 1)
    return float(np.polyval(coef, 12))
trend_preds = np.array([trend12(str(pd.Period(t, 'M') - 1)) for t in targets])

# Per-seed forecasts (5 seeds × 8 targets)
hcx_per_seed = np.array([[forecasts[t][s] for t in targets] for s in range(5)])  # (5, 8)

print(f"=== Per-seed ensemble analysis ===\n")

# Method per-seed RMSEs
def per_seed_rmses(predictions_5x8):
    return [float(np.sqrt(np.mean((predictions_5x8[s] - actual_arr)**2))) for s in range(5)]

# 1. HCX alone (5 seed RMSEs)
hcx_rmses = per_seed_rmses(hcx_per_seed)

# 2. Ensemble 50/50 per seed
ens50_per_seed = (0.5 * hcx_per_seed + 0.5 * trend_preds[None, :])
ens50_rmses = per_seed_rmses(ens50_per_seed)

# 3. Ensemble 70/30 per seed
ens73_per_seed = (0.7 * hcx_per_seed + 0.3 * trend_preds[None, :])
ens73_rmses = per_seed_rmses(ens73_per_seed)

# 4. Ensemble 30/70 per seed
ens37_per_seed = (0.3 * hcx_per_seed + 0.7 * trend_preds[None, :])
ens37_rmses = per_seed_rmses(ens37_per_seed)

# 5. Optimal weights — search per-seed
def optimal_weight(hcx_seed_preds, trend_preds, actuals, grid=np.arange(0, 1.05, 0.05)):
    best = (1e9, None)
    for w in grid:
        ens = w * hcx_seed_preds + (1-w) * trend_preds
        r = float(np.sqrt(np.mean((ens - actuals)**2)))
        if r < best[0]:
            best = (r, w)
    return best
opt_results = [optimal_weight(hcx_per_seed[s], trend_preds, actual_arr) for s in range(5)]
opt_rmses = [r for r, w in opt_results]
opt_weights = [w for r, w in opt_results]

# 6. Median ensemble (median of 3 methods: HCX-mean, Trend12, AR(1))
hcx_mean = np.mean(hcx_per_seed, axis=0)
# AR(1)
from sklearn.linear_model import LinearRegression
def ar1(origin_str):
    origin = pd.Period(origin_str, 'M')
    hist = df.loc[:str(origin), 'CPI_KR_YoY'].values
    X = hist[:-1].reshape(-1, 1); Y = hist[1:]
    m = LinearRegression().fit(X, Y)
    return float(m.predict([[hist[-1]]])[0])
ar1_preds = np.array([ar1(str(pd.Period(t, 'M') - 1)) for t in targets])
median_ensemble = np.median(np.stack([hcx_mean, trend_preds, ar1_preds]), axis=0)
median_ensemble_rmse = float(np.sqrt(np.mean((median_ensemble - actual_arr)**2)))

# Single-method reference RMSEs
trend12_rmse = float(np.sqrt(np.mean((trend_preds - actual_arr)**2)))
ar1_rmse = float(np.sqrt(np.mean((ar1_preds - actual_arr)**2)))

# Print
def stat_str(rmses): return f"{np.mean(rmses):.4f}±{np.std(rmses, ddof=1):.4f} [{min(rmses):.4f}, {max(rmses):.4f}]"

print(f"{'Method':40s} {'per-seed RMSE (mean±std [min, max])':>40s}")
print("-" * 90)
print(f"{'HCX forced CoT (mean per-seed)':40s} {stat_str(hcx_rmses):>40s}")
print(f"{'Trend12 (deterministic)':40s} {f'{trend12_rmse:.4f} (no variance)':>40s}")
print(f"{'AR(1) (deterministic)':40s} {f'{ar1_rmse:.4f} (no variance)':>40s}")
print(f"{'Median(HCX_mean, Trend12, AR1)':40s} {f'{median_ensemble_rmse:.4f} (deterministic)':>40s}")
print(f"{'Ensemble HCX 50% + Trend12 50%':40s} {stat_str(ens50_rmses):>40s}")
print(f"{'Ensemble HCX 70% + Trend12 30%':40s} {stat_str(ens73_rmses):>40s}")
print(f"{'Ensemble HCX 30% + Trend12 70%':40s} {stat_str(ens37_rmses):>40s}")
print(f"{'Optimal w per seed (oracle)':40s} {stat_str(opt_rmses):>40s}")
print(f"  optimal weights per seed: {[f'{w:.2f}' for w in opt_weights]}")

# Statistical tests
print("\n" + "=" * 90)
print("Welch's t-test: Ensemble 50/50 per-seed vs HCX alone per-seed")
t, p = stats.ttest_ind(ens50_rmses, hcx_rmses, equal_var=False)
print(f"  t = {t:+.3f}, p = {p:.4f}")
print(f"  Δ (ens50 - HCX) = {np.mean(ens50_rmses) - np.mean(hcx_rmses):+.4f}")
u, pu = stats.mannwhitneyu(ens50_rmses, hcx_rmses, alternative="two-sided")
print(f"  Mann-Whitney U = {u:.0f}, p = {pu:.4f}")

# Vs Trend12 (one-sample, since Trend12 is deterministic)
print("\nOne-sample t-test: Ensemble 50/50 per-seed vs Trend12 deterministic")
t1, p1 = stats.ttest_1samp(ens50_rmses, trend12_rmse)
print(f"  t = {t1:+.3f}, p = {p1:.4f}")
print(f"  ens50 mean = {np.mean(ens50_rmses):.4f}, Trend12 = {trend12_rmse:.4f}")

# Save
out = {
    "hcx_per_seed_rmses": hcx_rmses,
    "trend12_rmse": trend12_rmse,
    "ar1_rmse": ar1_rmse,
    "ensemble_50_per_seed_rmses": ens50_rmses,
    "ensemble_70_per_seed_rmses": ens73_rmses,
    "ensemble_30_per_seed_rmses": ens37_rmses,
    "optimal_weights_per_seed": opt_weights,
    "optimal_rmses": opt_rmses,
    "median_ensemble_rmse": median_ensemble_rmse,
}
with open("data/posthoc_ensemble_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved data/posthoc_ensemble_results.json")
