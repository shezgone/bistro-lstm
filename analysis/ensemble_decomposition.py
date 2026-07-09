"""Decompose ensemble win: where does HCX vs Trend12 each excel?"""
import json
import numpy as np
import pandas as pd

with open("data/rolling_2025_csi_forced_results.json") as f:
    d = json.load(f)
forecasts = d["forecasts"]
actuals = d["actuals"]
targets = sorted(forecasts.keys())
actual_arr = np.array([actuals[t] for t in targets])

df = pd.read_csv('data/macro_panel_optimal18.csv', index_col=0)
df.index = pd.PeriodIndex(df.index, freq='M')
def trend12(origin_str):
    origin = pd.Period(origin_str, 'M')
    last12 = df.loc[:str(origin), 'CPI_KR_YoY'].iloc[-12:].values
    coef = np.polyfit(range(12), last12, 1)
    return float(np.polyval(coef, 12))
trend_preds = np.array([trend12(str(pd.Period(t, 'M') - 1)) for t in targets])
hcx_means = np.array([np.mean(forecasts[t]) for t in targets])
ens_50 = 0.5 * hcx_means + 0.5 * trend_preds

# Per-target decomposition
print(f"\n{'Target':10s} {'Actual':>8s} | {'HCX':>7s} {'err':>7s} | {'Trend12':>8s} {'err':>7s} | {'Ens50':>7s} {'err':>7s} | who_wins?")
print("-" * 110)
for i, tgt in enumerate(targets):
    a = actual_arr[i]
    h, t, e = hcx_means[i], trend_preds[i], ens_50[i]
    eh, et, ee = h - a, t - a, e - a
    winner = "HCX" if abs(eh) < abs(et) else ("Trend12" if abs(et) < abs(eh) else "tie")
    benefit = "ens helps" if abs(ee) < min(abs(eh), abs(et)) else ("ens between" if min(abs(eh), abs(et)) <= abs(ee) <= max(abs(eh), abs(et)) else "ens hurts")
    print(f"{tgt:10s} {a:8.3f} | {h:7.3f} {eh:+7.3f} | {t:8.3f} {et:+7.3f} | {e:7.3f} {ee:+7.3f} | {winner:8s}, {benefit}")

# Sign analysis
hcx_err = hcx_means - actual_arr
trend_err = trend_preds - actual_arr
print(f"\nHCX errors: {hcx_err.round(3).tolist()}")
print(f"Trend12 errors: {trend_err.round(3).tolist()}")
print(f"Sign agreement: {(np.sign(hcx_err) == np.sign(trend_err)).sum()}/8")
print(f"  → If they ERR in different directions, ensemble helps")

# Correlation
r = np.corrcoef(hcx_err, trend_err)[0, 1]
print(f"\nError correlation HCX vs Trend12: r = {r:+.3f}")
print(f"  (lower = better diversification, ensemble more useful)")

# Theoretical optimal: weighted least squares minimizing ensemble RMSE
# Min over w of mean( (w*HCX + (1-w)*Trend12 - actual)^2 )
# = mean( (e_h*w + e_t*(1-w))^2 )
# = mean( (w*(e_h - e_t) + e_t)^2 )
# Take derivative, set 0: w_opt = -mean(e_t * (e_h - e_t)) / mean((e_h - e_t)^2)
diff = hcx_err - trend_err
w_opt = -np.mean(trend_err * diff) / np.mean(diff**2)
print(f"\nTheoretical optimal w (HCX weight) = {w_opt:.3f}")
ens_opt = w_opt * hcx_means + (1-w_opt) * trend_preds
rmse_opt = float(np.sqrt(np.mean((ens_opt - actual_arr)**2)))
print(f"Ensemble at optimal w RMSE = {rmse_opt:.4f}")

# Conditional analysis: when actual is rising vs falling, who wins?
print(f"\n=== Conditional analysis: ===")
for i, tgt in enumerate(targets):
    if i == 0: continue
    prev = actual_arr[i-1]; curr = actual_arr[i]
    direction = "rising" if curr > prev else "falling"
    hcx_better = abs(hcx_err[i]) < abs(trend_err[i])
    print(f"  {tgt} ({direction}, prev={prev:.2f} → curr={curr:.2f}): HCX_err {hcx_err[i]:+.2f}, Trend_err {trend_err[i]:+.2f}, "
          f"{'HCX' if hcx_better else 'Trend12'} better")
