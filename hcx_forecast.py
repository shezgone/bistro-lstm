"""Call HCX-32B-Think with 18 covariates + CPI context to forecast Korean CPI YoY."""
import os
import sys
import json
import re
import argparse
import requests
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--year", type=int, default=2023, choices=[2023, 2024])
args = ap.parse_args()
YEAR = args.year

CTX_START = {2023: "2020-01", 2024: "2021-01"}[YEAR]
CTX_END   = {2023: "2022-12", 2024: "2023-12"}[YEAR]
TGT_START = f"{YEAR}-01"
TGT_END   = f"{YEAR}-12"

API_URL = "https://namc-aigw.io.naver.com/v1/chat/completions"
MODEL = "HyperCLOVAX-SEED-32B-Think-Text"
API_KEY = os.environ.get("HCX_API_KEY")
if not API_KEY:
    print("ERROR: HCX_API_KEY env var not set", file=sys.stderr)
    sys.exit(1)

CSV = os.path.join(os.path.dirname(__file__), "data", "macro_panel_optimal18.csv")
df = pd.read_csv(CSV, index_col=0)
df.index = pd.PeriodIndex(df.index, freq="M")

ctx = df.loc[CTX_START:CTX_END].round(3)
ctx.index = ctx.index.astype(str)

system_msg = (
    "You are an expert macroeconomist forecasting Korean CPI YoY inflation. "
    "Use the provided 36 months of macro panel (CPI + 18 covariates) as context. "
    f"Forecast monthly Korean CPI YoY (%) for Jan-Dec {YEAR}, 12 values. "
    "Reason carefully about base effects, BoK monetary lag, commodity/FX trends, "
    "and global PPI signals. Return ONLY a JSON object: "
    '{"forecast": [v1, v2, ..., v12], "rationale": "1-3 sentences"}. '
    "Numbers must be plain decimals like 4.5, no % sign."
)

table = ctx.to_csv(sep="\t")
user_msg = (
    f"Macro panel ({CTX_START} to {CTX_END}, monthly, TSV):\n"
    f"{table}\n\n"
    f"Forecast CPI_KR_YoY for {TGT_START} through {TGT_END}. "
    "Output only the JSON specified in the system message."
)

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ],
    "temperature": 0.2,
    "max_tokens": 4096,
}

print(f"Calling {MODEL} ...", flush=True)
r = requests.post(API_URL, headers=headers, json=payload, timeout=180)
print(f"HTTP {r.status_code}", flush=True)
if r.status_code != 200:
    print(r.text[:1000])
    sys.exit(1)

resp = r.json()
content = resp["choices"][0]["message"]["content"]
print("---RAW RESPONSE---")
print(content[:2000])
print("---")

# Extract JSON
m = re.search(r"\{[\s\S]*\}", content)
if not m:
    print("No JSON object found in response", file=sys.stderr)
    sys.exit(1)
try:
    obj = json.loads(m.group(0))
except json.JSONDecodeError as e:
    print(f"JSON parse error: {e}", file=sys.stderr)
    print(m.group(0)[:500])
    sys.exit(1)

forecast = np.array(obj["forecast"], dtype=float)
if forecast.shape != (12,):
    print(f"Bad shape: {forecast.shape}", file=sys.stderr)
    sys.exit(1)

actual = df.loc[TGT_START:TGT_END, "CPI_KR_YoY"].values
rmse = float(np.sqrt(np.mean((forecast - actual) ** 2)))

print(f"\nForecast: {forecast.tolist()}")
print(f"RMSE: {rmse:.4f}pp")
print(f"Rationale: {obj.get('rationale', '')}")

ci_band = 0.8
out = f"data/hcx32_covariate_{YEAR}_result.npz" if YEAR != 2023 else "data/hcx32_covariate_result.npz"
np.savez_compressed(
    out,
    forecast_med=forecast,
    forecast_ci_lo=forecast - ci_band,
    forecast_ci_hi=forecast + ci_band,
    rmse=rmse,
    rationale=np.array(obj.get("rationale", "")),
)
print(f"Saved {out}")
