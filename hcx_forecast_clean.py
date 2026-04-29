"""HCX-32B-Think with cleaned context format: markdown table, units, sentinels masked."""
import os, sys, json, re, argparse
import requests, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--year", type=int, default=2023, choices=[2023, 2024])
ap.add_argument("--mode", choices=["univar", "cov"], default="cov")
args = ap.parse_args()
YEAR, MODE = args.year, args.mode

CTX_START = {2023: "2020-01", 2024: "2021-01"}[YEAR]
CTX_END   = {2023: "2022-12", 2024: "2023-12"}[YEAR]
TGT_START, TGT_END = f"{YEAR}-01", f"{YEAR}-12"

API_URL = "https://namc-aigw.io.naver.com/v1/chat/completions"
MODEL = "HyperCLOVAX-SEED-32B-Think-Text"
API_KEY = os.environ.get("HCX_API_KEY")
if not API_KEY: sys.exit("HCX_API_KEY missing")

# Variable metadata: (full name, unit)
VAR_META = {
    "CPI_KR_YoY": ("Korea CPI", "YoY %"),
    "AUD_USD": ("AUD per USD", "FX rate"),
    "CN_Interbank3M": ("China 3-month interbank rate", "%"),
    "US_UnempRate": ("US unemployment rate", "%"),
    "JP_REER": ("Japan real effective exchange rate", "index"),
    "JP_Interbank3M": ("Japan 3-month interbank rate", "%"),
    "JP_CoreCPI": ("Japan core CPI", "YoY %"),
    "KC_FSI": ("Korea financial stress index", "index"),
    "KR_MfgProd": ("Korea manufacturing production", "YoY %"),
    "Pork": ("Domestic pork wholesale price", "won/kg"),
    "US_NFP": ("US nonfarm payrolls change", "thousands MoM"),
    "US_TradeTransEmp": ("US trade & transport employment change", "thousands MoM"),
    "THB_USD": ("Thai baht per USD", "FX rate"),
    "PPI_CopperNickel": ("Copper/Nickel PPI", "YoY %"),
    "CN_PPI": ("China producer price index", "YoY %"),
    "US_Mortgage15Y": ("US 15-year mortgage rate", "%"),
    "UK_10Y_Bond": ("UK 10-year gilt yield", "%"),
    "US_ExportPI": ("US export price index", "YoY %"),
    "US_DepInstCredit": ("US depository institution credit growth", "%"),
}

# Sentinel values to mask
SENTINELS = {"JP_CoreCPI": -0.999, "PPI_CopperNickel": 11.111}

CSV = os.path.join(os.path.dirname(__file__), "data", "macro_panel_optimal18.csv")
df = pd.read_csv(CSV, index_col=0)
df.index = pd.PeriodIndex(df.index, freq="M")

cols = ["CPI_KR_YoY"] if MODE == "univar" else list(df.columns)
ctx = df.loc[CTX_START:CTX_END, cols].round(3).copy()

# Mask sentinels
for col, sv in SENTINELS.items():
    if col in ctx.columns:
        ctx.loc[ctx[col] == sv, col] = np.nan

ctx.index = ctx.index.astype(str)

# Build markdown table with units in header
units_row = []
for c in ctx.columns:
    name, unit = VAR_META.get(c, (c, ""))
    units_row.append(f"{c} ({unit})" if unit else c)

def fmt(v):
    if pd.isna(v):
        return "—"
    return f"{v:g}"

lines = ["| Date | " + " | ".join(units_row) + " |"]
lines.append("|" + "---|" * (len(ctx.columns) + 1))
for date, row in ctx.iterrows():
    lines.append("| " + date + " | " + " | ".join(fmt(row[c]) for c in ctx.columns) + " |")
table = "\n".join(lines)

# Variable definitions prelude (only for cov mode)
defs = ""
if MODE == "cov":
    defs_list = []
    for c in ctx.columns:
        name, unit = VAR_META.get(c, (c, ""))
        defs_list.append(f"- **{c}**: {name} ({unit})")
    defs = "Variable definitions:\n" + "\n".join(defs_list) + "\n\n"

system_msg = (
    "You are an expert macroeconomist forecasting Korean CPI YoY inflation. "
    f"You will receive a monthly macro panel ({CTX_START} to {CTX_END}, "
    f"{36 if YEAR==2023 else 36} rows in chronological order). "
    "Missing values are marked as '—'. "
    f"Forecast monthly Korean CPI YoY (%) for Jan-Dec {YEAR}, 12 values. "
    "Reason carefully about base effects, BoK monetary lag, commodity/FX trends, "
    "and global PPI signals. Return ONLY a JSON object: "
    '{"forecast": [v1, v2, ..., v12], "rationale": "1-3 sentences"}. '
    "Numbers must be plain decimals like 4.5, no % sign."
)
user_msg = (
    f"{defs}"
    f"Monthly macro panel ({CTX_START} to {CTX_END}, ordered chronologically):\n\n"
    f"{table}\n\n"
    f"Forecast CPI_KR_YoY for {TGT_START} through {TGT_END}. "
    "Output only the JSON specified in the system message."
)

print(f"[{MODE} {YEAR}] context chars={len(user_msg)}", flush=True)
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
payload = {"model": MODEL, "messages": [
    {"role": "system", "content": system_msg},
    {"role": "user", "content": user_msg},
], "temperature": 0.2, "max_tokens": 16384}

r = requests.post(API_URL, headers=headers, json=payload, timeout=180)
print(f"HTTP {r.status_code}")
if r.status_code != 200:
    print(r.text[:600]); sys.exit(1)

content = r.json()["choices"][0]["message"]["content"]
if content is None:
    print("content was None, raw response:"); print(json.dumps(r.json(), indent=2)[:1500]); sys.exit(2)
print("---RAW---"); print(content[:1200]); print("---")

m = re.search(r"\{[\s\S]*\}", content)
obj = json.loads(m.group(0))
forecast = np.array(obj["forecast"], dtype=float)
actual = df.loc[TGT_START:TGT_END, "CPI_KR_YoY"].values
rmse = float(np.sqrt(np.mean((forecast - actual)**2)))

print(f"\nForecast: {forecast.tolist()}")
print(f"RMSE: {rmse:.4f}pp")
print(f"Rationale: {obj.get('rationale', '')}")

out = f"data/hcx32_clean_{MODE}_{YEAR}.npz"
np.savez_compressed(out,
    forecast_med=forecast,
    forecast_ci_lo=forecast - 0.8,
    forecast_ci_hi=forecast + 0.8,
    rmse=rmse,
    rationale=np.array(obj.get("rationale", "")),
)
print(f"Saved {out}")
