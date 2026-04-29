"""HCX-32B-Think with reasoning disabled — compare to thinking-on baseline."""
import os, sys, json, re, argparse
import requests, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--year", type=int, default=2023, choices=[2023, 2024])
ap.add_argument("--mode", choices=["univar", "cov"], default="cov")
ap.add_argument("--method", choices=["param", "token", "both"], default="param",
                help="param: chat_template_kwargs; token: /no_think prefix")
args = ap.parse_args()
YEAR, MODE, METHOD = args.year, args.mode, args.method

CTX_START = {2023: "2020-01", 2024: "2021-01"}[YEAR]
CTX_END   = {2023: "2022-12", 2024: "2023-12"}[YEAR]
TGT_START, TGT_END = f"{YEAR}-01", f"{YEAR}-12"

API_URL = "https://namc-aigw.io.naver.com/v1/chat/completions"
MODEL = "HyperCLOVAX-SEED-32B-Think-Text"
API_KEY = os.environ.get("HCX_API_KEY")
if not API_KEY: sys.exit("HCX_API_KEY missing")

CSV = os.path.join(os.path.dirname(__file__), "data", "macro_panel_optimal18.csv")
df = pd.read_csv(CSV, index_col=0)
df.index = pd.PeriodIndex(df.index, freq="M")
cols = ["CPI_KR_YoY"] if MODE == "univar" else list(df.columns)
ctx = df.loc[CTX_START:CTX_END, cols].round(3)
ctx.index = ctx.index.astype(str)
table = ctx.to_csv(sep="\t")

system_msg = (
    "You are an expert macroeconomist forecasting Korean CPI YoY inflation. "
    f"Use the provided 36 months of macro panel as context. "
    f"Forecast monthly Korean CPI YoY (%) for Jan-Dec {YEAR}, 12 values. "
    "Reason carefully about base effects, BoK monetary lag, commodity/FX trends, "
    "and global PPI signals. Return ONLY a JSON object: "
    '{"forecast": [v1, v2, ..., v12], "rationale": "1-3 sentences"}. '
    "Numbers must be plain decimals like 4.5, no % sign."
)
ctx_label = "Macro panel" if MODE == "cov" else "Korean CPI YoY history"
user_msg = (
    f"{ctx_label} ({CTX_START} to {CTX_END}, monthly, TSV):\n{table}\n\n"
    f"Forecast CPI_KR_YoY for {TGT_START} through {TGT_END}. "
    "Output only the JSON specified in the system message."
)

# Apply no-think methods
if METHOD in ("token", "both"):
    user_msg = user_msg + " /no_think"

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
if METHOD in ("param", "both"):
    payload["chat_template_kwargs"] = {"enable_thinking": False}

print(f"[{MODE} {YEAR} method={METHOD}] no-think attempt", flush=True)
r = requests.post(API_URL, headers=headers, json=payload, timeout=180)
print(f"HTTP {r.status_code}")
if r.status_code != 200:
    print(r.text[:1000]); sys.exit(1)

resp = r.json()
msg = resp["choices"][0]["message"]
content = msg.get("content")
rc = msg.get("reasoning_content") or ""
finish = resp["choices"][0].get("finish_reason")
usage = resp.get("usage", {})
print(f"finish_reason={finish}, usage={usage}")
print(f"reasoning_content length: {len(rc)} chars")
if rc:
    print(f"reasoning_content (first 400 chars): {rc[:400]}")
print(f"---CONTENT---")
print(content[:1500] if content else "(empty)")
print("---")

if not content:
    print("Empty content; cannot parse forecast", file=sys.stderr); sys.exit(2)

m = re.search(r"\{[\s\S]*\}", content)
obj = json.loads(m.group(0))
forecast = np.array(obj["forecast"], dtype=float)
actual = df.loc[TGT_START:TGT_END, "CPI_KR_YoY"].values
rmse = float(np.sqrt(np.mean((forecast - actual)**2)))
print(f"\nForecast: {forecast.tolist()}")
print(f"RMSE: {rmse:.4f}pp")

out = f"data/hcx32_nothink_{METHOD}_{MODE}_{YEAR}.npz"
np.savez_compressed(out,
    forecast_med=forecast,
    forecast_ci_lo=forecast - 0.8,
    forecast_ci_hi=forecast + 0.8,
    rmse=rmse,
    rationale=np.array(obj.get("rationale", "")),
    reasoning_len=len(rc),
)
print(f"Saved {out}")
