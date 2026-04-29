"""HCX-32B-Think n-seed ablation: (cov/univar) x (think on/off) x (2023/2024)."""
import os, sys, json, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, numpy as np, pandas as pd

API_URL = "https://namc-aigw.io.naver.com/v1/chat/completions"
MODEL = "HyperCLOVAX-SEED-32B-Think-Text"
API_KEY = os.environ.get("HCX_API_KEY")
if not API_KEY: sys.exit("HCX_API_KEY missing")

CSV = os.path.join(os.path.dirname(__file__), "data", "macro_panel_optimal18.csv")
df = pd.read_csv(CSV, index_col=0)
df.index = pd.PeriodIndex(df.index, freq="M")

CTX_RANGES = {2023: ("2020-01", "2022-12"), 2024: ("2021-01", "2023-12")}
N_PER_CELL = 8
TEMP = 0.7
MAX_TOKENS = 8192  # bigger to handle thinking + answer

def build_messages(year, mode):
    cs, ce = CTX_RANGES[year]
    cols = ["CPI_KR_YoY"] if mode == "univar" else list(df.columns)
    ctx = df.loc[cs:ce, cols].round(3)
    ctx.index = ctx.index.astype(str)
    table = ctx.to_csv(sep="\t")
    sys_msg = (
        "You are an expert macroeconomist forecasting Korean CPI YoY inflation. "
        f"Use the provided 36 months of macro panel as context. "
        f"Forecast monthly Korean CPI YoY (%) for Jan-Dec {year}, 12 values. "
        "Reason carefully about base effects, BoK monetary lag, commodity/FX trends, "
        "and global PPI signals. Return ONLY a JSON object: "
        '{"forecast": [v1, v2, ..., v12], "rationale": "1-3 sentences"}. '
        "Numbers must be plain decimals like 4.5, no % sign."
    )
    label = "Macro panel" if mode == "cov" else "Korean CPI YoY history"
    user_msg = (
        f"{label} ({cs} to {ce}, monthly, TSV):\n{table}\n\n"
        f"Forecast CPI_KR_YoY for {year}-01 through {year}-12. "
        "Output only the JSON specified in the system message."
    )
    return sys_msg, user_msg

def call_once(year, mode, think_on, attempt):
    sys_msg, user_msg = build_messages(year, mode)
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": TEMP,
        "max_tokens": MAX_TOKENS,
    }
    if not think_on:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    for retry in range(3):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=240)
            if r.status_code != 200:
                if retry < 2: time.sleep(2 ** retry); continue
                return None, f"HTTP {r.status_code}: {r.text[:200]}"
            msg = r.json()["choices"][0]["message"]
            content = msg.get("content")
            if not content:
                if retry < 2: time.sleep(2 ** retry); continue
                return None, "empty content"
            m = re.search(r"\{[\s\S]*\}", content)
            if not m:
                if retry < 2: time.sleep(2 ** retry); continue
                return None, "no JSON"
            obj = json.loads(m.group(0))
            forecast = np.array(obj["forecast"], dtype=float)
            if forecast.shape != (12,):
                if retry < 2: time.sleep(2 ** retry); continue
                return None, f"bad shape {forecast.shape}"
            return forecast, None
        except Exception as e:
            if retry < 2: time.sleep(2 ** retry); continue
            return None, f"exception: {e}"
    return None, "max retries"

def main():
    cells = []
    for year in [2023, 2024]:
        for mode in ["cov", "univar"]:
            for think_on in [True, False]:
                for i in range(N_PER_CELL):
                    cells.append((year, mode, think_on, i))
    print(f"Total calls: {len(cells)} ({len(cells)//N_PER_CELL} cells × {N_PER_CELL})", flush=True)

    results = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(call_once, *c): c for c in cells}
        done = 0
        for fut in as_completed(futures):
            year, mode, think_on, i = futures[fut]
            forecast, err = fut.result()
            key = (year, mode, think_on)
            if forecast is not None:
                results.setdefault(key, []).append(forecast)
            done += 1
            tag = f"{year} {mode} think={'ON' if think_on else 'OFF'}"
            print(f"[{done}/{len(cells)}] {tag} attempt {i}: {'OK' if forecast is not None else f'FAIL ({err})'}", flush=True)

    # Compute RMSEs and stats
    print("\n" + "=" * 90)
    print(f"{'Year':5s} {'Mode':6s} {'Think':6s} {'n':3s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'median':>8s}")
    print("-" * 90)
    rows = []
    for (year, mode, think_on), fcs in sorted(results.items()):
        actual = df.loc[f"{year}-01":f"{year}-12", "CPI_KR_YoY"].values
        rmses = [float(np.sqrt(np.mean((f - actual) ** 2))) for f in fcs]
        rmses = np.array(rmses)
        rows.append({
            "year": year, "mode": mode, "think": "ON" if think_on else "OFF",
            "n": len(rmses), "mean": rmses.mean(), "std": rmses.std(ddof=1),
            "min": rmses.min(), "max": rmses.max(), "median": np.median(rmses),
            "rmses": rmses.tolist(),
            "forecasts": [f.tolist() for f in fcs],
        })
        print(f"{year:5d} {mode:6s} {'ON' if think_on else 'OFF':6s} {len(rmses):3d} "
              f"{rmses.mean():8.4f} {rmses.std(ddof=1):8.4f} {rmses.min():8.4f} {rmses.max():8.4f} {np.median(rmses):8.4f}")

    # Pairwise tests for think on vs off
    print("\n" + "=" * 90)
    print("Welch's t-test: think ON vs think OFF (per year, per mode)")
    print("-" * 90)
    from scipy import stats
    for year in [2023, 2024]:
        for mode in ["cov", "univar"]:
            on = [r["rmses"] for r in rows if r["year"] == year and r["mode"] == mode and r["think"] == "ON"]
            off = [r["rmses"] for r in rows if r["year"] == year and r["mode"] == mode and r["think"] == "OFF"]
            if not on or not off: continue
            on, off = on[0], off[0]
            t, p_t = stats.ttest_ind(on, off, equal_var=False)
            u, p_u = stats.mannwhitneyu(on, off, alternative="two-sided")
            mean_diff = np.mean(on) - np.mean(off)
            tag = f"{year} {mode}"
            print(f"{tag:14s} | mean ON={np.mean(on):.4f} OFF={np.mean(off):.4f} | "
                  f"Δ={mean_diff:+.4f} | t={t:+.2f} p_t={p_t:.4f} | U={u:.0f} p_u={p_u:.4f}")

    # cov vs univar (think on)
    print("\n" + "=" * 90)
    print("Welch's t-test: COV vs UNIVAR (think ON)")
    print("-" * 90)
    for year in [2023, 2024]:
        cov = [r["rmses"] for r in rows if r["year"] == year and r["mode"] == "cov" and r["think"] == "ON"]
        uni = [r["rmses"] for r in rows if r["year"] == year and r["mode"] == "univar" and r["think"] == "ON"]
        if not cov or not uni: continue
        cov, uni = cov[0], uni[0]
        t, p_t = stats.ttest_ind(cov, uni, equal_var=False)
        u, p_u = stats.mannwhitneyu(cov, uni, alternative="two-sided")
        mean_diff = np.mean(cov) - np.mean(uni)
        print(f"{year} think=ON  | mean COV={np.mean(cov):.4f} UNI={np.mean(uni):.4f} | "
              f"Δ={mean_diff:+.4f} | t={t:+.2f} p_t={p_t:.4f} | U={u:.0f} p_u={p_u:.4f}")
    for year in [2023, 2024]:
        cov = [r["rmses"] for r in rows if r["year"] == year and r["mode"] == "cov" and r["think"] == "OFF"]
        uni = [r["rmses"] for r in rows if r["year"] == year and r["mode"] == "univar" and r["think"] == "OFF"]
        if not cov or not uni: continue
        cov, uni = cov[0], uni[0]
        t, p_t = stats.ttest_ind(cov, uni, equal_var=False)
        u, p_u = stats.mannwhitneyu(cov, uni, alternative="two-sided")
        mean_diff = np.mean(cov) - np.mean(uni)
        print(f"{year} think=OFF | mean COV={np.mean(cov):.4f} UNI={np.mean(uni):.4f} | "
              f"Δ={mean_diff:+.4f} | t={t:+.2f} p_t={p_t:.4f} | U={u:.0f} p_u={p_u:.4f}")

    # Save raw
    out = {"rows": rows, "n_per_cell": N_PER_CELL, "temperature": TEMP}
    with open("data/hcx32_ablation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved data/hcx32_ablation_results.json")

if __name__ == "__main__":
    main()
