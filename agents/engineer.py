"""Engineer agent — runs HCX rolling evaluation given a Candidate config.

Not an LLM agent. A deterministic Python function that:
  1. Loads the requested data panel (Candidate.data_panel).
  2. Picks origins from Candidate.eval_set ("rolling_2025" / "stress_2022_h1").
  3. For each origin and seed (origins × n_seeds):
       - renders panel slice (+ optional fewshot examples) into TSV table,
       - composes system+user messages from candidate's templates,
       - calls HCX-32B-Think,
       - parses {"forecast": v} and stores it.
  4. Aggregates per-target / per-seed RMSE+MAE → Metric.
  5. Computes deterministic baselines (RW / AR(1) / Trend12 / MA12) on the same
     origins so the Researcher can rank LLM vs simple baselines.

HCX endpoint and parsing logic mirror rolling_2025*.py to keep results comparable.

Locked levers (Minimal Prompt Principle, see cognition.json):
  - Table is rendered as raw TSV. No markdown, no unit headers, no Date column.
"""
from __future__ import annotations
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .schemas import Candidate, Metric

API_URL = "https://namc-aigw.io.naver.com/v1/chat/completions"
MODEL = "HyperCLOVAX-SEED-32B-Think-Text"
ROOT = Path(__file__).resolve().parent.parent

# Eval sets — 8 origins each, 1-step-ahead
ORIGINS_BY_SET: dict[str, pd.PeriodIndex] = {
    "rolling_2025":    pd.period_range("2025-04", "2025-11", freq="M"),
    "stress_2022_h1":  pd.period_range("2021-12", "2022-07", freq="M"),  # surge regime
}

JSON_OUTPUT_INSTRUCTION = (
    "\n\nReturn ONLY a JSON object of the form "
    "{\"forecast\": v, \"rationale\": \"...\"} "
    "where v is a single decimal number like 2.3 (no % sign). "
    "Do not output anything outside the JSON object."
)

# Whitelist of placeholders the runner substitutes in system / user prompts.
_PLACEHOLDER_RE = re.compile(r"\{(table|origin|target|fewshot_block|cols_summary)\}")


def _render_template(template: str, mapping: dict) -> str:
    return _PLACEHOLDER_RE.sub(lambda m: str(mapping[m.group(1)]), template)


def _load_panel(data_panel: str) -> pd.DataFrame:
    fname = {
        "optimal18":     "macro_panel_optimal18.csv",
        "full_no_gt":    "macro_panel_full.csv",
        "full_with_gt":  "macro_panel_full.csv",
        "optimal18_aug": "macro_panel_aug.csv",
    }[data_panel]
    df = pd.read_csv(ROOT / "data" / fname, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="M")
    if data_panel == "full_no_gt":
        df = df.drop(columns=[c for c in df.columns if c.startswith("GT_")])
    return df


def _select_cols(df: pd.DataFrame, cand: Candidate) -> list:
    target = "CPI_KR_YoY"
    if cand.cov_mode == "univar":
        return [target]
    if cand.cov_mode == "all":
        return list(df.columns)
    keep = [target] + [c for c in cand.cov_subset if c != target and c in df.columns]
    return keep


def _render_panel(df_slice: pd.DataFrame, cand: Candidate, target_col: str = "CPI_KR_YoY") -> str:
    """Render dataframe slice as TSV. If `cand.blinded`, anonymize columns/index."""
    df = df_slice.round(3).copy()
    if cand.blinded:
        cols = [target_col] + [c for c in df.columns if c != target_col]
        df = df[cols]
        df.columns = [f"var_{i+1}" for i in range(len(df.columns))]
        df.index = [f"t={i+1}" for i in range(len(df))]
        df.index.name = "step"
    else:
        df.index = df.index.astype(str)
        df.index.name = "month"
    return df.to_csv(sep="\t")


def _diverse_regime_origins(df: pd.DataFrame, origin: pd.Period, k: int,
                            min_history: int) -> list:
    """Pick `k` past origins covering distinct CPI regimes (rising / falling / stable),
    vintage-safe relative to `origin`.

    Strategy: for each candidate past origin (with enough history before it), classify
    by 3-month CPI YoY change Δ. Group into:
      rising:  Δ > +0.5pp
      falling: Δ < -0.5pp
      stable:  |Δ| <= 0.3pp
    For k=3: latest of each regime. For k=6: 2 of each (latest + earlier). For k=12:
    fall back to balanced 4-of-each, with "any-regime" fill if a bucket runs out.
    Returns origins in chronological order.
    """
    candidates = []
    earliest = df.index[0] + min_history  # need history before this candidate
    for c in pd.period_range(earliest, origin - 1, freq="M"):
        if c < df.index[0] or c > df.index[-1]: continue
        cpi_now = float(df.loc[c, "CPI_KR_YoY"])
        c_prev = c - 3
        if c_prev < df.index[0]: continue
        cpi_prev = float(df.loc[c_prev, "CPI_KR_YoY"])
        delta = cpi_now - cpi_prev
        if delta > 0.5:    regime = "rising"
        elif delta < -0.5: regime = "falling"
        elif abs(delta) <= 0.3: regime = "stable"
        else:              regime = "transition"
        candidates.append((c, regime, delta))

    per_bucket = max(1, k // 3)
    buckets = {"rising": [], "falling": [], "stable": []}
    for c, regime, _ in reversed(candidates):
        if regime in buckets and len(buckets[regime]) < per_bucket:
            buckets[regime].append(c)

    picked = buckets["rising"] + buckets["falling"] + buckets["stable"]
    # If a bucket fell short, fill with most recent transitions/anything else
    if len(picked) < k:
        used = set(picked)
        for c, _, _ in reversed(candidates):
            if c in used: continue
            picked.append(c); used.add(c)
            if len(picked) >= k: break
    picked = picked[:k]
    picked.sort()  # chronological
    return picked


def _render_fewshot_block(df: pd.DataFrame, cols: list, origin: pd.Period, cand: Candidate) -> str:
    """Render `fewshot_k` examples per `fewshot_strategy`. Vintage-safe."""
    if cand.fewshot_k == 0:
        return ""

    if cand.fewshot_strategy == "recent":
        ex_origins = [origin - k for k in range(cand.fewshot_k, 0, -1)]
    elif cand.fewshot_strategy == "diverse_regime":
        ex_origins = _diverse_regime_origins(df, origin, cand.fewshot_k,
                                             min_history=cand.fewshot_panel_len + 3)
    else:
        raise ValueError(f"unknown fewshot_strategy {cand.fewshot_strategy}")

    parts = []
    for i, ex_origin in enumerate(ex_origins, 1):
        ex_target = ex_origin + 1
        ex_start = ex_origin - cand.fewshot_panel_len + 1
        if ex_start < df.index[0]:
            continue
        ex_panel = df.loc[str(ex_start):str(ex_origin), cols]
        ex_table = _render_panel(ex_panel, cand)
        ex_actual = float(df.loc[ex_target, "CPI_KR_YoY"])
        if cand.blinded:
            n = cand.fewshot_panel_len
            parts.append(
                f"==== EXAMPLE {i} ====\n"
                f"Panel (t=1..{n}):\n{ex_table}\n"
                f"Actual var_1 at t={n+1}: {ex_actual:.3f}\n"
            )
        else:
            parts.append(
                f"==== EXAMPLE {i} ====\n"
                f"Panel ending {ex_origin} ({cand.fewshot_panel_len} months):\n{ex_table}\n"
                f"Actual CPI_KR_YoY for {ex_target}: {ex_actual:.3f}\n"
            )
    return "\n".join(parts) + "\n"


def _hcx_call(api_key: str, sys_msg: str, user_msg: str, cand: Candidate, timeout: int = 240):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": cand.temperature,
        "max_tokens": cand.max_tokens,
    }
    for retry in range(3):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
            if r.status_code != 200:
                if retry < 2: time.sleep(2 ** retry); continue
                return None, None, f"HTTP {r.status_code}"
            content = r.json()["choices"][0]["message"].get("content")
            if not content:
                if retry < 2: time.sleep(2 ** retry); continue
                return None, None, "empty"
            m = re.search(r"\{[\s\S]*\}", content)
            obj = json.loads(m.group(0))
            v = obj["forecast"]
            v = float(v[0]) if isinstance(v, list) else float(v)
            return v, obj.get("rationale", ""), None
        except Exception as e:
            if retry < 2: time.sleep(2 ** retry); continue
            return None, None, str(e)[:120]
    return None, None, "max retries"


def _build_messages(df: pd.DataFrame, cols: list, origin: pd.Period, cand: Candidate):
    target = origin + 1
    ctx_start = origin - cand.ctx_len + 1
    panel = df.loc[str(ctx_start):str(origin), cols]
    table = _render_panel(panel, cand)
    fewshot_block = _render_fewshot_block(df, cols, origin, cand) if cand.fewshot_k else ""

    placeholders = {
        "table": table,
        "origin": str(origin),
        "target": str(target),
        "fewshot_block": fewshot_block,
        "cols_summary": ", ".join(cols) if not cand.blinded else
                        f"var_1..var_{len(cols)} (var_1 = target)",
    }
    user_msg = _render_template(cand.user_prompt_template, placeholders)
    sys_body = _render_template(cand.system_prompt, placeholders).rstrip()
    sys_msg = sys_body + JSON_OUTPUT_INSTRUCTION
    return sys_msg, user_msg


def deterministic_baselines(df: pd.DataFrame, origins: pd.PeriodIndex,
                            target_col: str = "CPI_KR_YoY") -> dict:
    """Compute RW / AR(1) / Trend12 / MA12 on the given origins, returning RMSE/MAE
    against the actuals. AR(1) is fit per-origin on history strictly before origin."""
    from sklearn.linear_model import LinearRegression
    targets = [str(o + 1) for o in origins]
    actuals = np.array([float(df.loc[t, target_col]) for t in targets])

    rw, ar1, trend12, ma12 = [], [], [], []
    for o in origins:
        hist = df.loc[:str(o), target_col]
        last_val = float(hist.iloc[-1])
        rw.append(last_val)

        y = hist.values
        X = y[:-1].reshape(-1, 1)
        Y = y[1:]
        m = LinearRegression().fit(X, Y)
        ar1.append(float(m.predict([[last_val]])[0]))

        last12 = hist.iloc[-12:].values
        coef = np.polyfit(np.arange(12), last12, 1)
        trend12.append(float(np.polyval(coef, 12)))

        ma12.append(float(hist.iloc[-12:].mean()))

    out = {}
    for name, preds in [("RW", rw), ("AR(1)", ar1), ("Trend12", trend12), ("MA12", ma12)]:
        preds_arr = np.array(preds)
        out[name] = {
            "rmse": float(np.sqrt(np.mean((preds_arr - actuals) ** 2))),
            "mae":  float(np.mean(np.abs(preds_arr - actuals))),
            "preds": [float(x) for x in preds_arr],
        }
    out["actuals"] = {t: float(actuals[i]) for i, t in enumerate(targets)}
    return out


def run_candidate(cand: Candidate, candidate_id: int, *, max_workers: int = 10,
                  verbose: bool = True) -> Metric:
    """Execute a Candidate against the chosen eval_set's origins × n_seeds."""
    api_key = os.environ.get("HCX_API_KEY")
    if not api_key:
        return Metric(candidate_id=candidate_id, status="failed", rmse_mc=None,
                      mae_mc=None, error="HCX_API_KEY missing")
    cand.validate()

    origins = ORIGINS_BY_SET[cand.eval_set]
    df = _load_panel(cand.data_panel)
    cols = _select_cols(df, cand)
    target_col = "CPI_KR_YoY"
    targets = [str(o + 1) for o in origins]
    actuals = {t: float(df.loc[t, target_col]) for t in targets}
    actual_arr = np.array([actuals[t] for t in targets])

    earliest_origin = origins[0] - max(cand.fewshot_k, 0) - 3  # extra slack for diverse_regime
    earliest_start = earliest_origin - max(cand.ctx_len, cand.fewshot_panel_len) + 1
    if df.index[0] > earliest_start:
        return Metric(candidate_id=candidate_id, status="failed", rmse_mc=None, mae_mc=None,
                      error=f"insufficient history; need >= {earliest_start}, have {df.index[0]}")

    per_origin_msgs = {o: _build_messages(df, cols, o, cand) for o in origins}

    forecasts: dict = {}
    rationales: dict = {}
    jobs = [(o, s) for o in origins for s in range(cand.n_seeds)]
    n_failed = 0
    t0 = time.time()

    if verbose:
        s, u = per_origin_msgs[origins[0]]
        print(f"  [engineer] eval_set={cand.eval_set}  origins={[str(o) for o in origins]}")
        print(f"  [engineer] sample prompt: system={len(s)} chars, user={len(u)} chars")
        print(f"  [engineer] {len(jobs)} HCX calls "
              f"({len(origins)} origins × {cand.n_seeds} seeds), cols={len(cols)}",
              flush=True)

    def _job(origin, seed_idx):
        sys_msg, user_msg = per_origin_msgs[origin]
        v, rat, err = _hcx_call(api_key, sys_msg, user_msg, cand)
        return origin, seed_idx, v, rat, err

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_job, o, s): (o, s) for o, s in jobs}
        done = 0
        for fut in as_completed(futures):
            o, s, v, rat, err = fut.result()
            tgt = str(o + 1)
            if v is not None:
                forecasts.setdefault(tgt, []).append(v)
                rationales.setdefault(tgt, []).append(rat or "")
            else:
                n_failed += 1
            done += 1
            if verbose:
                print(f"    [{done}/{len(jobs)}] {tgt} s{s}: "
                      f"{'%.3f' % v if v is not None else f'FAIL ({err})'}", flush=True)
    elapsed = time.time() - t0

    per_target = {}
    means_per_target = []
    for t in targets:
        fc = forecasts.get(t, [])
        per_target[t] = {
            "actual": actuals[t],
            "mean": float(np.mean(fc)) if fc else None,
            "sd": float(np.std(fc, ddof=1)) if len(fc) > 1 else None,
            "forecasts": [float(x) for x in fc],
        }
        means_per_target.append(per_target[t]["mean"] if fc else np.nan)

    means_arr = np.array(means_per_target, dtype=float)
    rmse_mc = float(np.sqrt(np.nanmean((means_arr - actual_arr) ** 2))) if not np.all(np.isnan(means_arr)) else None
    mae_mc = float(np.nanmean(np.abs(means_arr - actual_arr))) if not np.all(np.isnan(means_arr)) else None

    per_seed_rmse = []
    for s in range(cand.n_seeds):
        preds = []
        for t in targets:
            fc = forecasts.get(t, [])
            preds.append(fc[s] if s < len(fc) else np.nan)
        preds = np.array(preds)
        if not np.any(np.isnan(preds)):
            per_seed_rmse.append(float(np.sqrt(np.mean((preds - actual_arr) ** 2))))

    rationales_sample = {t: (rationales.get(t, [""])[0] or "")[:600] for t in targets}

    n_calls = len(jobs)
    if n_failed == 0:
        status = "ok"
    elif n_failed < n_calls:
        status = "partial"
    else:
        status = "failed"

    return Metric(
        candidate_id=candidate_id, status=status,
        rmse_mc=rmse_mc, mae_mc=mae_mc,
        per_seed_rmse=per_seed_rmse,
        per_target=per_target,
        rationales_sample=rationales_sample,
        n_calls=n_calls, n_failed=n_failed,
        elapsed_sec=elapsed,
    )


if __name__ == "__main__":
    # Smoke test the diverse_regime fewshot picker on stress_2022_h1
    df = _load_panel("full_no_gt")
    origin = ORIGINS_BY_SET["stress_2022_h1"][0]
    print(f"diverse_regime fewshot picker for origin {origin}:")
    for k in (3, 6):
        picks = _diverse_regime_origins(df, origin, k, min_history=15)
        print(f"  k={k}: {[str(p) for p in picks]}")
        for p in picks:
            cpi = float(df.loc[p, "CPI_KR_YoY"])
            cpi3 = float(df.loc[p - 3, "CPI_KR_YoY"])
            print(f"    {p}: CPI {cpi3:.2f} → {cpi:.2f} (Δ={cpi-cpi3:+.2f})")
    print("\nbaselines on stress_2022_h1:")
    bl = deterministic_baselines(df, ORIGINS_BY_SET["stress_2022_h1"])
    for k, v in bl.items():
        if k == "actuals": continue
        print(f"  {k}: RMSE={v['rmse']:.4f} MAE={v['mae']:.4f}")
