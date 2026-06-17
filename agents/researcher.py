"""Researcher agent — Claude Opus proposes the next Candidate config.

Reads experiments.db + cognition.json via store.py, composes an OPRO-style prompt
(past configs sorted by RMSE + accumulated lessons), asks Claude to emit one JSON
Candidate, validates it, ensures the hash is novel, and returns the Candidate.
"""
from __future__ import annotations
import json
import os
import re
import sys

import anthropic

from . import store
from .schemas import Candidate, best_known_baseline_summary

DEFAULT_MODEL = os.environ.get("AGENTS_RESEARCHER_MODEL", "claude-opus-4-7")
MAX_TOKENS = 4096
TOP_K_HISTORY = 12

SCHEMA_DOC = """\
Candidate JSON shape (all keys required unless marked optional):

{
  "eval_set":       one of ["rolling_2025", "stress_2022_h1"]
                    rolling_2025    = 8 origins 2025-04..2025-11 (CPI 1.6-2.5%, stable; well-mapped, Tier-1 ceiling ~0.25)
                    stress_2022_h1  = 8 origins 2021-12..2022-07 (CPI 3.77→6.33%, surge; CSI INVERTS vs 2025; UNEXPLORED)
  "data_panel":     one of ["optimal18", "full_no_gt", "full_with_gt", "optimal18_aug"]
                    optimal18      = 18 macro covariates + CPI_KR_YoY (no CSI, no Google Trends)
                    full_no_gt     = 18 macro + CPI + BoK_CSI (Korean Consumer Sentiment Index, 100=neutral)
                    full_with_gt   = 18 macro + CPI + BoK_CSI + 5 Google Trends queries
                    optimal18_aug  = 18 macro + CPI + 5 Google Trends (no CSI)
  "cov_mode":       one of ["all", "univar", "subset"]
                    "univar" = use only CPI_KR_YoY (ignore other columns).
  "cov_subset":     list[str] when cov_mode=="subset" else null. Column names from chosen panel.
  "ctx_len":        int in [12, 60]. Months of history per query (default: 36).
  "blinded":        bool. If true, columns become var_1..var_N (target = var_1) and
                    row labels become t=1..t=N (anonymized). Use only as a contamination
                    probe — labeled mode performs better when honest comparison is allowed.
  "fewshot_k":      0, 3, 6, or 12.
  "fewshot_strategy": one of ["recent", "diverse_regime"] (only used when fewshot_k > 0).
                    recent          = k preceding origins (vintage-safe, but anchoring trap on stable regimes)
                    diverse_regime  = k examples sampled across rising/falling/stable CPI regimes
                    (the open hypothesis from the experiment chronicle)
  "fewshot_panel_len": int (months per fewshot example, e.g. 12).
  "system_prompt":  str. Free-form. The fixed JSON-output instruction is appended by the
                    runner, so do NOT include any "return JSON" wording yourself.
                    Placeholders {origin}, {target}, {cols_summary} are interpolated here too.
  "user_prompt_template": template using placeholders:
                    {table}          REQUIRED — rendered TSV panel
                    {origin}         e.g. "2025-07"
                    {target}         e.g. "2025-08" (= origin + 1)
                    {fewshot_block}  rendered fewshot examples (empty when k=0)
                    {cols_summary}   column-name summary string
  "n_seeds":        1..10 (default 5). Each seed is a fresh HCX call (temperature-sampled).
  "temperature":    0.0..1.5 (default 0.7).
  "max_tokens":     int (default 8192).
  "hypothesis":     1-3 sentences naming which lever you're moving and why.
}

Note: table format is locked to raw TSV (Minimal Prompt Principle, 4-time-verified).
Markdown / unit headers / Date column / lag columns / cleaned format ALL hurt HCX.
If you want format variation, vary `system_prompt` instead.
"""

SYSTEM_PROMPT = f"""\
You are the Researcher in a multi-agent loop optimizing HCX-32B-Think (a 32B Korean LLM \
made by Naver) for in-context Korean CPI YoY 1-step-ahead forecasting. \
The Engineer runs Candidates through HCX, the Analyzer distills lessons. \
Your job is to propose ONE next Candidate that has a credible chance of beating the \
historical best on the chosen eval_set.

The score is per-seed RMSE (mean across 8 origins, then RMSE; lower is better). \
Pre-existing 2025-rolling SOTA: HCX forced 4-step CSI CoT 0.250 (single LLM); \
HCX+Trend12 50/50 ensemble 0.235. The 2022-H1 stress regime is unexplored — its open \
question is whether forced-CoT-with-CSI still helps when CSI's correlation with CPI \
flips sign vs the 2025 stable regime.

{SCHEMA_DOC}

CONSTRAINTS
- Your `system_prompt` will get a fixed JSON-output instruction appended by the runner; \
do NOT include any "return JSON" wording in `system_prompt` yourself.
- `user_prompt_template` MUST contain `{{table}}`. Other placeholders are optional but \
reuse them when relevant (especially `{{fewshot_block}}` when fewshot_k > 0).
- When `blinded` is true, your prompts must NOT mention "Korean", "CPI", "inflation", \
"BoK", "macroeconomist", or any specific covariate name — phrase the task purely in \
terms of var_1..var_N.
- Do NOT propose a Candidate whose structural hash duplicates one already in history.
- LESSONS are non-negotiable evidence — if a lesson says "X is bad", proposing X again \
WITHOUT explicit reason and a different framing is wasted compute. Prefer combinations \
of known-Tier-1 scaffolds (forced CoT + critique, stepback + forced) and the open \
hypotheses (diverse_regime fewshot, sign-conditional CSI framing for stress regimes).

HOW TO REASON
- For 2025 rolling: the prompt-engineering local optimum is well-mapped. Marginal gains \
require either (a) novel scaffold combinations, (b) diverse-regime fewshot, or (c) \
adaptive system_prompt for inflection vs stable origins.
- For 2022 H1 stress: the regime is UNEXPLORED. Test whether known-good prompts transfer, \
and try sign-agnostic CSI framing ("cite values, interpret direction conditionally on \
trend; do not assume CSI↑ ⇒ CPI↑") since 2022 H1 had INVERSE correlation.
- Spell your reasoning in `hypothesis` (1-3 sentences naming the lever and the lesson \
or open hypothesis you're acting on).

OUTPUT FORMAT
Return ONLY a single JSON object. No prose, no markdown fences. The JSON must parse and \
validate against the schema above.\
"""


def _format_history_block(history: list, top_k: int = TOP_K_HISTORY) -> str:
    if not history:
        return "(no prior runs)"
    lines = []
    for h in history[:top_k]:
        cfg = h["config"]
        rmse = f"{h['rmse_mc']:.4f}" if h["rmse_mc"] is not None else "—"
        seeds = ",".join(f"{x:.3f}" for x in h["per_seed_rmse"]) if h["per_seed_rmse"] else "—"
        sample_target = next(iter(h["rationales_sample"])) if h["rationales_sample"] else None
        rat = h["rationales_sample"].get(sample_target, "") if sample_target else ""
        rat = rat[:240].replace("\n", " ")
        lines.append(
            f"[id={h['id']} round={h['round_idx']} hash={h['config_hash']} status={h['status']}] "
            f"RMSE_mc={rmse} per-seed=[{seeds}]\n"
            f"  eval_set={cfg.get('eval_set')} data_panel={cfg.get('data_panel')} "
            f"cov_mode={cfg.get('cov_mode')} ctx_len={cfg.get('ctx_len')} "
            f"blinded={cfg.get('blinded')} fewshot_k={cfg.get('fewshot_k')} "
            f"strategy={cfg.get('fewshot_strategy')} n_seeds={cfg.get('n_seeds')} "
            f"T={cfg.get('temperature')}\n"
            f"  hypothesis: {(h.get('hypothesis') or '').strip()[:300]}\n"
            f"  system_prompt[:200]: {cfg.get('system_prompt','')[:200].replace(chr(10),' ')}\n"
            f"  sample rationale ({sample_target}): {rat}"
        )
    return "\n".join(lines)


def _format_lessons_block(lessons) -> str:
    if not lessons:
        return "(no lessons yet — this is round 0 territory)"
    out = []
    for l in lessons:
        tag = f"[{l.tag}] " if l.tag else ""
        prefix = "(seed)" if l.round_idx == -1 else f"(round {l.round_idx})"
        out.append(f"- {prefix} {tag}{l.text}"
                   + (f"  // evidence: {l.evidence}" if l.evidence else ""))
    return "\n".join(out)


def _extract_json(text: str) -> dict:
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence:
        text = fence.group(1)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
    return json.loads(m.group(0))


def propose(round_idx: int, *, model: str = DEFAULT_MODEL,
            max_retries: int = 3, eval_set_hint: str = None) -> Candidate:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("[researcher] ANTHROPIC_API_KEY missing")

    history = store.fetch_history(limit=TOP_K_HISTORY * 2)
    lessons = store.load_lessons()

    baselines_2025 = best_known_baseline_summary("rolling_2025")
    baselines_stress = best_known_baseline_summary("stress_2022_h1")

    hint = ""
    if eval_set_hint:
        hint = (f"\n[ORCHESTRATOR HINT] This round, propose a Candidate with "
                f"eval_set='{eval_set_hint}' (you can still freely choose other fields).\n")

    user_prompt = (
        f"ROUND {round_idx}.{hint}\n\n"
        f"PRIOR HISTORY (best RMSE first; '—' means failed/missing):\n"
        f"{_format_history_block(history)}\n\n"
        f"CUMULATIVE LESSONS (seed = pre-loop established findings):\n"
        f"{_format_lessons_block(lessons)}\n\n"
        f"BASELINES — rolling_2025:\n{baselines_2025}\n\n"
        f"BASELINES — stress_2022_h1:\n{baselines_stress}\n\n"
        f"Propose the next Candidate JSON now."
    )

    client = anthropic.Anthropic()
    feedback = ""
    last_err = None
    for attempt in range(max_retries):
        msg_user = user_prompt if not feedback else (
            user_prompt + "\n\n[REVISION REQUIRED]\n" + feedback
        )
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=[
                {"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}},
            ],
            messages=[{"role": "user", "content": msg_user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        try:
            obj = _extract_json(text)
            cand = Candidate.from_dict(obj)
            cand.validate()
        except Exception as e:
            last_err = e
            feedback = f"Previous response failed parsing/validation: {e!r}. Output ONLY a single JSON object matching the schema."
            print(f"[researcher] retry {attempt+1}/{max_retries}: {e}", flush=True)
            continue
        h = cand.hash()
        if store.has_config_hash(h):
            last_err = f"duplicate hash {h}"
            feedback = (f"That Candidate hash {h} already exists in history. "
                        "Vary at least one structural field (eval_set, data_panel, cov_mode, "
                        "ctx_len, blinded, fewshot_k, fewshot_strategy, fewshot_panel_len) or "
                        "rewrite system_prompt/user_prompt_template non-trivially.")
            print(f"[researcher] retry {attempt+1}/{max_retries}: duplicate hash {h}", flush=True)
            continue
        return cand
    raise RuntimeError(f"researcher failed after {max_retries} attempts: {last_err}")


if __name__ == "__main__":
    store.init_db()
    cand = propose(round_idx=store.latest_round() + 1)
    print(json.dumps(cand.canonical_dict(), ensure_ascii=False, indent=2)[:1200])
    print("\nhypothesis:", cand.hypothesis)
