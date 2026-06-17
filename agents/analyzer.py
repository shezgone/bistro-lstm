"""Analyzer agent — Claude Opus distills transferable lessons from the latest round.

Reads the last evaluated Candidate + its Metric and the recent history, asks Claude for
1-3 lessons describing what worked / didn't work, and appends them to cognition.json.
Lessons are short, actionable, and tagged by lever (format, fewshot, covariate, ...).
"""
from __future__ import annotations
import json
import os
import re
import sys
from typing import Optional

import anthropic

from . import store
from .schemas import Lesson

DEFAULT_MODEL = os.environ.get("AGENTS_ANALYZER_MODEL", "claude-opus-4-7")
MAX_TOKENS = 2048

SYSTEM_PROMPT = """\
You are the Analyzer in a multi-agent loop optimizing HCX-32B-Think for in-context Korean \
CPI forecasting on the rolling 2025-04..2025-11 evaluation. Lower per-seed RMSE is better.

Your job: given the LATEST round's Candidate config + its metrics + sample rationales, plus \
the RECENT context (top-N history and existing lessons), distill 1-3 SHORT, TRANSFERABLE \
lessons that should guide future Researcher proposals.

A good lesson:
- names the lever (e.g. "fewshot_k", "fewshot_strategy", "blinded", "eval_set", \
  "system_prompt phrasing", "covariate emphasis", "scaffold combination").
- states a comparative finding ("X helps when Y", "X is on par with Y", "X hurts because ...").
- is grounded in actual evidence from the metrics or rationales, not speculation.
- is NOT a duplicate or near-duplicate of an existing lesson.

A bad lesson is generic ("try harder"), tautological ("lower RMSE is better"), or unsupported.

Output ONLY a JSON object of the shape:
{
  "lessons": [
    {"text": "...", "evidence": "...", "tag": "format|fewshot|covariate|reasoning|prompt|data|other"}
  ]
}

If the latest round produced nothing actionable (e.g. it failed, or it merely confirmed an \
existing lesson with no new angle), return {"lessons": []} — do not pad.
"""


def _format_round_block(latest: dict) -> str:
    cfg = latest["config"]
    rmse = f"{latest['rmse_mc']:.4f}" if latest["rmse_mc"] is not None else "—"
    seeds = ",".join(f"{x:.4f}" for x in latest["per_seed_rmse"]) if latest["per_seed_rmse"] else "—"
    rats = []
    for tgt, rat in (latest["rationales_sample"] or {}).items():
        rat = (rat or "").strip().replace("\n", " ")
        if rat:
            rats.append(f"  [{tgt}] {rat[:300]}")
    rats_str = "\n".join(rats[:4]) if rats else "  (no rationales captured)"

    per_target_lines = []
    for tgt, info in (latest.get("per_target") or {}).items():
        if info.get("mean") is None:
            continue
        per_target_lines.append(
            f"  {tgt}: actual={info['actual']:.3f}  "
            f"forecast_mean={info['mean']:.3f} (sd={info['sd']:.3f})"
            if info.get("sd") is not None else
            f"  {tgt}: actual={info['actual']:.3f}  forecast_mean={info['mean']:.3f}"
        )
    pt_str = "\n".join(per_target_lines) if per_target_lines else "  (no per-target data)"

    return (
        f"Round {latest['round_idx']} (id={latest['id']} hash={latest['config_hash']} "
        f"status={latest['status']})\n"
        f"  RMSE_mc={rmse}  per-seed=[{seeds}]  n_failed={latest['n_failed']}\n"
        f"  hypothesis: {(latest['hypothesis'] or '').strip()}\n"
        f"  config: eval_set={cfg.get('eval_set')} data_panel={cfg.get('data_panel')} "
        f"cov_mode={cfg.get('cov_mode')} ctx_len={cfg.get('ctx_len')} "
        f"blinded={cfg.get('blinded')} fewshot_k={cfg.get('fewshot_k')} "
        f"strategy={cfg.get('fewshot_strategy')} n_seeds={cfg.get('n_seeds')} "
        f"T={cfg.get('temperature')}\n"
        f"  system_prompt[:300]: {cfg.get('system_prompt','')[:300].replace(chr(10),' ')}\n"
        f"  user_prompt_template[:200]: {cfg.get('user_prompt_template','')[:200].replace(chr(10),' ')}\n"
        f"  per-target:\n{pt_str}\n"
        f"  rationales (1st seed):\n{rats_str}"
    )


def _format_history_block(history: list[dict], top_k: int = 8) -> str:
    if not history:
        return "(no history)"
    lines = []
    for h in history[:top_k]:
        rmse = f"{h['rmse_mc']:.4f}" if h["rmse_mc"] is not None else "—"
        cfg = h["config"]
        lines.append(
            f"  id={h['id']} round={h['round_idx']} RMSE={rmse} "
            f"eval={cfg.get('eval_set')} data={cfg.get('data_panel')} "
            f"cov={cfg.get('cov_mode')} blinded={cfg.get('blinded')} "
            f"fewshot={cfg.get('fewshot_k')}/{cfg.get('fewshot_strategy')}  "
            f"hyp: {(h.get('hypothesis') or '')[:120]}"
        )
    return "\n".join(lines)


def _format_lessons_block(lessons) -> str:
    if not lessons:
        return "(none yet)"
    return "\n".join(f"- (round {l.round_idx}) [{l.tag or 'other'}] {l.text}"
                    for l in lessons)


def _extract_json(text: str) -> dict:
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence:
        text = fence.group(1)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
    return json.loads(m.group(0))


def analyze(round_idx: int, candidate_id: int, *, model: str = DEFAULT_MODEL,
            max_retries: int = 2) -> list[Lesson]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("[analyzer] ANTHROPIC_API_KEY missing")

    history = store.fetch_history(limit=20)
    latest: Optional[dict] = next((h for h in history if h["id"] == candidate_id), None)
    if latest is None:
        # latest run isn't ranked at the top because it failed; pull directly
        with store._conn() as c:  # noqa: SLF001
            row = c.execute("SELECT id FROM candidates WHERE id=?", (candidate_id,)).fetchone()
        if row is None:
            print(f"[analyzer] candidate id {candidate_id} not found", flush=True)
            return []
        # Re-fetch full history to find this row
        full = store.fetch_history(limit=10**6)
        latest = next((h for h in full if h["id"] == candidate_id), None)
        if latest is None:
            return []

    other_history = [h for h in history if h["id"] != candidate_id]
    lessons = store.load_lessons()

    user_prompt = (
        f"LATEST ROUND:\n{_format_round_block(latest)}\n\n"
        f"RECENT HISTORY (top by RMSE, latest excluded):\n{_format_history_block(other_history)}\n\n"
        f"EXISTING LESSONS:\n{_format_lessons_block(lessons)}\n\n"
        f"Distill 1-3 transferable lessons from this round. Return JSON {{\"lessons\": [...]}}. "
        f"If nothing new is learned, return {{\"lessons\": []}}."
    )

    client = anthropic.Anthropic()
    last_err = None
    for attempt in range(max_retries):
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=[
                {"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}},
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        try:
            obj = _extract_json(text)
            raw = obj.get("lessons", [])
            out = []
            for d in raw:
                t = (d.get("text") or "").strip()
                if not t:
                    continue
                out.append(Lesson(
                    round_idx=round_idx,
                    text=t,
                    evidence=(d.get("evidence") or "").strip(),
                    tag=(d.get("tag") or "").strip().lower(),
                ))
            return out
        except Exception as e:
            last_err = e
            print(f"[analyzer] retry {attempt+1}/{max_retries}: {e}", flush=True)
            continue
    print(f"[analyzer] giving up after {max_retries} attempts: {last_err}", flush=True)
    return []


if __name__ == "__main__":
    store.init_db()
    rnd = store.latest_round()
    history = store.fetch_history(limit=1)
    if not history:
        print("(no candidates yet)"); raise SystemExit
    cid = history[0]["id"]
    new = analyze(rnd, cid)
    print(json.dumps([l.__dict__ for l in new], ensure_ascii=False, indent=2))
