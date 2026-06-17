"""Manual-mode helper for the multi-agent loop without Anthropic API.

Researcher / Analyzer roles are played by the human + assistant in the chat.
This module exposes thin CLI subcommands the assistant uses each round:

  info                 — print history + lessons summary (read DB + cognition.json).
  run --candidate-file F --round N
                       — insert Candidate, execute Engineer (HCX), save Metric.
  lessons --lessons-file F --round N
                       — append lesson(s) to cognition.json.

Candidate JSON shape: see agents/schemas.py Candidate fields.
Lessons JSON shape: {"lessons": [{"text", "evidence", "tag"}, ...]}.
"""
from __future__ import annotations
import argparse
import json
import sys

from . import engineer, store
from .schemas import Candidate, Lesson, best_known_baseline_summary


def _cmd_info() -> None:
    store.init_db()
    history = store.fetch_history(limit=20)
    lessons = store.load_lessons()
    print(f"=== state ===")
    print(f"  candidates in DB:  {len(history)}")
    print(f"  lessons accumulated: {len(lessons)}  "
          f"(seed: {sum(1 for l in lessons if l.round_idx == -1)})")
    print(f"  latest_round: {store.latest_round()}")

    print("\n=== history (best RMSE_mc first) ===")
    if not history:
        print("  (empty)")
    for h in history:
        rmse = f"{h['rmse_mc']:.4f}" if h["rmse_mc"] is not None else "—"
        cfg = h["config"]
        print(f"  id={h['id']:>3} round={h['round_idx']:>2} hash={h['config_hash']} "
              f"status={h['status']} RMSE_mc={rmse}  "
              f"eval={cfg.get('eval_set')} data={cfg.get('data_panel')} "
              f"cov={cfg.get('cov_mode')} blinded={cfg.get('blinded')} "
              f"fs={cfg.get('fewshot_k')}/{cfg.get('fewshot_strategy')}")
        if h.get("hypothesis"):
            print(f"      hyp: {h['hypothesis'][:200]}")

    print("\n=== lessons ===")
    if not lessons:
        print("  (none)")
    for l in lessons:
        prefix = "(seed)" if l.round_idx == -1 else f"(round {l.round_idx})"
        tag = f"[{l.tag}] " if l.tag else ""
        print(f"  - {prefix} {tag}{l.text}")
        if l.evidence:
            print(f"      // evidence: {l.evidence}")

    print("\n=== baselines (rolling_2025) ===")
    print(best_known_baseline_summary("rolling_2025"))
    print("\n=== baselines (stress_2022_h1) ===")
    print(best_known_baseline_summary("stress_2022_h1"))


def _cmd_run(candidate_file: str, round_idx: int) -> None:
    store.init_db()
    with open(candidate_file) as f:
        d = json.load(f)
    cand = Candidate.from_dict(d)
    cand.validate()
    h = cand.hash()
    if store.has_config_hash(h):
        print(json.dumps({"error": f"duplicate hash {h}"}))
        sys.exit(2)

    cid = store.insert_candidate(round_idx, cand)
    print(f"[manual_run] inserted candidate id={cid} hash={h} round={round_idx}", flush=True)
    print(f"[manual_run] eval_set={cand.eval_set} data_panel={cand.data_panel} "
          f"cov_mode={cand.cov_mode} blinded={cand.blinded} "
          f"fewshot_k={cand.fewshot_k}/{cand.fewshot_strategy} n_seeds={cand.n_seeds}", flush=True)
    print(f"[manual_run] running engineer (HCX) ...", flush=True)

    m = engineer.run_candidate(cand, candidate_id=cid, verbose=True)
    store.insert_metric(m)

    summary = {
        "candidate_id": cid,
        "round_idx": round_idx,
        "hash": h,
        "status": m.status,
        "rmse_mc": m.rmse_mc,
        "mae_mc": m.mae_mc,
        "per_seed_rmse": m.per_seed_rmse,
        "n_calls": m.n_calls,
        "n_failed": m.n_failed,
        "elapsed_sec": m.elapsed_sec,
        "error": m.error,
        "per_target_means": {t: info.get("mean") for t, info in m.per_target.items()},
        "actuals": {t: info.get("actual") for t, info in m.per_target.items()},
    }
    print("\n=== ROUND RESULT ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_lessons(lessons_file: str, round_idx: int) -> None:
    store.init_db()
    with open(lessons_file) as f:
        d = json.load(f)
    raw = d.get("lessons", []) if isinstance(d, dict) else d
    objs = []
    for r in raw:
        text = (r.get("text") or "").strip()
        if not text: continue
        objs.append(Lesson(
            round_idx=round_idx,
            text=text,
            evidence=(r.get("evidence") or "").strip(),
            tag=(r.get("tag") or "").strip().lower(),
        ))
    if not objs:
        print("(no non-empty lessons in file)")
        return
    store.append_lessons(objs)
    print(f"appended {len(objs)} lesson(s) for round {round_idx}")


def _cmd_show_round(candidate_id: int) -> None:
    """Dump the per-target forecasts and rationales for a single candidate."""
    store.init_db()
    history = store.fetch_history(limit=10**6)
    h = next((x for x in history if x["id"] == candidate_id), None)
    if h is None:
        print(f"candidate {candidate_id} not found")
        sys.exit(2)
    print(f"=== candidate id={candidate_id} round={h['round_idx']} hash={h['config_hash']} ===")
    print(f"  status: {h['status']}  RMSE_mc: {h['rmse_mc']}  MAE_mc: {h['mae_mc']}")
    print(f"  per_seed_rmse: {h['per_seed_rmse']}")
    print(f"  hypothesis: {h.get('hypothesis')}")
    cfg = h["config"]
    print(f"\n--- config ---\n{json.dumps(cfg, ensure_ascii=False, indent=2)}")
    print(f"\n--- per-target ---")
    for t, info in (h.get("per_target") or {}).items():
        actual = info.get("actual")
        mean = info.get("mean")
        sd = info.get("sd")
        fcs = info.get("forecasts", [])
        err = (mean - actual) if (actual is not None and mean is not None) else None
        print(f"  {t}: actual={actual:.3f}  forecast_mean={mean:.3f}  sd={sd:.3f}  "
              f"err={err:+.3f}  forecasts={fcs}" if mean is not None else
              f"  {t}: actual={actual}  (no forecast)")
    print(f"\n--- rationales (1st seed) ---")
    for t, rat in (h.get("rationales_sample") or {}).items():
        print(f"  [{t}] {rat[:500]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    pr = sub.add_parser("run")
    pr.add_argument("--candidate-file", required=True)
    pr.add_argument("--round", type=int, required=True, dest="round_idx")
    pl = sub.add_parser("lessons")
    pl.add_argument("--lessons-file", required=True)
    pl.add_argument("--round", type=int, required=True, dest="round_idx")
    ps = sub.add_parser("show")
    ps.add_argument("--id", type=int, required=True, dest="candidate_id")
    args = p.parse_args()
    if args.cmd == "info":     _cmd_info()
    elif args.cmd == "run":    _cmd_run(args.candidate_file, args.round_idx)
    elif args.cmd == "lessons": _cmd_lessons(args.lessons_file, args.round_idx)
    elif args.cmd == "show":   _cmd_show_round(args.candidate_id)


if __name__ == "__main__":
    main()
