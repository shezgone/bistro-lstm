"""Orchestrator — runs N rounds of Researcher → Engineer → Analyzer.

Usage:
    python -m agents.orchestrator --rounds 5
    python -m agents.orchestrator --rounds 1 --dry-run     # propose only, skip HCX call
    python -m agents.orchestrator --rounds 3 --n-seeds 3   # override n_seeds for cheaper iterations

Env required:
  HCX_API_KEY        for HCX-32B (Engineer)
  ANTHROPIC_API_KEY  for Claude Opus (Researcher / Analyzer)
"""
from __future__ import annotations
import argparse
import json
import sys
import time

from . import analyzer, engineer, researcher, store
from .schemas import Candidate


def _print_summary(round_idx: int, cand: Candidate, m, lessons, dry_run: bool) -> None:
    print("\n" + "=" * 78)
    print(f"ROUND {round_idx} SUMMARY")
    print("-" * 78)
    print(f"  hash:        {cand.hash()}")
    print(f"  eval_set:    {cand.eval_set}")
    print(f"  data_panel:  {cand.data_panel}   cov_mode: {cand.cov_mode}   "
          f"ctx_len: {cand.ctx_len}")
    print(f"  blinded:     {cand.blinded}   fewshot_k: {cand.fewshot_k}   "
          f"strategy: {cand.fewshot_strategy}")
    print(f"  n_seeds:     {cand.n_seeds}   T: {cand.temperature}")
    print(f"  hypothesis:  {cand.hypothesis[:200]}")
    if dry_run:
        print("  [dry-run] engineer skipped, no metric.")
    else:
        rmse = f"{m.rmse_mc:.4f}" if m.rmse_mc is not None else "—"
        mae = f"{m.mae_mc:.4f}" if m.mae_mc is not None else "—"
        seeds = ",".join(f"{x:.4f}" for x in m.per_seed_rmse) if m.per_seed_rmse else "—"
        print(f"  status:      {m.status}   RMSE_mc={rmse}   MAE_mc={mae}")
        print(f"  per-seed:    [{seeds}]")
        print(f"  failed:      {m.n_failed}/{m.n_calls}   elapsed: {m.elapsed_sec:.1f}s")
        if m.error:
            print(f"  error:       {m.error}")
    if lessons:
        print(f"  new lessons ({len(lessons)}):")
        for l in lessons:
            print(f"    - [{l.tag or 'other'}] {l.text}")
    else:
        print("  new lessons: (none)")
    print("=" * 78, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=int, default=1, help="how many R→E→A cycles to run")
    p.add_argument("--dry-run", action="store_true",
                   help="researcher only, skip HCX call (for prompt-engineering Researcher itself)")
    p.add_argument("--skip-analyzer", action="store_true",
                   help="run R→E only, no lesson distillation")
    p.add_argument("--n-seeds", type=int, default=None,
                   help="override Researcher's n_seeds (cheaper iterations)")
    p.add_argument("--max-workers", type=int, default=10, help="HCX concurrent workers")
    p.add_argument("--researcher-model", default=None)
    p.add_argument("--analyzer-model", default=None)
    p.add_argument("--eval-set", default=None,
                   choices=["rolling_2025", "stress_2022_h1"],
                   help="hint Researcher to target this eval set (still researcher's choice)")
    args = p.parse_args()

    store.init_db()
    start_round = store.latest_round() + 1
    print(f"[orchestrator] starting from round {start_round}, "
          f"running {args.rounds} cycles", flush=True)

    for i in range(args.rounds):
        round_idx = start_round + i
        print(f"\n[orchestrator] === ROUND {round_idx} ===", flush=True)

        # 1. Researcher
        t0 = time.time()
        try:
            cand = researcher.propose(
                round_idx,
                model=args.researcher_model or researcher.DEFAULT_MODEL,
                eval_set_hint=args.eval_set,
            )
        except Exception as e:
            print(f"[orchestrator] researcher failed: {e}", flush=True)
            break
        if args.n_seeds is not None:
            cand.n_seeds = args.n_seeds
        cand.validate()
        cand_id = store.insert_candidate(round_idx, cand)
        print(f"[orchestrator] researcher proposed (id={cand_id}, hash={cand.hash()}) "
              f"in {time.time()-t0:.1f}s", flush=True)

        # 2. Engineer
        m = None
        if args.dry_run:
            print("[orchestrator] dry-run: skipping engineer", flush=True)
        else:
            try:
                m = engineer.run_candidate(
                    cand, candidate_id=cand_id, max_workers=args.max_workers,
                )
            except Exception as e:
                print(f"[orchestrator] engineer crashed: {e}", flush=True)
                from .schemas import Metric
                m = Metric(candidate_id=cand_id, status="failed",
                           rmse_mc=None, mae_mc=None, error=str(e)[:200])
            store.insert_metric(m)

        # 3. Analyzer
        new_lessons = []
        if not args.skip_analyzer and m is not None and m.status != "failed":
            try:
                new_lessons = analyzer.analyze(
                    round_idx, cand_id,
                    model=args.analyzer_model or analyzer.DEFAULT_MODEL,
                )
                if new_lessons:
                    store.append_lessons(new_lessons)
            except Exception as e:
                print(f"[orchestrator] analyzer failed: {e}", flush=True)

        _print_summary(round_idx, cand, m, new_lessons, dry_run=args.dry_run)

    # Final ranking snapshot
    print("\n[orchestrator] final ranking (top 10 by RMSE_mc):")
    history = store.fetch_history(limit=10)
    for h in history:
        rmse = f"{h['rmse_mc']:.4f}" if h["rmse_mc"] is not None else "—"
        cfg = h["config"]
        print(f"  id={h['id']:>3} round={h['round_idx']:>2} RMSE={rmse} "
              f"eval={cfg.get('eval_set')} data={cfg.get('data_panel')} "
              f"cov={cfg.get('cov_mode')} blinded={cfg.get('blinded')} "
              f"fs_k={cfg.get('fewshot_k')}/{cfg.get('fewshot_strategy')}")


if __name__ == "__main__":
    main()
