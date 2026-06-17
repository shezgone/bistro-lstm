"""Candidate / Metric / Lesson schemas shared across agents."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import hashlib
import json

# Allowed enums (kept tight so Researcher cannot break the runner)
DATA_PANELS = ("optimal18", "full_no_gt", "full_with_gt", "optimal18_aug")
COV_MODES = ("all", "univar", "subset")
EVAL_SETS = ("rolling_2025", "stress_2022_h1")
FEWSHOT_STRATEGIES = ("recent", "diverse_regime")


@dataclass
class Candidate:
    """A single HCX in-context configuration to be evaluated.

    Structural fields shape the runner; `system_prompt` and `user_prompt_template`
    are free-form text the Researcher writes. Placeholders inside templates:
        {table}, {origin}, {target}, {fewshot_block}, {cols_summary}.
    A fixed JSON output instruction is appended by the engineer regardless.

    Locked levers (per the 16 portable lessons; see cognition.json):
      - table_format is TSV-only (Minimal Prompt Principle: any user-msg decoration
        — markdown, units headers, lag columns, examples — degraded RMSE).
        If you want format variation, vary `system_prompt` instead.
    """
    eval_set: str              # one of EVAL_SETS — which rolling evaluation set to score on
    data_panel: str            # one of DATA_PANELS
    cov_mode: str              # one of COV_MODES
    cov_subset: Optional[list] # list[str] when cov_mode=="subset" else None
    ctx_len: int               # months of history per query
    blinded: bool              # var_N + t=N anonymization (probe / contamination test)
    fewshot_k: int             # 0/3/6/12
    fewshot_strategy: str      # one of FEWSHOT_STRATEGIES — only used when fewshot_k > 0
    fewshot_panel_len: int     # months per fewshot example
    system_prompt: str         # full HCX system message body (no JSON output line)
    user_prompt_template: str  # template, see placeholders above
    n_seeds: int = 5
    temperature: float = 0.7
    max_tokens: int = 8192
    hypothesis: str = ""

    def validate(self) -> None:
        if self.eval_set not in EVAL_SETS:
            raise ValueError(f"eval_set must be one of {EVAL_SETS}")
        if self.data_panel not in DATA_PANELS:
            raise ValueError(f"data_panel must be one of {DATA_PANELS}")
        if self.cov_mode not in COV_MODES:
            raise ValueError(f"cov_mode must be one of {COV_MODES}")
        if self.cov_mode == "subset" and not self.cov_subset:
            raise ValueError("cov_subset required when cov_mode=='subset'")
        if self.fewshot_strategy not in FEWSHOT_STRATEGIES:
            raise ValueError(f"fewshot_strategy must be one of {FEWSHOT_STRATEGIES}")
        if self.fewshot_k not in (0, 3, 6, 12):
            raise ValueError("fewshot_k must be 0, 3, 6, or 12")
        if self.ctx_len < 12 or self.ctx_len > 60:
            raise ValueError("ctx_len must be in [12, 60]")
        if self.n_seeds < 1 or self.n_seeds > 10:
            raise ValueError("n_seeds must be in [1, 10]")
        if not (0.0 <= self.temperature <= 1.5):
            raise ValueError("temperature out of range")
        if "{table}" not in self.user_prompt_template:
            raise ValueError("user_prompt_template must contain {table}")
        if self.fewshot_k > 0 and "{fewshot_block}" not in self.user_prompt_template:
            raise ValueError("user_prompt_template must contain {fewshot_block} when fewshot_k > 0")
        if not self.system_prompt.strip():
            raise ValueError("system_prompt is empty")

    def canonical_dict(self) -> dict:
        d = asdict(self)
        d.pop("hypothesis", None)
        return d

    def hash(self) -> str:
        s = json.dumps(self.canonical_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        kwargs = {f.name: d[f.name] for f in cls.__dataclass_fields__.values() if f.name in d}
        return cls(**kwargs)


@dataclass
class Metric:
    candidate_id: int
    status: str
    rmse_mc: Optional[float]
    mae_mc: Optional[float]
    per_seed_rmse: list = field(default_factory=list)
    per_target: dict = field(default_factory=dict)
    rationales_sample: dict = field(default_factory=dict)
    n_calls: int = 0
    n_failed: int = 0
    elapsed_sec: float = 0.0
    error: Optional[str] = None


@dataclass
class Lesson:
    round_idx: int
    text: str
    evidence: str = ""
    tag: str = ""


def best_known_baseline_summary(eval_set: str = "rolling_2025") -> str:
    """Eval-set-aware grounding for the Researcher (frozen historical numbers).

    For new eval sets without prior LLM history, only deterministic baselines are listed.
    """
    if eval_set == "rolling_2025":
        return (
            "Prior HCX-32B-Think rolling-2025 results (8 origins 2025-04..2025-11, 5 seeds; "
            "lower RMSE = better):\n"
            "  HCX cov_csi_forced (18+CSI, 4-step CSI CoT) ........ RMSE_mc 0.250  ← best single LLM\n"
            "  HCX critique (Reflexion-style CoT)  ................. RMSE_mc 0.243\n"
            "  HCX stepback CoT  ................................... RMSE_mc 0.253\n"
            "  HCX cov_base (18, mild prompt) ...................... RMSE_mc 0.296\n"
            "  HCX naive few-shot k=3 (recent, +CSI CoT) ........... RMSE_mc 0.268  ← worse, anchoring\n"
            "  HCX format header (Date+units) ...................... RMSE_mc 0.279  ← MPP violator\n"
            "  HCX blinded (var_N, t=N, generic CoT) ............... RMSE_mc 0.315  ← contamination probe\n"
            "Deterministic baselines on the same split:\n"
            "  Trend12 .............................................  RMSE 0.2575\n"
            "  AR(1)  ..............................................  RMSE 0.2641\n"
            "  Random Walk .........................................  RMSE 0.2690\n"
            "Ensemble (post-hoc): HCX 50% + Trend12 50% ............  RMSE 0.235  ← current SOTA\n"
            "Goal: per-seed RMSE < 0.25 (Tier 1). Tier 1 ceiling is well-mapped."
        )
    if eval_set == "stress_2022_h1":
        return (
            "Eval set: 2022-H1 inflation surge (8 origins 2021-12..2022-07, "
            "targets 2022-01..2022-08, CPI 3.77% → 5.72% with peak 6.33% in Jul; "
            "BoK_CSI 104 → 86 — INVERSE correlation vs 2025).\n"
            "Deterministic baselines (computed on this exact split):\n"
            "  Random Walk .........................................  RMSE 0.4706  ← best simple baseline here\n"
            "  AR(1)  ..............................................  RMSE 0.5199\n"
            "  Trend12 .............................................  RMSE 0.5338  (linear extrap can't keep up with surge)\n"
            "  MA12  ...............................................  RMSE 1.8024  (lagging mean is catastrophic)\n"
            "Note: this baseline ladder INVERTS vs rolling_2025. RW is strongest "
            "because surge is near-monotonic from origin; Trend12 / MA12 underestimate.\n"
            "No prior HCX rolling results on this set yet — UNEXPLORED prompt territory.\n"
            "Open hypotheses worth testing here:\n"
            "  - Forced 4-step CSI CoT may MISFIRE because CSI fell while CPI rose\n"
            "    (need sign-agnostic CSI framing: 'cite values, interpret direction conditionally')\n"
            "  - Diverse-regime few-shot (rising + falling + stable) may anchor better than recent\n"
            "  - Combined critique+stepback+forced may exceed single scaffolds in stress regime\n"
            "  - Goal: beat RW 0.4706 with HCX (so far on stress 2023, blinded HCX 1.80 vs Trend12 3.33).\n"
        )
    return f"(no baseline summary for eval_set={eval_set!r})"
