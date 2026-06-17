"""SQLite (experiments.db) + JSON (cognition.json) persistence.

experiments.db
  candidates(id, round_idx, config_json, config_hash UNIQUE, hypothesis, created_at)
  metrics(candidate_id PK, status, rmse_mc, mae_mc, per_seed_json, per_target_json,
          rationales_json, n_calls, n_failed, elapsed_sec, error, completed_at)

cognition.json
  {"lessons": [{round_idx, text, evidence, tag, created_at}, ...]}
"""
from __future__ import annotations
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .schemas import Candidate, Lesson, Metric

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "experiments.db"
COG_PATH = ROOT / "data" / "cognition.json"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            round_idx INTEGER NOT NULL,
            config_json TEXT NOT NULL,
            config_hash TEXT NOT NULL UNIQUE,
            hypothesis TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS metrics (
            candidate_id INTEGER PRIMARY KEY,
            status TEXT NOT NULL,
            rmse_mc REAL,
            mae_mc REAL,
            per_seed_json TEXT,
            per_target_json TEXT,
            rationales_json TEXT,
            n_calls INTEGER,
            n_failed INTEGER,
            elapsed_sec REAL,
            error TEXT,
            completed_at TEXT,
            FOREIGN KEY(candidate_id) REFERENCES candidates(id)
        );
        """)
    if not COG_PATH.exists():
        COG_PATH.parent.mkdir(parents=True, exist_ok=True)
        COG_PATH.write_text(json.dumps({"lessons": []}, ensure_ascii=False, indent=2))


def has_config_hash(h: str) -> bool:
    with _conn() as c:
        row = c.execute("SELECT 1 FROM candidates WHERE config_hash = ?", (h,)).fetchone()
        return row is not None


def insert_candidate(round_idx: int, cand: Candidate) -> int:
    cand.validate()
    h = cand.hash()
    if has_config_hash(h):
        raise ValueError(f"duplicate candidate hash {h}")
    cfg = json.dumps(cand.canonical_dict(), ensure_ascii=False)
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO candidates(round_idx, config_json, config_hash, hypothesis, created_at) "
            "VALUES (?,?,?,?,?)",
            (round_idx, cfg, h, cand.hypothesis, datetime.utcnow().isoformat()),
        )
        return cur.lastrowid


def insert_metric(m: Metric) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO metrics("
            "candidate_id, status, rmse_mc, mae_mc, per_seed_json, per_target_json, "
            "rationales_json, n_calls, n_failed, elapsed_sec, error, completed_at"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                m.candidate_id, m.status, m.rmse_mc, m.mae_mc,
                json.dumps(m.per_seed_rmse),
                json.dumps(m.per_target, ensure_ascii=False),
                json.dumps(m.rationales_sample, ensure_ascii=False),
                m.n_calls, m.n_failed, m.elapsed_sec, m.error,
                datetime.utcnow().isoformat(),
            ),
        )


def fetch_history(limit: int = 50) -> list[dict]:
    """Return list of {round_idx, candidate, metric} dicts sorted by RMSE asc (best first).
    Failed runs are appended last."""
    sql = """
    SELECT c.id, c.round_idx, c.config_json, c.hypothesis, c.config_hash,
           m.status, m.rmse_mc, m.mae_mc, m.per_seed_json, m.per_target_json,
           m.rationales_json, m.n_failed, m.error
    FROM candidates c LEFT JOIN metrics m ON c.id = m.candidate_id
    ORDER BY c.id ASC
    """
    with _conn() as c:
        rows = c.execute(sql).fetchall()
    out = []
    for r in rows:
        cfg = json.loads(r["config_json"])
        per_seed = json.loads(r["per_seed_json"]) if r["per_seed_json"] else []
        per_target = json.loads(r["per_target_json"]) if r["per_target_json"] else {}
        rats = json.loads(r["rationales_json"]) if r["rationales_json"] else {}
        out.append({
            "id": r["id"],
            "round_idx": r["round_idx"],
            "config_hash": r["config_hash"],
            "hypothesis": r["hypothesis"] or "",
            "config": cfg,
            "status": r["status"],
            "rmse_mc": r["rmse_mc"],
            "mae_mc": r["mae_mc"],
            "per_seed_rmse": per_seed,
            "per_target": per_target,
            "rationales_sample": rats,
            "n_failed": r["n_failed"] or 0,
            "error": r["error"],
        })
    # sort: completed first by RMSE, then partial/failed
    def _key(x):
        if x["status"] == "ok" and x["rmse_mc"] is not None:
            return (0, x["rmse_mc"])
        if x["status"] == "partial" and x["rmse_mc"] is not None:
            return (1, x["rmse_mc"])
        return (2, 1e9)
    out.sort(key=_key)
    return out[:limit]


def latest_round() -> int:
    with _conn() as c:
        row = c.execute("SELECT MAX(round_idx) AS r FROM candidates").fetchone()
        return int(row["r"]) if row["r"] is not None else -1


def load_lessons() -> list[Lesson]:
    if not COG_PATH.exists():
        return []
    obj = json.loads(COG_PATH.read_text())
    out = []
    for d in obj.get("lessons", []):
        out.append(Lesson(
            round_idx=d.get("round_idx", -1),
            text=d.get("text", ""),
            evidence=d.get("evidence", ""),
            tag=d.get("tag", ""),
        ))
    return out


def append_lessons(new: list[Lesson]) -> None:
    obj = {"lessons": []}
    if COG_PATH.exists():
        obj = json.loads(COG_PATH.read_text())
    now = datetime.utcnow().isoformat()
    for l in new:
        d = asdict(l)
        d["created_at"] = now
        obj.setdefault("lessons", []).append(d)
    COG_PATH.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
