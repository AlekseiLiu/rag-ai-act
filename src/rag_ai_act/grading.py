"""Blind grading export and import. Grading itself runs outside.

export_blind writes, into out_dir:
  blind.jsonl     one per (system, answerable qid): {"blind_id", "question", "gold_answer", "gold_ids",
                  "answer", "cited_ids", "declined"}; no system name, no qid; order shuffled with `seed`
  blind_key.json  {"seed": int, "map": {blind_id: [system, qid]}}
blind_id = "B" + zero-padded running number in shuffled order (e.g. "B001").
"""

from __future__ import annotations

from pathlib import Path


def export_blind(items: list[dict], answers: dict[str, list[dict]], out_dir: Path, seed: int = 20261104) -> tuple[Path, Path]:
    """Unanswerable items are not exported (their score is mechanical). Returns (blind.jsonl, blind_key.json)."""
    raise NotImplementedError


def load_verdicts(verdicts_path: Path, key_path: Path) -> dict[tuple[str, str], bool]:
    """Map verdict records back to (system, qid) -> correct. Raises ValueError on unknown or missing blind_ids."""
    raise NotImplementedError


def agreement(v1: dict[tuple[str, str], bool], v2: dict[tuple[str, str], bool]) -> dict:
    """{"n": agreeing, "of": total, "pct": ..., "disagreements": [(system, qid), ...]}."""
    raise NotImplementedError
