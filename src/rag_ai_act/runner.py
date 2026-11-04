"""Run loop over questions and systems with budget stop. Used by cli eval; testable with fakes."""

from __future__ import annotations

from collections.abc import Callable

from rag_ai_act.budget import Budget
from rag_ai_act.systems import Answer


def run_questions(items: list[dict], systems: dict[str, Callable[[str], Answer]], budget: Budget) -> dict:
    """For each system in order, for each item: call systems[name](item["question"]) and build the answer
    record of systems.py's JSONL format (qid, system, answer fields, latency_s). On BudgetExceeded stop
    everything and keep what was produced. Returns {"answers": {system: [records]}, "stopped": bool,
    "stopped_at": {"system", "qid"} | None}."""
    raise NotImplementedError
