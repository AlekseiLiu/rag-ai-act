"""Run loop over questions and systems with budget stop. Used by cli eval; testable with fakes."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict

from rag_ai_act.budget import Budget, BudgetExceeded
from rag_ai_act.systems import Answer


def run_questions(items: list[dict], systems: dict[str, Callable[[str], Answer]], budget: Budget) -> dict:
    """For each system in order, for each item: call systems[name](item["question"]) and build the answer
    record of systems.py's JSONL format (qid, system, answer fields, latency_s). On BudgetExceeded stop
    everything and keep what was produced. Returns {"answers": {system: [records]}, "stopped": bool,
    "stopped_at": {"system", "qid"} | None}."""
    answers: dict[str, list[dict]] = {name: [] for name in systems}
    for name, run in systems.items():
        for item in items:
            start = time.perf_counter()
            try:
                result = run(item["question"])
            except BudgetExceeded:
                return {"answers": answers, "stopped": True, "stopped_at": {"system": name, "qid": item["qid"]}}
            answers[name].append(
                {"qid": item["qid"], "system": name, **asdict(result), "latency_s": time.perf_counter() - start}
            )
    return {"answers": answers, "stopped": False, "stopped_at": None}
