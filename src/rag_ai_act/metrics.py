"""Metrics and verdict: H1, H2, H3, guard, diagnostics. Pure functions, no API calls.

Question item (eval/*.jsonl, one per line): {"qid", "question", "type": "single"|"multi"|"unanswerable",
"era": "2024"|"2026"|"n/a", "gold_answer", "gold_ids": [...], ...extra keys ignored}.
Verdict record (results/<run>/verdicts_<n>.jsonl): {"blind_id", "correct": bool, "reason": str}.
"""

from __future__ import annotations


def evidence_ok(item: dict, cited_ids: list[str]) -> bool:
    """single: at least one gold ID matched (ids.matches); multi: every gold ID matched; unanswerable: False."""
    raise NotImplementedError


def strict_correct(item: dict, answer: dict, correct: bool) -> bool:
    """answer is an answers_*.jsonl record. True iff item is answerable, answer valid and not declined,
    correct (grader verdict) and evidence_ok."""
    raise NotImplementedError


def recall_share(gold_ids: list[str], ids: list[str]) -> float:
    """Share of gold IDs matched by any of ids (ids.matches). Used for recall@5 (first 5 retrieved_ids)
    and for S3 evidence recall (all retrieved_ids)."""
    raise NotImplementedError


def citation_precision(gold_ids: list[str], cited_ids: list[str]) -> float | None:
    """Share of cited IDs matching some gold ID; None if nothing cited."""
    raise NotImplementedError


def summarize(items: list[dict], answers: dict[str, list[dict]], correct: dict[tuple[str, str], bool] | None) -> dict:
    """answers: system -> answer records; correct: (system, qid) -> grader verdict (answerables only), or
    None before grading: then H1, H2, h1_by_era and multi_hop are None and only mechanical metrics are filled
    (this is what draft_report.md shows). Items with a "paraphrase_of" key are ignored here.

    Returns per system: {"H1": {"n": k, "of": 24, "pct": ...}, "H2": {...era 2026 only}, "H3": {"n", "of"},
    "false_declines": {"n", "of"}, "h1_by_era": {"2024": {...}, "2026": {...}},
    "recall_at_5": float|None (S1, S2), "evidence_recall": float|None (S3), "citation_precision":
    {"mean": float|None, "n_citing_nothing": k} (answerable items with valid answers only), "multi_hop": {"n", "of"}, "invalid": k, "eur": x}.
    H3 counts declined answers among unanswerables (an invalid answer is not a decline); false_declines
    counts declined answers among answerables. Every count dict has n, of and pct (pct = 100*n/of, rounded to 1 decimal)."""
    raise NotImplementedError


def verdict_band(summary: dict) -> str:
    """'useless' | 'good' | 'partial' (checked in that order, first match wins):
    Written as shares so they work for any set size (24 answerable / 6 unanswerable on the held-out set,
    16 / 4 on the dev set); d = (S3 H1 n - S0 H1 n) / of.
    useless: d <= 0.05 (on 24: at most 1 question better, incl. any score below S0), or S3 H1 n <= S1 H1 n.
    good: S3 H1 n / of >= 0.70 (17 of 24) and d >= 0.25 (6 of 24) and S3 H1 n > S1 H1 n and
    S3 H3 n / of >= 5/6 and S3 false_declines n / of <= 2/24.
    partial: otherwise."""
    raise NotImplementedError


def paraphrase_stability(orig: list[bool], para: list[bool]) -> dict:
    """Strict-correct flags of the same 10 items, original vs paraphrase: {"orig": n, "para": n, "within_2": bool}."""
    raise NotImplementedError
