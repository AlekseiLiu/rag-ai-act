"""Metrics and verdict: H1, H2, H3, guard, diagnostics. Pure functions, no API calls.

Question item (eval/*.jsonl, one per line): {"qid", "question", "type": "single"|"multi"|"unanswerable",
"era": "2024"|"2026"|"n/a", "gold_answer", "gold_ids": [...], ...extra keys ignored}.
Verdict record (results/<run>/verdicts_<n>.jsonl): {"blind_id", "correct": bool, "reason": str}.
"""

from __future__ import annotations

from rag_ai_act.ids import matches


def evidence_ok(item: dict, cited_ids: list[str]) -> bool:
    """single: at least one gold ID matched (ids.matches); multi: every gold ID matched; unanswerable: False."""
    gold = item.get("gold_ids", [])
    if item.get("type") == "unanswerable" or not gold:
        return False
    hits = [any(matches(c, g) for c in cited_ids) for g in gold]
    return all(hits) if item["type"] == "multi" else any(hits)


def strict_correct(item: dict, answer: dict, correct: bool) -> bool:
    """answer is an answers_*.jsonl record. True iff item is answerable, answer valid and not declined,
    correct (grader verdict) and evidence_ok."""
    if item.get("type") == "unanswerable":
        return False
    if not answer["valid"] or answer["declined"] or not correct:
        return False
    return evidence_ok(item, answer["cited_ids"])


def recall_share(gold_ids: list[str], ids: list[str]) -> float:
    """Share of gold IDs matched by any of ids (ids.matches). Used for recall@5 (first 5 retrieved_ids)
    and for S3 evidence recall (all retrieved_ids)."""
    if not gold_ids:
        return 0.0
    return sum(any(matches(i, g) for i in ids) for g in gold_ids) / len(gold_ids)


def citation_precision(gold_ids: list[str], cited_ids: list[str]) -> float | None:
    """Share of cited IDs matching some gold ID; None if nothing cited."""
    if not cited_ids:
        return None
    return sum(any(matches(c, g) for g in gold_ids) for c in cited_ids) / len(cited_ids)


def summarize(items: list[dict], answers: dict[str, list[dict]], correct: dict[tuple[str, str], bool] | None) -> dict:
    """answers: system -> answer records; correct: (system, qid) -> grader verdict (answerables only), or
    None before grading: then H1, H2, h1_by_era and multi_hop are None and only mechanical metrics are filled
    (this is what draft_report.md shows). Items with a "paraphrase_of" key are ignored here.

    Returns per system: {"H1": {"n": k, "of": 24, "pct": ...}, "H2": {...era 2026 only}, "H3": {"n", "of"},
    "false_declines": {"n", "of"}, "h1_by_era": {"2024": {...}, "2026": {...}},
    "recall_at_5": float|None (S1, S2), "evidence_recall": float|None (S3), "citation_precision":
    {"mean": float|None, "n_citing_nothing": k} (answerable items with valid answers only), "multi_hop": {"n", "of"}, "invalid": k, "eur": x}.
    H3 counts declined answers among unanswerables (an invalid answer is not a decline); false_declines
    counts declined answers among answerables. Every count dict has n, of and pct (pct = 100*n/of, rounded to 1 decimal).

    Missing answers (a run stopped at the budget) count as failures in every metric, and every denominator
    is the full item set: a missing answer is never correct, never a decline of an unanswerable question,
    counts as a false decline of an answerable one, and retrieves nothing (recall 0). "missing" gives their
    number per system; "invalid" counts invalid answers that were given. Citation precision covers the
    valid answers that were given."""
    main = [it for it in items if "paraphrase_of" not in it]
    answerable = [it for it in main if it["type"] != "unanswerable"]
    unanswerable = [it for it in main if it["type"] == "unanswerable"]
    out = {}
    for system, records in answers.items():
        by_qid = {r["qid"]: r for r in records}
        # A missing answer stands in as an invalid, non-declined answer that cites and retrieves nothing.
        pairs = [(it, by_qid.get(it["qid"]) or _missing(it["qid"])) for it in answerable]
        upairs = [(it, by_qid.get(it["qid"]) or _missing(it["qid"])) for it in unanswerable]
        graded = correct is not None

        def strict(it, rec, system=system):
            return strict_correct(it, rec, correct.get((system, it["qid"]), False))

        def count(n, of):
            return {"n": n, "of": of, "pct": round(100 * n / of, 1) if of else 0.0}

        def h1(subset):
            return count(sum(strict(it, rec) for it, rec in subset), len(subset))

        def declined(rec):
            return rec["declined"] and rec["valid"]

        def false_decline(rec):
            return declined(rec) or rec.get("missing", False)

        precisions = [citation_precision(it["gold_ids"], rec["cited_ids"]) for it, rec in pairs if rec["valid"]]
        cited = [p for p in precisions if p is not None]
        recall = None
        if system in ("S1", "S2") and pairs:
            recall = sum(recall_share(it["gold_ids"], rec["retrieved_ids"][:5]) for it, rec in pairs) / len(pairs)
        evidence = None
        if system == "S3" and pairs:
            evidence = sum(recall_share(it["gold_ids"], rec["retrieved_ids"]) for it, rec in pairs) / len(pairs)
        everything = pairs + upairs
        out[system] = {
            "H1": h1(pairs) if graded else None,
            "H2": h1([p for p in pairs if p[0]["era"] == "2026"]) if graded else None,
            "H3": count(sum(declined(rec) for _, rec in upairs), len(upairs)),
            "false_declines": count(sum(false_decline(rec) for _, rec in pairs), len(pairs)),
            "h1_by_era": {
                era: h1([p for p in pairs if p[0]["era"] == era]) for era in ("2024", "2026")
            }
            if graded
            else None,
            "recall_at_5": recall,
            "evidence_recall": evidence,
            "citation_precision": {
                "mean": sum(cited) / len(cited) if cited else None,
                "n_citing_nothing": len(precisions) - len(cited),
            },
            "multi_hop": h1([p for p in pairs if p[0]["type"] == "multi"]) if graded else None,
            "invalid": sum(not rec["valid"] and not rec.get("missing", False) for _, rec in everything),
            "missing": sum(rec.get("missing", False) for _, rec in everything),
            "eur": sum(rec.get("usage", {}).get("eur", 0) for _, rec in everything),
        }
    return out


def _missing(qid: str) -> dict:
    """Stand-in record for an answer that was never produced: invalid, not declined, nothing cited."""
    return {"qid": qid, "answer": "", "cited_ids": [], "declined": False, "valid": False, "missing": True,
            "retrieved_ids": [], "trace": [], "usage": {}, "latency_s": 0.0}


def missing_answers(items: list[dict], answers: dict[str, list[dict]]) -> list[tuple[str, str]]:
    """(system, qid) pairs of every item, paraphrases included, that has no answer record for a system."""
    out = []
    for system, records in answers.items():
        have = {r["qid"] for r in records}
        out.extend((system, it["qid"]) for it in items if it["qid"] not in have)
    return out


def verdict_band(summary: dict) -> str:
    """'useless' | 'good' | 'partial' (checked in that order, first match wins):
    Written as shares so they work for any set size (24 answerable / 6 unanswerable on the held-out set,
    16 / 4 on the dev set); d = (S3 H1 n - S0 H1 n) / of.
    useless: d <= 0.05 (on 24: at most 1 question better, incl. any score below S0), or S3 H1 n <= S1 H1 n.
    good: S3 H1 n / of >= 0.70 (17 of 24) and d >= 0.25 (6 of 24) and S3 H1 n > S1 H1 n and
    S3 H3 n / of >= 5/6 and S3 false_declines n / of <= 2/24.
    partial: otherwise."""
    s0, s1, s3 = summary["S0"]["H1"], summary["S1"]["H1"], summary["S3"]["H1"]
    h3, fd = summary["S3"]["H3"], summary["S3"]["false_declines"]
    gain = s3["n"] - s0["n"]
    # Integer comparisons of the shares in the docstring (no float edge cases).
    if gain * 20 <= s3["of"] or s3["n"] <= s1["n"]:
        return "useless"
    if (
        s3["n"] * 10 >= 7 * s3["of"]
        and gain * 4 >= s3["of"]
        and h3["n"] * 6 >= 5 * h3["of"]
        and fd["n"] * 12 <= fd["of"]
    ):
        return "good"
    return "partial"


def paraphrase_stability(orig: list[bool], para: list[bool]) -> dict:
    """Strict-correct flags of the same 10 items, original vs paraphrase: {"orig": n, "para": n, "within_2": bool}."""
    o, p = sum(orig), sum(para)
    return {"orig": o, "para": p, "within_2": abs(o - p) <= 2}
