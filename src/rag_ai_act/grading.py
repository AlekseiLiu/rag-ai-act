"""Blind grading export and import. Grading itself runs outside.

export_blind writes, into out_dir:
  blind.jsonl     one per (system, answerable qid): {"blind_id", "question", "gold_answer", "gold_ids",
                  "answer", "cited_ids", "declined"}; no system name, no qid; order shuffled with `seed`
  blind_key.json  {"seed": int, "map": {blind_id: [system, qid]}}
blind_id = "B" + zero-padded running number in shuffled order (e.g. "B001").

blind.jsonl still holds the gold and cited IDs, so it is not what the grader sees. export_stripped writes
the grader's input, blind_stripped.jsonl, with only STRIPPED_KEYS per line, into a folder of its own.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

STRIPPED_KEYS = ("blind_id", "question", "gold_answer", "answer", "declined")
STRIPPED_FILE = "blind_stripped.jsonl"


def export_blind(items: list[dict], answers: dict[str, list[dict]], out_dir: Path, seed: int = 20261104) -> tuple[Path, Path]:
    """items may include paraphrase items (they need verdicts too). Unanswerable items are not exported
    (their score is mechanical). Returns (blind.jsonl, blind_key.json)."""
    entries = []
    for system, records in answers.items():
        by_qid = {r["qid"]: r for r in records}
        for item in items:
            if item["type"] != "unanswerable" and item["qid"] in by_qid:
                entries.append((system, item, by_qid[item["qid"]]))
    random.Random(seed).shuffle(entries)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, mapping = [], {}
    for n, (system, item, rec) in enumerate(entries, 1):
        blind_id = f"B{n:03}"
        mapping[blind_id] = [system, item["qid"]]
        rows.append(
            {
                "blind_id": blind_id,
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "gold_ids": item["gold_ids"],
                "answer": rec["answer"],
                "cited_ids": rec["cited_ids"],
                "declined": rec["declined"],
            }
        )
    blind_path, key_path = out_dir / "blind.jsonl", out_dir / "blind_key.json"
    blind_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    key_path.write_text(json.dumps({"seed": seed, "map": mapping}, indent=1), encoding="utf-8")
    return blind_path, key_path


def export_stripped(blind_path: Path, out_dir: Path) -> Path:
    """Write out_dir/blind_stripped.jsonl: every line of blind.jsonl reduced to STRIPPED_KEYS (no gold IDs,
    no cited IDs), in the same order. out_dir is created if needed and must be empty; it must not be the
    results folder holding blind.jsonl or lie inside it, so a grader working in out_dir cannot see the key
    or the answer files. Raises ValueError otherwise. Returns the written path."""
    blind_path, out_dir = Path(blind_path).resolve(), Path(out_dir).resolve()
    results_dir = blind_path.parent
    if out_dir == results_dir or results_dir in out_dir.parents:
        raise ValueError(f"{out_dir} is inside the results folder; choose a folder outside it")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"{out_dir} is not an empty folder")
    rows = []
    for line in blind_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append({k: row[k] for k in STRIPPED_KEYS})
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / STRIPPED_FILE
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def load_verdicts(verdicts_path: Path, key_path: Path) -> dict[tuple[str, str], bool]:
    """Map verdict records back to (system, qid) -> correct. Raises ValueError on unknown or missing
    blind_ids, and when "correct" is missing or not a JSON boolean (the string "false" is rejected, not
    read as true)."""
    mapping = json.loads(key_path.read_text(encoding="utf-8"))["map"]
    result: dict[tuple[str, str], bool] = {}
    seen: set[str] = set()
    for line in verdicts_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        blind_id = row["blind_id"]
        if blind_id not in mapping:
            raise ValueError(f"unknown blind_id {blind_id}")
        if blind_id in seen:
            raise ValueError(f"duplicate blind_id {blind_id}")
        seen.add(blind_id)
        correct = row.get("correct")
        if not isinstance(correct, bool):
            # ValueError like every other malformed verdict file, so callers handle one exception type
            raise ValueError(f"verdict for {blind_id}: \"correct\" must be true or false, got {correct!r}")  # noqa: TRY004
        system, qid = mapping[blind_id]
        result[(system, qid)] = correct
    missing = sorted(set(mapping) - seen)
    if missing:
        raise ValueError(f"missing verdicts for blind_ids {missing}")
    return result


def agreement(v1: dict[tuple[str, str], bool], v2: dict[tuple[str, str], bool]) -> dict:
    """{"n": agreeing, "of": total, "pct": round(100*n/of, 1), "disagreements": [(system, qid), ...]}."""
    if v1.keys() != v2.keys():
        raise ValueError("verdict sets cover different (system, qid) pairs")
    disagreements = sorted(k for k in v1 if v1[k] != v2[k])
    of = len(v1)
    n = of - len(disagreements)
    return {"n": n, "of": of, "pct": round(100 * n / of, 1) if of else 0.0, "disagreements": disagreements}
