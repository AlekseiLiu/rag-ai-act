import json
import os
import re
from pathlib import Path

import pytest

from rag_ai_act.grading import agreement, export_blind, load_verdicts


def grading_data():
    items = [
        {
            "qid": f"q{n}",
            "question": f"Question number {n}?",
            "gold_answer": f"Gold number {n}.",
            "gold_ids": ["AIA:Art6(2)"],
            "type": "single",
            "era": "2026",
        }
        for n in range(1, 5)
    ]
    items.append(
        {
            "qid": "u1",
            "question": "Unknown?",
            "gold_answer": "Decline",
            "gold_ids": [],
            "type": "unanswerable",
            "era": "n/a",
        }
    )
    answers = {
        s: [
            {
                "qid": i["qid"],
                "system": s,
                "answer": f"Response {offset * 10 + n}",
                "cited_ids": ["AIA:Art6(2)"],
                "declined": False,
                "valid": True,
            }
            for n, i in enumerate(items)
        ]
        for offset, s in enumerate(("S0", "S3"))
    }
    return items, answers


def test_blind_export_hides_labels_excludes_unanswerables_and_key_round_trips(tmp_path):
    items, answers = grading_data()
    blind_path, key_path = export_blind(items, answers, tmp_path, seed=27)
    assert blind_path.name == "blind.jsonl" and key_path.name == "blind_key.json"
    rows = [json.loads(line) for line in blind_path.read_text().splitlines()]
    key = json.loads(key_path.read_text())
    assert key["seed"] == 27 and len(key["map"]) == len(rows) == 8
    assert [r["blind_id"] for r in rows] == [f"B{n:03}" for n in range(1, 9)]
    assert {tuple(pair) for pair in key["map"].values()} == {
        (s, f"q{n}") for s in answers for n in range(1, 5)
    }
    for row in rows:
        assert set(row) == {
            "blind_id",
            "question",
            "gold_answer",
            "gold_ids",
            "answer",
            "cited_ids",
            "declined",
        }
        system, qid = key["map"][row["blind_id"]]
        item = next(i for i in items if i["qid"] == qid)
        answer = next(a for a in answers[system] if a["qid"] == qid)
        for field in ("question", "gold_answer", "gold_ids"):
            assert row[field] == item[field]
        for field in ("answer", "cited_ids", "declined"):
            assert row[field] == answer[field]
    serialized = blind_path.read_text()
    assert not re.search(r"\b(?:S0|S3|q[1-4]|u1)\b", serialized)


def test_blind_shuffle_is_reproducible_and_seed_sensitive(tmp_path):
    items, answers = grading_data()
    exports = []
    for n, seed in enumerate((27, 27, 28, 29)):
        directory = tmp_path / str(n)
        directory.mkdir()
        blind, key = export_blind(items, answers, directory, seed=seed)
        exports.append((blind.read_text(), key.read_text()))
    assert exports[0] == exports[1]
    assert len({pair[0] for pair in exports}) > 1


def test_blind_export_includes_paraphrase_items(tmp_path):
    items, answers = grading_data()
    items.append({**items[0], "qid": "p1", "paraphrase_of": "q1", "question": "Rephrased question?"})
    for system, rows in answers.items():
        rows.append({**rows[0], "qid": "p1", "system": system})
    blind, key = export_blind(items, answers, tmp_path)
    mapping = json.loads(key.read_text())["map"]
    assert {tuple(pair) for pair in mapping.values() if pair[1] == "p1"} == {("S0", "p1"), ("S3", "p1")}
    assert (
        sum(
            row["question"] == "Rephrased question?"
            for row in map(json.loads, blind.read_text().splitlines())
        )
        == 2
    )


@pytest.mark.parametrize("case", ["complete", "unknown", "missing"])
def test_verdict_import_requires_exact_blind_ids(tmp_path, case):
    key = tmp_path / "blind_key.json"
    key.write_text(json.dumps({"seed": 27, "map": {"B001": ["S0", "q1"], "B002": ["S3", "q1"]}}))
    rows = [
        {"blind_id": "B001", "correct": True, "reason": "Supported"},
        {"blind_id": "B002", "correct": False, "reason": "Incorrect"},
    ]
    if case == "missing":
        rows.pop()
    elif case == "unknown":
        rows.append({"blind_id": "B999", "correct": True, "reason": "Unknown"})
    path = tmp_path / "verdicts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if case == "complete":
        assert load_verdicts(path, key) == {("S0", "q1"): True, ("S3", "q1"): False}
    else:
        with pytest.raises(ValueError):
            load_verdicts(path, key)


def test_agreement_counts_disagreements_and_percentage():
    first = {("S0", "q1"): True, ("S1", "q1"): False, ("S2", "q1"): False, ("S3", "q1"): True}
    second = {**first, ("S1", "q1"): True}
    assert agreement(first, second) == {"n": 3, "of": 4, "pct": 75.0, "disagreements": [("S1", "q1")]}


def test_no_evaluation_answers_embedded_in_code():
    directory = os.environ.get("HELDOUT_DIR")
    if not directory:
        pytest.skip("HELDOUT_DIR is unset")
    records = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if {"qid", "question", "gold_answer"} <= row.keys():
                records.append(row)
    assert records, "No evaluation records loaded"
    root = Path(__file__).resolve().parents[1]
    files = [
        p
        for folder in ("src", "tests")
        for p in (root / folder).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ]
    assert files
    for path in files:
        content = path.read_bytes().decode("utf-8", errors="replace")
        for row in records:
            leaked = bool(re.search(r"\b" + re.escape(str(row["qid"])) + r"\b", content))
            leaked |= row["question"] in content or row["gold_answer"] in content
            assert not leaked, f"Evaluation content found in {path.relative_to(root)}"
