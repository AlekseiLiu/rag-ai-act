"""The report command on a tiny synthetic results folder with hand-computed numbers."""

import json
from pathlib import Path

import pytest

from rag_ai_act.cli import main

A, B, C, D = "AIA:Art6(2)", "AIA:Art5(1)", "GLHR:para204", "AIA:Art113(3)"

ITEMS = [
    {"qid": "Q1", "question": "q1", "type": "single", "era": "2024", "gold_answer": "a1", "gold_ids": [A]},
    {"qid": "Q2", "question": "q2", "type": "single", "era": "2026", "gold_answer": "a2", "gold_ids": [D]},
    {"qid": "Q3", "question": "q3", "type": "single", "era": "2026", "gold_answer": "a3", "gold_ids": [C]},
    {"qid": "Q4", "question": "q4", "type": "multi", "era": "2026", "gold_answer": "a4", "gold_ids": [A, B]},
    {"qid": "U1", "question": "u1", "type": "unanswerable", "era": "n/a", "gold_answer": "none", "gold_ids": []},
    {"qid": "P2", "question": "p2", "type": "single", "era": "2026", "gold_answer": "a2", "gold_ids": [D],
     "paraphrase_of": "Q2"},
]


def rec(system, qid, cited, declined=False):
    return {"qid": qid, "system": system, "answer": "x", "cited_ids": cited, "declined": declined, "valid": True,
            "retrieved_ids": [], "trace": [], "usage": {"eur": 0.01}, "latency_s": 0.1}


# cited IDs per system and qid
CITED = {
    "S0": {"Q1": [], "Q2": [], "Q3": [], "Q4": [], "P2": []},
    "S1": {"Q1": [A], "Q2": [D], "Q3": [], "Q4": [], "P2": []},
    "S2": {"Q1": [A], "Q2": [], "Q3": [], "Q4": [], "P2": []},
    "S3": {"Q1": [A], "Q2": [D], "Q3": [C], "Q4": [A], "P2": [C]},  # Q4 needs A and B; P2 cites the wrong ID
}
DECLINED_U1 = {"S0": False, "S1": False, "S2": False, "S3": True}

# grader verdicts after the author settled disagreements (verdicts_final)
FINAL = {
    "S0": {"Q1": True, "Q2": True, "Q3": False, "Q4": False, "P2": True},
    "S1": {"Q1": True, "Q2": False, "Q3": False, "Q4": False, "P2": False},
    "S2": {"Q1": True, "Q2": False, "Q3": False, "Q4": False, "P2": False},
    "S3": {"Q1": True, "Q2": True, "Q3": True, "Q4": True, "P2": True},
}
# pass 1 differs from final in one cell; pass 2 differs from pass 1 in two cells
PASS1 = {s: dict(v) for s, v in FINAL.items()}
PASS1["S1"]["Q2"] = True
PASS2 = {s: dict(v) for s, v in PASS1.items()}
PASS2["S0"]["Q3"] = True
PASS2["S2"]["Q1"] = False


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def results(tmp_path):
    write_jsonl(tmp_path / "items.jsonl", ITEMS)
    mapping = {}
    for system, cited in CITED.items():
        rows = [rec(system, qid, ids) for qid, ids in cited.items()]
        rows.append(rec(system, "U1", [], declined=DECLINED_U1[system]))
        write_jsonl(tmp_path / f"answers_{system}.jsonl", rows)
        for qid in cited:
            mapping[f"B{len(mapping) + 1:03}"] = [system, qid]
    (tmp_path / "blind_key.json").write_text(json.dumps({"seed": 1, "map": mapping}), encoding="utf-8")

    def verdicts(table):
        return [{"blind_id": b, "correct": table[s][q], "reason": ""} for b, (s, q) in mapping.items()]

    write_jsonl(tmp_path / "verdicts_1.jsonl", verdicts(PASS1))
    write_jsonl(tmp_path / "verdicts_2.jsonl", verdicts(PASS2))
    write_jsonl(tmp_path / "verdicts_final.jsonl", verdicts(FINAL))
    return tmp_path


def run_report(folder: Path) -> dict:
    assert main(["report", "--results", str(folder)]) == 0
    assert (folder / "report.md").read_text(encoding="utf-8").strip()
    return json.loads((folder / "report.json").read_text(encoding="utf-8"))


def test_report_headline_numbers_from_final_verdicts(results):
    rep = run_report(results)
    assert rep["settled"] is True and rep["verdicts_source"] == "verdicts_final"
    s = rep["summary"]
    # S3: strict correct on Q1, Q2, Q3 (Q4 cites only one of its two gold IDs): H1 3/4; 2026 items Q2..Q4: 2/3
    assert (s["S3"]["H1"]["n"], s["S3"]["H1"]["of"], s["S3"]["H1"]["pct"]) == (3, 4, 75.0)
    assert (s["S3"]["H2"]["n"], s["S3"]["H2"]["of"]) == (2, 3)
    assert (s["S3"]["H3"]["n"], s["S3"]["H3"]["of"]) == (1, 1)
    assert (s["S3"]["false_declines"]["n"], s["S3"]["false_declines"]["of"]) == (0, 4)
    # S0 cites nothing, so strict accuracy is 0; S1 is right on Q1 only (Q2 was settled to incorrect)
    assert s["S0"]["H1"]["n"] == 0 and s["S1"]["H1"]["n"] == 1 and s["S2"]["H1"]["n"] == 1
    assert s["S0"]["H3"]["n"] == 0 and s["S1"]["H3"]["n"] == 0


def test_report_verdict_band(results):
    # S3 3/4 >= 0.70, gain over S0 3/4 >= 0.25, above S1, H3 1/1, no false declines
    assert run_report(results)["band"] == "good"


def test_report_answer_only_correctness_next_to_strict(results):
    ao = run_report(results)["answer_only"]
    assert (ao["S0"]["all"]["n"], ao["S0"]["all"]["of"]) == (2, 4)  # Q1, Q2 correct; paraphrase excluded
    assert (ao["S0"]["2024"]["n"], ao["S0"]["2024"]["of"]) == (1, 1)
    assert (ao["S0"]["2026"]["n"], ao["S0"]["2026"]["of"]) == (1, 3)
    assert ao["S3"]["all"]["n"] == 4 and ao["S3"]["2026"]["n"] == 3


def test_report_agreement_and_paraphrase_stability(results):
    rep = run_report(results)
    # 5 graded cells per system x 4 systems = 20; pass 1 and pass 2 differ in two
    a = rep["agreement"]
    assert (a["n"], a["of"], a["pct"]) == (18, 20, 90.0)
    assert sorted(a["disagreements"]) == [["S0", "Q3"], ["S2", "Q1"]]
    # S3: Q2 strict correct, paraphrase P2 cites the wrong ID; S0 correct on neither strictly
    st = rep["paraphrase_stability"]
    assert (st["S3"]["orig"], st["S3"]["para"], st["S3"]["within_2"]) == (1, 0, True)
    assert (st["S0"]["orig"], st["S0"]["para"]) == (0, 0)


def test_report_without_final_verdicts_is_unsettled(results):
    (results / "verdicts_final.jsonl").unlink()
    rep = run_report(results)
    assert rep["settled"] is False and rep["verdicts_source"] == "verdicts_1"
    assert rep["summary"]["S1"]["H1"]["n"] == 2  # pass 1 still counted Q2 as correct
    assert "unsettled" in (results / "report.md").read_text(encoding="utf-8")
