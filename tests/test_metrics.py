import pytest

from rag_ai_act.metrics import (
    citation_precision,
    evidence_ok,
    paraphrase_stability,
    recall_share,
    strict_correct,
    summarize,
    verdict_band,
)


def question(qid="q1", kind="single", era="2026", ids=None):
    return {
        "qid": qid,
        "question": f"Question {qid}?",
        "type": kind,
        "era": era,
        "gold_answer": "Expected answer",
        "gold_ids": ids if ids is not None else ["AIA:Art6(2)"],
    }


def record(qid="q1", *, cited=None, valid=True, declined=False, retrieved=None, eur=0):
    return {
        "qid": qid,
        "answer": "Response",
        "cited_ids": cited if cited is not None else ["AIA:Art6(2)"],
        "valid": valid,
        "declined": declined,
        "retrieved_ids": retrieved or [],
        "usage": {"eur": eur},
        "trace": [],
        "latency_s": 0.1,
    }


@pytest.mark.parametrize(
    ("kind", "cited", "expected"),
    [
        ("single", ["AIA:Art6(2)"], True),
        ("single", ["GLHR:para204"], True),
        ("single", [], False),
        ("multi", ["AIA:Art6(2)"], False),
        ("multi", ["AIA:Art6(2)", "GLHR:para204"], True),
        ("multi", ["AIA:Art6", "GLHR:para204"], False),
        ("multi", ["AIA:Art6(2)", "AIA:Art6(2)"], False),
        ("unanswerable", ["AIA:Art6(2)", "GLHR:para204"], False),
    ],
)
def test_evidence_single_alternative_vs_multi_all(kind, cited, expected):
    item = question(kind=kind, ids=["AIA:Art6(2)", "GLHR:para204"])
    assert evidence_ok(item, cited) is expected


@pytest.mark.parametrize(
    ("kind", "valid", "declined", "correct", "cited", "expected"),
    [
        ("single", True, False, True, ["AIA:Art6(2)"], True),
        ("single", True, True, True, ["AIA:Art6(2)"], False),
        ("single", False, False, True, ["AIA:Art6(2)"], False),
        ("single", True, False, False, ["AIA:Art6(2)"], False),
        ("single", True, False, True, [], False),
        ("single", True, False, True, ["AIA:Art6"], False),
        ("unanswerable", True, False, True, ["AIA:Art6(2)"], False),
    ],
)
def test_strict_correct_requires_every_condition(kind, valid, declined, correct, cited, expected):
    assert (
        strict_correct(question(kind=kind), record(valid=valid, declined=declined, cited=cited), correct)
        is expected
    )


def test_recall_and_precision_are_directional_shares():
    gold = ["AIA:Art6(2)", "GLHR:para204"]
    assert recall_share(gold, ["Article 6(2)", "AIA:Art6(2)"]) == pytest.approx(0.5)
    assert recall_share(gold, ["AIA:Art6"]) == 0
    assert recall_share(gold, []) == 0
    assert recall_share(["AIA:Art6"], ["AIA:Art6(2)"]) == 1
    assert citation_precision(gold, ["Art. 6(2)", "unreadable", "AIA:Art7"]) == pytest.approx(1 / 3)
    assert citation_precision(gold, []) is None


def test_summary_denominators_era_declines_invalids_and_cost():
    items = [
        question("q1", era="2024"),
        question("q2", "multi", ids=["AIA:Art6(2)", "GLHR:para204"]),
        question("q3", ids=["AIA:Art113(3)"]),
        question("q4", era="2024", ids=["AIA:AnnexIII(2)"]),
        question("q5", ids=["GLHR:para204"]),
        *[question(f"u{n}", "unanswerable", "n/a", []) for n in range(1, 4)],
    ]
    rows = [
        record("q1", eur=0.01),
        record("q2", eur=0.02),
        record("q3", cited=[], declined=True),
        record("q4", valid=False),
        record("q5", cited=[]),
        record("u1", cited=[], declined=True),
        record("u2", cited=[], declined=True, valid=False),
        record("u3", cited=[]),
    ]
    correct = {("S3", f"q{n}"): n != 5 for n in range(1, 6)}
    result = summarize(items, {"S3": rows}, correct)["S3"]
    assert result["H1"] == {"n": 1, "of": 5, "pct": 20.0}
    assert result["H2"] == {"n": 0, "of": 3, "pct": 0.0}
    assert result["H3"] == {"n": 1, "of": 3, "pct": 33.3}
    assert result["false_declines"] == {"n": 1, "of": 5, "pct": 20.0}
    assert result["h1_by_era"]["2024"] == {"n": 1, "of": 2, "pct": 50.0}
    assert result["h1_by_era"]["2026"] == {"n": 0, "of": 3, "pct": 0.0}
    assert result["multi_hop"] == {"n": 0, "of": 1, "pct": 0.0}
    assert result["invalid"] == 2
    assert result["eur"] == pytest.approx(0.03)
    assert result["citation_precision"] == {"mean": 1.0, "n_citing_nothing": 2}


def test_summary_retrieval_uses_top_five_for_rag_and_all_for_agent():
    item = question(kind="multi", ids=["AIA:Art6(2)", "GLHR:para204"])
    retrieved = ["AIA:Art6(2)", "AIA:Art7", "AIA:Art8", "AIA:Art9", "AIA:Art10", "GLHR:para204"]
    answers = {
        system: [record(cited=["AIA:Art6(2)", "AIA:Art7"], retrieved=[] if system == "S0" else retrieved)]
        for system in ("S0", "S1", "S2", "S3")
    }
    summary = summarize([item], answers, {(s, "q1"): True for s in answers})
    assert summary["S1"]["recall_at_5"] == pytest.approx(0.5)
    assert summary["S2"]["recall_at_5"] == pytest.approx(0.5)
    assert summary["S3"]["evidence_recall"] == pytest.approx(1.0)
    assert summary["S3"]["citation_precision"]["mean"] == pytest.approx(0.5)
    for system in ("S0", "S3"):
        assert summary[system]["recall_at_5"] is None
    for system in ("S0", "S1", "S2"):
        assert summary[system]["evidence_recall"] is None


def test_ungraded_summary_defers_correctness_metrics():
    items = [question(), question("u1", "unanswerable", "n/a", [])]
    rows = [record(declined=True, cited=[], eur=0.01), record("u1", declined=True, cited=[], eur=0.02)]
    result = summarize(items, {"S3": rows}, None)["S3"]
    for metric in ("H1", "H2", "h1_by_era", "multi_hop"):
        assert result[metric] is None
    assert result["false_declines"] == {"n": 1, "of": 1, "pct": 100.0}
    assert result["H3"] == {"n": 1, "of": 1, "pct": 100.0}
    assert result["invalid"] == 0
    assert result["eur"] == pytest.approx(0.03)
    assert result["citation_precision"] == {"mean": None, "n_citing_nothing": 1}


def test_summary_excludes_paraphrases_from_main_counts():
    original = question()
    paraphrase = {**question("p1"), "paraphrase_of": "q1"}
    result = summarize(
        [original, paraphrase],
        {"S3": [record(), record("p1", declined=True, cited=[], valid=False)]},
        {("S3", "q1"): True, ("S3", "p1"): False},
    )["S3"]
    assert result["H1"] == {"n": 1, "of": 1, "pct": 100.0}
    assert result["false_declines"] == {"n": 0, "of": 1, "pct": 0.0}
    assert result["H2"] == {"n": 1, "of": 1, "pct": 100.0}
    assert result["invalid"] == 0


def band_input(s0, s1, s3, h3=5, declines=2, answerable=24, unanswerable=6):
    def counts(n, of):
        return {"n": n, "of": of, "pct": round(100 * n / of, 1)}

    return {
        "S0": {"H1": counts(s0, answerable)},
        "S1": {"H1": counts(s1, answerable)},
        "S3": {
            "H1": counts(s3, answerable),
            "H3": counts(h3, unanswerable),
            "false_declines": counts(declines, answerable),
        },
    }


@pytest.mark.parametrize(
    ("s0", "s1", "s3", "h3", "declines", "expected"),
    [
        (10, 14, 17, 5, 2, "good"),
        (11, 16, 17, 5, 2, "good"),
        (10, 17, 17, 5, 2, "useless"),
        (18, 14, 17, 5, 2, "useless"),
        (16, 14, 17, 5, 2, "useless"),
        (15, 14, 17, 5, 2, "partial"),
        (10, 14, 16, 5, 2, "partial"),
        (10, 14, 17, 4, 2, "partial"),
        (10, 14, 17, 5, 3, "partial"),
    ],
)
def test_verdict_thresholds_and_useless_precedence(s0, s1, s3, h3, declines, expected):
    assert verdict_band(band_input(s0, s1, s3, h3, declines)) == expected


def test_verdict_thresholds_scale_with_set_size():
    assert verdict_band(band_input(7, 10, 12, 4, 1, 16, 4)) == "good"
    assert verdict_band(band_input(7, 10, 12, 3, 1, 16, 4)) == "partial"
    assert verdict_band(band_input(13, 12, 14, 5, 1, 20, 6)) == "useless"


@pytest.mark.parametrize(("orig_n", "para_n", "stable"), [(7, 5, True), (7, 4, False), (4, 7, False)])
def test_paraphrase_stability_counts_and_boundary(orig_n, para_n, stable):
    orig = [True] * orig_n + [False] * (10 - orig_n)
    para = [True] * para_n + [False] * (10 - para_n)
    assert paraphrase_stability(orig, para) == {"orig": orig_n, "para": para_n, "within_2": stable}
