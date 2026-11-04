"""Accepted alternative IDs (item["also_ok_ids"]): a gold ID also counts as matched through an alternative."""

from rag_ai_act.metrics import citation_precision, evidence_ok, recall_share


def test_single_hop_accepts_an_alternative_and_rejects_unrelated():
    item = {"type": "single", "gold_ids": ["GLHR:para204"], "also_ok_ids": {"GLHR:para204": ["GLHR:para205"]}}
    assert evidence_ok(item, ["GLHR:para205"])
    assert not evidence_ok(item, ["GLHR:para206"])
    assert not evidence_ok({"type": "single", "gold_ids": ["GLHR:para204"]}, ["GLHR:para205"])


def test_multi_hop_needs_every_part_each_possibly_through_its_alternative():
    item = {"type": "multi", "gold_ids": ["AIA:Art111(2)", "AIA:Art113(3)"],
            "also_ok_ids": {"AIA:Art111(2)": ["GLHR:para449"]}}
    assert evidence_ok(item, ["GLHR:para449", "AIA:Art113(3)"])
    assert not evidence_ok(item, ["GLHR:para449"])  # the second part is still missing
    assert not evidence_ok(item, ["AIA:Art113(3)", "AIA:Art113(4)"])


def test_recall_and_precision_use_alternatives_and_default_to_gold_only():
    alts = {"AIA:Art6(2)": ["AIA:AnnexIII(2)"]}
    assert recall_share(["AIA:Art6(2)"], ["AIA:AnnexIII(2)"], alts) == 1.0
    assert recall_share(["AIA:Art6(2)"], ["AIA:AnnexIII(2)"]) == 0.0
    assert citation_precision(["AIA:Art6(2)"], ["AIA:AnnexIII(2)", "AIA:Art9(1)"], alts) == 0.5
    assert citation_precision(["AIA:Art6(2)"], ["AIA:AnnexIII(2)"]) == 0.0
