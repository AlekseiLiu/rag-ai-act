import pytest

from rag_ai_act.ids import InvalidId, SourceId, matches, normalize_id, parent_id, parse_id


@pytest.mark.parametrize(
    "canonical",
    [
        "AIA:Art6",
        "AIA:Art6(2)",
        "AIA:Art4a",
        "AIA:Art6(1a)",
        "AIA:Art3(14)",
        "AIA:Art113(3)",
        "AIA:AnnexIII(2)",
        "AIA:AnnexI(A)",
        "AIA:AnnexI(B)",
        "AIA:AnnexIV",
        "AIA:AnnexX(1)",
        "AIA:AnnexVIII(A)",
        "AIA:AnnexVIII(B)",
        "AIA:AnnexVIII(C)",
        "AIA:AnnexXI(1)",
        "AIA:AnnexXI(2)",
        "GLHR:para204",
        "GLHR:sec3.2.7",
        "GLDEF:para43",
    ],
)
def test_canonical_round_trip(canonical):
    parsed = parse_id(canonical)
    assert isinstance(parsed, SourceId)
    assert str(parsed) == canonical
    assert normalize_id(canonical) == canonical
    assert parse_id(str(parsed)) == parsed


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("AIA:Art6(2)(a)", "AIA:Art6(2)"),
        ("Art. 6(2)", "AIA:Art6(2)"),
        ("Article 6(2)", "AIA:Art6(2)"),
        ("Article 6(2)(a)", "AIA:Art6(2)"),
        ("AIA: Art 6 (2)", "AIA:Art6(2)"),
        ("Annex III point 2", "AIA:AnnexIII(2)"),
        ("GLHR para 204", "GLHR:para204"),
        ("AIA:AnnexIII(1)(a)", "AIA:AnnexIII(1)"),
        ("Annex XI Section 1", "AIA:AnnexXI(1)"),
    ],
)
def test_loose_and_subpoint_ids(raw, canonical):
    assert normalize_id(raw) == canonical


def test_parsed_fields_and_distinct_annex_sections():
    assert parse_id("Article 6(1a)") == SourceId("AIA", "art", "6", "1a")
    assert parse_id("GLHR:sec3.2.7") == SourceId("GLHR", "sec", "3.2.7")
    assert parse_id("AIA:AnnexI(A)") != parse_id("AIA:AnnexI(B)")


@pytest.mark.parametrize("raw", ["", "nonsense", "para 204", "XYZ:Art6", "AIA:Art", "GLHR:para"])
def test_invalid_ids_raise(raw):
    with pytest.raises(InvalidId):
        parse_id(raw)
    with pytest.raises(InvalidId):
        normalize_id(raw)


@pytest.mark.parametrize(
    ("cited", "gold", "expected"),
    [
        ("Art. 6(2)", "AIA:Art6(2)", True),
        ("AIA:Art6(2)(a)", "AIA:Art6(2)", True),
        ("AIA:Art6(2)", "AIA:Art6", True),
        ("AIA:Art6", "AIA:Art6(2)", False),
        ("AIA:Art60(2)", "AIA:Art6", False),
        ("AIA:AnnexI(A)", "AIA:AnnexI(B)", False),
        ("unreadable", "AIA:Art6", False),
        ("GLDEF:para204", "GLHR:para204", False),
    ],
)
def test_matching_is_directional_and_document_aware(cited, gold, expected):
    assert matches(cited, gold) is expected


def test_invalid_gold_raises():
    with pytest.raises(InvalidId):
        matches("AIA:Art6", "unreadable")


@pytest.mark.parametrize(
    ("source_id", "expected"),
    [
        ("AIA:Art6(2)", "AIA:Art6"),
        ("AIA:Art6(1a)", "AIA:Art6"),
        ("AIA:AnnexIII(2)", "AIA:AnnexIII"),
        ("AIA:AnnexI(A)", "AIA:AnnexI"),
        ("AIA:AnnexVIII(C)", "AIA:AnnexVIII"),
        ("AIA:AnnexXI(2)", "AIA:AnnexXI"),
        ("AIA:Art4a", None),
        ("AIA:AnnexIII", None),
        ("GLHR:para204", None),
        ("GLHR:sec3.2.7", None),
        ("GLDEF:para43", None),
    ],
)
def test_parent_ids(source_id, expected):
    assert parent_id(source_id) == expected
