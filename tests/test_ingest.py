import re
from dataclasses import replace
from pathlib import Path

import pytest
import tiktoken

from rag_ai_act.ids import normalize_id
from rag_ai_act.ingest import (
    CORPUS,
    CoverageError,
    check_coverage,
    chunk_units,
    clean_text,
    parse_units,
)


def parse_snippet(snippet, name, doc="AIA", part=None):
    return parse_units(clean_text(snippet(name), doc), doc, part)


def test_clean_text_removes_markers_headers_and_joins_words(snippet):
    text = clean_text(snippet("act_article6.txt"), "AIA")
    assert "▼B" not in text and "▼M1" not in text
    assert "02024R1689 — EN" not in text
    assert "\f" not in text
    assert "malfunctioning" in text
    assert "electromagnetic" in text
    assert any(line.lstrip().startswith("1a.") and line[0].isspace() for line in text.splitlines())


def test_clean_text_removes_footnotes_and_page_footers(snippet):
    text = clean_text(snippet("guidelines_electricity.txt"), "GLHR")
    assert "hydrogen sector" not in text
    assert "energy storage" in text
    assert not any(line.strip() in {"46", "47", "29"} for line in text.splitlines())


def test_article_numbering_insertions_and_embedded_points(snippet):
    units = parse_snippet(snippet, "act_article6.txt")
    expected = [f"AIA:Art6({n})" for n in ("1", "1a", "1b", "1c", "2", "3", "4", "5", "6", "7", "8")]
    assert [u.source_id for u in units] == expected
    first = units[0].text
    assert "(a)" in first and "(b)" in first
    assert "third-party conformity assessment" in first
    assert "non-safety related aspects" in units[1].text
    assert "narrow procedural task" in units[5].text
    assert "Notwithstanding the first subparagraph" in units[5].text


def test_article3_definitions_are_individual_units(snippet):
    units = parse_snippet(snippet, "act_article3.txt")
    ids = [u.source_id for u in units]
    assert "AIA:Art3(1)" in ids and "AIA:Art3(14)" in ids
    assert "AIA:Art3(14a)" in ids and "AIA:Art3(14b)" in ids
    assert len(ids) == len(set(ids))
    safety = next(u for u in units if u.source_id == "AIA:Art3(14)")
    assert "safety function" in safety.text
    assert "micro, small" not in safety.text


def test_article113_unnumbered_paragraphs_keep_points_together(snippet):
    units = parse_snippet(snippet, "act_article113.txt")
    assert [u.source_id for u in units] == [f"AIA:Art113({n})" for n in range(1, 5)]
    assert "twentieth day" in units[0].text
    assert "2 August 2026" in units[1].text
    assert all(f"({letter})" in units[2].text for letter in "abcd")
    assert "Annex III" in units[2].text and "Annex I" in units[2].text
    assert "binding in its entirety" in units[3].text


def test_annex_point_and_subpoints(snippet):
    units = parse_snippet(snippet, "act_annexiii.txt")
    assert [u.source_id for u in units] == ["AIA:AnnexIII(1)", "AIA:AnnexIII(2)"]
    assert "(a)" in units[0].text and "(c)" in units[0].text
    assert "electricity" in units[1].text


@pytest.mark.parametrize(
    ("name", "annex", "sections"),
    [
        (
            "act_annexi_sections.txt",
            "I",
            {
                "A": ["safety of toys", "in vitro diagnostic medical devices"],
                "B": ["civil aviation", "2023/1230"],
            },
        ),
        (
            "act_annexviii_sections.txt",
            "VIII",
            {
                "A": ["contact details of the provider", "additional information (optional)"],
                "B": ["contact details of the provider", "Article 6(3)"],
                "C": ["contact details of the deployer", "fundamental rights impact assessment"],
            },
        ),
        (
            "act_annexxi_sections.txt",
            "XI",
            {
                "1": ["general description", "computational resources"],
                "2": ["evaluation strategies", "adversarial testing", "system architecture"],
            },
        ),
    ],
)
def test_annex_sections_keep_their_numbered_points(snippet, name, annex, sections):
    units = parse_snippet(snippet, name)
    assert [u.source_id for u in units] == [f"AIA:Annex{annex}({section})" for section in sections]
    for unit, phrases in zip(units, sections.values(), strict=True):
        assert unit.doc == "AIA" and unit.header.strip() and unit.draft is False
        assert all(phrase in unit.text for phrase in phrases)
    if annex == "VIII":
        # Both sections start at point 1, but their units must remain distinct.
        assert "contact details of the deployer" not in units[1].text
        assert "additional information (optional)" not in units[1].text
    elif annex == "XI":
        assert "(a)" in units[0].text and "(f)" in units[0].text
        assert "adversarial testing" not in units[0].text


def test_deleted_paragraph_does_not_create_a_unit(snippet):
    units = parse_snippet(snippet, "act_article10.txt")
    assert [u.source_id for u in units] == [f"AIA:Art10({n})" for n in (1, 2, 3, 4, 6)]
    assert all("__________" not in u.text for u in units)


def test_guidelines_page_break_and_all_example_boxes(snippet):
    units = parse_snippet(snippet, "guidelines_electricity.txt", "GLHR", "annex_III")
    assert [u.source_id for u in units] == [
        "GLHR:para203",
        "GLHR:para204",
        "GLHR:para205",
        "GLHR:para206",
        "GLHR:sec3.2.7",
    ]
    para = units[0].text
    assert "electricity undertakings" in para and "energy storage" in para
    box = units[-1].text
    for phrase in (
        "physical perimeter protection",
        "power load distribution",
        "forecasting the energy demand",
        "cybersecurity monitoring",
        "quality assurance",
        "precautionary outage prevention",
        "future improvements",
        "consumption at transmission",
        "grid imbalances",
    ):
        assert phrase in box
    assert "nuclear energy production" not in box


def test_ordinary_examples_join_preceding_paragraph_inside_numbered_section(snippet):
    units = parse_snippet(snippet, "guidelines_inline_examples.txt", "GLHR", "annex_III")
    assert [u.source_id for u in units] == [f"GLHR:para{n}" for n in range(86, 109)]
    paragraph = next(u for u in units if u.source_id == "GLHR:para108")
    assert "Example:" in paragraph.text and "public benefits" in paragraph.text
    assert "Assessment:" in paragraph.text and "supplementary information" in paragraph.text
    assert all("public benefits" not in u.text for u in units if u is not paragraph)


def test_stray_guidelines_number_is_not_a_paragraph(snippet):
    units = parse_snippet(snippet, "guidelines_number_noise.txt", "GLHR", "annex_III")
    ids = [u.source_id for u in units]
    assert "GLHR:para179" in ids and "GLHR:para180" in ids
    assert "GLHR:para2" not in ids


@pytest.mark.parametrize(
    ("name", "doc", "part"),
    [
        ("act_article6.txt", "AIA", None),
        ("guidelines_electricity.txt", "GLHR", "annex_III"),
        ("definition_paragraph43.txt", "GLDEF", None),
    ],
)
def test_unit_identity_headers_and_draft_flags(snippet, name, doc, part):
    units = parse_snippet(snippet, name, doc, part)
    assert units
    for unit in units:
        assert normalize_id(unit.source_id) == unit.source_id
        assert unit.header.strip()
        assert unit.doc == doc and unit.part == part
        assert unit.draft is (doc == "GLHR")
    assert [u.order for u in units] == sorted({u.order for u in units})
    if doc == "GLDEF":
        assert [u.source_id for u in units] == ["GLDEF:para43"]


def test_long_unit_splits_without_losing_text_or_identity(unit_factory):
    encoding = tiktoken.get_encoding("cl100k_base")
    text = " ".join(f"token{n}" for n in range(1800))
    unit = unit_factory("GLHR:para204", text, draft=True, header="Electricity")
    assert len(encoding.encode(text)) > 1000
    chunks = chunk_units([unit], max_tokens=1000)
    assert len(chunks) > 1
    assert [c.chunk_id for c in chunks] == [f"GLHR:para204#{i}" for i in range(len(chunks))]
    assert [c.part_index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.source_id == unit.source_id and chunk.doc == "GLHR" and chunk.draft
        assert chunk.text.startswith(unit.header + "\n")
        assert len(encoding.encode(chunk.text)) <= 1000
    bodies = [c.text.removeprefix(unit.header + "\n") for c in chunks]
    assert re.sub(r"\s+", "", "".join(bodies)) == re.sub(r"\s+", "", text)


def test_short_unit_is_one_chunk(unit_factory):
    unit = unit_factory("AIA:Art6(2)", "Annex III systems are high-risk.")
    chunks = chunk_units([unit])
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "AIA:Art6(2)#0"
    assert chunks[0].text == unit.header + "\n" + unit.text
    assert chunks[0].draft is False


def test_coverage_checks_only_present_documents(unit_factory):
    units = [unit_factory(f"GLHR:para{n}", order=n - 1, draft=True) for n in range(1, 452)]
    coverage = check_coverage(units)
    assert coverage["glhr_paras"] == 451
    assert coverage["missing"] == [] and coverage["duplicates"] == []


def test_coverage_names_missing_and_duplicate_ids(unit_factory):
    units = [unit_factory(f"GLHR:para{n}", order=n - 1, draft=True) for n in range(1, 452)]
    incomplete = [u for u in units if u.source_id != "GLHR:para204"]
    with pytest.raises(CoverageError) as exc:
        check_coverage(incomplete)
    assert set(exc.value.missing) == {"GLHR:para204"}
    assert exc.value.duplicates == []
    with pytest.raises(CoverageError) as exc:
        check_coverage([*units, replace(units[0])])
    assert set(exc.value.duplicates) == {"GLHR:para1"}
    assert exc.value.missing == []


def test_definition_coverage_is_contiguous_to_maximum(unit_factory):
    units = [unit_factory(f"GLDEF:para{n}", order=n - 1) for n in range(1, 4)]
    assert check_coverage(units)["gldef_paras"] == 3
    with pytest.raises(CoverageError) as exc:
        check_coverage([units[0], units[2]])
    assert exc.value.missing == ["GLDEF:para2"]


@pytest.mark.corpus
def test_full_corpus_coverage():
    text_dir = Path(__file__).resolve().parents[1] / "data" / "text"
    units = []
    for filename, (doc, part) in CORPUS.items():
        raw = (text_dir / Path(filename).with_suffix(".txt")).read_text()
        units.extend(parse_units(clean_text(raw, doc), doc, part))
    coverage = check_coverage(units)
    assert coverage["aia_articles"] >= 113 and coverage["glhr_paras"] == 451
    assert coverage["missing"] == [] and coverage["duplicates"] == []
    ids = {u.source_id for u in units}
    assert {f"GLHR:para{n}" for n in range(1, 452)} <= ids
    for n in range(1, 114):
        assert any(sid == f"AIA:Art{n}" or sid.startswith(f"AIA:Art{n}(") for sid in ids)
