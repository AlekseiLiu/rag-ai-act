import re
from pathlib import Path

import pytest

from rag_ai_act.index import BM25Retriever
from rag_ai_act.ingest import chunk_units, clean_text, parse_units
from rag_ai_act.tools import NOT_FOUND, DateClauses, ProvisionStore, format_units, make_tools


@pytest.fixture
def provision_units(unit_factory):
    return [
        unit_factory("AIA:Art6(2)", "Annex III systems are high-risk.", order=2),
        unit_factory("AIA:Art6(1a)", "Non-safety assistance is excluded.", order=1),
        unit_factory("AIA:Art6(1)", "Safety components in Annex I.", order=0),
        unit_factory("AIA:Art60(1)", "Other provision.", order=3),
    ]


def test_provision_exact_loose_and_article_order(provision_units):
    store = ProvisionStore(provision_units)
    assert store.get("AIA:Art6(2)") == [provision_units[0]]
    assert store.get("Article 6(2)(a)") == [provision_units[0]]
    assert [u.source_id for u in store.get("AIA:Art6")] == [
        "AIA:Art6(1)",
        "AIA:Art6(1a)",
        "AIA:Art6(2)",
    ]


@pytest.mark.parametrize("source_id", ["AIA:Art999", "cannot parse this"])
def test_unknown_provision_and_tool_not_found(provision_units, source_id):
    store = ProvisionStore(provision_units)
    assert store.get(source_id) == []
    assert format_units(store.get(source_id)) == NOT_FOUND
    retriever = BM25Retriever(chunk_units(provision_units))
    tools = {t.name: t for t in make_tools(retriever, store, DateClauses(provision_units))}
    assert tools["get_provision"].invoke({"source_id": source_id}) == NOT_FOUND


def test_format_units_keeps_identity_text_and_draft_label(unit_factory):
    law = unit_factory("AIA:Art6(2)", "Binding text", header="Classification")
    draft = unit_factory("GLHR:para204", "Draft text", header="Electricity", draft=True)
    assert format_units([law, draft]) == (
        "[AIA:Art6(2)] Classification\nBinding text\n\n[GLHR:para204] (DRAFT) Electricity\nDraft text"
    )
    assert format_units([]) == NOT_FOUND


@pytest.fixture
def date_units(snippet, unit_factory):
    units = []
    for name in ("act_article111.txt", "act_article113.txt"):
        units.extend(parse_units(clean_text(snippet(name), "AIA"), "AIA"))
    units.append(unit_factory("AIA:Art6(2)", "Annex III and Article 6(1) are mentioned here."))
    units.append(unit_factory("GLHR:para449", "Annex III and Article 6(1) draft discussion.", draft=True))
    return units


@pytest.mark.parametrize(
    ("query", "clause", "date"),
    [
        ("Annex III", "AIA:Art113(3)", "2 December 2027"),
        ("Article 6(1)", "AIA:Art113(3)", "2 August 2028"),
        ("AIA:Art6(1)", "AIA:Art113(3)", "2 August 2028"),
        ("Annex X", "AIA:Art111(1)", "31 December 2030"),
    ],
)
def test_dates_return_relevant_corpus_clause_only(date_units, query, clause, date):
    result = DateClauses(date_units).lookup(query)
    assert any(u.source_id == clause and date in u.text for u in result)
    assert all(u in date_units for u in result)
    assert all(u.source_id.startswith(("AIA:Art113(", "AIA:Art111(")) for u in result)


def test_unknown_date_query(date_units):
    assert DateClauses(date_units).lookup("Article 999") == []


def test_three_tools_route_queries_to_real_data(date_units):
    retriever = BM25Retriever(chunk_units(date_units))
    tools = {t.name: t for t in make_tools(retriever, ProvisionStore(date_units), DateClauses(date_units))}
    assert set(tools) == {"search", "get_provision", "applicability_date"}
    exact = tools["get_provision"].invoke({"source_id": "Article 113(3)"})
    assert "[AIA:Art113(3)]" in exact and "Annex III" in exact
    dates = tools["applicability_date"].invoke({"provision": "Annex III"})
    assert "[AIA:Art113(3)]" in dates and "2 December 2027" in dates
    search = tools["search"].invoke({"query": "Annex III"})
    assert "[AIA:" in search
    assert tools["applicability_date"].invoke({"provision": "Article 999"}) == NOT_FOUND


def test_source_has_no_date_literals():
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src").rglob("*.py"))
    assert sources
    months = "January|February|March|April|May|June|July|August|September|October|November|December"
    pattern = re.compile(rf"\b\d{{1,2}} ({months}) \d{{4}}\b|\b(19|20)\d\d-\d\d-\d\d\b")
    violations = [str(p.relative_to(root)) for p in sources if pattern.search(p.read_text())]
    assert violations == []
