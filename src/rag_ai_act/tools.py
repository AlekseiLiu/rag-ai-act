"""Agent tools: search, get_provision, applicability_date. No date literals anywhere in src/.

Every tool returns plain text built by `format_units`. When a tool is invoked with a tool call (as an agent
does), the resulting ToolMessage also carries `artifact`: the list of source IDs it returned, in order,
without repeats (empty for NOT_FOUND).
"""

from __future__ import annotations

import re

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from rag_ai_act.ids import InvalidId, normalize_id, parent_id, parse_id
from rag_ai_act.index import TOP_K, Retriever
from rag_ai_act.ingest import Chunk, Unit

NOT_FOUND = "NOT_FOUND"


# ------------------------------------------------------------------------------------------ provisions


class ProvisionStore:
    """Exact lookup of units by ID. An article-level ID returns all its paragraphs in document order."""

    def __init__(self, units: list[Unit]):
        self._exact: dict[str, list[Unit]] = {}
        self._children: dict[str, list[Unit]] = {}
        for unit in units:
            self._exact.setdefault(unit.source_id, []).append(unit)
            try:
                parent = parent_id(unit.source_id)
            except InvalidId:
                parent = None
            if parent is not None:
                self._children.setdefault(parent, []).append(unit)
        for children in self._children.values():
            children.sort(key=lambda u: u.order)  # stable: equal orders keep input order

    def get(self, source_id: str) -> list[Unit]:
        """Accepts loose IDs (ids.parse_id). Unknown or unparseable ID -> [] (never raises).

        Lookup order: the unit with exactly this ID; else, for an article or annex ID, all its paragraphs,
        points or sections sorted by `order`; else, for a paragraph or point ID whose article or annex is
        stored as one single unit (e.g. "Article 4a(1)" when only AIA:Art4a exists), that unit.
        """
        try:
            sid = normalize_id(source_id)
        except InvalidId:
            return []
        if sid in self._exact:
            return list(self._exact[sid])
        if sid in self._children:
            return list(self._children[sid])
        try:
            parent = parent_id(sid)
        except InvalidId:
            return []
        if parent is not None and parent in self._exact:
            return list(self._exact[parent])
        return []


# --------------------------------------------------------------------------------------- date clauses

_DATE_ARTICLES = ("111", "113")
_ORDINAL_PARAGRAPH = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6"}

# A reference keyword followed by a list of numbers / numerals, e.g. "Articles 102 to 110",
# "Article 5(1a) and (1b)", "Chapters I and II", "Chapter III, Sections 1, 2, and 3".
_KEYWORD = re.compile(r"\b(?i:(article|annex|chapter)(?:s|es)?)\s+")
_SEP = r"\s*(?:,\s*(?:and|or)?|and|or)\s*"
_ARTICLE_ITEM = re.compile(r"(\d+[a-z]?)(?:\((\d+[a-z]?)\))?")
_ARTICLE_RANGE = re.compile(r"\s+to\s+(\d+)\b")
_ARTICLE_EXTRA_PARA = re.compile(_SEP + r"\((\d+[a-z]?)\)")
_ARTICLE_ORDINAL = re.compile(r",?\s*(first|second|third|fourth|fifth|sixth)\s+paragraph\b", re.IGNORECASE)
_ROMAN_ITEM = re.compile(r"([IVXL]+)\b")
_SECTIONS = re.compile(r"[\s,]*(?i:sections?)\s+")
_NUMBER_ITEM = re.compile(r"(\d+)\b")
_NEXT = re.compile(_SEP)

_QUERY_CHAPTER = re.compile(
    r"^(?:AIA\s*:\s*)?chapter\s+([IVXL]+|\d+)(?:[\s,]+section\s+(\d+))?\.?$", re.IGNORECASE
)
_QUERY_SECTION = re.compile(r"^(?:AIA\s*:\s*)?section\s+(\d+)\.?$", re.IGNORECASE)


def _roman(number: int) -> str:
    out = ""
    for value, numeral in ((50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while number >= value:
            out += numeral
            number -= value
    return out


def _item_list(text: str, pos: int, item: re.Pattern[str]) -> tuple[list[re.Match[str]], int]:
    """Items matching `item` from `pos`, separated by commas / "and" / "or". Returns (matches, end)."""
    found: list[re.Match[str]] = []
    m = item.match(text, pos)
    while m is not None:
        found.append(m)
        pos = m.end()
        sep = _NEXT.match(text, pos)
        nxt = item.match(text, sep.end()) if sep else None
        if nxt is None:
            break
        m = nxt
    return found, pos


def _article_mentions(text: str, pos: int) -> set[tuple]:
    out: set[tuple] = set()
    while True:
        m = _ARTICLE_ITEM.match(text, pos)
        if m is None:
            return out
        number, sub = m.group(1).lower(), m.group(2)
        pos = m.end()
        ordinal = _ARTICLE_ORDINAL.match(text, pos) if sub is None else None
        if ordinal:  # "Article 113, third paragraph"
            sub = _ORDINAL_PARAGRAPH[ordinal.group(1).lower()]
            pos = ordinal.end()
        out.add(("art", number, sub))
        rng = _ARTICLE_RANGE.match(text, pos)
        if rng and number.isdigit():  # "Articles 102 to 110"
            for n in range(int(number) + 1, int(rng.group(1)) + 1):
                out.add(("art", str(n), None))
            pos = rng.end()
        extra = _ARTICLE_EXTRA_PARA.match(text, pos)
        while extra:  # "Article 5(1a) and (1b)"
            out.add(("art", number, extra.group(1)))
            pos = extra.end()
            extra = _ARTICLE_EXTRA_PARA.match(text, pos)
        sep = _NEXT.match(text, pos)
        if sep is None or not re.match(r"\d", text[sep.end():sep.end() + 1]):
            return out
        pos = sep.end()


def _mentions(text: str) -> set[tuple]:
    """Provisions a legal text names: ("art", number, paragraph|None), ("annex", roman),
    ("chapter", roman) and ("section", chapter roman, section number)."""
    text = " ".join(text.split())
    out: set[tuple] = set()
    for key in _KEYWORD.finditer(text):
        kind = key.group(1).lower()
        if kind == "article":
            out |= _article_mentions(text, key.end())
            continue
        numerals, end = _item_list(text, key.end(), _ROMAN_ITEM)
        for m in numerals:
            out.add((kind, m.group(1)))
        if kind == "chapter" and len(numerals) == 1:
            sec = _SECTIONS.match(text, end)
            if sec:
                numbers, _ = _item_list(text, sec.end(), _NUMBER_ITEM)
                out.update(("section", numerals[0].group(1), m.group(1)) for m in numbers)
    return out


def _query_refs(provision: str) -> set[tuple] | None:
    """The reference a query names, as a one-element set of mention tuples (see `_mentions`); an empty
    set for guideline IDs (never named in the Act); None if the query is not a provision reference."""
    text = " ".join(provision.split())
    m = _QUERY_CHAPTER.match(text)
    if m:
        chapter = m.group(1).upper() if not m.group(1).isdigit() else _roman(int(m.group(1)))
        if m.group(2):
            return {("section", chapter, m.group(2))}
        return {("chapter", chapter)}
    m = _QUERY_SECTION.match(text)
    if m:
        return {("section", None, m.group(1))}
    try:
        sid = parse_id(text)
    except InvalidId:
        return None
    if sid.kind == "art":
        return {("art", sid.number, sid.sub)}
    if sid.kind == "annex":
        return {("annex", sid.number)}
    return set()


def _refers_to(query: tuple, mention: tuple) -> bool:
    """True if a clause naming `mention` speaks about the queried provision. A whole article covers its
    paragraphs and vice versa; a chapter covers its sections and vice versa."""
    kind = query[0]
    if kind == "art":
        same_article = mention[0] == "art" and mention[1] == query[1]
        return same_article and (mention[2] == query[2] or None in (query[2], mention[2]))
    if kind == "annex":
        return mention == query
    if kind == "chapter":
        return mention[0] in ("chapter", "section") and mention[1] == query[1]
    # section: (None or chapter, number)
    _, chapter, number = query
    if mention[0] == "section":
        return mention[2] == number and chapter in (None, mention[1])
    return mention[0] == "chapter" and chapter is not None and mention[1] == chapter


class DateClauses:
    """Clauses of AIA Art. 113 and Art. 111 (only these), built from the units at construction.

    Each clause is indexed by the provisions its text names: articles with paragraphs, article lists and
    ranges ("Articles 102 to 110"), annexes, chapters and chapter sections ("Chapter III, Sections 1, 2,
    and 3"). Dates are never stored separately; they are read from the returned clause text.
    """

    def __init__(self, units: list[Unit]):
        self.units: list[Unit] = []
        for unit in units:
            try:
                sid = parse_id(unit.source_id)
            except InvalidId:
                continue
            if sid.doc == "AIA" and sid.kind == "art" and sid.number in _DATE_ARTICLES:
                self.units.append(unit)
        self._mentions = [_mentions(u.text) for u in self.units]

    def lookup(self, provision: str) -> list[Unit]:
        """Units of Art. 113 / 111 whose text mentions `provision`, which may be an ID or a phrase such as
        "Annex III", "Article 6(1)", "Chapter III", "Section 4". Returns [] if none.

        IDs and references ("AIA:Art6(1)", "Article 6(1)(a)", "Annex III point 2", "Chapter 3") match by
        the provisions a clause names: a paragraph query also matches a clause naming its whole article,
        an article query matches clauses naming any of its paragraphs, a chapter query matches its
        sections. Guideline IDs never match. Other text matches clauses containing it as a phrase
        (case-insensitive, whole words). Results keep the order of the units given at construction.
        """
        if not isinstance(provision, str) or not provision.strip():
            return []
        query = _query_refs(provision)
        if query is not None:
            return [
                u
                for u, names in zip(self.units, self._mentions, strict=True)
                if any(_refers_to(q, m) for q in query for m in names)
            ]
        phrase = re.compile(r"(?<!\w)" + re.escape(" ".join(provision.split())) + r"(?!\w)", re.IGNORECASE)
        return [u for u in self.units if phrase.search(" ".join(u.text.split()))]


# -------------------------------------------------------------------------------------------- tools


def format_units(units: list[Unit]) -> str:
    """Tool output text: for each unit "[<source_id>] <header>\n<text>"; a draft unit's first line is
    "[<source_id>] (DRAFT) <header>". Units separated by blank lines; NOT_FOUND if units is empty."""
    if not units:
        return NOT_FOUND
    blocks = []
    for u in units:
        label = "(DRAFT) " if u.draft else ""
        blocks.append(f"[{u.source_id}] {label}{u.header}\n{u.text}")
    return "\n\n".join(blocks)


def _chunk_as_unit(chunk: Chunk) -> Unit:
    """A retrieved chunk in Unit form: its first line is the unit header, the rest is (part of) the text."""
    header, _, body = chunk.text.partition("\n")
    return Unit(chunk.source_id, chunk.doc, None, header, body, chunk.draft, chunk.part_index)


def _result(units: list[Unit]) -> tuple[str, list[str]]:
    return format_units(units), list(dict.fromkeys(u.source_id for u in units))


class _SearchArgs(BaseModel):
    query: str = Field(description="Free-text search query, e.g. 'AI systems in electricity supply'.")


class _ProvisionArgs(BaseModel):
    source_id: str = Field(description="Source ID such as 'AIA:Art6(2)', 'AIA:AnnexIII(2)' or 'GLHR:para204'.")


class _DateArgs(BaseModel):
    provision: str = Field(
        description="Provision to date, e.g. 'Annex III', 'Article 6(1)', 'Article 50', 'Chapter III'."
    )


_SEARCH_DOC = f"""Search the indexed corpus for passages relevant to a free-text query. The corpus is the
consolidated EU AI Act (Regulation (EU) 2024/1689, no recitals), the European Commission's DRAFT guidelines
on high-risk classification (GLHR) and the Commission guidelines on the AI system definition (GLDEF).
Returns the {TOP_K} best-matching passages, best first. Each passage starts with a line
"[<source_id>] <header>" followed by its text; passages from the draft guidelines show "(DRAFT)" after the
ID. Long provisions may appear as a part only: call get_provision with the ID for the full text. Cite the
IDs exactly as shown."""

_PROVISION_DOC = """Fetch the exact text of a provision by its source ID.
Accepted IDs (canonical form first; loose forms such as 'Article 6(2)', 'Art. 6(2)(a)', 'Annex III point 2',
'GLHR para 204' are also accepted):
- AI Act article paragraph: 'AIA:Art6(2)'; inserted numbers keep their letter: 'AIA:Art4a', 'AIA:Art6(1a)'.
- Whole article: 'AIA:Art6' returns all its paragraphs in order (Article 3 is long: prefer one definition).
- Article 3 definition point: 'AIA:Art3(14)'.
- Article 113's paragraphs are unnumbered in the text and counted in order: 'AIA:Art113(3)' is the third.
- Annex point: 'AIA:AnnexIII(2)'; whole annex: 'AIA:AnnexIII'. Annexes I, VIII and XI are stored per
  section, not per point: 'AIA:AnnexI(A)', 'AIA:AnnexI(B)', 'AIA:AnnexVIII(A)', 'AIA:AnnexXI(1)'.
- Draft high-risk guidelines paragraph: 'GLHR:para204'; unnumbered example boxes: 'GLHR:sec3.2.7'.
- AI system definition guidelines paragraph: 'GLDEF:para43'.
Points below paragraph level such as '(a)' are dropped and the whole paragraph is returned.
Returns each unit as a line "[<source_id>] <header>" followed by its text ("(DRAFT)" marks draft
guidelines), or NOT_FOUND if the ID is unknown or cannot be read."""

_DATE_DOC = """Find the dates from which parts of the AI Act apply. Looks only in Article 113 (entry into
force and application) and Article 111 (systems and models already on the market) and returns every clause
of those two articles that names the given provision, as "[<source_id>] <header>" plus the clause text;
read the date from the clause and cite its ID (e.g. AIA:Art113(3)).
Accepts an article ('Article 6(1)', 'AIA:Art50'), an annex ('Annex III'), a chapter ('Chapter III'), a
chapter section ('Chapter III Section 4' or 'Section 4'), or a short phrase occurring in those articles.
A paragraph query also matches clauses naming the whole article; an article query matches clauses naming
any of its paragraphs; article ranges such as 'Articles 102 to 110' are understood.
Returns NOT_FOUND if no clause names the provision: then query the chapter or annex it belongs to, or
fetch 'AIA:Art113' with get_provision for the general dates."""


def make_tools(retriever: Retriever, store: ProvisionStore, dates: DateClauses) -> list[BaseTool]:
    """Three LangChain tools named exactly "search" (arg: query), "get_provision" (arg: source_id) and
    "applicability_date" (arg: provision); each returns format_units(...) text.

    Invoked with a plain argument dict, a tool returns the text. Invoked with a tool call, it returns a
    ToolMessage whose content is that text and whose `artifact` is the list of returned source IDs in
    order, without repeats. Exceptions from the retriever or the stores propagate unchanged.
    """

    def search(query: str) -> tuple[str, list[str]]:
        return _result([_chunk_as_unit(c) for c in retriever.retrieve(query)])

    def get_provision(source_id: str) -> tuple[str, list[str]]:
        return _result(store.get(source_id))

    def applicability_date(provision: str) -> tuple[str, list[str]]:
        return _result(dates.lookup(provision))

    specs = (
        (search, "search", _SEARCH_DOC, _SearchArgs),
        (get_provision, "get_provision", _PROVISION_DOC, _ProvisionArgs),
        (applicability_date, "applicability_date", _DATE_DOC, _DateArgs),
    )
    return [
        StructuredTool.from_function(
            func=func,
            name=name,
            description=doc,
            args_schema=schema,
            response_format="content_and_artifact",
        )
        for func, name, doc, schema in specs
    ]
