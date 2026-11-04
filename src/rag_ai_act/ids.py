"""Source IDs: parsing, normalisation and matching.

Canonical forms (paragraph level is the finest gold level):
  AIA:Art6            whole article (or single-paragraph article)
  AIA:Art6(2)         numbered paragraph; inserted numbers keep their letter: AIA:Art4a, AIA:Art6(1a)
  AIA:Art3(14)        Art. 3 definition point 14
  AIA:Art113(3)       third (unnumbered) paragraph of Art. 113
  AIA:AnnexIII(2)     annex point; AIA:AnnexI(A) / AIA:AnnexI(B) annex sections (letters kept)
  GLHR:para204        draft high-risk guidelines paragraph; GLHR:sec3.2.7 unnumbered example box
  GLDEF:para43        definition guidelines paragraph
  Other annexes: AIA:Annex{roman} or AIA:Annex{roman}({point}); annex sub-points are dropped
  (AnnexIII(1)(a) -> AnnexIII(1)). Annexes whose sections restart their point numbering have one unit per
  SECTION, not per point: AIA:AnnexI(A), AIA:AnnexI(B); AIA:AnnexVIII(A), (B), (C); AIA:AnnexXI(1), (2)
  (Annex XI "Section 1" -> AIA:AnnexXI(1)); points inside those sections are part of the section unit. A paragraph reference without a doc prefix and without "Art"/"Annex"
  (e.g. "para 204") is InvalidId.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DOCS = ("AIA", "GLHR", "GLDEF")


class InvalidId(ValueError):
    """Raised when a string cannot be read as a source ID."""


@dataclass(frozen=True)
class SourceId:
    doc: str  # one of DOCS
    kind: str  # "art" | "annex" | "para" | "sec"
    number: str  # article number ("6", "4a", "113"), annex roman ("III"), paragraph ("204"), section ("3.2.7")
    sub: str | None = None  # paragraph / point / annex section, e.g. "2", "1a", "14", "A"; None = whole unit

    def __str__(self) -> str:  # canonical string, e.g. "AIA:Art6(2)"
        if self.kind == "para":
            return f"{self.doc}:para{self.number}"
        if self.kind == "sec":
            return f"{self.doc}:sec{self.number}"
        if self.kind == "art":
            base = f"{self.doc}:Art{self.number}"
        elif self.kind == "annex":
            base = f"{self.doc}:Annex{self.number}"
        else:
            raise InvalidId(f"unknown kind {self.kind!r}")
        return base if self.sub is None else f"{base}({self.sub})"


_ORDINALS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6"}

# Doc prefix: "AIA:", "GLHR: " (any word before a colon must be a known doc), or loose "GLHR para 204".
_PREFIX = re.compile(r"^([A-Za-z]+)\s*:\s*(.*)$", re.DOTALL)
_LOOSE_PREFIX = re.compile(r"^(AIA|GLHR|GLDEF)\s+(.*)$", re.DOTALL | re.IGNORECASE)
# Trailing references to the act itself: "Art. 6(2) AI Act", "Article 6 of the AI Act".
_ACT_SUFFIX = re.compile(r"[\s,]*(?:of\s+(?:the\s+)?)?(?:AI\s+Act|AIA)\s*$", re.IGNORECASE)
# Anything below the kept level, which is dropped: "(a)", "(iii)", ", point (a)", "first subparagraph".
_TAIL = re.compile(
    r"^(?:[\s,]*(?:\(\s*[a-z]{1,4}\s*\)|points?\s*\(?\s*[a-z]{1,4}\s*\)?"
    r"|(?:first|second|third|fourth|fifth|sixth)\s+subparagraph))*[\s.,;]*$",
    re.IGNORECASE,
)
_ANNEX_DROP = re.compile(
    r"^(?:[\s,]*(?:\(\s*[A-Za-z0-9]{1,4}\s*\)|(?:point|pt\.?)\s*\(?\s*[A-Za-z0-9]{1,4}\s*\)?))*", re.IGNORECASE
)
_ARTICLE = re.compile(r"^(?:art(?:icle)?s?\.?)\s*(\d+[a-z]?)\b(.*)$", re.IGNORECASE | re.DOTALL)
_ANNEX = re.compile(r"^annex\s*([IVX]+)(?![A-Za-z])(.*)$", re.IGNORECASE | re.DOTALL)
_PARA = re.compile(r"^(?:(?:para(?:graph)?s?\.?|§)\s*)?\(?\s*(\d+)\s*\)?[\s.]*$", re.IGNORECASE)
_SEC = re.compile(r"^(?:sec(?:tion)?\.?)\s*(\d+(?:\.\d+)*)\.?\s*$", re.IGNORECASE)
_ROMAN = re.compile(r"^(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def _article_sub(rest: str) -> str | None:
    """Paragraph after an article number: "(2)", "(1a)", " paragraph 2", ", third paragraph"; else None."""
    rest = rest.strip()
    sub = None
    m = re.match(r"^\(\s*(\d+[a-z]?)\s*\)(.*)$", rest, re.IGNORECASE | re.DOTALL)
    if m is None:
        m = re.match(r"^[\s,]*(?:para(?:graph)?\.?)\s*(\d+[a-z]?)\b(.*)$", rest, re.IGNORECASE | re.DOTALL)
    if m is not None:
        sub, rest = m.group(1).lower(), m.group(2)
    else:
        m = re.match(r"^[\s,]*(first|second|third|fourth|fifth|sixth)\s+paragraph\b(.*)$", rest, re.IGNORECASE | re.DOTALL)
        if m is not None:
            sub, rest = _ORDINALS[m.group(1).lower()], m.group(2)
    if not _TAIL.match(rest):
        raise InvalidId(f"cannot read article reference tail {rest!r}")
    return sub


def _annex_sub(rest: str) -> str | None:
    """Point or section after an annex numeral: "(2)", "(A)", " point 2", " Section 1"; else None.
    Lower levels ("(a)" sub-points, points inside a section) are dropped."""
    rest = rest.strip()
    sub = None
    m = re.match(r"^\(\s*([A-Z]|\d+[a-z]?)\s*\)(.*)$", rest, re.DOTALL)
    if m is None:
        m = re.match(r"^[\s,]*(?:point|pt\.?|section)\s*([A-Z]|\d+[a-z]?)\b\.?(.*)$", rest, re.IGNORECASE | re.DOTALL)
    if m is not None:
        sub, rest = m.group(1), m.group(2)
        sub = sub.upper() if sub.isalpha() else sub.lower()
    rest = _ANNEX_DROP.sub("", rest)
    if not _TAIL.match(rest):
        raise InvalidId(f"cannot read annex reference tail {rest!r}")
    return sub


def parse_id(raw: str) -> SourceId:
    """Parse a canonical or loose ID into a SourceId at paragraph level.

    Tolerant input: "AIA:Art6(2)", "Art. 6(2)", "Article 6(2)", "Article 6(2)(a)", "AIA: Art 6 (2)",
    "Annex III point 2", "GLHR para 204", "GLHR:para204". A missing doc prefix on an article or annex means AIA.
    Article point letters below paragraph level are dropped ("Art6(2)(a)" -> sub "2").
    Annex section letters are kept ("AnnexI(A)"). Raises InvalidId otherwise.
    Also accepted: "Article 113, third paragraph" (-> sub "3"), "Article 6 paragraph 2", a trailing
    "AI Act", and "GLHR:204" / "GLHR (204)" for guideline paragraphs.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidId(f"empty source ID {raw!r}")
    text = " ".join(raw.split())
    doc = None
    m = _PREFIX.match(text)
    if m is not None:
        doc, text = m.group(1).upper(), m.group(2)
        if doc not in DOCS:
            raise InvalidId(f"unknown document prefix in {raw!r}")
    else:
        m = _LOOSE_PREFIX.match(text)
        if m is not None:
            doc, text = m.group(1).upper(), m.group(2)
    text = _ACT_SUFFIX.sub("", text).strip()
    text = re.sub(r"\s+\(", "(", text)  # "Art 6 (2)" -> "Art 6(2)"

    m = _ARTICLE.match(text)
    if m is not None:
        if doc not in (None, "AIA"):
            raise InvalidId(f"articles exist only in AIA: {raw!r}")
        return SourceId("AIA", "art", m.group(1).lower(), _article_sub(m.group(2)))

    m = _ANNEX.match(text)
    if m is not None:
        roman = m.group(1).upper()
        if doc not in (None, "AIA") or not _ROMAN.match(roman):
            raise InvalidId(f"not an AI Act annex: {raw!r}")
        return SourceId("AIA", "annex", roman, _annex_sub(m.group(2)))

    if doc in ("GLHR", "GLDEF"):
        m = _PARA.match(text)
        if m is not None:
            return SourceId(doc, "para", str(int(m.group(1))))
        m = _SEC.match(text)
        if m is not None:
            return SourceId(doc, "sec", m.group(1))
    raise InvalidId(f"cannot parse source ID {raw!r}")


def normalize_id(raw: str) -> str:
    """str(parse_id(raw)). Raises InvalidId."""
    return str(parse_id(raw))


def parent_id(source_id: str) -> str | None:
    """Parent of a sub-unit ID: "AIA:Art6(2)" -> "AIA:Art6", "AIA:AnnexIII(2)" -> "AIA:AnnexIII",
    "AIA:AnnexI(A)" -> "AIA:AnnexI". None for top-level IDs and for every GLHR / GLDEF ID."""
    sid = parse_id(source_id)
    if sid.kind not in ("art", "annex") or sid.sub is None:
        return None
    return str(SourceId(sid.doc, sid.kind, sid.number))


def matches(cited: str, gold: str) -> bool:
    """True if `cited` supports `gold`: equal after normalisation, or gold is article level and
    cited is one of its paragraphs. A bare article citation does NOT match a paragraph-level gold.
    Unparseable `cited` -> False (never raises for cited); unparseable gold -> InvalidId.
    Annexes follow the same rule: an annex-level gold is supported by any of its points or sections."""
    g = parse_id(gold)
    try:
        c = parse_id(cited)
    except InvalidId:
        return False
    if c == g:
        return True
    return (
        g.kind in ("art", "annex")
        and g.sub is None
        and c.sub is not None
        and (c.doc, c.kind, c.number) == (g.doc, g.kind, g.number)
    )
