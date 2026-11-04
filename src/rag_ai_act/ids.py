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
        raise NotImplementedError


def parse_id(raw: str) -> SourceId:
    """Parse a canonical or loose ID into a SourceId at paragraph level.

    Tolerant input: "AIA:Art6(2)", "Art. 6(2)", "Article 6(2)", "Article 6(2)(a)", "AIA: Art 6 (2)",
    "Annex III point 2", "GLHR para 204", "GLHR:para204". A missing doc prefix on an article or annex means AIA.
    Article point letters below paragraph level are dropped ("Art6(2)(a)" -> sub "2").
    Annex section letters are kept ("AnnexI(A)"). Raises InvalidId otherwise.
    """
    raise NotImplementedError


def normalize_id(raw: str) -> str:
    """str(parse_id(raw)). Raises InvalidId."""
    raise NotImplementedError


def parent_id(source_id: str) -> str | None:
    """Parent of a sub-unit ID: "AIA:Art6(2)" -> "AIA:Art6", "AIA:AnnexIII(2)" -> "AIA:AnnexIII",
    "AIA:AnnexI(A)" -> "AIA:AnnexI". None for top-level IDs and for every GLHR / GLDEF ID."""
    raise NotImplementedError


def matches(cited: str, gold: str) -> bool:
    """True if `cited` supports `gold`: equal after normalisation, or gold is article level and
    cited is one of its paragraphs. A bare article citation does NOT match a paragraph-level gold.
    Unparseable `cited` -> False (never raises for cited); unparseable gold -> InvalidId."""
    raise NotImplementedError
