"""Ingest: PDF -> clean text -> legal units with IDs -> chunks.

Outputs of `build_corpus`: data/processed/units.jsonl and data/processed/chunks.jsonl, one JSON object per line
with exactly the fields of Unit / Chunk below (dataclasses.asdict).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Corpus files (data/raw) -> doc code and part. The OJ 2024 original is never indexed.
CORPUS = {
    "CELEX_02024R1689-20260727_EN_TXT.pdf": ("AIA", None),
    "Draft_Guidelines_on_the_classification_on_highrisk_AI_general_principles_UQmTa2BdJ3IH8bIsWJyDqaWOY_128559.pdf": ("GLHR", "general_principles"),
    "Draft_Guidelines_on_the_classification_of_high_risk_AI_Annex_I_E9XHEqRLx555OO6iijKwJF32Cvc_128560.pdf": ("GLHR", "annex_I"),
    "Draft_Guidelines_on_the_classification_of_high_risk_AI_Annex_III_7MXR3YIz2GW3uPpJPWvvNDd8IOI_128561.pdf": ("GLHR", "annex_III"),
    "Commission_Guidelines_on_the_definition_of_an_artificial_intelligence_system_established_by_Regulation_EU_20241689_AI_ActEnglish_nf2skCqfRtJdFgGJAvCODoPCWZ4_112455.PDF": ("GLDEF", None),
}
DRAFT_DOCS = {"GLHR"}  # every unit of these docs has draft=True
MAX_CHUNK_TOKENS = 1000  # tokens counted with tiktoken cl100k_base


class CoverageError(RuntimeError):
    """Raised by check_coverage when units are missing or duplicated. Attributes `missing` and
    `duplicates` (lists of source IDs) say why."""

    def __init__(self, missing: list[str], duplicates: list[str]):
        super().__init__(f"missing={missing} duplicates={duplicates}")
        self.missing = missing
        self.duplicates = duplicates


@dataclass
class Unit:
    source_id: str  # canonical ID (ids.normalize_id form)
    doc: str  # "AIA" | "GLHR" | "GLDEF"
    part: str | None  # GLHR part name, else None
    header: str  # context header, e.g. "AI Act, Article 6 (Classification rules for high-risk AI systems), paragraph 2"
    text: str  # cleaned unit text without the header
    draft: bool
    order: int  # position in its document, 0-based, for stable ordering


@dataclass
class Chunk:
    chunk_id: str  # f"{source_id}#{part_index}"
    source_id: str
    doc: str
    draft: bool
    part_index: int  # 0 unless the unit was split
    text: str  # header + "\n" + unit text (or this part of it)


def extract_text(pdf_path: Path) -> str:
    """Run `pdftotext -layout <pdf> -` and return the text. Raises RuntimeError if pdftotext fails."""
    raise NotImplementedError


def clean_text(raw: str, doc: str) -> str:
    """Remove page headers/footers, consolidation markers (lines like '▼B', '▼M1', '▼C1'), footnote blocks,
    and join words hyphenated at line ends. Keeps paragraph-number indentation intact."""
    raise NotImplementedError


def parse_units(text: str, doc: str, part: str | None = None) -> list[Unit]:
    """Split cleaned text into legal units with canonical IDs (see ids.py).

    AIA: one unit per article paragraph ("Article N" headings; numbered paragraphs "1.", "2.", "1a."); Art. 3
    one unit per definition point; Art. 113 one unit per (unnumbered) paragraph; single-paragraph articles
    -> AIA:ArtN; annex points (AIA:AnnexIII(2)), Annex I sections (A)/(B). Points (a), (b) stay inside their
    paragraph. Deleted provisions produce no unit: in the consolidated text a deletion is a consolidation
    marker line (e.g. '▼M1') followed by a line of underscores ('__________') where the provision was.
    GLHR / GLDEF: one unit per numbered paragraph "(n)" at line start; unnumbered example boxes inside a
    section are merged into ONE unit GLHR:sec{x.y.z} per section (all boxes of 3.2.7 -> one unit
    GLHR:sec3.2.7, split into chunks if long). A paragraph broken by a page break stays one unit; stray number lines that
    are not paragraphs are ignored.
    """
    raise NotImplementedError


def chunk_units(units: list[Unit], max_tokens: int = MAX_CHUNK_TOKENS) -> list[Chunk]:
    """One chunk per unit; units longer than max_tokens split into consecutive parts with the same source_id."""
    raise NotImplementedError


def check_coverage(units: list[Unit]) -> dict:
    """Return {"aia_articles": n, "glhr_paras": n, "gldef_paras": n, "missing": [...], "duplicates": [...]}.
    Checks only the docs present in `units` (a list with only GLHR units is not checked for AIA):
    AIA: every article 1..113 (plus inserted ones found) has at least one unit; GLHR: every paragraph
    1..451 present; GLDEF: paragraphs 1..max found, no gaps. Raises CoverageError if anything is missing or
    any source_id occurs on two units (units are pre-chunking, so split parts never count)."""
    raise NotImplementedError


def build_corpus(raw_dir: Path, out_dir: Path) -> tuple[list[Unit], list[Chunk]]:
    """Extract, clean, parse, coverage-check and chunk every CORPUS file in raw_dir; write units.jsonl and
    chunks.jsonl to out_dir; return both lists."""
    raise NotImplementedError
