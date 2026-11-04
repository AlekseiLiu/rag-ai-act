"""Ingest: PDF -> clean text -> legal units with IDs -> chunks.

Outputs of `build_corpus`: data/processed/units.jsonl and data/processed/chunks.jsonl, one JSON object per line
with exactly the fields of Unit / Chunk below (dataclasses.asdict).
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken

# Corpus files (data/raw) -> doc code and part. The OJ 2024 original is never indexed.
CORPUS = {
    "CELEX_02024R1689-20260727_EN_TXT.pdf": ("AIA", None),
    "Draft_Guidelines_on_the_classification_on_highrisk_AI_general_principles_UQmTa2BdJ3IH8bIsWJyDqaWOY_128559.pdf": ("GLHR", "general_principles"),
    "Draft_Guidelines_on_the_classification_of_high_risk_AI_Annex_I_E9XHEqRLx555OO6iijKwJF32Cvc_128560.pdf": ("GLHR", "annex_I"),
    "Draft_Guidelines_on_the_classification_of_high_risk_AI_Annex_III_7MXR3YIz2GW3uPpJPWvvNDd8IOI_128561.pdf": ("GLHR", "annex_III"),
    "Commission_Guidelines_on_the_definition_of_an_artificial_intelligence_system_established_by_Regulation_EU_20241689_AI_ActEnglish_nf2skCqfRtJdFgGJAvCODoPCWZ4_112455.PDF": ("GLDEF", None),
}
DRAFT_DOCS = {"GLHR"}  # every unit of these docs has draft=True
MAX_CHUNK_TOKENS = 1000  # tokens of the full chunk text (header included), tiktoken cl100k_base

AIA_ARTICLES = 113  # articles 1..113 (plus inserted ones such as 4a) must all be present
GLHR_PARAS = 451  # draft high-risk guideline paragraphs (1)..(451), numbered across the three parts
DOC_TITLES = {
    "AIA": "AI Act",
    "GLHR": "Draft high-risk guidelines",
    "GLDEF": "AI system definition guidelines",
}
# Annexes whose sections restart their point numbering: one unit per section.
SECTION_ANNEXES = {"I", "VIII", "XI"}


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


# --------------------------------------------------------------------------------------------- extraction


def extract_text(pdf_path: Path) -> str:
    """Run `pdftotext -layout <pdf> -` and return the text. Raises RuntimeError if pdftotext fails."""
    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"], capture_output=True, check=False
        )
    except OSError as exc:  # pdftotext missing or not executable
        raise RuntimeError(f"pdftotext could not be run: {exc}") from exc
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"pdftotext failed on {pdf_path} (exit {proc.returncode}): {err}")
    return proc.stdout.decode("utf-8", "replace")


# ------------------------------------------------------------------------------------------------ cleaning

# Running page header of the consolidated act: "<CELEX number> — EN — <dd.mm.yyyy> — <version> — <page>".
_AIA_PAGE_HEADER = re.compile(r"^\s*\d{5}[A-Z]\d{4}\s+—\s+[A-Z]{2}\s+—.*—\s*\d+\s*$")
# Consolidation markers: "▼B", "▼M1", "▼C1", "►B" (start of amended text), "◄" (end).
_MARKER_LINE = re.compile(r"^\s*(?:[▼►][A-Z]\d*|◄)\s*$")
_MARKER_TOKEN = re.compile(r"^(\s*)[▼►][A-Z]\d*\s+")
# Act footnotes sit at a page bottom and cite another legal act: "(1) Regulation (EU) No 1025/2012 ...".
_AIA_FOOTNOTE = re.compile(r"^\s*\(\d+\)\s+(?:Regulation|Directive|Decision)\b")
# Guideline footnotes start with a line holding only the footnote number, near the left margin.
_GL_FOOTNOTE = re.compile(r"^\s{0,20}\d{1,3}\s*$")
# Guideline page numbers: a number alone on a line, far to the right.
_GL_PAGE_NUMBER = re.compile(r"^\s{40,}\d{1,3}\s*$")
_GL_LANG_LINE = re.compile(r"^\s*EN\s+EN\s*$")
_TOC_LINE = re.compile(r"\.{6,}\s*\d*\s*$")
_PLACEHOLDER = re.compile(r"^\s*\[See [^\]]*\]\s*$", re.IGNORECASE)


def _clean_page(lines: list[str], doc: str) -> list[str]:
    """Drop the footnote block at the bottom of one page and the page's running header/number lines."""
    footnote = _AIA_FOOTNOTE if doc == "AIA" else _GL_FOOTNOTE
    for i, line in enumerate(lines):
        if footnote.match(line) and not (doc != "AIA" and _GL_PAGE_NUMBER.match(line)):
            # Everything from the first footnote line to the page end is footnote text.
            lines = lines[:i]
            break
    out = []
    for line in lines:
        if doc == "AIA" and _AIA_PAGE_HEADER.match(line):
            continue
        if doc != "AIA" and (_GL_PAGE_NUMBER.match(line) or _GL_LANG_LINE.match(line)):
            continue
        out.append(line)
    return out


def clean_text(raw: str, doc: str) -> str:
    """Remove page headers/footers, consolidation markers (lines like '▼B', '▼M1', '▼C1'), footnote blocks,
    and join words hyphenated at line ends. Keeps paragraph-number indentation intact.

    A consolidation marker line becomes an empty line (it always separates provisions). Words split by a
    soft hyphen are joined without the hyphen ("malfunc-/tioning" -> "malfunctioning"); words split after
    a hard hyphen keep it ("decision-/making" -> "decision-making"). Table-of-contents lines (dot leaders),
    "[See ...]" placeholders and deletion stubs (lines of underscores) are kept out of the units by
    parse_units, except dot-leader lines, which are dropped here.
    """
    lines: list[str] = []
    for page in raw.split("\f"):
        lines.extend(_clean_page(page.split("\n"), doc))
    out = []
    for line in lines:
        line = line.rstrip()
        if _MARKER_LINE.match(line):
            out.append("")
            continue
        line = _MARKER_TOKEN.sub(r"\1", line)
        if _TOC_LINE.search(line):
            continue
        out.append(line)
    text = "\n".join(out)
    # Soft hyphen at a line end (possibly across removed page furniture): join, drop the hyphen.
    text = re.sub(r"­[ \t]*\n(?:[ \t]*\n)*[ \t]*", "", text)
    text = text.replace("­", "")
    # Hard hyphen at a line end inside a word: join, keep the hyphen.
    text = re.sub(r"(?<=[A-Za-z])-[ \t]*\n(?:[ \t]*\n)*[ \t]*(?=[a-z])", "-", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip("\n") + "\n"


# ----------------------------------------------------------------------------------------- text helpers

_LIST_START = re.compile(r"^(?:\(\w{1,5}\)|[-—–•]\s|\d+(?:\.\d+)*\.\s|[a-z]\)\s|[ivx]+\.\s)")


def _tidy(lines: list[str]) -> str:
    """Join layout lines into readable text: one line per block or list item, blocks separated by newlines.
    A block that continues a sentence cut by a page break is joined to the block before it."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or set(line) == {"_"}:
            if current:
                blocks.append(current)
                current = []
            continue
        if not current and blocks and line[0].islower() and not re.search(r"[.:;!?]$", blocks[-1][-1]):
            # A sentence broken by a page break (blank lines in between) continues its block.
            current = blocks.pop()
        if current and _LIST_START.match(line):
            blocks.append(current)
            current = []
        current.append(line)
    if current:
        blocks.append(current)
    return "\n".join(" ".join(block) for block in blocks)


def _blocks_after(lines: list[str], start: int, stop: re.Pattern | None = None, limit: int = 3) -> int:
    """Index just past the non-blank block that starts at `start` (at most `limit` lines; a line matching
    `stop` ends it early)."""
    i = start
    while i < len(lines) and i - start < limit and lines[i].strip():
        if i > start and stop is not None and stop.match(lines[i]):
            break
        i += 1
    return i


def _next_nonblank(lines: list[str], i: int) -> int:
    while i < len(lines) and not lines[i].strip():
        i += 1
    return i


def _num_key(number: str) -> tuple[int, str]:
    m = re.match(r"(\d+)([a-z]*)", number)
    return (int(m.group(1)), m.group(2)) if m else (0, "")


def _is_successor(prev: str | None, cand: str, gap_ok: bool) -> bool:
    """Whether numbered item `cand` may follow `prev` ("1" -> "1a" -> "1b" -> "2"; gaps only if gap_ok)."""
    if prev is None:
        return True
    (pb, pl), (cb, cl) = _num_key(prev), _num_key(cand)
    if cb == pb:
        return cl > pl
    if cb == pb + 1:
        return True
    return cb > pb and gap_ok


# ------------------------------------------------------------------------------------------- AI Act parser

_ART_HEAD = re.compile(r"^\s{8,}Article (\d+[a-z]?)\s*$")
_ANNEX_HEAD = re.compile(r"^\s{8,}ANNEX ([IVX]+)\s*$")
_CHAPTER_HEAD = re.compile(r"^\s*(?:CHAPTER [IVX]+|SECTION \d+)\s*$")
_PARA_NUM = re.compile(r"^\s{0,14}(\d+[a-z]?)\.(?:\s+\S|\s*$)")
_DEF_NUM = re.compile(r"^\s{0,14}\((\d+[a-z]?)\)\s+\S")
_ANNEX_SECTION = re.compile(r"^\s*Section ([A-Z]|\d+)(?:\s*$|\.\s|\s+—)")
_DELETED = re.compile(r"^\s*_{3,}\s*$")
_POINT_START = re.compile(r"^\s*\(\w{1,5}\)\s")


class _Provision:
    """Lines of one article or annex, split into numbered items as they are read."""

    def __init__(self, kind: str, number: str, title: str):
        self.kind, self.number, self.title = kind, number, title
        self.preamble: list[str] = []
        self.items: list[tuple[str, list[str]]] = []  # (sub, lines)
        self.deleted_since_last = False

    def add(self, line: str) -> None:
        (self.items[-1][1] if self.items else self.preamble).append(line)


def _act_units(prov: _Provision) -> list[tuple[str, str, str]]:
    """(source_id, header, text) for one article or annex."""
    name = "Article" if prov.kind == "art" else "Annex"
    label = "Art" if prov.kind == "art" else "Annex"
    base = f"AIA:{label}{prov.number}"
    head = f"AI Act, {name} {prov.number}" + (f" ({prov.title})" if prov.title else "")
    items = [(sub, lines) for sub, lines in prov.items if _tidy(lines)]
    if not prov.items and prov.kind == "art" and prov.number == "113":
        items = [(str(i), lines) for i, lines in enumerate(_unnumbered_paragraphs(prov.preamble), 1)]
        prov.preamble = []
    if not items:
        text = _tidy(prov.preamble)
        return [(base, head, text)] if text else []
    if prov.preamble and _tidy(prov.preamble):
        # Introductory words before the first numbered item ("the following definitions apply:").
        items[0] = (items[0][0], prov.preamble + [""] + items[0][1])
    if prov.kind == "art":
        what = "point" if prov.number == "3" else "paragraph"
    else:
        what = "Section" if prov.number in SECTION_ANNEXES else "point"
    return [(f"{base}({sub})", f"{head}, {what} {sub}", _tidy(lines)) for sub, lines in items]


def _unnumbered_paragraphs(lines: list[str]) -> list[list[str]]:
    """Split an article without paragraph numbers into its paragraphs: blank-line blocks, where a block
    starting with a point label "(a)" or following a block that does not end a sentence continues the
    previous paragraph."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines + [""]:
        if line.strip() and not _DELETED.match(line):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    paragraphs: list[list[str]] = []
    for block in blocks:
        prev_open = paragraphs and not re.search(r"[.;]$", paragraphs[-1][-1].strip())
        if paragraphs and (_POINT_START.match(block[0]) or prev_open):
            paragraphs[-1].extend([""] + block)
        else:
            paragraphs.append(list(block))
    return paragraphs


def _parse_act(lines: list[str]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    prov: _Provision | None = None
    last_article: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m_art, m_annex = _ART_HEAD.match(line), _ANNEX_HEAD.match(line)
        if m_art and _is_successor(last_article, m_art.group(1), gap_ok=True) or m_annex:
            if prov is not None:
                out.extend(_act_units(prov))
            # The title is the block right after the heading.
            start = _next_nonblank(lines, i + 1)
            end = _blocks_after(lines, start, stop=re.compile(r"^\s*(?:\d+[a-z]?\.|\(\w+\))\s"))
            title = " ".join(t.strip() for t in lines[start:end])
            if m_art:
                last_article = m_art.group(1)
                prov = _Provision("art", m_art.group(1), title)
            else:
                prov = _Provision("annex", m_annex.group(1), title)
            i = end
            continue
        if _CHAPTER_HEAD.match(line):
            # Chapter / section heading and its title block belong to no provision.
            if prov is not None:
                out.extend(_act_units(prov))
                prov = None
            i = _blocks_after(lines, _next_nonblank(lines, i + 1))
            continue
        if prov is None:
            i += 1
            continue
        if _DELETED.match(line):
            prov.deleted_since_last = True
            i += 1
            continue
        sub = _item_start(prov, lines, i)
        if sub is not None:
            prov.items.append((sub, []))
            prov.deleted_since_last = False
        prov.add(line)
        i += 1
    if prov is not None:
        out.extend(_act_units(prov))
    return out


def _item_start(prov: _Provision, lines: list[str], i: int) -> str | None:
    """Sub-ID if line i opens a new numbered paragraph, definition, annex point or annex section."""
    line = lines[i]
    prev_blank = i == 0 or not lines[i - 1].strip()
    last = prov.items[-1][0] if prov.items else None
    if prov.kind == "annex" and prov.number in SECTION_ANNEXES:
        m = _ANNEX_SECTION.match(line)
        if m and (last is None or m.group(1) != last):
            return m.group(1)
        return None
    m = (_DEF_NUM if prov.kind == "art" and prov.number == "3" else _PARA_NUM).match(line)
    if m is None or not prev_blank:
        return None
    sub = m.group(1)
    if last is None:
        return sub if _num_key(sub)[0] <= 2 or prov.deleted_since_last else None
    return sub if _is_successor(last, sub, gap_ok=prov.deleted_since_last) else None


# -------------------------------------------------------------------------------------- guidelines parser

_GL_PARA = re.compile(r"^\s{0,16}\((\d{1,3})\)(?:\s+\S|\s*$)")
_GL_PART = re.compile(r"^\s*([IVX]+)\.\s+(\S.*)$")
_GL_SECTION = re.compile(r"^\s*(\d+(?:\.\d+)*)(\.?)\s+(\S.*)$")
_GL_SUBHEAD = re.compile(r"^\s*([ivx]+|[a-z])[.)]\s+(\S.*)$")
_PRACTICAL = re.compile(r"^Practical examples?\b")
_TITLE_START = re.compile(r"^[A-Z‘'\"“]")


def _section_successor(cur: tuple[int, ...] | None, cand: tuple[int, ...]) -> bool:
    if cur is None:
        return True
    if cand == cur + (1,):
        return True
    return any(cand == cur[: k - 1] + (cur[k - 1] + 1,) for k in range(1, len(cur) + 1))


class _GuidelineState:
    def __init__(self, doc: str):
        self.doc = doc
        self.part_title: str | None = None
        self.section: tuple[int, ...] | None = None
        self.section_title = ""
        self.last_para: int | None = None
        self.units: list[dict] = []  # {"sid", "context", "lines", "pos"}
        self.boxes: dict[str, dict] = {}
        self.current: dict | None = None  # unit receiving continuation lines
        self.orphan: list[str] = []  # unnumbered text after a heading, before the next paragraph

    def context(self) -> str:
        if self.section is not None:
            return f"{'.'.join(map(str, self.section))} {self.section_title}".strip()
        return self.part_title or ""


def _heading(lines: list[str], i: int, state: _GuidelineState) -> tuple[str, int, tuple] | None:
    """Recognise a guideline heading at line i: ("part" | "section" | "sub" | "box", end index, data)."""
    line = lines[i]
    prev_blank = i == 0 or not lines[i - 1].strip()
    end = _blocks_after(lines, i, stop=_GL_PARA)
    if end - i > 2 and end < len(lines) and lines[end].strip() and not _GL_PARA.match(lines[end]):
        return None  # a long block is body text, not a heading
    text = " ".join(" ".join(lines[i:end]).split())
    m = _GL_PART.match(line)
    if m and prev_blank and _TITLE_START.match(m.group(2)):
        return "part", end, (text,)
    m = _GL_SECTION.match(line)
    if m and _TITLE_START.match(m.group(3)) and not re.search(r"[.;:,]$", text):
        number = tuple(int(n) for n in m.group(1).split("."))
        if (len(number) > 1 or m.group(2)) and _section_successor(state.section, number):
            title = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s+", "", text)
            return "section", end, (number, title)
    m = _GL_SUBHEAD.match(line)
    if m and prev_blank and _TITLE_START.match(m.group(2)) and not re.search(r"[.;:,]$", text):
        return ("box" if _PRACTICAL.match(m.group(2)) else "sub"), end, (text,)
    # An unlabelled one-line heading right above a numbered paragraph or a labelled heading
    # ("Basic data processing", "Second use case: ...").
    j = _next_nonblank(lines, end)
    nxt = lines[j] if j < len(lines) else ""
    labelled = (_GL_SUBHEAD.match(nxt) or _GL_SECTION.match(nxt)) and _TITLE_START.match(
        re.sub(r"^\s*\S+\s+", "", nxt)
    )
    m = _GL_PARA.match(nxt)
    para_next = m is not None and int(m.group(1)) > (state.last_para or 0)
    short = len(text) <= 110 and not re.search(r"[.;:,]$", text)
    if prev_blank and end == i + 1 and short and _TITLE_START.match(text) and (para_next or labelled):
        return "sub", end, (text,)
    return None


def _parse_guidelines(lines: list[str], doc: str) -> list[dict]:
    st = _GuidelineState(doc)
    i = 0
    while i < len(lines):
        line = lines[i]
        if _PLACEHOLDER.match(line):
            i += 1
            continue
        m = _GL_PARA.match(line)
        if m:
            n = int(m.group(1))
            prev = lines[i - 1] if i > 0 else ""
            prev_open = not prev.strip() or _heading(lines, max(i - 1, 0), st) is not None
            if st.last_para is None or n == st.last_para + 1 or (n > st.last_para and prev_open):
                unit = {"sid": f"{doc}:para{n}", "context": st.context(), "lines": [], "pos": i}
                if st.orphan and st.units:
                    unit["lines"] = st.orphan + [""]
                st.orphan = []
                st.units.append(unit)
                st.current = unit
                st.last_para = n
                unit["lines"].append(line)
                i += 1
                continue
        if line.strip():
            head = _heading(lines, i, st)
            if head is not None:
                kind, end, data = head
                if kind == "part":
                    st.part_title, st.section, st.section_title = data[0], None, ""
                    st.current = None
                elif kind == "section":
                    st.section, st.section_title = data
                    st.current = None
                elif kind == "sub":
                    st.current = None
                elif kind == "box" and st.section is not None:
                    key = ".".join(map(str, st.section[:3]))
                    box = st.boxes.get(key)
                    if box is None:
                        box = {"sid": f"{doc}:sec{key}", "context": st.context(), "lines": [], "pos": i}
                        st.boxes[key] = box
                        st.units.append(box)
                    elif box["lines"]:
                        box["lines"].append("")
                    box["lines"].extend(lines[i:end])
                    st.current = box
                elif kind == "box" and st.current is not None:
                    st.current["lines"].extend(lines[i:end])
                i = end
                continue
        if st.current is not None:
            st.current["lines"].append(line)
        elif line.strip() and st.units:
            st.orphan.append(line)
        i += 1
    if st.orphan and st.units:
        # Trailing unnumbered text with no paragraph after it joins the last unit.
        st.units[-1]["lines"].extend([""] + st.orphan)
    return st.units


def _gl_header(doc: str, context: str, sid: str) -> str:
    kind = "practical examples" if ":sec" in sid else f"para {sid.split(':para')[-1]}"
    parts = [DOC_TITLES[doc]] + ([context] if context else []) + [kind]
    return ", ".join(parts)


# ----------------------------------------------------------------------------------------------- units


def parse_units(text: str, doc: str, part: str | None = None) -> list[Unit]:
    """Split cleaned text into legal units with canonical IDs (see ids.py).

    AIA: one unit per article paragraph ("Article N" headings; numbered paragraphs "1.", "2.", "1a."); Art. 3
    one unit per definition point; Art. 113 one unit per (unnumbered) paragraph; single-paragraph articles
    -> AIA:ArtN; annex points (AIA:AnnexIII(2)); one unit per section for Annexes I, VIII and XI (see ids.py). Points (a), (b) stay inside their
    paragraph. Deleted provisions produce no unit: in the consolidated text a deletion is a consolidation
    marker line (e.g. '▼M1') followed by a line of underscores ('__________') where the provision was.
    GLHR / GLDEF: one unit per numbered paragraph "(n)" at line start; unnumbered example boxes inside a
    section (the blocks headed "Practical example" / "Practical examples" inside a numbered section x.y.z;
    any other "Example:" lines join the preceding numbered paragraph) are merged into ONE unit
    GLHR:sec{x.y.z} per section (all boxes of 3.2.7 -> one unit
    GLHR:sec3.2.7, split into chunks if long). A paragraph broken by a page break stays one unit; stray number lines that
    are not paragraphs are ignored.

    Further rules: a numbered item starts only on a line after a blank line and only with the next number
    in sequence (an inserted "1a" after "1"; a gap only after a deletion), so wrapped lines that begin with
    a number are not mistaken for items. Text before the first numbered item of an article or annex is
    prepended to that item. Chapter and section headings of the Act belong to no unit. A box ID uses the
    first three levels of its section number (boxes of 3.1.2.1 -> GLHR:sec3.1.2). Guideline headings are
    kept out of paragraph text and used for the header; unnumbered text between a heading and the next
    numbered paragraph is prepended to that paragraph.
    """
    lines = text.split("\n")
    if doc == "AIA":
        triples = _parse_act(lines)
    else:
        triples = [
            (u["sid"], _gl_header(doc, u["context"], u["sid"]), _tidy(u["lines"]))
            for u in sorted(_parse_guidelines(lines, doc), key=lambda u: u["pos"])
        ]
    draft = doc in DRAFT_DOCS
    return [
        Unit(sid, doc, part, header, body, draft, order)
        for order, (sid, header, body) in enumerate(t for t in triples if t[2])
    ]


# ---------------------------------------------------------------------------------------------- chunks

_ENCODING = None


def _encoding():
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def _n_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def _split_body(prefix: str, body: str, max_tokens: int, seps: tuple[str, ...] = ("\n", " ")) -> list[str]:
    """Greedy split of body into parts so that prefix + part stays within max_tokens; prefers the first
    separator (block breaks) and falls back to finer ones (words, then characters)."""
    if _n_tokens(prefix + body) <= max_tokens:
        return [body]
    if not seps:
        # A single "word" longer than the budget: cut by characters.
        parts, current = [], ""
        for ch in body:
            if current and _n_tokens(prefix + current + ch) > max_tokens:
                parts.append(current)
                current = ""
            current += ch
        return parts + ([current] if current else [])
    sep, finer = seps[0], seps[1:]
    pieces = body.split(sep)
    parts: list[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else current + sep + piece
        if _n_tokens(prefix + candidate) <= max_tokens:
            current = candidate
            continue
        if current:
            parts.append(current)
        if _n_tokens(prefix + piece) <= max_tokens:
            current = piece
        else:
            sub = _split_body(prefix, piece, max_tokens, finer)
            parts.extend(sub[:-1])
            current = sub[-1]
    if current:
        parts.append(current)
    return parts


def chunk_units(units: list[Unit], max_tokens: int = MAX_CHUNK_TOKENS) -> list[Chunk]:
    """One chunk per unit; units longer than max_tokens split into consecutive parts with the same source_id.

    The token count covers the whole chunk text (header, newline and body). Splits fall between text
    blocks where possible, then between words; parts are roughly equal in size and lose no text.
    """
    chunks: list[Chunk] = []
    for unit in units:
        prefix = unit.header + "\n"
        if _n_tokens(prefix) >= max_tokens:
            raise ValueError(f"header of {unit.source_id} alone exceeds {max_tokens} tokens")
        total = _n_tokens(prefix + unit.text)
        if total <= max_tokens:
            bodies = [unit.text]
        else:
            # Aim for equal parts: a budget a little above total / n_parts, capped at max_tokens.
            n_parts = math.ceil(total / max_tokens)
            budget = min(max_tokens, math.ceil(total / n_parts) + _n_tokens(prefix) + 20)
            bodies = [b.strip() for b in _split_body(prefix, unit.text, budget)]
            bodies = [b for b in bodies if b]
        for index, body in enumerate(bodies):
            chunks.append(
                Chunk(f"{unit.source_id}#{index}", unit.source_id, unit.doc, unit.draft, index, prefix + body)
            )
    return chunks


# -------------------------------------------------------------------------------------------- coverage


def check_coverage(units: list[Unit]) -> dict:
    """Return {"aia_articles": n, "glhr_paras": n, "gldef_paras": n, "missing": [...], "duplicates": [...]}.
    Checks only the docs present in `units` (a list with only GLHR units is not checked for AIA):
    AIA: every article 1..113 (plus inserted ones found) has at least one unit; GLHR: every paragraph
    1..451 present; GLDEF: paragraphs 1..max found, no gaps. Raises CoverageError if anything is missing or
    any source_id occurs on two units (units are pre-chunking, so split parts never count)."""
    from rag_ai_act.ids import parse_id

    seen: set[str] = set()
    duplicates: list[str] = []
    articles: set[str] = set()
    paras: dict[str, set[int]] = {"GLHR": set(), "GLDEF": set()}
    docs = set()
    for unit in units:
        docs.add(unit.doc)
        if unit.source_id in seen and unit.source_id not in duplicates:
            duplicates.append(unit.source_id)
        seen.add(unit.source_id)
        sid = parse_id(unit.source_id)
        if sid.kind == "art":
            articles.add(sid.number)
        elif sid.kind == "para":
            paras[sid.doc].add(int(sid.number))
    missing: list[str] = []
    if "AIA" in docs:
        missing += [f"AIA:Art{n}" for n in range(1, AIA_ARTICLES + 1) if str(n) not in articles]
    if "GLHR" in docs:
        missing += [f"GLHR:para{n}" for n in range(1, GLHR_PARAS + 1) if n not in paras["GLHR"]]
    if "GLDEF" in docs and paras["GLDEF"]:
        missing += [f"GLDEF:para{n}" for n in range(1, max(paras["GLDEF"]) + 1) if n not in paras["GLDEF"]]
    if missing or duplicates:
        raise CoverageError(missing, duplicates)
    return {
        "aia_articles": len(articles),
        "glhr_paras": len(paras["GLHR"]),
        "gldef_paras": len(paras["GLDEF"]),
        "missing": missing,
        "duplicates": duplicates,
    }


# ---------------------------------------------------------------------------------------------- corpus


def build_corpus(raw_dir: Path, out_dir: Path) -> tuple[list[Unit], list[Chunk]]:
    """Extract, clean, parse, coverage-check and chunk every CORPUS file in raw_dir; write units.jsonl and
    chunks.jsonl to out_dir; return both lists.

    The raw (pre-cleaning) extraction of each CORPUS file is also written to `<raw_dir>/../text/<stem>.txt`
    (the folder is created), so the text the parser saw can be inspected and reused. Unit order is made
    unique per document: the GLHR parts are numbered one after another in CORPUS order.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    text_dir = raw_dir.parent / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    units: list[Unit] = []
    next_order: dict[str, int] = {}
    for filename, (doc, part) in CORPUS.items():
        raw = extract_text(raw_dir / filename)
        (text_dir / Path(filename).with_suffix(".txt").name).write_text(raw, encoding="utf-8")
        parsed = parse_units(clean_text(raw, doc), doc, part)
        offset = next_order.get(doc, 0)
        for unit in parsed:
            unit.order += offset
        next_order[doc] = offset + len(parsed)
        units.extend(parsed)
    check_coverage(units)
    chunks = chunk_units(units)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("units.jsonl", units), ("chunks.jsonl", chunks)):
        with open(out_dir / name, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in rows)
    return units, chunks
