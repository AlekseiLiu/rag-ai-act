"""Agent tools: search, get_provision, applicability_date. No date literals anywhere in src/."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from rag_ai_act.index import Retriever
from rag_ai_act.ingest import Unit

NOT_FOUND = "NOT_FOUND"


class ProvisionStore:
    """Exact lookup of units by ID. An article-level ID returns all its paragraphs in document order."""

    def __init__(self, units: list[Unit]):
        raise NotImplementedError

    def get(self, source_id: str) -> list[Unit]:
        """Accepts loose IDs (ids.parse_id). Unknown or unparseable ID -> [] (never raises)."""
        raise NotImplementedError


class DateClauses:
    """Clauses of AIA Art. 113 and Art. 111 (only these), built from the units at construction."""

    def __init__(self, units: list[Unit]):
        raise NotImplementedError

    def lookup(self, provision: str) -> list[Unit]:
        """Units of Art. 113 / 111 whose text mentions `provision`, which may be an ID or a phrase such as
        "Annex III", "Article 6(1)", "Chapter III", "Section 4". Returns [] if none."""
        raise NotImplementedError


def format_units(units: list[Unit]) -> str:
    """Tool output text: for each unit "[<source_id>] <header>\n<text>" (draft units get " (DRAFT)" after the
    ID), separated by blank lines; NOT_FOUND if units is empty."""
    raise NotImplementedError


def make_tools(retriever: Retriever, store: ProvisionStore, dates: DateClauses) -> list[BaseTool]:
    """Three LangChain tools named exactly "search" (arg: query), "get_provision" (arg: source_id) and
    "applicability_date" (arg: provision); each returns format_units(...) text."""
    raise NotImplementedError
