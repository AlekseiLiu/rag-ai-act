"""Retrieval indexes over chunks: FAISS dense (Azure embeddings) and BM25. Both return top-k chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from langchain_core.embeddings import Embeddings

from rag_ai_act.ingest import Chunk

TOP_K = 5


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = TOP_K) -> list[Chunk]: ...


class DenseRetriever:
    """FAISS inner-product index over L2-normalised embeddings of chunk.text."""

    def __init__(self, chunks: list[Chunk], embeddings: Embeddings):
        raise NotImplementedError

    def retrieve(self, query: str, k: int = TOP_K) -> list[Chunk]:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path, chunks: list[Chunk], embeddings: Embeddings) -> DenseRetriever:
        raise NotImplementedError


class BM25Retriever:
    """rank_bm25.BM25Okapi over lower-cased word tokens of chunk.text."""

    def __init__(self, chunks: list[Chunk]):
        raise NotImplementedError

    def retrieve(self, query: str, k: int = TOP_K) -> list[Chunk]:
        raise NotImplementedError
