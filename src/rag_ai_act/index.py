"""Retrieval indexes over chunks: FAISS dense (Azure embeddings) and BM25. Both return top-k chunks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from rag_ai_act.ingest import Chunk

TOP_K = 5

_INDEX_FILE = "index.faiss"
_META_FILE = "chunks.json"


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = TOP_K) -> list[Chunk]: ...


def _normalised(vectors: list[list[float]]) -> np.ndarray:
    """Float32 matrix with every row scaled to unit L2 norm (all-zero rows stay zero)."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-d list of vectors, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def chunks_sha256(chunks: list[Chunk]) -> str:
    """sha256 over the ordered chunk IDs and texts: changes when any chunk's text, ID or position changes."""
    digest = hashlib.sha256()
    for c in chunks:
        for part in (c.chunk_id, c.text):
            data = part.encode("utf-8")
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
    return digest.hexdigest()


def embedding_deployment(embeddings: Embeddings) -> str | None:
    """The deployment (or model) name of an embeddings client, looking through wrappers that keep the
    wrapped client in `.inner`; None if it has none."""
    seen = 0
    while hasattr(embeddings, "inner") and seen < 10:
        embeddings, seen = embeddings.inner, seen + 1
    for attr in ("deployment", "model"):
        value = getattr(embeddings, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


class DenseRetriever:
    """FAISS inner-product index over L2-normalised embeddings of chunk.text.

    Inner product of unit vectors is cosine similarity. All embedding calls go through the given
    `embeddings` object (one `embed_documents` call when building, one `embed_query` call per query), so a
    budget-capping wrapper sees every call. `save` / `load` persist the index next to the chunk IDs it was
    built from, a sha256 of the ordered chunk texts and the embedding deployment name; `load` refuses
    chunks or a deployment that do not match.
    """

    def __init__(self, chunks: list[Chunk], embeddings: Embeddings):
        self.chunks = list(chunks)
        self.embeddings = embeddings
        self.deployment = embedding_deployment(embeddings)
        self.index: faiss.Index | None = None
        if self.chunks:
            matrix = _normalised(embeddings.embed_documents([c.text for c in self.chunks]))
            if matrix.shape[0] != len(self.chunks):
                raise ValueError(f"got {matrix.shape[0]} embeddings for {len(self.chunks)} chunks")
            self.index = faiss.IndexFlatIP(matrix.shape[1])
            self.index.add(matrix)

    def retrieve(self, query: str, k: int = TOP_K) -> list[Chunk]:
        """The k chunks with the highest cosine similarity to the query, best first."""
        if self.index is None or k <= 0:
            return []
        vector = _normalised([self.embeddings.embed_query(query)])
        if vector.shape[1] != self.index.d:
            raise ValueError(f"query embedding has {vector.shape[1]} dimensions, index has {self.index.d}")
        _, ids = self.index.search(vector, min(k, self.index.ntotal))
        return [self.chunks[i] for i in ids[0] if i >= 0]

    def save(self, path: Path) -> None:
        """Write the FAISS index, the ordered chunk IDs, the chunk-text hash (chunks_sha256) and the
        embedding deployment name into directory `path` (created if needed)."""
        if self.index is None:
            raise ValueError("cannot save an empty index")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / _INDEX_FILE))
        meta = {
            "chunk_ids": [c.chunk_id for c in self.chunks],
            "chunks_sha256": chunks_sha256(self.chunks),
            "embed_deployment": self.deployment,
            "dim": int(self.index.d),
        }
        (path / _META_FILE).write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, chunks: list[Chunk], embeddings: Embeddings) -> DenseRetriever:
        """Read an index written by `save`. `chunks` must be the same chunks, with the same texts and in
        the same order, that the index was built from (checked by chunk ID and chunks_sha256), and
        `embeddings` must use the same deployment (embedding_deployment); ValueError otherwise, also for an
        index saved without these records. No document embeddings are computed; `embeddings` is used only
        for queries."""
        path = Path(path)
        meta = json.loads((path / _META_FILE).read_text(encoding="utf-8"))
        if meta.get("chunk_ids") != [c.chunk_id for c in chunks]:
            raise ValueError(f"saved index at {path} was built from different chunks; rebuild it")
        if meta.get("chunks_sha256") != chunks_sha256(chunks):
            raise ValueError(f"saved index at {path} was built from different chunk texts; rebuild it")
        deployment = embedding_deployment(embeddings)
        if "embed_deployment" not in meta or meta["embed_deployment"] != deployment:
            raise ValueError(
                f"saved index at {path} was built with embedding deployment {meta.get('embed_deployment')!r}, "
                f"not {deployment!r}; rebuild it"
            )
        index = faiss.read_index(str(path / _INDEX_FILE))
        if index.ntotal != len(chunks):
            raise ValueError(f"saved index holds {index.ntotal} vectors for {len(chunks)} chunks")
        retriever = cls.__new__(cls)
        retriever.chunks = list(chunks)
        retriever.embeddings = embeddings
        retriever.deployment = deployment
        retriever.index = index
        return retriever


_WORD = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25Retriever:
    """rank_bm25.BM25Okapi over lower-cased word tokens of chunk.text."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = list(chunks)
        self.bm25 = BM25Okapi([_tokens(c.text) for c in self.chunks]) if self.chunks else None

    def retrieve(self, query: str, k: int = TOP_K) -> list[Chunk]:
        """The k chunks with the highest BM25 score, best first; ties keep document order."""
        if self.bm25 is None or k <= 0:
            return []
        scores = self.bm25.get_scores(_tokens(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], i))
        return [self.chunks[i] for i in ranked[:k]]
