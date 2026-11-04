import pytest

from rag_ai_act.index import BM25Retriever, DenseRetriever
from tests.fakes import WordEmbeddings


@pytest.mark.parametrize("kind", ["bm25", "dense"])
def test_retrieval_returns_relevant_real_chunks_and_respects_k(kind, mini_chunks):
    retriever = (
        BM25Retriever(mini_chunks) if kind == "bm25" else DenseRetriever(mini_chunks, WordEmbeddings())
    )
    target = next(c for c in mini_chunks if c.source_id == "GLHR:para205")
    query = "treaties" if kind == "bm25" else target.text
    found = retriever.retrieve(query, k=2)
    assert len(found) == 2 and all(c in mini_chunks for c in found)
    assert found[0].source_id == "GLHR:para205"
    assert len(retriever.retrieve(query)) == 5


def test_dense_save_load_preserves_ranking(tmp_path, mini_chunks):
    embeddings = WordEmbeddings()
    retriever = DenseRetriever(mini_chunks, embeddings)
    query = "electricity storage distribution transmission"
    before = retriever.retrieve(query)
    assert len(before) == 5 and all(c in mini_chunks for c in before)
    path = tmp_path / "dense"
    retriever.save(path)
    restored = DenseRetriever.load(path, mini_chunks, embeddings)
    assert [c.chunk_id for c in restored.retrieve(query)] == [c.chunk_id for c in before]
