"""Opt-in service integration, excluded from the default test run."""

import json
from pathlib import Path

import pytest

from rag_ai_act.budget import Budget
from rag_ai_act.config import BudgetedEmbeddings, make_embeddings, make_llm
from rag_ai_act.index import BM25Retriever, DenseRetriever
from rag_ai_act.systems import Answer, run_agent, run_rag, run_s0
from rag_ai_act.tools import DateClauses, ProvisionStore

pytestmark = pytest.mark.live


@pytest.mark.parametrize("system", ["S0", "S1", "S2", "S3"])
def test_live_smoke_each_system(system, mini_units, mini_chunks):
    path = Path(__file__).resolve().parents[1] / "eval" / "dev_v0.jsonl"
    with path.open() as stream:
        question = json.loads(next(stream))["question"]
    budget = Budget(warn_eur=0.10, stop_eur=0.25)
    llm = make_llm()
    if system == "S0":
        answer = run_s0(question, llm, budget)
    elif system == "S1":
        answer = run_rag(question, llm, BM25Retriever(mini_chunks), budget)
    else:
        dense = DenseRetriever(mini_chunks, BudgetedEmbeddings(make_embeddings(), budget))
        if system == "S2":
            answer = run_rag(question, llm, dense, budget)
        else:
            answer = run_agent(
                question, llm, dense, ProvisionStore(mini_units), DateClauses(mini_units), budget
            )
    assert isinstance(answer, Answer) and answer.valid is True
    assert isinstance(answer.answer, str) and answer.answer.strip()
    assert isinstance(answer.cited_ids, list) and isinstance(answer.declined, bool)
    if answer.declined:
        assert answer.cited_ids == []
