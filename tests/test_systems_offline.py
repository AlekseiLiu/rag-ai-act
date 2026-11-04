import json
from itertools import islice

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from rag_ai_act.budget import Budget, BudgetExceeded
from rag_ai_act.index import BM25Retriever, DenseRetriever
from rag_ai_act.systems import Answer, run_agent, run_rag, run_s0
from rag_ai_act.tools import DateClauses, ProvisionStore
from tests.fakes import ScriptedChat, WordEmbeddings, chat, response

QUESTION = "When is an electricity system considered high-risk?"
FINAL = json.dumps(
    {
        "answer": "The draft describes critical electricity entities.",
        "cited_ids": ["GLHR:para204"],
        "declined": False,
    }
)


def assert_answer(answer, budget):
    assert isinstance(answer, Answer)
    assert answer.valid is True and answer.declined is False
    assert answer.answer == "The draft describes critical electricity entities."
    assert answer.cited_ids == ["GLHR:para204"]
    assert {"tokens_in", "tokens_out", "eur"} <= answer.usage.keys()
    assert all(answer.usage[k] >= 0 for k in ("tokens_in", "tokens_out", "eur"))
    assert budget.reserved_eur == pytest.approx(0)
    assert budget.spent_eur > 0
    assert answer.usage["eur"] == pytest.approx(budget.spent_eur)
    assert answer.usage["tokens_in"] > 0


def test_s0_offline_answer_and_question_delivery():
    model = chat([response(FINAL)])
    budget = Budget()
    answer = run_s0(QUESTION, model, budget)
    assert_answer(answer, budget)
    assert answer.retrieved_ids == [] and answer.trace == []
    assert len(model.seen) == 1
    assert QUESTION in "\n".join(str(m.content) for m in model.seen[0])


@pytest.mark.parametrize("system", ["S1", "S2"])
def test_rag_offline_answer_retrieval_and_context(system, mini_chunks):
    retriever = (
        BM25Retriever(mini_chunks) if system == "S1" else DenseRetriever(mini_chunks, WordEmbeddings())
    )
    model = chat([response(FINAL)])
    budget = Budget()
    answer = run_rag(QUESTION, model, retriever, budget)
    assert_answer(answer, budget)
    expected = retriever.retrieve(QUESTION)
    assert len(answer.retrieved_ids) == 5
    assert answer.retrieved_ids == [c.source_id for c in expected]
    assert len(model.seen) == 1
    prompt = "\n".join(str(m.content) for m in model.seen[0])
    assert QUESTION in prompt
    for chunk in expected:
        assert chunk.text in prompt and chunk.source_id in prompt


def call(name, args, call_id):
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


def test_agent_executes_tools_records_returned_ids_and_parses_final(mini_units, mini_chunks):
    retriever = BM25Retriever(mini_chunks)
    model = chat(
        [
            call("search", {"query": "electricity critical entity"}, "search-1"),
            call("get_provision", {"source_id": "GLHR:para204"}, "provision-1"),
            call("applicability_date", {"provision": "Annex III"}, "date-1"),
            response(FINAL),
        ]
    )
    budget = Budget()
    answer = run_agent(
        QUESTION, model, retriever, ProvisionStore(mini_units), DateClauses(mini_units), budget
    )
    assert_answer(answer, budget)
    assert [entry["tool"] for entry in answer.trace] == ["search", "get_provision", "applicability_date"]
    expected = [
        [c.source_id for c in retriever.retrieve("electricity critical entity")],
        ["GLHR:para204"],
        ["AIA:Art113(3)"],
    ]
    for entry, ids in zip(answer.trace, expected, strict=True):
        assert entry["args"]
        assert entry["returned_ids"]
        assert set(entry["returned_ids"]) == set(ids)
    flat = [sid for entry in answer.trace for sid in entry["returned_ids"]]
    assert answer.retrieved_ids == list(dict.fromkeys(flat))
    tool_messages = [m for messages in model.seen for m in messages if isinstance(m, ToolMessage)]
    assert any("[GLHR:para204]" in str(m.content) for m in tool_messages)
    assert any("[AIA:Art113(3)]" in str(m.content) for m in tool_messages)


@pytest.mark.parametrize(("limit", "batch_size"), [(1, 1), (2, 1), (2, 4)])
def test_agent_bounds_actual_tool_calls_even_with_repeated_or_batched_requests(
    mini_units,
    mini_chunks,
    monkeypatch,
    limit,
    batch_size,
):
    calls = []
    original = ProvisionStore.get

    def counted(store, source_id):
        calls.append(source_id)
        return original(store, source_id)

    monkeypatch.setattr(ProvisionStore, "get", counted)

    def requests():
        for turn in range(30):
            yield AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_provision",
                        "args": {"source_id": "GLHR:para204"},
                        "id": f"request-{turn}-{i}",
                        "type": "tool_call",
                    }
                    for i in range(batch_size)
                ],
            )

    model = chat(list(islice(requests(), 30)))
    answer = run_agent(
        QUESTION,
        model,
        BM25Retriever(mini_chunks),
        ProvisionStore(mini_units),
        DateClauses(mini_units),
        Budget(),
        max_tool_calls=limit,
    )
    assert isinstance(answer, Answer) and isinstance(answer.valid, bool)
    assert 0 < len(answer.trace) <= limit
    assert 0 < len(calls) <= limit
    assert len(model.seen) <= limit + 2
    assert all(entry["returned_ids"] == ["GLHR:para204"] for entry in answer.trace)


def test_rag_malformed_model_output_does_not_crash(mini_chunks):
    answer = run_rag(QUESTION, chat([response("broken JSON")]), BM25Retriever(mini_chunks), Budget())
    assert answer.valid is False and len(answer.retrieved_ids) == 5


@pytest.mark.parametrize("system", ["S0", "S1", "S2", "S3"])
@pytest.mark.parametrize("origin", ["reservation", "model"])
def test_systems_propagate_budget_stop(system, origin, mini_units, mini_chunks, monkeypatch):
    model = chat([response(FINAL)])
    budget = Budget(warn_eur=0, stop_eur=0) if origin == "reservation" else Budget()
    error = BudgetExceeded("Synthetic model budget stop")
    calls = []

    def stopped(*args, **kwargs):
        calls.append(True)
        raise error

    if origin == "model":
        monkeypatch.setattr(ScriptedChat, "_generate", stopped)
    if system != "S0":
        retriever = (
            BM25Retriever(mini_chunks) if system == "S1" else DenseRetriever(mini_chunks, WordEmbeddings())
        )
    if system == "S3":
        store, dates = ProvisionStore(mini_units), DateClauses(mini_units)
    with pytest.raises(BudgetExceeded) as exc:
        if system == "S0":
            run_s0(QUESTION, model, budget)
        elif system == "S3":
            run_agent(QUESTION, model, retriever, store, dates, budget)
        else:
            run_rag(QUESTION, model, retriever, budget)
    if origin == "model":
        assert exc.value is error and calls


def test_agent_propagates_tool_budget_error(mini_units, mini_chunks, monkeypatch):
    error = BudgetExceeded("Synthetic tool budget stop")
    calls = []

    def stopped(*args, **kwargs):
        calls.append(True)
        raise error

    store, dates = ProvisionStore(mini_units), DateClauses(mini_units)
    monkeypatch.setattr(ProvisionStore, "get", stopped)
    model = chat([call("get_provision", {"source_id": "GLHR:para204"}, "budget-call"), response(FINAL)])
    with pytest.raises(BudgetExceeded) as exc:
        run_agent(QUESTION, model, BM25Retriever(mini_chunks), store, dates, Budget())
    assert exc.value is error and calls


@pytest.mark.parametrize("error_kind", ["model", "tool"])
def test_agent_converts_service_errors_to_invalid_answers(error_kind, mini_units, mini_chunks, monkeypatch):
    calls = []

    def broken(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("Synthetic service failure")

    model = chat([call("get_provision", {"source_id": "GLHR:para204"}, "bad-call"), response(FINAL)])
    if error_kind == "model":
        monkeypatch.setattr(ScriptedChat, "_generate", broken)
    else:
        monkeypatch.setattr(ProvisionStore, "get", broken)
    answer = run_agent(
        QUESTION,
        model,
        BM25Retriever(mini_chunks),
        ProvisionStore(mini_units),
        DateClauses(mini_units),
        Budget(),
    )
    assert isinstance(answer, Answer) and answer.valid is False
    assert calls


@pytest.mark.parametrize("system", ["S0", "S1", "S2"])
def test_baselines_convert_model_errors_to_invalid_answers(system, mini_chunks, monkeypatch):
    calls = []

    def broken(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("Synthetic model failure")

    model = chat([response(FINAL)])
    monkeypatch.setattr(ScriptedChat, "_generate", broken)
    if system == "S0":
        answer = run_s0(QUESTION, model, Budget())
    else:
        retriever = (
            BM25Retriever(mini_chunks) if system == "S1" else DenseRetriever(mini_chunks, WordEmbeddings())
        )
        answer = run_rag(QUESTION, model, retriever, Budget())
    assert isinstance(answer, Answer) and answer.valid is False
    assert calls
