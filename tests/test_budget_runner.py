import json
import logging
from dataclasses import asdict

import pytest
import tiktoken

from rag_ai_act.budget import EUR_PER_USD, PRICES_USD_PER_M, Budget, BudgetExceeded, count_tokens
from rag_ai_act.config import BudgetedEmbeddings, run_metadata
from rag_ai_act.runner import run_questions
from rag_ai_act.systems import Answer
from tests.fakes import WordEmbeddings


@pytest.mark.parametrize("kind", ["chat", "embed"])
def test_reservations_stop_before_overspend(kind):
    budget = Budget(warn_eur=0.00001, stop_eur=0.0001)
    with pytest.raises(BudgetExceeded):
        if kind == "chat":
            budget.reserve_chat(1_000_000, 600)
        else:
            budget.reserve_embed(1_000_000)
    assert budget.spent_eur == 0 and budget.reserved_eur == 0


def test_stop_includes_existing_spend_and_open_reservations():
    cost = PRICES_USD_PER_M["chat_in"] * EUR_PER_USD
    budget = Budget(warn_eur=10, stop_eur=cost * 2.5, spent_eur=cost)
    reservation = budget.reserve_chat(1_000_000, 0)
    assert reservation == pytest.approx(cost)
    with pytest.raises(BudgetExceeded):
        budget.reserve_chat(1_000_000, 0)
    assert budget.reserved_eur == pytest.approx(cost)


@pytest.mark.parametrize("kind", ["chat", "embed"])
def test_budget_warning_is_emitted_once(kind, caplog):
    budget = Budget(warn_eur=0.000001, stop_eur=10)
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            if kind == "chat":
                budget.reserve_chat(1000, 1000)
            else:
                budget.reserve_embed(1000)
            assert budget.warned is True
            assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1
    assert budget.warned is True
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1


@pytest.mark.parametrize("kind", ["chat", "embed"])
def test_commit_releases_only_its_reservation_and_charges_actual_usage(kind):
    budget = Budget(warn_eur=10, stop_eur=20)
    if kind == "chat":
        reservation = budget.reserve_chat(10000, 1000)
        tokens_out = 20
        expected = 100 * PRICES_USD_PER_M["chat_in"] + 20 * PRICES_USD_PER_M["chat_out"]
    else:
        reservation = budget.reserve_embed(10000)
        tokens_out = 0
        expected = 100 * PRICES_USD_PER_M["embed_in"]
    other = budget.reserve_embed(2000)
    expected *= EUR_PER_USD / 1_000_000
    actual = budget.commit(reservation, kind, tokens_in=100, tokens_out=tokens_out)
    assert actual == pytest.approx(expected)
    assert budget.spent_eur == pytest.approx(expected)
    assert budget.reserved_eur == pytest.approx(other)
    assert budget.log == [{"kind": kind, "tokens_in": 100, "tokens_out": tokens_out, "eur": actual}]


def test_token_count_uses_cl100k():
    text = "Electricity distribution: safety components and AI."
    assert count_tokens(text) == len(tiktoken.get_encoding("cl100k_base").encode(text))
    assert count_tokens("") == 0


def test_runner_preserves_completed_records_and_stops_every_system():
    items = [{"qid": f"q{n}", "question": f"Question {n}"} for n in range(1, 4)]
    calls = []
    answer = Answer("Response", ["AIA:Art6(2)"], False)

    def successful(question):
        calls.append(("S0", question))
        return answer

    def limited(question):
        calls.append(("S1", question))
        if question == "Question 2":
            raise BudgetExceeded("Budget exhausted")
        return answer

    def later(question):
        calls.append(("S2", question))
        return answer

    result = run_questions(items, {"S0": successful, "S1": limited, "S2": later}, Budget())
    assert result["stopped"] is True
    assert result["stopped_at"] == {"system": "S1", "qid": "q2"}
    assert [r["qid"] for r in result["answers"]["S0"]] == ["q1", "q2", "q3"]
    assert [r["qid"] for r in result["answers"]["S1"]] == ["q1"]
    assert result["answers"].get("S2", []) == []
    assert calls == [("S0", i["question"]) for i in items] + [("S1", "Question 1"), ("S1", "Question 2")]
    for system, records in result["answers"].items():
        for row in records:
            assert row["system"] == system and row["latency_s"] >= 0
            assert {k: row[k] for k in asdict(answer)} == asdict(answer)


def test_runner_completes_without_a_budget_stop():
    answer = Answer("Cannot answer", [], True)
    result = run_questions([{"qid": "q1", "question": "Question"}], {"S0": lambda _: answer}, Budget())
    assert result["stopped"] is False and result["stopped_at"] is None
    assert result["answers"]["S0"][0]["declined"] is True


def test_metadata_excludes_key_and_has_reproducibility_fields(monkeypatch):
    key = "dummy-test-key-do-not-use"
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", key)
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "test-chat")
    monkeypatch.setenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "test-embed")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "test-version")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://service.invalid/?sig=dummy-query-secret")
    metadata = run_metadata()
    assert key not in json.dumps(metadata)
    assert "dummy-query-secret" not in json.dumps(metadata)
    assert {
        "chat_deployment",
        "embed_deployment",
        "api_version",
        "packages",
        "python",
        "pdftotext",
        "git_commit",
        "started_at",
    } <= metadata.keys()
    assert metadata["chat_deployment"] == "test-chat"
    assert metadata["embed_deployment"] == "test-embed"
    assert metadata["api_version"] == "test-version"


@pytest.mark.parametrize("mode", ["documents", "query"])
def test_embedding_wrapper_accounts_for_each_input(mode):
    texts = ["Electricity safety", "Critical infrastructure"]
    budget = Budget(warn_eur=10, stop_eur=20)
    encoding = tiktoken.get_encoding("cl100k_base")
    inputs = texts if mode == "documents" else texts[:1]
    expected_tokens = sum(len(encoding.encode(text)) for text in inputs)
    expected = expected_tokens * PRICES_USD_PER_M["embed_in"] * EUR_PER_USD / 1_000_000
    calls = []

    class ObservedEmbeddings(WordEmbeddings):
        def check_reservation(self, actual_inputs):
            assert actual_inputs == inputs
            assert budget.reserved_eur == pytest.approx(expected)
            assert budget.spent_eur == pytest.approx(len(calls) * expected)
            calls.append(list(actual_inputs))

        def embed_documents(self, texts):
            self.check_reservation(texts)
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            self.check_reservation([text])
            return [1.0, 0.0]

    wrapper = BudgetedEmbeddings(ObservedEmbeddings(), budget)
    for n in range(1, 3):
        if mode == "documents":
            assert wrapper.embed_documents(texts) == [[1.0, 0.0], [1.0, 0.0]]
        else:
            assert wrapper.embed_query(texts[0]) == [1.0, 0.0]
        assert len(calls) == n
        assert budget.spent_eur == pytest.approx(n * expected)
        assert budget.reserved_eur == pytest.approx(0)
    assert sum(row["tokens_in"] for row in budget.log) == 2 * expected_tokens
    assert all(row["kind"] == "embed" for row in budget.log)


@pytest.mark.parametrize("mode", ["documents", "query"])
def test_embedding_wrapper_stops_before_service_call(mode):
    class ObservedEmbeddings(WordEmbeddings):
        def embed_documents(self, texts):
            raise AssertionError("Service must not be called after budget stop")

        def embed_query(self, text):
            raise AssertionError("Service must not be called after budget stop")

    wrapper = BudgetedEmbeddings(ObservedEmbeddings(), Budget(warn_eur=0, stop_eur=0))
    with pytest.raises(BudgetExceeded):
        if mode == "documents":
            wrapper.embed_documents(["Electricity"])
        else:
            wrapper.embed_query("Electricity")


@pytest.mark.parametrize("mode", ["documents", "query"])
def test_embedding_wrapper_propagates_inner_budget_error(mode):
    error = BudgetExceeded("Synthetic embedding budget stop")
    calls = []

    class StoppedEmbeddings(WordEmbeddings):
        def embed_documents(self, texts):
            calls.append(texts)
            raise error

        def embed_query(self, text):
            calls.append(text)
            raise error

    wrapper = BudgetedEmbeddings(StoppedEmbeddings(), Budget(warn_eur=10, stop_eur=20))
    with pytest.raises(BudgetExceeded) as exc:
        if mode == "documents":
            wrapper.embed_documents(["Electricity"])
        else:
            wrapper.embed_query("Electricity")
    assert exc.value is error and len(calls) == 1
