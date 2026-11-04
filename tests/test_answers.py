import json

import pytest

from rag_ai_act.systems import Answer, answer_prompt, parse_answer


@pytest.mark.parametrize("fenced", [False, True])
def test_json_and_fenced_json(fenced):
    data = {"answer": "It is high-risk.", "cited_ids": ["AIA:Art6(2)"], "declined": False}
    text = json.dumps(data)
    if fenced:
        text = f"```json\n{text}\n```"
    answer = parse_answer(text)
    assert isinstance(answer, Answer) and answer.valid
    assert answer.answer == data["answer"]
    assert answer.cited_ids == data["cited_ids"] and answer.declined is False


@pytest.mark.parametrize(
    "text",
    [
        "This is not JSON",
        "{",
        "[]",
        "null",
        "{}",
        '{"answer":"x","declined":false}',
        '{"answer":"x","cited_ids":[]}',
        '{"cited_ids":[],"declined":false}',
        '{"answer":3,"cited_ids":[],"declined":false}',
        '{"answer":"x","cited_ids":"AIA:Art6","declined":false}',
        '{"answer":"x","cited_ids":[7],"declined":false}',
        '{"answer":"x","cited_ids":[],"declined":"false"}',
    ],
)
def test_malformed_missing_and_wrong_typed_outputs_are_invalid(text):
    answer = parse_answer(text)
    assert answer.valid is False
    assert answer.answer == text
    assert answer.cited_ids == [] and answer.declined is False


def test_decline_clears_citations():
    answer = parse_answer(
        json.dumps({"answer": "Insufficient information.", "cited_ids": ["AIA:Art6(2)"], "declined": True})
    )
    assert answer.valid and answer.declined and answer.cited_ids == []


def test_loose_citations_normalized_and_unknown_retained():
    answer = parse_answer(
        json.dumps(
            {
                "answer": "Answer",
                "declined": False,
                "cited_ids": ["Article 6(2)(a)", "GLHR para 204", "unparseable source"],
            }
        )
    )
    assert answer.valid
    assert answer.cited_ids == ["AIA:Art6(2)", "GLHR:para204", "unparseable source"]


def test_prompt_includes_question_context_and_answer_contract():
    question = "Which systems are covered?"
    context = "[AIA:Art6(2)] Annex III systems are high-risk."
    prompt = answer_prompt(question, context, with_id_scheme=False)
    assert question in prompt and context in prompt
    assert all(key in prompt for key in ("answer", "cited_ids", "declined"))
    assert "draft" in prompt.lower() and "declin" in prompt.lower()
    no_context = answer_prompt(question, None, with_id_scheme=True)
    assert question in no_context and "AIA:" in no_context and "GLHR:" in no_context
