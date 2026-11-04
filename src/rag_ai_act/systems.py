"""The four systems S0..S3 and the answer contract. In every run_* function BudgetExceeded propagates.

Every system returns an Answer. JSONL record of an answer (results/<run>/answers_<system>.jsonl), one per line:
  {"qid", "system", "answer", "cited_ids", "declined", "valid", "retrieved_ids", "trace", "usage", "latency_s"}

All four systems share one model, one set of answer rules (ANSWER_RULES) and the same chunk formatting:
  S0  no retrieval: the rules plus a description of the ID scheme, one model call;
  S1  BM25 top-5 chunks as context, one model call;
  S2  dense (FAISS) top-5 chunks as context, one model call;
  S3  a LangGraph ReAct agent with the tools search, get_provision and applicability_date.
Every chat call reserves its worst-case cost on the Budget before it is made and commits the API's reported
usage afterwards. `Answer.usage` covers every call made for the question: chat tokens in and out, and the
cost of all calls (chat and, when the retriever embeds through a budgeted wrapper, query embeddings).
"""

from __future__ import annotations

import json
import re
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool

from rag_ai_act.budget import Budget, BudgetExceeded, count_tokens
from rag_ai_act.ids import InvalidId, normalize_id
from rag_ai_act.index import Retriever
from rag_ai_act.ingest import Chunk
from rag_ai_act.tools import DateClauses, ProvisionStore, _chunk_as_unit, format_units, make_tools

SYSTEMS = ("S0", "S1", "S2", "S3")
MAX_TOOL_CALLS = 6
MAX_ANSWER_TOKENS = 600


@dataclass
class Answer:
    answer: str
    cited_ids: list[str]  # canonical where parseable; unparseable kept verbatim
    declined: bool
    valid: bool = True  # False if the model output broke the contract (counted as wrong, never a crash)
    retrieved_ids: list[str] = field(default_factory=list)  # S1/S2: source_ids of the k retrieved chunks in
    # rank order (k entries; two parts of one unit give the same ID twice; metrics use the set);
    # S3: deduplicated source_ids from all tool outputs in first-seen order; S0: []
    trace: list[dict] = field(default_factory=list)  # S3: [{"tool", "args", "returned_ids"}] per call
    usage: dict = field(default_factory=dict)  # {"tokens_in", "tokens_out", "eur"}


ANSWER_RULES = """\
You answer questions about the EU AI Act (Regulation (EU) 2024/1689, consolidated text) and the European \
Commission's guidelines on it.

Rules:
1. Answer only from the sources you are given. Each source starts with a line "[<source_id>] <header>". \
Do not add facts, dates or provisions that the sources do not state.
2. Put the IDs of the sources that support your answer in "cited_ids", written exactly as they appear \
between the square brackets (for example AIA:Art6(2), AIA:AnnexIII(2), GLHR:para204, GLDEF:para43). Cite \
every source your answer relies on and no other. If the question has several parts, cite the source for \
each part.
3. Sources marked "(DRAFT)" come from draft guidelines that are not final. When your answer relies on one, \
say in the answer text that this comes from draft guidelines. Where a draft guideline and the AI Act \
disagree, the AI Act prevails.
4. If the sources do not state the specific fact the question asks for, set "declined" to true, leave \
"cited_ids" empty, and say in "answer" that the sources do not answer the question; you may add what the \
sources do state. Do not guess.
5. Keep the answer short: a few sentences at most.
6. Reply with one JSON object and nothing else, in exactly this form:
{"answer": "<your answer>", "cited_ids": ["<source_id>", "..."], "declined": false}"""

_NO_SOURCES = """\
You are given no sources for this question: answer from your own knowledge of these documents instead. \
Rule 1 does not apply; all other rules do. Decline as in rule 4 when you are not sure of the answer.
Cite with IDs in this scheme:
- AI Act (consolidated text, prefix AIA): article paragraph AIA:Art6(2); whole article AIA:Art6; inserted \
articles and paragraphs keep their letter: AIA:Art4a, AIA:Art6(1a); Article 3 definition point 14: \
AIA:Art3(14); the unnumbered paragraphs of Article 113 are counted in order: AIA:Art113(3) is the third; \
annex point AIA:AnnexIII(2); Annex I sections AIA:AnnexI(A) and AIA:AnnexI(B).
- Commission guidelines on the classification of high-risk AI systems, a DRAFT (prefix GLHR): paragraph \
GLHR:para204; unnumbered example boxes by section number: GLHR:sec3.2.7.
- Commission guidelines on the definition of an AI system (prefix GLDEF): paragraph GLDEF:para43.
Points such as (a) are not part of an ID: cite AIA:Art6(2), not AIA:Art6(2)(a)."""

_TOOL_GUIDANCE = """\
Find the sources with your three tools; their results are the sources the rules refer to. Every result \
lists provisions as "[<source_id>] <header>" followed by the text, or says NOT_FOUND.
- search(query): the 5 passages that best match a free-text query, best first. A long provision may show \
only one part: fetch it with get_provision for the full text.
- get_provision(source_id): the full text of a provision by its ID, e.g. AIA:Art6(2); AIA:Art6 for a whole \
article; AIA:Art3(14) for a definition; AIA:Art113(3) for the third paragraph of Article 113; \
AIA:AnnexIII(2); AIA:AnnexI(A); GLHR:para204; GLHR:sec3.2.7; GLDEF:para43. Use it to read a provision \
that another text refers to.
- applicability_date(provision): the clauses of Article 113 and Article 111 that name a provision such as \
"Annex III", "Article 6(1)" or "Chapter III". Read the date in the clause and cite the clause's ID.
You may make at most {max_tool_calls} tool calls for one question; calls made together count one each, and \
calls beyond the limit are refused. Then give your final answer as the JSON object, without calling a tool."""

_REFUSED = (
    "REFUSED: the limit of {limit} tool calls for this question is reached; this call was not run. Answer "
    "now from the sources you have, as the JSON object only."
)


def agent_system_prompt() -> str:
    """S3 system prompt = ANSWER_RULES verbatim + tool guidance only (what each of the three tools returns,
    the ID formats they accept, the tool-call limit). No other rules, so S3 differs from S2 only in tools and
    control flow."""
    return ANSWER_RULES + "\n\n" + _TOOL_GUIDANCE.format(max_tool_calls=MAX_TOOL_CALLS)


def answer_prompt(question: str, context: str | None, with_id_scheme: bool) -> str:
    """The shared answer prompt (S0..S2), built on ANSWER_RULES: rules (answer only from the context when given; cite source IDs;
    say when a cited source is a draft; decline with declined=true if the context does not state the
    specific fact asked for), the JSON contract, the context block (None for S0), and for S0 a description of the ID scheme."""
    parts = [ANSWER_RULES]
    if with_id_scheme:
        parts.append(_NO_SOURCES)
    if context is not None:
        parts.append("Sources:\n\n" + context)
    parts.append("Question: " + question)
    return "\n\n".join(parts)


def format_chunks(chunks: list[Chunk]) -> str:
    """Context block for S1/S2: the chunks in rank order, formatted exactly like the agent's search tool
    output ("[<source_id>] <header>" or "[<source_id>] (DRAFT) <header>", then the text)."""
    return format_units([_chunk_as_unit(c) for c in chunks])


# ------------------------------------------------------------------------------------------- parsing

_FENCE = re.compile(r"^```[A-Za-z]*[ \t]*\n?(.*?)\n?```$", re.DOTALL)


def parse_answer(text: str) -> Answer:
    """Parse model output into an Answer. Accepts a JSON object, also inside ```json fences. Normalises
    cited_ids via ids.normalize_id where possible. A declined answer has cited_ids == []. Anything else
    (not JSON, missing keys, wrong types) -> Answer(answer=text, cited_ids=[], declined=False, valid=False).

    Repeated citations (also a loose and the canonical form of one ID) are kept once, in first-seen order."""
    if not isinstance(text, str):
        return Answer(answer=str(text), cited_ids=[], declined=False, valid=False)
    invalid = Answer(answer=text, cited_ids=[], declined=False, valid=False)
    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1).strip()
    try:
        data = json.loads(body)
    except (ValueError, RecursionError):
        return invalid
    if not isinstance(data, dict):
        return invalid
    answer, cited, declined = data.get("answer"), data.get("cited_ids"), data.get("declined")
    if not isinstance(answer, str) or not isinstance(declined, bool) or not isinstance(cited, list):
        return invalid
    if not all(isinstance(c, str) for c in cited):
        return invalid
    if declined:
        return Answer(answer=answer, cited_ids=[], declined=True)
    ids: list[str] = []
    for raw in cited:
        try:
            ids.append(normalize_id(raw))
        except InvalidId:
            ids.append(raw)
    return Answer(answer=answer, cited_ids=list(dict.fromkeys(ids)), declined=False)


# ------------------------------------------------------------------------------------ budgeted calls


def _text(message: BaseMessage) -> str:
    """The text of a message whose content is a string or a list of content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else str(block.get("text", ""))
        for block in content
        if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
    )


def _message_tokens(messages: list[BaseMessage]) -> int:
    """Token estimate of a chat prompt: content, tool-call arguments and a small per-message overhead."""
    total = 3
    for m in messages:
        total += 4 + count_tokens(_text(m))
        for call in getattr(m, "tool_calls", None) or []:
            total += count_tokens(call.get("name", "")) + count_tokens(json.dumps(call.get("args", {})))
    return total


def _max_tokens(llm: BaseChatModel) -> int:
    value = getattr(llm, "max_tokens", None)
    return value if isinstance(value, int) and value > 0 else MAX_ANSWER_TOKENS


def _reported_usage(message: Any) -> tuple[int, int] | None:
    """(input, output) tokens as reported by the API for one response, or None if not reported."""
    meta = getattr(message, "usage_metadata", None)
    if meta and "input_tokens" in meta and "output_tokens" in meta:
        return int(meta["input_tokens"]), int(meta["output_tokens"])
    usage = (getattr(message, "response_metadata", None) or {}).get("token_usage") or {}
    if "prompt_tokens" in usage and "completion_tokens" in usage:
        return int(usage["prompt_tokens"]), int(usage["completion_tokens"])
    return None


def _budgeted_invoke(runnable: Any, messages: list[BaseMessage], budget: Budget, max_tokens: int,
                     extra_prompt_tokens: int = 0) -> AIMessage:
    """One chat call under the budget: reserve (prompt estimate + max_tokens), call, commit the reported
    usage (the token estimate if the response reports none). A failed call is charged its prompt estimate,
    since the service may have billed it; then the error propagates."""
    prompt_tokens = _message_tokens(messages) + extra_prompt_tokens
    reservation = budget.reserve_chat(prompt_tokens, max_tokens)
    try:
        response = runnable.invoke(messages)
    except BaseException:
        budget.commit(reservation, "chat", tokens_in=prompt_tokens, tokens_out=0)
        raise
    usage = _reported_usage(response)
    if usage is None:
        out = count_tokens(_text(response)) + sum(
            count_tokens(json.dumps(c.get("args", {}))) for c in getattr(response, "tool_calls", None) or []
        )
        usage = (prompt_tokens, out)
    budget.commit(reservation, "chat", tokens_in=usage[0], tokens_out=usage[1])
    return response


def _usage_since(budget: Budget, start: int) -> dict:
    """Usage of the budget log entries from index `start`: chat tokens in and out, and the cost of all calls."""
    rows = budget.log[start:]
    chat = [r for r in rows if r["kind"] == "chat"]
    return {
        "tokens_in": sum(r["tokens_in"] for r in chat),
        "tokens_out": sum(r["tokens_out"] for r in chat),
        "eur": sum(r["eur"] for r in rows),
    }


def _error_answer(exc: BaseException, **fields: Any) -> Answer:
    """Invalid Answer for a failed call. The answer text is a fixed code with the exception's class name
    only ("error: RateLimitError"); the exception message is never recorded, since service errors can
    echo request details."""
    return Answer(answer=f"error: {type(exc).__name__}", cited_ids=[], declined=False, valid=False, **fields)


def _single_call(prompt: str, llm: BaseChatModel, budget: Budget, start: int,
                 retrieved_ids: list[str]) -> Answer:
    """One budgeted model call on `prompt`, parsed; a failed call gives an invalid Answer."""
    try:
        response = _budgeted_invoke(llm, [HumanMessage(content=prompt)], budget, _max_tokens(llm))
    except BudgetExceeded:
        raise
    except Exception as exc:  # noqa: BLE001  (any model failure is an invalid answer, never a crash)
        return _error_answer(exc, retrieved_ids=retrieved_ids, usage=_usage_since(budget, start))
    answer = parse_answer(_text(response))
    answer.retrieved_ids = retrieved_ids
    answer.usage = _usage_since(budget, start)
    return answer


# ------------------------------------------------------------------------------------------- systems


def run_s0(question: str, llm: BaseChatModel, budget: Budget) -> Answer:
    """S0: no retrieval. The shared answer prompt with the ID-scheme description and no sources, one call."""
    start = len(budget.log)
    return _single_call(answer_prompt(question, None, with_id_scheme=True), llm, budget, start, [])


def run_rag(question: str, llm: BaseChatModel, retriever: Retriever, budget: Budget) -> Answer:
    """S1 (BM25) and S2 (dense): top-5 chunks as context, one call."""
    start = len(budget.log)
    try:
        chunks = retriever.retrieve(question)
    except BudgetExceeded:
        raise
    except Exception as exc:  # noqa: BLE001  (e.g. a failed query embedding)
        return _error_answer(exc, usage=_usage_since(budget, start))
    prompt = answer_prompt(question, format_chunks(chunks), with_id_scheme=False)
    return _single_call(prompt, llm, budget, start, [c.source_id for c in chunks])


def run_agent(question: str, llm: BaseChatModel, retriever: Retriever, store: ProvisionStore,
              dates: DateClauses, budget: Budget, max_tool_calls: int = MAX_TOOL_CALLS) -> Answer:
    """S3: langgraph.prebuilt.create_react_agent with tools.make_tools(...). After max_tool_calls tool calls
    the agent must answer without further tools. Final message parsed with parse_answer; trace filled.
    Model or tool errors give an invalid Answer instead of a crash, EXCEPT BudgetExceeded, which always
    propagates (also out of LangChain callbacks: use raise_error=True or explicit wrappers).

    Control flow: tool calls are granted in the order the model requests them, also within one batched
    request, until max_tool_calls have been granted; every further call gets a refusal message and its tool
    is not run. Once the allowance is used up, the next model call is made with tool use disabled
    (tool_choice "none") and any tool call in that final response is dropped, so the run ends there. Tool
    exceptions are never shown to the model: they end the run with an invalid Answer (BudgetExceeded
    propagates). The trace lists the granted calls only, each with the source IDs it returned."""
    from langgraph.prebuilt import ToolNode, create_react_agent
    from langgraph.warnings import LangGraphDeprecatedSinceV10

    start = len(budget.log)
    tools = make_tools(retriever, store, dates)
    tool_tokens = sum(count_tokens(json.dumps(convert_to_openai_tool(t))) for t in tools)
    with_tools = llm.bind_tools(tools)
    without_tools = llm.bind_tools(tools, tool_choice="none")
    max_tokens = _max_tokens(llm)

    lock = threading.Lock()  # tool calls of one batched request may run in parallel threads
    trace: list[dict] = []
    granted: dict[str, dict] = {}

    def call_model(messages: list[BaseMessage]) -> AIMessage:
        final = len(trace) >= max_tool_calls
        response = _budgeted_invoke(without_tools if final else with_tools, messages, budget, max_tokens,
                                    extra_prompt_tokens=tool_tokens)
        if final and getattr(response, "tool_calls", None):
            response = AIMessage(content=response.content, id=response.id,
                                 usage_metadata=response.usage_metadata,
                                 response_metadata=response.response_metadata)
        with lock:
            for call in getattr(response, "tool_calls", None) or []:
                if len(trace) >= max_tool_calls:
                    break
                entry = {"tool": call["name"], "args": dict(call.get("args") or {}), "returned_ids": []}
                trace.append(entry)
                granted[str(call.get("id"))] = entry
        return response

    def select_model(state: Any, runtime: Any) -> RunnableLambda:
        return RunnableLambda(call_model)

    def guard_tool_call(request: Any, execute: Any) -> ToolMessage:
        call = request.tool_call
        with lock:
            entry = granted.pop(str(call.get("id")), None)
        if entry is None:
            return ToolMessage(content=_REFUSED.format(limit=max_tool_calls), name=call["name"],
                               tool_call_id=call["id"], artifact=[])
        message = execute(request)
        artifact = getattr(message, "artifact", None)
        entry["returned_ids"] = list(artifact) if isinstance(artifact, list) else []
        return message

    with warnings.catch_warnings():  # the prebuilt agent is deprecated in favour of a package not used here
        warnings.simplefilter("ignore", LangGraphDeprecatedSinceV10)
        agent = create_react_agent(
            select_model,
            ToolNode(tools, handle_tool_errors=False, wrap_tool_call=guard_tool_call),
            prompt=SystemMessage(content=agent_system_prompt()),
        )

    def retrieved() -> list[str]:
        return list(dict.fromkeys(sid for entry in trace for sid in entry["returned_ids"]))

    try:
        state = agent.invoke(
            {"messages": [HumanMessage(content="Question: " + question)]},
            config={"recursion_limit": 2 * max_tool_calls + 6},
        )
    except BudgetExceeded:
        raise
    except Exception as exc:  # noqa: BLE001  (model or tool failure, runaway loop: invalid, never a crash)
        return _error_answer(exc, retrieved_ids=retrieved(), trace=trace, usage=_usage_since(budget, start))
    answer = parse_answer(_text(state["messages"][-1]))
    answer.retrieved_ids = retrieved()
    answer.trace = trace
    answer.usage = _usage_since(budget, start)
    return answer
