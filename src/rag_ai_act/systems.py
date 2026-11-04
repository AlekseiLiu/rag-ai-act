"""The four systems S0..S3 and the answer contract.

Every system returns an Answer. JSONL record of an answer (results/<run>/answers_<system>.jsonl), one per line:
  {"qid", "system", "answer", "cited_ids", "declined", "valid", "retrieved_ids", "trace", "usage", "latency_s"}
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel

from rag_ai_act.budget import Budget
from rag_ai_act.index import Retriever
from rag_ai_act.tools import DateClauses, ProvisionStore

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


def answer_prompt(question: str, context: str | None, with_id_scheme: bool) -> str:
    """The shared answer prompt (S0..S2): rules (answer only from the context when given; cite source IDs;
    say when a cited source is a draft; decline with declined=true if the context does not contain the
    answer), the JSON contract, the context block (None for S0), and for S0 a description of the ID scheme."""
    raise NotImplementedError


def parse_answer(text: str) -> Answer:
    """Parse model output into an Answer. Accepts a JSON object, also inside ```json fences. Normalises
    cited_ids via ids.normalize_id where possible. A declined answer has cited_ids == []. Anything else
    (not JSON, missing keys, wrong types) -> Answer(answer=text, cited_ids=[], declined=False, valid=False)."""
    raise NotImplementedError


def run_s0(question: str, llm: BaseChatModel, budget: Budget) -> Answer:
    raise NotImplementedError


def run_rag(question: str, llm: BaseChatModel, retriever: Retriever, budget: Budget) -> Answer:
    """S1 (BM25) and S2 (dense): top-5 chunks as context, one call."""
    raise NotImplementedError


def run_agent(question: str, llm: BaseChatModel, retriever: Retriever, store: ProvisionStore,
              dates: DateClauses, budget: Budget, max_tool_calls: int = MAX_TOOL_CALLS) -> Answer:
    """S3: langgraph.prebuilt.create_react_agent with tools.make_tools(...). After max_tool_calls tool calls
    the agent must answer without further tools. Final message parsed with parse_answer; trace filled."""
    raise NotImplementedError
