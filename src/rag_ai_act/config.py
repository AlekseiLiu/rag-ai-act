"""Azure clients from .env. The key is read from the environment only; never logged or printed.

.env is loaded with python-dotenv (override=False) only inside make_llm / make_embeddings, never at
import time and never in run_metadata.
.env keys: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION,
AZURE_OPENAI_CHAT_DEPLOYMENT, AZURE_OPENAI_EMBED_DEPLOYMENT.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from rag_ai_act.budget import Budget


def make_llm(max_tokens: int = 600) -> BaseChatModel:
    """AzureChatOpenAI(temperature=0, max_tokens=max_tokens) from .env."""
    raise NotImplementedError


def make_embeddings() -> Embeddings:
    """AzureOpenAIEmbeddings from .env."""
    raise NotImplementedError


class BudgetedEmbeddings(Embeddings):
    """Wraps any Embeddings: before each embed_documents / embed_query call, reserve the cost of the input
    tokens (budget.count_tokens) with budget.reserve_embed, then commit kind "embed". BudgetExceeded
    propagates. Used for index building and query embedding, so every embedding call is capped."""

    def __init__(self, inner: Embeddings, budget: Budget):
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


def run_metadata() -> dict:
    """{"chat_deployment", "embed_deployment", "api_version", "packages": {name: version}, "python",
    "pdftotext", "git_commit", "started_at"}; never contains the key or the endpoint's secret parts."""
    raise NotImplementedError
