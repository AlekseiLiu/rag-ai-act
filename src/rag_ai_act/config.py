"""Azure clients from .env. The key is read from the environment only; never logged or printed.

.env is loaded with python-dotenv (override=False) only inside make_llm / make_embeddings, never at
import time and never in run_metadata.
.env keys: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION,
AZURE_OPENAI_CHAT_DEPLOYMENT, AZURE_OPENAI_EMBED_DEPLOYMENT.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from rag_ai_act.budget import Budget, count_tokens

# Azure answers 429 (rate limit) with a retry-after hint; the OpenAI client waits and retries this many times.
MAX_RETRIES = 10
# Inputs per embedding request, so one request stays well under the per-minute token limit.
EMBED_BATCH = 64

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGES = (
    "faiss-cpu",
    "langchain-core",
    "langchain-openai",
    "langchain-text-splitters",
    "langgraph",
    "langgraph-checkpoint",
    "langgraph-prebuilt",
    "numpy",
    "openai",
    "python-dotenv",
    "rank-bm25",
    "tiktoken",
)


def _load_env() -> None:
    """Load .env into the environment without overriding variables that are already set. The file is
    searched from the working directory upwards; if none is found there, the repository's .env is used."""
    from dotenv import find_dotenv, load_dotenv

    path = find_dotenv(usecwd=True)
    if not path and (_REPO_ROOT / ".env").is_file():
        path = str(_REPO_ROOT / ".env")
    if path:
        load_dotenv(path, override=False)


def _require(*names: str) -> dict[str, str]:
    """The named environment variables; a missing one raises RuntimeError naming the variable only."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise RuntimeError(f"missing environment variables (set them in .env): {', '.join(missing)}")
    return {n: os.environ[n] for n in names}


def make_llm(max_tokens: int = 600) -> BaseChatModel:
    """AzureChatOpenAI(temperature=0, max_tokens=max_tokens) from .env."""
    from langchain_openai import AzureChatOpenAI

    _load_env()
    env = _require("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION",
                   "AZURE_OPENAI_CHAT_DEPLOYMENT")
    return AzureChatOpenAI(
        azure_endpoint=env["AZURE_OPENAI_ENDPOINT"],
        api_key=env["AZURE_OPENAI_API_KEY"],
        api_version=env["AZURE_OPENAI_API_VERSION"],
        azure_deployment=env["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        temperature=0,
        max_tokens=max_tokens,
        timeout=120,
        max_retries=MAX_RETRIES,
    )


def make_embeddings() -> Embeddings:
    """AzureOpenAIEmbeddings from .env."""
    from langchain_openai import AzureOpenAIEmbeddings

    _load_env()
    env = _require("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION",
                   "AZURE_OPENAI_EMBED_DEPLOYMENT")
    return AzureOpenAIEmbeddings(
        azure_endpoint=env["AZURE_OPENAI_ENDPOINT"],
        api_key=env["AZURE_OPENAI_API_KEY"],
        api_version=env["AZURE_OPENAI_API_VERSION"],
        azure_deployment=env["AZURE_OPENAI_EMBED_DEPLOYMENT"],
        chunk_size=EMBED_BATCH,
        timeout=120,
        max_retries=MAX_RETRIES,
    )


class BudgetedEmbeddings(Embeddings):
    """Wraps any Embeddings: before each embed_documents / embed_query call, reserve the cost of the input
    tokens (budget.count_tokens) with budget.reserve_embed, then commit kind "embed". BudgetExceeded
    propagates. Used for index building and query embedding, so every embedding call is capped.

    The committed usage is the counted input tokens (the embedding service bills input tokens only, and the
    wrapped client does not report usage). A call that fails is still charged its input tokens, since the
    service may have billed it; then the error propagates."""

    def __init__(self, inner: Embeddings, budget: Budget):
        self.inner = inner
        self.budget = budget

    def _call(self, texts: list[str], run):
        tokens = sum(count_tokens(t) for t in texts)
        reservation = self.budget.reserve_embed(tokens)
        try:
            result = run()
        finally:
            self.budget.commit(reservation, "embed", tokens_in=tokens)
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = list(texts)
        return self._call(texts, lambda: self.inner.embed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._call([text], lambda: self.inner.embed_query(text))


def _command_output(args: list[str], cwd: Path | None = None, require_success: bool = True) -> str | None:
    """First non-empty output line of a local command (stdout, then stderr), or None if it cannot run."""
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=10, cwd=cwd, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if require_success and proc.returncode != 0:
        return None
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        if line.strip():
            return line.strip()
    return None


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def run_metadata() -> dict:
    """{"chat_deployment", "embed_deployment", "api_version", "packages": {name: version}, "python",
    "pdftotext", "git_commit", "started_at"}; never contains the key or the endpoint's secret parts.

    Deployments and API version come from the current environment (.env is not read here: call make_llm /
    make_embeddings first, as an evaluation run does). The endpoint is not recorded at all. Also records
    "git_dirty" (uncommitted changes to tracked files) next to the commit hash; values that cannot be
    determined are None. started_at is the current UTC time."""
    commit = dirty = None
    if (_REPO_ROOT / ".git").exists():
        commit = _command_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT)
        status = _command_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=_REPO_ROOT)
        if commit is not None:
            dirty = status is not None
    return {
        "chat_deployment": os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        "embed_deployment": os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT"),
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION"),
        "packages": {name: _package_version(name) for name in _PACKAGES},
        "python": sys.version.split()[0],
        "pdftotext": _command_output(["pdftotext", "-v"], require_success=False),
        "git_commit": commit,
        "git_dirty": dirty,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
