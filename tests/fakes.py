"""Deterministic local substitutes for model and embedding services."""

import hashlib
import math
import re
from itertools import chain, repeat
from typing import Any

import pytest
import tiktoken
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pydantic import Field


class WordEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * 128
        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:2]) % len(vector)] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class ScriptedChat(GenericFakeChatModel):
    seen: list[list[Any]] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        if len(self.seen) > 40:
            pytest.fail("Agent failed to terminate")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def response(text):
    return AIMessage(
        content=text, usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    )


def chat(messages):
    encoding = tiktoken.get_encoding("cl100k_base")
    script = list(messages)
    if not script:
        raise ValueError("A chat script needs at least one message")
    return ScriptedChat(messages=chain(script, repeat(script[-1])), custom_get_token_ids=encoding.encode)
