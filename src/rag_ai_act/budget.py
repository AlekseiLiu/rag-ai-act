"""Cost control: reserve the estimated cost before each API call; warn at 1 EUR, stop at 3 EUR."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

# List prices, USD per 1M tokens (Azure OpenAI global standard; check before the final run) and conversion.
PRICES_USD_PER_M = {
    "chat_in": 0.40,
    "chat_out": 1.60,
    "embed_in": 0.02,
}
EUR_PER_USD = 0.92
WARN_EUR = 1.0
STOP_EUR = 3.0
_EUR_PER_TOKEN = EUR_PER_USD / 1_000_000


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


class BudgetExceeded(RuntimeError):
    """Raised by reserve_* when the reservation would pass the stop limit. Partial results must be kept."""


def count_tokens(text: str) -> int:
    """tiktoken cl100k_base token count."""
    return len(_encoding().encode(text))


@dataclass
class Budget:
    warn_eur: float = WARN_EUR
    stop_eur: float = STOP_EUR
    spent_eur: float = 0.0  # committed actual cost
    reserved_eur: float = 0.0  # open reservations
    warned: bool = False
    log: list[dict] = field(default_factory=list)  # one entry per call: kind, tokens_in, tokens_out, eur

    def _reserve(self, cost: float) -> float:
        total = self.spent_eur + self.reserved_eur + cost
        if total > self.stop_eur:
            raise BudgetExceeded(
                f"reservation of {cost:.4f} EUR would bring spend to {total:.4f} EUR, past the {self.stop_eur} EUR limit"
            )
        self.reserved_eur += cost
        if not self.warned and total > self.warn_eur:
            self.warned = True
            logging.getLogger(__name__).warning(
                "budget warning: %.4f EUR spent or reserved (including this reservation), warn limit %s EUR",
                total,
                self.warn_eur,
            )
        return cost

    def reserve_chat(self, prompt_tokens: int, max_tokens: int) -> float:
        """Reserve chat cost; return the reservation (EUR). Raise BudgetExceeded if spent+reserved+this > stop.
        Emits one warning (logging.warning, sets warned) the first time spent+reserved INCLUDING this
        reservation passes warn_eur."""
        cost = (prompt_tokens * PRICES_USD_PER_M["chat_in"] + max_tokens * PRICES_USD_PER_M["chat_out"]) * _EUR_PER_TOKEN
        return self._reserve(cost)

    def reserve_embed(self, input_tokens: int) -> float:
        """As reserve_chat, for an embedding call."""
        return self._reserve(input_tokens * PRICES_USD_PER_M["embed_in"] * _EUR_PER_TOKEN)

    def commit(self, reservation_eur: float, kind: str, tokens_in: int, tokens_out: int = 0) -> float:
        """kind is "chat" or "embed". Release the reservation, add the actual cost from the API's usage, append to log; return actual EUR."""
        if kind == "chat":
            actual = (tokens_in * PRICES_USD_PER_M["chat_in"] + tokens_out * PRICES_USD_PER_M["chat_out"]) * _EUR_PER_TOKEN
        elif kind == "embed":
            actual = tokens_in * PRICES_USD_PER_M["embed_in"] * _EUR_PER_TOKEN
        else:
            raise ValueError(f"unknown call kind {kind!r}")
        self.reserved_eur = max(0.0, self.reserved_eur - reservation_eur)
        self.spent_eur += actual
        self.log.append({"kind": kind, "tokens_in": tokens_in, "tokens_out": tokens_out, "eur": actual})
        return actual
