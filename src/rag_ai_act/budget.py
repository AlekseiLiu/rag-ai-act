"""Cost control: reserve the estimated cost before each API call; warn at 1 EUR, stop at 3 EUR."""

from __future__ import annotations

from dataclasses import dataclass, field

# List prices, USD per 1M tokens (Azure OpenAI global standard; check before the final run) and conversion.
PRICES_USD_PER_M = {
    "chat_in": 0.40,
    "chat_out": 1.60,
    "embed_in": 0.02,
}
EUR_PER_USD = 0.92
WARN_EUR = 1.0
STOP_EUR = 3.0


class BudgetExceeded(RuntimeError):
    """Raised by reserve_* when the reservation would pass the stop limit. Partial results must be kept."""


def count_tokens(text: str) -> int:
    """tiktoken cl100k_base token count."""
    raise NotImplementedError


@dataclass
class Budget:
    warn_eur: float = WARN_EUR
    stop_eur: float = STOP_EUR
    spent_eur: float = 0.0  # committed actual cost
    reserved_eur: float = 0.0  # open reservations
    warned: bool = False
    log: list[dict] = field(default_factory=list)  # one entry per call: kind, tokens_in, tokens_out, eur

    def reserve_chat(self, prompt_tokens: int, max_tokens: int) -> float:
        """Reserve chat cost; return the reservation (EUR). Raise BudgetExceeded if spent+reserved+this > stop.
        Emits one warning (logging.warning, sets warned) the first time spent+reserved INCLUDING this
        reservation passes warn_eur."""
        raise NotImplementedError

    def reserve_embed(self, input_tokens: int) -> float:
        """As reserve_chat, for an embedding call."""
        raise NotImplementedError

    def commit(self, reservation_eur: float, kind: str, tokens_in: int, tokens_out: int = 0) -> float:
        """kind is "chat" or "embed". Release the reservation, add the actual cost from the API's usage, append to log; return actual EUR."""
        raise NotImplementedError
