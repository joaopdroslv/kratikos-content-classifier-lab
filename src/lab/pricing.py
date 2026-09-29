"""OpenAI list prices, to turn the `usage` that `lab.llm` records into dollars.

US$ per 1M tokens, standard (non-batch) tier, from developers.openai.com/api/docs/models/<model>,
read on 2026-09-29. Reasoning tokens are billed as output (they are included in `output`).
"""

from collections.abc import Iterable

PRICES = {
    # model: (input, cached input, output)
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-6-luna": (0.10, 0.01, 0.50),
}


def cost(model: str, usages: Iterable[dict]) -> float:
    """Total US$ for the `usage` dicts of a model's answers."""

    price_in, price_cached, price_out = PRICES[model]
    total = 0.0
    for u in usages:
        total += (u["input"] - u["cached"]) * price_in
        total += u["cached"] * price_cached + u["output"] * price_out
    return total / 1_000_000
