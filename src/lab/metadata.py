"""The minimum per-row metadata every experiment output carries — see README § Metadata standard.

- LLM answers: `model`, `prompt_version`, `created_at` (plus `taxonomy_version` when the prompt
  embeds a taxonomy). `lab.llm.run` writes `model` and `created_at`; the caller passes the rest
  through `extra`.
- Rows read from the live database or Qdrant: `sampled_at`, the moment of the read.
"""

from datetime import datetime, timezone


def now() -> str:
    """UTC, ISO 8601, to the second: the format of every `created_at` / `sampled_at`."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
