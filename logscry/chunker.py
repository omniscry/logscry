"""Split logs on line boundaries to fit a token budget."""

from __future__ import annotations

from collections.abc import Callable

CountTokens = Callable[[str], int]

# Conservative: overestimate tokens so packed chunks stay under n_ctx.
CHARS_PER_TOKEN = 2


def estimate_tokens(text: str) -> int:
    """Estimate tokens from character count. Empty text is 0 tokens."""
    if not text:
        return 0
    return max((len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN, 1)


def chunk_log(
    text: str,
    count_tokens: CountTokens,
    max_tokens: int,
) -> list[str]:
    """Split *text* into line-aware chunks that stay under *max_tokens*.

    A single line that exceeds the budget is kept as its own chunk.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")

    lines = text.splitlines(keepends=True)
    if not lines:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = max(count_tokens(line), 1)
        if current and current_tokens + line_tokens > max_tokens:
            chunks.append("".join(current))
            current = [line]
            current_tokens = line_tokens
        else:
            current.append(line)
            current_tokens += line_tokens

    if current:
        chunks.append("".join(current))
    return chunks
