"""Split logs on line boundaries to fit a token budget."""

from __future__ import annotations

from collections.abc import Callable

CountTokens = Callable[[str], int]


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
