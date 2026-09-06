"""Analyze a log with the local model, chunking when needed."""

from __future__ import annotations

from logscry.chunker import chunk_log, estimate_tokens
from logscry.config import PromptConfig
from logscry.engine import Engine
from logscry.progress import ProgressBar, count_lines, status

TOKEN_MARGIN = 384  # chat-template / role overhead beyond raw token counts

MERGE_SYSTEM_PROMPT = (
    "You are a log analyzer. Combine the partial analyses below into one "
    "human-readable report with exactly two sections:\n\n"
    "-- Summary\n"
    "A short overall overview.\n\n"
    "-- Findings\n"
    "A numbered list of notable issues. Deduplicate repeats and rank by severity. "
    "Do not invent events that are not in the partial analyses."
)


def analyze(
    engine: Engine,
    config: PromptConfig,
    log_text: str,
    *,
    show_progress: bool = True,
) -> str:
    total_lines = count_lines(log_text)
    status("preparing log chunks...", enabled=show_progress)
    budget = _prompt_budget(engine, config)
    chunks = chunk_log(log_text, estimate_tokens, budget)
    total_chunks = len(chunks)
    merge_needed = total_chunks > 1
    progress = ProgressBar(
        total_lines=total_lines,
        total_steps=total_chunks + (1 if merge_needed else 0),
        enabled=show_progress,
    )
    progress.start()

    done_lines = 0
    if len(chunks) == 1:
        progress.update(
            completed_steps=0,
            completed_lines=0,
            label=f"Analyzing chunk 1/{total_chunks}",
        )
        result = engine.complete(config, _user_message(chunks[0]))
        progress.finish()
        return result

    partials = []
    for index, chunk in enumerate(chunks, start=1):
        progress.update(
            completed_steps=index - 1,
            completed_lines=done_lines,
            label=f"Analyzing chunk {index}/{total_chunks}",
        )
        user = _user_message(chunk, index=index, total=total_chunks)
        partials.append(engine.complete(config, user))
        done_lines += count_lines(chunk)
        progress.update(
            completed_steps=index,
            completed_lines=done_lines,
            label=f"Analyzing chunk {index}/{total_chunks}",
        )

    progress.update(
        completed_steps=total_chunks,
        completed_lines=total_lines,
        label="Merging report",
    )
    result = merge_partials(
        engine,
        config,
        partials,
        progress=progress,
        show_progress=show_progress,
    )
    progress.finish()
    return result


def merge_partials(
    engine: Engine,
    config: PromptConfig,
    partials: list[str],
    *,
    progress: ProgressBar | None = None,
    show_progress: bool = True,
) -> str:
    """Hierarchically merge partial reports so each call fits in context."""
    if not partials:
        return ""
    if len(partials) == 1:
        return partials[0]

    merge_config = PromptConfig(
        system_prompt=MERGE_SYSTEM_PROMPT,
        temperature=config.temperature,
        top_p=config.top_p,
        num_predict=config.num_predict,
        path=config.path,
    )
    budget = _prompt_budget(engine, merge_config)
    current = list(partials)
    round_num = 1

    while len(current) > 1:
        # Leave headroom for chat-template tokens that count_tokens does not see.
        pack_budget = max(int(budget * 0.85), 256)
        batches = _pack_for_merge(current, engine.count_tokens, pack_budget)
        next_round: list[str] = []
        total_batches = len(batches)
        for batch_index, batch in enumerate(batches, start=1):
            if progress is not None:
                progress.update(
                    label=f"Merging round {round_num} ({batch_index}/{total_batches})",
                )
            else:
                status(
                    f"merging round {round_num} batch {batch_index}/{total_batches}",
                    enabled=show_progress,
                )
            prompt = "\n\n".join(
                f"--- Partial {index} of {len(batch)} ---\n{text}"
                for index, text in enumerate(batch, start=1)
            )
            prompt = _fit_text(prompt, engine.count_tokens, pack_budget)
            next_round.append(engine.complete(merge_config, prompt))
        if len(next_round) >= len(current):
            # Pathological case: packing could not shrink; force pairwise merge.
            left = _fit_text(current[0], engine.count_tokens, pack_budget // 2)
            right = _fit_text(current[1], engine.count_tokens, pack_budget // 2)
            prompt = (
                f"--- Partial 1 of 2 ---\n{left}\n\n"
                f"--- Partial 2 of 2 ---\n{right}"
            )
            merged = engine.complete(merge_config, _fit_text(prompt, engine.count_tokens, pack_budget))
            current = [merged, *current[2:]]
        else:
            current = next_round
        round_num += 1

    return current[0]


def _pack_for_merge(
    partials: list[str],
    count_tokens,
    max_tokens: int,
) -> list[list[str]]:
    """Group partials into batches that fit under *max_tokens*."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")

    batches: list[list[str]] = []
    current: list[str] = []

    for text in partials:
        body = _fit_text(text, count_tokens, max(max_tokens - 48, 32))
        candidate = current + [body]
        prompt = "\n\n".join(
            f"--- Partial {index} of {len(candidate)} ---\n{item}"
            for index, item in enumerate(candidate, start=1)
        )
        if current and count_tokens(prompt) > max_tokens:
            batches.append(current)
            current = [body]
        else:
            current = candidate

    if current:
        batches.append(current)
    return batches


def _fit_text(text: str, count_tokens, max_tokens: int) -> str:
    """Truncate *text* so its token count stays within *max_tokens*."""
    if count_tokens(text) <= max_tokens:
        return text

    low, high = 0, len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid] + "\n...(truncated)..."
        if count_tokens(candidate) <= max_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best or text[: max(max_tokens, 1)]


def _prompt_budget(engine: Engine, config: PromptConfig) -> int:
    used = engine.count_tokens(config.system_prompt) + config.num_predict + TOKEN_MARGIN
    budget = engine.n_ctx - used
    if budget < 256:
        raise ValueError(
            f"context window too small for analysis "
            f"(n_ctx={engine.n_ctx}, reserved={used})"
        )
    return budget


def _user_message(chunk: str, index: int | None = None, total: int | None = None) -> str:
    if index is not None and total is not None:
        header = f"Analyze this log excerpt (chunk {index} of {total}):"
    else:
        header = "Analyze this log:"
    return f"{header}\n\n{chunk}"
