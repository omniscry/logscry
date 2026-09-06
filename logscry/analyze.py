"""Analyze a log with the local model, chunking when needed."""

from __future__ import annotations

from logscry.chunker import chunk_log
from logscry.config import PromptConfig
from logscry.engine import Engine
from logscry.progress import ProgressBar, count_lines, status

TOKEN_MARGIN = 64

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
    budget = _chunk_budget(engine, config)
    chunks = chunk_log(log_text, engine.count_tokens, budget)
    total_chunks = len(chunks)
    merge_needed = total_chunks > 1
    total_steps = total_chunks + (1 if merge_needed else 0)
    progress = ProgressBar(
        total_lines=total_lines,
        total_steps=total_steps,
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

    merge_config = PromptConfig(
        system_prompt=MERGE_SYSTEM_PROMPT,
        temperature=config.temperature,
        top_p=config.top_p,
        num_predict=config.num_predict,
        path=config.path,
    )
    merged_input = "\n\n".join(
        f"--- Chunk {index} of {total_chunks} ---\n{text}"
        for index, text in enumerate(partials, start=1)
    )
    progress.update(
        completed_steps=total_chunks,
        completed_lines=total_lines,
        label="Merging report",
    )
    result = engine.complete(merge_config, merged_input)
    progress.finish()
    return result


def _chunk_budget(engine: Engine, config: PromptConfig) -> int:
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
