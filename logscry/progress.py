"""Simple stderr progress display for log analysis."""

from __future__ import annotations

import sys
from dataclasses import dataclass


def status(message: str, *, enabled: bool = True) -> None:
    if not enabled:
        return
    sys.stderr.write(f"logscry: {message}\n")
    sys.stderr.flush()


@dataclass
class ProgressBar:
    total_lines: int
    total_steps: int
    enabled: bool = True

    completed_lines: int = 0
    completed_steps: int = 0
    label: str = "Starting"

    def start(self) -> None:
        self._render()

    def update(
        self,
        *,
        completed_steps: int | None = None,
        completed_lines: int | None = None,
        label: str | None = None,
    ) -> None:
        if label is not None:
            self.label = label
        if completed_steps is not None:
            self.completed_steps = min(max(completed_steps, 0), self.total_steps)
        if completed_lines is not None:
            self.completed_lines = min(max(completed_lines, 0), self.total_lines)
        self._render()

    def finish(self) -> None:
        if not self.enabled:
            return
        self.completed_lines = self.total_lines
        self.completed_steps = self.total_steps
        self.label = "Done"
        self._render(final=True)

    def _render(self, *, final: bool = False) -> None:
        if not self.enabled:
            return
        width = 30
        ratio = self.completed_steps / max(self.total_steps, 1)
        filled = min(width, int(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        message = (
            f"\r{self.label}: |{bar}| "
            f"chunks {self.completed_steps}/{self.total_steps} "
            f"lines {self.completed_lines}/{self.total_lines}"
        )
        # Pad to clear leftover characters from longer previous labels.
        message = message.ljust(100)
        sys.stderr.write(message if not final else message.rstrip() + "\n")
        sys.stderr.flush()


def count_lines(text: str) -> int:
    if not text:
        return 0
    return len(text.splitlines())
