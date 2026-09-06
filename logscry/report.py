"""Format a human-readable Summary + Findings report."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HEADING = re.compile(
    r"(?im)^\s*(?:(?:--+|#+)\s*)?(summary|findings)\s*:?\s*$"
)


@dataclass(frozen=True)
class Report:
    summary: str
    findings: str
    raw: str

    def render(
        self,
        *,
        logfile: Path | str,
        prompt: Path | str,
        model: Path | str,
    ) -> str:
        lines = [
            "logscry report",
            f"Logfile: {logfile}",
            f"Prompt: {prompt}",
            f"Model: {model}",
            "",
            "-- Summary",
            self.summary or "(no summary)",
            "",
            "-- Findings",
            self.findings or "(no findings)",
            "",
        ]
        return "\n".join(lines)


def parse_report(text: str) -> Report:
    """Pull Summary and Findings from model output; keep raw as fallback."""
    raw = text.strip()
    if not raw:
        return Report(summary="", findings="", raw="")

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in raw.splitlines():
        match = _HEADING.match(line)
        if match:
            current = match.group(1).lower()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)

    summary = "\n".join(sections.get("summary", [])).strip()
    findings = "\n".join(sections.get("findings", [])).strip()
    if not summary and not findings:
        summary = raw
    return Report(summary=summary, findings=findings, raw=raw)


def write_report(report_text: str, output: Path | None) -> None:
    if output is None:
        print(report_text, end="" if report_text.endswith("\n") else "\n")
        return
    output.write_text(report_text if report_text.endswith("\n") else report_text + "\n", encoding="utf-8")
