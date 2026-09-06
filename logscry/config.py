"""Load JSON prompt/config files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
KNOWN_PROMPTS = ("generic", "syslog", "messages")

_ALLOWED_TOP_LEVEL = frozenset({"system_prompt", "options"})
_ALLOWED_OPTIONS = frozenset({"temperature", "top_p", "num_predict"})


class ConfigError(ValueError):
    """Invalid prompt config."""


@dataclass(frozen=True)
class PromptConfig:
    system_prompt: str
    temperature: float = 0.0
    top_p: float = 0.9
    num_predict: int = 512
    path: Path | None = None

    def with_overrides(
        self,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        num_predict: int | None = None,
    ) -> PromptConfig:
        return PromptConfig(
            system_prompt=self.system_prompt,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p,
            num_predict=self.num_predict if num_predict is None else num_predict,
            path=self.path,
        )


def resolve_prompt_path(name: str | None = None, config: str | None = None) -> Path:
    if config:
        path = Path(config)
        if not path.is_file():
            raise ConfigError(f"prompt file not found: {path}")
        return path
    prompt_name = name or "generic"
    if prompt_name not in KNOWN_PROMPTS:
        known = ", ".join(KNOWN_PROMPTS)
        raise ConfigError(f"unknown prompt {prompt_name!r}; choose one of: {known}")
    path = PROMPT_DIR / f"{prompt_name}.prompt"
    if not path.is_file():
        raise ConfigError(f"bundled prompt missing: {path}")
    return path


def load_prompt(path: Path) -> PromptConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: prompt file must be a JSON object")

    unknown = set(raw) - _ALLOWED_TOP_LEVEL
    if unknown:
        keys = ", ".join(sorted(unknown))
        raise ConfigError(f"{path}: unknown top-level key(s): {keys}")

    system_prompt = raw.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ConfigError(f"{path}: system_prompt must be a non-empty string")

    options = raw.get("options", {})
    if not isinstance(options, dict):
        raise ConfigError(f"{path}: options must be an object")
    unknown_opts = set(options) - _ALLOWED_OPTIONS
    if unknown_opts:
        keys = ", ".join(sorted(unknown_opts))
        raise ConfigError(f"{path}: unknown options key(s): {keys}")

    temperature = _as_float(options.get("temperature", 0.0), "temperature", path)
    top_p = _as_float(options.get("top_p", 0.9), "top_p", path)
    num_predict = _as_int(options.get("num_predict", 512), "num_predict", path)
    if num_predict < 1:
        raise ConfigError(f"{path}: num_predict must be >= 1")

    return PromptConfig(
        system_prompt=system_prompt.strip(),
        temperature=temperature,
        top_p=top_p,
        num_predict=num_predict,
        path=path,
    )


def _as_float(value: object, name: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: {name} must be a number")
    return float(value)


def _as_int(value: object, name: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}: {name} must be an integer")
    return value
