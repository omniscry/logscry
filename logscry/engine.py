"""In-process llama.cpp engine (CPU)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from logscry.config import PromptConfig


class EngineError(RuntimeError):
    """Model load or inference failure."""


@dataclass
class Engine:
    model_path: Path
    n_ctx: int
    n_threads: int | None = None
    n_gpu_layers: int = 0
    chat_format: str | None = None

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        if not self.model_path.is_file():
            raise EngineError(f"model not found: {self.model_path}")
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise EngineError(
                "llama-cpp-python is required. Install with: pip install llama-cpp-python"
            ) from exc

        kwargs: dict = {
            "model_path": str(self.model_path),
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "verbose": False,
        }
        if self.n_threads is not None:
            kwargs["n_threads"] = self.n_threads
        if self.chat_format:
            kwargs["chat_format"] = self.chat_format
        try:
            self._llm = Llama(**kwargs)
        except Exception as exc:
            raise EngineError(f"failed to load model: {exc}") from exc

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False))

    def complete(self, config: PromptConfig, user_message: str) -> str:
        try:
            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.num_predict,
            )
        except Exception as exc:
            raise EngineError(f"inference failed: {exc}") from exc
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EngineError("model returned an unexpected response") from exc
        return (content or "").strip()
