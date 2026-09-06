import unittest
from unittest.mock import patch

from logscry.analyze import analyze, merge_partials, _pack_for_merge
from logscry.chunker import estimate_tokens
from logscry.config import PromptConfig


class FakeEngine:
    def __init__(self, n_ctx: int = 4096) -> None:
        self.n_ctx = n_ctx
        self.calls = 0
        self.prompts: list[str] = []

    def count_tokens(self, text: str) -> int:
        return max(len(text) // 8, 1) if text else 0

    def complete(self, config: PromptConfig, user_message: str) -> str:
        self.calls += 1
        self.prompts.append(user_message)
        return "-- Summary\nOK\n\n-- Findings\n1. none\n"


class AnalyzeTests(unittest.TestCase):
    def test_single_chunk(self) -> None:
        engine = FakeEngine()
        config = PromptConfig(system_prompt="You are a log analyzer.")
        text = analyze(engine, config, "line1\nline2\n", show_progress=False)
        self.assertEqual(engine.calls, 1)
        self.assertIn("-- Summary", text)

    def test_hierarchical_merge_when_partials_overflow(self) -> None:
        engine = FakeEngine(n_ctx=1024)
        config = PromptConfig(
            system_prompt="You are a log analyzer.",
            num_predict=128,
        )
        # Each partial is large enough that all five cannot fit in one merge call.
        partials = [
            f"-- Summary\npart {i}\n-- Findings\n" + ("issue line with details\n" * 40)
            for i in range(5)
        ]
        result = merge_partials(engine, config, partials, show_progress=False)
        self.assertIn("-- Summary", result)
        self.assertGreater(engine.calls, 1)

    def test_pack_for_merge_splits(self) -> None:
        partials = ["aaaa" * 20, "bbbb" * 20, "cccc" * 20]
        batches = _pack_for_merge(partials, lambda text: len(text), max_tokens=100)
        self.assertGreater(len(batches), 1)
        self.assertEqual(sum(len(batch) for batch in batches), 3)

    def test_multi_chunk_uses_merge(self) -> None:
        engine = FakeEngine()
        config = PromptConfig(system_prompt="You are a log analyzer.", num_predict=64)
        chunks = ["a\nb\n", "c\n", "d\ne\n"]
        with patch("logscry.analyze.chunk_log", return_value=chunks):
            text = analyze(engine, config, "a\nb\nc\nd\ne\n", show_progress=False)
        self.assertEqual(engine.calls, 4)  # 3 analyze + 1 merge
        self.assertIn("-- Summary", text)

    def test_chunk_log_uses_estimate_tokens(self) -> None:
        engine = FakeEngine()
        config = PromptConfig(system_prompt="You are a log analyzer.")
        with patch("logscry.analyze.chunk_log", return_value=["line1\n"]) as mocked:
            analyze(engine, config, "line1\n", show_progress=False)
        _text, count_tokens, _budget = mocked.call_args.args
        self.assertIs(count_tokens, estimate_tokens)


if __name__ == "__main__":
    unittest.main()
