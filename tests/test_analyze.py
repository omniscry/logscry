import unittest

from logscry.analyze import analyze
from logscry.config import PromptConfig


class FakeEngine:
    def __init__(self) -> None:
        self.n_ctx = 4096
        self.calls = 0

    def count_tokens(self, text: str) -> int:
        return max(len(text) // 8, 1) if text else 0

    def complete(self, config: PromptConfig, user_message: str) -> str:
        self.calls += 1
        return "-- Summary\nOK\n\n-- Findings\n1. none\n"


class AnalyzeTests(unittest.TestCase):
    def test_single_chunk(self) -> None:
        engine = FakeEngine()
        config = PromptConfig(system_prompt="You are a log analyzer.")
        text = analyze(engine, config, "line1\nline2\n", show_progress=False)
        self.assertEqual(engine.calls, 1)
        self.assertIn("-- Summary", text)


if __name__ == "__main__":
    unittest.main()
