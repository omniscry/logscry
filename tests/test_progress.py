import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from logscry.analyze import analyze
from logscry.config import PromptConfig
from logscry.progress import ProgressBar, count_lines, status


class ProgressTests(unittest.TestCase):
    def test_count_lines(self) -> None:
        self.assertEqual(count_lines("a\nb\n"), 2)
        self.assertEqual(count_lines(""), 0)

    def test_status(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            status("loading model ...")
        self.assertIn("logscry: loading model ...", stderr.getvalue())

    def test_progress_bar_renders(self) -> None:
        bar = ProgressBar(total_lines=10, total_steps=2, enabled=True)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            bar.start()
            bar.update(completed_steps=0, completed_lines=0, label="Analyzing chunk 1/2")
            bar.update(completed_steps=1, completed_lines=5, label="Analyzing chunk 1/2")
            bar.finish()
        output = stderr.getvalue()
        self.assertIn("chunks 0/2", output)
        self.assertIn("chunks 1/2", output)
        self.assertIn("lines 5/10", output)
        self.assertIn("Analyzing chunk 1/2", output)

    def test_progress_disabled(self) -> None:
        bar = ProgressBar(total_lines=10, total_steps=2, enabled=False)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            status("loading", enabled=False)
            bar.start()
            bar.update(completed_steps=1, completed_lines=5)
            bar.finish()
        self.assertEqual(stderr.getvalue(), "")


class FakeEngine:
    def __init__(self, n_ctx: int = 4096) -> None:
        self.n_ctx = n_ctx
        self.calls = 0

    def count_tokens(self, text: str) -> int:
        return max(len(text), 1)

    def complete(self, config: PromptConfig, user_message: str) -> str:
        self.calls += 1
        return "-- Summary\nOK\n\n-- Findings\n1. none\n"


class AnalyzeProgressTests(unittest.TestCase):
    def test_multi_chunk_progress(self) -> None:
        engine = FakeEngine()
        config = PromptConfig(system_prompt="You are a log analyzer.")
        log_text = "a\nb\nc\nd\ne\n"
        chunks = ["a\nb\n", "c\n", "d\ne\n"]
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with patch("logscry.analyze.chunk_log", return_value=chunks):
                analyze(engine, config, log_text, show_progress=True)
        output = stderr.getvalue()
        self.assertIn("preparing log chunks", output)
        self.assertIn("Analyzing chunk 1/3", output)
        self.assertIn("lines 5/5", output)
        self.assertIn("chunks 4/4", output)
        self.assertIn("Merging report", output)

    def test_single_chunk_shows_progress_before_inference(self) -> None:
        engine = FakeEngine()
        config = PromptConfig(system_prompt="You are a log analyzer.")
        seen: list[str] = []

        def slow_complete(config: PromptConfig, user_message: str) -> str:
            # Capture stderr mid-inference to prove progress already rendered.
            seen.append(sys_stderr_snapshot[0].getvalue())
            return "-- Summary\nOK\n\n-- Findings\n1. none\n"

        sys_stderr_snapshot: list[io.StringIO] = []
        stderr = io.StringIO()
        sys_stderr_snapshot.append(stderr)
        engine.complete = slow_complete  # type: ignore[method-assign]
        with redirect_stderr(stderr):
            analyze(engine, config, "line1\n", show_progress=True)
        self.assertTrue(any("Analyzing chunk 1/1" in text for text in seen))


if __name__ == "__main__":
    unittest.main()
