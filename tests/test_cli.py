import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from logscry.cli import LogscryError, main, read_log_text
from logscry.download import DEFAULT_FILE

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "sample.syslog"


class CliTests(unittest.TestCase):
    def test_help_exits_zero(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            main(["--help"])
        self.assertEqual(caught.exception.code, 0)

    def test_missing_logfile(self) -> None:
        self.assertEqual(main(["missing.log", "--model", "missing.gguf"]), 1)

    def test_missing_model(self) -> None:
        with patch("logscry.cli.stdin_available", return_value=False):
            self.assertEqual(main([str(SAMPLE)]), 1)

    def test_tty_without_logfile_errors(self) -> None:
        with patch("logscry.cli.stdin_available", return_value=False):
            self.assertEqual(main(["--model", "missing.gguf"]), 1)

    def test_download_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            model = dest / DEFAULT_FILE
            model.write_bytes(b"gguf")
            with patch("logscry.cli.stdin_available", return_value=False):
                with patch("logscry.cli.download_model", return_value=model) as mocked:
                    self.assertEqual(
                        main(
                            [
                                "--download",
                                "--models-dir",
                                tmp,
                                "--model-name",
                                "Qwen/Qwen2.5-3B-Instruct-GGUF",
                            ]
                        ),
                        0,
                    )
                mocked.assert_called_once_with(tmp, "Qwen/Qwen2.5-3B-Instruct-GGUF")

    def test_read_log_text_from_dash(self) -> None:
        fake_stdin = io.StringIO("Sep 6 host sshd: hello\n")
        with patch("logscry.cli.sys.stdin", fake_stdin):
            text, label = read_log_text("-")
        self.assertIn("sshd", text)
        self.assertEqual(label, "stdin")

    def test_read_log_text_from_pipe_when_omitted(self) -> None:
        fake_stdin = io.StringIO("line one\nline two\n")
        with patch("logscry.cli.stdin_available", return_value=True):
            with patch("logscry.cli.sys.stdin", fake_stdin):
                text, label = read_log_text(None)
        self.assertEqual(text, "line one\nline two\n")
        self.assertEqual(label, "stdin")

    def test_empty_stdin_errors(self) -> None:
        fake_stdin = io.StringIO("   \n")
        with patch("logscry.cli.sys.stdin", fake_stdin):
            with self.assertRaises(LogscryError) as caught:
                read_log_text("-")
        self.assertIn("stdin is empty", str(caught.exception))

    def test_empty_stdin_via_main(self) -> None:
        fake_stdin = io.StringIO("")
        with patch("logscry.cli.stdin_available", return_value=True):
            with patch("logscry.cli.sys.stdin", fake_stdin):
                self.assertEqual(main(["--model", "dummy.gguf"]), 1)

    def test_piped_input_runs_analyze(self) -> None:
        fake_stdin = io.StringIO("Sep 6 host kernel: oom\n")
        with patch("logscry.cli.stdin_available", return_value=True):
            with patch("logscry.cli.sys.stdin", fake_stdin):
                with patch("logscry.cli.run", return_value="report\n") as mocked_run:
                    code = main(["--model", "dummy.gguf", "--prompt", "syslog"])
        self.assertEqual(code, 0)
        mocked_run.assert_called_once()
        args = mocked_run.call_args.args[0]
        self.assertIsNone(args.logfile)


if __name__ == "__main__":
    unittest.main()
