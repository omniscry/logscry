import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from logscry.cli import main
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
        self.assertEqual(main([str(SAMPLE)]), 1)

    def test_download_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            model = dest / DEFAULT_FILE
            model.write_bytes(b"gguf")
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


if __name__ == "__main__":
    unittest.main()
