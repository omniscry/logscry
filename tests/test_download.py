import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from logscry.download import DEFAULT_FILE, DEFAULT_REPO, download_model, parse_model_name


class DownloadTests(unittest.TestCase):
    def test_parse_repo_only(self) -> None:
        self.assertEqual(parse_model_name(None), (DEFAULT_REPO, None))
        self.assertEqual(parse_model_name("Qwen/Qwen2.5-3B-Instruct-GGUF"), ("Qwen/Qwen2.5-3B-Instruct-GGUF", None))

    def test_parse_repo_and_file(self) -> None:
        self.assertEqual(
            parse_model_name("Qwen/Foo:bar.gguf"),
            ("Qwen/Foo", "bar.gguf"),
        )

    def test_parse_filename_only(self) -> None:
        self.assertEqual(parse_model_name("tiny.gguf"), (DEFAULT_REPO, "tiny.gguf"))

    def test_skips_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            existing = dest / DEFAULT_FILE
            existing.write_bytes(b"gguf")
            path = download_model(dest)
            self.assertEqual(path, existing.resolve())

    def test_downloads_named_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            fake = dest / "tiny.gguf"
            seen: dict[str, str] = {}

            def _fake_download(**kwargs: str) -> str:
                seen.update(kwargs)
                return str(fake)

            hub = SimpleNamespace(hf_hub_download=_fake_download)
            with patch.dict(sys.modules, {"huggingface_hub": hub}):
                path = download_model(dest, model_name="Qwen/Foo:tiny.gguf")
            self.assertEqual(seen["repo_id"], "Qwen/Foo")
            self.assertEqual(seen["filename"], "tiny.gguf")
            self.assertEqual(path, fake.resolve())
