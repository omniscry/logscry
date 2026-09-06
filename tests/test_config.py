import json
import tempfile
import unittest
from pathlib import Path

from logscry.config import ConfigError, load_prompt, resolve_prompt_path


class ConfigTests(unittest.TestCase):
    def test_load_bundled_prompts(self) -> None:
        for name in ("generic", "syslog", "messages"):
            path = resolve_prompt_path(name=name)
            config = load_prompt(path)
            self.assertTrue(config.system_prompt)
            self.assertEqual(config.num_predict, 512)

    def test_unknown_prompt_name(self) -> None:
        with self.assertRaises(ConfigError):
            resolve_prompt_path(name="nope")

    def test_missing_file(self) -> None:
        with self.assertRaises(ConfigError):
            resolve_prompt_path(config="does-not-exist.prompt")

    def test_unknown_top_level_key(self) -> None:
        path = self._write({"system_prompt": "ok", "extra": 1})
        with self.assertRaises(ConfigError):
            load_prompt(path)

    def test_missing_system_prompt(self) -> None:
        path = self._write({"options": {"temperature": 0.1}})
        with self.assertRaises(ConfigError):
            load_prompt(path)

    def test_option_fallbacks(self) -> None:
        path = self._write({"system_prompt": "You analyze logs."})
        config = load_prompt(path)
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.num_predict, 512)

    def test_overrides(self) -> None:
        path = self._write(
            {
                "system_prompt": "You analyze logs.",
                "options": {"temperature": 0.2, "top_p": 0.8, "num_predict": 256},
            }
        )
        config = load_prompt(path).with_overrides(temperature=0.5, num_predict=64)
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.top_p, 0.8)
        self.assertEqual(config.num_predict, 64)

    def _write(self, payload: dict) -> Path:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".prompt", delete=False, encoding="utf-8"
        )
        with handle:
            json.dump(payload, handle)
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return Path(handle.name)


if __name__ == "__main__":
    unittest.main()
