import unittest

from logscry.chunker import chunk_log


def _chars(text: str) -> int:
    return len(text)


class ChunkerTests(unittest.TestCase):
    def test_fits_in_one_chunk(self) -> None:
        text = "one\ntwo\n"
        self.assertEqual(chunk_log(text, _chars, 100), [text])

    def test_splits_on_lines(self) -> None:
        text = "aaaa\nbbbb\ncccc\n"
        chunks = chunk_log(text, _chars, 6)
        self.assertEqual(chunks, ["aaaa\n", "bbbb\n", "cccc\n"])

    def test_keeps_oversized_line(self) -> None:
        text = "toolongline\nshort\n"
        chunks = chunk_log(text, _chars, 6)
        self.assertEqual(chunks[0], "toolongline\n")
        self.assertEqual(chunks[1], "short\n")

    def test_empty_text(self) -> None:
        self.assertEqual(chunk_log("", _chars, 10), [""])


if __name__ == "__main__":
    unittest.main()
