import unittest

from logscry.report import parse_report


class ReportTests(unittest.TestCase):
    def test_dashed_headings(self) -> None:
        report = parse_report("-- Summary\nAll quiet.\n\n-- Findings\n1. None\n")
        self.assertEqual(report.summary, "All quiet.")
        self.assertEqual(report.findings, "1. None")

    def test_markdown_headings(self) -> None:
        report = parse_report("## Summary\nHost is unhealthy.\n## Findings\nOOM kill\n")
        self.assertEqual(report.summary, "Host is unhealthy.")
        self.assertEqual(report.findings, "OOM kill")

    def test_unstructured_fallback(self) -> None:
        report = parse_report("Something went wrong on the host.")
        self.assertEqual(report.summary, "Something went wrong on the host.")
        self.assertEqual(report.findings, "")

    def test_render(self) -> None:
        report = parse_report("-- Summary\nOK\n-- Findings\nNone\n")
        text = report.render(logfile="a.log", prompt="generic.prompt", model="m.gguf")
        self.assertIn("logscry report", text)
        self.assertIn("-- Summary", text)
        self.assertIn("OK", text)
        self.assertIn("-- Findings", text)


if __name__ == "__main__":
    unittest.main()
