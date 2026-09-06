"""logscry command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from logscry.analyze import analyze
from logscry.config import ConfigError, load_prompt, resolve_prompt_path
from logscry.download import DownloadError, download_model
from logscry.engine import Engine, EngineError
from logscry.progress import status
from logscry.report import parse_report, write_report


class LogscryError(Exception):
    """CLI-level failure."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="logscry",
        description="Analyze a log file or stdin with a local GGUF model.",
    )
    parser.add_argument(
        "logfile",
        nargs="?",
        help="path to the log file, or - for stdin",
    )
    parser.add_argument(
        "--model",
        help="path to a GGUF model file (required unless --download is used)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download a GGUF model into --models-dir",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-7B-Instruct-GGUF",
        help=(
            "Hugging Face model for --download: org/repo, "
            "org/repo:file.gguf, or file.gguf "
            "(default: Qwen/Qwen2.5-7B-Instruct-GGUF)"
        ),
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="directory for --download (default: models)",
    )
    parser.add_argument(
        "--prompt",
        default="generic",
        choices=["generic", "syslog", "messages"],
        help="bundled prompt config (default: generic)",
    )
    parser.add_argument(
        "--config",
        help="path to a JSON .prompt file (overrides --prompt)",
    )
    parser.add_argument("--n-ctx", type=int, default=4096, help="context size (default: 4096)")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads")
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="GPU layers to offload (default: 0, CPU only)",
    )
    parser.add_argument("--chat-format", default=None, help="override GGUF chat template")
    parser.add_argument("--temperature", type=float, default=None, help="override prompt temperature")
    parser.add_argument("--top-p", type=float, default=None, help="override prompt top_p")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="override prompt num_predict",
    )
    parser.add_argument("-o", "--output", help="write the report to a file instead of stdout")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the analysis progress bar",
    )
    return parser


def stdin_available() -> bool:
    """True when stdin is piped/redirected (not an interactive TTY)."""
    try:
        return not sys.stdin.isatty()
    except Exception:
        return False


def has_log_input(args: argparse.Namespace) -> bool:
    return bool(args.logfile) or stdin_available()


def read_log_text(logfile: str | None) -> tuple[str, str]:
    """Return (log_text, label) from a file path, '-', or piped stdin."""
    if logfile in (None, "-"):
        if logfile is None and not stdin_available():
            raise LogscryError("logfile is required (or pipe input via stdin)")
        text = sys.stdin.read()
        if not text.strip():
            raise LogscryError("stdin is empty")
        return text, "stdin"

    path = Path(logfile)
    if not path.is_file():
        raise LogscryError(f"logfile not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise LogscryError(f"logfile is empty: {path}")
    return text, str(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.download:
            model_path = download_model(args.models_dir, args.model_name)
            print(f"logscry: model ready at {model_path}", file=sys.stderr)
            if not has_log_input(args):
                return 0
            if not args.model:
                args.model = str(model_path)
        elif not has_log_input(args):
            raise LogscryError("logfile is required (or use --download, or pipe via stdin)")
        elif not args.model:
            raise LogscryError("--model is required (or use --download)")
        report_text = run(args)
    except (LogscryError, ConfigError, DownloadError, EngineError, OSError, ValueError) as exc:
        print(f"logscry: {exc}", file=sys.stderr)
        return 1
    write_report(report_text, Path(args.output) if args.output else None)
    return 0


def run(args: argparse.Namespace) -> str:
    if args.n_ctx < 512:
        raise LogscryError("--n-ctx must be >= 512")

    log_text, source_label = read_log_text(args.logfile)
    prompt_path = resolve_prompt_path(name=args.prompt, config=args.config)
    config = load_prompt(prompt_path).with_overrides(
        temperature=args.temperature,
        top_p=args.top_p,
        num_predict=args.max_tokens,
    )

    show_progress = not args.no_progress
    status(f"loading model {args.model} ...", enabled=show_progress)
    engine = Engine(
        model_path=Path(args.model),
        n_ctx=args.n_ctx,
        n_threads=args.threads,
        n_gpu_layers=args.n_gpu_layers,
        chat_format=args.chat_format,
    )
    status("model loaded", enabled=show_progress)
    raw = analyze(engine, config, log_text, show_progress=show_progress)
    report = parse_report(raw)
    return report.render(logfile=source_label, prompt=prompt_path, model=args.model)


if __name__ == "__main__":
    raise SystemExit(main())
