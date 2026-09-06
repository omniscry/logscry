# logscry

A local LLM-based log analyzer for Linux.

`logscry` reads a log file and a JSON prompt config, runs a GGUF model in-process with [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) on CPU, and prints a human-readable report with **Summary** and **Findings**. Large logs are split on line boundaries and merged.

This uses llama.cpp rather than vLLM so it can run without a GPU.

## Install

### From PyPI

```bash
pip install logscry
```

### From source

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

For packaging/publishing tools:

```bash
pip install -e ".[dev]"
```

## Usage

Download the 3B GGUF, then analyze:

```bash
logscry --download --model-name Qwen/Qwen2.5-3B-Instruct-GGUF:qwen2.5-3b-instruct-q4_k_m.gguf
logscry examples/sample.syslog --model ./models/qwen2.5-3b-instruct-q4_k_m.gguf --prompt syslog
```

Or download and analyze in one step:

```bash
logscry examples/sample.syslog --download --model-name Qwen/Qwen2.5-3B-Instruct-GGUF:qwen2.5-3b-instruct-q4_k_m.gguf --prompt syslog
```

`--model` is a local `.gguf` path. `--model-name` is the Hugging Face repo (or `repo:file.gguf`) used by `--download`.

```bash
logscry /var/log/syslog --model ./models/qwen2.5-3b-instruct-q4_k_m.gguf --prompt syslog
logscry examples/sample.syslog --model ./models/qwen2.5-3b-instruct-q4_k_m.gguf -o report.txt
python -m logscry examples/sample.syslog --model ./models/qwen2.5-3b-instruct-q4_k_m.gguf
```

### Pipe from other logs

Progress goes to stderr; the Summary/Findings report goes to stdout.

```bash
journalctl -n 500 --no-pager | logscry --model ./models/qwen2.5-3b-instruct-q4_k_m.gguf --prompt syslog
journalctl -u ssh --no-pager | logscry - --model ./models/qwen2.5-3b-instruct-q4_k_m.gguf --prompt syslog
```

Omit the logfile (or pass `-`) when piping. Use `--no-pager` with `journalctl` so output is plain text.

| Flag | Meaning |
| --- | --- |
| `--download` | Download a GGUF into `--models-dir` |
| `--model-name` | Hugging Face repo, `repo:file.gguf`, or `file.gguf` |
| `--models-dir` | Download directory (default: `models`) |
| `--model` | Path to a GGUF file (required unless `--download`) |
| `--prompt` | Bundled config: `generic`, `syslog`, or `messages` |
| `--config` | Path to a custom JSON `.prompt` file |
| `--n-ctx` | Context size (default 4096) |
| `--threads` | CPU threads |
| `--temperature`, `--top-p`, `--max-tokens` | Override values from the prompt file |
| `-o` / `--output` | Write the report to a file |
| `--no-progress` | Disable the analysis progress bar |

## Prompt format

```json
{
  "system_prompt": "You are a log analyzer etcetc.....",
  "options": {
    "temperature": 0.0,
    "top_p": 0.9,
    "num_predict": 512
  }
}
```

## Tests

```bash
python -m unittest discover -s tests
```

## Publish to PyPI

Build the wheel and source distribution:

```bash
pip install -e ".[dev]"
python -m build
```

Check and upload (TestPyPI first recommended):

```bash
twine check dist/*
twine upload --repository testpypi dist/*
twine upload dist/*
```

Requires a PyPI API token and an account that owns the `logscry` project name.
