"""Download a GGUF model for local analysis."""

from __future__ import annotations

from pathlib import Path

DEFAULT_REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"
DEFAULT_FILE = "qwen2.5-7b-instruct-q4_k_m.gguf"
DEFAULT_MODELS_DIR = Path("models")


class DownloadError(RuntimeError):
    """Model download failure."""


def parse_model_name(model_name: str | None) -> tuple[str, str | None]:
    """Return (repo_id, filename_or_None).

    Accepts ``org/repo``, ``org/repo:file.gguf``, or ``file.gguf``.
    """
    name = (model_name or DEFAULT_REPO).strip()
    if not name:
        raise DownloadError("model name is empty")
    if name.endswith(".gguf") and "/" not in name and ":" not in name:
        return DEFAULT_REPO, Path(name).name
    if ":" in name:
        repo, filename = name.split(":", 1)
        repo, filename = repo.strip(), filename.strip()
        if not repo or not filename:
            raise DownloadError("use --model-name org/repo:file.gguf")
        return repo, filename
    return name, None


def download_model(
    dest_dir: Path | str | None = None,
    model_name: str | None = None,
) -> Path:
    dest = Path(dest_dir) if dest_dir is not None else DEFAULT_MODELS_DIR
    dest.mkdir(parents=True, exist_ok=True)
    repo_id, filename = parse_model_name(model_name)
    if filename is None:
        filename = _pick_gguf_file(repo_id)

    target = dest / Path(filename).name
    if target.is_file() and target.stat().st_size > 0:
        return target.resolve()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise DownloadError(
            "huggingface_hub is required to download models. "
            "Install with: pip install huggingface_hub"
        ) from exc

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(dest),
        )
    except Exception as exc:
        raise DownloadError(f"failed to download {repo_id} ({filename}): {exc}") from exc
    return Path(path).resolve()


def _pick_gguf_file(repo_id: str) -> str:
    if repo_id == DEFAULT_REPO:
        return DEFAULT_FILE
    try:
        from huggingface_hub import list_repo_files
    except ImportError as exc:
        raise DownloadError(
            "huggingface_hub is required to download models. "
            "Install with: pip install huggingface_hub"
        ) from exc
    try:
        files = [name for name in list_repo_files(repo_id) if name.endswith(".gguf")]
    except Exception as exc:
        raise DownloadError(f"failed to list files for {repo_id}: {exc}") from exc
    if not files:
        raise DownloadError(f"no GGUF files found in {repo_id}")
    preferred = [name for name in files if "q4_k_m" in name.lower()]
    return Path(preferred[0] if preferred else files[0]).name
