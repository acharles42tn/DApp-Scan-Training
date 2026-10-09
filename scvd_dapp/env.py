"""Process-environment setup for HPC runs (carried over from scvd.env unchanged in behaviour).

Call :func:`configure_environment` **before** importing any HuggingFace library.
It silences the tqdm/transformers log spam that floods SLURM ``.out`` files and
points the HF caches at a writable directory. Model loads fall back to
``local_files_only`` when a compute node loses internet mid-job.
"""

from __future__ import annotations

import os
from functools import partialmethod
from pathlib import Path
from typing import Any, Callable, Dict


def configure_environment(cache_dir: str | None = None) -> None:
    """Quiet the logs and (optionally) pin HF caches. Call before HF imports."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("DISABLE_CUSTOM_GENERATE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if cache_dir:
        hf_cache = Path(cache_dir) / "huggingface"
        (hf_cache / "hub").mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_cache)
        os.environ["HF_DATASETS_CACHE"] = str(hf_cache / "datasets")

    try:
        from tqdm import tqdm as _tqdm

        _tqdm.__init__ = partialmethod(_tqdm.__init__, disable=True)
    except ImportError:
        pass

    try:
        import transformers

        transformers.logging.set_verbosity_error()
    except ImportError:
        pass


def _is_repo_not_found(err: BaseException) -> bool:
    """True if the error means the repo is missing/gated (404), not a network blip."""
    try:
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

        cause: BaseException | None = err
        while cause is not None:
            if isinstance(cause, (RepositoryNotFoundError, GatedRepoError)):
                return True
            cause = cause.__cause__
    except ImportError:
        pass
    msg = str(err).lower()
    return ("not a valid model identifier" in msg or "repository not found" in msg
            or "gated repo" in msg)


def load_with_offline_fallback(loader: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run ``loader(*args, **kwargs)``; on a network error retry with ``local_files_only=True``.

    A 404 (repo missing, private, or gated) is permanent, so it is re-raised as-is.
    """
    try:
        return loader(*args, **kwargs)
    except (ConnectionError, OSError, RuntimeError) as err:
        if _is_repo_not_found(err):
            raise
        print(f"  Network error ({type(err).__name__}); retrying with local_files_only=True", flush=True)
        kwargs["local_files_only"] = True
        return loader(*args, **kwargs)


def describe_runtime() -> Dict[str, Any]:
    """Small dict of python/torch/transformers/CUDA facts, logged at job start."""
    import platform

    info: Dict[str, Any] = {"python": platform.python_version()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except ImportError:
        info["torch"] = None
    for mod in ("transformers", "peft"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            info[mod] = None
    return info
