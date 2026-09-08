"""Utilities for locating the external MACE-OFF23 model used by HADES."""

from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path


MACE_MODEL_FILENAME = "MACE-OFF23_small.model"
MACE_MODEL_URL = (
    "https://github.com/ACEsuit/mace-off/raw/main/"
    "mace_off23/MACE-OFF23_small.model"
)


def default_mace_model_path() -> Path:
    """Return a user-writable cache path for the default MACE model."""
    configured_cache = os.environ.get("HADES_CACHE_DIR")

    if configured_cache:
        cache_dir = Path(configured_cache)
    elif os.environ.get("XDG_CACHE_HOME"):
        cache_dir = Path(os.environ["XDG_CACHE_HOME"]) / "hades"
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        cache_dir = Path(os.environ["LOCALAPPDATA"]) / "hades" / "Cache"
    else:
        cache_dir = Path.home() / ".cache" / "hades"

    return cache_dir / MACE_MODEL_FILENAME


def ensure_mace_model(model_path: str | os.PathLike[str] | None = None) -> Path:
    """Download the default model atomically when it is not already present."""
    destination = Path(model_path) if model_path is not None else default_mace_model_path()
    destination = destination.expanduser().resolve()

    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".download",
        dir=destination.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        urllib.request.urlretrieve(MACE_MODEL_URL, temporary_path)

        if temporary_path.stat().st_size == 0:
            raise RuntimeError("Downloaded MACE model is empty")

        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return destination
