"""Filesystem helpers for source checkouts and installed distributions."""

from __future__ import annotations

from pathlib import Path


def repository_root(start: str | Path | None = None) -> Path:
    """Locate the repository root, falling back to the current directory.

    Paper runners need the checkout root for source fingerprints. Installed
    callers may not have a checkout and therefore use their current directory.
    """
    origin = Path(start).resolve() if start is not None else Path(__file__).resolve()
    search_root = origin if origin.is_dir() else origin.parent
    for candidate in (search_root, *search_root.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "loraphymac"
        ).is_dir():
            return candidate
    return Path.cwd().resolve()


__all__ = ["repository_root"]
