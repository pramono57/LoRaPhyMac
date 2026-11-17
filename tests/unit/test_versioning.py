from __future__ import annotations

import re
from pathlib import Path

from loraphymac import (
    SIMULATOR_API_VERSION,
    SIMULATOR_VERSION,
    Simulator,
    __version__,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _declared_version(path: Path, pattern: str) -> str:
    match = re.search(pattern, path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    assert match is not None, f"Version declaration is missing from {path}"
    return match.group(1)


def test_release_version_is_v1_and_public_runtime_values_are_consistent():
    assert __version__ == "1.0.0"
    assert SIMULATOR_VERSION == __version__
    assert Simulator.version == __version__
    assert SIMULATOR_API_VERSION == "simulator_api_v1"
    assert Simulator.api_version == SIMULATOR_API_VERSION


def test_packaging_and_citation_versions_match_runtime_source():
    package_version = _declared_version(
        PROJECT_ROOT / "pyproject.toml",
        r'^version\s*=\s*"([^"]+)"',
    )
    citation_version = _declared_version(
        PROJECT_ROOT / "CITATION.cff",
        r"^version:\s*([^\s]+)",
    )

    assert package_version == __version__
    assert citation_version == __version__
