"""Shared configuration and paths for publication-only artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


PUBLICATION_CONFIG_KEYS = frozenset(
    {"publication_id", "source_experiment_dir", "sources", "output_dir", "options"}
)


def _resolved_relative_to(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


@dataclass(frozen=True)
class PublicationSpec:
    """Resolved source and destination for one publication workflow."""

    publication_id: str
    source_experiment_dir: Path | None
    sources: Mapping[str, Path]
    output_dir: Path
    options: Mapping[str, Any]
    config_path: Path


def load_publication_spec(path: str | Path) -> PublicationSpec:
    """Load a strict publication mapping without modifying the filesystem."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Publication configuration must be a JSON object")
    unknown = set(payload).difference(PUBLICATION_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Publication configuration contains unknown keys: {sorted(unknown)}")
    missing = {"publication_id", "output_dir"}.difference(payload)
    if missing:
        raise ValueError(f"Publication configuration is missing keys: {sorted(missing)}")
    publication_id = str(payload["publication_id"]).strip()
    if not publication_id:
        raise ValueError("publication_id must not be empty")
    options = payload.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("Publication configuration options must be a JSON object")
    has_single_source = "source_experiment_dir" in payload
    has_multi_source = "sources" in payload
    if has_single_source == has_multi_source:
        raise ValueError(
            "Publication configuration must define exactly one of "
            "source_experiment_dir or sources"
        )

    source_experiment_dir: Path | None = None
    resolved_sources: dict[str, Path] = {}
    if has_single_source:
        source_experiment_dir = _resolved_relative_to(
            config_path,
            payload["source_experiment_dir"],
        )
    else:
        raw_sources = payload["sources"]
        if not isinstance(raw_sources, dict) or not raw_sources:
            raise ValueError("Publication configuration sources must be a non-empty object")
        for raw_role, raw_path in raw_sources.items():
            role = str(raw_role).strip()
            if not role:
                raise ValueError("Publication source roles must not be empty")
            if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
                raise ValueError(f"Publication source {role!r} must be a non-empty path")
            resolved_sources[role] = _resolved_relative_to(config_path, raw_path)

    output_dir = _resolved_relative_to(config_path, payload["output_dir"])
    source_paths = (
        [source_experiment_dir]
        if source_experiment_dir is not None
        else list(resolved_sources.values())
    )
    if any(output_dir == source or output_dir.is_relative_to(source) for source in source_paths):
        raise ValueError("Publication output_dir must be separate from every source experiment")
    return PublicationSpec(
        publication_id=publication_id,
        source_experiment_dir=source_experiment_dir,
        sources=MappingProxyType(resolved_sources),
        output_dir=output_dir,
        options=MappingProxyType(dict(options)),
        config_path=config_path,
    )


@dataclass(frozen=True)
class PublicationArtifactPaths:
    """Canonical tables and figures below one publication output directory."""

    publication_dir: Path

    @property
    def manifest(self) -> Path:
        return self.publication_dir / "publication_manifest.json"

    @property
    def tables(self) -> Path:
        return self.publication_dir / "tables"

    @property
    def figures(self) -> Path:
        return self.publication_dir / "figures"

    def ensure(self) -> "PublicationArtifactPaths":
        self.tables.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        return self


__all__ = [
    "PUBLICATION_CONFIG_KEYS",
    "PublicationArtifactPaths",
    "PublicationSpec",
    "load_publication_spec",
]
