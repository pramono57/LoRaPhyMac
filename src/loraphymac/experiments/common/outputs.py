"""Versioned directory contract for generated paper artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


OUTPUT_LAYOUT_VERSION = "paper_artifacts_v1"


@dataclass(frozen=True)
class ExperimentArtifactPaths:
    """Canonical paths below one versioned experiment directory."""

    experiment_dir: Path

    @classmethod
    def from_spec(cls, spec: Any) -> "ExperimentArtifactPaths":
        """Build paths from an object exposing ``output_dir`` and ``experiment_id``."""
        return cls(Path(spec.output_dir) / str(spec.experiment_id))

    @property
    def runs(self) -> Path:
        return self.experiment_dir / "runs"

    @property
    def master(self) -> Path:
        return self.experiment_dir / "master"

    @property
    def scientific_master(self) -> Path:
        """Atomically published opt-in scientific master cohort."""
        return self.master / "scientific_master"

    @property
    def aggregate(self) -> Path:
        return self.experiment_dir / "aggregate"

    @property
    def tables(self) -> Path:
        return self.experiment_dir / "tables"

    @property
    def figures(self) -> Path:
        return self.experiment_dir / "figures"

    @property
    def logs(self) -> Path:
        return self.experiment_dir / "logs"

    @property
    def manifest(self) -> Path:
        return self.experiment_dir / "experiment_manifest.json"

    @property
    def status(self) -> Path:
        return self.experiment_dir / "experiment_status.json"

    def ensure(self) -> "ExperimentArtifactPaths":
        """Create the complete output skeleton and return this path object."""
        for path in (
            self.runs,
            self.master,
            self.aggregate,
            self.tables,
            self.figures,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def directory_map(self, *, absolute: bool = False) -> dict[str, str]:
        """Return stable machine-readable names for manifest and status files."""
        directories = {
            "experiment": self.experiment_dir,
            "runs": self.runs,
            "master": self.master,
            "aggregate": self.aggregate,
            "tables": self.tables,
            "figures": self.figures,
            "logs": self.logs,
        }
        if absolute:
            return {name: str(path.resolve()) for name, path in directories.items()}
        return {
            name: "."
            if path == self.experiment_dir
            else path.relative_to(self.experiment_dir).as_posix()
            for name, path in directories.items()
        }


def resolve_master_table_path(experiment_dir: str | Path, filename: str) -> Path:
    """Prefer finalized scientific master output, then the legacy master path."""
    paths = ExperimentArtifactPaths(Path(experiment_dir))
    candidates = [paths.scientific_master / filename]
    if filename.endswith(".csv"):
        candidates.insert(0, paths.scientific_master / f"{filename}.gz")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return paths.master / filename


__all__ = [
    "ExperimentArtifactPaths",
    "OUTPUT_LAYOUT_VERSION",
    "resolve_master_table_path",
]
