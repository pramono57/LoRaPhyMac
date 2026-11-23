"""Versioned directory contract for generated paper artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


OUTPUT_LAYOUT_VERSION = "paper_artifacts_v1"
SWEEP_OUTPUT_LAYOUT_VERSION = "sweep_cohort_v1"


@dataclass(frozen=True)
class ExperimentArtifactPaths:
    """Canonical paths below one versioned experiment directory."""

    experiment_dir: Path
    layout_version: str = OUTPUT_LAYOUT_VERSION

    @classmethod
    def from_spec(cls, spec: Any) -> "ExperimentArtifactPaths":
        """Build paths from an object exposing ``output_dir`` and ``experiment_id``."""
        return cls(
            Path(spec.output_dir) / str(spec.experiment_id),
            getattr(spec, "output_layout_version", OUTPUT_LAYOUT_VERSION),
        )

    @classmethod
    def for_experiment(cls, experiment_dir: str | Path) -> "ExperimentArtifactPaths":
        """Resolve the layout from a cohort manifest, retaining legacy defaults."""
        directory = Path(experiment_dir)
        manifest = directory / "experiment_manifest.json"
        if manifest.is_file():
            try:
                import json

                payload = json.loads(manifest.read_text(encoding="utf-8"))
                version = str(payload.get("output_layout_version", OUTPUT_LAYOUT_VERSION))
            except (OSError, ValueError, TypeError):
                version = OUTPUT_LAYOUT_VERSION
        else:
            version = OUTPUT_LAYOUT_VERSION
        return cls(directory, version)

    @property
    def is_sweep_standard(self) -> bool:
        return self.layout_version == SWEEP_OUTPUT_LAYOUT_VERSION

    @property
    def runs(self) -> Path:
        return self.experiment_dir / "runs"

    @property
    def master(self) -> Path:
        if self.is_sweep_standard:
            return self.aggregate
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
        if self.is_sweep_standard:
            return self.analysis / "tables"
        return self.experiment_dir / "tables"

    @property
    def figures(self) -> Path:
        if self.is_sweep_standard:
            return self.analysis / "figures"
        return self.experiment_dir / "figures"

    @property
    def analysis(self) -> Path:
        return self.experiment_dir / "analysis"

    @property
    def analysis_manifest(self) -> Path:
        if self.is_sweep_standard:
            return self.analysis / "analysis_manifest.json"
        return self.experiment_dir / "analysis_manifest.json"

    @property
    def logs(self) -> Path:
        if self.is_sweep_standard:
            return self.runs / ".logs"
        return self.experiment_dir / "logs"

    @property
    def manifest(self) -> Path:
        return self.experiment_dir / "experiment_manifest.json"

    @property
    def status(self) -> Path:
        return self.experiment_dir / "experiment_status.json"

    def ensure(self) -> "ExperimentArtifactPaths":
        """Create the complete output skeleton and return this path object."""
        paths = [self.runs, self.aggregate]
        if self.is_sweep_standard:
            paths.extend((self.figures, self.tables))
        else:
            paths.extend((self.master, self.tables, self.figures, self.logs))
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
        return self

    def directory_map(self, *, absolute: bool = False) -> dict[str, str]:
        """Return stable machine-readable names for manifest and status files."""
        directories = {
            "experiment": self.experiment_dir,
            "runs": self.runs,
            "aggregate": self.aggregate,
            "tables": self.tables,
            "figures": self.figures,
            "logs": self.runs / ".logs" if self.is_sweep_standard else self.logs,
        }
        if not self.is_sweep_standard:
            directories["master"] = self.master
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
    paths = ExperimentArtifactPaths.for_experiment(experiment_dir)
    if paths.is_sweep_standard:
        candidates = [paths.aggregate / filename]
        if filename.endswith(".csv"):
            candidates.insert(0, paths.aggregate / f"{filename}.gz")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return paths.aggregate / filename
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
    "SWEEP_OUTPUT_LAYOUT_VERSION",
    "resolve_master_table_path",
]
