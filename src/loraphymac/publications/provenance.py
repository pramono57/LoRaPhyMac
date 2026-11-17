"""Validation and provenance records for publication artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from math import prod
from pathlib import Path
from typing import Any

from loraphymac._version import SIMULATOR_VERSION
from loraphymac.experiments.common.multiseed_pipeline import (
    atomic_write_json,
    file_sha256,
)
from loraphymac.experiments.common.outputs import ExperimentArtifactPaths
from loraphymac.paths import repository_root
from loraphymac.publications.common import PublicationArtifactPaths, PublicationSpec


PUBLICATION_PROVENANCE_VERSION = "publication_provenance_v1"
REQUIRED_VERSION_FIELDS = (
    "pipeline_version",
    "simulator_api_version",
    "output_layout_version",
)


class PublicationProvenanceError(ValueError):
    """Raised when experiment evidence is unsafe for publication."""


@dataclass(frozen=True)
class ValidatedExperimentEvidence:
    """Read-only, validated evidence used by a publication workflow."""

    paths: ExperimentArtifactPaths
    manifest: Mapping[str, Any]
    status: Mapping[str, Any]
    core_metrics: Path


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PublicationProvenanceError(f"Missing {label}: {path}")
    try:
        import json

        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationProvenanceError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PublicationProvenanceError(f"{label} must contain a JSON object: {path}")
    return value


def _required_positive_integer(mapping: Mapping[str, Any], key: str, label: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PublicationProvenanceError(f"{label}.{key} must be a positive integer")
    return value


def _required_nonnegative_integer(mapping: Mapping[str, Any], key: str, label: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PublicationProvenanceError(f"{label}.{key} must be a non-negative integer")
    return value


def _declared_grid_size(spec: Mapping[str, Any]) -> int:
    factors = spec.get("factors")
    seeds = spec.get("seeds")
    if not isinstance(factors, Mapping) or not factors:
        raise PublicationProvenanceError("experiment manifest spec.factors must be non-empty")
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence) or not seeds:
        raise PublicationProvenanceError("experiment manifest spec.seeds must be non-empty")
    factor_sizes = []
    for name, values in factors.items():
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
            raise PublicationProvenanceError(
                f"experiment manifest factor {name!r} must contain values"
            )
        factor_sizes.append(len(values))
    return len(seeds) * prod(factor_sizes)


def validate_completed_experiment(
    paths: ExperimentArtifactPaths,
    *,
    expected_run_count: int | None = None,
    contract_versions: Mapping[str, str] | None = None,
) -> ValidatedExperimentEvidence:
    """Validate a completed source experiment without modifying it."""
    if not paths.experiment_dir.is_dir():
        raise PublicationProvenanceError(
            f"Source experiment directory does not exist: {paths.experiment_dir}"
        )

    manifest = _read_json_object(paths.manifest, "experiment manifest")
    status = _read_json_object(paths.status, "experiment status")
    if status.get("status") != "PASS":
        raise PublicationProvenanceError(
            f"Source experiment status must be PASS, got {status.get('status')!r}"
        )
    if status.get("seed_grid_complete") is not True:
        raise PublicationProvenanceError("Source experiment seed grid is incomplete")

    requested = _required_positive_integer(status, "requested_run_count", "experiment status")
    passed_on_disk = _required_nonnegative_integer(
        status, "passed_run_count_on_disk", "experiment status"
    )
    if passed_on_disk != requested:
        raise PublicationProvenanceError(
            "Source experiment run evidence is incomplete: "
            f"passed_on_disk={passed_on_disk}, requested={requested}"
        )
    if expected_run_count is not None and requested != expected_run_count:
        raise PublicationProvenanceError(
            f"Source experiment must contain {expected_run_count} runs, got {requested}"
        )

    status_counts = status.get("status_counts")
    if not isinstance(status_counts, Mapping):
        raise PublicationProvenanceError("experiment status.status_counts must be an object")
    failed = _required_nonnegative_integer(status_counts, "FAILED", "status_counts")
    if failed:
        raise PublicationProvenanceError(f"Source experiment contains {failed} failed run(s)")

    manifest_spec = manifest.get("spec")
    if not isinstance(manifest_spec, Mapping):
        raise PublicationProvenanceError("experiment manifest.spec must be an object")
    experiment_id = status.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise PublicationProvenanceError("experiment status.experiment_id must be non-empty")
    if manifest_spec.get("experiment_id") != experiment_id:
        raise PublicationProvenanceError(
            "Experiment ID differs between experiment manifest and status"
        )
    declared_grid_size = _declared_grid_size(manifest_spec)
    if declared_grid_size != requested:
        raise PublicationProvenanceError(
            "Experiment manifest grid size differs from requested_run_count: "
            f"grid={declared_grid_size}, requested={requested}"
        )

    for field in REQUIRED_VERSION_FIELDS:
        manifest_value = manifest.get(field)
        if not isinstance(manifest_value, str) or not manifest_value:
            raise PublicationProvenanceError(f"experiment manifest.{field} is missing")
        if status.get(field) != manifest_value:
            raise PublicationProvenanceError(
                f"{field} differs between experiment manifest and status"
            )

    simulator_version = manifest.get("simulator_version")
    if simulator_version is not None:
        if not isinstance(simulator_version, str) or not simulator_version:
            raise PublicationProvenanceError(
                "experiment manifest.simulator_version must be a non-empty string"
            )
        if status.get("simulator_version") != simulator_version:
            raise PublicationProvenanceError(
                "simulator_version differs between experiment manifest and status"
            )
    elif "simulator_version" in status:
        raise PublicationProvenanceError("simulator_version is present only in experiment status")

    profile = manifest_spec.get("base_config", {}).get("experiment_profile")
    if contract_versions is not None:
        expected_contract = contract_versions.get(str(profile))
        if expected_contract is None:
            raise PublicationProvenanceError(
                f"Unsupported source experiment profile for publication: {profile!r}"
            )
        if manifest.get("experiment_contract_version") != expected_contract:
            raise PublicationProvenanceError(
                "Source experiment contract version does not match its profile"
            )

    if not isinstance(manifest.get("spec_hash"), str) or not manifest["spec_hash"]:
        raise PublicationProvenanceError("experiment manifest.spec_hash is missing")
    source_fingerprint = manifest.get("source_fingerprint")
    if not isinstance(source_fingerprint, Mapping):
        raise PublicationProvenanceError("experiment manifest.source_fingerprint is missing")
    if source_fingerprint.get("algorithm") != "sha256" or not isinstance(
        source_fingerprint.get("files"), Mapping
    ):
        raise PublicationProvenanceError("experiment source fingerprint is malformed")
    if not source_fingerprint["files"]:
        raise PublicationProvenanceError("experiment source fingerprint has no files")

    core_metrics = paths.aggregate / "core_metrics_long.csv"
    if not core_metrics.is_file():
        raise PublicationProvenanceError(f"Missing authoritative aggregate: {core_metrics}")
    return ValidatedExperimentEvidence(
        paths=paths,
        manifest=manifest,
        status=status,
        core_metrics=core_metrics,
    )


def _path_label(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _package_version() -> str:
    return SIMULATOR_VERSION


def _artifact_records(
    paths: PublicationArtifactPaths,
    artifacts: Sequence[str | Path],
) -> list[dict[str, Any]]:
    publication_root = paths.publication_dir.resolve()
    unique = sorted({Path(item).resolve() for item in artifacts}, key=str)
    records = []
    for artifact in unique:
        if not artifact.is_file():
            raise PublicationProvenanceError(f"Generated artifact is missing: {artifact}")
        try:
            relative = artifact.relative_to(publication_root)
        except ValueError as exc:
            raise PublicationProvenanceError(
                f"Generated artifact is outside publication output: {artifact}"
            ) from exc
        top_level = relative.parts[0] if relative.parts else ""
        kind = "table" if top_level == "tables" else "figure" if top_level == "figures" else "other"
        records.append(
            {
                "path": relative.as_posix(),
                "kind": kind,
                "format": artifact.suffix.lower().lstrip("."),
                "size_bytes": artifact.stat().st_size,
                "sha256": file_sha256(artifact),
            }
        )
    return records


def build_publication_manifest(
    *,
    spec: PublicationSpec,
    evidence: ValidatedExperimentEvidence,
    output_paths: PublicationArtifactPaths,
    artifacts: Sequence[str | Path],
    workflow_version: str,
    generator_module: str,
    generator_files: Sequence[str | Path],
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a complete provenance record for generated publication files."""
    root = repository_root(spec.config_path)
    generator_fingerprint = {
        _path_label(Path(source), root): file_sha256(Path(source))
        for source in sorted({Path(item).resolve() for item in generator_files}, key=str)
    }
    records = _artifact_records(output_paths, artifacts)
    table_count = sum(record["kind"] == "table" for record in records)
    figure_count = sum(record["kind"] == "figure" for record in records)
    status = evidence.status
    experiment_manifest = evidence.manifest
    return {
        "publication_provenance_version": PUBLICATION_PROVENANCE_VERSION,
        "publication_id": spec.publication_id,
        "status": "PASS",
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "workflow_version": workflow_version,
        "generator": {
            "module": generator_module,
            "loraphymac_version": _package_version(),
            "source_fingerprint": {
                "algorithm": "sha256",
                "files": generator_fingerprint,
            },
        },
        "publication_config": {
            "path": _path_label(spec.config_path, root),
            "sha256": file_sha256(spec.config_path),
            "options": dict(spec.options),
        },
        "source_experiment": {
            "path": _path_label(evidence.paths.experiment_dir, root),
            "experiment_id": status["experiment_id"],
            "status": status["status"],
            "requested_run_count": status["requested_run_count"],
            "passed_run_count_on_disk": status["passed_run_count_on_disk"],
            "status_counts": dict(status["status_counts"]),
            "seed_grid_complete": status["seed_grid_complete"],
            "pipeline_version": experiment_manifest["pipeline_version"],
            "simulator_version": experiment_manifest.get("simulator_version"),
            "simulator_api_version": experiment_manifest["simulator_api_version"],
            "output_layout_version": experiment_manifest["output_layout_version"],
            "experiment_contract_version": experiment_manifest.get("experiment_contract_version"),
            "spec_hash": experiment_manifest["spec_hash"],
            "source_fingerprint": experiment_manifest["source_fingerprint"],
            "experiment_manifest_sha256": file_sha256(evidence.paths.manifest),
            "experiment_status_sha256": file_sha256(evidence.paths.status),
            "core_metrics_sha256": file_sha256(evidence.core_metrics),
        },
        "artifact_counts": {
            "tables": table_count,
            "figures": figure_count,
            "total": len(records),
        },
        "artifacts": records,
    }


def write_publication_manifest(
    paths: PublicationArtifactPaths,
    manifest: Mapping[str, Any],
) -> Path:
    """Atomically publish a provenance manifest after all outputs exist."""
    atomic_write_json(paths.manifest, dict(manifest))
    return paths.manifest


__all__ = [
    "PUBLICATION_PROVENANCE_VERSION",
    "PublicationProvenanceError",
    "ValidatedExperimentEvidence",
    "build_publication_manifest",
    "validate_completed_experiment",
    "write_publication_manifest",
]
