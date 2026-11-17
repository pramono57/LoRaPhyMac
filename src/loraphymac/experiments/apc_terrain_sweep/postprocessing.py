"""Idempotent tables, figures, and preview metadata for a completed terrain sweep."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loraphymac.experiments.common.multiseed_pipeline import (
    atomic_write_json,
    file_sha256,
    load_experiment_spec,
)
from loraphymac.experiments.common.outputs import ExperimentArtifactPaths
from loraphymac.publications.paper_rapc_multihop_lora.analysis import (
    DEFAULT_COMPARISON_TERRAIN_DELTA_H,
    build_publication_tables,
)
from loraphymac.publications.paper_rapc_multihop_lora.figures import generate_all_figures


TERRAIN_SWEEP_ANALYSIS_MANIFEST_VERSION = "apc_terrain_sweep_analysis_v1"
WORKFLOW_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = WORKFLOW_DIR / "configs" / "production_v1.json"


def terrain_sweep_paths(
    config: str | Path = DEFAULT_CONFIG,
    *,
    experiment_dir: str | Path | None = None,
) -> ExperimentArtifactPaths:
    """Resolve the versioned terrain-sweep output directory."""
    if experiment_dir is not None:
        return ExperimentArtifactPaths(Path(experiment_dir).expanduser().resolve())
    return ExperimentArtifactPaths.from_spec(load_experiment_spec(config))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _validate_completed_sweep(paths: ExperimentArtifactPaths) -> dict[str, Any]:
    status = _read_json(paths.status, "terrain-sweep experiment status")
    if status.get("status") != "PASS":
        raise ValueError(f"Terrain-sweep status must be PASS: {paths.status}")
    if status.get("seed_grid_complete") is not True:
        raise ValueError(f"Terrain-sweep seed grid is incomplete: {paths.status}")
    if status.get("passed_run_count_on_disk") != status.get("requested_run_count"):
        raise ValueError(f"Terrain-sweep run evidence is incomplete: {paths.status}")
    aggregate = paths.aggregate / "core_metrics_long.csv"
    if not aggregate.is_file():
        raise ValueError(f"Terrain-sweep aggregate is missing: {aggregate}")
    return status


def build_terrain_sweep_artifacts(
    config: str | Path = DEFAULT_CONFIG,
    *,
    experiment_dir: str | Path | None = None,
    terrain_delta_h: float = DEFAULT_COMPARISON_TERRAIN_DELTA_H,
) -> dict[str, Any]:
    """Build experiment-owned tables and figures without executing a simulation."""
    paths = terrain_sweep_paths(config, experiment_dir=experiment_dir)
    status = _validate_completed_sweep(paths)
    table_files = build_publication_tables(
        paths.experiment_dir,
        publication_dir=paths.experiment_dir,
        terrain_delta_h=terrain_delta_h,
        manuscript_names=False,
    )
    figure_files = generate_all_figures(
        paths.experiment_dir,
        publication_dir=paths.experiment_dir,
        terrain_delta_h=terrain_delta_h,
        manuscript_names=False,
    )
    generated = sorted(
        {*table_files.values(), *figure_files},
        key=lambda path: str(path),
    )
    records = [
        {
            "path": path.resolve().relative_to(paths.experiment_dir.resolve()).as_posix(),
            "format": path.suffix.lower().lstrip("."),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in generated
    ]
    manifest_path = paths.experiment_dir / "analysis_manifest.json"
    aggregate = paths.aggregate / "core_metrics_long.csv"
    atomic_write_json(
        manifest_path,
        {
            "analysis_manifest_version": TERRAIN_SWEEP_ANALYSIS_MANIFEST_VERSION,
            "status": "PASS",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment_id": status.get("experiment_id"),
            "experiment_status_sha256": file_sha256(paths.status),
            "core_metrics_sha256": file_sha256(aggregate),
            "comparison_terrain_delta_h": float(terrain_delta_h),
            "artifact_counts": {
                "tables": sum(path.suffix.lower() == ".csv" for path in generated),
                "figures": sum(path.suffix.lower() in {".png", ".pdf"} for path in generated),
                "total": len(generated),
            },
            "artifacts": records,
        },
    )
    return {
        "status": "PASS",
        "experiment_id": status.get("experiment_id"),
        "experiment_dir": str(paths.experiment_dir),
        "table_count": sum(path.suffix.lower() == ".csv" for path in generated),
        "figure_count": sum(path.suffix.lower() in {".png", ".pdf"} for path in generated),
        "artifact_count": len(generated),
        "tables": [str(path) for path in generated if path.suffix.lower() == ".csv"],
        "figures": [
            str(path) for path in generated if path.suffix.lower() in {".png", ".pdf"}
        ],
        "analysis_manifest": str(manifest_path),
    }


__all__ = [
    "TERRAIN_SWEEP_ANALYSIS_MANIFEST_VERSION",
    "build_terrain_sweep_artifacts",
    "terrain_sweep_paths",
]
