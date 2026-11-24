"""Idempotent tables, figures, and preview metadata for a completed terrain sweep."""

from __future__ import annotations

import json
import argparse
import itertools
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loraphymac.experiments.common.multiseed_pipeline import (
    atomic_write_json,
    file_sha256,
    load_experiment_spec,
)
from loraphymac.experiments.common.analysis_publication import AnalysisPublisher
from loraphymac.experiments.common.outputs import ExperimentArtifactPaths
from loraphymac.experiments.common.multiseed_pipeline import _read_verified_scientific_tables
from loraphymac.experiments.common.multiseed_pipeline import content_hash
from loraphymac.experiments.apc_terrain_profile_diagnostic.contract import (
    APC_TERRAIN_PROFILE_DIAGNOSTIC_PROFILE,
    APC_TERRAIN_PROFILE_DIAGNOSTIC_RUN_COUNT,
    APC_TERRAIN_PROFILE_DIAGNOSTIC_SEEDS,
    APC_TERRAIN_PROFILE_DIAGNOSTIC_TERRAIN,
    diagnostic_config_violations,
)
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
        return ExperimentArtifactPaths.for_experiment(Path(experiment_dir).expanduser().resolve())
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
    if status.get("status") not in {"PASS", "COMPLETE"}:
        raise ValueError(f"Terrain-sweep status must be PASS or COMPLETE: {paths.status}")
    if status.get("seed_grid_complete") is not True:
        raise ValueError(f"Terrain-sweep seed grid is incomplete: {paths.status}")
    if status.get("passed_run_count_on_disk") != status.get("requested_run_count"):
        raise ValueError(f"Terrain-sweep run evidence is incomplete: {paths.status}")
    aggregate = paths.aggregate / "core_metrics_long.csv"
    if not aggregate.is_file():
        raise ValueError(f"Terrain-sweep aggregate is missing: {aggregate}")
    if paths.is_sweep_standard:
        manifest = _read_json(paths.manifest, "terrain-sweep experiment manifest")
        run_ids = manifest.get("run_ids")
        grid = manifest.get("parameter_grid", {})
        factors, seeds = grid.get("factors"), grid.get("seeds")
        if (not isinstance(run_ids, list) or not run_ids
                or len(run_ids) != len(set(run_ids))
                or not isinstance(factors, dict) or not isinstance(seeds, list) or not seeds):
            raise ValueError("Terrain manifest must contain unique expected run IDs")
        if manifest.get("expected_run_count") != len(run_ids):
            raise ValueError("Terrain expected_run_count conflicts with run_ids")
        matrix = grid.get("run_matrix")
        spec = manifest.get("spec", {})
        profile = spec.get("base_config", {}).get("experiment_profile")
        if profile == APC_TERRAIN_PROFILE_DIAGNOSTIC_PROFILE:
            if not isinstance(matrix, list) or len(matrix) != APC_TERRAIN_PROFILE_DIAGNOSTIC_RUN_COUNT:
                raise ValueError("Terrain diagnostic manifest must contain its exact ten-row run_matrix")
            expected = {
                (mode, mode, APC_TERRAIN_PROFILE_DIAGNOSTIC_TERRAIN, seed)
                for mode in ("fixed", "adaptive")
                for seed in APC_TERRAIN_PROFILE_DIAGNOSTIC_SEEDS
            }
            try:
                declared = {
                    (
                        row["node_ptx_mode"], row["relay_ptx_mode"],
                        int(row["terrain_delta_h"]), int(row["seed"]),
                    )
                    for row in matrix
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Terrain diagnostic run_matrix contains an invalid row") from exc
            if len(declared) != len(matrix) or declared != expected:
                raise ValueError("Terrain diagnostic run_matrix is missing, duplicated, or unexpected")
        elif profile == "apc_terrain_profile_full_sweep_v1":
            expected = {
                (mode_n, mode_r, int(delta), int(seed))
                for mode_n, mode_r, delta, seed in itertools.product(
                    factors.get("node_ptx_mode", []), factors.get("relay_ptx_mode", []),
                    factors.get("terrain_delta_h", []), seeds or [],
                )
            }
        elif isinstance(matrix, list):
            expected = set()
            for row in matrix:
                try:
                    expected.add((
                        row["node_ptx_mode"], row["relay_ptx_mode"],
                        int(row["terrain_delta_h"]), int(row["seed"]),
                    ))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("Terrain run_matrix contains an invalid row") from exc
            if len(expected) != len(matrix):
                raise ValueError("Terrain run_matrix contains duplicate rows")
        else:
            expected = {
                (mode_n, mode_r, int(delta), int(seed))
                for mode_n, mode_r, delta, seed in itertools.product(
                    factors.get("node_ptx_mode", []), factors.get("relay_ptx_mode", []),
                    factors.get("terrain_delta_h", []), seeds or [],
                )
            }
        actual_dirs = {
            path.name for path in paths.runs.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        } if paths.runs.is_dir() else set()
        if actual_dirs != set(run_ids):
            raise ValueError(
                "Terrain run directories do not match the manifest: "
                f"missing={sorted(set(run_ids) - actual_dirs)[:10]}, "
                f"unexpected={sorted(actual_dirs - set(run_ids))[:10]}"
            )
        observed = set()
        declared_statuses = manifest.get("run_statuses", {})
        if not isinstance(declared_statuses, dict):
            raise ValueError("Terrain manifest run_statuses must be an object")
        for run_id in run_ids:
            run_dir = paths.runs / run_id
            run_status_path = run_dir / "status.json"
            if not run_status_path.is_file():
                raise ValueError(f"Missing run status for {run_id}")
            run_status = _read_json(run_status_path, "terrain run status")
            if run_status.get("run_id") != run_id or run_status.get("status") != "PASS":
                raise ValueError(f"Terrain run is not complete and passing: {run_id}")
            if declared_statuses.get(run_id) != "PASS":
                raise ValueError(f"Terrain cohort manifest does not record PASS for {run_id}")
            resolved_path = run_dir / "resolved_config.json"
            if not resolved_path.is_file():
                raise ValueError(f"Missing resolved configuration for {run_id}")
            resolved = _read_json(resolved_path, "resolved run configuration")
            if resolved.get("run_id") != run_id:
                raise ValueError(f"Resolved run ID mismatch for {run_id}")
            factors_run = resolved.get("factors", {})
            config_run = resolved.get("scientific_config", {})
            cell = (
                factors_run.get("node_ptx_mode"), factors_run.get("relay_ptx_mode"),
                int(factors_run.get("terrain_delta_h", -1)), int(resolved.get("seed", -1)),
            )
            if cell not in expected or cell in observed:
                raise ValueError(f"Unexpected or duplicate terrain condition/seed: {run_id}")
            if (config_run.get("node_ptx_mode") != cell[0]
                    or config_run.get("relay_ptx_mode") != cell[1]
                    or int(config_run.get("terrain_delta_h", -1)) != cell[2]
                    or int(config_run.get("seed", -1)) != cell[3]):
                raise ValueError(f"Resolved terrain modes conflict with factors: {run_id}")
            if profile == APC_TERRAIN_PROFILE_DIAGNOSTIC_PROFILE:
                violations = diagnostic_config_violations(config_run)
                if violations:
                    raise ValueError(f"Resolved diagnostic contract failed for {run_id}: {violations}")
            expected_run_id = _terrain_run_id(manifest, resolved, factors)
            if expected_run_id != run_id:
                raise ValueError(f"Resolved run identity/hash mismatch for {run_id}")
            _read_verified_scientific_tables(run_dir)
            observed.add(cell)
        if observed != expected:
            raise ValueError("Terrain resolved configurations do not cover the configured grid")
    return status


def _terrain_run_id(
    manifest: dict[str, Any], resolved: dict[str, Any], factors: dict[str, Any]
) -> str:
    """Reconstruct the pipeline run ID from resolved scientific provenance."""
    factor_values = resolved.get("factors", {})
    factor_slug = "__".join(
        f"{name}-{str(factor_values[name]).replace(' ', '')}"
        for name in factors
    )[:120].rstrip("-_")
    digest = content_hash(resolved.get("scientific_config", {}), length=10)
    return (
        f"{manifest.get('cohort_id')}__{factor_slug}"
        f"__s{resolved.get('seed')}__{digest}"
    )


def build_terrain_sweep_artifacts(
    config: str | Path = DEFAULT_CONFIG,
    *,
    experiment_dir: str | Path | None = None,
    terrain_delta_h: float = DEFAULT_COMPARISON_TERRAIN_DELTA_H,
) -> dict[str, Any]:
    """Build experiment-owned tables and figures without executing a simulation."""
    paths = terrain_sweep_paths(config, experiment_dir=experiment_dir)
    status = _validate_completed_sweep(paths)
    publisher = AnalysisPublisher(paths.experiment_dir) if paths.is_sweep_standard else None
    tables_destination = publisher.tables_dir if publisher else paths.tables
    figures_destination = publisher.figures_dir if publisher else paths.figures
    table_files = build_publication_tables(
        paths.experiment_dir,
        publication_dir=tables_destination if paths.is_sweep_standard else paths.experiment_dir,
        terrain_delta_h=terrain_delta_h,
        manuscript_names=False,
    )
    figure_files = generate_all_figures(
        paths.experiment_dir,
        publication_dir=figures_destination if paths.is_sweep_standard else paths.experiment_dir,
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
    manifest_path = paths.analysis_manifest
    aggregate = paths.aggregate / "core_metrics_long.csv"
    analysis_manifest = {
            "analysis_manifest_version": TERRAIN_SWEEP_ANALYSIS_MANIFEST_VERSION,
            "status": "COMPLETE" if paths.is_sweep_standard else "PASS",
            "analysis_id": f"{status.get('experiment_id')}-analysis-v1",
            "analysis_version": 1,
            "source_cohort_id": status.get("experiment_id"),
            "experiment_status_sha256": file_sha256(paths.status),
            "core_metrics_sha256": file_sha256(aggregate),
            "comparison_terrain_delta_h": float(terrain_delta_h),
            "metrics": {
                "pdr_percent": "per-run delivered packets / offered packets * 100",
                "energy_per_delivered_packet_J": "per-run corrected network energy / delivered packets",
                "aggregation": "seed-level means and Student-t confidence intervals per frozen analysis contract",
            },
            "grouping_dimensions": ["node_ptx_mode", "relay_ptx_mode", "terrain_delta_h"],
            "seed_aggregation": {
                "method": "arithmetic mean across configured seeds",
                "confidence_interval": "two-sided Student-t, df=n-1",
                "confidence_level": 0.95,
            },
            "included_model_variants": ["AA", "AF", "FA", "FF"],
            "generator": {
                "path": Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[4]).as_posix(),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
            "artifact_counts": {
                "tables": sum(path.suffix.lower() == ".csv" for path in generated),
                "figures": sum(path.suffix.lower() in {".png", ".pdf", ".svg"} for path in generated),
                "total": len(generated),
            },
            "inputs": [
                {
                    "path": aggregate.relative_to(paths.experiment_dir).as_posix(),
                    "size_bytes": aggregate.stat().st_size,
                    "sha256": file_sha256(aggregate),
                }
            ],
            "artifacts": records,
        }
    if not paths.is_sweep_standard:
        analysis_manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    else:
        analysis_manifest["inputs"] = [
            {
                "path": path.relative_to(paths.experiment_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted([paths.status, *paths.aggregate.glob("*.csv")])
            if path.is_file()
        ]
    if publisher is not None:
        reused = publisher.find_reusable(analysis_manifest)
        if reused is not None:
            experiment_manifest = _read_json(paths.manifest, "terrain-sweep experiment manifest")
            relative_manifest = Path(reused["analysis_manifest"]).relative_to(
                paths.experiment_dir
            ).as_posix()
            if (experiment_manifest.get("analysis_status") != "COMPLETE"
                    or experiment_manifest.get("analysis_manifest_path") != relative_manifest):
                experiment_manifest["analysis_status"] = "COMPLETE"
                experiment_manifest["analysis_manifest_path"] = relative_manifest
                atomic_write_json(paths.manifest, experiment_manifest)
            publisher.close()
            return reused
        result = publisher.publish(analysis_manifest, generated)
        manifest_path = Path(result["analysis_manifest"])
    else:
        atomic_write_json(manifest_path, analysis_manifest)
    if paths.is_sweep_standard:
        experiment_manifest = _read_json(paths.manifest, "terrain-sweep experiment manifest")
        manifest_relative = manifest_path.relative_to(paths.experiment_dir).as_posix()
        if (experiment_manifest.get("analysis_status") != "COMPLETE"
                or experiment_manifest.get("analysis_manifest_path") != manifest_relative):
            experiment_manifest["analysis_status"] = "COMPLETE"
            experiment_manifest["analysis_manifest_path"] = manifest_relative
            atomic_write_json(paths.manifest, experiment_manifest)
    if publisher is not None:
        publisher.close()
        generated = [paths.experiment_dir / path for path in result["tables"] + result["figures"]]
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_terrain_sweep_artifacts(
        args.config, experiment_dir=args.experiment_dir
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
