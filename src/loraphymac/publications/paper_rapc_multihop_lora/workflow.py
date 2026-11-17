"""Audited multi-experiment R-APC multi-hop LoRa publication workflow."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from loraphymac.experiments.apc_closed_loop_verification.contract import (
    APC_CLOSED_LOOP_CONTRACT_VERSION,
    APC_CLOSED_LOOP_PROFILE,
)
from loraphymac.experiments.apc_terrain_sweep.contract import (
    APC_TERRAIN_SWEEP_CONTRACT_VERSION,
    APC_TERRAIN_SWEEP_EXPECTED_RUN_COUNT,
    APC_TERRAIN_SWEEP_PROFILE,
    PAPER1_APC_CONTRACT_VERSION,
    PAPER1_APC_PROFILE,
)
from loraphymac.experiments.common.multiseed_pipeline import file_sha256
from loraphymac.experiments.common.outputs import ExperimentArtifactPaths
from loraphymac.paths import repository_root
from loraphymac.publications import common as publication_common_module
from loraphymac.publications import provenance as provenance_module
from loraphymac.publications.common import PublicationArtifactPaths, PublicationSpec
from loraphymac.publications.paper_rapc_multihop_lora import analysis as analysis_module
from loraphymac.publications.paper_rapc_multihop_lora import figures as figures_module
from loraphymac.publications.paper_rapc_multihop_lora.analysis import (
    DEFAULT_CONFIG,
    build_publication_tables,
    publication_paths,
    publication_sources,
    publication_spec,
)
from loraphymac.publications.paper_rapc_multihop_lora.figures import generate_all_figures
from loraphymac.publications.provenance import (
    PublicationProvenanceError,
    ValidatedExperimentEvidence,
    build_publication_manifest,
    validate_completed_experiment,
    write_publication_manifest,
)


RAPC_MULTIHOP_LORA_PUBLICATION_WORKFLOW_VERSION = (
    "paper_rapc_multihop_lora_publication_v2_multi_experiment"
)
PAPER1_SOURCE_CONTRACT_VERSIONS = {
    APC_TERRAIN_SWEEP_PROFILE: APC_TERRAIN_SWEEP_CONTRACT_VERSION,
    PAPER1_APC_PROFILE: PAPER1_APC_CONTRACT_VERSION,
}

TERRAIN_SOURCE_TO_PUBLICATION = {
    "tables/terrain_apc_performance_delta_h_90.csv": (
        "tables/table_09_apc_performance_delta_h_90.csv"
    ),
    "tables/terrain_apc_performance_delta_h_90_long.csv": (
        "tables/table_09_apc_performance_delta_h_90_long.csv"
    ),
    "tables/terrain_constrained_energy_delta_h_90.csv": (
        "tables/table_10_constrained_energy_delta_h_90.csv"
    ),
    "figures/terrain_energy_breakdown_delta_h_90.png": (
        "figures/figure_01_energy_breakdown_delta_h_90.png"
    ),
    "figures/terrain_energy_breakdown_delta_h_90.pdf": (
        "figures/figure_01_energy_breakdown_delta_h_90.pdf"
    ),
    "figures/terrain_final_node_ptx.png": "figures/figure_02_final_node_ptx.png",
    "figures/terrain_final_node_ptx.pdf": "figures/figure_02_final_node_ptx.pdf",
    "figures/terrain_final_relay_ptx.png": "figures/figure_03_final_relay_ptx.png",
    "figures/terrain_final_relay_ptx.pdf": "figures/figure_03_final_relay_ptx.pdf",
    "figures/terrain_pdr.png": "figures/figure_04_pdr.png",
    "figures/terrain_pdr.pdf": "figures/figure_04_pdr.pdf",
    "figures/terrain_energy_per_delivered_packet.png": (
        "figures/figure_05_energy_per_delivered_packet.png"
    ),
    "figures/terrain_energy_per_delivered_packet.pdf": (
        "figures/figure_05_energy_per_delivered_packet.pdf"
    ),
    "figures/terrain_delay.png": "figures/figure_06_delay.png",
    "figures/terrain_delay.pdf": "figures/figure_06_delay.pdf",
}

CLOSED_LOOP_SOURCE_TO_PUBLICATION = {
    "tables/table_iv_representative_apc_decision_validation.csv": (
        "tables/table_04_representative_apc_decision_validation.csv"
    ),
    "figures/fig_single_node_margin_convergence.png": (
        "figures/figure_07_node_margin_convergence.png"
    ),
    "figures/fig_single_node_margin_convergence.pdf": (
        "figures/figure_07_node_margin_convergence.pdf"
    ),
    "figures/fig_single_relay_margin_convergence.png": (
        "figures/figure_08_relay_margin_convergence.png"
    ),
    "figures/fig_single_relay_margin_convergence.pdf": (
        "figures/figure_08_relay_margin_convergence.pdf"
    ),
    "figures/fig_single_node_ptx_next_tx.png": "figures/figure_09_node_ptx_next_tx.png",
    "figures/fig_single_node_ptx_next_tx.pdf": "figures/figure_09_node_ptx_next_tx.pdf",
    "figures/fig_single_relay_ptx_next_tx.png": "figures/figure_10_relay_ptx_next_tx.png",
    "figures/fig_single_relay_ptx_next_tx.pdf": "figures/figure_10_relay_ptx_next_tx.pdf",
}


@dataclass(frozen=True)
class ValidatedClosedLoopEvidence:
    """Validated one-run closed-loop evidence selected for publication."""

    experiment_dir: Path
    report_path: Path
    report: Mapping[str, Any]
    selected_artifacts: tuple[Path, ...]


@dataclass(frozen=True)
class ValidatedPublicationSources:
    """Role-labelled evidence for one R-APC multi-experiment publication."""

    terrain_sweep: ValidatedExperimentEvidence
    closed_loop_verification: ValidatedClosedLoopEvidence | None

    @property
    def paths(self) -> ExperimentArtifactPaths:
        return self.terrain_sweep.paths

    @property
    def status(self) -> Mapping[str, Any]:
        return self.terrain_sweep.status


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PublicationProvenanceError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationProvenanceError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PublicationProvenanceError(f"{label} must contain a JSON object: {path}")
    return value


def _validate_selected_files(root: Path, relative_paths: Mapping[str, str]) -> tuple[Path, ...]:
    selected = tuple(root / relative for relative in relative_paths)
    missing = [str(path) for path in selected if not path.is_file()]
    if missing:
        raise PublicationProvenanceError(
            "Required experiment-owned publication artifacts are missing: " + ", ".join(missing)
        )
    return selected


def _validate_terrain_analysis(evidence: ValidatedExperimentEvidence) -> tuple[Path, ...]:
    manifest_path = evidence.paths.experiment_dir / "analysis_manifest.json"
    manifest = _read_json_object(manifest_path, "terrain-sweep analysis manifest")
    if manifest.get("status") != "PASS":
        raise PublicationProvenanceError("Terrain-sweep analysis manifest status must be PASS")
    if manifest.get("experiment_id") != evidence.status.get("experiment_id"):
        raise PublicationProvenanceError(
            "Terrain-sweep analysis manifest experiment_id does not match experiment status"
        )
    if manifest.get("core_metrics_sha256") != file_sha256(evidence.core_metrics):
        raise PublicationProvenanceError(
            "Terrain-sweep analysis manifest does not match the current core aggregate"
        )
    selected = _validate_selected_files(
        evidence.paths.experiment_dir,
        TERRAIN_SOURCE_TO_PUBLICATION,
    )
    records = {
        str(record.get("path")): record
        for record in manifest.get("artifacts", [])
        if isinstance(record, Mapping)
    }
    for path in selected:
        relative = path.relative_to(evidence.paths.experiment_dir).as_posix()
        record = records.get(relative)
        if record is None or record.get("sha256") != file_sha256(path):
            raise PublicationProvenanceError(
                f"Terrain-sweep analysis artifact hash is missing or stale: {relative}"
            )
    return selected


def _validate_closed_loop_source(path: Path) -> ValidatedClosedLoopEvidence:
    if not path.is_dir():
        raise PublicationProvenanceError(
            f"Closed-loop source experiment directory does not exist: {path}"
        )
    report_path = path / "verification_report.json"
    report = _read_json_object(report_path, "closed-loop verification report")
    if report.get("status") != "PASS":
        raise PublicationProvenanceError("Closed-loop verification status must be PASS")
    if report.get("experiment_profile") != APC_CLOSED_LOOP_PROFILE:
        raise PublicationProvenanceError("Closed-loop verification profile is not supported")
    if report.get("experiment_contract_version") != APC_CLOSED_LOOP_CONTRACT_VERSION:
        raise PublicationProvenanceError("Closed-loop verification contract version is invalid")
    if report.get("seed") != 42 or float(report.get("terrain_delta_h", -1)) != 90.0:
        raise PublicationProvenanceError(
            "Closed-loop publication evidence must remain seed 42 at terrain_delta_h=90"
        )
    checks = report.get("checks")
    if not isinstance(checks, Mapping):
        raise PublicationProvenanceError("Closed-loop verification checks are missing")
    if checks.get("closed_loop_contract") is not True or checks.get("stage1_status") != "PASS":
        raise PublicationProvenanceError("Closed-loop verification contract did not pass")
    for invariant in (
        "packet_fate_conservation_delta",
        "collision_outcome_conservation_delta",
    ):
        if checks.get(invariant) != 0:
            raise PublicationProvenanceError(
                f"Closed-loop verification invariant {invariant} must be zero"
            )
    selected = _validate_selected_files(path, CLOSED_LOOP_SOURCE_TO_PUBLICATION)
    return ValidatedClosedLoopEvidence(
        experiment_dir=path,
        report_path=report_path,
        report=report,
        selected_artifacts=selected,
    )


def validate_publication_source(
    config: str | Path = DEFAULT_CONFIG,
) -> tuple[PublicationSpec, PublicationArtifactPaths, ValidatedPublicationSources]:
    """Validate all configured source experiments without writing output."""
    spec = publication_spec(config)
    output_paths = publication_paths(config)
    sources = publication_sources(config)
    terrain_dir = sources.get("terrain_sweep")
    if terrain_dir is None:
        raise PublicationProvenanceError("Publication sources must include terrain_sweep")
    terrain = validate_completed_experiment(
        ExperimentArtifactPaths(terrain_dir),
        expected_run_count=APC_TERRAIN_SWEEP_EXPECTED_RUN_COUNT,
        contract_versions=PAPER1_SOURCE_CONTRACT_VERSIONS,
    )
    closed_loop: ValidatedClosedLoopEvidence | None = None
    if spec.sources:
        closed_loop_dir = sources.get("closed_loop_verification")
        if closed_loop_dir is None:
            raise PublicationProvenanceError(
                "Multi-experiment publication sources must include closed_loop_verification"
            )
        _validate_terrain_analysis(terrain)
        closed_loop = _validate_closed_loop_source(closed_loop_dir)
    return (
        spec,
        output_paths,
        ValidatedPublicationSources(
            terrain_sweep=terrain,
            closed_loop_verification=closed_loop,
        ),
    )


def _generator_files() -> tuple[Path, ...]:
    return (
        Path(__file__),
        Path(analysis_module.__file__),
        Path(figures_module.__file__),
        Path(publication_common_module.__file__),
        Path(provenance_module.__file__),
        Path(__file__).with_name("run.py"),
    )


def _base_summary(
    spec: PublicationSpec,
    output_paths: PublicationArtifactPaths,
    evidence: ValidatedPublicationSources,
) -> dict[str, Any]:
    status = evidence.terrain_sweep.status
    source_summary: dict[str, Any] = {
        "terrain_sweep": {
            "path": str(evidence.terrain_sweep.paths.experiment_dir),
            "experiment_id": status["experiment_id"],
            "status": status["status"],
            "run_count": status["passed_run_count_on_disk"],
        }
    }
    if evidence.closed_loop_verification is not None:
        report = evidence.closed_loop_verification.report
        source_summary["closed_loop_verification"] = {
            "path": str(evidence.closed_loop_verification.experiment_dir),
            "run_id": report.get("run_id"),
            "status": report.get("status"),
            "seed": report.get("seed"),
            "terrain_delta_h": report.get("terrain_delta_h"),
        }
    return {
        "publication_id": spec.publication_id,
        "source_experiments": source_summary,
        "source_experiment_dir": str(evidence.terrain_sweep.paths.experiment_dir),
        "source_experiment_id": status["experiment_id"],
        "requested_run_count": status["requested_run_count"],
        "passed_run_count_on_disk": status["passed_run_count_on_disk"],
        "seed_grid_complete": status["seed_grid_complete"],
        "publication_dir": str(output_paths.publication_dir),
    }


def _artifact_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "format": path.suffix.lower().lstrip("."),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _copy_selected_artifacts(
    *,
    output_paths: PublicationArtifactPaths,
    source_root: Path,
    mapping: Mapping[str, str],
    source_role: str,
) -> tuple[list[Path], dict[Path, dict[str, Any]]]:
    generated: list[Path] = []
    lineage: dict[Path, dict[str, Any]] = {}
    for source_relative, output_relative in mapping.items():
        source = source_root / source_relative
        destination = output_paths.publication_dir / output_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        generated.append(destination)
        lineage[destination.resolve()] = {
            "source_role": source_role,
            "source_path": source_relative,
            "source_sha256": file_sha256(source),
        }
    return generated, lineage


def _multi_source_manifest(
    *,
    spec: PublicationSpec,
    output_paths: PublicationArtifactPaths,
    evidence: ValidatedPublicationSources,
    generated_files: list[Path],
    lineage: Mapping[Path, Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = build_publication_manifest(
        spec=spec,
        evidence=evidence.terrain_sweep,
        output_paths=output_paths,
        artifacts=generated_files,
        workflow_version=RAPC_MULTIHOP_LORA_PUBLICATION_WORKFLOW_VERSION,
        generator_module="loraphymac.publications.paper_rapc_multihop_lora.workflow",
        generator_files=_generator_files(),
    )
    closed = evidence.closed_loop_verification
    if closed is None:
        return manifest
    terrain_record = dict(manifest["source_experiment"])
    terrain_record["analysis_manifest_sha256"] = file_sha256(
        evidence.terrain_sweep.paths.experiment_dir / "analysis_manifest.json"
    )
    terrain_record["selected_artifacts"] = [
        _artifact_record(evidence.terrain_sweep.paths.experiment_dir, path)
        for path in _validate_selected_files(
            evidence.terrain_sweep.paths.experiment_dir,
            TERRAIN_SOURCE_TO_PUBLICATION,
        )
    ]
    root = repository_root(spec.config_path)
    try:
        closed_path = closed.experiment_dir.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        closed_path = str(closed.experiment_dir.resolve())
    closed_record = {
        "path": closed_path,
        "experiment_profile": closed.report["experiment_profile"],
        "experiment_contract_version": closed.report["experiment_contract_version"],
        "run_id": closed.report["run_id"],
        "status": closed.report["status"],
        "seed": closed.report["seed"],
        "terrain_delta_h": closed.report["terrain_delta_h"],
        "simulator_version": closed.report.get("simulator_version"),
        "simulator_api_version": closed.report.get("simulator_api_version"),
        "resolved_config_hash": closed.report.get("resolved_config_hash"),
        "verification_report_sha256": file_sha256(closed.report_path),
        "selected_artifacts": [
            _artifact_record(closed.experiment_dir, path) for path in closed.selected_artifacts
        ],
    }
    manifest["source_experiments"] = {
        "terrain_sweep": terrain_record,
        "closed_loop_verification": closed_record,
    }
    for record in manifest["artifacts"]:
        destination = (output_paths.publication_dir / record["path"]).resolve()
        record.update(lineage[destination])
    return manifest


def run_publication(
    config: str | Path = DEFAULT_CONFIG,
    *,
    check_only: bool = False,
) -> dict[str, Any]:
    """Validate and assemble all configured R-APC publication artifacts."""
    spec, output_paths, evidence = validate_publication_source(config)
    summary = _base_summary(spec, output_paths, evidence)
    if check_only:
        return {
            **summary,
            "status": "READY",
            "check_only": True,
        }

    output_paths.manifest.unlink(missing_ok=True)
    closed = evidence.closed_loop_verification
    if closed is None:
        table_files = build_publication_tables(config=config)
        figure_files = generate_all_figures(config=config)
        generated_files = [*set(table_files.values()), *figure_files]
        lineage: dict[Path, dict[str, Any]] = {}
    else:
        output_paths.ensure()
        terrain_files, terrain_lineage = _copy_selected_artifacts(
            output_paths=output_paths,
            source_root=evidence.terrain_sweep.paths.experiment_dir,
            mapping=TERRAIN_SOURCE_TO_PUBLICATION,
            source_role="terrain_sweep",
        )
        closed_files, closed_lineage = _copy_selected_artifacts(
            output_paths=output_paths,
            source_root=closed.experiment_dir,
            mapping=CLOSED_LOOP_SOURCE_TO_PUBLICATION,
            source_role="closed_loop_verification",
        )
        generated_files = [*terrain_files, *closed_files]
        lineage = {**terrain_lineage, **closed_lineage}

    manifest = _multi_source_manifest(
        spec=spec,
        output_paths=output_paths,
        evidence=evidence,
        generated_files=generated_files,
        lineage=lineage,
    )
    manifest_path = write_publication_manifest(output_paths, manifest)
    return {
        **summary,
        "status": "PASS",
        "check_only": False,
        "table_count": manifest["artifact_counts"]["tables"],
        "figure_count": manifest["artifact_counts"]["figures"],
        "artifact_count": manifest["artifact_counts"]["total"],
        "publication_manifest": str(manifest_path),
    }


__all__ = [
    "CLOSED_LOOP_SOURCE_TO_PUBLICATION",
    "PAPER1_SOURCE_CONTRACT_VERSIONS",
    "RAPC_MULTIHOP_LORA_PUBLICATION_WORKFLOW_VERSION",
    "TERRAIN_SOURCE_TO_PUBLICATION",
    "ValidatedClosedLoopEvidence",
    "ValidatedPublicationSources",
    "run_publication",
    "validate_publication_source",
]
