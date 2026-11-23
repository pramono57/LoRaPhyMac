"""Versioned, no-overwrite cohort creation for future Sweep executions."""

from __future__ import annotations

import json
import hashlib
import re
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from loraphymac.experiments.common.outputs import SWEEP_OUTPUT_LAYOUT_VERSION


SWEEP_DESIGN_VERSION = 1
PROJECT_TIMEZONE = "Asia/Jakarta"
RESERVATION_FILE = ".cohort_reservation.json"


def _validate_external_references(value: Any, project_root: Path) -> None:
    """Validate declared external path/hash pairs before reserving a cohort."""
    if isinstance(value, Mapping):
        raw_path = value.get("path")
        expected = value.get("sha256")
        if isinstance(raw_path, str) and isinstance(expected, str):
            path = Path(raw_path)
            path = path if path.is_absolute() else project_root / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Required external reference is missing: {path}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"External reference SHA-256 mismatch: {path}")
        for child in value.values():
            _validate_external_references(child, project_root)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_external_references(child, project_root)


def reserve_sweep_cohort(
    spec: Any,
    *,
    project_root: str | Path,
    scenario: str,
    output_root: str | Path | None = None,
    design_version: int = SWEEP_DESIGN_VERSION,
    now: datetime | None = None,
) -> Any:
    """Atomically reserve a new cohort path and return a cohort-specific spec.

    An existing cohort is never reused. The returned spec is authorized to
    initialize only the empty directory reserved by this call.
    """
    normalized = re.sub(r"[^a-z0-9]+", "_", scenario.lower()).strip("_")
    if not normalized:
        raise ValueError("scenario must contain at least one alphanumeric character")
    instant = (now or datetime.now(ZoneInfo(PROJECT_TIMEZONE))).astimezone(
        ZoneInfo(PROJECT_TIMEZONE)
    )
    date_text = instant.strftime("%Y%m%d")
    resolved_root = Path(project_root).resolve()
    external_references = dict(getattr(spec, "external_references", {}))
    _validate_external_references(external_references, resolved_root)
    input_config_path = getattr(spec, "input_config_path", None)
    portable_config_path = None
    if input_config_path is not None:
        config_path = Path(input_config_path).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Required scenario configuration is missing: {config_path}")
        try:
            portable_config_path = config_path.relative_to(resolved_root).as_posix()
        except ValueError:
            portable_config_path = str(config_path)
        external_references["scenario_configuration"] = {
            "path": portable_config_path,
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        }
    artifact_root = Path(output_root).expanduser().resolve() if output_root is not None else (
        resolved_root / "artifacts" / "experiments"
    )
    cohort_root = artifact_root / f"APC_{normalized.upper()}"
    cohort_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{normalized}_v{int(design_version)}_r"
    rerun = 1
    while True:
        cohort_id = f"{prefix}{rerun:02d}_{date_text}"
        target = cohort_root / cohort_id
        execution_id = str(uuid.uuid4())
        try:
            target.mkdir()
            break
        except FileExistsError:
            rerun += 1
    reservation = {
        "cohort_id": cohort_id,
        "execution_id": execution_id,
        "created_at": instant.isoformat(),
        "timezone": PROJECT_TIMEZONE,
    }
    reservation_path = target / RESERVATION_FILE
    try:
        reservation_path.write_text(
            json.dumps(reservation, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as exc:
        failure = {
            "cohort_id": cohort_id,
            "execution_id": execution_id,
            "status": "FAILED",
            "failure": f"{type(exc).__name__}: {exc}",
        }
        try:
            (target / "reservation_status.json").write_text(
                json.dumps(failure, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
        except OSError:
            pass
        raise
    return replace(
        spec,
        experiment_id=cohort_id,
        output_dir=cohort_root,
        output_layout_version=SWEEP_OUTPUT_LAYOUT_VERSION,
        design_version=int(design_version),
        rerun_number=rerun,
        execution_id=execution_id,
        created_at=instant.isoformat(),
        timezone=PROJECT_TIMEZONE,
        cohort_reservation_path=reservation_path,
        external_references=external_references,
    )


def validate_cohort_reservation(spec: Any) -> Mapping[str, Any]:
    """Validate that a standard spec owns its still-empty reserved directory."""
    reservation_path = getattr(spec, "cohort_reservation_path", None)
    if reservation_path is None:
        raise ValueError(
            "Sweep standard cohorts must be created with reserve_sweep_cohort(); "
            "existing directories cannot be resumed or overwritten"
        )
    reservation_path = Path(reservation_path)
    cohort_dir = Path(spec.output_dir) / spec.experiment_id
    if reservation_path != cohort_dir / RESERVATION_FILE:
        raise ValueError("cohort reservation path does not match the spec output path")
    if not reservation_path.is_file() or reservation_path.is_symlink():
        raise ValueError("cohort reservation is missing or is not a regular file")
    payload = json.loads(reservation_path.read_text(encoding="utf-8"))
    if payload.get("execution_id") != getattr(spec, "execution_id", None):
        raise ValueError("cohort reservation execution identity mismatch")
    unexpected = [path.name for path in cohort_dir.iterdir() if path != reservation_path]
    if unexpected:
        raise FileExistsError(
            f"Reserved cohort directory is no longer empty: {sorted(unexpected)}"
        )
    return payload


def consume_cohort_reservation(spec: Any) -> None:
    """Remove the one-use reservation marker immediately before manifest init."""
    path = Path(spec.cohort_reservation_path)
    validate_cohort_reservation(spec)
    path.unlink()


def standard_manifest_fields(
    spec: Any,
    *,
    expected_run_count: int,
    run_ids: list[str],
    git_head: str | None,
) -> dict[str, Any]:
    return {
        "scenario": Path(spec.output_dir).name.removeprefix("APC_").lower(),
        "cohort_id": spec.experiment_id,
        "design_version": int(spec.design_version),
        "rerun_number": int(spec.rerun_number),
        "execution_id": spec.execution_id,
        "created_at": spec.created_at,
        "timezone": spec.timezone,
        "parameter_grid": {
            "factors": {key: list(values) for key, values in spec.factors.items()},
            "seeds": list(spec.seeds),
            "run_matrix": (
                [dict(row) for row in spec.run_matrix]
                if spec.run_matrix is not None
                else None
            ),
        },
        "expected_run_count": int(expected_run_count),
        "completed_run_count": 0,
        "failed_run_count": 0,
        "incomplete_run_count": int(expected_run_count),
        "run_ids": run_ids,
        "run_statuses": {run_id: "PENDING" for run_id in run_ids},
        "sweep_profile": str(getattr(spec, "run_annotations", {}).get("sweep_profile", "production")),
        "is_non_production_pilot": getattr(spec, "run_annotations", {}).get("sweep_profile") == "pilot",
        "git_head": git_head,
        "external_references": dict(getattr(spec, "external_references", {})),
        "output_schema_version": SWEEP_OUTPUT_LAYOUT_VERSION,
        "aggregate_status": "PENDING",
        "analysis_status": "PENDING",
    }


def write_cohort_readme(
    cohort_dir: Path,
    *,
    spec: Any,
    run_count: int,
    seed_list: list[int],
    source_fingerprint_hash: str,
) -> None:
    """Write a factual README for a completed cohort, without replay claims."""
    factor_lines = [
        f"- `{key}`: {', '.join(map(str, values))}"
        for key, values in spec.factors.items()
    ] or ["- Explicit run matrix (see `experiment_manifest.json`)."]
    if spec.run_matrix is not None:
        dimension_values: dict[str, set[str]] = {}
        for row in spec.run_matrix:
            for name, value in row.get("dimensions", {}).items():
                dimension_values.setdefault(str(name), set()).add(str(value))
        for name, values in sorted(dimension_values.items()):
            factor_lines.append(f"- `{name}`: {', '.join(sorted(values))}")
    profile = str(spec.base_config.get("experiment_profile", ""))
    analysis_commands = {
        "apc_load_sweep_standard_v1": (
            "python -m loraphymac.experiments.apc_load_sweep.standard_analysis <cohort-directory>"
        ),
        "apc_load_sweep_pilot_v1": (
            "python -m loraphymac.experiments.apc_load_sweep.standard_analysis <cohort-directory>"
        ),
        "apc_feedback_robustness_sweep_v1": (
            "python -m loraphymac.experiments.apc_feedback_robustness_sweep.standard_analysis <cohort-directory>"
        ),
        "apc_terrain_sweep_v1": (
            "python -m loraphymac.experiments.apc_terrain_sweep.postprocessing "
            "--experiment-dir <cohort-directory>"
        ),
    }
    analysis_command = analysis_commands.get(
        profile,
        "python -m loraphymac.experiments.apc_compare_model_terrain_sweep.postprocessing "
        "<archived-standard-config> --experiment-dir <cohort-directory>",
    )
    content = "\n".join(
        [
            f"# {spec.experiment_id}",
            "",
            f"Scenario: `{spec.base_config.get('experiment_profile', 'unspecified')}`.",
            f"Selected sweep profile: `{getattr(spec, 'run_annotations', {}).get('sweep_profile', 'production')}`.",
            *(
                ["**Non-production integration evidence. Do not use as publication results.**"]
                if getattr(spec, "run_annotations", {}).get("sweep_profile") == "pilot"
                else []
            ),
            f"Design version: {spec.design_version}; cohort: `{spec.experiment_id}`.",
            "",
            "## Parameter grid",
            *factor_lines,
            f"- Seeds: {', '.join(map(str, seed_list))}",
            f"- Expected runs: {run_count}",
            f"- Actual runs: {run_count}",
            "",
            "## Source identity",
            f"- Git HEAD: `{_manifest_git_head(cohort_dir) or 'unavailable'}`",
            f"- Dirty source fingerprint SHA-256: `{source_fingerprint_hash}`",
            "",
            "## Directory structure",
            "- `runs/<run_id>/`: resolved configuration, scientific output manifest, tables, and run-local logs.",
            "- `aggregate/`: all-run core summary and aggregate statistics.",
            "- `analysis/figures/`: cohort analysis figures.",
            "- `analysis/tables/`: cohort analysis tables.",
            "- `analysis/analysis_manifest.json`: analysis provenance and completion state.",
            "",
            "## Reproduction",
            "Run the scenario's runner with its archived configuration and source environment to repeat the simulation as a new cohort. For postprocessing, run:",
            "",
            f"`{analysis_command}`",
            "",
            "This README records the source identity; it does not claim full source replay reproducibility.",
            "",
            "Analysis outputs are recorded in `analysis/analysis_manifest.json` when generated.",
            "",
        ]
    )
    (cohort_dir / "README.md").write_text(content, encoding="utf-8")


def _manifest_git_head(cohort_dir: Path) -> str | None:
    path = cohort_dir / "experiment_manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("git_head")
    except (OSError, ValueError, TypeError):
        return None


__all__ = [
    "PROJECT_TIMEZONE",
    "SWEEP_DESIGN_VERSION",
    "consume_cohort_reservation",
    "reserve_sweep_cohort",
    "standard_manifest_fields",
    "validate_cohort_reservation",
    "write_cohort_readme",
]
