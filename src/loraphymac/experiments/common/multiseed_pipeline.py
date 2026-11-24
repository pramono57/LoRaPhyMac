from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import platform
import re
import subprocess
import sys
import traceback
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from loraphymac.experiments.common.contract_registry import (
    experiment_aggregation_metrics,
    experiment_contract_version,
    experiment_run_tables,
    experiment_table_filenames,
    validate_experiment_grid_contract,
    validate_experiment_spec_contract,
    validate_run_output_contract,
)
from loraphymac.experiments.common.outputs import (
    OUTPUT_LAYOUT_VERSION,
    SWEEP_OUTPUT_LAYOUT_VERSION,
    ExperimentArtifactPaths,
)
from loraphymac.experiments.common.sweep_standard import (
    consume_cohort_reservation,
    standard_manifest_fields,
    validate_cohort_reservation,
    write_cohort_readme,
)
from loraphymac.experiments.common.scientific_output_adapter import (
    persist_multiseed_run,
    persistence_route,
)
from loraphymac.experiments.common.scientific_output_manager import (
    ScientificOutputManager,
    collect_complete_source_provenance,
)
from loraphymac.config.validation import (
    ConfigurationError,
    validate_config_keys,
    validate_seed,
)
from loraphymac.simulator import (
    SIMULATOR_API_VERSION,
    SIMULATOR_VERSION,
    default_config,
    resolve_configuration,
    simulate,
)


PIPELINE_VERSION = "multiseed_pipeline_v5"
AGGREGATION_VERSION = "paper_grade_t_ci_v1"
DEFAULT_CONFIG = default_config()

MASTER_TABLE_MODES = frozenset({"full", "aggregate_inputs"})

PIPELINE_CONTROL_KEYS = {
    "run_id",
    "seed",
    "output_dir",
    "export_summary",
    "export_results",
    "export_debug",
    "export_internal",
    "plot_routing",
    "print_summary",
    "print_detail",
    "print_debug",
    "print_internal",
    "debug_preview_rows",
    "verbose",
}

RUN_TABLES = {
    "packet_fate_summary_df": "packet_fate_summary.csv",
    "packet_fate_df": "packet_fates.csv",
    "relay_tx_queue_metrics_df": "relay_queue_metrics.csv",
    "tx_queue_traffic_class_metrics_df": "queue_traffic_class_metrics.csv",
    "relay_rx_stats_df": "relay_rx_metrics.csv",
    "relay_duty_cycle_df": "relay_duty_cycle.csv",
    "per_channel_duty_cycle_df": "relay_channel_duty_cycle.csv",
    "demodulator_pool_summary_df": "receiver_demod_metrics.csv",
    "collision_summary_df": "receiver_collision_metrics.csv",
    "topology_parent_by_run_df": "topology_parent_by_run.csv",
    "apc_feedback_lifecycle_df": "apc_feedback_lifecycle.csv",
    "tx_airtime_by_entity_mtype_ptx_channel_df": (
        "tx_airtime_by_entity_mtype_ptx_channel.csv"
    ),
}

DEBUG_RUN_TABLES = {
    "event_log_df": "event_log.csv",
    "tx_queue_lifecycle_df": "queue_lifecycle.csv",
    "demodulator_event_log_df": "demodulator_events.csv",
    "collision_event_log_df": "collision_decisions.csv",
    "collision_interference_edges_df": "collision_interference_edges.csv",
}

MASTER_TABLE_NAMES = {filename: f"per_run_{filename}" for filename in RUN_TABLES.values()}
MASTER_TABLE_NAMES["relay_bottleneck_metrics.csv"] = "per_run_relay_bottleneck_metrics.csv"


def master_table_names_for_profile(profile: str | None = None) -> dict[str, str]:
    names = dict(MASTER_TABLE_NAMES)
    for filename in experiment_table_filenames(profile):
        names.setdefault(filename, f"per_run_{filename}")
    return names


CORE_PAPER_METRICS = [
    "delivery_by_horizon_percent",
    "terminal_loss_percent",
    "censored_fraction_percent",
    "packet_fate_offered_count",
    "packet_fate_delivered_count",
    "packet_fate_terminal_drop_count",
    "packet_fate_censored_count",
    "packet_fate_censored_waiting_count",
    "packet_fate_censored_in_service_count",
    "packet_fate_censored_rx_count",
    "packet_fate_censored_in_flight_count",
    "packet_fate_conservation_delta",
    "pdr_percent",
    "served_node_ratio_95_percent",
    "avg_e2e_delay_ms",
    "delivered_delay_p50_ms",
    "delivered_delay_p95_ms",
    "delivered_delay_p99_ms",
    "collision_opportunity_count",
    "collision_drop_decision_count",
    "collision_loss_rate_percent",
    "collision_capture_success_rate_percent",
    "collision_outcome_conservation_delta",
    "relay_fifo_wait_mean_ms_packet_weighted",
    "relay_fifo_wait_p95_ms",
    "relay_time_to_tx_start_mean_ms",
    "relay_system_sojourn_mean_ms",
    "relay_max_waiting_end_count",
    "relay_max_worker_occupied_fraction",
    "total_node_energy_J",
    "total_relay_energy_J",
    "total_gateway_energy_J",
    "total_network_energy_J",
    "energy_per_delivered_packet_J",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, tuple, set)):
        return list(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
        allow_nan=False,
    )


def content_hash(value: Any, length: int = 16) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:length]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                default=_json_default,
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".csv.gz"):
        ScientificOutputManager._atomic_dataframe(path, frame, gzip=True)
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    output_dir: Path
    base_config: Dict[str, Any]
    factors: Dict[str, Sequence[Any]]
    seeds: Sequence[int]
    seed_config_overrides: Mapping[int, Mapping[str, Any]] = field(
        default_factory=dict
    )
    run_annotations: Mapping[str, Any] = field(default_factory=dict)
    retain_debug_tables: bool = False
    confidence_level: float = 0.95
    master_table_mode: str = "full"
    scientific_master_policy: Mapping[str, Any] | None = None
    run_matrix: Sequence[Mapping[str, Any]] | None = None
    output_layout_version: str = OUTPUT_LAYOUT_VERSION
    design_version: int = 1
    rerun_number: int = 1
    execution_id: str | None = None
    created_at: str | None = None
    timezone: str | None = None
    cohort_reservation_path: Path | None = None
    external_references: Mapping[str, Any] = field(default_factory=dict)
    input_config_path: Path | None = None

    def serializable(self) -> dict:
        value = asdict(self)
        value["output_dir"] = str(self.output_dir)
        for runtime_field in (
            "cohort_reservation_path", "execution_id", "created_at", "timezone",
            "rerun_number",
            "input_config_path",
        ):
            value.pop(runtime_field, None)
        value["factors"] = {key: list(values) for key, values in self.factors.items()}
        value["seeds"] = list(self.seeds)
        value["external_references"] = dict(self.external_references)
        if self.run_matrix is not None:
            value["run_matrix"] = [dict(row) for row in self.run_matrix]
        else:
            value.pop("run_matrix", None)
        value["seed_config_overrides"] = {
            str(seed): dict(overrides)
            for seed, overrides in sorted(self.seed_config_overrides.items())
        }
        if not value["seed_config_overrides"]:
            value.pop("seed_config_overrides")
        value["run_annotations"] = dict(self.run_annotations)
        if not value["run_annotations"]:
            value.pop("run_annotations")
        if value.get("scientific_master_policy") is None:
            value.pop("scientific_master_policy")
        return value


def _validate_config_keys(config: Mapping[str, Any], label: str) -> None:
    validate_config_keys(config, DEFAULT_CONFIG, label=label)
    forbidden = set(config).intersection(PIPELINE_CONTROL_KEYS)
    if forbidden:
        raise ValueError(f"{label} contains pipeline-controlled keys: {sorted(forbidden)}")


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Experiment spec must be a JSON object")
    common_required = {"experiment_id", "output_dir", "base_config"}
    missing = common_required.difference(raw)
    if missing:
        raise ValueError(f"Experiment spec missing keys: {sorted(missing)}")
    if "run_matrix" not in raw:
        missing_grid = {"factors", "seeds"}.difference(raw)
        if missing_grid:
            raise ValueError(
                f"Experiment spec missing keys: {sorted(missing_grid)}"
            )
    allowed = common_required | {
        "factors",
        "seeds",
        "run_matrix",
        "retain_debug_tables",
        "confidence_level",
        "master_table_mode",
        "seed_config_overrides",
        "run_annotations",
        "scientific_master_policy",
        "output_layout_version",
        "design_version",
    }
    unknown = set(raw).difference(allowed)
    if unknown:
        raise ValueError(f"Experiment spec contains unknown keys: {sorted(unknown)}")

    if not isinstance(raw["experiment_id"], str):
        raise ValueError("experiment_id must be a string")
    experiment_id = raw["experiment_id"].strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment_id):
        raise ValueError("experiment_id must contain only letters, numbers, _, -, .")
    if len(experiment_id) > 64:
        raise ValueError("experiment_id must be at most 64 characters")
    if not isinstance(raw["output_dir"], str) or not raw["output_dir"].strip():
        raise ValueError("output_dir must be a non-empty path string")
    output_dir = Path(raw["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()

    if not isinstance(raw["base_config"], Mapping):
        raise ValueError("base_config must be a JSON object")
    raw_factors = raw.get("factors", {})
    if not isinstance(raw_factors, Mapping):
        raise ValueError("factors must be a JSON object")
    base_config = dict(raw["base_config"])
    factors = {}
    for key, values in raw_factors.items():
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError(f"factor {key!r} must contain a JSON array of values")
        factors[str(key)] = list(values)
    explicit_matrix = raw.get("run_matrix")
    if not factors and explicit_matrix is None:
        raise ValueError("factors must contain at least one experiment factor")
    if any(not values for values in factors.values()):
        raise ValueError("Every factor must contain at least one value")
    overlap = set(base_config).intersection(factors)
    if overlap:
        raise ValueError(f"Keys cannot appear in base_config and factors: {sorted(overlap)}")
    _validate_config_keys(base_config, "base_config")
    _validate_config_keys({key: None for key in factors}, "factors")

    raw_seeds = raw.get("seeds", [])
    if (
        isinstance(raw_seeds, (str, bytes))
        or not isinstance(raw_seeds, Sequence)
        or (not raw_seeds and explicit_matrix is None)
    ):
        raise ValueError("seeds must be a non-empty list of unique integers")
    seeds = [validate_seed(seed) for seed in raw_seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a non-empty list of unique integers")
    run_matrix = None
    if explicit_matrix is not None:
        if not isinstance(explicit_matrix, Sequence) or isinstance(
            explicit_matrix, (str, bytes)
        ) or not explicit_matrix:
            raise ValueError("run_matrix must be a non-empty array of run objects")
        matrix_rows = []
        expected_keys = set(factors) | {"seed"}
        for index, raw_row in enumerate(explicit_matrix):
            if not isinstance(raw_row, Mapping):
                raise ValueError(f"run_matrix[{index}] must be a JSON object")
            row = dict(raw_row)
            if set(row) != expected_keys:
                raise ValueError(
                    f"run_matrix[{index}] keys must be exactly {sorted(expected_keys)}"
                )
            for key, allowed_values in factors.items():
                if row[key] not in allowed_values:
                    raise ValueError(
                        f"run_matrix[{index}].{key}={row[key]!r} is outside factors"
                    )
            row["seed"] = validate_seed(row["seed"])
            if seeds and row["seed"] not in seeds:
                raise ValueError(
                    f"run_matrix[{index}].seed={row['seed']} is outside seeds"
                )
            matrix_rows.append(row)
        row_keys = [json.dumps(row, sort_keys=True) for row in matrix_rows]
        if len(row_keys) != len(set(row_keys)):
            raise ValueError("run_matrix contains duplicate rows")
        if seeds and {row["seed"] for row in matrix_rows} != set(seeds):
            raise ValueError("run_matrix must include every configured seed")
        run_matrix = matrix_rows
    raw_seed_overrides = raw.get("seed_config_overrides", {})
    if not isinstance(raw_seed_overrides, Mapping):
        raise ValueError("seed_config_overrides must be a JSON object")
    seed_config_overrides: dict[int, dict[str, Any]] = {}
    for raw_seed, raw_overrides in raw_seed_overrides.items():
        try:
            override_seed = validate_seed(int(raw_seed))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"seed_config_overrides contains invalid seed {raw_seed!r}"
            ) from exc
        if str(override_seed) != str(raw_seed):
            raise ValueError(
                f"seed_config_overrides seed {raw_seed!r} is not canonical"
            )
        if not isinstance(raw_overrides, Mapping):
            raise ValueError(
                f"seed_config_overrides[{raw_seed!r}] must be a JSON object"
            )
        overrides = dict(raw_overrides)
        _validate_config_keys(overrides, f"seed_config_overrides[{raw_seed!r}]")
        overlap = set(overrides).intersection(factors)
        if overlap:
            raise ValueError(
                "seed_config_overrides must not replace experiment factors: "
                f"{sorted(overlap)}"
            )
        seed_config_overrides[override_seed] = overrides
    if seed_config_overrides and set(seed_config_overrides) != set(seeds):
        raise ValueError(
            "seed_config_overrides must bind every configured seed exactly once"
        )
    run_annotations = raw.get("run_annotations", {})
    if not isinstance(run_annotations, Mapping):
        raise ValueError("run_annotations must be a JSON object")
    raw_confidence = raw.get("confidence_level", 0.95)
    if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, Real):
        raise ValueError("confidence_level must be a finite number between 0 and 1")
    confidence = float(raw_confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    retain_debug_tables = raw.get("retain_debug_tables", False)
    if not isinstance(retain_debug_tables, bool):
        raise ValueError("retain_debug_tables must be a boolean")

    master_table_mode = raw.get("master_table_mode", "full")
    if master_table_mode not in MASTER_TABLE_MODES:
        raise ValueError(
            "master_table_mode must be one of "
            f"{sorted(MASTER_TABLE_MODES)}"
        )

    master_policy = _validate_scientific_master_policy(
        raw.get("scientific_master_policy"),
        persistence_profile=base_config.get("persistence_profile"),
        experiment_profile=base_config.get("experiment_profile"),
    )

    spec = ExperimentSpec(
        experiment_id=experiment_id,
        output_dir=output_dir,
        base_config=base_config,
        factors=factors,
        seeds=seeds,
        run_matrix=run_matrix,
        seed_config_overrides=seed_config_overrides,
        run_annotations=dict(run_annotations),
        retain_debug_tables=retain_debug_tables,
        confidence_level=confidence,
        master_table_mode=master_table_mode,
        scientific_master_policy=master_policy,
        output_layout_version=str(raw.get("output_layout_version", OUTPUT_LAYOUT_VERSION)),
        design_version=int(raw.get("design_version", 1)),
        input_config_path=config_path,
    )
    validate_experiment_spec_contract(spec)
    return spec


def _validate_scientific_master_policy(
    raw: Mapping[str, Any] | None,
    *,
    persistence_profile: str | None,
    experiment_profile: str | None,
) -> dict[str, Any] | None:
    """Validate an opt-in, single-object policy for scientific master output."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("scientific_master_policy must be an object")
    if persistence_profile not in {"scientific", "diagnostic"}:
        raise ValueError(
            "scientific_master_policy requires persistence_profile scientific or diagnostic"
        )
    expected_keys = {"materialization", "table_keys", "compression"}
    if set(raw) != expected_keys:
        raise ValueError(
            "scientific_master_policy must contain exactly: "
            "materialization, table_keys, compression"
        )
    materialization = raw["materialization"]
    if materialization not in {"deferred", "on_complete"}:
        raise ValueError(
            "scientific_master_policy.materialization must be deferred or on_complete"
        )
    table_keys = raw["table_keys"]
    if (
        isinstance(table_keys, (str, bytes))
        or not isinstance(table_keys, Sequence)
        or not table_keys
        or any(not isinstance(key, str) for key in table_keys)
        or len(set(table_keys)) != len(table_keys)
    ):
        raise ValueError("scientific_master_policy.table_keys must be unique table keys")
    required_core = {"core_summary_df", "relay_bottleneck_metrics_df"}
    if not required_core.issubset(table_keys):
        raise ValueError(
            "scientific master selection must include core_summary_df and "
            "relay_bottleneck_metrics_df for cohort validation/aggregation"
        )
    allowed = set(RUN_TABLES) | {"core_summary_df", "relay_bottleneck_metrics_df"}
    if experiment_profile is not None:
        from loraphymac.experiments.common.scientific_output_contract import (
            output_contract_for,
        )

        profile_config = {"experiment_profile": experiment_profile}
        contract = output_contract_for(profile_config)
        if contract is not None:
            allowed.update(contract.required_additional_tables)
        model_ids = (
            "ff", "aa", "islam_dbpc_itm6", "islam_dbpc_802", "wang_atp_sf7_a2only"
        )
        allowed.update(experiment_run_tables(profile_config))
        for model_id in model_ids:
            allowed.update(
                experiment_run_tables(
                    {**profile_config, "tx_power_model_id": model_id}
                )
            )
    unknown = set(table_keys) - allowed
    if unknown:
        raise ValueError(
            "scientific_master_policy contains unregistered table keys: "
            + ", ".join(sorted(unknown))
        )
    compression = raw["compression"]
    if compression not in {"inherit", "csv", "gzip"}:
        raise ValueError(
            "scientific_master_policy.compression must be inherit, csv, or gzip"
        )
    return {
        "materialization": str(materialization),
        "table_keys": list(table_keys),
        "compression": str(compression),
    }


def _slug(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    text = str(value).replace(" ", "")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "value"


def build_experiment_grid(spec: ExperimentSpec) -> list[dict]:
    validate_experiment_spec_contract(spec)
    factor_names = list(spec.factors)
    runs = []
    if spec.run_matrix is not None:
        entries = []
        for row in spec.run_matrix:
            reserved = {"seed", "config_overrides", "dimensions"}
            factor_values = {key: value for key, value in row.items() if key not in reserved}
            dimensions = dict(row.get("dimensions", {}))
            if set(factor_values) != set(spec.factors):
                raise ValueError("run_matrix rows must provide every configured factor")
            factor_values.update(dimensions)
            entries.append((factor_values, row["seed"], dict(row.get("config_overrides", {}))))
    else:
        cells = itertools.product(*(spec.factors[name] for name in factor_names))
        entries = (
            (dict(zip(factor_names, values)), seed, {})
            for values in cells
            for seed in spec.seeds
        )
    for factor_values, seed, run_config_overrides in entries:
        validated_seed = validate_seed(seed)
        scientific_config = dict(spec.base_config)
        scientific_config.update(
            {key: value for key, value in factor_values.items() if key in spec.factors}
        )
        scientific_config.update(run_config_overrides)
        if scientific_config.get("experiment_profile") in {
            "apc_load_sweep_standard_v1",
            "apc_load_sweep_pilot_v1",
        }:
            mode = factor_values.get("node_ptx_mode")
            if mode not in {"adaptive", "fixed"}:
                raise ValueError("Standard APC load runs require adaptive or fixed mode")
            scientific_config["relay_ptx_mode"] = mode
        seed_overrides = spec.seed_config_overrides.get(validated_seed)
        if spec.seed_config_overrides and seed_overrides is None:
            raise ValueError(
                f"No seed_config_overrides binding for seed {validated_seed}"
            )
        if seed_overrides is not None:
            scientific_config.update(seed_overrides)
        scientific_config["seed"] = validated_seed
        try:
            resolve_configuration(scientific_config)
        except ConfigurationError as exc:
            raise ConfigurationError(
                "Invalid experiment grid cell "
                f"factors={factor_values!r}, seed={validated_seed}: {exc}"
            ) from exc
        slug = "__".join(
            f"{_slug(name)}-{_slug(value)}" for name, value in factor_values.items()
        )
        slug = slug[:120].rstrip("-_")
        config_digest = content_hash(scientific_config, length=10)
        run_id = f"{spec.experiment_id}__{slug}__s{validated_seed}__{config_digest}"
        runs.append(
            {
                "run_id": run_id,
                "seed": validated_seed,
                "factors": factor_values,
                "scientific_config": scientific_config,
            }
        )
    run_ids = [run["run_id"] for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Experiment grid produced duplicate run_id values")
    validate_experiment_grid_contract(spec, runs)
    return runs


def _source_fingerprint(project_root: Path) -> dict:
    source_root = project_root / "src" / "loraphymac"
    source_paths = list(source_root.rglob("*.py")) if source_root.exists() else []

    files = {}
    for path in sorted(set(source_paths)):
        relative = path.relative_to(project_root)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        files[relative.as_posix()] = file_sha256(path)
    return {"algorithm": "sha256", "files": files}


def _git_head(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _git_worktree_provenance(project_root: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.CalledProcessError):
        return {"available": False}
    status = result.stdout
    return {
        "available": True,
        "worktree_dirty": bool(status),
        "status_porcelain_sha256": hashlib.sha256(status).hexdigest(),
    }


def _portable_standard_spec(spec: ExperimentSpec, project_root: Path) -> dict[str, Any]:
    value = spec.serializable()
    value["output_dir"] = "."

    def normalize(item: Any, key: str | None = None) -> Any:
        if isinstance(item, dict):
            return {name: normalize(child, name) for name, child in item.items()}
        if isinstance(item, list):
            return [normalize(child, key) for child in item]
        if key and key.endswith("_manifest") and isinstance(item, str):
            path = Path(item).expanduser()
            if path.is_absolute():
                try:
                    return path.resolve().relative_to(project_root.resolve()).as_posix()
                except ValueError:
                    return item
        return item

    return normalize(value)


def _update_sweep_manifest_progress(
    paths: ExperimentArtifactPaths,
    run_id: str,
    status: str,
) -> None:
    if not paths.is_sweep_standard or not paths.manifest.is_file():
        return
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    run_statuses = dict(manifest.get("run_statuses", {}))
    run_statuses[run_id] = "PASS" if status == "SKIPPED" else status
    manifest.update(
        {
            "run_statuses": run_statuses,
            "completed_run_count": sum(value == "PASS" for value in run_statuses.values()),
            "failed_run_count": sum(value == "FAILED" for value in run_statuses.values()),
            "incomplete_run_count": sum(
                value not in {"PASS", "FAILED"} for value in run_statuses.values()
            ),
            "status": "RUNNING",
        }
    )
    atomic_write_json(paths.manifest, manifest)




def _environment_manifest() -> dict:
    packages = {}
    for module in (np, pd):
        packages[module.__name__] = getattr(module, "__version__", None)
    try:
        import scipy

        packages["scipy"] = scipy.__version__
    except Exception:
        packages["scipy"] = None
    try:
        import simpy

        packages["simpy"] = simpy.__version__
    except Exception:
        packages["simpy"] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def initialize_experiment(spec: ExperimentSpec, project_root: Path) -> Path:
    paths = ExperimentArtifactPaths.from_spec(spec)
    experiment_dir = paths.experiment_dir
    manifest_path = paths.manifest
    spec_value = (
        _portable_standard_spec(spec, project_root)
        if paths.is_sweep_standard
        else spec.serializable()
    )
    spec_digest = content_hash(spec_value)
    if paths.is_sweep_standard:
        validate_cohort_reservation(spec)
        if manifest_path.exists():
            raise FileExistsError(
                f"Sweep cohort already has a manifest and is immutable: {experiment_dir}"
            )
        grid = build_experiment_grid(spec)
        git_head = _git_head(project_root)
        source_fingerprint = _source_fingerprint(project_root)
        source_git = _git_worktree_provenance(project_root)
        consume_cohort_reservation(spec)
        paths.ensure()
        experiment_profile = spec.base_config.get(
            "experiment_profile", DEFAULT_CONFIG["experiment_profile"]
        )
        manifest = {
            "pipeline_version": PIPELINE_VERSION,
            "simulator_version": SIMULATOR_VERSION,
            "simulator_api_version": SIMULATOR_API_VERSION,
            "output_layout_version": SWEEP_OUTPUT_LAYOUT_VERSION,
            "artifact_directories": paths.directory_map(),
            "aggregation_version": AGGREGATION_VERSION,
            "experiment_contract_version": experiment_contract_version(experiment_profile),
            "created_at": spec.created_at,
            "execution_id": spec.execution_id,
            "timezone": spec.timezone,
            "spec_hash": spec_digest,
            "spec": spec_value,
            "source_fingerprint": source_fingerprint,
            "source_fingerprint_sha256": content_hash(source_fingerprint),
            "dirty_source_fingerprint_sha256": content_hash(source_fingerprint),
            "source_git": {"head": git_head, **source_git},
            "environment": _environment_manifest(),
            "input_configuration_hash": content_hash(spec.base_config),
            "git_head": git_head,
            "status": "INITIALIZED",
            **standard_manifest_fields(
                spec,
                expected_run_count=len(grid),
                run_ids=[row["run_id"] for row in grid],
                git_head=git_head,
            ),
        }
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(
            paths.analysis_manifest,
            {
                "analysis_id": f"{spec.experiment_id}-analysis-v1",
                "analysis_version": 1,
                "source_cohort_id": spec.experiment_id,
                "inputs": [],
                "generator": None,
                "metrics": {},
                "grouping_dimensions": list(spec.factors),
                "seed_aggregation": {
                    "method": "mean and sample standard deviation",
                    "confidence_interval": "Student-t",
                    "confidence_level": spec.confidence_level,
                },
                "included_model_variants": None,
                "figures": [],
                "tables": [],
                "status": "PENDING",
            },
        )
        return experiment_dir

    experiment_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("spec_hash") != spec_digest:
            raise ValueError(
                "Existing experiment directory has a different specification; "
                "use a new experiment_id or output_dir"
            )
        current_source = _source_fingerprint(project_root)
        if existing.get("source_fingerprint") != current_source:
            raise ValueError(
                "Source fingerprint differs from the existing experiment "
                "manifest; use a new experiment_id to avoid mixed models"
            )
        if existing.get("output_layout_version") != OUTPUT_LAYOUT_VERSION:
            raise ValueError(
                "Output layout differs from the existing experiment manifest; "
                "use a new experiment_id to avoid mixed artifact layouts"
            )
        paths.ensure()
        return experiment_dir

    paths.ensure()
    experiment_profile = spec.base_config.get(
        "experiment_profile", DEFAULT_CONFIG["experiment_profile"]
    )
    atomic_write_json(
        manifest_path,
        {
            "pipeline_version": PIPELINE_VERSION,
            "simulator_version": SIMULATOR_VERSION,
            "simulator_api_version": SIMULATOR_API_VERSION,
            "output_layout_version": spec.output_layout_version,
            "artifact_directories": paths.directory_map(),
            "aggregation_version": AGGREGATION_VERSION,
            "experiment_contract_version": experiment_contract_version(experiment_profile),
            "created_at": utc_now(),
            "spec_hash": spec_digest,
            "spec": spec_value,
            "source_fingerprint": _source_fingerprint(project_root),
            "environment": _environment_manifest(),
        },
    )
    return experiment_dir


def _annotate_table(
    frame: pd.DataFrame,
    spec: ExperimentSpec,
    run: Mapping[str, Any],
) -> pd.DataFrame:
    if frame is None:
        frame = pd.DataFrame()
    result = frame.copy()
    annotations = {
        "experiment_id": spec.experiment_id,
        **dict(spec.run_annotations),
        "run_id": run["run_id"],
        "seed": run["seed"],
        **run["factors"],
    }
    if (
        spec.base_config.get("experiment_profile") in {
            "apc_load_sweep_standard_v1", "apc_load_sweep_pilot_v1"
        }
        and "node_ptx_mode" in run["factors"]
    ):
        annotations["apc_mode"] = (
            "AA" if run["factors"]["node_ptx_mode"] == "adaptive" else "FF"
        )
    for key, value in reversed(list(annotations.items())):
        if key in result.columns:
            result[key] = value
        else:
            result.insert(0, key, value)
    return result


def _prefix_nonkeys(frame: pd.DataFrame, prefix: str, keys: Iterable[str]) -> pd.DataFrame:
    keys = set(keys)
    return frame.rename(
        columns={column: f"{prefix}{column}" for column in frame.columns if column not in keys}
    )


def build_relay_bottleneck_df(outputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Create one receiver/entity row containing queue and radio bottlenecks."""
    queue = outputs.get("relay_tx_queue_metrics_df", pd.DataFrame()).copy()
    if not queue.empty:
        queue = _prefix_nonkeys(queue, "queue_", {"entity_id", "entity_type"})

    rx = outputs.get("relay_rx_stats_df", pd.DataFrame()).copy()
    if not rx.empty:
        rx = rx.rename(columns={"relay": "entity_id"})
        rx = _prefix_nonkeys(rx, "rx_", {"entity_id"})

    duty = outputs.get("relay_duty_cycle_df", pd.DataFrame()).copy()
    if not duty.empty:
        duty = _prefix_nonkeys(duty, "duty_", {"entity_id", "entity_type"})

    demod = outputs.get("demodulator_pool_summary_df", pd.DataFrame()).copy()
    if not demod.empty:
        demod = demod.rename(columns={"receiver_id": "entity_id"})
        demod = _prefix_nonkeys(demod, "demod_", {"entity_id"})

    collision = outputs.get("collision_summary_df", pd.DataFrame()).copy()
    if not collision.empty:
        count_columns = [
            column
            for column in [
                "decision_count",
                "singleton_success_count",
                "collision_opportunity_count",
                "capture_success_count",
                "sinr_success_count",
                "contended_success_count",
                "collision_drop_count",
                "directed_interference_edge_count",
                "outcome_conservation_delta",
            ]
            if column in collision.columns
        ]
        collision = (
            collision.groupby("receiver_id", dropna=False)[count_columns]
            .sum(numeric_only=True)
            .reset_index()
            .rename(columns={"receiver_id": "entity_id"})
        )
        collision = _prefix_nonkeys(collision, "collision_", {"entity_id"})

    frames = [frame for frame in (queue, rx, duty, demod, collision) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = frames[0]
    for frame in frames[1:]:
        duplicate_type = "entity_type" in combined.columns and "entity_type" in frame.columns
        combined = combined.merge(frame, on="entity_id", how="outer")
        if duplicate_type:
            combined["entity_type"] = combined.pop("entity_type_x").combine_first(
                combined.pop("entity_type_y")
            )
    if "entity_type" not in combined.columns:
        combined["entity_type"] = combined["entity_id"].map(
            lambda value: "gateway" if str(value).startswith("GW") else "relay"
        )
    return combined.sort_values("entity_id", kind="stable").reset_index(drop=True)


def _resolved_simulator_config(run: Mapping[str, Any]) -> dict:
    config = dict(run["scientific_config"])
    config.update(
        {
            "run_id": run["run_id"],
            "output_dir": None,
            "export_summary": False,
            "export_results": False,
            "export_debug": False,
            "export_internal": False,
            "plot_routing": False,
            "print_summary": False,
            "print_detail": False,
            "print_debug": False,
            "print_internal": False,
            "verbose": False,
        }
    )
    return resolve_configuration(config)


def execute_one_run(
    spec: ExperimentSpec,
    experiment_dir: Path,
    run: Mapping[str, Any],
    resume: bool = True,
) -> str:
    resolved_simulator_config = _resolved_simulator_config(run)
    paths = ExperimentArtifactPaths.for_experiment(experiment_dir).ensure()
    run_dir = paths.runs / run["run_id"]
    tables_dir = run_dir / "tables"
    status_path = run_dir / "status.json"
    summary_path = run_dir / "core_summary.csv"
    managed_output = persistence_route(resolved_simulator_config).manager_enabled
    if managed_output and (run_dir.exists() or (paths.runs / f".{run['run_id']}.lock").exists()):
        # Managed output is immutable. The runner deliberately does not add a
        # new safe-to-skip rule; reuse requires a separately validated resume.
        atomic_write_json(
            paths.logs / f"{run['run_id']}.failed.json",
            {
                "run_id": run["run_id"],
                "status": "FAILED",
                "error_type": "ImmutableRunOutputExists",
                "error": "managed run output exists; refusing overwrite or implicit skip",
            },
        )
        return "FAILED"
    if not managed_output and resume and status_path.exists() and summary_path.exists():
        with status_path.open("r", encoding="utf-8") as handle:
            if json.load(handle).get("status") == "PASS":
                return "SKIPPED"

    if not managed_output:
        run_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = perf_counter()
    resolved_config_payload = {
        "pipeline_version": PIPELINE_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "experiment_id": spec.experiment_id,
        "run_id": run["run_id"],
        "seed": run["seed"],
        "factors": run["factors"],
        "scientific_config": run["scientific_config"],
        "sweep_profile": str(spec.run_annotations.get("sweep_profile", "production")),
        "run_annotations": dict(spec.run_annotations),
        "resolved_simulator_config": resolved_simulator_config,
        "config_hash": content_hash(resolved_simulator_config),
    }
    if not managed_output:
        atomic_write_json(run_dir / "resolved_config.json", resolved_config_payload)
        atomic_write_json(
            status_path,
            {"run_id": run["run_id"], "status": "RUNNING", "started_at": started_at},
        )

    try:
        source_provenance = (
            collect_complete_source_provenance(Path(__file__).resolve().parents[4])
            if managed_output
            else None
        )
        simulation = simulate(resolved_simulator_config)
        core = simulation.summary
        outputs = simulation.tables
        validate_run_output_contract(
            resolved_simulator_config,
            core,
            outputs,
        )
        core_frame = _annotate_table(pd.DataFrame([dict(core)]), spec, run)

        run_tables = dict(RUN_TABLES)
        run_tables.update(experiment_run_tables(resolved_simulator_config))
        bottleneck = _annotate_table(build_relay_bottleneck_df(outputs), spec, run)
        if managed_output:
            status_payload = {
                "run_id": run["run_id"],
                "status": "PASS",
                "started_at": started_at,
                "finished_at": utc_now(),
                "elapsed_wall_s": perf_counter() - started,
                "core_summary": str(summary_path.relative_to(experiment_dir)),
            }
            annotated_tables = {
                key: _annotate_table(value, spec, run)
                for key, value in outputs.items()
                if isinstance(value, pd.DataFrame)
            }
            for key in run_tables:
                if key not in annotated_tables:
                    annotated_tables[key] = _annotate_table(pd.DataFrame(), spec, run)
            if "core_summary_df" not in annotated_tables:
                annotated_tables["core_summary_df"] = core_frame.copy()
            persist_multiseed_run(
                experiment_dir=experiment_dir,
                experiment_id=spec.experiment_id,
                run_id=run["run_id"],
                config=resolved_simulator_config,
                source_provenance=source_provenance,
                summary=core,
                outputs=outputs,
                annotated_core_summary=core_frame,
                annotated_tables=annotated_tables,
                annotated_relay_bottleneck=bottleneck,
                status_payload=status_payload,
                resolved_config_payload=resolved_config_payload,
                pipeline_table_filenames=run_tables,
                validation_results={"experiment_output_contract": True},
            )
        else:
            atomic_write_csv(summary_path, core_frame)
            for output_key, filename in run_tables.items():
                table = _annotate_table(outputs.get(output_key, pd.DataFrame()), spec, run)
                atomic_write_csv(tables_dir / filename, table)
            atomic_write_csv(tables_dir / "relay_bottleneck_metrics.csv", bottleneck)
            if spec.retain_debug_tables:
                for output_key, filename in DEBUG_RUN_TABLES.items():
                    table = _annotate_table(outputs.get(output_key, pd.DataFrame()), spec, run)
                    atomic_write_csv(tables_dir / filename, table)
            status_payload = {
                "run_id": run["run_id"],
                "status": "PASS",
                "started_at": started_at,
                "finished_at": utc_now(),
                "elapsed_wall_s": perf_counter() - started,
                "core_summary": str(summary_path.relative_to(experiment_dir)),
            }
            atomic_write_json(status_path, status_payload)
        return "PASS"
    except Exception as exc:
        failure_payload = {
            "run_id": run["run_id"],
            "status": "FAILED",
            "started_at": started_at,
            "finished_at": utc_now(),
            "elapsed_wall_s": perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if managed_output:
            atomic_write_json(paths.logs / f"{run['run_id']}.failed.json", failure_payload)
        else:
            atomic_write_json(status_path, failure_payload)
        return "FAILED"


def _format_progress_context(run: Mapping[str, Any]) -> str:
    """Build a compact, paper-agnostic description for live progress output."""
    factors = dict(run.get("factors", {}))
    parts: list[str] = []

    node_mode = factors.pop("node_ptx_mode", None)
    relay_mode = factors.pop("relay_ptx_mode", None)
    mode_codes = {"fixed": "F", "adaptive": "A"}
    if node_mode in mode_codes and relay_mode in mode_codes:
        parts.append(f"mode={mode_codes[node_mode]}{mode_codes[relay_mode]}")
    else:
        if node_mode is not None:
            parts.append(f"node_ptx_mode={node_mode}")
        if relay_mode is not None:
            parts.append(f"relay_ptx_mode={relay_mode}")

    for key, value in factors.items():
        label = "delta_h" if key == "terrain_delta_h" else key
        parts.append(f"{label}={value}")
    parts.append(f"seed={run.get('seed')}")
    return " | ".join(parts)


def _print_progress(message: str, *, enabled: bool) -> None:
    """Print one immediately visible progress line in terminals and notebooks."""
    if enabled:
        print(message, flush=True)


def _read_passed_run_dirs(experiment_dir: Path) -> list[Path]:
    passed = []
    runs_dir = ExperimentArtifactPaths.for_experiment(experiment_dir).runs
    if not runs_dir.exists():
        return passed
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        status_path = run_dir / "status.json"
        if not status_path.exists():
            continue
        with status_path.open("r", encoding="utf-8") as handle:
            if json.load(handle).get("status") == "PASS":
                passed.append(run_dir)
    return passed


def _read_verified_scientific_tables(run_dir: Path) -> dict[str, tuple[str, Path]]:
    """Return checksum-verified managed table files keyed by table key."""
    manifest_path = run_dir / "scientific_output_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Scientific run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("tables")
    artifacts = manifest.get("artifacts")
    compatibility = manifest.get("compatibility_artifacts")
    required = manifest.get("required_table_keys")
    if (
        manifest.get("status") != "COMPLETED"
        or not isinstance(records, dict)
        or not isinstance(artifacts, dict)
        or not isinstance(compatibility, dict)
        or not isinstance(required, list)
        or set(records) != set(required)
    ):
        raise ValueError(f"Scientific run manifest is incomplete: {manifest_path}")

    all_records = [*records.values(), *artifacts.values(), *compatibility.values()]
    expected_paths = sorted(str(record.get("path", "")) for record in all_records)
    if manifest.get("required_artifacts") != expected_paths:
        raise ValueError(f"Scientific run artifact list is inconsistent: {manifest_path}")
    for record in all_records:
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"Unsafe scientific artifact path in {manifest_path}: {relative}")
        path = run_dir / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Scientific artifact is missing: {path}")
        if (
            path.stat().st_size != record.get("size_bytes")
            or file_sha256(path) != record.get("sha256")
        ):
            raise ValueError(f"Scientific artifact checksum mismatch: {path}")

    verified: dict[str, tuple[str, Path]] = {}
    filenames: set[str] = set()
    for key, record in records.items():
        relative = Path(str(record.get("path", "")))
        path = run_dir / relative
        filename = relative.name
        if filename in filenames:
            raise ValueError(f"Duplicate scientific table filename: {filename}")
        filenames.add(filename)
        verified[str(key)] = (filename, path)
    return verified


def _scientific_master_frames(
    experiment_dir: Path,
    run_dirs: Sequence[Path],
    table_keys: Sequence[str],
    run_configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read only requested canonical per-run tables and aggregate deterministically."""
    collected: dict[str, list[pd.DataFrame]] = {
        "all_runs_core_summary.csv": [],
        "per_run_relay_bottleneck_metrics.csv": [],
    }
    verified_by_run: dict[str, dict[str, tuple[str, Path]]] = {}
    selected_filenames: dict[str, str] = {}
    selected_schemas: dict[str, tuple[str, ...]] = {}
    for run_dir in run_dirs:
        verified = _read_verified_scientific_tables(run_dir)
        verified_by_run[run_dir.name] = verified
        selected: dict[str, tuple[str, Path]] = {}
        for key in table_keys:
            if key == "relay_bottleneck_metrics_df":
                record = verified.get(key)
            else:
                record = verified.get(key)
            if record is None or not record[1].is_file():
                config = (run_configs or {}).get(run_dir.name, {})
                contract = None
                if config:
                    from loraphymac.experiments.common.scientific_output_contract import (
                        output_contract_for,
                    )

                    contract = output_contract_for(config)
                model = str(config.get("tx_power_model_id", config.get("model_id", "")))
                if not model and config.get("node_ptx_mode") == "fixed":
                    model = "ff"
                elif not model and config.get("node_ptx_mode") == "adaptive":
                    model = "aa"
                semantics = (
                    contract.conditional_table_semantics.get(key, {}).get(model)
                    if contract is not None
                    else None
                )
                if semantics == "not_applicable_empty":
                    continue
                raise ValueError(f"Selected scientific master table {key} missing in {run_dir}")
            previous_filename = selected_filenames.setdefault(key, record[0])
            if previous_filename != record[0]:
                raise ValueError(
                    f"Selected table {key} has inconsistent per-run filenames: "
                    f"{previous_filename} vs {record[0]}"
                )
            selected[key] = record
        for key, (filename, path) in selected.items():
            try:
                frame = pd.read_csv(path, compression="infer")
            except pd.errors.EmptyDataError as exc:
                raise ValueError(f"Scientific table has no schema: {path}") from exc
            columns = tuple(frame.columns)
            previous_schema = selected_schemas.setdefault(key, columns)
            # These configuration/provenance tables intentionally carry
            # model-specific fields (for example Wang window counters or DBPC
            # calibration diagnostics). Per-run validators are authoritative;
            # concat preserves their union, as legacy master aggregation does.
            if key not in {"core_summary_df", "run_metadata_df"} and previous_schema != columns:
                raise ValueError(f"Selected table {key} has inconsistent per-run schemas")
            if key == "core_summary_df":
                collected["all_runs_core_summary.csv"].append(frame)
            elif key == "relay_bottleneck_metrics_df":
                collected["per_run_relay_bottleneck_metrics.csv"].append(frame)
            else:
                stem = filename.removesuffix(".gz").removesuffix(".csv")
                master_name = f"per_run_{stem}.csv"
                collected.setdefault(master_name, []).append(frame)
    result = {
        name: pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        for name, frames in collected.items()
    }
    for key in table_keys:
        master_name = (
            "all_runs_core_summary.csv"
            if key == "core_summary_df"
            else "per_run_relay_bottleneck_metrics.csv"
            if key == "relay_bottleneck_metrics_df"
            else next(
                (
                    f"per_run_{filename.removesuffix('.gz').removesuffix('.csv')}.csv"
                    for run_dir in run_dirs
                    for table_key, (filename, _) in verified_by_run[run_dir.name].items()
                    if table_key == key
                ),
                f"per_run_{key.removesuffix('_df')}.csv",
            )
        )
        if result.get(master_name, pd.DataFrame()).empty and not result.get(master_name, pd.DataFrame()).columns.size:
            schema = None
            for config in (run_configs or {}).values():
                from loraphymac.experiments.common.scientific_output_contract import (
                    output_contract_for,
                )

                contract = output_contract_for(config)
                if contract is not None and key in contract.table_schemas:
                    schema = contract.table_schemas[key]
                    break
            if schema:
                result[master_name] = pd.DataFrame(
                    columns=_annotated_empty_schema(experiment_dir, schema)
                )
    return result


def _master_filename_for_table_key(
    table_key: str,
    run_configs: Mapping[str, Mapping[str, Any]],
) -> str:
    """Map a canonical per-run key to the stable scientific master filename."""
    if table_key == "core_summary_df":
        return "all_runs_core_summary.csv"
    if table_key == "relay_bottleneck_metrics_df":
        return "per_run_relay_bottleneck_metrics.csv"
    filename = next(
        (
            experiment_run_tables(config)[table_key]
            for config in run_configs.values()
            if table_key in experiment_run_tables(config)
        ),
        RUN_TABLES.get(table_key, f"{table_key.removesuffix('_df')}.csv"),
    )
    stem = filename.removesuffix(".gz").removesuffix(".csv")
    return f"per_run_{stem}.csv"


def _read_experiment_spec_from_manifest(experiment_dir: Path) -> ExperimentSpec:
    manifest_path = ExperimentArtifactPaths.for_experiment(experiment_dir).manifest
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Experiment manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = manifest.get("spec")
    if not isinstance(raw, Mapping):
        raise ValueError("Experiment manifest has no valid serialized spec")
    if not raw.get("scientific_master_policy"):
        raise ValueError("Experiment has no scientific_master_policy")
    spec = ExperimentSpec(
        experiment_id=str(raw["experiment_id"]),
        output_dir=Path(raw["output_dir"]),
        base_config=dict(raw["base_config"]),
        factors={key: list(values) for key, values in raw["factors"].items()},
        seeds=list(raw["seeds"]),
        seed_config_overrides={
            int(seed): dict(value)
            for seed, value in raw.get("seed_config_overrides", {}).items()
        },
        run_annotations=dict(raw.get("run_annotations", {})),
        retain_debug_tables=bool(raw.get("retain_debug_tables", False)),
        confidence_level=float(raw.get("confidence_level", 0.95)),
        master_table_mode=str(raw.get("master_table_mode", "full")),
        scientific_master_policy=_validate_scientific_master_policy(
            raw["scientific_master_policy"],
            persistence_profile=raw["base_config"].get("persistence_profile"),
            experiment_profile=raw["base_config"].get("experiment_profile"),
        ),
    )
    if manifest.get("spec_hash") != content_hash(spec.serializable()):
        raise ValueError("Experiment manifest spec checksum does not match its serialized spec")
    validate_experiment_spec_contract(spec)
    return spec


def rebuild_scientific_master_tables(
    experiment_dir: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Validate the full expected cohort, then atomically publish selected masters.

    Rebuild is idempotent for an identical, checksum-valid finalized cohort and
    refuses to replace any existing scientific master directory.
    """
    experiment_dir = Path(experiment_dir).resolve()
    spec = _read_experiment_spec_from_manifest(experiment_dir)
    policy = dict(spec.scientific_master_policy or {})
    planned = build_experiment_grid(spec)
    expected_ids = [row["run_id"] for row in planned]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Expected run matrix contains duplicate run IDs")
    paths = ExperimentArtifactPaths.for_experiment(experiment_dir)
    run_dirs_by_id = {
        directory.name: directory
        for directory in paths.runs.iterdir()
        if directory.is_dir() and not directory.name.startswith(".")
    } if paths.runs.exists() else {}
    missing = sorted(set(expected_ids) - set(run_dirs_by_id))
    unexpected = sorted(set(run_dirs_by_id) - set(expected_ids))
    if missing or unexpected:
        raise ValueError(
            f"Expected run matrix mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[4]
    cohort_manifest_path = paths.manifest
    cohort_manifest = json.loads(cohort_manifest_path.read_text(encoding="utf-8"))
    current_source = _source_fingerprint(root)
    if cohort_manifest.get("source_fingerprint") != current_source:
        raise ValueError("Cohort source fingerprint differs from current source")
    source_provenance = collect_complete_source_provenance(root)

    expected_records = []
    run_dirs = []
    run_configs: dict[str, Mapping[str, Any]] = {}
    for planned_run in planned:
        run_id = planned_run["run_id"]
        run_dir = run_dirs_by_id[run_id]
        status_path = run_dir / "status.json"
        if not status_path.is_file() or json.loads(status_path.read_text()).get("status") != "PASS":
            raise ValueError(f"Run is not completed PASS: {run_id}")
        config = _resolved_simulator_config(planned_run)
        run_configs[run_id] = config
        compatibility_keys = set(RUN_TABLES) | set(experiment_run_tables(config))
        compatibility_keys.add("relay_bottleneck_metrics_df")
        check = ScientificOutputManager(experiment_dir).validate_completed(
            experiment_id=spec.experiment_id,
            run_id=run_id,
            config=config,
            source_provenance=source_provenance,
            persistence_profile=str(config.get("persistence_profile")),
            additional_table_keys=tuple(sorted(compatibility_keys)),
        )
        if not check.safe_to_skip:
            raise ValueError(
                f"Run provenance/output validation failed for {run_id}: "
                + "; ".join(check.reasons)
            )
        run_manifest = run_dir / "scientific_output_manifest.json"
        expected_records.append({
            "run_id": run_id,
            "manifest_sha256": file_sha256(run_manifest),
            "config_fingerprint_sha256": check.manifest.get("config_fingerprint_sha256"),
        })
        run_dirs.append(run_dir)

    selected_keys = list(policy["table_keys"])
    candidate_frames = _scientific_master_frames(
        experiment_dir, run_dirs, selected_keys, run_configs=run_configs
    )
    compression = policy["compression"]
    gzip_master = (
        bool(spec.base_config.get("gzip_scientific_tables", False))
        if compression == "inherit"
        else compression == "gzip"
    )
    cohort_digest = hashlib.sha256(
        canonical_json({
            "expected_run_ids": expected_ids,
            "run_manifests": expected_records,
            "source_fingerprint": current_source,
            "policy": policy,
        }).encode("utf-8")
    ).hexdigest()
    output_dir = paths.scientific_master
    if output_dir.exists():
        existing_path = output_dir / "aggregation_manifest.json"
        if not existing_path.is_file():
            raise FileExistsError(f"Refusing to overwrite unmanifested scientific master: {output_dir}")
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if existing.get("cohort_sha256") != cohort_digest:
            raise FileExistsError(f"Scientific master belongs to a different cohort: {output_dir}")
        status_path = paths.status
        if status_path.is_file():
            recorded = json.loads(status_path.read_text(encoding="utf-8")).get(
                "scientific_master_manifest_sha256"
            )
            if recorded and recorded != file_sha256(existing_path):
                raise ValueError("Finalized scientific master manifest checksum mismatch")
        for record in existing.get("files", {}).values():
            file_path = output_dir / record["path"]
            if not file_path.is_file() or file_path.stat().st_size != record["size_bytes"] or file_sha256(file_path) != record["sha256"]:
                raise ValueError(f"Finalized scientific master checksum mismatch: {file_path}")
        result = {}
        for name, record in existing["files"].items():
            result[name] = pd.read_csv(output_dir / record["path"], compression="infer")
        _mark_scientific_cohort_finalized(paths, experiment_dir, output_dir)
        return result

    paths.master.mkdir(parents=True, exist_ok=True)
    staging = paths.master / f".scientific-master-{uuid.uuid4().hex}.partial"
    staging.mkdir()
    try:
        file_records = {}
        result = {}
        master_key_by_name = {
            "all_runs_core_summary.csv": "core_summary_df",
            "per_run_relay_bottleneck_metrics.csv": "relay_bottleneck_metrics_df",
        }
        for run_dir in run_dirs:
            for table_key, (table_filename, _) in _read_verified_scientific_tables(run_dir).items():
                if table_key in selected_keys and table_key not in master_key_by_name.values():
                    stem = table_filename.removesuffix(".gz").removesuffix(".csv")
                    master_key_by_name[f"per_run_{stem}.csv"] = table_key
        for table_key in selected_keys:
            if table_key not in master_key_by_name.values() and table_key not in {
                "core_summary_df", "relay_bottleneck_metrics_df"
            }:
                master_key_by_name[
                    f"per_run_{table_key.removesuffix('_df')}.csv"
                ] = table_key
        for master_name, frame in candidate_frames.items():
            base = master_name
            key = master_key_by_name.get(master_name)
            if key is None or key not in selected_keys:
                continue
            if frame.empty and len(frame.columns) == 0:
                raise ValueError(
                    f"Selected table {key} has no rows and no registered empty schema"
                )
            filename = base[:-4] + ".csv.gz" if gzip_master else base
            path = staging / filename
            atomic_write_csv(path, frame)
            file_records[master_name] = {
                "path": filename,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "rows": len(frame),
                "columns": list(frame.columns),
                "table_key": key,
            }
            result[master_name] = frame
        manifest_value = {
            "schema_version": "scientific_master_v1",
            "status": "FINALIZED",
            "experiment_id": spec.experiment_id,
            "experiment_contract_version": experiment_contract_version(
                spec.base_config.get("experiment_profile")
            ),
            "source_fingerprint_sha256": hashlib.sha256(
                canonical_json(current_source).encode("utf-8")
            ).hexdigest(),
            "cohort_sha256": cohort_digest,
            "expected_run_count": len(expected_ids),
            "expected_run_ids": expected_ids,
            "expected_run_matrix": [
                {"run_id": row["run_id"], "seed": row["seed"], "factors": row["factors"]}
                for row in planned
            ],
            "runs": expected_records,
            "policy": policy,
            "compression": "gzip" if gzip_master else "csv",
            "files": file_records,
        }
        atomic_write_json(staging / "aggregation_manifest.json", manifest_value)
        if output_dir.exists():
            raise FileExistsError(f"Scientific master appeared during publish: {output_dir}")
        os.rename(staging, output_dir)
    except Exception:
        # Keep interrupted staging evidence; no finalized directory is exposed.
        raise

    _mark_scientific_cohort_finalized(paths, experiment_dir, output_dir)
    return result


def _mark_scientific_cohort_finalized(
    paths: ExperimentArtifactPaths, experiment_dir: Path, output_dir: Path
) -> None:
    status_path = paths.status
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.is_file()
        else {"experiment_id": experiment_dir.name}
    )
    status["status"] = "PASS"
    status["aggregation_status"] = "FINALIZED"
    status["cohort_finalized"] = True
    status["scientific_master_manifest"] = str(
        (output_dir / "aggregation_manifest.json").relative_to(experiment_dir)
    )
    status["scientific_master_manifest_sha256"] = file_sha256(
        output_dir / "aggregation_manifest.json"
    )
    atomic_write_json(status_path, status)


def _annotated_empty_schema(
    experiment_dir: Path,
    producer_columns: Sequence[str],
) -> list[str]:
    """Recreate producer columns plus the pipeline's run annotations."""
    columns = list(producer_columns)
    manifest_path = ExperimentArtifactPaths.for_experiment(experiment_dir).manifest
    if not manifest_path.is_file():
        return columns
    spec = json.loads(manifest_path.read_text(encoding="utf-8")).get("spec", {})
    annotations = ["experiment_id"]
    annotations.extend((spec.get("run_annotations") or {}).keys())
    annotations.extend(("run_id", "seed"))
    annotations.extend((spec.get("factors") or {}).keys())
    for key in reversed(annotations):
        if key not in columns:
            columns.insert(0, key)
    return columns


def rebuild_master_tables(
    experiment_dir: Path,
    experiment_profile: str | None = None,
    master_table_mode: str = "full",
    persistence_profile: str | None = None,
) -> dict[str, pd.DataFrame]:
    if master_table_mode not in MASTER_TABLE_MODES:
        raise ValueError(
            "master_table_mode must be one of "
            f"{sorted(MASTER_TABLE_MODES)}"
        )
    artifact_paths = ExperimentArtifactPaths.for_experiment(experiment_dir).ensure()
    master_dir = artifact_paths.aggregate if artifact_paths.is_sweep_standard else artifact_paths.master
    passed_dirs = _read_passed_run_dirs(experiment_dir)
    table_names = master_table_names_for_profile(experiment_profile)
    managed = persistence_profile in {"scientific", "diagnostic"}
    managed_by_run = (
        {run_dir: _read_verified_scientific_tables(run_dir) for run_dir in passed_dirs}
        if managed
        else {}
    )
    if managed:
        # Managed runs may include contract-required tables outside the legacy
        # RUN_TABLES list. Add their manifested filenames without changing the
        # legacy master-table surface.
        for tables in managed_by_run.values():
            for filename, _ in tables.values():
                if filename != "core_summary.csv":
                    table_names.setdefault(filename, f"per_run_{filename}")
    if master_table_mode == "aggregate_inputs":
        table_names = {
            "relay_bottleneck_metrics.csv": (
                MASTER_TABLE_NAMES["relay_bottleneck_metrics.csv"]
            )
        }
    collected: dict[str, list[pd.DataFrame]] = {
        "all_runs_core_summary.csv": [],
        **{master: [] for master in table_names.values()},
    }
    for run_dir in passed_dirs:
        core_path = run_dir / "core_summary.csv"
        if core_path.exists():
            collected["all_runs_core_summary.csv"].append(pd.read_csv(core_path))
        managed_tables = managed_by_run.get(run_dir) if managed else None
        for run_filename, master_filename in table_names.items():
            if managed_tables is not None:
                match = next(
                    (
                        (key, path)
                        for key, (filename, path) in managed_tables.items()
                        if filename == run_filename
                    ),
                    None,
                )
                # An absent artifact is not inferred to be an empty table: it
                # was not selected for this model/contract and is N/A here.
                if match is None:
                    continue
                table_key, path = match
            else:
                table_key = None
                path = run_dir / "tables" / run_filename
            if path.exists():
                try:
                    frame = pd.read_csv(path)
                except pd.errors.EmptyDataError:
                    schema = None
                    if table_key is not None:
                        from loraphymac.experiments.common.scientific_output_contract import (
                            output_contract_for,
                        )

                        contract = output_contract_for(
                            {"experiment_profile": experiment_profile}
                        )
                        schema = contract.table_schemas.get(table_key) if contract else None
                    if not schema:
                        raise ValueError(
                            f"Empty scientific table has no registered schema: {path}"
                        )
                    frame = pd.DataFrame(
                        columns=_annotated_empty_schema(
                            Path(experiment_dir), schema
                        )
                    )
                if not frame.empty or managed:
                    collected[master_filename].append(frame)

    result = {}
    for filename, frames in collected.items():
        combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        atomic_write_csv(master_dir / filename, combined)
        result[filename] = combined
    return result


def validate_seed_completeness(
    frame: pd.DataFrame,
    factor_columns: Sequence[str],
    expected_seeds: Sequence[int],
    expected_run_matrix: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    report_columns = [
        *factor_columns,
        "expected_seed_count",
        "observed_seed_count",
        "missing_seeds",
        "unexpected_seeds",
        "duplicate_seed_count",
        "complete",
    ]
    if frame.empty and expected_run_matrix is None:
        return pd.DataFrame(columns=report_columns)
    required = {"run_id", "seed", *factor_columns}
    missing = required.difference(frame.columns)
    if missing and not (frame.empty and expected_run_matrix is not None):
        raise ValueError(f"Master summary missing columns: {sorted(missing)}")
    if expected_run_matrix is not None:
        if not expected_run_matrix:
            raise ValueError("expected_run_matrix must not be empty")
        expected_by_cell: dict[tuple[Any, ...], set[int]] = {}
        declared_identities = []
        seed_contract = {int(value) for value in expected_seeds}
        for index, run in enumerate(expected_run_matrix):
            try:
                cell = tuple(run[name] for name in factor_columns)
                seed = int(run["seed"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"expected_run_matrix[{index}] lacks a factor or integer seed"
                ) from exc
            if seed not in seed_contract:
                raise ValueError(
                    f"expected_run_matrix[{index}].seed is outside expected_seeds"
                )
            expected_by_cell.setdefault(cell, set()).add(seed)
            declared_identities.append((*cell, seed))
        if len(declared_identities) != len(set(declared_identities)):
            raise ValueError("expected_run_matrix contains duplicate factor/seed rows")
        if not frame.empty and frame["run_id"].duplicated().any():
            duplicates = frame.loc[frame["run_id"].duplicated(), "run_id"].tolist()
            raise ValueError(f"Duplicate run_id rows: {duplicates[:10]}")

        observed_by_cell: dict[tuple[Any, ...], pd.DataFrame] = {}
        if not frame.empty:
            if factor_columns:
                grouped = frame.groupby(list(factor_columns), dropna=False, sort=True)
            else:
                working = frame.assign(__single_cell__="all")
                grouped = working.groupby("__single_cell__", dropna=False, sort=True)
            for keys, subset in grouped:
                cell = keys if isinstance(keys, tuple) else (keys,)
                observed_by_cell[tuple(cell)] = subset

        rows = []
        for cell in sorted(set(expected_by_cell) | set(observed_by_cell), key=repr):
            subset = observed_by_cell.get(cell, frame.iloc[0:0])
            observed_values = (
                pd.to_numeric(subset["seed"], errors="raise").astype(int)
                if not subset.empty else pd.Series(dtype=int)
            )
            observed = set(observed_values.tolist())
            expected = expected_by_cell.get(cell, set())
            duplicate_count = int(observed_values.duplicated().sum())
            row = dict(zip(factor_columns, cell))
            row.update(
                {
                    "expected_seed_count": len(expected),
                    "observed_seed_count": len(observed),
                    "missing_seeds": ",".join(map(str, sorted(expected - observed))),
                    "unexpected_seeds": ",".join(map(str, sorted(observed - expected))),
                    "duplicate_seed_count": duplicate_count,
                    "complete": bool(
                        expected and observed == expected and duplicate_count == 0
                    ),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=report_columns)

    if missing:
        raise ValueError(f"Master summary missing columns: {sorted(missing)}")
    if frame["run_id"].duplicated().any():
        duplicates = frame.loc[frame["run_id"].duplicated(), "run_id"].tolist()
        raise ValueError(f"Duplicate run_id rows: {duplicates[:10]}")

    expected = {int(seed) for seed in expected_seeds}
    working = frame.copy()
    if factor_columns:
        grouped = working.groupby(list(factor_columns), dropna=False, sort=True)
    else:
        working["__single_cell__"] = "all"
        grouped = working.groupby("__single_cell__", dropna=False, sort=True)
    rows = []
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        observed_values = pd.to_numeric(sub["seed"], errors="raise").astype(int)
        observed = set(observed_values.tolist())
        duplicate_count = int(observed_values.duplicated().sum())
        row = dict(zip(factor_columns, keys))
        row.update(
            {
                "expected_seed_count": len(expected),
                "observed_seed_count": len(observed),
                "missing_seeds": ",".join(map(str, sorted(expected - observed))),
                "unexpected_seeds": ",".join(map(str, sorted(observed - expected))),
                "duplicate_seed_count": duplicate_count,
                "complete": (observed == expected and duplicate_count == 0),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=report_columns)


def aggregate_paper_grade(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    output_columns = [
        *group_columns,
        "metric",
        "n",
        "mean",
        "std",
        "sem",
        "confidence_level",
        "ci_method",
        "ci_lower",
        "ci_upper",
        "minimum",
        "p05",
        "p25",
        "median",
        "p75",
        "p95",
        "maximum",
    ]
    if frame.empty:
        return pd.DataFrame(columns=output_columns)
    metrics = [column for column in metric_columns if column in frame.columns]
    if not metrics:
        raise ValueError("No requested numeric metrics exist in the input table")
    for column in group_columns:
        if column not in frame.columns:
            raise ValueError(f"Missing aggregation group column: {column}")

    if group_columns:
        grouped = frame.groupby(list(group_columns), dropna=False, sort=True)
    else:
        grouped = [((), frame)]
    rows = []
    for keys, sub in grouped:
        if group_columns and not isinstance(keys, tuple):
            keys = (keys,)
        elif not group_columns:
            keys = ()
        group_values = dict(zip(group_columns, keys))
        for metric in metrics:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            n = int(len(values))
            if n == 0:
                continue
            mean = float(values.mean())
            if n >= 2:
                std = float(values.std(ddof=1))
                sem = std / math.sqrt(n)
                critical = float(student_t.ppf((1.0 + confidence_level) / 2.0, n - 1))
                ci_lower = mean - critical * sem
                ci_upper = mean + critical * sem
                ci_method = "student_t"
            else:
                std = sem = ci_lower = ci_upper = float("nan")
                ci_method = "not_estimable_n_lt_2"
            rows.append(
                {
                    **group_values,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                    "confidence_level": confidence_level,
                    "ci_method": ci_method,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "minimum": float(values.min()),
                    "p05": float(values.quantile(0.05)),
                    "p25": float(values.quantile(0.25)),
                    "median": float(values.quantile(0.50)),
                    "p75": float(values.quantile(0.75)),
                    "p95": float(values.quantile(0.95)),
                    "maximum": float(values.max()),
                }
            )
    return pd.DataFrame(rows, columns=output_columns)


def aggregate_experiment(
    spec: ExperimentSpec,
    experiment_dir: Path,
    master_tables: Mapping[str, pd.DataFrame],
    allow_incomplete: bool = False,
) -> dict[str, pd.DataFrame]:
    aggregate_dir = ExperimentArtifactPaths.for_experiment(experiment_dir).ensure().aggregate
    factor_columns = list(spec.factors)
    for run in build_experiment_grid(spec):
        for name in run["factors"]:
            if name not in factor_columns:
                factor_columns.append(name)
    core = master_tables.get("all_runs_core_summary.csv", pd.DataFrame())
    seed_report = validate_seed_completeness(
        core,
        factor_columns,
        spec.seeds,
        expected_run_matrix=spec.run_matrix,
    )
    atomic_write_csv(aggregate_dir / "seed_completeness.csv", seed_report)
    if not allow_incomplete and (seed_report.empty or not seed_report["complete"].all()):
        raise ValueError("Seed grid is incomplete; inspect aggregate/seed_completeness.csv")

    metric_contract = experiment_aggregation_metrics(
        spec.base_config.get("experiment_profile"),
        CORE_PAPER_METRICS,
    )
    core_aggregate = aggregate_paper_grade(
        core,
        group_columns=factor_columns,
        metric_columns=metric_contract,
        confidence_level=spec.confidence_level,
    )
    atomic_write_csv(aggregate_dir / "core_metrics_long.csv", core_aggregate)

    relay = master_tables.get("per_run_relay_bottleneck_metrics.csv", pd.DataFrame())
    relay_aggregate = pd.DataFrame()
    if not relay.empty:
        relay_group_columns = [*factor_columns, "entity_id"]
        excluded = {
            "seed",
            "run_id",
            "experiment_id",
            "entity_type",
            *relay_group_columns,
        }
        relay_metrics = [
            column
            for column in relay.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(relay[column])
        ]
        relay_aggregate = aggregate_paper_grade(
            relay,
            group_columns=relay_group_columns,
            metric_columns=relay_metrics,
            confidence_level=spec.confidence_level,
        )
        atomic_write_csv(
            aggregate_dir / "relay_bottleneck_metrics_long.csv",
            relay_aggregate,
        )
    return {
        "seed_completeness": seed_report,
        "core_metrics": core_aggregate,
        "relay_bottleneck_metrics": relay_aggregate,
    }


def run_experiment(
    spec: ExperimentSpec,
    project_root: str | Path,
    *,
    resume: bool = True,
    allow_incomplete: bool = False,
    fail_fast: bool = False,
    max_runs: int | None = None,
    dry_run: bool = False,
    show_progress: bool = True,
) -> dict:
    project_root = Path(project_root).resolve()
    planned_grid = build_experiment_grid(spec)
    grid = planned_grid
    if max_runs is not None:
        grid = grid[: max(0, int(max_runs))]
    if dry_run:
        paths = ExperimentArtifactPaths.from_spec(spec)
        return {
            "pipeline_version": PIPELINE_VERSION,
            "simulator_version": SIMULATOR_VERSION,
            "simulator_api_version": SIMULATOR_API_VERSION,
            "output_layout_version": spec.output_layout_version or OUTPUT_LAYOUT_VERSION,
            "experiment_id": spec.experiment_id,
            "experiment_dir": str(paths.experiment_dir),
            "artifact_directories": paths.directory_map(),
            "run_count": len(grid),
            "runs": grid,
        }

    try:
        experiment_dir = initialize_experiment(spec, project_root)
    except Exception as exc:
        if spec.output_layout_version == SWEEP_OUTPUT_LAYOUT_VERSION:
            _record_standard_initialization_failure(spec, project_root, exc)
        raise
    paths = ExperimentArtifactPaths.for_experiment(experiment_dir)
    counts = {"PASS": 0, "FAILED": 0, "SKIPPED": 0}
    total_runs = len(grid)
    _print_progress(
        (
            f"[PIPELINE] experiment={spec.experiment_id} | runs={total_runs} | "
            f"resume={resume}"
        ),
        enabled=show_progress,
    )
    for index, run in enumerate(grid, start=1):
        prefix = f"[{index:03d}/{total_runs:03d}]"
        context = _format_progress_context(run)
        _print_progress(
            f"{prefix} RUNNING | {context}",
            enabled=show_progress,
        )
        run_started = perf_counter()
        status = execute_one_run(spec, experiment_dir, run, resume=resume)
        counts[status] += 1
        _update_sweep_manifest_progress(paths, run["run_id"], status)
        _print_progress(
            f"{prefix} {status} | elapsed={perf_counter() - run_started:.2f}s",
            enabled=show_progress,
        )
        if status == "FAILED" and fail_fast:
            _print_progress(
                f"[PIPELINE] STOPPED | fail_fast=True | processed={sum(counts.values())}",
                enabled=show_progress,
            )
            break

    _print_progress(
        f"[PIPELINE] AGGREGATING | processed={sum(counts.values())}/{total_runs}",
        enabled=show_progress,
    )
    experiment_profile = spec.base_config.get(
        "experiment_profile", DEFAULT_CONFIG["experiment_profile"]
    )
    policy = spec.scientific_master_policy
    passed_dirs = _read_passed_run_dirs(experiment_dir)
    passed_ids = {path.name for path in passed_dirs}
    expected_ids = {run["run_id"] for run in planned_grid}
    cohort_runs_complete = passed_ids == expected_ids
    master_build_error = None
    if policy is None:
        master = rebuild_master_tables(
            experiment_dir,
            experiment_profile=experiment_profile,
            master_table_mode=spec.master_table_mode,
            persistence_profile=spec.base_config.get("persistence_profile"),
        )
    else:
        # Aggregate summaries can be computed from canonical per-run tables in
        # memory. Deferred policy avoids publishing any master files here.
        master = _scientific_master_frames(
            experiment_dir,
            sorted(passed_dirs),
            policy["table_keys"],
            run_configs={
                run["run_id"]: _resolved_simulator_config(run)
                for run in planned_grid
                if run["run_id"] in passed_ids
            },
        )
        if (
            cohort_runs_complete
            and policy["materialization"] == "on_complete"
        ):
            try:
                master = rebuild_scientific_master_tables(
                    experiment_dir,
                    project_root=project_root,
                )
            except Exception as exc:
                master_build_error = f"{type(exc).__name__}: {exc}"
                master = _scientific_master_frames(
                    experiment_dir,
                    sorted(passed_dirs),
                    policy["table_keys"],
                    run_configs={
                        run["run_id"]: _resolved_simulator_config(run)
                        for run in planned_grid
                        if run["run_id"] in passed_ids
                    },
                )
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "simulator_api_version": SIMULATOR_API_VERSION,
        "output_layout_version": spec.output_layout_version,
        "artifact_directories": paths.directory_map(),
        "experiment_id": spec.experiment_id,
        "master_table_mode": spec.master_table_mode,
        "scientific_master_policy": policy,
        "finished_at": utc_now(),
        "requested_run_count": len(grid),
        "status_counts": counts,
        "passed_run_count_on_disk": len(_read_passed_run_dirs(experiment_dir)),
        "experiment_dir": str(experiment_dir),
        "seed_grid_complete": False,
        "cohort_run_matrix_complete": cohort_runs_complete,
        "aggregation_status": (
            "FINALIZED"
            if policy is not None
            and policy["materialization"] == "on_complete"
            and cohort_runs_complete
            and master_build_error is None
            else "FAILED" if master_build_error is not None
            else "PENDING" if policy is not None and cohort_runs_complete else "INCOMPLETE"
        ),
        "cohort_finalized": bool(
            policy is None
            or (
                policy["materialization"] == "on_complete"
                and cohort_runs_complete
                and master_build_error is None
            )
        ),
        "master_build_error": master_build_error,
        "status": "AGGREGATING",
    }
    status_path = paths.status
    atomic_write_json(status_path, summary)
    try:
        aggregate = aggregate_experiment(
            spec,
            experiment_dir,
            master,
            allow_incomplete=allow_incomplete,
        )
    except Exception as exc:
        summary.update(
            {
                "finished_at": utc_now(),
                "status": "INCOMPLETE",
                "aggregation_error_type": type(exc).__name__,
                "aggregation_error": str(exc),
            }
        )
        atomic_write_json(status_path, summary)
        _print_progress(
            f"[PIPELINE] INCOMPLETE | {type(exc).__name__}: {exc}",
            enabled=show_progress,
        )
        raise

    summary.update(
        {
            "finished_at": utc_now(),
            "status": (
                "RUNS_COMPLETE"
                if policy is not None
                and (
                    policy["materialization"] == "deferred"
                    or master_build_error is not None
                )
                and cohort_runs_complete
                else "PASS" if counts["FAILED"] == 0 else "PARTIAL"
            ),
            "seed_grid_complete": bool(
                not aggregate["seed_completeness"].empty
                and aggregate["seed_completeness"]["complete"].all()
            ),
        }
    )
    if paths.is_sweep_standard:
        if policy is None and cohort_runs_complete and master_build_error is None:
            summary["aggregation_status"] = "FINALIZED"
        complete = bool(
            cohort_runs_complete
            and summary["seed_grid_complete"]
            and summary.get("aggregation_status") == "FINALIZED"
            and counts["FAILED"] == 0
        )
        summary["status"] = "COMPLETE" if complete else "INCOMPLETE"
    atomic_write_json(status_path, summary)
    if paths.is_sweep_standard:
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        run_statuses = {}
        for run in planned_grid:
            run_status_path = paths.runs / run["run_id"] / "status.json"
            if run_status_path.is_file():
                try:
                    run_statuses[run["run_id"]] = json.loads(
                        run_status_path.read_text(encoding="utf-8")
                    ).get("status", "UNKNOWN")
                except (OSError, ValueError, TypeError):
                    run_statuses[run["run_id"]] = "INVALID_STATUS"
            elif (paths.runs / run["run_id"] / "scientific_output_manifest.json").is_file():
                run_statuses[run["run_id"]] = "PASS"
            else:
                run_statuses[run["run_id"]] = "INCOMPLETE"
        manifest.update(
            {
                "status": summary["status"],
                "finished_at": summary["finished_at"],
                "completed_run_count": sum(value == "PASS" for value in run_statuses.values()),
                "failed_run_count": sum(value == "FAILED" for value in run_statuses.values()),
                "incomplete_run_count": sum(
                    value not in {"PASS", "FAILED"} for value in run_statuses.values()
                ),
                "run_statuses": run_statuses,
                "aggregate_status": summary.get("aggregation_status", "INCOMPLETE"),
            }
        )
        atomic_write_json(paths.manifest, manifest)
        if summary["status"] == "COMPLETE":
            write_cohort_readme(
                experiment_dir,
                spec=spec,
                run_count=len(planned_grid),
                seed_list=list(spec.seeds),
                source_fingerprint_hash=content_hash(manifest["source_fingerprint"]),
            )
    _print_progress(
        (
            f"[PIPELINE] {summary['status']} | passed_on_disk="
            f"{summary['passed_run_count_on_disk']}/{total_runs} | "
            f"seed_grid_complete={summary['seed_grid_complete']}"
        ),
        enabled=show_progress,
    )
    return summary


def _record_standard_initialization_failure(
    spec: ExperimentSpec, project_root: Path, error: Exception
) -> None:
    """Leave machine-readable failure evidence for an owned reservation."""
    reservation_path = spec.cohort_reservation_path
    if reservation_path is None:
        return
    paths = ExperimentArtifactPaths.from_spec(spec)
    current: dict[str, Any] = {}
    if paths.manifest.is_file():
        try:
            current = json.loads(paths.manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            current = {}
    if current.get("execution_id") not in {None, spec.execution_id}:
        return
    if current.get("status") == "COMPLETE":
        return
    failure = f"{type(error).__name__}: {error}"
    current.update({
        "experiment_id": spec.experiment_id,
        "execution_id": spec.execution_id,
        "output_layout_version": SWEEP_OUTPUT_LAYOUT_VERSION,
        "status": "FAILED",
        "initialization_error_type": type(error).__name__,
        "initialization_error": failure,
    })
    atomic_write_json(paths.manifest, current)
    atomic_write_json(paths.status, {
        "experiment_id": spec.experiment_id,
        "execution_id": spec.execution_id,
        "status": "FAILED",
        "seed_grid_complete": False,
        "initialization_error_type": type(error).__name__,
        "initialization_error": failure,
    })
    if Path(reservation_path).is_file():
        try:
            reservation = json.loads(Path(reservation_path).read_text(encoding="utf-8"))
            if reservation.get("execution_id") == spec.execution_id:
                reservation["status"] = "FAILED"
                reservation["failure"] = failure
                atomic_write_json(Path(reservation_path), reservation)
        except (OSError, ValueError, TypeError):
            pass
