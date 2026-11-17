"""Contract for the isolated FF/AA terrain-profile diagnostic cohort."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from loraphymac.experiments.apc_terrain_sweep.contract import (
    PAPER1_APC_AGGREGATION_METRICS,
    PAPER1_APC_ALWAYS_RUN_TABLES,
    PAPER1_APC_COMPARISON_RUN_TABLES,
    PAPER1_APC_LOCKED_CONFIG,
    PAPER1_APC_REQUIRED_OUTPUT_TABLES,
    PAPER1_APC_REQUIRED_SUMMARY_FIELDS,
)


APC_TERRAIN_PROFILE_DIAGNOSTIC_PROFILE = "apc_terrain_profile_diagnostic_v1"
APC_TERRAIN_PROFILE_DIAGNOSTIC_VERSION = "apc_terrain_profile_diagnostic_contract_v1.0"
APC_TERRAIN_PROFILE_DIAGNOSTIC_TERRAIN = 90
APC_TERRAIN_PROFILE_DIAGNOSTIC_SEEDS = (30, 36, 42, 48, 56)
APC_TERRAIN_PROFILE_DIAGNOSTIC_MODES = ("fixed", "adaptive")
APC_TERRAIN_PROFILE_DIAGNOSTIC_RUN_COUNT = 10

_EXPECTED_ROWS = {
    (node_mode, relay_mode, APC_TERRAIN_PROFILE_DIAGNOSTIC_TERRAIN, seed)
    for node_mode, relay_mode in (("fixed", "fixed"), ("adaptive", "adaptive"))
    for seed in APC_TERRAIN_PROFILE_DIAGNOSTIC_SEEDS
}
_DIAGNOSTIC_TABLES = {
    **PAPER1_APC_ALWAYS_RUN_TABLES,
    **PAPER1_APC_COMPARISON_RUN_TABLES,
    "application_queue_metrics_df": "application_queue_metrics.csv",
}


def diagnostic_config_violations(config: Mapping[str, Any]) -> list[str]:
    errors = []
    if config.get("experiment_profile") != APC_TERRAIN_PROFILE_DIAGNOSTIC_PROFILE:
        errors.append("experiment_profile must identify the diagnostic contract")
    for key, expected in PAPER1_APC_LOCKED_CONFIG.items():
        if key in {"node_ptx_mode", "relay_ptx_mode", "terrain_delta_h"}:
            continue
        if key not in config:
            errors.append(f"{key} is missing from resolved diagnostic config")
        elif isinstance(expected, tuple) and tuple(config[key]) != expected:
            errors.append(f"{key} must remain {expected!r} (got {config[key]!r})")
        elif not isinstance(expected, tuple) and config[key] != expected:
            errors.append(f"{key} must remain {expected!r} (got {config[key]!r})")
    expected_profile = {
        "node_rx_collision_enabled": True,
        "application_scheduling_mode": "independent_fifo_v1",
        "application_rng_policy": "sha256_seed_node_v1",
        "application_queue_policy": "unbounded_fifo_v1",
        "application_horizon_policy": "right_censor_waiting_v1",
    }
    for key, expected in expected_profile.items():
        if config.get(key) != expected:
            errors.append(f"{key} must be {expected!r} (got {config.get(key)!r})")
    if config.get("terrain_delta_h") != APC_TERRAIN_PROFILE_DIAGNOSTIC_TERRAIN:
        errors.append("terrain_delta_h must be exactly 90 m")
    if config.get("node_ptx_mode") != config.get("relay_ptx_mode") or config.get(
        "node_ptx_mode"
    ) not in APC_TERRAIN_PROFILE_DIAGNOSTIC_MODES:
        errors.append("diagnostic mode must be FF (fixed/fixed) or AA (adaptive/adaptive)")
    if config.get("seed") not in APC_TERRAIN_PROFILE_DIAGNOSTIC_SEEDS:
        errors.append(
            f"seed must be one of {list(APC_TERRAIN_PROFILE_DIAGNOSTIC_SEEDS)}"
        )
    return errors


def validate_diagnostic_spec(spec: Any) -> None:
    config = getattr(spec, "base_config", {})
    if config.get("experiment_profile") != APC_TERRAIN_PROFILE_DIAGNOSTIC_PROFILE:
        return
    errors = []
    factors = getattr(spec, "factors", {})
    expected_factor_names = {"node_ptx_mode", "relay_ptx_mode", "terrain_delta_h"}
    if set(factors) != expected_factor_names:
        errors.append("factor keys must be node_ptx_mode, relay_ptx_mode, terrain_delta_h")
    else:
        if tuple(factors["node_ptx_mode"]) != APC_TERRAIN_PROFILE_DIAGNOSTIC_MODES:
            errors.append("node_ptx_mode factor must be ['fixed', 'adaptive']")
        if tuple(factors["relay_ptx_mode"]) != APC_TERRAIN_PROFILE_DIAGNOSTIC_MODES:
            errors.append("relay_ptx_mode factor must be ['fixed', 'adaptive']")
        if tuple(factors["terrain_delta_h"]) != (APC_TERRAIN_PROFILE_DIAGNOSTIC_TERRAIN,):
            errors.append("terrain_delta_h factor must contain only 90")
    if tuple(getattr(spec, "seeds", ())) != APC_TERRAIN_PROFILE_DIAGNOSTIC_SEEDS:
        errors.append("seeds must be exactly [30, 36, 42, 48, 56]")
    matrix = getattr(spec, "run_matrix", None)
    if matrix is None:
        errors.append("run_matrix must be explicit for the diagnostic experiment")
    else:
        actual = {
            (
                row.get("node_ptx_mode"),
                row.get("relay_ptx_mode"),
                row.get("terrain_delta_h"),
                row.get("seed"),
            )
            for row in matrix
        }
        if len(matrix) != APC_TERRAIN_PROFILE_DIAGNOSTIC_RUN_COUNT:
            errors.append("run_matrix must contain exactly 10 rows")
        if actual != _EXPECTED_ROWS:
            errors.append("run_matrix must contain exactly FF and AA for each of five seeds at Δh=90")
    if config.get("node_rx_collision_enabled") is not True:
        errors.append("base_config must enable node_rx_collision_enabled")
    for key, expected in (
        ("application_scheduling_mode", "independent_fifo_v1"),
        ("application_rng_policy", "sha256_seed_node_v1"),
        ("application_queue_policy", "unbounded_fifo_v1"),
        ("application_horizon_policy", "right_censor_waiting_v1"),
    ):
        if config.get(key) != expected:
            errors.append(f"base_config {key} must be {expected!r}")
    if errors:
        raise ValueError("Diagnostic experiment contract violation: " + "; ".join(errors))


def diagnostic_run_tables(config: Mapping[str, Any]) -> dict[str, str]:
    return dict(_DIAGNOSTIC_TABLES)


def validate_diagnostic_outputs(
    config: Mapping[str, Any], summary: Mapping[str, Any], outputs: Mapping[str, Any]
) -> None:
    violations = diagnostic_config_violations(config)
    if violations:
        raise ValueError("Diagnostic run configuration violation: " + "; ".join(violations))
    missing_fields = sorted(PAPER1_APC_REQUIRED_SUMMARY_FIELDS.difference(summary))
    missing_tables = sorted(PAPER1_APC_REQUIRED_OUTPUT_TABLES.difference(outputs))
    missing_tables.extend(sorted(set(_DIAGNOSTIC_TABLES).difference(outputs)))
    if missing_fields or missing_tables:
        raise ValueError(
            "Diagnostic output missing required evidence: "
            f"summary_fields={missing_fields}, tables={sorted(set(missing_tables))}"
        )
    invalid = sorted(
        key for key in set(_DIAGNOSTIC_TABLES) | set(PAPER1_APC_REQUIRED_OUTPUT_TABLES)
        if not isinstance(outputs[key], pd.DataFrame)
    )
    if invalid:
        raise ValueError(f"Diagnostic output tables must be DataFrames: {invalid}")
    for key in ("packet_fate_conservation_delta", "collision_outcome_conservation_delta"):
        if int(summary[key]) != 0:
            raise ValueError(f"Diagnostic invariant {key} must be zero")
    if summary["stage1_status"] != "PASS":
        raise ValueError("Diagnostic stage1_status must be PASS")


__all__ = [
    "APC_TERRAIN_PROFILE_DIAGNOSTIC_AGGREGATION_METRICS",
    "APC_TERRAIN_PROFILE_DIAGNOSTIC_PROFILE",
    "APC_TERRAIN_PROFILE_DIAGNOSTIC_RUN_COUNT",
    "APC_TERRAIN_PROFILE_DIAGNOSTIC_VERSION",
    "diagnostic_config_violations",
    "diagnostic_run_tables",
    "validate_diagnostic_outputs",
    "validate_diagnostic_spec",
]

APC_TERRAIN_PROFILE_DIAGNOSTIC_AGGREGATION_METRICS = PAPER1_APC_AGGREGATION_METRICS
