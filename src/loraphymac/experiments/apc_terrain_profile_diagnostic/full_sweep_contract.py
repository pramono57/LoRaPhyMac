"""Separate 300-run contract for the diagnostic terrain profile."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd

from loraphymac.experiments.apc_terrain_profile_diagnostic.contract import (
    diagnostic_run_tables,
)
from loraphymac.experiments.apc_terrain_sweep.contract import (
    APC_TERRAIN_SWEEP_LOCKED_CONFIG,
    APC_TERRAIN_SWEEP_MODE_VALUES,
    APC_TERRAIN_SWEEP_SEEDS,
    APC_TERRAIN_SWEEP_TERRAIN_DELTA_H_VALUES,
    PAPER1_APC_REQUIRED_OUTPUT_TABLES,
    PAPER1_APC_REQUIRED_SUMMARY_FIELDS,
)
from loraphymac.metrics.instrumentation_metrics import (
    CHANNEL_OCCUPANCY_COLUMNS,
    NODE_TX_DUTY_CYCLE_COLUMNS,
)


APC_TERRAIN_PROFILE_FULL_SWEEP_PROFILE = "apc_terrain_profile_full_sweep_v1"
APC_TERRAIN_PROFILE_FULL_SWEEP_VERSION = "apc_terrain_profile_full_sweep_contract_v1.0"
APC_TERRAIN_PROFILE_FULL_SWEEP_RUN_COUNT = 300
_REQUIRED_PROFILE = {
    "node_rx_collision_enabled": True,
    "application_scheduling_mode": "independent_fifo_v1",
    "application_rng_policy": "sha256_seed_node_v1",
    "application_queue_policy": "unbounded_fifo_v1",
    "application_horizon_policy": "right_censor_waiting_v1",
}
_FULL_SWEEP_TABLES = {
    "node_tx_duty_cycle_df": "node_tx_duty_cycle.csv",
    "channel_occupancy_df": "channel_occupancy.csv",
}
_MODE_PAIRS = (
    ("fixed", "fixed"),
    ("fixed", "adaptive"),
    ("adaptive", "fixed"),
    ("adaptive", "adaptive"),
)
_EXPECTED_ROWS = {
    (node_mode, relay_mode, terrain, seed)
    for node_mode, relay_mode in _MODE_PAIRS
    for terrain in APC_TERRAIN_SWEEP_TERRAIN_DELTA_H_VALUES
    for seed in APC_TERRAIN_SWEEP_SEEDS
}


def full_sweep_config_violations(config: Mapping[str, Any]) -> list[str]:
    errors = []
    if config.get("experiment_profile") != APC_TERRAIN_PROFILE_FULL_SWEEP_PROFILE:
        errors.append("experiment_profile must identify the diagnostic full-sweep contract")
    for key, expected in APC_TERRAIN_SWEEP_LOCKED_CONFIG.items():
        if key in {"node_ptx_mode", "relay_ptx_mode", "terrain_delta_h"}:
            continue
        if key not in config:
            errors.append(f"{key} is missing from resolved config")
        elif isinstance(expected, tuple):
            if tuple(config[key]) != expected:
                errors.append(f"{key} must remain {expected!r} (got {config[key]!r})")
        elif config[key] != expected:
            errors.append(f"{key} must remain {expected!r} (got {config[key]!r})")
    for key, expected in _REQUIRED_PROFILE.items():
        if config.get(key) != expected:
            errors.append(f"{key} must be {expected!r} (got {config.get(key)!r})")
    if config.get("node_ptx_mode") not in APC_TERRAIN_SWEEP_MODE_VALUES:
        errors.append("node_ptx_mode is outside the four official APC modes")
    if config.get("relay_ptx_mode") not in APC_TERRAIN_SWEEP_MODE_VALUES:
        errors.append("relay_ptx_mode is outside the four official APC modes")
    if config.get("terrain_delta_h") not in APC_TERRAIN_SWEEP_TERRAIN_DELTA_H_VALUES:
        errors.append("terrain_delta_h is outside the 15 official terrain values")
    if config.get("seed") not in APC_TERRAIN_SWEEP_SEEDS:
        errors.append("seed is outside the five official seeds")
    return errors


def validate_full_sweep_spec(spec: Any) -> None:
    config = getattr(spec, "base_config", {})
    if config.get("experiment_profile") != APC_TERRAIN_PROFILE_FULL_SWEEP_PROFILE:
        return
    errors = []
    factors = getattr(spec, "factors", {})
    expected_factors = {
        "node_ptx_mode": APC_TERRAIN_SWEEP_MODE_VALUES,
        "relay_ptx_mode": APC_TERRAIN_SWEEP_MODE_VALUES,
        "terrain_delta_h": APC_TERRAIN_SWEEP_TERRAIN_DELTA_H_VALUES,
    }
    if set(factors) != set(expected_factors):
        errors.append("factor keys must match the official terrain sweep")
    else:
        for key, expected in expected_factors.items():
            if tuple(factors[key]) != tuple(expected):
                errors.append(f"{key} factor must exactly match official ordered values")
    if tuple(getattr(spec, "seeds", ())) != tuple(APC_TERRAIN_SWEEP_SEEDS):
        errors.append("seeds must exactly match the official five-seed grid")
    matrix = getattr(spec, "run_matrix", None)
    if matrix is None:
        errors.append("run_matrix must explicitly declare the diagnostic full sweep")
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
        if len(matrix) != APC_TERRAIN_PROFILE_FULL_SWEEP_RUN_COUNT:
            errors.append("run_matrix must contain exactly 300 rows")
        if actual != _EXPECTED_ROWS:
            errors.append("run_matrix is missing, duplicating, or adding official combinations")
    for key, expected in _REQUIRED_PROFILE.items():
        if config.get(key) != expected:
            errors.append(f"base_config {key} must be {expected!r}")
    if errors:
        raise ValueError("Diagnostic full-sweep contract violation: " + "; ".join(errors))


def validate_full_sweep_outputs(
    config: Mapping[str, Any], summary: Mapping[str, Any], outputs: Mapping[str, Any]
) -> None:
    violations = full_sweep_config_violations(config)
    if violations:
        raise ValueError("Diagnostic full-sweep config violation: " + "; ".join(violations))
    missing_fields = sorted(PAPER1_APC_REQUIRED_SUMMARY_FIELDS.difference(summary))
    required_tables = set(PAPER1_APC_REQUIRED_OUTPUT_TABLES) | set(
        full_sweep_run_tables(config)
    )
    missing_tables = sorted(required_tables.difference(outputs))
    if missing_fields or missing_tables:
        raise ValueError(
            "Diagnostic full-sweep output missing evidence: "
            f"summary_fields={missing_fields}, tables={missing_tables}"
        )
    invalid = sorted(
        key for key in required_tables if not isinstance(outputs[key], pd.DataFrame)
    )
    if invalid:
        raise ValueError(f"Diagnostic full-sweep tables must be DataFrames: {invalid}")
    for key, expected_columns in (
        ("node_tx_duty_cycle_df", NODE_TX_DUTY_CYCLE_COLUMNS),
        ("channel_occupancy_df", CHANNEL_OCCUPANCY_COLUMNS),
    ):
        frame = outputs[key]
        if list(frame.columns) != expected_columns:
            raise ValueError(
                f"Diagnostic full-sweep table {key} has invalid schema; "
                f"expected {expected_columns}"
            )
        if frame.empty:
            raise ValueError(f"Diagnostic full-sweep table {key} must not be empty")
        if frame.isna().any().any():
            raise ValueError(f"Diagnostic full-sweep table {key} contains null values")
        numeric_columns = (
            [
                "sim_time_s", "observed_tx_airtime_s", "tx_duty_fraction",
                "tx_duty_cycle_percent",
            ]
            if key == "node_tx_duty_cycle_df"
            else [
                "channel", "sim_time_s", "sum_offered_tx_airtime_s",
                "sum_observed_tx_airtime_s", "channel_busy_union_s",
                "channel_busy_fraction", "channel_busy_percent",
            ]
        )
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            raise ValueError(f"Diagnostic full-sweep table {key} has invalid numbers")
        if any(
            not math.isfinite(float(value))
            for value in numeric.to_numpy().flat
        ):
            raise ValueError(f"Diagnostic full-sweep table {key} has non-finite numbers")
        if key == "node_tx_duty_cycle_df":
            if frame["entity_id"].duplicated().any() or not frame["entity_type"].eq("node").all():
                raise ValueError("Diagnostic full-sweep node duty rows must be unique nodes")
            expected_nodes = int(config.get("node_num", 0))
            if expected_nodes <= 0 or len(frame) != expected_nodes:
                raise ValueError(
                    "Diagnostic full-sweep node duty table must contain one row "
                    f"per configured end node ({expected_nodes})"
                )
            if not frame["measurement_scope"].eq(
                "physical_tx_airtime_over_simulation_horizon"
            ).all():
                raise ValueError("Diagnostic full-sweep node duty scope is invalid")
            if (numeric["sim_time_s"] <= 0).any():
                raise ValueError("Diagnostic full-sweep node duty horizon must be positive")
            if (
                (numeric["observed_tx_airtime_s"] < 0).any()
                or (numeric["tx_duty_fraction"] < 0).any()
                or (numeric["tx_duty_fraction"] > 1).any()
                or (numeric["tx_duty_cycle_percent"] < 0).any()
                or (numeric["tx_duty_cycle_percent"] > 100).any()
            ):
                raise ValueError("Diagnostic full-sweep node duty values are outside valid bounds")
            expected_fraction = (
                numeric["observed_tx_airtime_s"] / numeric["sim_time_s"]
            )
            if not expected_fraction.sub(numeric["tx_duty_fraction"]).abs().le(1e-12).all():
                raise ValueError("Diagnostic full-sweep node duty fraction is inconsistent")
            expected_percent = expected_fraction * 100.0
            if not expected_percent.sub(
                numeric["tx_duty_cycle_percent"]
            ).abs().le(1e-10).all():
                raise ValueError("Diagnostic full-sweep node duty percentage is inconsistent")
        else:
            if frame["channel"].duplicated().any():
                raise ValueError("Diagnostic full-sweep occupancy channels must be unique")
            if not frame["occupancy_method"].eq(
                "union_of_simulated_tx_intervals"
            ).all():
                raise ValueError("Diagnostic full-sweep occupancy method is invalid")
            if (numeric["channel"] < 0).any() or not numeric["channel"].map(
                lambda value: float(value).is_integer()
            ).all():
                raise ValueError("Diagnostic full-sweep occupancy channel ID is invalid")
            if (numeric["sim_time_s"] <= 0).any():
                raise ValueError("Diagnostic full-sweep occupancy horizon must be positive")
            if (
                (numeric["sum_offered_tx_airtime_s"] < 0).any()
                or (numeric["sum_observed_tx_airtime_s"] < 0).any()
                or (
                    numeric["sum_observed_tx_airtime_s"]
                    > numeric["sum_offered_tx_airtime_s"] + 1e-12
                ).any()
                or (numeric["channel_busy_union_s"] < 0).any()
                or (
                    numeric["channel_busy_union_s"]
                    > numeric["sum_observed_tx_airtime_s"] + 1e-12
                ).any()
            ):
                raise ValueError("Diagnostic full-sweep occupancy airtime is inconsistent")
            if (
                (numeric["channel_busy_fraction"] < 0).any()
                or (numeric["channel_busy_fraction"] > 1).any()
                or (numeric["channel_busy_percent"] < 0).any()
                or (numeric["channel_busy_percent"] > 100).any()
            ):
                raise ValueError("Diagnostic full-sweep occupancy is outside [0, 100]")
            fraction = numeric["channel_busy_union_s"] / numeric["sim_time_s"]
            if not fraction.sub(numeric["channel_busy_fraction"]).abs().le(1e-12).all():
                raise ValueError("Diagnostic full-sweep occupancy fraction is inconsistent")
            percent = fraction * 100.0
            if not percent.sub(numeric["channel_busy_percent"]).abs().le(1e-10).all():
                raise ValueError("Diagnostic full-sweep occupancy percentage is inconsistent")
    for key in ("packet_fate_conservation_delta", "collision_outcome_conservation_delta"):
        if int(summary[key]) != 0:
            raise ValueError(f"Diagnostic full-sweep invariant {key} must be zero")
    if summary["stage1_status"] != "PASS":
        raise ValueError("Diagnostic full-sweep stage1_status must be PASS")


def full_sweep_run_tables(config: Mapping[str, Any]) -> dict[str, str]:
    return {**diagnostic_run_tables(config), **_FULL_SWEEP_TABLES}


__all__ = [
    "APC_TERRAIN_PROFILE_FULL_SWEEP_PROFILE",
    "APC_TERRAIN_PROFILE_FULL_SWEEP_RUN_COUNT",
    "APC_TERRAIN_PROFILE_FULL_SWEEP_VERSION",
    "full_sweep_config_violations",
    "full_sweep_run_tables",
    "validate_full_sweep_outputs",
    "validate_full_sweep_spec",
]
