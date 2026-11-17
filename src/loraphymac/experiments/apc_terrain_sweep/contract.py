"""Executable scientific contract for the APC mode/terrain sweep.

The contract is publication-independent: it defines the grid, aggregation
fields, retained evidence, and post-run invariants for any publication that
uses this experiment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any


PAPER1_APC_PROFILE = "paper1_apc_v1"
PAPER1_APC_CONTRACT_VERSION = "paper1_apc_contract_v1.0"

APC_TERRAIN_SWEEP_PROFILE = "apc_terrain_sweep_v1"
APC_TERRAIN_SWEEP_CONTRACT_VERSION = "apc_terrain_sweep_contract_v1.0"
APC_TERRAIN_SWEEP_SUPPORTED_PROFILES = frozenset(
    {APC_TERRAIN_SWEEP_PROFILE, PAPER1_APC_PROFILE}
)

PAPER1_APC_TERRAIN_DELTA_H_VALUES = tuple(range(20, 161, 10))
PAPER1_APC_SEEDS = (30, 36, 42, 48, 56)
PAPER1_APC_COMPARISON_TERRAIN_DELTA_H = 90
PAPER1_APC_MODE_VALUES = ("fixed", "adaptive")
PAPER1_APC_EXPECTED_RUN_COUNT = 300


def _is_apc_terrain_sweep_profile(value: Any) -> bool:
    return value in APC_TERRAIN_SWEEP_SUPPORTED_PROFILES


def apc_terrain_sweep_contract_version(profile: Any) -> str:
    if profile == PAPER1_APC_PROFILE:
        return PAPER1_APC_CONTRACT_VERSION
    return APC_TERRAIN_SWEEP_CONTRACT_VERSION

# These values reproduce the 300-run experiment recorded by the current
# outputs_ch6 manifest/table, while applying the current EIRP implementation.
# Operational output switches and the four experimental variables are listed
# separately below.
PAPER1_APC_LOCKED_CONFIG = MappingProxyType(
    {
        "area_x": 45_000,
        "area_y": 15_000,
        "sim_time": 3_600,
        "routing_assumed_sf": 7,
        "routing_ch_id": 0,
        "routing_hop_penalty_alpha": 2,
        "feasible_margin_db": 0,
        "delta_snr": 3,
        "collision_mode": "capture",
        "capture_threshold_db": 6.0,
        "collision_enforce_sf_match": True,
        "collision_sf_interference_mode": "sir_matrix",
        "collision_sir_matrix_preset": "rakibul_islam_2023_table4",
        "collision_sir_matrix": None,
        "collision_sir_matrix_version": None,
        "collision_noise_floor_dbm": -120.0,
        "collision_destructive": True,
        "node_max_eirp_dbm": 20.0,
        "relay_max_eirp_dbm": 26.0,
        "gw_max_eirp_dbm": 26.0,
        "node_feeder_loss_db": 0.0,
        "relay_feeder_loss_db": 0.0,
        "gw_feeder_loss_db": 0.0,
        "frame_enabled": True,
        "frame_validation_enabled": False,
        "frame_version": 0,
        "frame_default_ack_req": False,
        "apc_downlink_mode": "physical_all",
        "apc_fopts_size": 4,
        "apc_downlink_delay_s": 1.0,
        "apc_retry_timeout_s": 5.0,
        "apc_downlink_ptx_dbm": None,
        "apc_downlink_node_ptx_dbm": None,
        "apc_downlink_relay_ptx_dbm": None,
        "node_rx_window_enabled": True,
        "node_rx1_delay_s": 1.0,
        "node_rx_window_duration_s": 1.0,
        "include_mcu_energy": False,
        "node_num": 100,
        "node_sf_list": (7,),
        "node_bw": 125_000,
        "node_ptx_init": None,
        "node_payload_size": 50,
        "node_period": 60,
        "node_jitter": 2,
        "node_ant_height": 2.0,
        "node_ant_gain": 2.15,
        "node_target_margin_db": 6.0,
        "relay_num": 2,
        "relay_sf": 7,
        "relay_bw_hz": 125_000,
        "relay_ptx_init": None,
        "relay_ant_height": 25.0,
        "relay_ant_gain": 5.8,
        "relay_tx_ch_ids": (1,),
        "relay_tx_waiting_queue_capacity_packets": None,
        "gateway_tx_waiting_queue_capacity_packets": None,
        "tx_queue_overflow_policy": "drop_tail",
        "relay_duty_cycle": None,
        "rx_idle_poll_interval_ms": 2.0,
        "max_rx_idle_wait_ms": 3_000.0,
        "relay_target_margin_db": 6.0,
        "gw_ant_height": 25.0,
        "gw_ant_gain": 5.8,
    }
)

# E0 predates the explicit accounting selector. Missing therefore means the
# historical legacy behavior; an explicit non-legacy value is contract drift.
PAPER1_APC_LEGACY_DEFAULT_CONFIG = MappingProxyType(
    {
        "energy_accounting_mode": "legacy",
        "application_scheduling_mode": "legacy_post_tx_done_v1",
        "application_rng_policy": "legacy_global_random_v1",
        "application_queue_policy": "none",
        "application_horizon_policy": "physical_tx_offer_v1",
        "application_initial_delay_max_s": 30.0,
        "node_rx_collision_enabled": False,
        "routing_strategy": "terrain_specific",
        "routing_reference_delta_h": None,
        "routing_snapshot_manifest": None,
        "routing_snapshot_manifest_sha256": None,
    }
)

# These keys belong to the E1 comparison profile and are intentionally not E0
# invariants. Their semantics are validated by the apc_compare_model contract.
PAPER1_APC_E1_ONLY_CONFIG_KEYS = frozenset(
    {
        "tx_power_policy_id",
        "tx_power_model_id",
        "tx_power_calibration_id",
        "tx_power_calibration_manifest",
        "tx_power_calibration_manifest_sha256",
        "tx_power_calibration_margin_db",
    }
)

PAPER1_APC_VARIABLE_CONFIG_KEYS = frozenset(
    {"terrain_delta_h", "seed", "node_ptx_mode", "relay_ptx_mode"}
)
PAPER1_APC_OPERATIONAL_CONFIG_KEYS = frozenset(
    {
        "output_dir",
        "output_profile",
        "export_summary",
        "export_results",
        "export_debug",
        "export_internal",
        "print_summary",
        "print_detail",
        "print_debug",
        "print_internal",
        "debug_preview_rows",
        "plot_routing",
        "verbose",
        "run_id",
    }
)

PAPER1_APC_FACTOR_KEYS = (
    "node_ptx_mode",
    "relay_ptx_mode",
    "terrain_delta_h",
)

PAPER1_APC_AGGREGATION_METRICS = (
    "delivery_by_horizon_percent",
    "terminal_loss_percent",
    "censored_fraction_percent",
    "served_node_ratio_95_percent",
    "avg_e2e_delay_ms",
    "total_node_energy_J",
    "total_relay_energy_J",
    "total_gateway_energy_J",
    "total_network_energy_J",
    "total_constrained_energy_J",
    "energy_per_delivered_packet_J",
    "energy_per_delivered_packet_constrained_J",
    "avg_final_node_ptx",
    "avg_final_relay_ptx",
    "avg_final_node_decision_margin",
    "avg_final_node_apc_rx_margin",
    "avg_final_relay_decision_margin",
    "avg_final_relay_apc_rx_margin",
    "apc_total_tx",
    "apc_total_rx",
    "apc_total_drop",
    "apc_apply_success_percent",
    "apc_down_node_success_percent",
    "apc_down_relay_success_percent",
    "ptx_update_count",
    "packet_fate_conservation_delta",
    "collision_outcome_conservation_delta",
)

PAPER1_APC_ALWAYS_RUN_TABLES = MappingProxyType(
    {
        "energy_breakdown_df": "energy_breakdown.csv",
        "final_ptx_distribution_df": "final_ptx_distribution.csv",
        "frame_summary_df": "frame_summary.csv",
        "run_metadata_df": "run_metadata.csv",
    }
)
PAPER1_APC_COMPARISON_RUN_TABLES = MappingProxyType(
    {
        "ptx_history_df": "ptx_history.csv",
        "margin_history_df": "margin_history.csv",
        "apc_control_history_df": "apc_control_history.csv",
    }
)

PAPER1_APC_REQUIRED_OUTPUT_TABLES = frozenset(
    {
        *PAPER1_APC_ALWAYS_RUN_TABLES,
        *PAPER1_APC_COMPARISON_RUN_TABLES,
        "packet_fate_summary_df",
    }
)

PAPER1_APC_REQUIRED_SUMMARY_FIELDS = frozenset(
    {
        "delivery_by_horizon_percent",
        "terminal_loss_percent",
        "censored_fraction_percent",
        "avg_e2e_delay_ms",
        "total_node_energy_J",
        "total_relay_energy_J",
        "total_gateway_energy_J",
        "total_network_energy_J",
        "total_constrained_energy_J",
        "energy_per_delivered_packet_J",
        "energy_per_delivered_packet_constrained_J",
        "avg_final_node_ptx",
        "avg_final_relay_ptx",
        "apc_total_tx",
        "apc_total_rx",
        "apc_total_drop",
        "ptx_update_count",
        "stage1_status",
        "packet_fate_conservation_delta",
        "collision_outcome_conservation_delta",
    }
)


class Paper1APCContractError(ValueError):
    """Raised when a Paper 1 design or result violates its frozen contract."""


def _equivalent(actual: Any, expected: Any) -> bool:
    if isinstance(expected, tuple):
        if isinstance(actual, (str, bytes)) or not isinstance(actual, Sequence):
            return False
        return tuple(actual) == expected
    return actual == expected


def paper1_apc_config_violations(config: Mapping[str, Any]) -> list[str]:
    """Return all Paper 1 per-run configuration violations."""
    violations = []
    for key, expected in PAPER1_APC_LOCKED_CONFIG.items():
        if key not in config:
            violations.append(f"{key} is missing (expected {expected!r})")
            continue
        if not _equivalent(config[key], expected):
            violations.append(f"{key} must remain {expected!r} (got {config[key]!r})")
    for key, expected in PAPER1_APC_LEGACY_DEFAULT_CONFIG.items():
        actual = config.get(key, expected)
        if not _equivalent(actual, expected):
            violations.append(f"{key} must remain {expected!r} (got {actual!r})")

    terrain = config.get("terrain_delta_h")
    if terrain not in PAPER1_APC_TERRAIN_DELTA_H_VALUES:
        violations.append(
            "terrain_delta_h must be one of "
            f"{list(PAPER1_APC_TERRAIN_DELTA_H_VALUES)} (got {terrain!r})"
        )
    seed = config.get("seed")
    if seed not in PAPER1_APC_SEEDS:
        violations.append(f"seed must be one of {list(PAPER1_APC_SEEDS)} (got {seed!r})")
    for key in ("node_ptx_mode", "relay_ptx_mode"):
        if config.get(key) not in PAPER1_APC_MODE_VALUES:
            violations.append(
                f"{key} must be one of {list(PAPER1_APC_MODE_VALUES)} (got {config.get(key)!r})"
            )
    return violations


def validate_paper1_apc_experiment_spec(spec: Any) -> None:
    """Validate the complete 4-mode × 15-terrain × 5-seed Paper 1 design."""
    base_config = getattr(spec, "base_config", {})
    factors = getattr(spec, "factors", {})
    seeds = tuple(getattr(spec, "seeds", ()))

    if not _is_apc_terrain_sweep_profile(base_config.get("experiment_profile")):
        return

    errors = []
    missing_locked = sorted(set(PAPER1_APC_LOCKED_CONFIG).difference(base_config))
    if missing_locked:
        errors.append(
            f"base_config must explicitly freeze all locked keys; missing {missing_locked}"
        )
    for key, expected in PAPER1_APC_LEGACY_DEFAULT_CONFIG.items():
        if key in base_config and not _equivalent(base_config[key], expected):
            errors.append(f"{key} must remain {expected!r} (got {base_config[key]!r})")
    unexpected_factors = set(factors).symmetric_difference(PAPER1_APC_FACTOR_KEYS)
    if unexpected_factors:
        errors.append(
            f"factor keys must be exactly {list(PAPER1_APC_FACTOR_KEYS)} "
            f"(difference {sorted(unexpected_factors)})"
        )
    else:
        if tuple(factors["node_ptx_mode"]) != PAPER1_APC_MODE_VALUES:
            errors.append(f"node_ptx_mode factor must be {list(PAPER1_APC_MODE_VALUES)}")
        if tuple(factors["relay_ptx_mode"]) != PAPER1_APC_MODE_VALUES:
            errors.append(f"relay_ptx_mode factor must be {list(PAPER1_APC_MODE_VALUES)}")
        if tuple(factors["terrain_delta_h"]) != PAPER1_APC_TERRAIN_DELTA_H_VALUES:
            errors.append("terrain_delta_h factor must be the ordered 20..160 m sweep")
    if seeds != PAPER1_APC_SEEDS:
        errors.append(f"seeds must be exactly {list(PAPER1_APC_SEEDS)}")
    if float(getattr(spec, "confidence_level", 0.0)) != 0.95:
        errors.append("confidence_level must be 0.95")
    if bool(getattr(spec, "retain_debug_tables", False)):
        errors.append(
            "retain_debug_tables must be false; the contract retains selected "
            "Paper 1 evidence separately"
        )

    if errors:
        raise Paper1APCContractError("Paper 1 APC grid contract violation: " + "; ".join(errors))


def paper1_apc_run_tables(config: Mapping[str, Any]) -> dict[str, str]:
    """Return paper-specific tables retained for this resolved run."""
    if not _is_apc_terrain_sweep_profile(config.get("experiment_profile")):
        return {}
    tables = dict(PAPER1_APC_ALWAYS_RUN_TABLES)
    if config.get("terrain_delta_h") == PAPER1_APC_COMPARISON_TERRAIN_DELTA_H:
        tables.update(PAPER1_APC_COMPARISON_RUN_TABLES)
    return tables


def _finite_nonnegative(summary: Mapping[str, Any], key: str) -> float:
    try:
        value = float(summary[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise Paper1APCContractError(f"Paper 1 APC output field {key!r} must be numeric") from exc
    if not math.isfinite(value) or value < 0.0:
        raise Paper1APCContractError(
            f"Paper 1 APC output field {key!r} must be finite and non-negative"
        )
    return value


def _close(actual: float, expected: float, tolerance: float = 1e-5) -> bool:
    return abs(float(actual) - float(expected)) <= tolerance


def validate_paper1_apc_outputs(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> None:
    """Validate evidence and invariants for one completed Paper 1 run."""
    if not _is_apc_terrain_sweep_profile(config.get("experiment_profile")):
        return
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is a project dependency
        raise Paper1APCContractError("pandas is required for output validation") from exc

    missing_fields = sorted(PAPER1_APC_REQUIRED_SUMMARY_FIELDS.difference(summary))
    if missing_fields:
        raise Paper1APCContractError(
            f"Paper 1 APC summary is missing required fields: {missing_fields}"
        )
    missing_tables = sorted(PAPER1_APC_REQUIRED_OUTPUT_TABLES.difference(outputs))
    if missing_tables:
        raise Paper1APCContractError(
            f"Paper 1 APC outputs are missing required tables: {missing_tables}"
        )
    invalid_tables = sorted(
        name
        for name in PAPER1_APC_REQUIRED_OUTPUT_TABLES
        if not isinstance(outputs[name], pd.DataFrame)
    )
    if invalid_tables:
        raise Paper1APCContractError(f"Paper 1 APC outputs must be DataFrames: {invalid_tables}")

    percentage_fields = (
        "delivery_by_horizon_percent",
        "terminal_loss_percent",
        "censored_fraction_percent",
    )
    percentages = [_finite_nonnegative(summary, key) for key in percentage_fields]
    if any(value > 100.0 for value in percentages):
        raise Paper1APCContractError("Paper 1 packet-fate percentages must be within 0..100")
    if not _close(sum(percentages), 100.0, tolerance=0.01):
        raise Paper1APCContractError("Paper 1 packet-fate percentages must sum to 100%")

    energy_fields = (
        "avg_e2e_delay_ms",
        "total_node_energy_J",
        "total_relay_energy_J",
        "total_gateway_energy_J",
        "total_network_energy_J",
        "total_constrained_energy_J",
        "energy_per_delivered_packet_J",
        "energy_per_delivered_packet_constrained_J",
        "avg_final_node_ptx",
        "avg_final_relay_ptx",
    )
    values = {key: _finite_nonnegative(summary, key) for key in energy_fields}
    if not _close(
        values["total_network_energy_J"],
        values["total_node_energy_J"]
        + values["total_relay_energy_J"]
        + values["total_gateway_energy_J"],
    ):
        raise Paper1APCContractError(
            "Paper 1 total_network_energy_J does not conserve component energy"
        )
    if not _close(
        values["total_constrained_energy_J"],
        values["total_node_energy_J"] + values["total_relay_energy_J"],
    ):
        raise Paper1APCContractError(
            "Paper 1 total_constrained_energy_J must equal node plus relay energy"
        )

    for key in ("packet_fate_conservation_delta", "collision_outcome_conservation_delta"):
        if int(summary[key]) != 0:
            raise Paper1APCContractError(f"Paper 1 invariant {key} must be zero")
    if summary["stage1_status"] != "PASS":
        raise Paper1APCContractError(
            f"Paper 1 APC stage1_status must be PASS, got {summary['stage1_status']!r}"
        )

    apc_tx = int(summary["apc_total_tx"])
    apc_rx = int(summary["apc_total_rx"])
    apc_drop = int(summary["apc_total_drop"])
    update_count = int(summary["ptx_update_count"])
    if min(apc_tx, apc_rx, apc_drop, update_count) < 0:
        raise Paper1APCContractError("Paper 1 APC counters must be non-negative")
    if apc_tx != apc_rx + apc_drop:
        raise Paper1APCContractError(
            "Paper 1 APC accounting requires apc_total_tx = apc_total_rx + apc_total_drop"
        )
    if apc_rx != update_count:
        raise Paper1APCContractError(
            "Paper 1 APC accounting requires apc_total_rx = ptx_update_count"
        )

    history = outputs["ptx_history_df"]
    if len(history) != update_count:
        raise Paper1APCContractError("Paper 1 ptx_history row count must equal ptx_update_count")
    node_mode = config["node_ptx_mode"]
    relay_mode = config["relay_ptx_mode"]
    if node_mode == "fixed" and not history.empty:
        if (history["entity_type"].astype(str) == "node").any():
            raise Paper1APCContractError("fixed node mode must not contain node PTx updates")
    if relay_mode == "fixed" and not history.empty:
        if (history["entity_type"].astype(str) == "relay").any():
            raise Paper1APCContractError("fixed relay mode must not contain relay PTx updates")
    if node_mode == relay_mode == "fixed" and any(
        value != 0 for value in (apc_tx, apc_rx, apc_drop, update_count)
    ):
        raise Paper1APCContractError(
            "Paper 1 fixed/fixed baseline must not generate APC traffic or PTx updates"
        )

    if node_mode == "fixed" and not _close(values["avg_final_node_ptx"], 17.0):
        raise Paper1APCContractError("fixed node PTx must remain at 17 dBm")
    if relay_mode == "fixed" and not _close(values["avg_final_relay_ptx"], 20.0):
        raise Paper1APCContractError("fixed relay PTx must remain at 20 dBm")

    metadata = outputs["run_metadata_df"]
    expected_contract_version = apc_terrain_sweep_contract_version(
        config.get("experiment_profile")
    )
    if metadata.empty or metadata.iloc[0].get("experiment_contract_version") != (
        expected_contract_version
    ):
        raise Paper1APCContractError(
            "Paper 1 run metadata has a missing or incorrect experiment contract version"
        )


# Content-based public names. The Paper 1 names above remain compatibility
# aliases for the completed historical dataset and existing callers.
APC_TERRAIN_SWEEP_TERRAIN_DELTA_H_VALUES = PAPER1_APC_TERRAIN_DELTA_H_VALUES
APC_TERRAIN_SWEEP_SEEDS = PAPER1_APC_SEEDS
APC_TERRAIN_SWEEP_COMPARISON_TERRAIN_DELTA_H = PAPER1_APC_COMPARISON_TERRAIN_DELTA_H
APC_TERRAIN_SWEEP_MODE_VALUES = PAPER1_APC_MODE_VALUES
APC_TERRAIN_SWEEP_EXPECTED_RUN_COUNT = PAPER1_APC_EXPECTED_RUN_COUNT
APC_TERRAIN_SWEEP_LOCKED_CONFIG = PAPER1_APC_LOCKED_CONFIG
APC_TERRAIN_SWEEP_FACTOR_KEYS = PAPER1_APC_FACTOR_KEYS
APC_TERRAIN_SWEEP_AGGREGATION_METRICS = PAPER1_APC_AGGREGATION_METRICS
APC_TERRAIN_SWEEP_ALWAYS_RUN_TABLES = PAPER1_APC_ALWAYS_RUN_TABLES
APC_TERRAIN_SWEEP_COMPARISON_RUN_TABLES = PAPER1_APC_COMPARISON_RUN_TABLES
APCTerrainSweepContractError = Paper1APCContractError
apc_terrain_sweep_config_violations = paper1_apc_config_violations
validate_apc_terrain_sweep_experiment_spec = validate_paper1_apc_experiment_spec
apc_terrain_sweep_run_tables = paper1_apc_run_tables
validate_apc_terrain_sweep_outputs = validate_paper1_apc_outputs
