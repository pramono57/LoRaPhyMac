"""Versioned contract for the isolated R2.6 feedback robustness sweep."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd

APC_FEEDBACK_ROBUSTNESS_PROFILE = "apc_feedback_robustness_sweep_v1"
APC_FEEDBACK_ROBUSTNESS_VERSION = "apc_feedback_robustness_contract_v1.1"
APC_FEEDBACK_ROBUSTNESS_RUN_COUNT = 30
APC_FEEDBACK_ROBUSTNESS_SEEDS = (30, 36, 42, 48, 56)
APC_FEEDBACK_ROBUSTNESS_SCENARIOS = tuple(
    (period, loss)
    for period in (30, 90)
    for loss in (0.0, 0.1, 0.3)
)

ATTEMPT_COLUMNS = (
    "run_id", "feedback_id", "physical_attempt_id", "attempt_number",
    "controller_id", "target_transmitter_id", "transmitter_role", "mtype",
    "tx_timestamp_s", "channel_id", "tx_airtime_s", "ptx_dbm",
    "native_reception_status", "native_failure_reason", "logical_failure_reason",
    "synthetic_loss_probability", "synthetic_erasure_decision",
    "logical_delivery_status", "tpi_application_status", "tpi_version",
)
TRANSACTION_COLUMNS = (
    "run_id", "decision_id", "feedback_id", "target_transmitter_id",
    "controller_id", "decision_timestamp_s", "desired_tpi",
    "tpi_version", "tpi_state_changed", "physical_attempts", "native_successful_receptions",
    "synthetic_erasures", "application_timestamp_s", "application_latency_s",
    "retry_exhausted", "status",
)
STALE_TPI_TABLE_KEY = "stale_tpi_transmission_summary_df"
# These metadata are added by multiseed_pipeline._annotate_table when the
# scientific table is persisted. ``run_id`` is already a scientific key, so
# it appears only once in the complete on-disk schema.
STALE_TPI_METADATA_COLUMNS = (
    "experiment_id", "model_id", "revision", "energy_accounting_model_version",
    "seed", "node_period", "feedback_additional_loss_probability",
)
STALE_TPI_COLUMNS = (
    "run_id", "transmitter_id", "transmitter_role", "mtype", "control_scope_id",
    "total_data_physical_tx_count", "pre_decision_tx_count",
    "post_decision_data_tx_count", "eligible_data_tx_count",
    "stale_tpi_tx_count", "current_tpi_tx_count", "unclassified_tx_count",
    "version_invariant_violation_count", "stale_tpi_transmission_fraction",
)
STALE_TPI_PERSISTED_COLUMNS = (
    *STALE_TPI_METADATA_COLUMNS,
    *STALE_TPI_COLUMNS,
)
CONVERGENCE_COLUMNS = (
    "run_id", "feedback_id", "target_transmitter_id", "tpi_version",
    "decision_timestamp_s", "application_timestamp_s", "observation_timestamp_s",
    "margin_db", "margin_first_entry_latency_s", "observation_status",
)
AIRTIME_COLUMNS = (
    "run_id", "apc_control_tx_count", "apc_control_airtime_s",
    "total_observed_tx_count", "total_observed_tx_airtime_s",
    "apc_control_airtime_percent", "by_transmitter_role_json", "by_mtype_json",
)
APC_FEEDBACK_ROBUSTNESS_RUN_TABLES = {
    "feedback_attempt_metrics_df": "feedback_attempt_metrics.csv",
    "feedback_transaction_metrics_df": "feedback_transaction_metrics.csv",
    STALE_TPI_TABLE_KEY: "stale_tpi_by_transmitter.csv",
    "apc_convergence_metrics_df": "apc_convergence_metrics.csv",
    "apc_control_airtime_summary_df": "apc_control_airtime_summary.csv",
}
_LOCKED_CONFIG = {
    "output_profile": "summary", "persistence_profile": "scientific",
    "gzip_scientific_tables": False, "sim_time": 3600, "node_num": 100, "area_x": 45000,
    "area_y": 15000, "terrain_delta_h": 90, "node_sf_list": [7],
    "node_bw": 125000, "node_payload_size": 50, "node_jitter": 2,
    "node_ptx_mode": "adaptive", "node_target_margin_db": 6.0,
    "relay_num": 2, "relay_sf": 7, "relay_bw_hz": 125000,
    "relay_ptx_mode": "adaptive", "relay_duty_cycle": None,
    "relay_tx_waiting_queue_capacity_packets": None,
    "tx_queue_overflow_policy": "drop_tail", "collision_mode": "capture",
    "collision_enforce_sf_match": True,
    "collision_sf_interference_mode": "sir_matrix",
    "collision_sir_matrix_preset": "rakibul_islam_2023_table4",
    "node_rx_window_enabled": True, "node_rx_collision_enabled": True,
    "frame_default_ack_req": False, "energy_accounting_mode": "corrected",
    "tx_power_policy_id": "legacy",
    "application_scheduling_mode": "independent_fifo_v1",
    "application_rng_policy": "sha256_seed_node_v1",
    "application_queue_policy": "unbounded_fifo_v1",
    "application_horizon_policy": "right_censor_waiting_v1",
}
TABLE_SCHEMAS = {
    "feedback_attempt_metrics_df": ATTEMPT_COLUMNS,
    "feedback_transaction_metrics_df": TRANSACTION_COLUMNS,
    STALE_TPI_TABLE_KEY: STALE_TPI_COLUMNS,
    "apc_convergence_metrics_df": CONVERGENCE_COLUMNS,
    "apc_control_airtime_summary_df": AIRTIME_COLUMNS,
}


def feedback_robustness_config_violations(config: Mapping[str, Any]) -> list[str]:
    if config.get("experiment_profile") != APC_FEEDBACK_ROBUSTNESS_PROFILE:
        return []
    errors = []
    if config.get("node_ptx_mode") != "adaptive" or config.get("relay_ptx_mode") != "adaptive":
        errors.append("node_ptx_mode and relay_ptx_mode must be adaptive")
    if config.get("node_num") != 100 or config.get("relay_num") != 2:
        errors.append("sweep requires 100 nodes and exactly two relays")
    for key, expected in _LOCKED_CONFIG.items():
        if config.get(key) != expected:
            errors.append(f"{key} must remain {expected!r}")
    if config.get("node_period") not in {30, 90}:
        errors.append("node_period must be 30 or 90 seconds")
    p = config.get("feedback_additional_loss_probability")
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(float(p)) or p not in {0, 0.0, .1, .3}:
        errors.append("feedback_additional_loss_probability must be 0.0, 0.1, or 0.3")
    if config.get("seed") not in APC_FEEDBACK_ROBUSTNESS_SEEDS:
        errors.append("seed must be one of 30, 36, 42, 48, 56")
    return errors


def validate_feedback_robustness_spec(spec: Any) -> None:
    config = getattr(spec, "base_config", {})
    if config.get("experiment_profile") != APC_FEEDBACK_ROBUSTNESS_PROFILE:
        return
    factors = getattr(spec, "factors", {})
    expected = {
        "node_period": (30, 90),
        "feedback_additional_loss_probability": (0.0, 0.1, 0.3),
    }
    if set(factors) != set(expected) or any(tuple(factors[k]) != v for k, v in expected.items()):
        raise ValueError("feedback robustness factors must be the approved 2x3 grid")
    if tuple(getattr(spec, "seeds", ())) != APC_FEEDBACK_ROBUSTNESS_SEEDS:
        raise ValueError("feedback robustness seeds must be the approved five seeds")
    if getattr(spec, "run_annotations", {}).get("model_id") != "aa":
        raise ValueError("feedback robustness run_annotations.model_id must be 'aa'")
    if getattr(spec, "run_annotations", {}).get("energy_accounting_model_version") != "eq10_exclusive_transition_v1":
        raise ValueError("run_annotations must pin eq10_exclusive_transition_v1")
    from loraphymac.metrics.energy_metrics import energy_accounting_model_version

    actual_energy_version = energy_accounting_model_version("corrected")
    if actual_energy_version != "eq10_exclusive_transition_v1":
        raise ValueError(
            "corrected energy implementation changed: expected pinned version "
            f"eq10_exclusive_transition_v1, got {actual_energy_version!r}"
        )
    if getattr(spec, "run_matrix", None) is not None:
        raise ValueError("feedback robustness contract requires Cartesian factors")
    if len(getattr(spec, "factors", {})) != 2:
        raise ValueError("feedback robustness spec is invalid")


def feedback_robustness_run_tables(config: Mapping[str, Any]) -> dict[str, str]:
    return dict(APC_FEEDBACK_ROBUSTNESS_RUN_TABLES)


def validate_feedback_robustness_outputs(config, summary, outputs) -> None:
    violations = feedback_robustness_config_violations(config)
    required_summary = {
        "pdr_percent", "packet_fate_delivered_count", "avg_e2e_delay_ms",
        "total_network_energy_J", "energy_per_delivered_packet_J",
        "packet_fate_conservation_delta", "collision_outcome_conservation_delta",
    }
    missing_summary = required_summary - set(summary)
    missing = set(APC_FEEDBACK_ROBUSTNESS_RUN_TABLES) - set(outputs)
    if violations or missing or missing_summary:
        raise ValueError(
            "feedback robustness output contract violation: "
            f"config={violations}; summary={sorted(missing_summary)}; "
            f"tables={sorted(missing)}"
        )
    pdr = float(summary["pdr_percent"])
    if not math.isfinite(pdr) or not 0.0 <= pdr <= 100.0:
        raise ValueError("pdr_percent must be finite and in [0, 100]")
    for key, columns in TABLE_SCHEMAS.items():
        frame = outputs[key]
        if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != tuple(columns):
            raise ValueError(f"{key} missing or has invalid schema")
        for column in frame.select_dtypes(include="number"):
            values = frame[column].dropna()
            if not values.map(lambda value: math.isfinite(float(value))).all():
                raise ValueError(f"{key}.{column} contains non-finite values")
    stale = outputs[STALE_TPI_TABLE_KEY]
    if stale.duplicated(
        ["run_id", "transmitter_id", "transmitter_role", "mtype", "control_scope_id"],
        keep=False,
    ).any():
        raise ValueError("stale-TPI transmitter rows must have unique grouping keys")
    for row in stale.itertuples(index=False):
        if row.total_data_physical_tx_count != (
            row.pre_decision_tx_count + row.post_decision_data_tx_count
        ):
            raise ValueError("stale-TPI total/pre/post accounting invariant failed")
        if row.post_decision_data_tx_count != (
            row.eligible_data_tx_count + row.unclassified_tx_count
        ):
            raise ValueError("stale-TPI post-decision accounting invariant failed")
        if row.eligible_data_tx_count != (
            row.stale_tpi_tx_count + row.current_tpi_tx_count
        ):
            raise ValueError("stale-TPI eligible accounting invariant failed")
        if row.version_invariant_violation_count > row.unclassified_tx_count:
            raise ValueError("stale-TPI version invariant count exceeds unclassified count")
        fraction = row.stale_tpi_transmission_fraction
        if row.eligible_data_tx_count == 0:
            if not pd.isna(fraction):
                raise ValueError("stale-TPI fraction must be null when denominator is zero")
        elif not math.isclose(
            float(fraction),
            row.stale_tpi_tx_count / row.eligible_data_tx_count,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("stale-TPI fraction disagrees with its counts")
    if len(outputs["apc_control_airtime_summary_df"]) != 1:
        raise ValueError("APC airtime summary must contain exactly one row")


def validate_feedback_robustness_grid(spec, runs) -> None:
    if getattr(spec, "base_config", {}).get("experiment_profile") != APC_FEEDBACK_ROBUSTNESS_PROFILE:
        return
    expected = {
        (period, loss, seed)
        for period, loss in APC_FEEDBACK_ROBUSTNESS_SCENARIOS
        for seed in APC_FEEDBACK_ROBUSTNESS_SEEDS
    }
    actual = {
        (run["factors"].get("node_period"), run["factors"].get("feedback_additional_loss_probability"), run["seed"])
        for run in runs
    }
    if len(runs) != APC_FEEDBACK_ROBUSTNESS_RUN_COUNT or actual != expected:
        raise ValueError("feedback robustness grid must contain exactly the approved 30 runs")
