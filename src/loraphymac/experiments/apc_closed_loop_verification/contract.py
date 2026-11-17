"""Scientific contract for one closed-loop APC behavior verification run."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from loraphymac.experiments.apc_terrain_sweep.contract import PAPER1_APC_LOCKED_CONFIG


APC_CLOSED_LOOP_PROFILE = "apc_closed_loop_verification_v1"
APC_CLOSED_LOOP_CONTRACT_VERSION = "apc_closed_loop_contract_v1.0"
APC_CLOSED_LOOP_EXPECTED_RUN_COUNT = 1

_locked = dict(PAPER1_APC_LOCKED_CONFIG)
_locked.update(
    {
        "node_sf_list": (7,),
        "relay_tx_ch_ids": (1,),
        "terrain_delta_h": 90,
        "seed": 42,
        "node_ptx_mode": "adaptive",
        "relay_ptx_mode": "adaptive",
    }
)
APC_CLOSED_LOOP_LOCKED_CONFIG = MappingProxyType(_locked)

APC_CLOSED_LOOP_AGGREGATION_METRICS = (
    "delivery_by_horizon_percent",
    "avg_e2e_delay_ms",
    "total_network_energy_J",
    "avg_final_node_ptx",
    "avg_final_relay_ptx",
    "apc_total_tx",
    "apc_total_rx",
    "apc_total_drop",
    "ptx_update_count",
)
APC_CLOSED_LOOP_RUN_TABLES = MappingProxyType(
    {
        "energy_breakdown_df": "energy_breakdown.csv",
        "final_ptx_distribution_df": "final_ptx_distribution.csv",
        "ptx_history_df": "ptx_history.csv",
        "margin_history_df": "margin_history.csv",
        "apc_control_history_df": "apc_control_history.csv",
        "frame_summary_df": "frame_summary.csv",
        "packet_fate_summary_df": "packet_fate_summary.csv",
        "run_metadata_df": "run_metadata.csv",
    }
)
APC_CLOSED_LOOP_REQUIRED_SUMMARY_FIELDS = frozenset(
    {
        "delivery_by_horizon_percent",
        "terminal_loss_percent",
        "censored_fraction_percent",
        "avg_e2e_delay_ms",
        "total_network_energy_J",
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


class APCClosedLoopContractError(ValueError):
    """Raised when the verification configuration or evidence is invalid."""


def _equivalent(actual: Any, expected: Any) -> bool:
    if isinstance(expected, tuple):
        if isinstance(actual, (str, bytes)) or not isinstance(actual, Sequence):
            return False
        return tuple(actual) == expected
    return actual == expected


def apc_closed_loop_config_violations(config: Mapping[str, Any]) -> list[str]:
    """Return drift from the frozen closed-loop verification scenario."""
    violations: list[str] = []
    for key, expected in APC_CLOSED_LOOP_LOCKED_CONFIG.items():
        if key not in config:
            violations.append(f"{key} is missing (expected {expected!r})")
        elif not _equivalent(config[key], expected):
            violations.append(f"{key} must remain {expected!r} (got {config[key]!r})")
    return violations


def validate_apc_closed_loop_experiment_spec(spec: Any) -> None:
    """Allow the profile through the grid API only as one fixed cell."""
    base_config = getattr(spec, "base_config", {})
    if base_config.get("experiment_profile") != APC_CLOSED_LOOP_PROFILE:
        return
    errors: list[str] = []
    if getattr(spec, "factors", {}):
        errors.append("factors must be empty")
    if tuple(getattr(spec, "seeds", ())) != (42,):
        errors.append("seeds must be exactly [42]")
    if errors:
        raise APCClosedLoopContractError(
            "APC closed-loop grid contract violation: " + "; ".join(errors)
        )


def apc_closed_loop_run_tables(config: Mapping[str, Any]) -> dict[str, str]:
    if config.get("experiment_profile") != APC_CLOSED_LOOP_PROFILE:
        return {}
    return dict(APC_CLOSED_LOOP_RUN_TABLES)


def _finite_nonnegative(summary: Mapping[str, Any], key: str) -> float:
    try:
        value = float(summary[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise APCClosedLoopContractError(f"Output field {key!r} must be numeric") from exc
    if not math.isfinite(value) or value < 0.0:
        raise APCClosedLoopContractError(
            f"Output field {key!r} must be finite and non-negative"
        )
    return value


def validate_apc_closed_loop_outputs(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> None:
    """Require observable node and relay APC feedback in one valid run."""
    if config.get("experiment_profile") != APC_CLOSED_LOOP_PROFILE:
        return
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise APCClosedLoopContractError("pandas is required for output validation") from exc

    missing_fields = sorted(APC_CLOSED_LOOP_REQUIRED_SUMMARY_FIELDS.difference(summary))
    if missing_fields:
        raise APCClosedLoopContractError(
            f"Summary is missing required fields: {missing_fields}"
        )
    missing_tables = sorted(set(APC_CLOSED_LOOP_RUN_TABLES).difference(outputs))
    if missing_tables:
        raise APCClosedLoopContractError(
            f"Outputs are missing required tables: {missing_tables}"
        )
    invalid_tables = sorted(
        name
        for name in APC_CLOSED_LOOP_RUN_TABLES
        if not isinstance(outputs[name], pd.DataFrame)
    )
    if invalid_tables:
        raise APCClosedLoopContractError(
            f"Required outputs must be DataFrames: {invalid_tables}"
        )

    percentages = [
        _finite_nonnegative(summary, key)
        for key in (
            "delivery_by_horizon_percent",
            "terminal_loss_percent",
            "censored_fraction_percent",
        )
    ]
    if any(value > 100.0 for value in percentages) or not math.isclose(
        sum(percentages),
        100.0,
        abs_tol=0.01,
    ):
        raise APCClosedLoopContractError(
            "Delivery, terminal-loss, and censored percentages must be within "
            "0..100 and sum to 100%"
        )
    for key in (
        "avg_e2e_delay_ms",
        "total_network_energy_J",
        "avg_final_node_ptx",
        "avg_final_relay_ptx",
    ):
        _finite_nonnegative(summary, key)
    if summary["stage1_status"] != "PASS":
        raise APCClosedLoopContractError("stage1_status must be PASS")
    for key in (
        "packet_fate_conservation_delta",
        "collision_outcome_conservation_delta",
    ):
        if int(summary[key]) != 0:
            raise APCClosedLoopContractError(f"Invariant {key} must be zero")

    apc_tx = int(summary["apc_total_tx"])
    apc_rx = int(summary["apc_total_rx"])
    apc_drop = int(summary["apc_total_drop"])
    updates = int(summary["ptx_update_count"])
    if min(apc_tx, apc_rx, apc_drop, updates) < 0:
        raise APCClosedLoopContractError("APC counters must be non-negative")
    if apc_tx != apc_rx + apc_drop:
        raise APCClosedLoopContractError(
            "apc_total_tx must equal apc_total_rx + apc_total_drop"
        )
    if apc_rx != updates:
        raise APCClosedLoopContractError(
            "apc_total_rx must equal ptx_update_count"
        )
    if updates == 0:
        raise APCClosedLoopContractError("No closed-loop PTx update was observed")

    history = outputs["ptx_history_df"]
    required_history_columns = {
        "entity_id",
        "entity_type",
        "time",
        "old_ptx",
        "new_ptx",
    }
    missing_history_columns = sorted(required_history_columns.difference(history.columns))
    if missing_history_columns:
        raise APCClosedLoopContractError(
            f"PTx history is missing required columns: {missing_history_columns}"
        )
    if len(history) != updates:
        raise APCClosedLoopContractError(
            "PTx-history row count must equal ptx_update_count"
        )
    observed_types = set(history["entity_type"].dropna().astype(str).str.lower())
    for entity_type in ("node", "relay"):
        if entity_type not in observed_types:
            raise APCClosedLoopContractError(
                f"No {entity_type} closed-loop PTx update was observed"
            )

    margin = outputs["margin_history_df"]
    required_margin_columns = {"entity_id", "entity_type", "time", "margin"}
    missing_margin_columns = sorted(required_margin_columns.difference(margin.columns))
    if missing_margin_columns or margin.empty:
        raise APCClosedLoopContractError(
            "Margin history must be non-empty with entity_id, entity_type, "
            "time, and margin"
        )
    margin_types = set(margin["entity_type"].dropna().astype(str).str.lower())
    if not {"node", "relay"}.issubset(margin_types):
        raise APCClosedLoopContractError(
            "Margin history must contain both node and relay observations"
        )

    metadata = outputs["run_metadata_df"]
    recorded_version = (
        None
        if metadata.empty
        else metadata.iloc[0].get("experiment_contract_version")
    )
    if recorded_version != APC_CLOSED_LOOP_CONTRACT_VERSION:
        raise APCClosedLoopContractError(
            "Run metadata has a missing or incorrect closed-loop contract version"
        )


__all__ = [
    "APC_CLOSED_LOOP_AGGREGATION_METRICS",
    "APC_CLOSED_LOOP_CONTRACT_VERSION",
    "APC_CLOSED_LOOP_EXPECTED_RUN_COUNT",
    "APC_CLOSED_LOOP_LOCKED_CONFIG",
    "APC_CLOSED_LOOP_PROFILE",
    "APC_CLOSED_LOOP_RUN_TABLES",
    "APCClosedLoopContractError",
    "apc_closed_loop_config_violations",
    "apc_closed_loop_run_tables",
    "validate_apc_closed_loop_experiment_spec",
    "validate_apc_closed_loop_outputs",
]
