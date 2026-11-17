"""Four-run Summary/Detailed channel-occupancy pilot contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loraphymac.experiments.apc_terrain_profile_diagnostic.full_sweep_contract import (
    APC_TERRAIN_PROFILE_FULL_SWEEP_PROFILE,
    full_sweep_config_violations,
    full_sweep_run_tables,
    validate_full_sweep_outputs,
)


APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_PROFILE = (
    "apc_terrain_channel_occupancy_pilot_v1"
)
APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_VERSION = (
    "apc_terrain_channel_occupancy_pilot_contract_v1.0"
)
APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_RUN_COUNT = 4
_MODES = ("fixed", "adaptive")
_OUTPUT_PROFILES = ("summary", "detailed")
_EXPECTED_ROWS = {
    (mode, mode, 90, 42, output_profile)
    for mode in _MODES
    for output_profile in _OUTPUT_PROFILES
}
_REQUIRED_PROFILE = {
    "node_rx_collision_enabled": True,
    "application_scheduling_mode": "independent_fifo_v1",
    "application_rng_policy": "sha256_seed_node_v1",
    "application_queue_policy": "unbounded_fifo_v1",
    "application_horizon_policy": "right_censor_waiting_v1",
}


def channel_occupancy_pilot_config_violations(
    config: Mapping[str, Any],
) -> list[str]:
    if config.get("experiment_profile") != APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_PROFILE:
        return ["experiment_profile must identify the channel-occupancy pilot"]
    if config.get("output_profile") not in _OUTPUT_PROFILES:
        return ["output_profile must be summary or detailed"]
    full_sweep_config = {
        **config,
        "experiment_profile": APC_TERRAIN_PROFILE_FULL_SWEEP_PROFILE,
    }
    return full_sweep_config_violations(full_sweep_config)


def validate_channel_occupancy_pilot_spec(spec: Any) -> None:
    config = getattr(spec, "base_config", {})
    if config.get("experiment_profile") != APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_PROFILE:
        return
    errors: list[str] = []
    expected_factors = {
        "node_ptx_mode": _MODES,
        "relay_ptx_mode": _MODES,
        "terrain_delta_h": (90,),
        "output_profile": _OUTPUT_PROFILES,
    }
    factors = getattr(spec, "factors", {})
    if set(factors) != set(expected_factors):
        errors.append("factor keys must be node_ptx_mode, relay_ptx_mode, terrain_delta_h, output_profile")
    else:
        for key, expected in expected_factors.items():
            if tuple(factors[key]) != expected:
                errors.append(f"{key} factor must be {list(expected)}")
    if tuple(getattr(spec, "seeds", ())) != (42,):
        errors.append("seeds must contain only 42")
    matrix = getattr(spec, "run_matrix", None)
    if matrix is None:
        errors.append("run_matrix must explicitly declare the four pilot runs")
    else:
        actual = {
            (
                row.get("node_ptx_mode"),
                row.get("relay_ptx_mode"),
                row.get("terrain_delta_h"),
                row.get("seed"),
                row.get("output_profile"),
            )
            for row in matrix
        }
        if len(matrix) != APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_RUN_COUNT:
            errors.append("run_matrix must contain exactly four rows")
        if actual != _EXPECTED_ROWS:
            errors.append("run_matrix must contain FF/AA × Summary/Detailed at Δh=90, seed=42")
    if errors:
        raise ValueError("Channel occupancy pilot contract violation: " + "; ".join(errors))


def validate_channel_occupancy_pilot_outputs(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> None:
    errors = channel_occupancy_pilot_config_violations(config)
    if errors:
        raise ValueError("Channel occupancy pilot config violation: " + "; ".join(errors))
    full_sweep_config = {
        **config,
        "experiment_profile": APC_TERRAIN_PROFILE_FULL_SWEEP_PROFILE,
    }
    validate_full_sweep_outputs(full_sweep_config, summary, outputs)


def channel_occupancy_pilot_run_tables(config: Mapping[str, Any]) -> dict[str, str]:
    return full_sweep_run_tables(config)


__all__ = [
    "APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_PROFILE",
    "APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_RUN_COUNT",
    "APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_VERSION",
    "channel_occupancy_pilot_config_violations",
    "channel_occupancy_pilot_run_tables",
    "validate_channel_occupancy_pilot_outputs",
    "validate_channel_occupancy_pilot_spec",
]
