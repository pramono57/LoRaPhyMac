"""Contracts for the IJIES R1.7 scalability/load follow-up profiles."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


R1_7_LOAD_PROFILE = "scalability_r1_7_load_v1"
R1_7_DUTY_PROFILE = "scalability_r1_7_relay_duty_v1"
APC_LOAD_SWEEP_PROFILE = "apc_load_sweep_v1"
R1_7_CONTRACT_VERSION = "scalability_r1_7_v1"
R1_7_SEEDS = (30, 36, 42, 48, 56)
R1_7_RUN_TABLES = {
    "node_tx_duty_cycle_df": "node_tx_duty_cycle.csv",
    "channel_occupancy_df": "channel_occupancy.csv",
}
R1_7_AGGREGATION_METRICS = (
    "pdr_percent",
    "avg_e2e_delay_ms",
    "packet_fate_delivered_count",
    "total_network_energy_J",
    "energy_per_delivered_packet_J",
    "collision_drop_decision_count",
    "relay_fifo_wait_mean_ms_packet_weighted",
)


def r1_7_config_violations(config: Mapping[str, Any]) -> list[str]:
    profile = config.get("experiment_profile")
    if str(profile) not in {
        R1_7_LOAD_PROFILE,
        R1_7_DUTY_PROFILE,
        APC_LOAD_SWEEP_PROFILE,
    }:
        return []
    errors = []
    if config.get("experiment_profile") != profile:
        errors.append("experiment_profile does not match R1.7 profile")
    if config.get("energy_accounting_mode") != "corrected":
        errors.append("energy_accounting_mode must be corrected")
    if config.get("sim_time") != 3600:
        errors.append("sim_time must be 3600 seconds")
    if config.get("node_sf_list") != [7] or config.get("relay_sf") != 7:
        errors.append("R1.7 requires fixed SF7")
    if config.get("node_bw") != 125000 or config.get("relay_bw_hz") != 125000:
        errors.append("R1.7 requires BW125 kHz")
    if config.get("node_payload_size") != 50:
        errors.append("R1.7 requires 50-byte application payload")
    if config.get("frame_default_ack_req") is not False:
        errors.append("R1.7 must not use application ACK/ARQ")
    if config.get("application_scheduling_mode") != "independent_fifo_v1":
        errors.append("R1.7 requires paired independent application scheduling")
    if config.get("application_rng_policy") != "sha256_seed_node_v1":
        errors.append("R1.7 requires stable per-node application RNG")
    if config.get("node_num") not in {50, 100, 200}:
        errors.append("R1.7 node_num must be one of 50, 100, 200")
    if config.get("node_period") not in {15, 30, 60}:
        errors.append("R1.7 node_period must be one of 15, 30, 60 seconds")
    return errors


def validate_r1_7_spec(spec: Any) -> None:
    profile = spec.base_config.get("experiment_profile")
    load_profile = profile in {R1_7_LOAD_PROFILE, APC_LOAD_SWEEP_PROFILE}
    expected_factors = (
        {"node_num", "node_period"}
        if load_profile
        else {"relay_duty_cycle"}
    )
    if set(spec.factors) != expected_factors:
        raise ValueError(f"{profile} factors must be exactly {sorted(expected_factors)}")
    if list(spec.seeds) != list(R1_7_SEEDS):
        raise ValueError(f"{profile} seeds must be {list(R1_7_SEEDS)}")
    if spec.run_annotations.get("model_id") not in {"ff", "aa"}:
        raise ValueError("R1.7 run_annotations.model_id must be ff or aa")
    model_id = spec.run_annotations["model_id"]
    expected_mode = "fixed" if model_id == "ff" else "adaptive"
    if (
        spec.base_config.get("node_ptx_mode") != expected_mode
        or spec.base_config.get("relay_ptx_mode") != expected_mode
    ):
        raise ValueError(f"R1.7 {model_id} profile has inconsistent PTx modes")
    if load_profile:
        if list(spec.factors["node_num"]) != [50, 100, 200]:
            raise ValueError("R1.7 load node_num levels must be [50, 100, 200]")
        if list(spec.factors["node_period"]) != [60, 30, 15]:
            raise ValueError("R1.7 load periods must be [60, 30, 15]")
        expected_count = 45
    else:
        if list(spec.factors["relay_duty_cycle"]) != [0.1, 0.01]:
            raise ValueError("R1.7 relay duty levels must be [0.1, 0.01]")
        if spec.base_config.get("node_num") != 100 or spec.base_config.get("node_period") != 30:
            raise ValueError("R1.7 relay duty sensitivity must use 100 nodes / 30 s")
        expected_count = 10
    if expected_count != len(spec.seeds) * math.prod(
        len(values) for values in spec.factors.values()
    ):
        raise ValueError(f"{profile} grid does not match its declared run count")


def validate_r1_7_outputs(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> None:
    del summary
    for key in R1_7_RUN_TABLES:
        frame = outputs.get(key)
        if frame is None or frame.empty:
            raise ValueError(f"R1.7 required output {key} is empty")


def r1_7_run_tables(config: Mapping[str, Any]) -> dict[str, str]:
    return dict(R1_7_RUN_TABLES)

