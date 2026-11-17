"""Contracts for versioned APC load-sweep designs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


APC_LOAD_SWEEP_FINAL_PROFILE = "apc_load_sweep_final_v1"
APC_LOAD_SWEEP_FINAL_CONTRACT = "apc_load_sweep_final_v1"
APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE = "apc_load_sweep_full_factorial_v1"
APC_LOAD_SWEEP_FULL_FACTORIAL_CONTRACT = "apc_load_sweep_full_factorial_v1"
APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_PILOT_PROFILE = (
    "apc_load_sweep_profile_acceptance_pilot_v1"
)
APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_PILOT_CONTRACT = (
    "apc_load_sweep_profile_acceptance_pilot_v1"
)
APC_LOAD_SWEEP_SEEDS = (30, 36, 42, 48, 56)
APC_LOAD_SWEEP_FULL_FACTORIAL_SCENARIOS = tuple(
    (nodes, period)
    for nodes in (50, 100, 150, 200)
    for period in (30, 60, 90)
)
APC_LOAD_SWEEP_FULL_FACTORIAL_BATCH = {
    "node_num": [50, 100, 150, 200],
    "node_period": [30, 60, 90],
}
APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_SCENARIOS = ((50, 90), (150, 60), (200, 30))
APC_LOAD_SWEEP_SCENARIOS = (
    (50, 90),
    (100, 90),
    (150, 30),
    (150, 60),
    (150, 90),
    (200, 90),
)
APC_LOAD_SWEEP_BATCHES = {
    "wide_nodes_90s": {"node_num": [50, 100, 200], "node_period": [90]},
    "node150_periods": {"node_num": [150], "node_period": [30, 60, 90]},
}
APC_LOAD_SWEEP_RUN_TABLES = {
    "energy_breakdown_df": "energy_breakdown.csv",
    "node_tx_duty_cycle_df": "node_tx_duty_cycle.csv",
    "channel_occupancy_df": "channel_occupancy.csv",
}


def apc_load_sweep_config_violations(config: Mapping[str, Any]) -> list[str]:
    if config.get("experiment_profile") != APC_LOAD_SWEEP_FINAL_PROFILE:
        return []
    errors = []
    if config.get("energy_accounting_mode") != "corrected":
        errors.append("energy_accounting_mode must be corrected")
    if config.get("sim_time") != 3600:
        errors.append("sim_time must be 3600 seconds")
    if config.get("node_sf_list") != [7] or config.get("relay_sf") != 7:
        errors.append("APC load sweep requires fixed SF7")
    if config.get("node_bw") != 125000 or config.get("relay_bw_hz") != 125000:
        errors.append("APC load sweep requires BW125 kHz")
    if config.get("node_payload_size") != 50:
        errors.append("APC load sweep requires 50-byte application payload")
    if config.get("frame_default_ack_req") is not False:
        errors.append("APC load sweep must not use application ACK/ARQ")
    if config.get("application_scheduling_mode") != "independent_fifo_v1":
        errors.append("APC load sweep requires independent application scheduling")
    if config.get("application_rng_policy") != "sha256_seed_node_v1":
        errors.append("APC load sweep requires stable per-node application RNG")
    node_num = config.get("node_num")
    node_period = config.get("node_period")
    if (node_num, node_period) not in set(APC_LOAD_SWEEP_SCENARIOS):
        errors.append("node_num/node_period is not in the final six-cell matrix")
    return errors


def apc_load_sweep_full_factorial_config_violations(
    config: Mapping[str, Any],
) -> list[str]:
    if config.get("experiment_profile") != APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE:
        return []
    errors = []
    common = dict(config)
    # Reuse the validated physical and traffic contract of the final APC sweep.
    common["experiment_profile"] = APC_LOAD_SWEEP_FINAL_PROFILE
    common["node_num"] = 50
    common["node_period"] = 90
    errors.extend(
        error
        for error in apc_load_sweep_config_violations(common)
        if "final six-cell matrix" not in error
    )
    if config.get("output_profile") != "summary":
        errors.append("output_profile must be summary")
    if config.get("node_rx_collision_enabled") is not True:
        errors.append("node_rx_collision_enabled must be true")
    if config.get("relay_num") != 2:
        errors.append("APC load sweep requires exactly two relays")
    if config.get("node_num") not in {50, 100, 150, 200}:
        errors.append("node_num must be one of 50, 100, 150, 200")
    if config.get("node_period") not in {30, 60, 90}:
        errors.append("node_period must be one of 30, 60, 90 seconds")
    return errors


def apc_load_sweep_profile_acceptance_config_violations(
    config: Mapping[str, Any],
) -> list[str]:
    """Validate the paired output-profile pilot without relaxing production."""
    if config.get("experiment_profile") != APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_PILOT_PROFILE:
        return []
    if config.get("output_profile") not in {"summary", "detailed"}:
        return ["output_profile must be summary or detailed for the acceptance pilot"]

    production_contract_config = dict(config)
    production_contract_config["experiment_profile"] = APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE
    production_contract_config["output_profile"] = "summary"
    errors = apc_load_sweep_full_factorial_config_violations(
        production_contract_config
    )
    scenario = (config.get("node_num"), config.get("node_period"))
    if scenario not in set(APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_SCENARIOS):
        errors.append("node_num/node_period is not an approved profile-acceptance pilot cell")
    if config.get("seed") != 30:
        errors.append("profile-acceptance pilot seed must be 30")
    return errors


def validate_apc_load_sweep_spec(spec: Any) -> None:
    if spec.base_config.get("experiment_profile") != APC_LOAD_SWEEP_FINAL_PROFILE:
        return
    if list(spec.seeds) != list(APC_LOAD_SWEEP_SEEDS):
        raise ValueError(f"Final APC load sweep seeds must be {list(APC_LOAD_SWEEP_SEEDS)}")
    model_id = spec.run_annotations.get("model_id")
    if model_id not in {"ff", "aa"}:
        raise ValueError("Final APC load sweep model_id must be ff or aa")
    expected_mode = "fixed" if model_id == "ff" else "adaptive"
    if (
        spec.base_config.get("node_ptx_mode") != expected_mode
        or spec.base_config.get("relay_ptx_mode") != expected_mode
    ):
        raise ValueError(f"Final APC load sweep {model_id} PTx mode is inconsistent")
    factors = {name: list(values) for name, values in spec.factors.items()}
    if factors not in list(APC_LOAD_SWEEP_BATCHES.values()):
        raise ValueError(
            "Final APC load sweep factors must be one of the two declared batches"
        )
    if len(spec.seeds) * len(factors["node_num"]) * len(factors["node_period"]) != 15:
        raise ValueError("Each final APC load-sweep batch must plan exactly 15 runs")


def validate_apc_load_sweep_outputs(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> None:
    del summary
    if config.get("experiment_profile") not in {
        APC_LOAD_SWEEP_FINAL_PROFILE,
        APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE,
    }:
        return
    required = ["node_tx_duty_cycle_df", "channel_occupancy_df"]
    if config.get("experiment_profile") == APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE:
        required.extend(
            [
                "energy_breakdown_df",
                "packet_fate_df",
                "packet_fate_summary_df",
                "relay_duty_cycle_df",
                "collision_summary_df",
                "apc_feedback_lifecycle_df",
                "core_summary_df",
            ]
        )
    for key in required:
        frame = outputs.get(key)
        if frame is None:
            raise ValueError(f"APC load sweep required output {key} is missing")
        if key in {
            "node_tx_duty_cycle_df",
            "channel_occupancy_df",
            "energy_breakdown_df",
            "packet_fate_df",
            "packet_fate_summary_df",
            "relay_duty_cycle_df",
        } and frame.empty:
            raise ValueError(f"APC load sweep required output {key} is empty")


def validate_apc_load_sweep_full_factorial_spec(spec: Any) -> None:
    if spec.base_config.get("experiment_profile") != APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE:
        return
    if list(spec.seeds) != list(APC_LOAD_SWEEP_SEEDS):
        raise ValueError(
            f"Full-factorial APC load sweep seeds must be {list(APC_LOAD_SWEEP_SEEDS)}"
        )
    model_id = spec.run_annotations.get("model_id")
    if model_id not in {"ff", "aa"}:
        raise ValueError("Full-factorial APC load sweep model_id must be ff or aa")
    expected_mode = "fixed" if model_id == "ff" else "adaptive"
    if (
        spec.base_config.get("node_ptx_mode") != expected_mode
        or spec.base_config.get("relay_ptx_mode") != expected_mode
    ):
        raise ValueError(f"Full-factorial APC load sweep {model_id} PTx mode is inconsistent")
    factors = {name: list(values) for name, values in spec.factors.items()}
    expected = APC_LOAD_SWEEP_FULL_FACTORIAL_BATCH
    if factors != expected:
        raise ValueError(f"Full-factorial factors must be exactly {expected}")
    if spec.master_table_mode != "aggregate_inputs":
        raise ValueError("Full-factorial APC load sweep requires aggregate_inputs")
    if len(spec.seeds) * len(expected["node_num"]) * len(expected["node_period"]) != 60:
        raise ValueError("Each full-factorial model must plan exactly 60 runs")
