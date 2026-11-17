"""Registry for versioned scientific experiment contracts.

Adding a future experiment requires registering its contract here; the shared
validator and multi-seed pipeline do not need experiment-specific branches.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loraphymac.experiments.apc_closed_loop_verification.contract import (
    APC_CLOSED_LOOP_AGGREGATION_METRICS,
    APC_CLOSED_LOOP_CONTRACT_VERSION,
    APC_CLOSED_LOOP_EXPECTED_RUN_COUNT,
    APC_CLOSED_LOOP_PROFILE,
    APC_CLOSED_LOOP_RUN_TABLES,
    apc_closed_loop_config_violations,
    apc_closed_loop_run_tables,
    validate_apc_closed_loop_experiment_spec,
    validate_apc_closed_loop_outputs,
)
from loraphymac.experiments.apc_terrain_sweep.contract import (
    APC_TERRAIN_SWEEP_AGGREGATION_METRICS,
    APC_TERRAIN_SWEEP_ALWAYS_RUN_TABLES,
    APC_TERRAIN_SWEEP_COMPARISON_RUN_TABLES,
    APC_TERRAIN_SWEEP_CONTRACT_VERSION,
    APC_TERRAIN_SWEEP_EXPECTED_RUN_COUNT,
    APC_TERRAIN_SWEEP_PROFILE,
    PAPER1_APC_ALWAYS_RUN_TABLES,
    PAPER1_APC_COMPARISON_RUN_TABLES,
    PAPER1_APC_CONTRACT_VERSION,
    PAPER1_APC_PROFILE,
    APCTerrainSweepContractError,
    apc_terrain_sweep_config_violations,
    apc_terrain_sweep_run_tables,
    validate_apc_terrain_sweep_experiment_spec,
    validate_apc_terrain_sweep_outputs,
)
from loraphymac.experiments.apc_terrain_profile_diagnostic.contract import (
    APC_TERRAIN_PROFILE_DIAGNOSTIC_AGGREGATION_METRICS,
    APC_TERRAIN_PROFILE_DIAGNOSTIC_PROFILE,
    APC_TERRAIN_PROFILE_DIAGNOSTIC_RUN_COUNT,
    APC_TERRAIN_PROFILE_DIAGNOSTIC_VERSION,
    diagnostic_config_violations,
    diagnostic_run_tables,
    validate_diagnostic_outputs,
    validate_diagnostic_spec,
)
from loraphymac.experiments.apc_terrain_profile_diagnostic.full_sweep_contract import (
    APC_TERRAIN_PROFILE_FULL_SWEEP_PROFILE,
    APC_TERRAIN_PROFILE_FULL_SWEEP_RUN_COUNT,
    APC_TERRAIN_PROFILE_FULL_SWEEP_VERSION,
    full_sweep_config_violations,
    full_sweep_run_tables,
    validate_full_sweep_outputs,
    validate_full_sweep_spec,
)
from loraphymac.experiments.apc_terrain_profile_diagnostic.channel_occupancy_pilot_contract import (
    APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_PROFILE,
    APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_RUN_COUNT,
    APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_VERSION,
    channel_occupancy_pilot_config_violations,
    channel_occupancy_pilot_run_tables,
    validate_channel_occupancy_pilot_outputs,
    validate_channel_occupancy_pilot_spec,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.contract import (
    APC_COMPARE_MODEL_TERRAIN_SWEEP_AGGREGATION_METRICS,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_CHILD_RUN_COUNT,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_CONTRACT_VERSION,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_RUN_TABLES,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_CONTRACT_VERSION,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_APPLICATION_RUN_TABLES,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_CHILD_RUN_COUNT,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_CONTRACT_VERSION,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_WANG_RUN_TABLES,
    MAIN_W4_CHILD_RUN_COUNT,
    MAIN_W4_CONTRACT_VERSION,
    MAIN_W4_PROFILE,
    PAIRED_ROUTING_PILOT_CHILD_RUN_COUNT,
    PAIRED_ROUTING_PILOT_CONTRACT_VERSION,
    PAIRED_ROUTING_PILOT_PROFILE,
    REVIEWER_R1_5_CHILD_RUN_COUNT,
    REVIEWER_R1_5_CONTRACT_VERSION,
    REVIEWER_R1_5_PROFILE,
    apc_compare_model_terrain_sweep_config_violations,
    apc_compare_model_terrain_sweep_run_tables,
    validate_apc_compare_model_terrain_sweep_outputs,
    validate_apc_compare_model_terrain_sweep_spec,
)
from loraphymac.experiments.scalability_relay_bottleneck.r1_7_contract import (
    APC_LOAD_SWEEP_PROFILE,
    R1_7_AGGREGATION_METRICS,
    R1_7_CONTRACT_VERSION,
    R1_7_DUTY_PROFILE,
    R1_7_LOAD_PROFILE,
    R1_7_RUN_TABLES,
    r1_7_config_violations,
    r1_7_run_tables,
    validate_r1_7_outputs,
    validate_r1_7_spec,
)
from loraphymac.experiments.apc_load_sweep.contract import (
    APC_LOAD_SWEEP_FINAL_CONTRACT,
    APC_LOAD_SWEEP_FINAL_PROFILE,
    APC_LOAD_SWEEP_FULL_FACTORIAL_CONTRACT,
    APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE,
    APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_PILOT_CONTRACT,
    APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_PILOT_PROFILE,
    APC_LOAD_SWEEP_RUN_TABLES,
    apc_load_sweep_config_violations,
    apc_load_sweep_full_factorial_config_violations,
    apc_load_sweep_profile_acceptance_config_violations,
    validate_apc_load_sweep_outputs,
    validate_apc_load_sweep_spec,
    validate_apc_load_sweep_full_factorial_spec,
)
from loraphymac.experiments.apc_feedback_robustness_sweep.contract import (
    APC_FEEDBACK_ROBUSTNESS_PROFILE,
    APC_FEEDBACK_ROBUSTNESS_RUN_COUNT,
    APC_FEEDBACK_ROBUSTNESS_VERSION,
    APC_FEEDBACK_ROBUSTNESS_RUN_TABLES,
    feedback_robustness_config_violations,
    feedback_robustness_run_tables,
    validate_feedback_robustness_grid,
    validate_feedback_robustness_outputs,
    validate_feedback_robustness_spec,
)


def _apc_terrain_sweep_contract(version: str) -> dict[str, Any]:
    return {
        "version": version,
        "expected_run_count": APC_TERRAIN_SWEEP_EXPECTED_RUN_COUNT,
        "config_violations": apc_terrain_sweep_config_violations,
        "validate_spec": validate_apc_terrain_sweep_experiment_spec,
        "validate_outputs": validate_apc_terrain_sweep_outputs,
        "run_tables": apc_terrain_sweep_run_tables,
        "aggregation_metrics": APC_TERRAIN_SWEEP_AGGREGATION_METRICS,
        "all_table_filenames": frozenset(
            {
                *APC_TERRAIN_SWEEP_ALWAYS_RUN_TABLES.values(),
                *APC_TERRAIN_SWEEP_COMPARISON_RUN_TABLES.values(),
            }
        ),
    }


def _apc_closed_loop_contract() -> dict[str, Any]:
    return {
        "version": APC_CLOSED_LOOP_CONTRACT_VERSION,
        "expected_run_count": APC_CLOSED_LOOP_EXPECTED_RUN_COUNT,
        "config_violations": apc_closed_loop_config_violations,
        "validate_spec": validate_apc_closed_loop_experiment_spec,
        "validate_outputs": validate_apc_closed_loop_outputs,
        "run_tables": apc_closed_loop_run_tables,
        "aggregation_metrics": APC_CLOSED_LOOP_AGGREGATION_METRICS,
        "all_table_filenames": frozenset(APC_CLOSED_LOOP_RUN_TABLES.values()),
    }


def _apc_compare_model_terrain_sweep_contract(
    *,
    version: str,
    include_wang_tables: bool = False,
    include_v2_output_tables: bool = False,
    expected_run_count: int = APC_COMPARE_MODEL_TERRAIN_SWEEP_CHILD_RUN_COUNT,
) -> dict[str, Any]:
    filenames = set(APC_COMPARE_MODEL_TERRAIN_SWEEP_RUN_TABLES.values())
    if include_wang_tables:
        filenames.update(APC_COMPARE_MODEL_TERRAIN_SWEEP_WANG_RUN_TABLES.values())
        filenames.update(
            APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_APPLICATION_RUN_TABLES.values()
        )
    if include_v2_output_tables:
        filenames.update(
            {
                "frame_summary.csv",
                "frame_mtype_summary.csv",
                "hop_summary.csv",
                "frame_hop_event_summary.csv",
                "frame_drop_reason_by_hop.csv",
                "tx_queue_waiting_summary.csv",
                "selected_next_hop_links.csv",
                "relay_chain_links.csv",
                "relay_tx_mgc_queue_metrics.csv",
                "node_tx_duty_cycle.csv",
                "channel_occupancy.csv",
            }
        )
    return {
        "version": version,
        "expected_run_count": expected_run_count,
        "config_violations": apc_compare_model_terrain_sweep_config_violations,
        "validate_spec": validate_apc_compare_model_terrain_sweep_spec,
        "validate_outputs": validate_apc_compare_model_terrain_sweep_outputs,
        "run_tables": apc_compare_model_terrain_sweep_run_tables,
        "aggregation_metrics": APC_COMPARE_MODEL_TERRAIN_SWEEP_AGGREGATION_METRICS,
        "all_table_filenames": frozenset(filenames),
    }


_CONTRACTS = {
    APC_FEEDBACK_ROBUSTNESS_PROFILE: {
        "version": APC_FEEDBACK_ROBUSTNESS_VERSION,
        "expected_run_count": APC_FEEDBACK_ROBUSTNESS_RUN_COUNT,
        "config_violations": feedback_robustness_config_violations,
        "validate_spec": validate_feedback_robustness_spec,
        "validate_grid": validate_feedback_robustness_grid,
        "validate_outputs": validate_feedback_robustness_outputs,
        "run_tables": feedback_robustness_run_tables,
        "aggregation_metrics": R1_7_AGGREGATION_METRICS,
        "all_table_filenames": frozenset(APC_FEEDBACK_ROBUSTNESS_RUN_TABLES.values()),
    },
    APC_LOAD_SWEEP_PROFILE: {
        "version": R1_7_CONTRACT_VERSION,
        "expected_run_count": 45,
        "config_violations": r1_7_config_violations,
        "validate_spec": validate_r1_7_spec,
        "validate_outputs": validate_r1_7_outputs,
        "run_tables": r1_7_run_tables,
        "aggregation_metrics": R1_7_AGGREGATION_METRICS,
        "all_table_filenames": frozenset(R1_7_RUN_TABLES.values()),
    },
    APC_LOAD_SWEEP_FINAL_PROFILE: {
        "version": APC_LOAD_SWEEP_FINAL_CONTRACT,
        "expected_run_count": 15,
        "config_violations": apc_load_sweep_config_violations,
        "validate_spec": validate_apc_load_sweep_spec,
        "validate_outputs": validate_apc_load_sweep_outputs,
        "run_tables": lambda config: dict(APC_LOAD_SWEEP_RUN_TABLES),
        "aggregation_metrics": R1_7_AGGREGATION_METRICS,
        "all_table_filenames": frozenset(APC_LOAD_SWEEP_RUN_TABLES.values()),
    },
    APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE: {
        "version": APC_LOAD_SWEEP_FULL_FACTORIAL_CONTRACT,
        "expected_run_count": 60,
        "config_violations": apc_load_sweep_full_factorial_config_violations,
        "validate_spec": validate_apc_load_sweep_full_factorial_spec,
        "validate_outputs": validate_apc_load_sweep_outputs,
        "run_tables": lambda config: dict(APC_LOAD_SWEEP_RUN_TABLES),
        "aggregation_metrics": R1_7_AGGREGATION_METRICS,
        "all_table_filenames": frozenset(APC_LOAD_SWEEP_RUN_TABLES.values()),
    },
    APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_PILOT_PROFILE: {
        "version": APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_PILOT_CONTRACT,
        "expected_run_count": 12,
        "config_violations": apc_load_sweep_profile_acceptance_config_violations,
        "validate_spec": lambda spec: None,
        "validate_outputs": validate_apc_load_sweep_outputs,
        "run_tables": lambda config: {},
        "aggregation_metrics": (),
        "all_table_filenames": frozenset(),
    },
    R1_7_LOAD_PROFILE: {
        "version": R1_7_CONTRACT_VERSION,
        "expected_run_count": 45,
        "config_violations": r1_7_config_violations,
        "validate_spec": validate_r1_7_spec,
        "validate_outputs": validate_r1_7_outputs,
        "run_tables": r1_7_run_tables,
        "aggregation_metrics": R1_7_AGGREGATION_METRICS,
        "all_table_filenames": frozenset(R1_7_RUN_TABLES.values()),
    },
    R1_7_DUTY_PROFILE: {
        "version": R1_7_CONTRACT_VERSION,
        "expected_run_count": 10,
        "config_violations": r1_7_config_violations,
        "validate_spec": validate_r1_7_spec,
        "validate_outputs": validate_r1_7_outputs,
        "run_tables": r1_7_run_tables,
        "aggregation_metrics": R1_7_AGGREGATION_METRICS,
        "all_table_filenames": frozenset(R1_7_RUN_TABLES.values()),
    },
    APC_CLOSED_LOOP_PROFILE: _apc_closed_loop_contract(),
    APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE: (
        _apc_compare_model_terrain_sweep_contract(
            version=APC_COMPARE_MODEL_TERRAIN_SWEEP_CONTRACT_VERSION
        )
    ),
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE: (
        _apc_compare_model_terrain_sweep_contract(
            version=APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_CONTRACT_VERSION,
            include_wang_tables=True,
            include_v2_output_tables=True,
        )
    ),
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE: (
        _apc_compare_model_terrain_sweep_contract(
            version=APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_CONTRACT_VERSION,
            include_wang_tables=True,
            include_v2_output_tables=True,
            expected_run_count=(
                APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_CHILD_RUN_COUNT
            ),
        )
    ),
    MAIN_W4_PROFILE: _apc_compare_model_terrain_sweep_contract(
        version=MAIN_W4_CONTRACT_VERSION,
        include_wang_tables=True,
        expected_run_count=MAIN_W4_CHILD_RUN_COUNT,
    ),
    PAIRED_ROUTING_PILOT_PROFILE: _apc_compare_model_terrain_sweep_contract(
        version=PAIRED_ROUTING_PILOT_CONTRACT_VERSION,
        include_wang_tables=True,
        expected_run_count=PAIRED_ROUTING_PILOT_CHILD_RUN_COUNT,
    ),
    REVIEWER_R1_5_PROFILE: _apc_compare_model_terrain_sweep_contract(
        version=REVIEWER_R1_5_CONTRACT_VERSION,
        include_wang_tables=True,
        expected_run_count=REVIEWER_R1_5_CHILD_RUN_COUNT,
    ),
    APC_TERRAIN_SWEEP_PROFILE: _apc_terrain_sweep_contract(
        APC_TERRAIN_SWEEP_CONTRACT_VERSION
    ),
    APC_TERRAIN_PROFILE_DIAGNOSTIC_PROFILE: {
        "version": APC_TERRAIN_PROFILE_DIAGNOSTIC_VERSION,
        "expected_run_count": APC_TERRAIN_PROFILE_DIAGNOSTIC_RUN_COUNT,
        "config_violations": diagnostic_config_violations,
        "validate_spec": validate_diagnostic_spec,
        "validate_outputs": validate_diagnostic_outputs,
        "run_tables": diagnostic_run_tables,
        "aggregation_metrics": APC_TERRAIN_PROFILE_DIAGNOSTIC_AGGREGATION_METRICS,
        "all_table_filenames": frozenset(
            {
                *PAPER1_APC_ALWAYS_RUN_TABLES.values(),
                *PAPER1_APC_COMPARISON_RUN_TABLES.values(),
                "application_queue_metrics.csv",
            }
        ),
    },
    APC_TERRAIN_PROFILE_FULL_SWEEP_PROFILE: {
        "version": APC_TERRAIN_PROFILE_FULL_SWEEP_VERSION,
        "expected_run_count": APC_TERRAIN_PROFILE_FULL_SWEEP_RUN_COUNT,
        "config_violations": full_sweep_config_violations,
        "validate_spec": validate_full_sweep_spec,
        "validate_outputs": validate_full_sweep_outputs,
        "run_tables": full_sweep_run_tables,
        "aggregation_metrics": APC_TERRAIN_PROFILE_DIAGNOSTIC_AGGREGATION_METRICS,
        "all_table_filenames": frozenset(
            {
                *PAPER1_APC_ALWAYS_RUN_TABLES.values(),
                *PAPER1_APC_COMPARISON_RUN_TABLES.values(),
                "application_queue_metrics.csv",
                "node_tx_duty_cycle.csv",
                "channel_occupancy.csv",
            }
        ),
    },
    APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_PROFILE: {
        "version": APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_VERSION,
        "expected_run_count": APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_RUN_COUNT,
        "config_violations": channel_occupancy_pilot_config_violations,
        "validate_spec": validate_channel_occupancy_pilot_spec,
        "validate_outputs": validate_channel_occupancy_pilot_outputs,
        "run_tables": channel_occupancy_pilot_run_tables,
        "aggregation_metrics": APC_TERRAIN_PROFILE_DIAGNOSTIC_AGGREGATION_METRICS,
        "all_table_filenames": frozenset(
            channel_occupancy_pilot_run_tables({})
            .values()
        ),
    },
    # Frozen historical profile retained so the completed 300-run dataset
    # remains auditable without rewriting its manifest or run metadata.
    PAPER1_APC_PROFILE: _apc_terrain_sweep_contract(PAPER1_APC_CONTRACT_VERSION),
}

# Legacy public name retained for downstream callers.
Paper1APCContractError = APCTerrainSweepContractError


def registered_contract_profiles() -> frozenset[str]:
    return frozenset(_CONTRACTS)


def experiment_contract_version(profile: Any) -> str | None:
    contract = _CONTRACTS.get(str(profile))
    return None if contract is None else str(contract["version"])


def experiment_config_violations(profile: Any, config: Mapping[str, Any]) -> list[str]:
    contract = _CONTRACTS.get(str(profile))
    if contract is None:
        return []
    return list(contract["config_violations"](config))


def validate_experiment_spec_contract(spec: Any) -> None:
    profile = getattr(spec, "base_config", {}).get("experiment_profile")
    contract = _CONTRACTS.get(str(profile))
    if contract is not None:
        contract["validate_spec"](spec)


def validate_experiment_grid_contract(spec: Any, runs: Sequence[Mapping]) -> None:
    profile = getattr(spec, "base_config", {}).get("experiment_profile")
    contract = _CONTRACTS.get(str(profile))
    if contract is None:
        return
    expected = int(contract["expected_run_count"])
    if len(runs) != expected:
        raise ValueError(
            f"{profile} grid contract violation: expected {expected} runs, got {len(runs)}"
        )
    validator = contract.get("validate_grid")
    if validator is not None:
        validator(spec, runs)


def validate_run_output_contract(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> None:
    contract = _CONTRACTS.get(str(config.get("experiment_profile")))
    if contract is not None:
        contract["validate_outputs"](config, summary, outputs)


def experiment_run_tables(config: Mapping[str, Any]) -> dict[str, str]:
    contract = _CONTRACTS.get(str(config.get("experiment_profile")))
    if contract is None:
        return {}
    return dict(contract["run_tables"](config))


def experiment_aggregation_metrics(profile: Any, default: Sequence[str]) -> Sequence[str]:
    contract = _CONTRACTS.get(str(profile))
    if contract is None:
        return default
    return contract["aggregation_metrics"]


def experiment_table_filenames(profile: Any) -> frozenset[str]:
    contract = _CONTRACTS.get(str(profile))
    if contract is None:
        return frozenset()
    return contract["all_table_filenames"]


__all__ = [
    "APC_FEEDBACK_ROBUSTNESS_PROFILE",
    "APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_PROFILE",
    "APC_LOAD_SWEEP_PROFILE",
    "R1_7_LOAD_PROFILE",
    "R1_7_DUTY_PROFILE",
    "APC_CLOSED_LOOP_CONTRACT_VERSION",
    "APC_CLOSED_LOOP_PROFILE",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_CONTRACT_VERSION",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE",
    "MAIN_W4_PROFILE",
    "PAIRED_ROUTING_PILOT_PROFILE",
    "REVIEWER_R1_5_PROFILE",
    "APC_TERRAIN_SWEEP_CONTRACT_VERSION",
    "APC_TERRAIN_SWEEP_PROFILE",
    "PAPER1_APC_CONTRACT_VERSION",
    "PAPER1_APC_PROFILE",
    "APCTerrainSweepContractError",
    "Paper1APCContractError",
    "experiment_aggregation_metrics",
    "experiment_config_violations",
    "experiment_contract_version",
    "experiment_run_tables",
    "experiment_table_filenames",
    "registered_contract_profiles",
    "validate_experiment_grid_contract",
    "validate_experiment_spec_contract",
    "validate_run_output_contract",
]
