"""Scientific output requirements and preflight profile compatibility.

This module describes evidence requirements separately from the experiment
grid contracts. Existing run-table registries remain the source of truth for
experiment-specific table names. The persistence selector is declarative at
this stage; the legacy writers are not changed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from loraphymac.experiments.apc_closed_loop_verification.contract import (
    APC_CLOSED_LOOP_PROFILE,
)
from loraphymac.experiments.apc_load_sweep.contract import (
    APC_LOAD_SWEEP_FINAL_PROFILE,
    APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE,
    APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_PILOT_PROFILE,
    APC_LOAD_SWEEP_STANDARD_PROFILE,
    APC_LOAD_SWEEP_PILOT_PROFILE,
)
from loraphymac.experiments.apc_feedback_robustness_sweep.contract import (
    APC_FEEDBACK_ROBUSTNESS_PROFILE,
    APC_FEEDBACK_ROBUSTNESS_RUN_TABLES,
    TABLE_SCHEMAS as FEEDBACK_ROBUSTNESS_TABLE_SCHEMAS,
)
from loraphymac.experiments.apc_terrain_sweep.contract import (
    APC_TERRAIN_SWEEP_PROFILE,
    PAPER1_APC_PROFILE,
)
from loraphymac.experiments.common.contract_registry import (
    experiment_aggregation_metrics,
    experiment_run_tables,
)
from loraphymac.metrics.apc_metrics import PTX_HISTORY_COLUMNS
from loraphymac.metrics.collision_metrics import COLLISION_SUMMARY_COLUMNS
from loraphymac.metrics.instrumentation_metrics import (
    APC_FEEDBACK_LIFECYCLE_COLUMNS,
    WANG_POWER_COMMAND_LIFECYCLE_COLUMNS,
    WANG_SNR_REPORT_LIFECYCLE_COLUMNS,
)
from loraphymac.metrics.packet_metrics import PACKET_FATE_COLUMNS
from loraphymac.experiments.apc_compare_model_terrain_sweep.contract import (
    APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_AGGREGATION_METRICS,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE,
)
from loraphymac.experiments.scalability_relay_bottleneck.r1_7_contract import (
    APC_LOAD_SWEEP_PROFILE,
    R1_7_DUTY_PROFILE,
    R1_7_LOAD_PROFILE,
)


PERSISTENCE_PROFILES = frozenset({"core", "scientific", "diagnostic"})
LEGACY_PERSISTENCE_PROFILE = "legacy"


@dataclass(frozen=True, slots=True)
class ScientificOutputContract:
    """Requirements shared by profiles belonging to one experiment family."""

    contract_id: str
    required_metrics: frozenset[str]
    required_additional_tables: frozenset[str]
    minimum_collection_profile: str
    minimum_persistence_profile: str
    producer_modules: tuple[str, ...]
    granularity_categories: tuple[str, ...]
    validator_dependencies: tuple[str, ...]
    postprocessor_dependencies: tuple[str, ...]
    provenance_fields: frozenset[str]
    # Core persistence is denied unless a contract explicitly lists its small
    # evidence set. Existing experiment contracts keep the safe empty default.
    core_table_keys: frozenset[str] = frozenset()
    # V2 event tables distinguish an inapplicable empty table from a missing
    # artifact, and from an applicable table whose schema is present but has no
    # events in this run.
    conditional_table_semantics: Mapping[str, Mapping[str, str]] = MappingProxyType({})
    table_schemas: Mapping[str, tuple[str, ...]] = MappingProxyType({})


_COMMON_METRICS = frozenset(
    {
        "pdr_percent",
        "avg_e2e_delay_ms",
        "total_network_energy_J",
        "packet_fate_conservation_delta",
        "collision_outcome_conservation_delta",
    }
)
_COMMON_TABLES = frozenset(
    {
        "core_summary_df",
        "packet_fate_summary_df",
        "collision_summary_df",
        "energy_breakdown_df",
        "tx_airtime_by_entity_mtype_ptx_channel_df",
        "topology_parent_by_run_df",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "run_id",
        "seed",
        "experiment_profile",
        "config_hash",
        "source_fingerprint",
        "terrain_delta_h",
        "node_num",
        "node_period",
        "model_id",
        "topology_sha256",
        "routing_sha256",
        "offered_schedule_sha256",
        "energy_accounting_mode",
    }
)

_CLOSED_LOOP = ScientificOutputContract(
    contract_id="apc_closed_loop_scientific_v1",
    required_metrics=_COMMON_METRICS
    | frozenset(
        {
            "delivery_by_horizon_percent",
            "terminal_loss_percent",
            "censored_fraction_percent",
            "apc_total_tx",
            "apc_total_rx",
            "apc_total_drop",
            "ptx_update_count",
        }
    ),
    required_additional_tables=_COMMON_TABLES
    | frozenset(
        {
            "run_metadata_df",
            "frame_summary_df",
            "final_ptx_distribution_df",
            "ptx_history_df",
            "margin_history_df",
            "apc_control_history_df",
            "apc_feedback_lifecycle_df",
            "relay_duty_cycle_df",
            "per_channel_duty_cycle_df",
        }
    ),
    minimum_collection_profile="summary",
    minimum_persistence_profile="scientific",
    producer_modules=(
        "loraphymac.metrics.statistics.Statistics",
        "loraphymac.metrics.output_bundles",
        "loraphymac.simulation.output_collection",
    ),
    granularity_categories=(
        "one row/run or grouped counters",
        "per entity/link/channel",
        "per APC/control transition",
    ),
    validator_dependencies=(
        "validate_apc_closed_loop_outputs",
        "packet_fate_conservation_delta",
        "collision_outcome_conservation_delta",
    ),
    postprocessor_dependencies=(
        "apc_closed_loop_verification.workflow",
        "apc_closed_loop_verification.run",
    ),
    provenance_fields=_PROVENANCE_FIELDS,
)

_TERRAIN_SWEEP = ScientificOutputContract(
    contract_id="apc_terrain_sweep_scientific_v1",
    required_metrics=_COMMON_METRICS
    | frozenset(
        {
            "delivery_by_horizon_percent",
            "terminal_loss_percent",
            "censored_fraction_percent",
            "energy_per_delivered_packet_J",
        }
    ),
    required_additional_tables=_COMMON_TABLES
    | frozenset(
        {
            "run_metadata_df",
            "frame_summary_df",
            "final_ptx_distribution_df",
            "packet_fate_df",
            "relay_duty_cycle_df",
            "per_channel_duty_cycle_df",
            "relay_tx_queue_metrics_df",
            "tx_queue_traffic_class_metrics_df",
            "demodulator_pool_summary_df",
        }
    ),
    minimum_collection_profile="summary",
    minimum_persistence_profile="scientific",
    producer_modules=(
        "loraphymac.metrics.statistics.Statistics",
        "loraphymac.metrics.output_bundles",
        "loraphymac.simulation.output_collection",
    ),
    granularity_categories=(
        "one row/run or grouped counters",
        "per node/relay/link/channel",
        "per logical application packet where pairing requires it",
        "per APC/PTx history row at contracted comparison cells",
    ),
    validator_dependencies=(
        "validate_apc_terrain_sweep_outputs",
        "packet_fate_conservation_delta",
        "collision_outcome_conservation_delta",
    ),
    postprocessor_dependencies=(
        "apc_terrain_sweep.postprocessing.build_terrain_sweep_artifacts",
        "apc_terrain_sweep.routing_audit",
    ),
    provenance_fields=_PROVENANCE_FIELDS,
)

_LOAD_SWEEP = ScientificOutputContract(
    contract_id="apc_load_sweep_scientific_v1",
    required_metrics=_COMMON_METRICS
    | frozenset(
        {
            "packet_fate_offered_count",
            "packet_fate_delivered_count",
            "packet_fate_terminal_drop_count",
            "packet_fate_censored_count",
            "energy_per_delivered_packet_J",
            "total_constrained_energy_J",
            "collision_drop_decision_count",
            "collision_opportunity_count",
            "collision_capture_success_rate_percent",
            "relay_fifo_wait_mean_ms_packet_weighted",
            "ptx_update_count",
            "apc_total_tx",
            "apc_total_rx",
            "apc_total_drop",
            "relay_max_waiting_end_count",
            "relay_max_worker_occupied_fraction",
            # Derived by the load-sweep reporter as delivered application bits
            # divided by the configured observation horizon.
            "throughput_bps",
        }
    ),
    required_additional_tables=_COMMON_TABLES
    | frozenset(
        {
            "packet_fate_df",
            "run_metadata_df",
            "frame_summary_df",
            "frame_mtype_summary_df",
            "hop_summary_df",
            "frame_hop_event_summary_df",
            "frame_drop_reason_by_hop_df",
            "relay_duty_cycle_df",
            "per_channel_duty_cycle_df",
            "node_tx_duty_cycle_df",
            "channel_occupancy_df",
            "relay_rx_stats_df",
            "relay_tx_queue_metrics_df",
            "relay_tx_mgc_queue_metrics_df",
            "tx_queue_traffic_class_metrics_df",
            "tx_queue_waiting_time_summary_df",
            "demodulator_pool_summary_df",
            "application_queue_metrics_df",
            "selected_next_hop_links_df",
            "relay_chain_links_df",
            "apc_feedback_lifecycle_df",
            "ptx_history_df",
            "final_ptx_distribution_df",
        }
    ),
    minimum_collection_profile="summary",
    minimum_persistence_profile="scientific",
    producer_modules=(
        "loraphymac.metrics.statistics.Statistics",
        "loraphymac.metrics.output_bundles",
        "loraphymac.simulation.output_collection",
    ),
    granularity_categories=(
        "one row/run or grouped counters",
        "per node/relay/channel/queue class",
        "per logical offered packet for schedule/fate pairing",
        "per APC lifecycle/PTx application event",
    ),
    validator_dependencies=(
        "validate_apc_load_sweep_outputs",
        "packet_fate_conservation_delta",
        "collision_outcome_conservation_delta",
        "offered_schedule_pairing",
    ),
    postprocessor_dependencies=(
        "apc_load_sweep.full_factorial_postprocessing",
        "apc_load_sweep.postprocessing",
    ),
    provenance_fields=_PROVENANCE_FIELDS,
    conditional_table_semantics=MappingProxyType(
        {
            "apc_feedback_lifecycle_df": MappingProxyType(
                {
                    "ff": "not_applicable_empty",
                    "aa": "schema_empty_allowed",
                }
            ),
            "ptx_history_df": MappingProxyType(
                {
                    "ff": "schema_empty_allowed",
                    "aa": "schema_empty_allowed",
                }
            ),
        }
    ),
    table_schemas=MappingProxyType(
        {
            "packet_fate_df": tuple(PACKET_FATE_COLUMNS),
            "collision_summary_df": tuple(COLLISION_SUMMARY_COLUMNS),
            "ptx_history_df": tuple(PTX_HISTORY_COLUMNS),
            "apc_feedback_lifecycle_df": tuple(APC_FEEDBACK_LIFECYCLE_COLUMNS),
        }
    ),
)

# V1 is a historical three-model contract. Keep its exact persisted evidence
# surface separate from the newer terrain-sweep scientific contract: V1 did
# not produce every newer table (for example node duty and channel occupancy).
_COMPARE_MODEL_V1_COMPAT = ScientificOutputContract(
    contract_id="apc_compare_model_terrain_sweep_v1_compat",
    required_metrics=_COMMON_METRICS
    | frozenset(APC_COMPARE_MODEL_TERRAIN_SWEEP_AGGREGATION_METRICS),
    required_additional_tables=_COMMON_TABLES
    | frozenset(
        {
            "run_metadata_df",
            "final_ptx_distribution_df",
            "ptx_history_df",
            "packet_fate_df",
            "relay_duty_cycle_df",
            "per_channel_duty_cycle_df",
            "relay_rx_stats_df",
            "relay_tx_queue_metrics_df",
            "tx_queue_traffic_class_metrics_df",
            "demodulator_pool_summary_df",
            "apc_feedback_lifecycle_df",
        }
    ),
    minimum_collection_profile="summary",
    minimum_persistence_profile="scientific",
    producer_modules=(
        "loraphymac.metrics.statistics.Statistics",
        "loraphymac.metrics.output_bundles",
        "loraphymac.simulation.output_collection",
        "loraphymac.experiments.common.multiseed_pipeline",
    ),
    granularity_categories=(
        "one row/run and grouped counters",
        "per logical packet for fate conservation",
        "per node/relay/link/channel for energy and routing evidence",
        "per PTx update event; an empty event table is valid for fixed baselines",
    ),
    validator_dependencies=(
        "validate_apc_compare_model_terrain_sweep_outputs",
        "packet_fate_conservation_delta",
        "collision_outcome_conservation_delta",
    ),
    postprocessor_dependencies=(
        "apc_compare_model_terrain_sweep.postprocessing",
        "apc_compare_model_terrain_sweep.run",
    ),
    provenance_fields=_PROVENANCE_FIELDS,
)

_V2_FIXED_MODELS = ("ff", "islam_dbpc_itm6", "islam_dbpc_802")
_COMPARE_MODEL_V2 = ScientificOutputContract(
    contract_id="apc_compare_model_terrain_sweep_v2_scientific",
    required_metrics=(
        _COMMON_METRICS
        | frozenset(APC_COMPARE_MODEL_TERRAIN_SWEEP_AGGREGATION_METRICS)
        | frozenset({"throughput_bps"})
    ),
    required_additional_tables=_COMMON_TABLES
    | frozenset(
        {
            "run_metadata_df",
            "frame_summary_df",
            "frame_mtype_summary_df",
            "hop_summary_df",
            "frame_hop_event_summary_df",
            "frame_drop_reason_by_hop_df",
            "final_ptx_distribution_df",
            "ptx_history_df",
            "packet_fate_df",
            "relay_duty_cycle_df",
            "per_channel_duty_cycle_df",
            "node_tx_duty_cycle_df",
            "channel_occupancy_df",
            "relay_rx_stats_df",
            "relay_tx_mgc_queue_metrics_df",
            "relay_tx_queue_metrics_df",
            "tx_queue_traffic_class_metrics_df",
            "tx_queue_waiting_time_summary_df",
            "demodulator_pool_summary_df",
            "selected_next_hop_links_df",
            "relay_chain_links_df",
            "apc_feedback_lifecycle_df",
            "application_queue_metrics_df",
            "wang_snr_report_lifecycle_df",
            "wang_power_command_lifecycle_df",
        }
    ),
    minimum_collection_profile="summary",
    minimum_persistence_profile="scientific",
    producer_modules=(
        "loraphymac.metrics.statistics.Statistics",
        "loraphymac.metrics.output_bundles",
        "loraphymac.simulation.output_collection",
        "loraphymac.simulation.runner",
        "loraphymac.experiments.common.multiseed_pipeline",
        "loraphymac.experiments.apc_compare_model_terrain_sweep.postprocessing._run_metrics",
    ),
    granularity_categories=(
        "one row/run and grouped core counters",
        "per offered logical packet for fate and pairing",
        "per node/relay/link/channel/queue for operational evidence",
        "per control lifecycle event and per Wang observation window",
    ),
    validator_dependencies=(
        "validate_apc_compare_model_terrain_sweep_outputs",
        "packet_fate_conservation_delta",
        "collision_outcome_conservation_delta",
        "corrected_energy_breakdown",
    ),
    postprocessor_dependencies=(
        "apc_compare_model_terrain_sweep.postprocessing",
        "apc_compare_model_terrain_sweep.run",
    ),
    provenance_fields=_PROVENANCE_FIELDS,
    conditional_table_semantics=MappingProxyType(
        {
            "application_queue_metrics_df": MappingProxyType(
                {model: "required_non_empty" for model in (
                    "ff", "aa", "islam_dbpc_itm6", "islam_dbpc_802",
                    "wang_atp_sf7_a2only",
                )}
            ),
            "ptx_history_df": MappingProxyType(
                {
                    **{model: "not_applicable_empty" for model in _V2_FIXED_MODELS},
                    "aa": "schema_empty_allowed",
                    "wang_atp_sf7_a2only": "schema_empty_allowed",
                }
            ),
            "apc_feedback_lifecycle_df": MappingProxyType(
                {
                    **{model: "not_applicable_empty" for model in (*_V2_FIXED_MODELS, "wang_atp_sf7_a2only")},
                    "aa": "schema_empty_allowed",
                }
            ),
            "wang_snr_report_lifecycle_df": MappingProxyType(
                {
                    **{model: "not_applicable_empty" for model in (
                        "ff", "aa", "islam_dbpc_itm6", "islam_dbpc_802"
                    )},
                    "wang_atp_sf7_a2only": "required_non_empty",
                }
            ),
            "wang_power_command_lifecycle_df": MappingProxyType(
                {
                    **{model: "not_applicable_empty" for model in (
                        "ff", "aa", "islam_dbpc_itm6", "islam_dbpc_802"
                    )},
                    "wang_atp_sf7_a2only": "schema_empty_allowed",
                }
            ),
            "wang_sink_decision_df": MappingProxyType(
                {
                    **{model: "not_applicable_empty" for model in (
                        "ff", "aa", "islam_dbpc_itm6", "islam_dbpc_802"
                    )},
                    "wang_atp_sf7_a2only": "required_non_empty",
                }
            ),
        }
    ),
    table_schemas=MappingProxyType(
        {
            "ptx_history_df": tuple(PTX_HISTORY_COLUMNS),
            "apc_feedback_lifecycle_df": tuple(APC_FEEDBACK_LIFECYCLE_COLUMNS),
            "wang_snr_report_lifecycle_df": tuple(WANG_SNR_REPORT_LIFECYCLE_COLUMNS),
            "wang_power_command_lifecycle_df": tuple(WANG_POWER_COMMAND_LIFECYCLE_COLUMNS),
        }
    ),
)

_PROFILE_TO_CONTRACT = MappingProxyType(
    {
        APC_FEEDBACK_ROBUSTNESS_PROFILE: ScientificOutputContract(
            contract_id="apc_feedback_robustness_scientific_v1_1",
            required_metrics=frozenset({
                "pdr_percent", "avg_e2e_delay_ms", "total_network_energy_J",
                "energy_per_delivered_packet_J", "packet_fate_conservation_delta",
                "collision_outcome_conservation_delta",
            }),
            required_additional_tables=frozenset(APC_FEEDBACK_ROBUSTNESS_RUN_TABLES),
            minimum_collection_profile="summary",
            minimum_persistence_profile="scientific",
            producer_modules=(
                "metrics/statistics.py", "simulation/output_collection.py",
                "runtime/node_rx.py", "runtime/relay_rx.py",
            ),
            granularity_categories=(
                "per-run summary", "feedback physical attempt", "logical transaction",
                "ephemeral TX-start TPI snapshot", "bounded per-transmitter/per-mtype counters",
            ),
            validator_dependencies=("experiments/apc_feedback_robustness_sweep/contract.py",),
            postprocessor_dependencies=("experiments/common/multiseed_pipeline.py",),
            provenance_fields=frozenset(_PROVENANCE_FIELDS),
            table_schemas=MappingProxyType(FEEDBACK_ROBUSTNESS_TABLE_SCHEMAS),
        ),
        APC_CLOSED_LOOP_PROFILE: _CLOSED_LOOP,
        APC_TERRAIN_SWEEP_PROFILE: _TERRAIN_SWEEP,
        PAPER1_APC_PROFILE: _TERRAIN_SWEEP,
        APC_LOAD_SWEEP_FINAL_PROFILE: _LOAD_SWEEP,
        APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE: _LOAD_SWEEP,
        # Pilot changes only the plan size/output profile; per-run scientific
        # evidence is the same as the full-factorial APC load-sweep contract.
        APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_PILOT_PROFILE: _LOAD_SWEEP,
        APC_LOAD_SWEEP_STANDARD_PROFILE: _LOAD_SWEEP,
        APC_LOAD_SWEEP_PILOT_PROFILE: _LOAD_SWEEP,
        APC_LOAD_SWEEP_PROFILE: _LOAD_SWEEP,
        R1_7_LOAD_PROFILE: _LOAD_SWEEP,
        R1_7_DUTY_PROFILE: _LOAD_SWEEP,
        APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE: _COMPARE_MODEL_V1_COMPAT,
        APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE: _COMPARE_MODEL_V2,
        APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE: _COMPARE_MODEL_V2,
    }
)

# The simulator's summary and result bundles provide these tables in both
# collection modes. Detailed additionally retains the raw diagnostic tables.
_SUMMARY_COLLECTION_TABLES = frozenset(
    {
        "link_df",
        "selected_next_hop_links_df",
        "relay_chain_links_df",
        "visual_links_df",
        "routing_table_df",
        "routing_map_df",
        "relay_hops_df",
        "core_summary_df",
        "energy_breakdown_df",
        "node_rx_matrix_df",
        "hop_summary_df",
        "frame_summary_df",
        "frame_mtype_summary_df",
        "frame_hop_event_summary_df",
        "frame_drop_reason_by_hop_df",
        "relay_duty_cycle_df",
        "per_channel_duty_cycle_df",
        "tx_queue_waiting_time_summary_df",
        "relay_tx_mgc_queue_metrics_df",
        "relay_tx_queue_metrics_df",
        "tx_queue_traffic_class_metrics_df",
        "final_ptx_distribution_df",
        "run_metadata_df",
        "ptx_history_df",
        "margin_history_df",
        "apc_control_history_df",
        "delivered_packet_df",
        "packet_failure_df",
        "relay_rx_stats_df",
        "rx_channel_usage_df",
        "demodulator_pool_summary_df",
        "collision_summary_df",
        "packet_fate_summary_df",
        "packet_fate_df",
        "topology_parent_by_run_df",
        "apc_feedback_lifecycle_df",
        "tx_airtime_by_entity_mtype_ptx_channel_df",
        "wang_snr_report_lifecycle_df",
        "wang_power_command_lifecycle_df",
        "application_queue_metrics_df",
        "node_tx_duty_cycle_df",
        "channel_occupancy_df",
        # The V2 runner appends this after collection from the gateway decision
        # engine; it is nevertheless part of the summary scientific bundle.
        "wang_sink_decision_df",
        "feedback_attempt_metrics_df",
        "feedback_transaction_metrics_df",
        "stale_tpi_transmission_summary_df",
        "apc_convergence_metrics_df",
        "apc_control_airtime_summary_df",
    }
)
_DIAGNOSTIC_COLLECTION_TABLES = frozenset(
    {
        "event_log_df",
        "tx_queue_lifecycle_df",
        "demodulator_event_log_df",
        "collision_event_log_df",
        "collision_interference_edges_df",
        "packet_fate_df",
    }
)


def output_contract_for(config: Mapping[str, Any]) -> ScientificOutputContract | None:
    """Return the scientific contract for a resolved experiment config."""
    return _PROFILE_TO_CONTRACT.get(str(config.get("experiment_profile")))


def effective_persistence_profile(config: Mapping[str, Any]) -> str:
    """Resolve an omitted/None persistence selector without mutating config."""
    value = config.get("persistence_profile")
    return LEGACY_PERSISTENCE_PROFILE if value is None else str(value)


def required_table_keys(config: Mapping[str, Any]) -> frozenset[str]:
    """Combine existing experiment table registrations with scientific minima."""
    contract = output_contract_for(config)
    if contract is None:
        return frozenset()
    registered = frozenset(experiment_run_tables(config))
    return registered | contract.required_additional_tables


def required_metrics(config: Mapping[str, Any]) -> frozenset[str]:
    """Combine common scientific metrics with the existing aggregation registry."""
    contract = output_contract_for(config)
    if contract is None:
        return frozenset()
    registered = experiment_aggregation_metrics(str(config.get("experiment_profile")), ())
    return contract.required_metrics | frozenset(registered)


def validate_conditional_table_semantics(
    config: Mapping[str, Any], tables: Mapping[str, Any]
) -> list[str]:
    """Validate model-specific empty/non-empty semantics for registered tables."""
    contract = output_contract_for(config)
    if contract is None:
        return []
    model = str(config.get("tx_power_model_id", config.get("model_id", "")))
    if not model and config.get("node_ptx_mode") == "fixed":
        model = "ff"
    elif not model and config.get("node_ptx_mode") == "adaptive":
        model = "aa"
    errors: list[str] = []
    for key, expected_schema in contract.table_schemas.items():
        frame = tables.get(key)
        if frame is not None and tuple(getattr(frame, "columns", ())) != expected_schema:
            errors.append(f"{key} schema does not match the registered producer schema")
    for key, per_model in contract.conditional_table_semantics.items():
        policy = per_model.get(model)
        if policy is None:
            continue
        frame = tables.get(key)
        if frame is None:
            # Presence is checked by required_table_keys/select_tables.
            continue
        if not hasattr(frame, "empty") or not hasattr(frame, "columns"):
            errors.append(f"{key} must be a DataFrame")
            continue
        if policy == "required_non_empty" and frame.empty:
            errors.append(f"{key} must contain rows for model {model}")
        elif policy == "not_applicable_empty" and not frame.empty:
            errors.append(f"{key} must be empty because it is not applicable to model {model}")
        elif policy not in {
            "required_non_empty", "not_applicable_empty", "schema_empty_allowed"
        }:
            errors.append(f"unknown conditional table policy {policy!r} for {key}")
    return errors


def summary_collection_table_keys() -> frozenset[str]:
    """Return the non-debug table keys currently built by output collection."""
    return _SUMMARY_COLLECTION_TABLES


def diagnostic_collection_table_keys() -> frozenset[str]:
    """Return raw tables that require detailed event collection."""
    return _DIAGNOSTIC_COLLECTION_TABLES


def preflight_output_violations(config: Mapping[str, Any]) -> list[str]:
    """Check collection and persistence compatibility before simulation starts."""
    persistence = effective_persistence_profile(config)
    if persistence not in {LEGACY_PERSISTENCE_PROFILE, *PERSISTENCE_PROFILES}:
        return ["persistence_profile must be one of core, scientific, diagnostic, or omitted"]

    contract = output_contract_for(config)
    if contract is None:
        return []

    errors: list[str] = []
    collection = str(config.get("output_profile", "detailed"))
    if collection not in {"summary", "detailed"}:
        return ["output_profile must be summary or detailed"]
    missing_collection = sorted(required_table_keys(config) - _SUMMARY_COLLECTION_TABLES)
    if missing_collection:
        errors.append(
            f"collection profile {collection!r} cannot provide required tables: "
            + ", ".join(missing_collection)
        )
    if persistence == "diagnostic" and collection != "detailed":
        errors.append(
            "persistence_profile='diagnostic' requires output_profile='detailed' "
            "for raw event tables: " + ", ".join(sorted(_DIAGNOSTIC_COLLECTION_TABLES))
        )

    if persistence == "core" and not contract.core_table_keys:
        errors.append(
            f"{contract.contract_id} requires persistence_profile='scientific' "
            "or 'diagnostic'; core would omit contract-required evidence: "
            + ", ".join(sorted(required_table_keys(config)))
        )
    elif persistence == "core":
        missing_core = sorted(contract.core_table_keys - _SUMMARY_COLLECTION_TABLES)
        if missing_core:
            errors.append(
                "core persistence tables are unavailable from collection profile: "
                + ", ".join(missing_core)
            )
    if persistence in PERSISTENCE_PROFILES and (persistence != "core" or contract.core_table_keys):
        rank = {"core": 0, "scientific": 1, "diagnostic": 2}
        minimum = contract.minimum_persistence_profile
        if minimum in rank and rank[persistence] < rank[minimum]:
            errors.append(
                f"{contract.contract_id} requires at least persistence_profile={minimum!r}"
            )
    return errors


__all__ = [
    "LEGACY_PERSISTENCE_PROFILE",
    "PERSISTENCE_PROFILES",
    "ScientificOutputContract",
    "effective_persistence_profile",
    "diagnostic_collection_table_keys",
    "output_contract_for",
    "preflight_output_violations",
    "required_metrics",
    "required_table_keys",
    "summary_collection_table_keys",
    "validate_conditional_table_semantics",
]
