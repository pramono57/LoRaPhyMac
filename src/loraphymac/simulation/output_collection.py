"""Collect all scalar and tabular evidence after a simulation horizon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from loraphymac.metrics.instrumentation_metrics import (
    build_channel_occupancy_df,
    build_node_tx_duty_cycle_df,
    build_topology_parent_by_run_df,
)
from loraphymac.experiments.apc_load_sweep.contract import (
    APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_PILOT_PROFILE,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.contract import (
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE,
)
from loraphymac.experiments.apc_terrain_profile_diagnostic.full_sweep_contract import (
    APC_TERRAIN_PROFILE_FULL_SWEEP_PROFILE,
)
from loraphymac.experiments.apc_terrain_profile_diagnostic.channel_occupancy_pilot_contract import (
    APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_PROFILE,
)
from loraphymac.simulation.context import RoutingArtifacts
from loraphymac.simulation.results import build_master_summary_row


_NODE_CHANNEL_DUTY_PROFILES = frozenset(
    {
        "apc_load_sweep_v1",
        "apc_load_sweep_final_v1",
        "apc_load_sweep_full_factorial_v1",
        APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_PILOT_PROFILE,
        APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
        APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE,
        APC_TERRAIN_PROFILE_FULL_SWEEP_PROFILE,
        APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_PROFILE,
    }
)


def collects_node_channel_duty(profile: str) -> bool:
    return profile.startswith("scalability_r1_7_") or profile in _NODE_CHANNEL_DUTY_PROFILES


@dataclass(frozen=True, slots=True)
class CollectedSimulationOutputs:
    """Complete internal output contract for one simulation run."""

    master_summary: dict[str, Any]
    ptx_history: pd.DataFrame
    margin_history: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    summary_bundle: dict[str, pd.DataFrame]
    result_bundle: dict[str, pd.DataFrame]
    debug_bundle: dict[str, pd.DataFrame]

    def to_legacy_tuple(
        self,
    ) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
        """Return the historical four-item engine result contract."""
        return (
            self.master_summary,
            self.ptx_history,
            self.margin_history,
            self.tables,
        )


def collect_simulation_outputs(
    *,
    statistics: Any,
    routing: RoutingArtifacts,
) -> CollectedSimulationOutputs:
    """Build summary, analysis-result, and raw-debug tables once."""
    all_nodes = routing.all_nodes
    summary_bundle = statistics.get_summary_bundle(all_nodes)
    result_bundle = statistics.get_result_bundle(
        all_nodes,
        selected_next_hop_links_df=routing.selected_next_hop_links_df,
        relay_chain_links_df=routing.relay_chain_links_df,
        visual_links_df=routing.visual_links_df,
    )
    output_profile = str(getattr(statistics, "output_profile", "detailed"))
    debug_bundle = (
        statistics.get_debug_bundle()
        if output_profile == "detailed"
        else {}
    )

    core_summary_df = summary_bundle["01_core_summary"]
    energy_breakdown_df = result_bundle["R01_energy_breakdown"]
    node_rx_matrix_df = result_bundle["R02_node_rx_matrix"]
    hop_summary_df = summary_bundle["04_hop_summary"]
    frame_summary_df = summary_bundle.get("03_frame_summary", pd.DataFrame())
    frame_mtype_summary_df = summary_bundle.get(
        "05_frame_mtype_summary",
        pd.DataFrame(),
    )
    frame_hop_event_summary_df = summary_bundle.get(
        "06_frame_hop_event_summary",
        pd.DataFrame(),
    )
    frame_drop_reason_by_hop_df = summary_bundle.get(
        "07_frame_drop_reason_by_hop",
        pd.DataFrame(),
    )
    relay_duty_cycle_df = summary_bundle["08_relay_duty_cycle_summary"]
    per_channel_duty_cycle_df = result_bundle["R03_per_channel_duty_cycle"]
    tx_queue_waiting_time_summary_df = summary_bundle[
        "09_tx_queue_waiting_summary"
    ]
    final_ptx_distribution_df = result_bundle["R04_final_ptx_distribution"]
    relay_tx_mgc_queue_metrics_df = result_bundle["R05_relay_tx_mgc_queue"]
    relay_tx_queue_metrics_df = result_bundle["R06_relay_tx_queue_metrics"]
    tx_queue_traffic_class_metrics_df = result_bundle["R07_tx_queue_traffic_class"]
    collision_summary_df = summary_bundle.get(
        "11_collision_summary",
        pd.DataFrame(),
    )
    packet_fate_summary_df = summary_bundle.get(
        "12_packet_fate_summary",
        pd.DataFrame(),
    )

    ptx_history_df = result_bundle.get("R11_ptx_history", pd.DataFrame())
    margin_history_df = result_bundle.get("R12_margin_history", pd.DataFrame())
    apc_control_history_df = result_bundle.get("R13_apc_control_history", pd.DataFrame())

    run_metadata_df = summary_bundle.get("02_run_metadata", pd.DataFrame())
    event_log_df = debug_bundle.get("D01_event_log", pd.DataFrame())
    delivered_packet_df = result_bundle.get("R14_delivered_packets", pd.DataFrame())
    packet_failure_df = result_bundle.get("R15_packet_failures", pd.DataFrame())
    tx_queue_waiting_time_df = debug_bundle.get(
        "D02_tx_queue_waiting_time",
        pd.DataFrame(),
    )
    tx_queue_lifecycle_df = debug_bundle.get(
        "D03_tx_queue_lifecycle",
        pd.DataFrame(),
    )
    relay_rx_stats_df = result_bundle.get("R16_relay_rx_statistics", pd.DataFrame())
    rx_channel_usage_df = result_bundle.get("R17_rx_channel_usage", pd.DataFrame())
    demodulator_event_log_df = debug_bundle.get(
        "D04_demodulator_events",
        pd.DataFrame(),
    )
    demodulator_pool_summary_df = summary_bundle.get(
        "10_demodulator_pool_summary",
        pd.DataFrame(),
    )
    collision_event_log_df = debug_bundle.get(
        "D05_collision_events",
        pd.DataFrame(),
    )
    collision_interference_edges_df = debug_bundle.get(
        "D06_collision_interference_edges",
        pd.DataFrame(),
    )
    packet_fate_df = (
        debug_bundle.get("D07_packet_fates", pd.DataFrame())
        if output_profile == "detailed"
        else statistics.get_packet_fate_df()
    )
    apc_feedback_lifecycle_df = result_bundle.get(
        "R18_apc_feedback_lifecycle",
        pd.DataFrame(),
    )
    tx_airtime_by_entity_mtype_ptx_channel_df = result_bundle.get(
        "R19_tx_airtime_by_entity_mtype_ptx_channel",
        pd.DataFrame(),
    )
    wang_snr_report_lifecycle_df = result_bundle.get(
        "R20_wang_snr_report_lifecycle",
        pd.DataFrame(),
    )
    wang_power_command_lifecycle_df = result_bundle.get(
        "R22_wang_power_command_lifecycle",
        pd.DataFrame(),
    )
    application_queue_metrics_df = result_bundle.get(
        "R23_application_queue_metrics",
        pd.DataFrame(),
    )
    topology_parent_by_run_df = build_topology_parent_by_run_df(
        routing,
        getattr(statistics, "run_metadata", {}),
    )

    routing_table_df = pd.DataFrame(
        [
            {"source": source, "next_hop": next_hop}
            for source, next_hop in routing.routing_table.items()
        ]
    )
    routing_map_df = pd.DataFrame(
        [
            {"source": source, "next_hop": next_hop}
            for source, next_hop in routing.routing_map.items()
        ]
    )
    relay_hops_df = pd.DataFrame(
        [
            {"entity_id": entity_id, "hop": hop}
            for entity_id, hop in routing.relay_hops.items()
        ]
    )

    tables: dict[str, pd.DataFrame] = {
        "link_df": routing.link_df,
        "selected_next_hop_links_df": routing.selected_next_hop_links_df,
        "relay_chain_links_df": routing.relay_chain_links_df,
        "visual_links_df": routing.visual_links_df,
        "routing_table_df": routing_table_df,
        "routing_map_df": routing_map_df,
        "relay_hops_df": relay_hops_df,
        "core_summary_df": core_summary_df,
        "energy_breakdown_df": energy_breakdown_df,
        "node_rx_matrix_df": node_rx_matrix_df,
        "hop_summary_df": hop_summary_df,
        "frame_summary_df": frame_summary_df,
        "frame_mtype_summary_df": frame_mtype_summary_df,
        "frame_hop_event_summary_df": frame_hop_event_summary_df,
        "frame_drop_reason_by_hop_df": frame_drop_reason_by_hop_df,
        "relay_duty_cycle_df": relay_duty_cycle_df,
        "per_channel_duty_cycle_df": per_channel_duty_cycle_df,
        "tx_queue_waiting_time_summary_df": tx_queue_waiting_time_summary_df,
        "relay_tx_mgc_queue_metrics_df": relay_tx_mgc_queue_metrics_df,
        "relay_tx_queue_metrics_df": relay_tx_queue_metrics_df,
        "tx_queue_traffic_class_metrics_df": tx_queue_traffic_class_metrics_df,
        "final_ptx_distribution_df": final_ptx_distribution_df,
        "run_metadata_df": run_metadata_df,
        "event_log_df": event_log_df,
        "ptx_history_df": ptx_history_df,
        "margin_history_df": margin_history_df,
        "apc_control_history_df": apc_control_history_df,
        "delivered_packet_df": delivered_packet_df,
        "packet_failure_df": packet_failure_df,
        "tx_queue_waiting_time_df": tx_queue_waiting_time_df,
        "tx_queue_lifecycle_df": tx_queue_lifecycle_df,
        "relay_rx_stats_df": relay_rx_stats_df,
        "rx_channel_usage_df": rx_channel_usage_df,
        "demodulator_event_log_df": demodulator_event_log_df,
        "demodulator_pool_summary_df": demodulator_pool_summary_df,
        "collision_event_log_df": collision_event_log_df,
        "collision_interference_edges_df": collision_interference_edges_df,
        "collision_summary_df": collision_summary_df,
        "packet_fate_summary_df": packet_fate_summary_df,
        "packet_fate_df": packet_fate_df,
        "topology_parent_by_run_df": topology_parent_by_run_df,
        "apc_feedback_lifecycle_df": apc_feedback_lifecycle_df,
        "tx_airtime_by_entity_mtype_ptx_channel_df": (
            tx_airtime_by_entity_mtype_ptx_channel_df
        ),
        "wang_snr_report_lifecycle_df": wang_snr_report_lifecycle_df,
        "wang_power_command_lifecycle_df": wang_power_command_lifecycle_df,
        "application_queue_metrics_df": application_queue_metrics_df,
    }
    profile = str(
        getattr(statistics, "run_metadata", {}).get("experiment_profile", "")
    )
    if collects_node_channel_duty(profile):
        if output_profile == "summary":
            tables["node_tx_duty_cycle_df"] = (
                statistics.get_summary_node_tx_duty_cycle_df(all_nodes)
            )
            tables["channel_occupancy_df"] = (
                statistics.get_summary_channel_occupancy_df()
            )
        else:
            tables["node_tx_duty_cycle_df"] = build_node_tx_duty_cycle_df(
                event_log_df,
                all_nodes,
                getattr(statistics, "run_metadata", {}),
            )
            tables["channel_occupancy_df"] = build_channel_occupancy_df(
                event_log_df,
                getattr(statistics, "run_metadata", {}),
            )
    if profile == "apc_feedback_robustness_sweep_v1":
        tables.update(statistics.get_feedback_robustness_tables())
    master_summary = build_master_summary_row(
        core_summary_df=core_summary_df,
        frame_summary_df=frame_summary_df,
        node_rx_matrix_df=node_rx_matrix_df,
        selected_next_hop_links_df=routing.selected_next_hop_links_df,
        packet_failure_df=packet_failure_df,
    )
    return CollectedSimulationOutputs(
        master_summary=master_summary,
        ptx_history=ptx_history_df,
        margin_history=margin_history_df,
        tables=tables,
        summary_bundle=summary_bundle,
        result_bundle=result_bundle,
        debug_bundle=debug_bundle,
    )
