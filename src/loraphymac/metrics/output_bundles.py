"""Stable output-bundle assembly for :class:`Statistics`.

This module intentionally depends on the public ``Statistics`` methods instead
of its internal storage.  Keeping that dependency direction lets
``metrics.statistics.Statistics`` remain the facade while metric families are
extracted incrementally.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_summary_bundle(statistics: Any, nodes: list) -> dict[str, pd.DataFrame]:
    """Build the compact scientific summary workbook payload."""
    return {
        "01_core_summary": statistics.get_core_summary_df(nodes),
        "02_run_metadata": statistics.get_run_metadata_df(),
        "03_frame_summary": statistics.get_frame_summary_df(),
        "04_hop_summary": statistics.get_hop_summary_df(),
        "05_frame_mtype_summary": statistics.get_frame_mtype_summary_df(),
        "06_frame_hop_event_summary": statistics.get_frame_hop_event_summary_df(),
        "07_frame_drop_reason_by_hop": statistics.get_frame_drop_reason_by_hop_df(),
        "08_relay_duty_cycle_summary": statistics.get_relay_duty_cycle_df(nodes),
        "09_tx_queue_waiting_summary": statistics.get_tx_queue_waiting_time_summary_df(),
        "10_demodulator_pool_summary": statistics.get_demodulator_pool_summary_df(),
        "11_collision_summary": statistics.get_collision_summary_df(),
        "12_packet_fate_summary": statistics.get_packet_fate_summary_df(),
    }


def build_result_bundle(
    statistics: Any,
    nodes: list,
    selected_next_hop_links_df: pd.DataFrame,
    relay_chain_links_df: pd.DataFrame | None = None,
    visual_links_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build analysis-grade tables used for figures and diagnostics."""
    empty = pd.DataFrame()
    return {
        "R01_energy_breakdown": statistics.get_energy_breakdown_df(nodes),
        "R02_node_rx_matrix": statistics.get_node_rx_matrix_df(),
        "R03_per_channel_duty_cycle": statistics.get_per_channel_duty_cycle_df(nodes),
        "R04_final_ptx_distribution": statistics.get_final_ptx_distribution_df(nodes),
        "R05_relay_tx_mgc_queue": statistics.get_relay_tx_mgc_queue_metrics_df(
            nodes,
            sim_time_s=statistics.run_metadata.get("sim_time", None),
        ),
        "R06_relay_tx_queue_metrics": statistics.get_relay_tx_queue_metrics_df(
            nodes,
            sim_time_s=statistics.run_metadata.get("sim_time", None),
        ),
        "R07_tx_queue_traffic_class": statistics.get_tx_queue_traffic_class_metrics_df(),
        "R08_selected_next_hop_links": selected_next_hop_links_df,
        "R09_relay_chain_links": (
            relay_chain_links_df if relay_chain_links_df is not None else empty
        ),
        "R10_visual_links": visual_links_df if visual_links_df is not None else empty,
        "R11_ptx_history": statistics.get_ptx_history_df(),
        "R12_margin_history": statistics.get_margin_history_df(),
        "R13_apc_control_history": statistics.get_apc_control_history_df(),
        "R14_delivered_packets": statistics.get_delivered_packet_df(),
        "R15_packet_failures": statistics.get_packet_failure_df(),
        "R16_relay_rx_statistics": statistics.get_relay_rx_stats_df(),
        "R17_rx_channel_usage": statistics.get_rx_channel_usage_df(),
        "R18_apc_feedback_lifecycle": (
            statistics.get_apc_feedback_lifecycle_df()
        ),
        "R19_tx_airtime_by_entity_mtype_ptx_channel": (
            statistics.get_tx_airtime_by_entity_mtype_ptx_channel_df(nodes)
        ),
        "R20_wang_snr_report_lifecycle": (
            statistics.get_wang_snr_report_lifecycle_df()
        ),
        "R22_wang_power_command_lifecycle": (
            statistics.get_wang_power_command_lifecycle_df()
        ),
        "R23_application_queue_metrics": (
            statistics.get_application_queue_metrics_df()
        ),
    }


def build_debug_bundle(statistics: Any) -> dict[str, pd.DataFrame]:
    """Build raw event/lifecycle evidence separately from result tables."""
    return {
        "D01_event_log": statistics.to_dataframe(),
        "D02_tx_queue_waiting_time": statistics.get_tx_queue_waiting_time_df(),
        "D03_tx_queue_lifecycle": statistics.get_tx_queue_lifecycle_df(),
        "D04_demodulator_events": statistics.get_demodulator_event_log_df(),
        "D05_collision_events": statistics.get_collision_event_log_df(),
        "D06_collision_interference_edges": (statistics.get_collision_interference_edges_df()),
        "D07_packet_fates": statistics.get_packet_fate_df(),
    }
