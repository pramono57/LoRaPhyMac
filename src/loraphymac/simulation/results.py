"""Assembly helpers for scalar simulation results."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_master_summary_row(
    core_summary_df,
    frame_summary_df,
    node_rx_matrix_df,
    selected_next_hop_links_df,
    packet_failure_df,
):
    """Combine core, APC, feasibility, service, and drop summary fields."""
    if isinstance(core_summary_df, pd.DataFrame) and not core_summary_df.empty:
        row = core_summary_df.iloc[0].to_dict()
    else:
        row = {}

    apc_cols = [
        "total_data_down_tx",
        "total_data_down_rx",
        "total_data_down_drop",
        "total_data_down_fwd_tx",
        "total_data_down_fwd_rx",
        "total_data_down_fwd_drop",
        "apc_total_tx",
        "apc_total_rx",
        "apc_total_drop",
        "apc_apply_success_percent",
        "apc_down_node_success_percent",
        "apc_down_relay_success_percent",
        "ptx_update_count",
        "stage1_status",
    ]

    if isinstance(frame_summary_df, pd.DataFrame) and not frame_summary_df.empty:
        frame = frame_summary_df.iloc[0].to_dict()
        for column in apc_cols:
            row[column] = frame.get(column, np.nan)
    else:
        for column in apc_cols:
            row[column] = np.nan

    if (
        isinstance(selected_next_hop_links_df, pd.DataFrame)
        and not selected_next_hop_links_df.empty
    ):
        node_links = selected_next_hop_links_df[
            selected_next_hop_links_df["Node"].astype(str).str.startswith("N")
        ]
        row["feasible_node_count"] = int(node_links["Valid"].sum())
        row["feasible_node_ratio_percent"] = (
            100.0 * node_links["Valid"].sum() / len(node_links)
            if len(node_links) > 0
            else np.nan
        )

    if isinstance(node_rx_matrix_df, pd.DataFrame) and not node_rx_matrix_df.empty:
        total_nodes = len(node_rx_matrix_df)
        served_nodes = int((node_rx_matrix_df["gateway"] > 0).sum())
        row["served_node_count"] = served_nodes
        row["served_node_ratio_percent"] = (
            100.0 * served_nodes / total_nodes if total_nodes > 0 else np.nan
        )

    row["drop_forwarding_queue_overflow_count"] = 0
    if (
        isinstance(packet_failure_df, pd.DataFrame)
        and not packet_failure_df.empty
        and "reason" in packet_failure_df.columns
    ):
        reasons = packet_failure_df["reason"].astype(str)
        weights = (
            packet_failure_df["event_count"]
            if "event_count" in packet_failure_df
            else pd.Series(1, index=packet_failure_df.index)
        )
        row["drop_not_feasible_count"] = int(weights.loc[reasons == "not_feasible"].sum())
        row["drop_collision_count"] = int(weights.loc[reasons == "collision"].sum())
        row["drop_half_duplex_count"] = int(weights.loc[reasons.str.contains("half_duplex")].sum())
        row["drop_demod_busy_count"] = int(weights.loc[reasons == "demod_busy"].sum())
        row["drop_node_rx_window_closed_count"] = int(weights.loc[reasons == "node_rx_window_closed"].sum())
        row["drop_forwarding_queue_overflow_count"] = int(weights.loc[reasons == "forwarding_queue_overflow"].sum())

    return row
