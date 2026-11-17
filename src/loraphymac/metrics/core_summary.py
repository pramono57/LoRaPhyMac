"""Cross-metric core summary assembly for the :class:`Statistics` facade."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_core_summary_df(statistics: Any, nodes: list) -> pd.DataFrame:
    """
    Returns a single-row DataFrame summarizing one run.
    """
    # ------------------------
    # Node-level summary
    # ------------------------
    node_df = statistics.get_node_rx_matrix_df()

    if node_df.empty:
        tx_packets = 0
        rx_packets_gw = 0
        served_node_ratio_95_percent = 0.0
        avg_e2e_delay_ms = 0.0
    else:
        tx_packets = int(node_df["tx"].sum())
        rx_packets_gw = int(node_df["gateway"].sum())

        source_nodes = node_df[node_df["tx"] > 0]
        served_nodes_95 = source_nodes[source_nodes["PDR %"] >= 95.0]

        served_node_ratio_95_percent = (
            len(served_nodes_95) / len(source_nodes) * 100.0 if len(source_nodes) > 0 else 0.0
        )

        delay_sum_s = sum(v.get("e2e_delay_sum_s", 0.0) for v in statistics.node_stats.values())
        delay_count = sum(v.get("e2e_delay_count", 0) for v in statistics.node_stats.values())

        avg_e2e_delay_ms = delay_sum_s / delay_count * 1000.0 if delay_count > 0 else 0.0

    pdr_ratio = (rx_packets_gw / tx_packets) if tx_packets > 0 else 0.0
    pdr_percent = pdr_ratio * 100.0

    fate_summary_df = statistics.get_packet_fate_summary_df()
    fate = fate_summary_df.iloc[0].to_dict()

    # ------------------------
    # Collision/capture lifecycle summary
    # ------------------------
    collision_df = statistics.get_collision_event_log_df()
    if collision_df.empty:
        collision_decision_count = 0
        collision_resolution_count = 0
        collision_opportunity_count = 0
        collision_opportunity_resolution_count = 0
        collision_singleton_success_count = 0
        collision_capture_success_count = 0
        collision_sinr_success_count = 0
        collision_contended_success_count = 0
        collision_drop_decision_count = 0
        collision_unique_overlap_pair_count = 0.0
        collision_loss_rate_percent = 0.0
        collision_capture_success_rate_percent = 0.0
        collision_outcome_conservation_delta = 0
    else:
        outcome = collision_df["outcome"].astype(str)
        direct = pd.to_numeric(collision_df["direct_interferer_count"], errors="coerce").fillna(0.0)
        contended_mask = direct > 0
        survived = collision_df["survived"].fillna(False).astype(bool)
        collision_decision_count = int(len(collision_df))
        collision_resolution_count = int(collision_df["resolution_id"].nunique())
        collision_opportunity_count = int(contended_mask.sum())
        collision_opportunity_resolution_count = int(
            collision_df.loc[contended_mask, "resolution_id"].nunique()
        )
        collision_singleton_success_count = int((outcome == "SINGLETON_SURVIVOR").sum())
        collision_capture_success_count = int((outcome == "CAPTURE_SURVIVOR").sum())
        collision_sinr_success_count = int((outcome == "SINR_SURVIVOR").sum())
        collision_contended_success_count = int((survived & contended_mask).sum())
        collision_drop_decision_count = int((outcome == "COLLISION_DROP").sum())
        collision_unique_overlap_pair_count = float(direct.sum()) / 2.0
        collision_loss_rate_percent = (
            100.0 * collision_drop_decision_count / collision_opportunity_count
            if collision_opportunity_count > 0
            else 0.0
        )
        capture_denominator = collision_capture_success_count + collision_drop_decision_count
        collision_capture_success_rate_percent = (
            100.0 * collision_capture_success_count / capture_denominator
            if capture_denominator > 0
            else 0.0
        )
        collision_outcome_conservation_delta = int(
            collision_decision_count
            - collision_singleton_success_count
            - collision_capture_success_count
            - collision_sinr_success_count
            - collision_drop_decision_count
        )

    collision_edge_df = statistics.get_collision_interference_edges_df()
    if collision_edge_df.empty:
        collision_interference_edge_count = 0
        collision_cross_sf_edge_count = 0
        collision_partial_spectral_edge_count = 0
        collision_destructive_edge_count = 0
        collision_dominant_edge_count = 0
    else:
        desired_sf = pd.to_numeric(collision_edge_df["desired_sf"], errors="coerce")
        interferer_sf = pd.to_numeric(collision_edge_df["interferer_sf"], errors="coerce")
        overlap_fraction = pd.to_numeric(
            collision_edge_df["spectral_overlap_fraction"],
            errors="coerce",
        )
        pair_pass = collision_edge_df["pair_pass"].fillna(False).astype(bool)
        dominant = collision_edge_df["dominant_interferer"].fillna(False).astype(bool)
        collision_interference_edge_count = int(len(collision_edge_df))
        collision_cross_sf_edge_count = int((desired_sf != interferer_sf).sum())
        collision_partial_spectral_edge_count = int(
            ((overlap_fraction > 0.0) & (overlap_fraction < 1.0)).sum()
        )
        collision_destructive_edge_count = int((~pair_pass).sum())
        collision_dominant_edge_count = int(dominant.sum())

    # ------------------------
    # Energy summary already in Joule
    # ------------------------
    energy_breakdown_df = statistics.get_energy_breakdown_df(nodes)
    if energy_breakdown_df.empty:
        total_node_energy_J = 0.0
        total_relay_energy_J = 0.0
        total_gateway_energy_J = 0.0
    else:
        total_node_energy_J = energy_breakdown_df[energy_breakdown_df["entity_type"] == "node"][
            "total_energy_J"
        ].sum()
        total_relay_energy_J = energy_breakdown_df[energy_breakdown_df["entity_type"] == "relay"][
            "total_energy_J"
        ].sum()
        total_gateway_energy_J = energy_breakdown_df[
            energy_breakdown_df["entity_type"] == "gateway"
        ]["total_energy_J"].sum()

    total_network_energy_J = total_node_energy_J + total_relay_energy_J + total_gateway_energy_J
    energy_per_delivered_packet_J = (
        total_network_energy_J / rx_packets_gw if rx_packets_gw > 0 else 0.0
    )

    total_constrained_energy_J = total_node_energy_J + total_relay_energy_J
    energy_per_delivered_packet_constrained_J = (
        total_constrained_energy_J / rx_packets_gw if rx_packets_gw > 0 else 0.0
    )

    # ------------------------
    # Final PTX distribution
    # ------------------------
    ptx_df = statistics.get_final_ptx_distribution_df(nodes)
    if ptx_df.empty:
        avg_final_node_ptx = 0.0
        avg_final_relay_ptx = 0.0
        avg_final_node_margin = 0.0
        avg_final_relay_margin = 0.0
        node_apc_rx_margin = 0.0
        relay_apc_rx_margin = 0.0
    else:
        node_ptx_df = ptx_df[ptx_df["entity_type"] == "node"]
        relay_ptx_df = ptx_df[ptx_df["entity_type"] == "relay"]

        avg_final_node_ptx = (
            node_ptx_df["final_ptx"].dropna().mean() if not node_ptx_df.empty else 0.0
        )
        avg_final_relay_ptx = (
            relay_ptx_df["final_ptx"].dropna().mean() if not relay_ptx_df.empty else 0.0
        )

        node_margin_col = (
            "last_decision_margin"
            if "last_decision_margin" in node_ptx_df.columns
            else "last_margin"
        )

        relay_margin_col = (
            "last_decision_margin"
            if "last_decision_margin" in relay_ptx_df.columns
            else "last_margin"
        )

        avg_final_node_margin = (
            node_ptx_df[node_margin_col].dropna().mean()
            if not node_ptx_df.empty and node_margin_col in node_ptx_df.columns
            else 0.0
        )

        avg_final_relay_margin = (
            relay_ptx_df[relay_margin_col].dropna().mean()
            if not relay_ptx_df.empty and relay_margin_col in relay_ptx_df.columns
            else 0.0
        )
        node_apc_rx_margin = (
            node_ptx_df["last_apc_rx_margin"].dropna().mean()
            if not node_ptx_df.empty and "last_apc_rx_margin" in node_ptx_df.columns
            else 0.0
        )

        relay_apc_rx_margin = (
            relay_ptx_df["last_apc_rx_margin"].dropna().mean()
            if not relay_ptx_df.empty and "last_apc_rx_margin" in relay_ptx_df.columns
            else 0.0
        )

    # ------------------------
    # Waiting Time (wq) summary
    # ------------------------
    queue_lifecycle_df = statistics.get_tx_queue_lifecycle_df()
    relay_queue_visits = queue_lifecycle_df[queue_lifecycle_df["entity_type"] == "relay"]
    relay_fifo_observed = relay_queue_visits[relay_queue_visits["queue_dequeue_time"].notna()]
    relay_tx_started = relay_queue_visits[relay_queue_visits["tx_start"].notna()]
    relay_completed_queue = relay_queue_visits[relay_queue_visits["status"] == "COMPLETED"]

    relay_fifo_wait_mean_ms = statistics._queue_mean_ms(
        relay_fifo_observed,
        "fifo_wait_s",
    )
    relay_fifo_wait_p95_ms = statistics._queue_percentile_ms(
        relay_fifo_observed,
        "fifo_wait_s",
        0.95,
    )
    relay_time_to_tx_start_mean_ms = statistics._queue_mean_ms(
        relay_tx_started,
        "time_to_tx_start_s",
    )
    relay_system_sojourn_mean_ms = statistics._queue_mean_ms(
        relay_completed_queue,
        "system_sojourn_s",
    )

    queue_metrics_df = statistics.get_relay_tx_queue_metrics_df(
        nodes,
        sim_time_s=statistics.run_metadata.get("sim_time", None),
    )
    relay_queue_metrics = (
        queue_metrics_df[queue_metrics_df["entity_type"] == "relay"]
        if not queue_metrics_df.empty
        else pd.DataFrame()
    )
    relay_max_waiting_end_count = (
        int(relay_queue_metrics["waiting_end_count"].max()) if not relay_queue_metrics.empty else 0
    )
    relay_max_worker_occupied_fraction = (
        float(relay_queue_metrics["worker_occupied_fraction"].max())
        if not relay_queue_metrics.empty
        else 0.0
    )

    # Backward-compatible core alias now uses the canonical packet-weighted
    # FIFO wait. Entity-mean aggregation (including idle gateways) is
    # intentionally retired in queue_metrics_version=2.0.
    avg_relay_waiting_time_wq_ms = relay_fifo_wait_mean_ms

    # ------------------------
    # Compose one row
    # ------------------------
    row = dict(statistics.run_metadata) if statistics.run_metadata else {}

    row.update(
        {
            "tx_packets": tx_packets,
            "rx_packets_gw": rx_packets_gw,
            "pdr_ratio": round(pdr_ratio, 6),
            "pdr_percent": round(pdr_percent, 3),
            "pdr_definition": "delivery_by_hard_horizon",
            "packet_fate_metrics_version": fate["packet_fate_metrics_version"],
            "packet_fate_offered_count": int(fate["offered_count"]),
            "packet_fate_delivered_count": int(fate["delivered_count"]),
            "packet_fate_terminal_drop_count": int(fate["terminal_drop_count"]),
            "packet_fate_censored_count": int(fate["censored_count"]),
            "packet_fate_censored_waiting_count": int(fate["censored_waiting_count"]),
            "packet_fate_censored_in_service_count": int(fate["censored_in_service_count"]),
            "packet_fate_censored_rx_count": int(fate["censored_rx_count"]),
            "packet_fate_censored_in_flight_count": int(fate["censored_in_flight_count"]),
            "delivery_by_horizon_percent": round(float(fate["delivery_by_horizon_percent"]), 6),
            "terminal_loss_percent": round(float(fate["terminal_loss_percent"]), 6),
            "censored_fraction_percent": round(float(fate["censored_fraction_percent"]), 6),
            "delivered_delay_p50_ms": round(float(fate["delivered_delay_p50_ms"]), 3),
            "delivered_delay_p95_ms": round(float(fate["delivered_delay_p95_ms"]), 3),
            "delivered_delay_p99_ms": round(float(fate["delivered_delay_p99_ms"]), 3),
            "packet_fate_terminal_conflict_count": int(fate["terminal_conflict_count"]),
            "packet_fate_conservation_delta": int(fate["packet_fate_conservation_delta"]),
            "collision_decision_count": collision_decision_count,
            "collision_resolution_count": collision_resolution_count,
            "collision_opportunity_count": collision_opportunity_count,
            "collision_opportunity_resolution_count": collision_opportunity_resolution_count,
            "collision_singleton_success_count": collision_singleton_success_count,
            "collision_capture_success_count": collision_capture_success_count,
            "collision_contended_success_count": collision_contended_success_count,
            "collision_drop_decision_count": collision_drop_decision_count,
            "collision_unique_overlap_pair_count": collision_unique_overlap_pair_count,
            "collision_interference_edge_count": collision_interference_edge_count,
            "collision_cross_sf_edge_count": collision_cross_sf_edge_count,
            "collision_partial_spectral_edge_count": collision_partial_spectral_edge_count,
            "collision_destructive_edge_count": collision_destructive_edge_count,
            "collision_dominant_edge_count": collision_dominant_edge_count,
            "collision_capture_success_rate_percent": round(
                collision_capture_success_rate_percent, 6
            ),
            "collision_loss_rate_percent": round(collision_loss_rate_percent, 6),
            "collision_outcome_conservation_delta": collision_outcome_conservation_delta,
            "served_node_ratio_95_percent": round(served_node_ratio_95_percent, 3),
            "avg_e2e_delay_ms": round(avg_e2e_delay_ms, 3),
            "avg_relay_waiting_time_wq_ms": round(avg_relay_waiting_time_wq_ms, 3),
            "queue_metrics_version": "2.0",
            "relay_fifo_wait_mean_ms_packet_weighted": round(relay_fifo_wait_mean_ms, 3),
            "relay_fifo_wait_p95_ms": round(relay_fifo_wait_p95_ms, 3),
            "relay_time_to_tx_start_mean_ms": round(relay_time_to_tx_start_mean_ms, 3),
            "relay_system_sojourn_mean_ms": round(relay_system_sojourn_mean_ms, 3),
            "relay_max_waiting_end_count": relay_max_waiting_end_count,
            "relay_max_worker_occupied_fraction": round(relay_max_worker_occupied_fraction, 6),
            "total_node_energy_J": round(float(total_node_energy_J), 6),
            "total_relay_energy_J": round(float(total_relay_energy_J), 6),
            "total_gateway_energy_J": round(float(total_gateway_energy_J), 6),
            "total_network_energy_J": round(float(total_network_energy_J), 6),
            "energy_per_delivered_packet_J": round(float(energy_per_delivered_packet_J), 6),
            "total_constrained_energy_J": round(float(total_constrained_energy_J), 6),
            "energy_per_delivered_packet_constrained_J": round(
                float(energy_per_delivered_packet_constrained_J), 6
            ),
            "avg_final_node_ptx": round(float(avg_final_node_ptx), 3)
            if pd.notna(avg_final_node_ptx)
            else 0.0,
            "avg_final_node_decision_margin": round(float(avg_final_node_margin), 3)
            if pd.notna(avg_final_node_margin)
            else 0.0,
            "avg_final_node_apc_rx_margin": round(float(node_apc_rx_margin), 3)
            if pd.notna(node_apc_rx_margin)
            else 0.0,
            "avg_final_relay_ptx": round(float(avg_final_relay_ptx), 3)
            if pd.notna(avg_final_relay_ptx)
            else 0.0,
            "avg_final_relay_decision_margin": round(float(avg_final_relay_margin), 3)
            if pd.notna(avg_final_relay_margin)
            else 0.0,
            "avg_final_relay_apc_rx_margin": round(float(relay_apc_rx_margin), 3)
            if pd.notna(relay_apc_rx_margin)
            else 0.0,
        }
    )

    return pd.DataFrame([row])
