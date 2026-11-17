from __future__ import annotations

import pandas as pd
import pytest

from loraphymac.metrics.core_summary import build_core_summary_df
from loraphymac.metrics.queue_metrics import (
    build_tx_queue_lifecycle_df,
    queue_mean_ms,
    queue_percentile_ms,
)


def _fate_summary(**overrides):
    values = {
        "packet_fate_metrics_version": "1.0",
        "offered_count": 20,
        "delivered_count": 19,
        "terminal_drop_count": 1,
        "censored_count": 0,
        "censored_waiting_count": 0,
        "censored_in_service_count": 0,
        "censored_rx_count": 0,
        "censored_in_flight_count": 0,
        "delivery_by_horizon_percent": 95.0,
        "terminal_loss_percent": 5.0,
        "censored_fraction_percent": 0.0,
        "delivered_delay_p50_ms": 400.0,
        "delivered_delay_p95_ms": 800.0,
        "delivered_delay_p99_ms": 900.0,
        "terminal_conflict_count": 0,
        "packet_fate_conservation_delta": 0,
    }
    values.update(overrides)
    return pd.DataFrame([values])


class CoreSummaryInputs:
    def __init__(self):
        self.run_metadata = {"run_id": "core-8h", "seed": 42, "sim_time": 100.0}
        self.node_stats = {
            "N1": {"e2e_delay_sum_s": 1.0, "e2e_delay_count": 2},
            "N2": {"e2e_delay_sum_s": 0.5, "e2e_delay_count": 1},
        }
        self.node_rx = pd.DataFrame(
            [
                {"node": "N1", "tx": 10, "gateway": 10, "PDR %": 100.0},
                {"node": "N2", "tx": 10, "gateway": 9, "PDR %": 90.0},
            ]
        )
        self.fate = _fate_summary()
        self.collisions = pd.DataFrame(
            [
                {
                    "resolution_id": "c1",
                    "outcome": "CAPTURE_SURVIVOR",
                    "direct_interferer_count": 1,
                    "survived": True,
                },
                {
                    "resolution_id": "c1",
                    "outcome": "COLLISION_DROP",
                    "direct_interferer_count": 1,
                    "survived": False,
                },
                {
                    "resolution_id": "c2",
                    "outcome": "SINGLETON_SURVIVOR",
                    "direct_interferer_count": 0,
                    "survived": True,
                },
            ]
        )
        self.collision_edges = pd.DataFrame(
            [
                {
                    "desired_sf": 7,
                    "interferer_sf": 8,
                    "spectral_overlap_fraction": 0.5,
                    "pair_pass": False,
                    "dominant_interferer": True,
                },
                {
                    "desired_sf": 7,
                    "interferer_sf": 7,
                    "spectral_overlap_fraction": 1.0,
                    "pair_pass": True,
                    "dominant_interferer": False,
                },
            ]
        )
        self.energy = pd.DataFrame(
            [
                {"entity_type": "node", "total_energy_J": 4.0},
                {"entity_type": "relay", "total_energy_J": 3.0},
                {"entity_type": "gateway", "total_energy_J": 2.0},
            ]
        )
        self.ptx = pd.DataFrame(
            [
                {
                    "entity_type": "node",
                    "final_ptx": 10.0,
                    "last_decision_margin": 6.0,
                    "last_apc_rx_margin": 4.0,
                },
                {
                    "entity_type": "node",
                    "final_ptx": 12.0,
                    "last_decision_margin": 8.0,
                    "last_apc_rx_margin": 6.0,
                },
                {
                    "entity_type": "relay",
                    "final_ptx": 14.0,
                    "last_decision_margin": 9.0,
                    "last_apc_rx_margin": 7.0,
                },
            ]
        )
        self.queue_lifecycle = build_tx_queue_lifecycle_df(
            [
                {
                    "queue_visit_id": "q1",
                    "entity_id": "R1",
                    "entity_type": "relay",
                    "status": "COMPLETED",
                    "queue_dequeue_time": 0.1,
                    "tx_start": 0.2,
                    "fifo_wait_s": 0.1,
                    "time_to_tx_start_s": 0.2,
                    "system_sojourn_s": 0.5,
                },
                {
                    "queue_visit_id": "q2",
                    "entity_id": "R1",
                    "entity_type": "relay",
                    "status": "COMPLETED",
                    "queue_dequeue_time": 0.3,
                    "tx_start": 0.4,
                    "fifo_wait_s": 0.3,
                    "time_to_tx_start_s": 0.4,
                    "system_sojourn_s": 0.7,
                },
                {
                    "queue_visit_id": "q3",
                    "entity_id": "GW",
                    "entity_type": "gateway",
                    "status": "COMPLETED",
                    "queue_dequeue_time": 0.5,
                    "tx_start": 0.6,
                    "fifo_wait_s": 10.0,
                    "time_to_tx_start_s": 10.0,
                    "system_sojourn_s": 10.0,
                },
            ]
        )
        self.queue_metrics = pd.DataFrame(
            [
                {
                    "entity_type": "relay",
                    "waiting_end_count": 2,
                    "worker_occupied_fraction": 0.5,
                },
                {
                    "entity_type": "relay",
                    "waiting_end_count": 5,
                    "worker_occupied_fraction": 0.7,
                },
                {
                    "entity_type": "gateway",
                    "waiting_end_count": 99,
                    "worker_occupied_fraction": 0.99,
                },
            ]
        )

    def get_node_rx_matrix_df(self):
        return self.node_rx

    def get_packet_fate_summary_df(self):
        return self.fate

    def get_collision_event_log_df(self):
        return self.collisions

    def get_collision_interference_edges_df(self):
        return self.collision_edges

    def get_energy_breakdown_df(self, nodes):
        return self.energy

    def get_final_ptx_distribution_df(self, nodes):
        return self.ptx

    def get_tx_queue_lifecycle_df(self):
        return self.queue_lifecycle

    @staticmethod
    def _queue_mean_ms(frame, column):
        return queue_mean_ms(frame, column)

    @staticmethod
    def _queue_percentile_ms(frame, column, quantile):
        return queue_percentile_ms(frame, column, quantile)

    def get_relay_tx_queue_metrics_df(self, nodes, sim_time_s=None):
        return self.queue_metrics


def test_core_summary_preserves_cross_metric_formulas_and_relay_only_queue_scope():
    inputs = CoreSummaryInputs()

    result = build_core_summary_df(inputs, nodes=[object()]).iloc[0]

    assert result["run_id"] == "core-8h"
    assert result["seed"] == 42
    assert result["tx_packets"] == 20
    assert result["rx_packets_gw"] == 19
    assert result["pdr_ratio"] == 0.95
    assert result["pdr_percent"] == 95.0
    assert result["served_node_ratio_95_percent"] == 50.0
    assert result["avg_e2e_delay_ms"] == 500.0

    assert result["packet_fate_offered_count"] == 20
    assert result["packet_fate_delivered_count"] == 19
    assert result["packet_fate_conservation_delta"] == 0

    assert result["collision_decision_count"] == 3
    assert result["collision_resolution_count"] == 2
    assert result["collision_opportunity_count"] == 2
    assert result["collision_opportunity_resolution_count"] == 1
    assert result["collision_singleton_success_count"] == 1
    assert result["collision_capture_success_count"] == 1
    assert result["collision_contended_success_count"] == 1
    assert result["collision_drop_decision_count"] == 1
    assert result["collision_unique_overlap_pair_count"] == 1.0
    assert result["collision_capture_success_rate_percent"] == 50.0
    assert result["collision_loss_rate_percent"] == 50.0
    assert result["collision_outcome_conservation_delta"] == 0
    assert result["collision_interference_edge_count"] == 2
    assert result["collision_cross_sf_edge_count"] == 1
    assert result["collision_partial_spectral_edge_count"] == 1
    assert result["collision_destructive_edge_count"] == 1
    assert result["collision_dominant_edge_count"] == 1

    assert result["total_node_energy_J"] == 4.0
    assert result["total_relay_energy_J"] == 3.0
    assert result["total_gateway_energy_J"] == 2.0
    assert result["total_network_energy_J"] == 9.0
    assert result["total_constrained_energy_J"] == 7.0
    assert result["energy_per_delivered_packet_J"] == pytest.approx(round(9.0 / 19.0, 6))
    assert result["energy_per_delivered_packet_constrained_J"] == pytest.approx(
        round(7.0 / 19.0, 6)
    )

    assert result["avg_final_node_ptx"] == 11.0
    assert result["avg_final_node_decision_margin"] == 7.0
    assert result["avg_final_node_apc_rx_margin"] == 5.0
    assert result["avg_final_relay_ptx"] == 14.0
    assert result["avg_final_relay_decision_margin"] == 9.0
    assert result["avg_final_relay_apc_rx_margin"] == 7.0

    assert result["avg_relay_waiting_time_wq_ms"] == 200.0
    assert result["relay_fifo_wait_mean_ms_packet_weighted"] == 200.0
    assert result["relay_fifo_wait_p95_ms"] == 290.0
    assert result["relay_time_to_tx_start_mean_ms"] == 300.0
    assert result["relay_system_sojourn_mean_ms"] == 600.0
    assert result["relay_max_waiting_end_count"] == 5
    assert result["relay_max_worker_occupied_fraction"] == 0.7
    assert result["queue_metrics_version"] == "2.0"


def test_core_summary_empty_ptx_and_metric_tables_return_zero_fallbacks():
    inputs = CoreSummaryInputs()
    inputs.node_rx = pd.DataFrame()
    inputs.node_stats = {}
    inputs.fate = _fate_summary(
        offered_count=0,
        delivered_count=0,
        terminal_drop_count=0,
        delivery_by_horizon_percent=0.0,
    )
    inputs.collisions = pd.DataFrame()
    inputs.collision_edges = pd.DataFrame()
    inputs.energy = pd.DataFrame()
    inputs.ptx = pd.DataFrame()
    inputs.queue_lifecycle = build_tx_queue_lifecycle_df([])
    inputs.queue_metrics = pd.DataFrame()

    result = build_core_summary_df(inputs, nodes=[]).iloc[0]

    assert result["tx_packets"] == 0
    assert result["rx_packets_gw"] == 0
    assert result["pdr_percent"] == 0.0
    assert result["total_network_energy_J"] == 0.0
    assert result["energy_per_delivered_packet_J"] == 0.0
    assert result["avg_final_node_ptx"] == 0.0
    assert result["avg_final_node_apc_rx_margin"] == 0.0
    assert result["avg_final_relay_ptx"] == 0.0
    assert result["avg_final_relay_apc_rx_margin"] == 0.0
    assert result["collision_decision_count"] == 0
    assert result["collision_interference_edge_count"] == 0
    assert result["relay_max_waiting_end_count"] == 0
