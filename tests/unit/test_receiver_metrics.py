from __future__ import annotations

import pandas as pd

from loraphymac.metrics.receiver_metrics import (
    build_hop_summary_df,
    build_node_rx_matrix_df,
    build_relay_rx_stats_df,
    build_rx_channel_usage_df,
)


def test_relay_rx_table_preserves_receive_forwarding_and_queue_counters():
    result = build_relay_rx_stats_df(
        {
            "R1": {
                "rx_arrival_count": 10,
                "rx_feasible": 9,
                "rx_not_feasible": 1,
                "rx_survivor": 7,
                "rx_collision": 2,
                "rx_decoded": 6,
                "demodulator_drop_count": 1,
                "half_duplex_drop_count": 2,
                "rx_interrupted_by_tx_count": 1,
                "rx_forwarded": 5,
                "rx_gateway": 0,
                "tx_queue_arrival_count": 5,
                "tx_queue_enqueue_attempt_count": 5,
                "tx_queue_accepted_count": 4,
                "tx_queue_overflow_drop_count": 1,
                "tx_queue_internal_drop_count": 0,
                "tx_done": 4,
            }
        }
    ).iloc[0]

    assert result.to_dict() == {
        "relay": "R1",
        "rx_arrival_count": 10,
        "feasible": 9,
        "rx_not_feasible": 1,
        "survivor": 7,
        "collision": 2,
        "decoded": 6,
        "demodulator_drop_count": 1,
        "half_duplex_drop_count": 2,
        "rx_interrupted_by_tx_count": 1,
        "forwarded": 5,
        "gateway": 0,
        "tx_queue_arrival_count": 5,
        "tx_queue_enqueue_attempt_count": 5,
        "tx_queue_accepted_count": 4,
        "tx_queue_overflow_drop_count": 1,
        "tx_queue_internal_drop_count": 0,
        "tx_done": 4,
    }


def test_rx_channel_usage_preserves_receiver_channel_order_and_airtime():
    result = build_rx_channel_usage_df(
        {"R1": {2: 3, 0: 1}, "GW": {1: 4}},
        {"R1": {2: 0.75, 0: 0.10}, "GW": {1: 1.25}},
    )

    assert result.to_dict("records") == [
        {"relay": "R1", "channel": 2, "packet_count": 3, "total_duration_s": 0.75},
        {"relay": "R1", "channel": 0, "packet_count": 1, "total_duration_s": 0.10},
        {"relay": "GW", "channel": 1, "packet_count": 4, "total_duration_s": 1.25},
    ]


def test_node_rx_matrix_keeps_only_nodes_and_preserves_pdr_delay_semantics():
    result = build_node_rx_matrix_df(
        {
            "R1": {"tx": 99, "rx_gateway": 99},
            "N2": {
                "hop_index": 1,
                "tx": 3,
                "rx_arrival_count": 4,
                "rx_feasible": 3,
                "rx_not_feasible": 1,
                "rx_survivor": 2,
                "rx_collision": 1,
                "demodulator_drop_count": 0,
                "half_duplex_drop_count": 1,
                "rx_interrupted_by_tx_count": 1,
                "rx_decoded": 2,
                "rx_forwarded": 2,
                "rx_gateway": 2,
                "e2e_delay_sum_s": 0.5,
                "e2e_delay_count": 2,
            },
            "N1": {"hop_index": None, "tx": 0, "rx_gateway": 0},
        }
    )

    assert result["node"].tolist() == ["N2", "N1"]
    node = result.iloc[0]
    assert node["relay_count"] == 1
    assert node["network_rx_events_by_source"] == 4
    assert node["PDR %"] == 66.67
    assert node["avg_e2e_delay_ms"] == 250.0
    assert pd.isna(result.iloc[1]["relay_count"])
    assert result.iloc[1]["PDR %"] == 0.0


def test_hop_summary_preserves_group_totals_rounding_and_nullable_hops():
    node_matrix = pd.DataFrame(
        [
            {
                "node": "N1",
                "relay_count": 1,
                "tx": 2,
                "gateway": 1,
                "PDR %": 50.0,
                "avg_e2e_delay_ms": 100.0,
            },
            {
                "node": "N2",
                "relay_count": 1,
                "tx": 3,
                "gateway": 2,
                "PDR %": 66.67,
                "avg_e2e_delay_ms": 200.0,
            },
            {
                "node": "N3",
                "relay_count": 2,
                "tx": 4,
                "gateway": 3,
                "PDR %": 75.0,
                "avg_e2e_delay_ms": 300.0,
            },
            {
                "node": "N4",
                "relay_count": pd.NA,
                "tx": 1,
                "gateway": 0,
                "PDR %": 0.0,
                "avg_e2e_delay_ms": 0.0,
            },
        ]
    )

    result = build_hop_summary_df(node_matrix)

    assert str(result["relay_count"].dtype) == "Int64"
    assert result.to_dict("records") == [
        {
            "relay_count": 1,
            "node_count": 2,
            "tx_sum": 5,
            "gateway_sum": 3,
            "avg_pdr_percent": 58.34,
            "avg_e2e_delay_ms": 150.0,
            "min_e2e_delay_ms": 100.0,
            "max_e2e_delay_ms": 200.0,
        },
        {
            "relay_count": 2,
            "node_count": 1,
            "tx_sum": 4,
            "gateway_sum": 3,
            "avg_pdr_percent": 75.0,
            "avg_e2e_delay_ms": 300.0,
            "min_e2e_delay_ms": 300.0,
            "max_e2e_delay_ms": 300.0,
        },
    ]


def test_empty_hop_summary_keeps_stable_schema():
    result = build_hop_summary_df(pd.DataFrame())

    assert result.empty
    assert list(result.columns) == [
        "relay_count",
        "node_count",
        "tx_sum",
        "gateway_sum",
        "avg_pdr_percent",
        "avg_e2e_delay_ms",
        "min_e2e_delay_ms",
        "max_e2e_delay_ms",
    ]
