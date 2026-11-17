"""Read-only receiver result tables for the :class:`Statistics` facade."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def build_relay_rx_stats_df(
    relay_stats: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Build the relay/gateway receive and forwarding counter matrix."""
    rows = []
    for relay_id, counters in relay_stats.items():
        rows.append(
            {
                "relay": relay_id,
                "rx_arrival_count": counters.get("rx_arrival_count", 0),
                "feasible": counters.get("rx_feasible", 0),
                "rx_not_feasible": counters.get("rx_not_feasible", 0),
                "survivor": counters.get("rx_survivor", 0),
                "collision": counters.get("rx_collision", 0),
                "decoded": counters.get("rx_decoded", 0),
                "demodulator_drop_count": counters.get(
                    "demodulator_drop_count",
                    0,
                ),
                "half_duplex_drop_count": counters.get(
                    "half_duplex_drop_count",
                    0,
                ),
                "rx_interrupted_by_tx_count": counters.get(
                    "rx_interrupted_by_tx_count",
                    0,
                ),
                "forwarded": counters.get("rx_forwarded", 0),
                "gateway": counters.get("rx_gateway", 0),
                "tx_queue_arrival_count": counters.get(
                    "tx_queue_arrival_count",
                    0,
                ),
                "tx_queue_enqueue_attempt_count": counters.get(
                    "tx_queue_enqueue_attempt_count",
                    0,
                ),
                "tx_queue_accepted_count": counters.get(
                    "tx_queue_accepted_count",
                    0,
                ),
                "tx_queue_overflow_drop_count": counters.get(
                    "tx_queue_overflow_drop_count",
                    0,
                ),
                "tx_queue_internal_drop_count": counters.get(
                    "tx_queue_internal_drop_count",
                    0,
                ),
                "tx_done": counters.get("tx_done", 0),
            }
        )

    return pd.DataFrame(rows)


def build_rx_channel_usage_df(
    rx_channel_use: Mapping[str, Mapping[Any, int]],
    rx_channel_duration: Mapping[str, Mapping[Any, float]],
) -> pd.DataFrame:
    """Build packet-count and receive-airtime metrics by receiver/channel."""
    rows = []
    for relay_id, channel_counts in rx_channel_use.items():
        for channel, packet_count in channel_counts.items():
            rows.append(
                {
                    "relay": relay_id,
                    "channel": channel,
                    "packet_count": packet_count,
                    "total_duration_s": rx_channel_duration[relay_id][channel],
                }
            )
    return pd.DataFrame(rows)


def build_node_rx_matrix_df(
    node_stats: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Build end-device TX/RX, PDR, hop, and delay metrics."""
    rows = []

    for node_id, counters in node_stats.items():
        if not str(node_id).startswith("N"):
            continue

        tx_count = counters.get("tx", 0)
        gateway_count = counters.get("rx_gateway", 0)
        pdr_percent = gateway_count / tx_count * 100 if tx_count > 0 else 0.0

        delay_count = counters.get("e2e_delay_count", 0)
        average_delay_s = (
            counters.get("e2e_delay_sum_s", 0.0) / delay_count if delay_count > 0 else 0.0
        )
        hop_index = counters.get("hop_index")

        rows.append(
            {
                "node": node_id,
                "relay_count": int(hop_index) if hop_index is not None else pd.NA,
                "tx": tx_count,
                "network_rx_events_by_source": counters.get(
                    "rx_arrival_count",
                    0,
                ),
                "feasible": counters.get("rx_feasible", 0),
                "not_feasible": counters.get("rx_not_feasible", 0),
                "survivor": counters.get("rx_survivor", 0),
                "collision": counters.get("rx_collision", 0),
                "demod_drop": counters.get("demodulator_drop_count", 0),
                "half_duplex_drop": counters.get("half_duplex_drop_count", 0),
                "rx_interrupted_by_tx": counters.get(
                    "rx_interrupted_by_tx_count",
                    0,
                ),
                "decoded": counters.get("rx_decoded", 0),
                "forwarded": counters.get("rx_forwarded", 0),
                "gateway": gateway_count,
                "PDR %": round(pdr_percent, 2),
                "avg_e2e_delay_ms": round(average_delay_s * 1000, 3),
            }
        )

    return pd.DataFrame(rows)


def build_hop_summary_df(node_rx_matrix: pd.DataFrame) -> pd.DataFrame:
    """Aggregate end-device metrics by number of relays traversed."""
    columns = [
        "relay_count",
        "node_count",
        "tx_sum",
        "gateway_sum",
        "avg_pdr_percent",
        "avg_e2e_delay_ms",
        "min_e2e_delay_ms",
        "max_e2e_delay_ms",
    ]

    if node_rx_matrix.empty:
        return pd.DataFrame(columns=columns)

    frame = node_rx_matrix[node_rx_matrix["relay_count"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        frame.groupby("relay_count", dropna=True)
        .agg(
            node_count=("node", "count"),
            tx_sum=("tx", "sum"),
            gateway_sum=("gateway", "sum"),
            avg_pdr_percent=("PDR %", "mean"),
            avg_e2e_delay_ms=("avg_e2e_delay_ms", "mean"),
            min_e2e_delay_ms=("avg_e2e_delay_ms", "min"),
            max_e2e_delay_ms=("avg_e2e_delay_ms", "max"),
        )
        .reset_index()
    )

    summary["relay_count"] = summary["relay_count"].astype("Int64")
    summary["avg_pdr_percent"] = summary["avg_pdr_percent"].round(2)
    summary["avg_e2e_delay_ms"] = summary["avg_e2e_delay_ms"].round(3)
    summary["min_e2e_delay_ms"] = summary["min_e2e_delay_ms"].round(3)
    summary["max_e2e_delay_ms"] = summary["max_e2e_delay_ms"].round(3)

    return summary
