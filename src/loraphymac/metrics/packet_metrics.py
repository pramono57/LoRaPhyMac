"""Read-only packet result tables for the :class:`Statistics` facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


PACKET_FATE_COLUMNS = [
    "run_id",
    "packet_uid",
    "src_id",
    "fcnt",
    "birth_time",
    "offered_time",
    "final_status",
    "terminal_reason",
    "final_time",
    "observation_end_time",
    "age_at_horizon_s",
    "latest_state",
    "latest_state_time",
    "latest_entity_id",
    "latest_hop_index",
    "latest_next_hop",
    "delivered_time",
    "e2e_delay_s",
    "e2e_delay_ms",
]

PACKET_FATE_STATUSES = (
    "DELIVERED",
    "TERMINAL_DROP",
    "CENSORED_WAITING",
    "CENSORED_IN_SERVICE",
    "CENSORED_RX",
    "CENSORED_IN_FLIGHT",
)


def build_packet_fate_df(
    packet_fates: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Return one deterministically ordered row per offered uplink packet."""
    if not packet_fates:
        return pd.DataFrame(columns=PACKET_FATE_COLUMNS)

    return (
        pd.DataFrame(list(packet_fates.values()))
        .reindex(columns=PACKET_FATE_COLUMNS)
        .sort_values(["src_id", "fcnt", "packet_uid"], kind="stable")
        .reset_index(drop=True)
    )


def build_packet_fate_summary_df(
    fate: pd.DataFrame,
    run_metadata: Mapping[str, Any],
    terminal_conflict_count: int,
) -> pd.DataFrame:
    """Summarize terminal and finite-horizon packet outcomes."""
    counts = {status: 0 for status in PACKET_FATE_STATUSES}
    if not fate.empty:
        counts.update(fate["final_status"].value_counts().to_dict())

    offered = int(len(fate))
    delivered = int(counts["DELIVERED"])
    terminal_drop = int(counts["TERMINAL_DROP"])
    censored = int(
        sum(counts[status] for status in PACKET_FATE_STATUSES if status.startswith("CENSORED_"))
    )
    delay = (
        pd.to_numeric(
            fate.loc[fate["final_status"] == "DELIVERED", "e2e_delay_ms"],
            errors="coerce",
        ).dropna()
        if not fate.empty
        else pd.Series(dtype="float64")
    )

    def percent(value: int) -> float:
        return 100.0 * value / offered if offered > 0 else 0.0

    return pd.DataFrame(
        [
            {
                "run_id": run_metadata.get("run_id"),
                "packet_fate_metrics_version": "1.0",
                "observation_end_time": run_metadata.get("sim_time"),
                "offered_count": offered,
                "delivered_count": delivered,
                "terminal_drop_count": terminal_drop,
                "censored_count": censored,
                "censored_waiting_count": int(counts["CENSORED_WAITING"]),
                "censored_in_service_count": int(counts["CENSORED_IN_SERVICE"]),
                "censored_rx_count": int(counts["CENSORED_RX"]),
                "censored_in_flight_count": int(counts["CENSORED_IN_FLIGHT"]),
                "delivery_by_horizon_percent": percent(delivered),
                "terminal_loss_percent": percent(terminal_drop),
                "censored_fraction_percent": percent(censored),
                "delivered_delay_mean_ms": (float(delay.mean()) if len(delay) else 0.0),
                "delivered_delay_p50_ms": (float(delay.quantile(0.50)) if len(delay) else 0.0),
                "delivered_delay_p95_ms": (float(delay.quantile(0.95)) if len(delay) else 0.0),
                "delivered_delay_p99_ms": (float(delay.quantile(0.99)) if len(delay) else 0.0),
                "terminal_conflict_count": int(terminal_conflict_count),
                "packet_fate_conservation_delta": (offered - delivered - terminal_drop - censored),
            }
        ]
    )


def build_delivered_packet_df(
    delivered_packets: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Return the packet-delivery event matrix."""
    return pd.DataFrame(delivered_packets)


def build_packet_failure_df(event_log: pd.DataFrame) -> pd.DataFrame:
    """Return DROP events with their original event-log schema."""
    if event_log.empty:
        return pd.DataFrame()
    return event_log[event_log["event"] == "DROP"].copy().reset_index(drop=True)
