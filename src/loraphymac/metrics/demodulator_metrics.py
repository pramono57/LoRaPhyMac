"""Read-only demodulator result tables for the :class:`Statistics` facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


def build_demodulator_event_log_df(
    events: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Return packet-level modem-pool transitions in deterministic order."""
    if not events:
        return pd.DataFrame()

    return (
        pd.DataFrame(events)
        .sort_values(["time", "receiver_id", "event"], kind="stable")
        .reset_index(drop=True)
    )


def build_demodulator_pool_summary_df(event_log: pd.DataFrame) -> pd.DataFrame:
    """Aggregate modem-pool occupancy and rejection evidence per receiver."""
    if event_log.empty:
        return pd.DataFrame()

    rows = []
    for receiver_id, sub in event_log.groupby("receiver_id", dropna=False):
        acquired = sub[sub["event"] == "ACQUIRE"]
        released = sub[sub["event"] == "RELEASE"]
        rejected = sub[sub["event"] == "REJECT"]

        def max_int(column: str) -> int:
            if column not in sub.columns or sub[column].dropna().empty:
                return 0
            return int(sub[column].dropna().max())

        def first_int(column: str) -> int:
            if column not in sub.columns or sub[column].dropna().empty:
                return 0
            return int(sub[column].dropna().iloc[0])

        rows.append(
            {
                "receiver_id": receiver_id,
                "timing_model": "rx_start_to_rx_end_v2",
                "capacity_a": first_int("capacity_a"),
                "capacity_b": first_int("capacity_b"),
                "capacity_total": first_int("capacity_total"),
                "acquire_attempt_count": int(len(acquired) + len(rejected)),
                "acquire_success_count": int(len(acquired)),
                "release_count": int(len(released)),
                "reject_count": int(len(rejected)),
                "demod_busy_count": int(
                    (rejected.get("reason", pd.Series(dtype=str)).astype(str) == "demod_busy").sum()
                ),
                "unsupported_sf_count": int(
                    (
                        rejected.get("reason", pd.Series(dtype=str)).astype(str)
                        == "demod_unsupported_sf"
                    ).sum()
                ),
                "duplicate_acquire_count": int(
                    (
                        rejected.get("reason", pd.Series(dtype=str)).astype(str)
                        == "demod_duplicate_acquire"
                    ).sum()
                ),
                "release_missing_count": int(
                    (sub.get("event", pd.Series(dtype=str)).astype(str) == "RELEASE_MISSING").sum()
                ),
                "pool_a_acquire_count": int(
                    (acquired.get("pool", pd.Series(dtype=str)).astype(str) == "A").sum()
                ),
                "pool_b_acquire_count": int(
                    (acquired.get("pool", pd.Series(dtype=str)).astype(str) == "B").sum()
                ),
                "peak_used_a": max_int("used_a_after"),
                "peak_used_b": max_int("used_b_after"),
                "peak_used_total": max_int("used_total_after"),
                "used_a_at_last_event": int(sub.iloc[-1].get("used_a_after", 0) or 0),
                "used_b_at_last_event": int(sub.iloc[-1].get("used_b_after", 0) or 0),
            }
        )

    return pd.DataFrame(rows).sort_values("receiver_id").reset_index(drop=True)
