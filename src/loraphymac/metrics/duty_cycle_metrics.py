"""Read-only duty-cycle result tables for the :class:`Statistics` facade."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd


def build_relay_duty_cycle_df(
    nodes: Sequence[Any] | None,
    run_metadata: Mapping[str, Any],
    infer_entity_type: Callable[[str | None], str],
) -> pd.DataFrame:
    """Build aggregate transmitter duty cycle per relay and gateway."""
    sim_time_s = run_metadata.get("sim_time", None)
    if sim_time_s is None or float(sim_time_s) <= 0 or nodes is None:
        return pd.DataFrame()

    sim_time_s = float(sim_time_s)
    configured_dc = run_metadata.get("relay_duty_cycle", None)
    configured_dc_percent = None if configured_dc is None else float(configured_dc) * 100.0
    rows = []

    for node in nodes:
        entity_id = getattr(node, "id", None)
        entity_type = infer_entity_type(entity_id)
        if entity_type not in ("relay", "gateway"):
            continue

        total_tx_time_s = float(getattr(node, "total_tx_time_s", 0.0))
        duty_cycle_percent = total_tx_time_s / sim_time_s * 100.0

        if configured_dc is None:
            duty_cycle_margin_percent = None
            dc_status = "unlimited"
        else:
            duty_cycle_margin_percent = configured_dc_percent - duty_cycle_percent
            dc_status = "OK" if duty_cycle_percent <= configured_dc_percent + 1e-9 else "VIOLATION"

        rows.append(
            {
                "run_id": run_metadata.get("run_id"),
                "entity_id": entity_id,
                "entity_type": entity_type,
                "total_tx_time_s": total_tx_time_s,
                "sim_time_s": sim_time_s,
                "relay_duty_cycle_percent": duty_cycle_percent,
                "configured_relay_duty_cycle": configured_dc,
                "configured_relay_duty_cycle_percent": configured_dc_percent,
                "duty_cycle_margin_percent": duty_cycle_margin_percent,
                "dc_status": dc_status,
                "relay_dc_mode": run_metadata.get("relay_dc_mode", None),
            }
        )

    return pd.DataFrame(rows)


def build_per_channel_duty_cycle_df(
    nodes: Sequence[Any],
    sim_time: float | None,
    run_metadata: Mapping[str, Any],
    infer_entity_type: Callable[[str], str],
) -> pd.DataFrame:
    """Build transmitter airtime and duty cycle per relay/gateway channel."""
    if sim_time is None:
        sim_time = float(run_metadata.get("sim_time", 0.0))

    configured_dc = run_metadata.get("relay_duty_cycle", None)
    configured_dc_percent = None if configured_dc is None else float(configured_dc) * 100.0
    rows = []

    for node in nodes:
        entity_type = infer_entity_type(node.id)
        if entity_type not in ("relay", "gateway"):
            continue

        airtime_map = getattr(node, "tx_airtime_per_channel", {})
        count_map = getattr(node, "tx_packet_count_per_channel", {})

        for channel_id in sorted(airtime_map.keys()):
            total_airtime_s = float(airtime_map.get(channel_id, 0.0))
            packet_count = int(count_map.get(channel_id, 0))
            per_channel_dc_percent = total_airtime_s / sim_time * 100.0 if sim_time > 0 else 0.0

            if configured_dc is None:
                duty_cycle_margin_percent = None
                dc_status = "unlimited"
            else:
                duty_cycle_margin_percent = configured_dc_percent - per_channel_dc_percent
                dc_status = (
                    "OK" if per_channel_dc_percent <= configured_dc_percent + 1e-9 else "VIOLATION"
                )

            rows.append(
                {
                    "run_id": run_metadata.get("run_id"),
                    "entity_id": node.id,
                    "entity_type": entity_type,
                    "channel": channel_id,
                    "sim_time_s": sim_time,
                    "configured_relay_duty_cycle": configured_dc,
                    "configured_relay_duty_cycle_percent": configured_dc_percent,
                    "tx_packet_count": packet_count,
                    "total_airtime_per_channel_s": total_airtime_s,
                    "per_channel_duty_cycle_percent": per_channel_dc_percent,
                    "duty_cycle_margin_percent": duty_cycle_margin_percent,
                    "dc_status": dc_status,
                }
            )

    return pd.DataFrame(rows)
