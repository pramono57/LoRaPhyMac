"""Read-only audit tables for routing, APC feedback, and transmitter airtime."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from loraphymac.mac.frame_defs import MType, mtype_name


TOPOLOGY_PARENT_COLUMNS = [
    "run_id",
    "seed",
    "node_ptx_mode",
    "relay_ptx_mode",
    "mode",
    "terrain_delta_h",
    "routing_strategy",
    "route_construction_delta_h",
    "physical_terrain_delta_h",
    "routing_snapshot_id",
    "routing_snapshot_sha256",
    "topology_sha256",
    "selected_parent_routing_sha256",
    "entity_id",
    "entity_type",
    "x_m",
    "y_m",
    "height_m",
    "parent_id",
    "route_scope",
    "selection_rule",
    "candidate_count",
    "routing_ch_id",
    "routing_freq_mhz",
    "distance_m",
    "pathloss_db",
    "rssi_dbm",
    "snr_db",
    "margin_db",
    "hop",
    "score",
    "valid",
]

TX_AIRTIME_COLUMNS = [
    "run_id",
    "entity_id",
    "entity_type",
    "traffic_class",
    "mtype",
    "mtype_name",
    "ptx_dbm",
    "channel",
    "sf",
    "bw_hz",
    "tx_packet_count",
    "total_airtime_s",
    "entity_grouped_airtime_s",
    "entity_recorded_total_tx_time_s",
    "airtime_consistency_delta_s",
    "airtime_consistent",
]

NODE_TX_DUTY_CYCLE_COLUMNS = [
    "run_id", "entity_id", "entity_type", "sim_time_s",
    "observed_tx_airtime_s", "tx_duty_fraction",
    "tx_duty_cycle_percent", "measurement_scope",
]
CHANNEL_OCCUPANCY_COLUMNS = [
    "run_id", "channel", "sim_time_s", "sum_offered_tx_airtime_s",
    "sum_observed_tx_airtime_s", "channel_busy_union_s",
    "channel_busy_fraction", "channel_busy_percent", "occupancy_method",
]

APC_FEEDBACK_LIFECYCLE_COLUMNS = [
    "run_id",
    "event_sequence",
    "feedback_id",
    "packet_uid",
    "event",
    "timestamp_s",
    "decision_time_s",
    "feedback_age_s",
    "retry_timeout_s",
    "stale_tpi",
    "controller_id",
    "controller_type",
    "target_id",
    "target_type",
    "src_id",
    "hop_index",
    "control_scope",
    "mtype",
    "mtype_name",
    "channel",
    "sf",
    "bw_hz",
    "control_airtime_s",
    "old_tx_power_index",
    "decision_tx_power_index",
    "applied_tx_power_index",
    "old_ptx_dbm",
    "applied_ptx_dbm",
    "decision_margin_db",
    "decision_rssi_dbm",
    "decision_snr_db",
    "apc_rx_margin_db",
    "apc_rx_rssi_dbm",
    "apc_rx_snr_db",
    "decoded",
    "collided",
    "reason",
]

WANG_SNR_REPORT_LIFECYCLE_COLUMNS = [
    "run_id",
    "event_sequence",
    "report_id",
    "packet_uid",
    "event",
    "timestamp_s",
    "observation_time_s",
    "report_age_s",
    "tx_id",
    "rx_id",
    "hop_index",
    "snr_db",
    "observed_ptx_dbm",
    "ptx_epoch",
    "last_applied_command_seq",
    "ptx_provenance",
    "carrier_hop_index",
    "receiver_id",
    "fopts_len",
    "reason",
]

WANG_POWER_COMMAND_LIFECYCLE_COLUMNS = [
    "run_id",
    "event_sequence",
    "command_id",
    "event",
    "timestamp_s",
    "target_id",
    "command_seq",
    "window_index",
    "expected_ptx_dbm",
    "expected_epoch",
    "target_ptx_dbm",
    "issued_at_s",
    "expiry_s",
    "packet_uid",
    "current_tx",
    "next_hop",
    "receiver_id",
    "hop_index",
    "mtype",
    "channel",
    "sf",
    "bw_hz",
    "fopts_len",
    "control_airtime_s",
    "reason",
]


def _entity_type(entity_id: Any) -> str:
    value = str(entity_id)
    if value == "GW" or value.startswith("GW"):
        return "gateway"
    if value.startswith("R"):
        return "relay"
    return "node"


def _metadata(run_metadata: Mapping[str, Any]) -> dict[str, Any]:
    node_mode = run_metadata.get("node_ptx_mode")
    relay_mode = run_metadata.get("relay_ptx_mode")
    mode = None
    if node_mode is not None and relay_mode is not None:
        node_code = "A" if str(node_mode).lower() == "adaptive" else "F"
        relay_code = "A" if str(relay_mode).lower() == "adaptive" else "F"
        mode = f"{node_code}{relay_code}"
    return {
        "run_id": run_metadata.get("run_id"),
        "seed": run_metadata.get("seed"),
        "node_ptx_mode": node_mode,
        "relay_ptx_mode": relay_mode,
        "mode": mode,
        "terrain_delta_h": run_metadata.get("terrain_delta_h"),
        "routing_strategy": run_metadata.get(
            "routing_strategy", "terrain_specific"
        ),
        "route_construction_delta_h": run_metadata.get(
            "route_construction_delta_h", run_metadata.get("terrain_delta_h")
        ),
        "physical_terrain_delta_h": run_metadata.get(
            "physical_terrain_delta_h", run_metadata.get("terrain_delta_h")
        ),
        "routing_snapshot_id": run_metadata.get("routing_snapshot_id"),
        "routing_snapshot_sha256": run_metadata.get(
            "routing_snapshot_sha256"
        ),
        "topology_sha256": run_metadata.get("topology_sha256"),
        "selected_parent_routing_sha256": run_metadata.get(
            "selected_parent_routing_sha256"
        ),
    }


def build_topology_parent_by_run_df(
    routing: Any,
    run_metadata: Mapping[str, Any],
) -> pd.DataFrame:
    """Describe coordinates and the selected parent for every run entity."""
    selected_frames = [
        frame
        for frame in (
            getattr(routing, "selected_next_hop_links_df", None),
            getattr(routing, "relay_chain_links_df", None),
        )
        if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    selected = (
        pd.concat(selected_frames, ignore_index=True, sort=False)
        if selected_frames
        else pd.DataFrame()
    )
    link_df = getattr(routing, "link_df", pd.DataFrame())
    routing_map = dict(getattr(routing, "routing_map", {}))
    prefix = _metadata(run_metadata)
    rows: list[dict[str, Any]] = []

    for entity in getattr(routing, "all_nodes", []):
        entity_id = str(getattr(entity, "id", entity))
        entity_type = _entity_type(entity_id)
        parent_id = routing_map.get(entity_id)
        loc = getattr(entity, "loc", None)

        route_row = None
        if not selected.empty and {"Node", "Target"}.issubset(selected.columns):
            matches = selected[
                (selected["Node"].astype(str) == entity_id)
                & (selected["Target"].astype(str) == str(parent_id))
            ]
            if not matches.empty:
                route_row = matches.iloc[0]

        if entity_type == "node":
            route_scope = "end_device_parent"
            selection_rule = "highest_score"
            if isinstance(link_df, pd.DataFrame) and "Node" in link_df.columns:
                candidate_count = int((link_df["Node"].astype(str) == entity_id).sum())
            else:
                candidate_count = None
        elif entity_type == "relay":
            route_scope = "relay_backbone_parent"
            selection_rule = "deterministic_linear_relay_chain"
            candidate_count = 1 if parent_id is not None else 0
        else:
            route_scope = "gateway_terminal"
            selection_rule = "terminal"
            candidate_count = 0

        def selected_value(column: str) -> Any:
            return None if route_row is None else route_row.get(column)

        rows.append(
            {
                **prefix,
                "entity_id": entity_id,
                "entity_type": entity_type,
                "x_m": getattr(loc, "x", None),
                "y_m": getattr(loc, "y", None),
                "height_m": getattr(entity, "h", None),
                "parent_id": parent_id,
                "route_scope": route_scope,
                "selection_rule": selection_rule,
                "candidate_count": candidate_count,
                "routing_ch_id": selected_value("ChID"),
                "routing_freq_mhz": selected_value("Freq(MHz)"),
                "distance_m": selected_value("Distance(m)"),
                "pathloss_db": selected_value("Pathloss(dB)"),
                "rssi_dbm": selected_value("RSSI(dBm)"),
                "snr_db": selected_value("SNR(dB)"),
                "margin_db": selected_value("Margin(dB)"),
                "hop": selected_value("Hop"),
                "score": selected_value("Score"),
                "valid": selected_value("Valid"),
            }
        )

    return pd.DataFrame(rows, columns=TOPOLOGY_PARENT_COLUMNS)


def _traffic_class(mtype: Any) -> str:
    if pd.isna(mtype):
        return "legacy_uplink"
    value = int(mtype)
    if value == int(MType.DATA_UP):
        return "uplink"
    if value == int(MType.DATA_UP_FWD):
        return "forwarding"
    if value in (int(MType.DATA_DOWN), int(MType.DATA_DOWN_FWD)):
        return "control_feedback"
    return "other"


def _kahan_add(total: float, compensation: float, value: float) -> tuple[float, float]:
    adjusted = value - compensation
    updated = total + adjusted
    return updated, (updated - total) - adjusted


def _normalized_numeric_key(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    return value


def build_tx_airtime_by_entity_mtype_ptx_channel_df(
    records: Sequence[Mapping[str, Any]],
    nodes: Sequence[Any],
    run_metadata: Mapping[str, Any],
    *,
    tolerance_s: float = 1e-9,
) -> pd.DataFrame:
    """Aggregate observed TX airtime without changing runtime accounting."""
    events = pd.DataFrame(records)
    if events.empty or "event" not in events.columns:
        return pd.DataFrame(columns=TX_AIRTIME_COLUMNS)

    tx = events[events["event"].astype(str) == "TX"].copy()
    if tx.empty:
        return pd.DataFrame(columns=TX_AIRTIME_COLUMNS)

    if "observed_tx_airtime_s" not in tx.columns:
        tx["observed_tx_airtime_s"] = pd.to_numeric(tx.get("tx_end"), errors="coerce") - pd.to_numeric(
            tx.get("tx_start"), errors="coerce"
        )
    tx["observed_tx_airtime_s"] = pd.to_numeric(
        tx["observed_tx_airtime_s"], errors="coerce"
    ).fillna(0.0)
    tx["entity_id"] = tx["node"].astype(str)
    tx["entity_type"] = tx["entity_id"].map(_entity_type)
    aggregates: dict[tuple[Any, ...], list[Any]] = {}
    for event in tx.itertuples(index=False):
        entity_id = str(event.node)
        mtype = _normalized_numeric_key(event.mtype)
        mtype = None if mtype is None else int(mtype)
        key = (
            entity_id,
            _entity_type(entity_id),
            _traffic_class(mtype),
            mtype,
            "LEGACY" if mtype is None else mtype_name(mtype),
            None if pd.isna(event.ptx) else float(event.ptx),
            _normalized_numeric_key(event.ch),
            _normalized_numeric_key(event.sf),
            _normalized_numeric_key(event.bw_hz),
        )
        aggregate = aggregates.setdefault(key, [0, 0.0, 0.0])
        aggregate[0] += 1
        aggregate[1], aggregate[2] = _kahan_add(
            aggregate[1], aggregate[2], float(event.observed_tx_airtime_s)
        )

    group_columns = [
        "entity_id", "entity_type", "traffic_class", "mtype", "mtype_name",
        "ptx_dbm", "channel", "sf", "bw_hz",
    ]
    rows = [
        {
            **dict(zip(group_columns, key, strict=True)),
            "tx_packet_count": values[0],
            "total_airtime_s": values[1],
        }
        for key, values in aggregates.items()
    ]
    grouped = pd.DataFrame(rows, columns=group_columns + ["tx_packet_count", "total_airtime_s"])
    sort_columns = [
        "entity_id", "entity_type", "traffic_class", "mtype", "mtype_name",
        "ptx_dbm", "channel", "sf", "bw_hz",
    ]
    grouped = grouped.sort_values(
        sort_columns, kind="stable", na_position="last"
    ).reset_index(drop=True)
    entity_grouped: dict[str, tuple[float, float]] = {}
    for row in grouped.itertuples(index=False):
        total, compensation = entity_grouped.get(row.entity_id, (0.0, 0.0))
        entity_grouped[row.entity_id] = _kahan_add(
            total, compensation, float(row.total_airtime_s)
        )
    entity_recorded = {
        str(getattr(node, "id", node)): float(getattr(node, "total_tx_time_s", 0.0))
        for node in nodes
    }
    grouped.insert(0, "run_id", run_metadata.get("run_id"))
    grouped["entity_grouped_airtime_s"] = grouped["entity_id"].map(
        lambda entity_id: entity_grouped[str(entity_id)][0]
    )
    grouped["entity_recorded_total_tx_time_s"] = grouped["entity_id"].map(entity_recorded)
    grouped["airtime_consistency_delta_s"] = (
        grouped["entity_grouped_airtime_s"]
        - grouped["entity_recorded_total_tx_time_s"]
    )
    grouped["airtime_consistent"] = (
        grouped["airtime_consistency_delta_s"].abs() <= float(tolerance_s)
    )
    return grouped.reindex(columns=TX_AIRTIME_COLUMNS)


def _tx_event_rows(event_log_df: pd.DataFrame) -> pd.DataFrame:
    """Return normalized TX rows from a simulator event table."""
    if event_log_df is None or event_log_df.empty or "event" not in event_log_df:
        return pd.DataFrame()
    tx = event_log_df.loc[event_log_df["event"].eq("TX")].copy()
    if not {"node", "time", "ch"}.issubset(tx.columns):
        return pd.DataFrame()
    if "toa_s" not in tx:
        tx["toa_s"] = 0.0
    if "observed_tx_airtime_s" not in tx:
        tx["observed_tx_airtime_s"] = tx["toa_s"]
    for column in ("time", "toa_s", "observed_tx_airtime_s"):
        tx[column] = pd.to_numeric(tx[column], errors="coerce")
    return tx.dropna(subset=["node", "time", "toa_s", "observed_tx_airtime_s"])


def build_node_tx_duty_cycle_df(
    event_log_df: pd.DataFrame,
    nodes: Sequence[Any],
    run_metadata: Mapping[str, Any],
) -> pd.DataFrame:
    """Measure observed TX duty fraction for every end node in a run.

    This is descriptive airtime accounting, not a duty-cycle limiter or a
    regulatory-compliance determination.
    """
    sim_time_s = float(run_metadata.get("sim_time", 0.0) or 0.0)
    if sim_time_s <= 0:
        return pd.DataFrame()
    tx = _tx_event_rows(event_log_df)
    airtime_by_node = (
        tx.groupby("node")["observed_tx_airtime_s"].sum()
        if not tx.empty
        else pd.Series(dtype=float)
    )
    rows = []
    for entity in nodes:
        entity_id = str(getattr(entity, "id", ""))
        if entity_id == "GW" or entity_id.startswith("GW") or entity_id.startswith("R"):
            continue
        airtime_s = float(airtime_by_node.get(entity_id, 0.0))
        rows.append(
            {
                "run_id": run_metadata.get("run_id"),
                "entity_id": entity_id,
                "entity_type": "node",
                "sim_time_s": sim_time_s,
                "observed_tx_airtime_s": airtime_s,
                "tx_duty_fraction": airtime_s / sim_time_s,
                "tx_duty_cycle_percent": airtime_s / sim_time_s * 100.0,
                "measurement_scope": "physical_tx_airtime_over_simulation_horizon",
            }
        )
    return pd.DataFrame(rows, columns=NODE_TX_DUTY_CYCLE_COLUMNS)


def build_channel_occupancy_df(
    event_log_df: pd.DataFrame,
    run_metadata: Mapping[str, Any],
) -> pd.DataFrame:
    """Separate summed TX airtime from per-channel interval-union busy time."""
    try:
        sim_time_s = float(run_metadata.get("sim_time", 0.0) or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("sim_time must be a finite non-negative number") from exc
    if not math.isfinite(sim_time_s) or sim_time_s < 0:
        raise ValueError("sim_time must be a finite non-negative number")
    if sim_time_s == 0:
        return pd.DataFrame(columns=CHANNEL_OCCUPANCY_COLUMNS)
    if event_log_df is None or event_log_df.empty:
        return pd.DataFrame(columns=CHANNEL_OCCUPANCY_COLUMNS)
    if "event" not in event_log_df.columns:
        raise ValueError("TX event log is missing required column 'event'")
    tx_source = event_log_df.loc[event_log_df["event"].eq("TX")].copy()
    if tx_source.empty:
        return pd.DataFrame(columns=CHANNEL_OCCUPANCY_COLUMNS)
    required = {"node", "time", "ch", "toa_s", "observed_tx_airtime_s"}
    missing = sorted(required - set(tx_source.columns))
    if missing:
        raise ValueError(f"TX event log is missing required columns: {missing}")
    if tx_source[list(required)].isna().any().any():
        raise ValueError("TX event log contains null node, time, channel, or airtime")
    tx = _tx_event_rows(event_log_df)
    if len(tx) != len(tx_source):
        raise ValueError("TX event log contains malformed numeric values")
    numeric_columns = ("time", "ch", "toa_s", "observed_tx_airtime_s")
    for column in numeric_columns:
        values = pd.to_numeric(tx[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise ValueError(f"TX event log column {column!r} must be finite numeric data")
        tx[column] = values
    if (tx["time"] < 0).any():
        raise ValueError("TX event timestamps must be non-negative")
    if (tx["toa_s"] < 0).any() or (tx["observed_tx_airtime_s"] < 0).any():
        raise ValueError("TX airtime values must be non-negative")
    if (tx["observed_tx_airtime_s"] > tx["toa_s"] + 1e-12).any():
        raise ValueError("observed TX airtime cannot exceed nominal ToA")
    if (tx["ch"] < 0).any() or not tx["ch"].map(
        lambda value: float(value).is_integer()
    ).all():
        raise ValueError("TX channel IDs must be non-negative integers")
    if tx["time"].diff().dropna().lt(0).any():
        raise ValueError("TX event timestamps must be ordered non-decreasingly")
    duplicate_columns = ["node", "time", "ch"]
    for column in ("packet_uid", "mtype", "toa_s", "observed_tx_airtime_s"):
        if column in tx:
            duplicate_columns.append(column)
    if tx.duplicated(subset=duplicate_columns).any():
        raise ValueError("TX event log contains a duplicate physical TX record")

    accumulators: dict[Any, list[Any]] = {}
    for event in tx.dropna(subset=["ch"]).itertuples(index=False):
        channel = _normalized_numeric_key(event.ch)
        state = accumulators.setdefault(channel, [0.0, 0.0, 0.0, 0.0, None, None, 0.0])
        state[0], state[1] = _kahan_add(state[0], state[1], float(event.toa_s))
        start = float(event.time)
        observed_airtime = min(
            float(event.observed_tx_airtime_s), max(sim_time_s - start, 0.0)
        )
        state[2], state[3] = _kahan_add(
            state[2], state[3], observed_airtime
        )
        end = min(sim_time_s, start + observed_airtime)
        if end <= start:
            continue
        if state[4] is None:
            state[4], state[5] = start, end
        elif start <= state[5]:
            state[5] = max(state[5], end)
        else:
            state[6] += state[5] - state[4]
            state[4], state[5] = start, end

    rows = []
    for channel, state in sorted(accumulators.items()):
        busy_union_s = state[6]
        if state[4] is not None:
            busy_union_s += state[5] - state[4]
        observed_airtime_s = state[2]
        tolerance_s = max(1e-12, observed_airtime_s * 1e-12)
        if busy_union_s > observed_airtime_s + tolerance_s:
            raise ValueError(
                f"channel {channel}: busy-union airtime exceeds observed TX airtime"
            )
        busy_union_s = min(busy_union_s, observed_airtime_s, sim_time_s)
        occupancy_percent = busy_union_s / sim_time_s * 100.0
        if not 0.0 <= occupancy_percent <= 100.0 + 1e-10:
            raise ValueError(f"channel {channel}: occupancy percentage is outside [0, 100]")
        rows.append({
            "run_id": run_metadata.get("run_id"),
            "channel": channel,
            "sim_time_s": sim_time_s,
            "sum_offered_tx_airtime_s": state[0],
            "sum_observed_tx_airtime_s": observed_airtime_s,
            "channel_busy_union_s": busy_union_s,
            "channel_busy_fraction": busy_union_s / sim_time_s,
            "channel_busy_percent": occupancy_percent,
            "occupancy_method": "union_of_simulated_tx_intervals",
        })
    return pd.DataFrame(rows, columns=CHANNEL_OCCUPANCY_COLUMNS)


def build_apc_feedback_lifecycle_df(
    records: Sequence[Mapping[str, Any]],
    run_metadata: Mapping[str, Any],
) -> pd.DataFrame:
    """Return schema-complete physical APC feedback lifecycle evidence."""
    if not records:
        return pd.DataFrame(columns=APC_FEEDBACK_LIFECYCLE_COLUMNS)
    frame = pd.DataFrame(records)
    if "run_id" not in frame.columns:
        frame.insert(0, "run_id", run_metadata.get("run_id"))
    return frame.reindex(columns=APC_FEEDBACK_LIFECYCLE_COLUMNS)


def build_wang_snr_report_lifecycle_df(
    records: Sequence[Mapping[str, Any]],
    run_metadata: Mapping[str, Any],
) -> pd.DataFrame:
    """Return schema-complete physical Wang SNR-report evidence."""
    if not records:
        return pd.DataFrame(columns=WANG_SNR_REPORT_LIFECYCLE_COLUMNS)
    frame = pd.DataFrame(records)
    if "run_id" not in frame.columns:
        frame.insert(0, "run_id", run_metadata.get("run_id"))
    return frame.reindex(columns=WANG_SNR_REPORT_LIFECYCLE_COLUMNS)


def build_wang_power_command_lifecycle_df(
    records: Sequence[Mapping[str, Any]],
    run_metadata: Mapping[str, Any],
) -> pd.DataFrame:
    """Return physical Wang command generation/delivery evidence."""
    if not records:
        return pd.DataFrame(columns=WANG_POWER_COMMAND_LIFECYCLE_COLUMNS)
    frame = pd.DataFrame(records)
    if "run_id" not in frame.columns:
        frame.insert(0, "run_id", run_metadata.get("run_id"))
    return frame.reindex(columns=WANG_POWER_COMMAND_LIFECYCLE_COLUMNS)


__all__ = [
    "APC_FEEDBACK_LIFECYCLE_COLUMNS",
    "CHANNEL_OCCUPANCY_COLUMNS",
    "NODE_TX_DUTY_CYCLE_COLUMNS",
    "TOPOLOGY_PARENT_COLUMNS",
    "TX_AIRTIME_COLUMNS",
    "WANG_SNR_REPORT_LIFECYCLE_COLUMNS",
    "WANG_POWER_COMMAND_LIFECYCLE_COLUMNS",
    "build_apc_feedback_lifecycle_df",
    "build_topology_parent_by_run_df",
    "build_tx_airtime_by_entity_mtype_ptx_channel_df",
    "build_node_tx_duty_cycle_df",
    "build_channel_occupancy_df",
    "build_wang_snr_report_lifecycle_df",
    "build_wang_power_command_lifecycle_df",
]
