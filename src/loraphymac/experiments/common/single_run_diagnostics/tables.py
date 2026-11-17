"""Shared diagnostic table builders for one-run experiment workflows."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from loraphymac import SimulationResult
from loraphymac.experiments.common.multiseed_pipeline import atomic_write_csv


APC_DECISION_TABLE_COLUMNS = (
    "Target",
    "Type",
    "Controller",
    "Feedback frame",
    "RSSI used for decision (dBm)",
    "Link margin (dB)",
    "APC margin (dB)",
    "Old TPI",
    "New TPI",
    "Old PTx (dBm)",
    "New PTx (dBm)",
    "APC decision",
    "APC feedback RX margin (dB)",
)

DATA_UP_DOWN_TABLE_COLUMNS = (
    "node",
    "data_up_ch",
    "data_down_tx_count",
    "data_down_rx_count",
    "data_down_drop_count",
    "data_down_tx_ch",
    "data_down_rx_ch",
    "data_down_drop_ch",
    "channel_match",
    "drop_reason_summary",
)


def _series_or_missing(frame: pd.DataFrame, primary: str, fallback: str | None = None):
    if primary in frame.columns:
        return frame[primary]
    if fallback is not None and fallback in frame.columns:
        return frame[fallback]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def build_apc_decision_validation_table(
    ptx_history: pd.DataFrame,
    *,
    target_margin_db: float = 6.0,
    node_limit: int = 4,
    relay_limit: int = 2,
) -> pd.DataFrame:
    """Build legacy Table IV from representative physical APC decisions."""
    if ptx_history is None or ptx_history.empty:
        raise ValueError("PTx history is empty; an adaptive single run is required")

    required = {"entity_id", "entity_type", "time", "old_ptx", "new_ptx"}
    missing = required.difference(ptx_history.columns)
    if missing:
        raise ValueError(f"PTx history is missing required columns: {sorted(missing)}")

    frame = ptx_history.copy()
    frame["entity_type"] = frame["entity_type"].astype(str).str.lower()
    frame["entity_key"] = frame["entity_id"].astype(str)
    frame["time_numeric"] = pd.to_numeric(frame["time"], errors="coerce")
    frame["link_margin_db"] = pd.to_numeric(
        _series_or_missing(frame, "decision_margin", "margin"),
        errors="coerce",
    )
    frame["decision_rssi_numeric"] = pd.to_numeric(
        _series_or_missing(frame, "decision_rssi", "rssi"),
        errors="coerce",
    )
    frame["apc_margin_db"] = frame["link_margin_db"] - float(target_margin_db)
    frame["old_tpi_numeric"] = pd.to_numeric(
        _series_or_missing(frame, "old_tx_power_index"),
        errors="coerce",
    )
    frame["new_tpi_numeric"] = pd.to_numeric(
        _series_or_missing(frame, "new_tx_power_index"),
        errors="coerce",
    )
    frame["delta_tpi"] = frame["new_tpi_numeric"] - frame["old_tpi_numeric"]
    frame["old_ptx_numeric"] = pd.to_numeric(frame["old_ptx"], errors="coerce")
    frame["new_ptx_numeric"] = pd.to_numeric(frame["new_ptx"], errors="coerce")
    frame["abs_delta_ptx"] = (frame["new_ptx_numeric"] - frame["old_ptx_numeric"]).abs()

    frame["APC decision"] = "Unknown"
    frame.loc[frame["delta_tpi"].gt(0), "APC decision"] = "Decrease PTx"
    frame.loc[frame["delta_tpi"].lt(0), "APC decision"] = "Increase PTx"
    frame.loc[frame["delta_tpi"].eq(0), "APC decision"] = "Keep PTx"

    apply_source = _series_or_missing(frame, "apply_source").fillna("").astype(str).str.lower()
    frame["Feedback frame"] = "APC feedback"
    frame.loc[apply_source.str.contains("data_down", na=False), "Feedback frame"] = "DATA_DOWN"
    frame.loc[apply_source.str.contains("data_down_fwd", na=False), "Feedback frame"] = (
        "DATA_DOWN_FWD"
    )

    changed = frame.loc[frame["delta_tpi"].fillna(0).ne(0)].copy()
    candidates = changed if not changed.empty else frame

    def representative(entity_type: str, limit: int) -> pd.DataFrame:
        subset = candidates.loc[candidates["entity_type"].eq(entity_type)].copy()
        if subset.empty or limit <= 0:
            return subset.iloc[0:0]
        return (
            subset.sort_values(
                ["abs_delta_ptx", "time_numeric", "entity_key"],
                ascending=[False, True, True],
                na_position="last",
            )
            .drop_duplicates("entity_key", keep="first")
            .head(limit)
        )

    selected = pd.concat(
        [representative("node", node_limit), representative("relay", relay_limit)],
        ignore_index=True,
    )
    if selected.empty:
        selected = (
            frame.sort_values(["time_numeric", "entity_key"], na_position="last")
            .drop_duplicates("entity_key", keep="last")
            .head(max(0, node_limit) + max(0, relay_limit))
        )

    table = pd.DataFrame(
        {
            "Target": selected["entity_key"],
            "Type": selected["entity_type"],
            "Controller": _series_or_missing(selected, "feedback_from"),
            "Feedback frame": selected["Feedback frame"],
            "RSSI used for decision (dBm)": selected["decision_rssi_numeric"],
            "Link margin (dB)": selected["link_margin_db"],
            "APC margin (dB)": selected["apc_margin_db"],
            "Old TPI": selected["old_tpi_numeric"],
            "New TPI": selected["new_tpi_numeric"],
            "Old PTx (dBm)": selected["old_ptx_numeric"],
            "New PTx (dBm)": selected["new_ptx_numeric"],
            "APC decision": selected["APC decision"],
            "APC feedback RX margin (dB)": pd.to_numeric(
                _series_or_missing(selected, "apc_rx_margin"),
                errors="coerce",
            ),
        },
        columns=list(APC_DECISION_TABLE_COLUMNS),
    )

    float_columns = (
        "RSSI used for decision (dBm)",
        "Link margin (dB)",
        "APC margin (dB)",
        "APC feedback RX margin (dB)",
    )
    table[list(float_columns)] = table[list(float_columns)].round(3)
    for column in ("Old TPI", "New TPI", "Old PTx (dBm)", "New PTx (dBm)"):
        table[column] = pd.to_numeric(table[column], errors="coerce").round().astype("Int64")
    return table.reset_index(drop=True)


def _mtype_to_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        name = str(value).rsplit(".", maxsplit=1)[-1].upper()
        return {"DATA_UP": 1, "DATA_DOWN": 3}.get(name)


def _node_sort_key(node_id: str) -> tuple[int, int | str]:
    value = str(node_id)
    suffix = value[1:]
    if value.upper().startswith("N") and suffix.isdigit():
        return (0, int(suffix))
    return (1, value)


def _channel_map(frame: pd.DataFrame, group_column: str) -> dict[str, list[Any]]:
    if frame.empty:
        return {}

    def unique_sorted(values: pd.Series) -> list[Any]:
        unique = values.dropna().unique().tolist()
        try:
            return sorted(unique)
        except TypeError:
            return sorted(unique, key=str)

    grouped = frame.dropna(subset=[group_column]).groupby(group_column)["ch"].agg(unique_sorted)
    return {str(key): value for key, value in grouped.items()}


def _count_map(frame: pd.DataFrame, group_column: str) -> dict[str, int]:
    if frame.empty:
        return {}
    counts = frame.dropna(subset=[group_column]).groupby(group_column).size()
    return {str(key): int(value) for key, value in counts.items()}


def build_data_up_down_debug_table(
    event_log: pd.DataFrame,
    *,
    node_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize DATA_UP and physical DATA_DOWN behavior for each node."""
    required = {"event", "mtype", "src", "next_hop", "node", "ch", "reason"}
    missing = required.difference(event_log.columns)
    if missing:
        raise ValueError(f"Event log is missing required columns: {sorted(missing)}")

    events = event_log.copy()
    events["event_normalized"] = events["event"].astype(str).str.upper()
    events["mtype_int"] = events["mtype"].map(_mtype_to_int)
    tx = events["event_normalized"].eq("TX")
    rx = events["event_normalized"].eq("RX")
    drop = events["event_normalized"].eq("DROP")
    data_up_tx = events.loc[tx & events["mtype_int"].eq(1)].copy()
    data_down_tx = events.loc[tx & events["mtype_int"].eq(3)].copy()
    data_down_rx = events.loc[rx & events["mtype_int"].eq(3)].copy()
    data_down_drop = events.loc[drop & events["mtype_int"].eq(3)].copy()

    if node_ids is None:
        observed = {
            *data_up_tx["src"].dropna().astype(str),
            *data_down_tx["next_hop"].dropna().astype(str),
            *data_down_rx["node"].dropna().astype(str),
            *data_down_drop["node"].dropna().astype(str),
        }
        resolved_nodes = sorted(observed, key=_node_sort_key)
    else:
        resolved_nodes = list(dict.fromkeys(str(node) for node in node_ids))

    up_channels = _channel_map(data_up_tx, "src")
    down_tx_channels = _channel_map(data_down_tx, "next_hop")
    down_rx_channels = _channel_map(data_down_rx, "node")
    down_drop_channels = _channel_map(data_down_drop, "node")
    down_tx_counts = _count_map(data_down_tx, "next_hop")
    down_rx_counts = _count_map(data_down_rx, "node")
    down_drop_counts = _count_map(data_down_drop, "node")

    reasons = data_down_drop.dropna(subset=["node", "reason"])
    reason_map: dict[str, str] = {}
    if not reasons.empty:
        grouped = reasons.groupby(["node", "reason"]).size().reset_index(name="count")
        grouped = grouped.sort_values(["node", "count", "reason"], ascending=[True, False, True])
        for node, subset in grouped.groupby("node", sort=False):
            reason_map[str(node)] = ", ".join(
                f"{row.reason}={int(row.count)}" for row in subset.itertuples(index=False)
            )

    rows = []
    for node in resolved_nodes:
        up = up_channels.get(node)
        down = down_tx_channels.get(node)
        rows.append(
            {
                "node": node,
                "data_up_ch": up,
                "data_down_tx_count": down_tx_counts.get(node, 0),
                "data_down_rx_count": down_rx_counts.get(node, 0),
                "data_down_drop_count": down_drop_counts.get(node, 0),
                "data_down_tx_ch": down,
                "data_down_rx_ch": down_rx_channels.get(node),
                "data_down_drop_ch": down_drop_channels.get(node),
                "channel_match": bool(up is not None and down is not None and up == down),
                "drop_reason_summary": reason_map.get(node, ""),
            }
        )
    return pd.DataFrame(rows, columns=list(DATA_UP_DOWN_TABLE_COLUMNS))


def generate_single_run_tables(
    result: SimulationResult,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Build and save all maintained single-run diagnostic tables."""
    run_directory = Path(result.resolved_config["output_dir"])
    destination = Path(output_dir) if output_dir is not None else run_directory / "tables"
    destination.mkdir(parents=True, exist_ok=True)

    node_count = int(result.resolved_config["node_num"])
    node_ids = [f"N{index}" for index in range(1, node_count + 1)]
    decision_table = build_apc_decision_validation_table(
        result.ptx_history,
        target_margin_db=float(result.resolved_config["node_target_margin_db"]),
    )
    data_up_down_table = build_data_up_down_debug_table(
        result.table("event_log_df"),
        node_ids=node_ids,
    )

    outputs = {
        "apc_decision_validation": (
            destination / "table_iv_representative_apc_decision_validation.csv"
        ),
        "data_up_down_debug": destination / "table_debug_data_up_down_by_node.csv",
    }
    atomic_write_csv(outputs["apc_decision_validation"], decision_table)
    atomic_write_csv(outputs["data_up_down_debug"], data_up_down_table)
    return outputs


__all__ = [
    "APC_DECISION_TABLE_COLUMNS",
    "DATA_UP_DOWN_TABLE_COLUMNS",
    "build_apc_decision_validation_table",
    "build_data_up_down_debug_table",
    "generate_single_run_tables",
]
