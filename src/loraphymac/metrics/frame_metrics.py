"""Read-only frame/event result tables for the :class:`Statistics` facade."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from loraphymac.mac.frame_defs import CID, MType


FRAME_EVENT_REQUIRED_COLUMNS = [
    "event",
    "mtype",
    "src",
    "dst_id",
    "current_tx",
    "next_hop",
    "hop_index",
    "hop_limit",
    "hop_count",
    "path",
    "fctrl",
    "fcnt",
    "fport",
    "fopts_len",
    "control_cid",
    "decoded",
    "reason",
    "time",
]

FRAME_EVENT_NUMERIC_COLUMNS = [
    "mtype",
    "hop_index",
    "hop_limit",
    "hop_count",
    "fctrl",
    "fcnt",
    "fport",
    "fopts_len",
    "control_cid",
    "time",
]


def normalize_frame_event_df(event_log: pd.DataFrame) -> pd.DataFrame:
    """Add the stable frame fields and coerce numeric event values."""
    if event_log.empty:
        return pd.DataFrame()

    frame = event_log.copy()
    for column in FRAME_EVENT_REQUIRED_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    for column in FRAME_EVENT_NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def _count_mtype(frame: pd.DataFrame, mtype: MType) -> int:
    mask = frame["mtype"] == int(mtype)
    return _weighted_count(frame, mask)


def _event_weights(frame: pd.DataFrame) -> pd.Series:
    if "event_count" not in frame.columns:
        return pd.Series(1, index=frame.index, dtype="int64")
    return pd.to_numeric(frame["event_count"], errors="raise").astype("int64")


def _weighted_count(frame: pd.DataFrame, mask=None) -> int:
    weights = _event_weights(frame)
    if mask is not None:
        weights = weights.loc[mask]
    return int(weights.sum())


def _percent(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) * 100.0 if denominator > 0 else 0.0


def build_frame_summary_df(
    frame_events: pd.DataFrame,
    run_metadata: Mapping[str, Any],
    ptx_update_count: int,
    delivered_count: int,
) -> pd.DataFrame:
    """Build the general LoRaPhyMAC frame and physical APC summary."""
    if frame_events.empty:
        return pd.DataFrame()

    frame = frame_events.copy()
    frame["mtype"] = pd.to_numeric(frame["mtype"], errors="coerce").fillna(-1).astype(int)
    frame["hop_index"] = pd.to_numeric(frame["hop_index"], errors="coerce").fillna(-1).astype(int)
    frame["hop_count"] = pd.to_numeric(frame["hop_count"], errors="coerce")
    frame["fctrl"] = pd.to_numeric(frame["fctrl"], errors="coerce").fillna(0).astype(int)
    frame["fopts_len"] = pd.to_numeric(frame["fopts_len"], errors="coerce").fillna(0).astype(int)
    frame["control_cid"] = pd.to_numeric(frame["control_cid"], errors="coerce")

    tx_frame = frame[frame["event"] == "TX"]
    rx_frame = frame[frame["event"] == "RX"]
    drop_frame = frame[frame["event"] == "DROP"]

    total_data_up = _count_mtype(frame, MType.DATA_UP)
    total_data_up_tx = _count_mtype(tx_frame, MType.DATA_UP)
    total_data_up_rx = _count_mtype(rx_frame, MType.DATA_UP)
    total_data_up_drop = _count_mtype(drop_frame, MType.DATA_UP)

    total_data_up_fwd = _count_mtype(frame, MType.DATA_UP_FWD)
    total_data_up_fwd_tx = _count_mtype(tx_frame, MType.DATA_UP_FWD)
    total_data_up_fwd_rx = _count_mtype(rx_frame, MType.DATA_UP_FWD)
    total_data_up_fwd_drop = _count_mtype(drop_frame, MType.DATA_UP_FWD)

    total_data_down = _count_mtype(frame, MType.DATA_DOWN)
    total_data_down_tx = _count_mtype(tx_frame, MType.DATA_DOWN)
    total_data_down_rx = _count_mtype(rx_frame, MType.DATA_DOWN)
    total_data_down_drop = _count_mtype(drop_frame, MType.DATA_DOWN)

    total_data_down_fwd = _count_mtype(frame, MType.DATA_DOWN_FWD)
    total_data_down_fwd_tx = _count_mtype(tx_frame, MType.DATA_DOWN_FWD)
    total_data_down_fwd_rx = _count_mtype(rx_frame, MType.DATA_DOWN_FWD)
    total_data_down_fwd_drop = _count_mtype(drop_frame, MType.DATA_DOWN_FWD)

    total_ack = _count_mtype(frame, MType.ACK)
    total_ack_tx = _count_mtype(tx_frame, MType.ACK)
    total_ack_rx = _count_mtype(rx_frame, MType.ACK)
    total_ack_drop = _count_mtype(drop_frame, MType.ACK)

    downlink = frame["mtype"].isin(
        (int(MType.DATA_DOWN), int(MType.DATA_DOWN_FWD))
    )
    # ``DATA_DOWN`` is a generic physical carrier shared by R-APC and Wang.
    # Count only wire-confirmed APC_REQ frames as R-APC. Requiring both the
    # encoded CID and APC bit excludes Wang and malformed/ambiguous controls
    # without changing generic downlink, airtime, collision, or energy totals.
    rapc = frame.loc[
        downlink
        & (frame["control_cid"] == int(CID.APC_REQ))
        & ((frame["fctrl"] & 0x80) != 0)
    ]
    rapc_tx = rapc.loc[rapc["event"] == "TX"]
    rapc_rx = rapc.loc[rapc["event"] == "RX"]
    rapc_drop = rapc.loc[rapc["event"] == "DROP"]
    apc_total_tx = _weighted_count(rapc_tx)
    apc_total_rx = _weighted_count(rapc_rx)
    apc_total_drop = _weighted_count(rapc_drop)
    apc_apply_success_percent = _percent(apc_total_rx, apc_total_tx)
    apc_down_node_success_percent = _percent(
        _count_mtype(rapc_rx, MType.DATA_DOWN),
        _count_mtype(rapc_tx, MType.DATA_DOWN),
    )
    apc_down_relay_success_percent = _percent(
        _count_mtype(rapc_rx, MType.DATA_DOWN_FWD),
        _count_mtype(rapc_tx, MType.DATA_DOWN_FWD),
    )

    original_tx_frame = tx_frame[
        (tx_frame["mtype"] == int(MType.DATA_UP)) & (tx_frame["hop_index"] == 0)
    ]
    original_tx_count = _weighted_count(original_tx_frame)
    frame_pdr_percent = _percent(delivered_count, original_tx_count)

    fctrl_series = frame["fctrl"].fillna(0).astype(int)
    fopts_len_series = frame["fopts_len"].fillna(0).astype(int)
    event_weights = _event_weights(frame)
    fctrl_nonzero_count = int(event_weights.loc[fctrl_series != 0].sum())
    fopts_nonzero_count = int(event_weights.loc[fopts_len_series != 0].sum())
    apc_flag_count = int(event_weights.loc[(fctrl_series & 0x80) != 0].sum())

    apc_downlink_mode = str(run_metadata.get("apc_downlink_mode", "physical_all")).lower().strip()
    node_ptx_mode = str(run_metadata.get("node_ptx_mode", "fixed")).lower().strip()
    relay_ptx_mode = str(run_metadata.get("relay_ptx_mode", "fixed")).lower().strip()
    frame_enabled = bool(run_metadata.get("frame_enabled", True))
    adaptive_enabled = node_ptx_mode == "adaptive" or relay_ptx_mode == "adaptive"
    wang_a2only = run_metadata.get("tx_power_model_id") == "wang_atp_sf7_a2only"

    if apc_downlink_mode != "physical_all":
        stage1_status = "CHECK_LEGACY_APC_MODE"
    elif adaptive_enabled and not frame_enabled:
        stage1_status = "CHECK_FRAME_DISABLED"
    elif adaptive_enabled and apc_total_tx == 0:
        stage1_status = "CHECK_ADAPTIVE_WITHOUT_APC_DOWNLINK_TX"
    elif ptx_update_count > 0 and apc_total_rx == 0 and not wang_a2only:
        stage1_status = "CHECK_PTX_UPDATE_WITHOUT_APC_RX"
    elif apc_total_tx > 0 and apc_total_rx == 0:
        stage1_status = "CHECK_APC_TX_WITHOUT_APC_RX"
    else:
        stage1_status = "PASS"

    original_weights = _event_weights(original_tx_frame)
    single_hop_original_tx = int(
        original_weights.loc[original_tx_frame["hop_count"] == 1].sum()
    )
    multi_hop_original_tx = int(
        original_weights.loc[original_tx_frame["hop_count"] > 1].sum()
    )
    valid_hop_count = original_tx_frame["hop_count"].notna()
    valid_hop_weight = original_weights.loc[valid_hop_count].sum()
    avg_hop_count = (
        round(
            float(
                (original_tx_frame.loc[valid_hop_count, "hop_count"]
                 * original_weights.loc[valid_hop_count]).sum()
                / valid_hop_weight
            ),
            3,
        )
        if valid_hop_weight > 0
        else 0.0
    )
    max_hop_count = (
        int(original_tx_frame["hop_count"].dropna().max())
        if original_tx_frame["hop_count"].notna().any()
        else 0
    )

    return pd.DataFrame(
        [
            {
                "run_id": run_metadata.get("run_id"),
                "apc_downlink_mode": apc_downlink_mode,
                "total_event": _weighted_count(frame),
                "total_tx_event": _weighted_count(tx_frame),
                "total_rx_event": _weighted_count(rx_frame),
                "total_drop_event": _weighted_count(drop_frame),
                "total_original_data_up_tx": original_tx_count,
                "total_data_up": total_data_up,
                "total_data_up_tx": total_data_up_tx,
                "total_data_up_rx": total_data_up_rx,
                "total_data_up_drop": total_data_up_drop,
                "total_data_up_fwd": total_data_up_fwd,
                "total_data_up_fwd_tx": total_data_up_fwd_tx,
                "total_data_up_fwd_rx": total_data_up_fwd_rx,
                "total_data_up_fwd_drop": total_data_up_fwd_drop,
                "total_data_down": total_data_down,
                "total_data_down_tx": total_data_down_tx,
                "total_data_down_rx": total_data_down_rx,
                "total_data_down_drop": total_data_down_drop,
                "total_data_down_fwd": total_data_down_fwd,
                "total_data_down_fwd_tx": total_data_down_fwd_tx,
                "total_data_down_fwd_rx": total_data_down_fwd_rx,
                "total_data_down_fwd_drop": total_data_down_fwd_drop,
                "apc_total_tx": apc_total_tx,
                "apc_total_rx": apc_total_rx,
                "apc_total_drop": apc_total_drop,
                "apc_apply_success_percent": round(apc_apply_success_percent, 3),
                "apc_down_node_success_percent": round(
                    apc_down_node_success_percent,
                    3,
                ),
                "apc_down_relay_success_percent": round(
                    apc_down_relay_success_percent,
                    3,
                ),
                "total_ack": total_ack,
                "total_ack_tx": total_ack_tx,
                "total_ack_rx": total_ack_rx,
                "total_ack_drop": total_ack_drop,
                "total_delivered_to_gateway": delivered_count,
                "frame_pdr_percent": round(frame_pdr_percent, 3),
                "single_hop_original_tx": single_hop_original_tx,
                "multi_hop_original_tx": multi_hop_original_tx,
                "avg_hop_count": avg_hop_count,
                "max_hop_count": max_hop_count,
                "fctrl_nonzero_count": fctrl_nonzero_count,
                "fopts_nonzero_count": fopts_nonzero_count,
                "apc_flag_count": apc_flag_count,
                "ptx_update_count": ptx_update_count,
                "stage1_status": stage1_status,
            }
        ]
    )


def build_frame_mtype_summary_df(frame_events: pd.DataFrame) -> pd.DataFrame:
    """Build the event distribution by LoRaPhyMAC MType."""
    if frame_events.empty:
        return pd.DataFrame(columns=["mtype", "mtype_name", "event", "count"])

    mtype_name_map = {
        int(MType.HELLO): "HELLO",
        int(MType.DATA_UP): "DATA_UP",
        int(MType.DATA_UP_FWD): "DATA_UP_FWD",
        int(MType.DATA_DOWN): "DATA_DOWN",
        int(MType.DATA_DOWN_FWD): "DATA_DOWN_FWD",
        int(MType.ACK): "ACK",
        int(MType.RFU_6): "RFU_6",
        int(MType.RFU_7): "RFU_7",
    }
    if "event_count" in frame_events:
        result = (
            frame_events.groupby(["mtype", "event"], dropna=False)["event_count"]
            .sum()
            .reset_index(name="count")
        )
    else:
        result = frame_events.groupby(["mtype", "event"], dropna=False).size().reset_index(name="count")
    result["mtype_name"] = result["mtype"].map(mtype_name_map).fillna("UNKNOWN")
    return (
        result[["mtype", "mtype_name", "event", "count"]]
        .sort_values(["mtype", "event"])
        .reset_index(drop=True)
    )


def build_frame_hop_event_summary_df(frame_events: pd.DataFrame) -> pd.DataFrame:
    """Build TX/RX/drop frame counts grouped by end-to-end hop count."""
    if frame_events.empty:
        return pd.DataFrame()

    rows = []
    weights = _event_weights(frame_events)
    for hop_count, sub in frame_events.groupby("hop_count", dropna=True):
        sub_weights = weights.loc[sub.index]
        tx_sub = sub[sub["event"] == "TX"]
        rx_sub = sub[sub["event"] == "RX"]
        drop_sub = sub[sub["event"] == "DROP"]
        drop_weights = sub_weights.loc[drop_sub.index]
        original_tx = tx_sub[(tx_sub["mtype"] == int(MType.DATA_UP)) & (tx_sub["hop_index"] == 0)]
        drop_reasons = drop_sub["reason"].astype(str)

        rows.append(
            {
                "hop_count": int(hop_count),
                "event_count": _weighted_count(sub),
                "tx_event": _weighted_count(tx_sub),
                "rx_event": _weighted_count(rx_sub),
                "drop_event": _weighted_count(drop_sub),
                "original_data_up_tx": _weighted_count(original_tx),
                "data_up_tx": _weighted_count(tx_sub, tx_sub["mtype"] == int(MType.DATA_UP)),
                "data_up_fwd_tx": _weighted_count(tx_sub, tx_sub["mtype"] == int(MType.DATA_UP_FWD)),
                "data_down_tx": _weighted_count(tx_sub, tx_sub["mtype"] == int(MType.DATA_DOWN)),
                "data_down_fwd_tx": _weighted_count(tx_sub, tx_sub["mtype"] == int(MType.DATA_DOWN_FWD)),
                "ack_tx": _weighted_count(tx_sub, tx_sub["mtype"] == int(MType.ACK)),
                "collision_drop": int(drop_weights.loc[drop_reasons == "collision"].sum()),
                "half_duplex_drop": int(drop_weights.loc[
                    drop_reasons.isin(
                        [
                            "half_duplex_tx_busy",
                            "half_duplex_rx_interrupted_by_tx",
                        ]
                    )].sum()),
                "not_feasible_drop": int(drop_weights.loc[drop_reasons.str.contains("not_feasible", na=False)].sum()),
                "demod_drop": int(drop_weights.loc[drop_reasons == "demod_busy"].sum()),
            }
        )

    return pd.DataFrame(rows).sort_values("hop_count").reset_index(drop=True)


def build_frame_drop_reason_by_hop_df(frame_events: pd.DataFrame) -> pd.DataFrame:
    """Build drop-reason counts grouped by end-to-end hop count."""
    empty_columns = ["hop_count", "reason", "drop_count"]
    if frame_events.empty:
        return pd.DataFrame(columns=empty_columns)

    drop_frame = frame_events[frame_events["event"] == "DROP"].copy()
    if drop_frame.empty:
        return pd.DataFrame(columns=empty_columns)

    if "event_count" in drop_frame:
        grouped = (
            drop_frame.groupby(["hop_count", "reason"], dropna=False)["event_count"]
            .sum()
            .reset_index(name="drop_count")
        )
    else:
        grouped = drop_frame.groupby(["hop_count", "reason"], dropna=False).size().reset_index(name="drop_count")
    return (
        grouped
        .sort_values(["hop_count", "drop_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
