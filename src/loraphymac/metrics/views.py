"""Pure DataFrame views used by console reporting."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def select_existing_columns(
    frame: pd.DataFrame | None,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Select available columns without raising on partial diagnostic data."""
    if frame is None or frame.empty:
        return pd.DataFrame()

    selected = [column for column in columns if column in frame.columns]
    return frame[selected] if selected else frame


def view_core_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return select_existing_columns(
        frame,
        [
            "run_id",
            "pdr_percent",
            "served_node_ratio_95_percent",
            "avg_e2e_delay_ms",
            "total_node_energy_J",
            "total_relay_energy_J",
            "total_gateway_energy_J",
            "total_network_energy_J",
            "total_constrained_energy_J",
            "energy_per_delivered_packet_J",
            "energy_per_delivered_packet_constrained_J",
            "avg_final_node_ptx",
            "avg_final_relay_ptx",
            "avg_final_node_decision_margin",
            "avg_final_node_apc_rx_margin",
            "avg_final_relay_decision_margin",
            "avg_final_relay_apc_rx_margin",
        ],
    )


def view_frame_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return select_existing_columns(
        frame,
        [
            "run_id",
            "total_original_data_up_tx",
            "total_data_up_tx",
            "total_data_up_fwd_tx",
            "total_delivered_to_gateway",
            "frame_pdr_percent",
            "single_hop_original_tx",
            "multi_hop_original_tx",
            "avg_hop_count",
            "max_hop_count",
            "fctrl_nonzero_count",
            "fopts_nonzero_count",
            "apc_flag_count",
            "total_data_down",
            "total_data_down_tx",
            "total_data_down_rx",
            "total_data_down_drop",
            "total_data_down_fwd",
            "total_data_down_fwd_tx",
            "total_data_down_fwd_rx",
            "total_data_down_fwd_drop",
            "apc_total_tx",
            "apc_total_rx",
            "apc_apply_success_percent",
            "ptx_update_count",
            "total_ack",
            "stage1_status",
        ],
    )


def view_relay_duty_cycle(frame: pd.DataFrame) -> pd.DataFrame:
    return select_existing_columns(
        frame,
        [
            "entity_id",
            "entity_type",
            "total_tx_time_s",
            "relay_duty_cycle_percent",
            "configured_relay_duty_cycle_percent",
            "duty_cycle_margin_percent",
            "dc_status",
        ],
    )


def view_per_channel_duty_cycle(frame: pd.DataFrame) -> pd.DataFrame:
    return select_existing_columns(
        frame,
        [
            "entity_id",
            "entity_type",
            "channel",
            "tx_packet_count",
            "total_airtime_per_channel_s",
            "per_channel_duty_cycle_percent",
            "duty_cycle_margin_percent",
            "dc_status",
        ],
    )


def view_energy_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    return select_existing_columns(
        frame,
        [
            "entity_id",
            "entity_type",
            "duplex_mode",
            "tx_full_duplex",
            "tx_time_s",
            "rx_time_s",
            "energy_tx_J",
            "energy_rx_J",
            "energy_sleep_J",
            "energy_transition_J",
            "total_energy_J",
        ],
    )


def view_final_ptx_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    return select_existing_columns(
        frame,
        [
            "entity_id",
            "entity_type",
            "ptx_mode",
            "final_ptx",
            "final_tx_power_index",
            "last_decision_margin",
            "last_apc_rx_margin",
            "last_feedback_from",
        ],
    )


def view_as_vertical(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    rows = list(frame.iloc[0].items())
    midpoint = (len(rows) + 1) // 2
    left = rows[:midpoint]
    right = rows[midpoint:]
    max_length = max(len(left), len(right))
    left += [("", "")] * (max_length - len(left))
    right += [("", "")] * (max_length - len(right))

    return pd.DataFrame(
        {
            "Metric": [item[0] for item in left],
            "Value": [item[1] for item in left],
            "Metric_": [item[0] for item in right],
            "Value_": [item[1] for item in right],
        }
    )


def view_apc_control_history(frame: pd.DataFrame) -> pd.DataFrame:
    return select_existing_columns(
        frame,
        [
            "time",
            "event",
            "sender_id",
            "receiver_id",
            "controller_id",
            "target_id",
            "target_type",
            "control_scope",
            "old_ptx",
            "new_ptx",
            "old_tx_power_index",
            "new_tx_power_index",
            "tx_power_index",
            "dr",
            "status",
            "decision_margin",
            "apc_rx_margin",
            "margin",
            "src_id",
            "hop_index",
        ],
    )


def view_frame_summary_vertical(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    row = frame.iloc[0]

    def get(key: str, default: object = 0) -> object:
        return row[key] if key in frame.columns else default

    data_down_tx = get("total_data_down_tx", 0)
    data_down_rx = get("total_data_down_rx", 0)
    data_down_fwd_tx = get("total_data_down_fwd_tx", 0)
    data_down_fwd_rx = get("total_data_down_fwd_rx", 0)

    node_down_success = data_down_rx / data_down_tx * 100.0 if data_down_tx > 0 else 0.0
    relay_down_success = (
        data_down_fwd_rx / data_down_fwd_tx * 100.0 if data_down_fwd_tx > 0 else 0.0
    )

    rows = [
        ("run_id", get("run_id", "")),
        ("stage1_status", get("stage1_status", "")),
        ("total_original_data_up_tx", int(get("total_original_data_up_tx", 0))),
        ("total_data_up_tx", int(get("total_data_up_tx", 0))),
        ("total_data_up_fwd_tx", int(get("total_data_up_fwd_tx", 0))),
        ("total_delivered_to_gateway", int(get("total_delivered_to_gateway", 0))),
        ("frame_pdr_percent", round(float(get("frame_pdr_percent", 0.0)), 3)),
        ("single_hop_original_tx", int(get("single_hop_original_tx", 0))),
        ("multi_hop_original_tx", int(get("multi_hop_original_tx", 0))),
        ("avg_hop_count", round(float(get("avg_hop_count", 0.0)), 3)),
        ("max_hop_count", int(get("max_hop_count", 0))),
        ("total_data_down_tx", int(data_down_tx)),
        ("total_data_down_rx", int(data_down_rx)),
        ("total_data_down_drop", int(get("total_data_down_drop", 0))),
        ("data_down_success_percent", round(float(node_down_success), 3)),
        ("total_data_down_fwd_tx", int(data_down_fwd_tx)),
        ("total_data_down_fwd_rx", int(data_down_fwd_rx)),
        ("total_data_down_fwd_drop", int(get("total_data_down_fwd_drop", 0))),
        ("data_down_fwd_success_percent", round(float(relay_down_success), 3)),
        ("apc_total_tx", int(get("apc_total_tx", 0))),
        ("apc_total_rx", int(get("apc_total_rx", 0))),
        (
            "apc_apply_success_percent",
            round(float(get("apc_apply_success_percent", 0.0)), 3),
        ),
        ("ptx_update_count", int(get("ptx_update_count", 0))),
        ("fctrl_nonzero_count", int(get("fctrl_nonzero_count", 0))),
        ("fopts_nonzero_count", int(get("fopts_nonzero_count", 0))),
        ("apc_flag_count", int(get("apc_flag_count", 0))),
        ("total_ack", int(get("total_ack", 0))),
    ]

    midpoint = (len(rows) + 1) // 2
    left = rows[:midpoint]
    right = rows[midpoint:]
    max_length = max(len(left), len(right))
    left += [("", "")] * (max_length - len(left))
    right += [("", "")] * (max_length - len(right))

    return pd.DataFrame(
        {
            "Metric": [item[0] for item in left],
            "Value": [item[1] for item in left],
            "Metric_": [item[0] for item in right],
            "Value_": [item[1] for item in right],
        }
    )
