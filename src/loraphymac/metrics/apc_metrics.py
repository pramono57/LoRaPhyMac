"""Read-only APC, PTx, and link-margin tables for the Statistics facade."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd


PTX_HISTORY_COLUMNS = (
    "run_id",
    "time",
    "entity_id",
    "entity_type",
    "old_ptx",
    "new_ptx",
    "old_tx_power_index",
    "new_tx_power_index",
    "tx_power_index",
    "margin",
    "rssi",
    "snr",
    "decision_margin",
    "decision_rssi",
    "decision_snr",
    "apc_rx_margin",
    "apc_rx_rssi",
    "apc_rx_snr",
    "feedback_from",
    "src_id",
    "hop_index",
    "apply_source",
    "control_mtype",
)


def build_ptx_history_df(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Build the physical-PTx and TxPowerIndex update history."""
    if not records:
        return pd.DataFrame(columns=PTX_HISTORY_COLUMNS)
    return pd.DataFrame(records)


def build_apc_control_history_df(
    records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Build the physical APC request/answer control history."""
    return pd.DataFrame(records)


def build_margin_history_df(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Build the observed link-margin sample history."""
    return pd.DataFrame(records)


def build_final_ptx_distribution_df(
    nodes: Sequence[Any],
    run_metadata: Mapping[str, Any],
    infer_entity_type: Callable[[str], str],
) -> pd.DataFrame:
    """Build final dBm, TxPowerIndex, and APC/decision margin values."""
    rows = []
    for node in nodes:
        row = (
            {
                "run_id": run_metadata.get("run_id"),
                "entity_id": node.id,
                "entity_type": infer_entity_type(node.id),
                "ptx_mode": getattr(node, "ptx_mode", None),
                # Physical dBm domain.
                "final_ptx": getattr(node, "default_ptx", None),
                # APC TxPowerIndex domain.
                "final_tx_power_index": getattr(
                    node,
                    "apc_tx_power_index",
                    None,
                ),
                # Backward-compatible aliases now represent the explicit
                # decision/uplink observations when those are available.
                "last_margin": getattr(
                    node,
                    "last_decision_margin",
                    getattr(node, "last_margin", None),
                ),
                "last_rssi": getattr(
                    node,
                    "last_decision_rssi",
                    getattr(node, "last_rssi", None),
                ),
                "last_snr": getattr(
                    node,
                    "last_decision_snr",
                    getattr(node, "last_snr", None),
                ),
                # Explicit decision/uplink observations.
                "last_decision_margin": getattr(
                    node,
                    "last_decision_margin",
                    None,
                ),
                "last_decision_rssi": getattr(
                    node,
                    "last_decision_rssi",
                    None,
                ),
                "last_decision_snr": getattr(
                    node,
                    "last_decision_snr",
                    None,
                ),
                # Explicit APC physical-downlink reception observations.
                "last_apc_rx_margin": getattr(
                    node,
                    "last_apc_rx_margin",
                    None,
                ),
                "last_apc_rx_rssi": getattr(
                    node,
                    "last_apc_rx_rssi",
                    None,
                ),
                "last_apc_rx_snr": getattr(
                    node,
                    "last_apc_rx_snr",
                    None,
                ),
                "last_feedback_from": getattr(
                    node,
                    "last_feedback_from",
                    None,
                ),
                "last_feedback_time": getattr(
                    node,
                    "last_feedback_time",
                    None,
                ),
            }
        )
        if run_metadata.get("tx_power_model_id") is not None:
            row.update(
                {
                    "tx_power_model_id": getattr(node, "tx_power_model_id", None),
                    "tx_power_policy_id": getattr(node, "tx_power_policy_id", None),
                    "calibration_id": getattr(node, "tx_power_calibration_id", None),
                    "calibration_manifest_sha256": getattr(
                        node, "tx_power_calibration_manifest_sha256", None
                    ),
                    "calibration_margin_db": getattr(
                        node, "tx_power_calibration_margin_db", None
                    ),
                    "selected_parent_id": getattr(node, "tx_power_parent_id", None),
                    "selected_parent_distance_m": getattr(
                        node, "tx_power_parent_distance_m", None
                    ),
                    "initial_ptx": getattr(node, "tx_power_initial_ptx_dbm", None),
                    "selected_at_max_ptx": getattr(
                        node, "tx_power_selected_at_max_ptx", None
                    ),
                    "outside_estimated_coverage": getattr(
                        node, "tx_power_outside_estimated_coverage", None
                    ),
                    "power_control_feedback_enabled": getattr(
                        node, "tx_power_feedback_enabled", None
                    ),
                }
            )
            if getattr(node, "tx_power_estimator_id", None) is not None:
                row.update(
                    {
                        "estimator_id": node.tx_power_estimator_id,
                        "estimator_frequency_mhz": (
                            node.tx_power_estimator_frequency_mhz
                        ),
                        "estimated_path_loss_db": (
                            node.tx_power_estimated_path_loss_db
                        ),
                        "receiver_sensitivity_dbm": (
                            node.tx_power_receiver_sensitivity_dbm
                        ),
                        "estimated_rssi_dbm": node.tx_power_estimated_rssi_dbm,
                        "estimated_margin_db": node.tx_power_estimated_margin_db,
                        "policy_margin_db": node.tx_power_policy_margin_db,
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)
