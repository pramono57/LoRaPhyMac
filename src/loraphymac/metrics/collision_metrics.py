"""Read-only collision/capture tables for the :class:`Statistics` facade."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


COLLISION_DECISION_COLUMNS = [
    "collision_model_version",
    "spectral_model_version",
    "collision_mode",
    "sf_interference_mode",
    "sir_matrix_version",
    "resolution_id",
    "receiver_id",
    "decision_time",
    "packet_object_id",
    "packet_id",
    "packet_uid",
    "src_id",
    "fcnt",
    "current_tx",
    "prev_hop",
    "next_hop",
    "hop_index",
    "mtype",
    "ch_id",
    "freq_mhz",
    "sf",
    "bw_hz",
    "rx_start",
    "rx_end",
    "rssi_dbm",
    "outcome",
    "survived",
    "captured",
    "demod_admitted",
    "demod_reason",
    "demod_pool",
    "demod_acquire_time",
    "demod_release_time",
    "direct_interferer_count",
    "contender_count",
    "strongest_interferer_packet_id",
    "strongest_interferer_src_id",
    "strongest_interferer_rssi_dbm",
    "capture_delta_db",
    "dominant_interferer_packet_id",
    "dominant_interferer_src_id",
    "dominant_interferer_rssi_dbm",
    "dominant_effective_interferer_rssi_dbm",
    "capture_threshold_db",
    "required_sir_db",
    "min_pair_margin_db",
    "max_pair_overlap_s",
    "total_pair_overlap_s",
    "enforce_sf_match",
    "destructive",
    "sinr_db",
    "reason",
]

COLLISION_INTERFERENCE_EDGE_COLUMNS = [
    "collision_model_version",
    "spectral_model_version",
    "collision_mode",
    "sf_interference_mode",
    "sir_matrix_version",
    "resolution_id",
    "receiver_id",
    "decision_time",
    "desired_packet_object_id",
    "desired_packet_id",
    "desired_packet_uid",
    "desired_src_id",
    "desired_fcnt",
    "desired_current_tx",
    "desired_prev_hop",
    "desired_next_hop",
    "desired_hop_index",
    "desired_mtype",
    "desired_ch_id",
    "desired_freq_mhz",
    "desired_sf",
    "desired_bw_hz",
    "desired_rssi_dbm",
    "desired_rx_start",
    "desired_rx_end",
    "interferer_packet_object_id",
    "interferer_packet_id",
    "interferer_packet_uid",
    "interferer_src_id",
    "interferer_fcnt",
    "interferer_current_tx",
    "interferer_prev_hop",
    "interferer_next_hop",
    "interferer_hop_index",
    "interferer_mtype",
    "interferer_ch_id",
    "interferer_freq_mhz",
    "interferer_sf",
    "interferer_bw_hz",
    "interferer_rssi_dbm",
    "interferer_rx_start",
    "interferer_rx_end",
    "time_overlap_s",
    "spectral_overlap_hz",
    "spectral_overlap_fraction",
    "effective_interferer_rssi_dbm",
    "required_sir_db",
    "pair_sir_db",
    "pair_margin_db",
    "pair_pass",
    "dominant_interferer",
]

COLLISION_SUMMARY_COLUMNS = [
    "collision_model_version",
    "spectral_model_version",
    "collision_mode",
    "sf_interference_mode",
    "sir_matrix_version",
    "receiver_id",
    "hop_index",
    "mtype",
    "ch_id",
    "sf",
    "decision_count",
    "resolution_count",
    "collision_opportunity_resolution_count",
    "singleton_success_count",
    "collision_opportunity_count",
    "capture_success_count",
    "sinr_success_count",
    "contended_success_count",
    "collision_drop_count",
    "directed_interference_edge_count",
    "overlap_pair_equivalent_count",
    "capture_success_rate_percent",
    "contended_success_rate_percent",
    "collision_loss_rate_percent",
    "avg_contender_count",
    "p95_contender_count",
    "avg_capture_delta_db",
    "p05_capture_delta_db",
    "p50_capture_delta_db",
    "avg_min_pair_margin_db",
    "p05_min_pair_margin_db",
    "avg_max_pair_overlap_s",
    "max_pair_overlap_s",
    "outcome_conservation_delta",
]

COLLISION_GROUP_COLUMNS = [
    "collision_model_version",
    "spectral_model_version",
    "collision_mode",
    "sf_interference_mode",
    "sir_matrix_version",
    "receiver_id",
    "hop_index",
    "mtype",
    "ch_id",
    "sf",
]


def _stable_mean(values: pd.Series) -> float:
    """Return an order-independent mean for profile-equivalent event sets."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return math.fsum(sorted(float(value) for value in numeric)) / len(numeric)


def _schema_complete_frame(
    records: Sequence[Mapping[str, Any]],
    required_columns: list[str],
    sort_columns: list[str],
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=required_columns)

    frame = pd.DataFrame(records)
    for column in required_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    extras = [column for column in frame.columns if column not in required_columns]
    return (
        frame[required_columns + extras]
        .sort_values(
            sort_columns,
            kind="stable",
            na_position="last",
        )
        .reset_index(drop=True)
    )


def build_collision_interference_edges_df(
    records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Return schema-complete pair-level RF interference evidence."""
    return _schema_complete_frame(
        records,
        COLLISION_INTERFERENCE_EDGE_COLUMNS,
        [
            "decision_time",
            "receiver_id",
            "resolution_id",
            "desired_packet_object_id",
            "interferer_packet_object_id",
        ],
    )


def build_collision_event_log_df(
    records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Return stable, schema-complete packet-level collision evidence."""
    return _schema_complete_frame(
        records,
        COLLISION_DECISION_COLUMNS,
        ["decision_time", "receiver_id", "resolution_id", "packet_object_id"],
    )


def build_collision_summary_df(decisions: pd.DataFrame) -> pd.DataFrame:
    """Build decision-centric collision/capture metrics for paper analysis."""
    if decisions.empty:
        return pd.DataFrame(columns=COLLISION_SUMMARY_COLUMNS)

    rows = []
    for keys, sub in decisions.groupby(
        COLLISION_GROUP_COLUMNS,
        dropna=False,
        sort=False,
    ):
        outcome = sub["outcome"].astype(str)
        direct = pd.to_numeric(
            sub["direct_interferer_count"],
            errors="coerce",
        ).fillna(0.0)
        contenders = pd.to_numeric(sub["contender_count"], errors="coerce").dropna()
        capture_delta = pd.to_numeric(
            sub["capture_delta_db"],
            errors="coerce",
        ).dropna()
        pair_margin = pd.to_numeric(
            sub["min_pair_margin_db"],
            errors="coerce",
        ).dropna()
        overlap = pd.to_numeric(
            sub["max_pair_overlap_s"],
            errors="coerce",
        ).dropna()
        contended_mask = direct > 0

        singleton_count = int((outcome == "SINGLETON_SURVIVOR").sum())
        capture_count = int((outcome == "CAPTURE_SURVIVOR").sum())
        sinr_count = int((outcome == "SINR_SURVIVOR").sum())
        collision_drop_count = int((outcome == "COLLISION_DROP").sum())
        contended_count = int(contended_mask.sum())
        survived = sub["survived"].fillna(False).astype(bool)
        contended_success_count = int((survived & contended_mask).sum())
        capture_denominator = capture_count + collision_drop_count

        row = dict(zip(COLLISION_GROUP_COLUMNS, keys))
        row.update(
            {
                "decision_count": int(len(sub)),
                "resolution_count": int(sub["resolution_id"].nunique()),
                "collision_opportunity_resolution_count": int(
                    sub.loc[contended_mask, "resolution_id"].nunique()
                ),
                "singleton_success_count": singleton_count,
                "collision_opportunity_count": contended_count,
                "capture_success_count": capture_count,
                "sinr_success_count": sinr_count,
                "contended_success_count": contended_success_count,
                "collision_drop_count": collision_drop_count,
                "directed_interference_edge_count": int(direct.sum()),
                "overlap_pair_equivalent_count": float(direct.sum()) / 2.0,
                "capture_success_rate_percent": (
                    100.0 * capture_count / capture_denominator if capture_denominator > 0 else 0.0
                ),
                "contended_success_rate_percent": (
                    100.0 * contended_success_count / contended_count
                    if contended_count > 0
                    else 0.0
                ),
                "collision_loss_rate_percent": (
                    100.0 * collision_drop_count / contended_count if contended_count > 0 else 0.0
                ),
                "avg_contender_count": (
                    _stable_mean(contenders) if not contenders.empty else 0.0
                ),
                "p95_contender_count": (
                    float(contenders.quantile(0.95)) if not contenders.empty else 0.0
                ),
                "avg_capture_delta_db": (
                    _stable_mean(capture_delta)
                    if not capture_delta.empty
                    else float("nan")
                ),
                "p05_capture_delta_db": (
                    float(capture_delta.quantile(0.05)) if not capture_delta.empty else float("nan")
                ),
                "p50_capture_delta_db": (
                    float(capture_delta.quantile(0.50)) if not capture_delta.empty else float("nan")
                ),
                "avg_min_pair_margin_db": (
                    _stable_mean(pair_margin)
                    if not pair_margin.empty
                    else float("nan")
                ),
                "p05_min_pair_margin_db": (
                    float(pair_margin.quantile(0.05)) if not pair_margin.empty else float("nan")
                ),
                "avg_max_pair_overlap_s": (
                    _stable_mean(overlap) if not overlap.empty else 0.0
                ),
                "max_pair_overlap_s": (float(overlap.max()) if not overlap.empty else 0.0),
                "outcome_conservation_delta": int(
                    len(sub) - singleton_count - capture_count - sinr_count - collision_drop_count
                ),
            }
        )
        rows.append(row)

    return (
        pd.DataFrame(rows, columns=COLLISION_SUMMARY_COLUMNS)
        .sort_values(
            ["receiver_id", "hop_index", "mtype", "ch_id", "sf"],
            kind="stable",
            na_position="last",
        )
        .reset_index(drop=True)
    )
