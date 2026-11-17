from __future__ import annotations

import pandas as pd
import pytest

from loraphymac.metrics.packet_metrics import (
    PACKET_FATE_COLUMNS,
    build_delivered_packet_df,
    build_packet_failure_df,
    build_packet_fate_df,
    build_packet_fate_summary_df,
)


def test_empty_packet_fate_table_preserves_schema():
    result = build_packet_fate_df({})

    assert result.empty
    assert list(result.columns) == PACKET_FATE_COLUMNS


def test_packet_fate_table_preserves_schema_and_deterministic_order():
    packet_fates = {
        "third": {"packet_uid": "UP:N2:1", "src_id": "N2", "fcnt": 1},
        "second": {"packet_uid": "UP:N1:2", "src_id": "N1", "fcnt": 2},
        "first": {"packet_uid": "UP:N1:1", "src_id": "N1", "fcnt": 1},
    }

    result = build_packet_fate_df(packet_fates)

    assert list(result.columns) == PACKET_FATE_COLUMNS
    assert result["packet_uid"].tolist() == ["UP:N1:1", "UP:N1:2", "UP:N2:1"]


def test_packet_fate_summary_preserves_counts_delay_and_conservation():
    fate = pd.DataFrame(
        [
            {"final_status": "DELIVERED", "e2e_delay_ms": 10.0},
            {"final_status": "DELIVERED", "e2e_delay_ms": 30.0},
            {"final_status": "TERMINAL_DROP", "e2e_delay_ms": None},
            {"final_status": "CENSORED_WAITING", "e2e_delay_ms": None},
        ]
    )

    result = build_packet_fate_summary_df(
        fate,
        {"run_id": "run-8b", "sim_time": 3600.0},
        terminal_conflict_count=2,
    ).iloc[0]

    assert result["run_id"] == "run-8b"
    assert result["packet_fate_metrics_version"] == "1.0"
    assert result["observation_end_time"] == 3600.0
    assert result["offered_count"] == 4
    assert result["delivered_count"] == 2
    assert result["terminal_drop_count"] == 1
    assert result["censored_count"] == 1
    assert result["delivery_by_horizon_percent"] == 50.0
    assert result["terminal_loss_percent"] == 25.0
    assert result["censored_fraction_percent"] == 25.0
    assert result["delivered_delay_mean_ms"] == 20.0
    assert result["delivered_delay_p50_ms"] == 20.0
    assert result["delivered_delay_p95_ms"] == pytest.approx(29.0)
    assert result["delivered_delay_p99_ms"] == pytest.approx(29.8)
    assert result["terminal_conflict_count"] == 2
    assert result["packet_fate_conservation_delta"] == 0


def test_delivered_and_failure_tables_keep_legacy_shapes():
    delivered_records = [{"packet_uid": "UP:N1:1", "delivery_status": "delivered"}]
    event_log = pd.DataFrame(
        [
            {"event": "TX", "packet_uid": "UP:N1:1", "reason": None},
            {"event": "DROP", "packet_uid": "UP:N1:2", "reason": "collision"},
            {"event": "RX", "packet_uid": "UP:N1:1", "reason": None},
        ],
        index=[4, 7, 9],
    )

    delivered = build_delivered_packet_df(delivered_records)
    failures = build_packet_failure_df(event_log)

    assert delivered.to_dict("records") == delivered_records
    assert failures.index.tolist() == [0]
    assert failures.to_dict("records") == [
        {"event": "DROP", "packet_uid": "UP:N1:2", "reason": "collision"}
    ]
