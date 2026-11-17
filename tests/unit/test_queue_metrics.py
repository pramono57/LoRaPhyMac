from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from loraphymac.metrics.queue_metrics import (
    TX_QUEUE_LIFECYCLE_COLUMNS,
    build_relay_tx_mgc_queue_metrics_legacy_df,
    build_relay_tx_queue_metrics_df,
    build_tx_queue_lifecycle_df,
    build_tx_queue_traffic_class_metrics_df,
    build_tx_queue_waiting_time_summary_df,
    queue_length_metrics,
    tx_queue_state_metrics,
)


def _infer_entity_type(entity_id):
    if str(entity_id).startswith("GW"):
        return "gateway"
    if str(entity_id).startswith("R"):
        return "relay"
    return "node"


def _relay(**overrides):
    values = {
        "id": "R1",
        "hw": SimpleNamespace(tx_phy_radio_count=2),
        "total_tx_time_s": 0.3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _queue_visit(visit_id, status, **overrides):
    values = {
        "queue_visit_id": visit_id,
        "entity_id": "R1",
        "entity_type": "relay",
        "traffic_class": "DATA_UP_FWD",
        "status": status,
        "queue_dequeue_time": None,
        "tx_start": None,
        "fifo_wait_s": None,
        "duty_wait_s": None,
        "half_duplex_wait_s": None,
        "airtime_s": None,
        "observed_airtime_s": 0.0,
        "worker_service_s": None,
        "time_to_tx_start_s": None,
        "system_sojourn_s": None,
    }
    values.update(overrides)
    return values


def test_lifecycle_builder_preserves_stable_schema_for_empty_and_partial_records():
    empty = build_tx_queue_lifecycle_df([])
    partial = build_tx_queue_lifecycle_df(
        [{"queue_visit_id": "q1", "entity_id": "R1", "status": "COMPLETED"}]
    )

    assert empty.columns.tolist() == TX_QUEUE_LIFECYCLE_COLUMNS
    assert partial.columns.tolist() == TX_QUEUE_LIFECYCLE_COLUMNS
    assert partial.loc[0, "queue_visit_id"] == "q1"
    assert pd.isna(partial.loc[0, "worker_id"])


def test_queue_length_is_time_weighted_over_the_observation_horizon():
    samples = [
        {"time": 2.0, "queue_length": 2},
        {"time": 5.0, "queue_length": 1},
        {"time": 10.0, "queue_length": 0},
    ]

    result = queue_length_metrics(samples, sim_time_s=10.0)

    assert result == {"avg_queue_length": 1.1, "max_queue_length": 2}


def test_queue_state_keeps_same_timestamp_event_order_and_finite_horizon_tail():
    samples = [
        {"time": 0.0, "waiting_count": 2, "in_service_count": 0, "system_count": 2},
        {"time": 0.0, "waiting_count": 1, "in_service_count": 1, "system_count": 2},
        {"time": 10.0, "waiting_count": 0, "in_service_count": 0, "system_count": 0},
    ]

    result = tx_queue_state_metrics(samples, sim_time_s=10.0)

    assert result["avg_waiting_queue_length_lq"] == 1.0
    assert result["avg_in_service_count"] == 1.0
    assert result["avg_system_count_l"] == 2.0
    assert result["max_waiting_queue_length"] == 2
    assert result["max_in_service_count"] == 1


def test_waiting_summary_preserves_percentiles_completion_and_occupancy():
    result = build_tx_queue_waiting_time_summary_df(
        waiting_times={"R1": [0.1, 0.3]},
        relay_stats={"R1": {"tx_queue_arrival_count": 4, "tx_done": 2}},
        queue_length_samples={
            "R1": [
                {"time": 0.0, "queue_length": 1},
                {"time": 5.0, "queue_length": 0},
            ]
        },
        run_metadata={"sim_time": 10.0},
    ).iloc[0]

    assert result["count"] == 2
    assert result["avg_waiting_time_wq_ms"] == 200.0
    assert result["p95_waiting_time_wq_ms"] == 290.0
    assert result["p99_waiting_time_wq_ms"] == 298.0
    assert result["avg_queue_length"] == 0.5
    assert result["tx_queue_backlog_count"] == 2
    assert result["tx_queue_completion_ratio_percent"] == 50.0


def test_queue_v2_preserves_censoring_load_and_conservation_equations():
    lifecycle = build_tx_queue_lifecycle_df(
        [
            _queue_visit(
                "q1",
                "COMPLETED",
                queue_dequeue_time=0.1,
                tx_start=0.2,
                fifo_wait_s=0.1,
                duty_wait_s=0.05,
                half_duplex_wait_s=0.05,
                airtime_s=0.2,
                observed_airtime_s=0.2,
                worker_service_s=0.3,
                time_to_tx_start_s=0.2,
                system_sojourn_s=0.4,
            ),
            _queue_visit("q2", "CENSORED_WAITING"),
            _queue_visit(
                "q3",
                "CENSORED_IN_SERVICE",
                queue_dequeue_time=0.5,
                tx_start=0.6,
                fifo_wait_s=0.2,
                observed_airtime_s=0.1,
            ),
            _queue_visit("q4", "OVERFLOW_DROP"),
        ]
    )
    relay_stats = {
        "R1": {
            "tx_queue_enqueue_attempt_count": 4,
            "tx_queue_accepted_count": 3,
            "tx_queue_overflow_drop_count": 1,
            "tx_queue_internal_drop_count": 0,
        }
    }
    state_samples = {
        "R1": [
            {"time": 0.0, "waiting_count": 2, "in_service_count": 1, "system_count": 3},
            {"time": 10.0, "waiting_count": 0, "in_service_count": 0, "system_count": 0},
        ]
    }

    result = build_relay_tx_queue_metrics_df(
        [_relay()],
        10.0,
        lifecycle,
        relay_stats,
        state_samples,
        _infer_entity_type,
    ).iloc[0]

    assert result["queue_metrics_version"] == "2.0"
    assert result["completed_count"] == 1
    assert result["waiting_end_count"] == 1
    assert result["in_service_end_count"] == 1
    assert result["attempt_conservation_delta"] == 0
    assert result["accepted_conservation_delta"] == 0
    assert result["offered_airtime_load_ratio"] == pytest.approx(0.03)
    assert result["effective_offered_load_estimate"] == pytest.approx(0.045)
    assert result["worker_occupied_fraction"] == pytest.approx(0.5)
    assert result["radio_tx_fraction"] == pytest.approx(0.015)


def test_traffic_class_and_legacy_mgc_tables_remain_available():
    lifecycle = build_tx_queue_lifecycle_df(
        [
            _queue_visit(
                "q1",
                "COMPLETED",
                queue_dequeue_time=0.1,
                fifo_wait_s=0.1,
                duty_wait_s=0.0,
                half_duplex_wait_s=0.0,
                worker_service_s=0.2,
                system_sojourn_s=0.3,
            ),
            _queue_visit("q2", "CENSORED_WAITING"),
        ]
    )
    traffic_class = build_tx_queue_traffic_class_metrics_df(
        lifecycle,
        _infer_entity_type,
    ).iloc[0]
    legacy = build_relay_tx_mgc_queue_metrics_legacy_df(
        [_relay(total_tx_time_s=2.0)],
        10.0,
        {"R1": {"tx_queue_arrival_count": 4, "tx_done": 2}},
        {"R1": [0.1, 0.3]},
        {
            "R1": [
                {"time": 0.0, "queue_length": 1},
                {"time": 5.0, "queue_length": 0},
            ]
        },
        _infer_entity_type,
    ).iloc[0]

    assert traffic_class["visit_count"] == 2
    assert traffic_class["completed_count"] == 1
    assert traffic_class["censored_waiting_count"] == 1
    assert legacy["lambda_pkt_s"] == 0.4
    assert legacy["mean_service_time_s"] == 1.0
    assert legacy["rho"] == 0.2
    assert legacy["tx_queue_completion_ratio_percent"] == 50.0
