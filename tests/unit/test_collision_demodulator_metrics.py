from __future__ import annotations

import pytest

from loraphymac.metrics.collision_metrics import (
    COLLISION_DECISION_COLUMNS,
    COLLISION_INTERFERENCE_EDGE_COLUMNS,
    build_collision_event_log_df,
    build_collision_interference_edges_df,
    build_collision_summary_df,
)
from loraphymac.metrics.demodulator_metrics import (
    build_demodulator_event_log_df,
    build_demodulator_pool_summary_df,
)


def _collision_decision(
    *,
    packet_object_id: int,
    decision_time: float,
    resolution_id: str,
    outcome: str,
    survived: bool,
    direct_interferer_count: int,
    contender_count: int,
    capture_delta_db: float | None = None,
    min_pair_margin_db: float | None = None,
    max_pair_overlap_s: float | None = None,
) -> dict[str, object]:
    return {
        "collision_model_version": "m1g_v3",
        "spectral_model_version": "spectral_v1",
        "collision_mode": "capture",
        "sf_interference_mode": "matrix",
        "sir_matrix_version": "sir_v1",
        "resolution_id": resolution_id,
        "receiver_id": "GW",
        "decision_time": decision_time,
        "packet_object_id": packet_object_id,
        "hop_index": 1,
        "mtype": 1,
        "ch_id": 0,
        "sf": 7,
        "outcome": outcome,
        "survived": survived,
        "direct_interferer_count": direct_interferer_count,
        "contender_count": contender_count,
        "capture_delta_db": capture_delta_db,
        "min_pair_margin_db": min_pair_margin_db,
        "max_pair_overlap_s": max_pair_overlap_s,
    }


def test_collision_logs_are_schema_complete_and_stably_sorted():
    decisions = [
        _collision_decision(
            packet_object_id=2,
            decision_time=2.0,
            resolution_id="res-2",
            outcome="COLLISION_DROP",
            survived=False,
            direct_interferer_count=1,
            contender_count=2,
        ),
        {
            **_collision_decision(
                packet_object_id=1,
                decision_time=1.0,
                resolution_id="res-1",
                outcome="SINGLETON_SURVIVOR",
                survived=True,
                direct_interferer_count=0,
                contender_count=1,
            ),
            "custom_evidence": "retained",
        },
    ]

    result = build_collision_event_log_df(decisions)

    assert result["packet_object_id"].tolist() == [1, 2]
    assert list(result.columns[: len(COLLISION_DECISION_COLUMNS)]) == (COLLISION_DECISION_COLUMNS)
    assert result.columns[-1] == "custom_evidence"


def test_empty_collision_edge_table_preserves_pair_schema():
    result = build_collision_interference_edges_df([])

    assert result.empty
    assert list(result.columns) == COLLISION_INTERFERENCE_EDGE_COLUMNS


def test_collision_summary_preserves_capture_loss_and_conservation_metrics():
    decisions = build_collision_event_log_df(
        [
            _collision_decision(
                packet_object_id=1,
                decision_time=1.0,
                resolution_id="res-1",
                outcome="SINGLETON_SURVIVOR",
                survived=True,
                direct_interferer_count=0,
                contender_count=1,
            ),
            _collision_decision(
                packet_object_id=2,
                decision_time=2.0,
                resolution_id="res-2",
                outcome="CAPTURE_SURVIVOR",
                survived=True,
                direct_interferer_count=1,
                contender_count=2,
                capture_delta_db=8.0,
                min_pair_margin_db=2.0,
                max_pair_overlap_s=0.10,
            ),
            _collision_decision(
                packet_object_id=3,
                decision_time=3.0,
                resolution_id="res-3",
                outcome="SINR_SURVIVOR",
                survived=True,
                direct_interferer_count=1,
                contender_count=2,
                min_pair_margin_db=1.0,
                max_pair_overlap_s=0.20,
            ),
            _collision_decision(
                packet_object_id=4,
                decision_time=4.0,
                resolution_id="res-4",
                outcome="COLLISION_DROP",
                survived=False,
                direct_interferer_count=1,
                contender_count=2,
                capture_delta_db=4.0,
                min_pair_margin_db=-1.0,
                max_pair_overlap_s=0.30,
            ),
        ]
    )

    result = build_collision_summary_df(decisions).iloc[0]

    assert result["decision_count"] == 4
    assert result["resolution_count"] == 4
    assert result["collision_opportunity_resolution_count"] == 3
    assert result["singleton_success_count"] == 1
    assert result["collision_opportunity_count"] == 3
    assert result["capture_success_count"] == 1
    assert result["sinr_success_count"] == 1
    assert result["contended_success_count"] == 2
    assert result["collision_drop_count"] == 1
    assert result["directed_interference_edge_count"] == 3
    assert result["overlap_pair_equivalent_count"] == 1.5
    assert result["capture_success_rate_percent"] == 50.0
    assert result["contended_success_rate_percent"] == pytest.approx(200.0 / 3.0)
    assert result["collision_loss_rate_percent"] == pytest.approx(100.0 / 3.0)
    assert result["avg_capture_delta_db"] == 6.0
    assert result["avg_min_pair_margin_db"] == pytest.approx(2.0 / 3.0)
    assert result["avg_max_pair_overlap_s"] == pytest.approx(0.20)
    assert result["max_pair_overlap_s"] == pytest.approx(0.30)
    assert result["outcome_conservation_delta"] == 0


def test_demodulator_summary_preserves_capacity_rejection_and_occupancy():
    events = [
        {
            "time": 3.0,
            "receiver_id": "GW",
            "event": "RELEASE",
            "pool": "A",
            "used_a_after": 0,
            "used_b_after": 0,
            "used_total_after": 0,
        },
        {
            "time": 1.0,
            "receiver_id": "GW",
            "event": "ACQUIRE",
            "pool": "A",
            "capacity_a": 8,
            "capacity_b": 8,
            "capacity_total": 16,
            "used_a_after": 1,
            "used_b_after": 0,
            "used_total_after": 1,
        },
        {
            "time": 2.0,
            "receiver_id": "GW",
            "event": "REJECT",
            "reason": "demod_busy",
            "used_a_after": 1,
            "used_b_after": 0,
            "used_total_after": 1,
        },
    ]

    event_log = build_demodulator_event_log_df(events)
    result = build_demodulator_pool_summary_df(event_log).iloc[0]

    assert event_log["time"].tolist() == [1.0, 2.0, 3.0]
    assert result["capacity_total"] == 16
    assert result["acquire_attempt_count"] == 2
    assert result["acquire_success_count"] == 1
    assert result["release_count"] == 1
    assert result["reject_count"] == 1
    assert result["demod_busy_count"] == 1
    assert result["pool_a_acquire_count"] == 1
    assert result["pool_b_acquire_count"] == 0
    assert result["peak_used_total"] == 1
    assert result["used_a_at_last_event"] == 0
