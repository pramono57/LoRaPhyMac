from types import SimpleNamespace

import pytest
import simpy

from loraphymac.mac.frame_defs import MType
from loraphymac.metrics.statistics import Statistics
from loraphymac.runtime.packet import Packet
from loraphymac.runtime.relay_tx import RelayTX


def make_packet(fcnt):
    return Packet(
        src_id="N1",
        current_tx="R1",
        next_hop="GW",
        path=["N1", "R1", "GW"],
        hop_index=1,
        ch_id=1,
        freq_mhz=921.6,
        sf=7,
        bw_hz=125_000,
        ptx_dbm=20,
        payload_size=62,
        toa=0.1,
        tx_start=0.0,
        tx_end=0.1,
        mtype=MType.DATA_UP_FWD,
        fcnt=fcnt,
    )


def make_relay_tx(capacity=None):
    env = simpy.Environment()
    relay = SimpleNamespace(
        id="R1",
        hw=SimpleNamespace(
            tx_phy_radio_count=1,
            tx_full_duplex=False,
        ),
    )
    stats = Statistics()
    tx = RelayTX(
        env=env,
        relay=relay,
        air_interface=SimpleNamespace(),
        statistics=stats,
        relay_sf=7,
        relay_bw_hz=125_000,
        relay_ch_ids=[1],
        channel_plan=SimpleNamespace(),
        relay_duty_cycle=None,
        tx_waiting_queue_capacity_packets=capacity,
    )
    return env, relay, stats, tx


def test_time_weighted_queue_state_tracks_waiting_service_and_system():
    stats = Statistics()
    stats.record_tx_queue_state("R1", 0, 0, 0, "start")
    stats.record_tx_queue_state("R1", 2, 2, 1, "enqueue")
    stats.record_tx_queue_state("R1", 5, 1, 1, "dequeue")
    stats.record_tx_queue_state("R1", 10, 0, 0, "end")

    metrics = stats._get_tx_queue_state_metrics("R1", 10)

    assert metrics["avg_waiting_queue_length_lq"] == pytest.approx(1.1)
    assert metrics["avg_in_service_count"] == pytest.approx(0.8)
    assert metrics["avg_system_count_l"] == pytest.approx(1.9)
    assert metrics["max_waiting_queue_length"] == 2
    assert metrics["max_in_service_count"] == 1
    assert metrics["max_system_count"] == 3


def test_finite_waiting_queue_drops_tail_without_blocking_relay_rx():
    _, _, stats, tx = make_relay_tx(capacity=1)
    first = make_packet(1)
    overflow = make_packet(2)

    assert tx.put(first) is True
    assert tx.put(overflow) is False

    assert len(tx.tx_waiting_queue.items) == 1
    assert overflow.reason == "forwarding_queue_overflow"
    assert stats.relay_stats["R1"]["tx_queue_enqueue_attempt_count"] == 2
    assert stats.relay_stats["R1"]["tx_queue_accepted_count"] == 1
    assert stats.relay_stats["R1"]["tx_queue_overflow_drop_count"] == 1
    lifecycle = stats.get_tx_queue_lifecycle_df()
    assert lifecycle["status"].tolist() == ["OVERFLOW_DROP"]


def test_unbounded_put_preserves_immediate_nonblocking_acceptance():
    _, _, stats, tx = make_relay_tx(capacity=None)

    for fcnt in range(100):
        assert tx.put(make_packet(fcnt)) is True

    assert len(tx.tx_waiting_queue.items) == 100
    assert stats.relay_stats["R1"]["tx_queue_accepted_count"] == 100
    assert stats.relay_stats["R1"]["tx_queue_overflow_drop_count"] == 0


def test_finalize_observation_is_idempotent_and_conserves_accepted_visits():
    env, relay, stats, tx = make_relay_tx(capacity=None)
    waiting = make_packet(1)
    in_service = make_packet(2)
    assert tx.put(waiting) is True
    assert tx.put(in_service) is True

    tx.tx_waiting_queue.items.remove(in_service)
    in_service.tx_queue_dequeue_time = 0.0
    in_service.tx_worker_id = 0
    tx.in_service_packets[0] = in_service
    tx._record_queue_state("test_dequeue")
    env.run(until=1.0)

    tx.finalize_observation(1.0)
    tx.finalize_observation(1.0)

    lifecycle = stats.get_tx_queue_lifecycle_df()
    assert lifecycle["status"].value_counts().to_dict() == {
        "CENSORED_WAITING": 1,
        "CENSORED_IN_SERVICE": 1,
    }
    waiting_row = lifecycle[
        lifecycle["status"] == "CENSORED_WAITING"
    ].iloc[0]
    assert waiting_row["tx_start"] is None
    assert waiting_row["time_to_tx_start_s"] is None

    metrics = stats.get_relay_tx_queue_metrics_df([relay], sim_time_s=1.0)
    row = metrics.iloc[0]
    assert row["accepted_count"] == 2
    assert row["waiting_end_count"] == 1
    assert row["in_service_end_count"] == 1
    assert row["accepted_conservation_delta"] == 0
    assert row["attempt_conservation_delta"] == 0
