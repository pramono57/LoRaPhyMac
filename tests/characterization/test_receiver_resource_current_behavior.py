"""Records current receiver-resource behavior; it is not necessarily ideal."""

import math

import pytest
import simpy

from loraphymac.mac.collision_engine import CollisionEngine
from loraphymac.runtime.demodulator_pool import DemodulatorPool
from tests.helpers.collision_fakes import make_collision_packet
from tests.helpers.demodulator_fakes import make_demod_packet


pytestmark = pytest.mark.characterization


def test_touching_time_intervals_currently_do_not_overlap():
    # Records current behavior; this is not a claim that the behavior is ideal.
    first = make_collision_packet("P1", rx_start=1, rx_end=2)
    second = make_collision_packet("P2", rx_start=2, rx_end=3)
    survivors, failed = CollisionEngine().resolve([first, second])
    assert survivors == [first, second]
    assert failed == []


def test_exact_capture_threshold_currently_allows_strongest_to_survive():
    # Records current behavior; this is not a claim that the behavior is ideal.
    strong = make_collision_packet("strong", rssi=-100)
    weak = make_collision_packet("weak", rssi=-106)
    survivors, failed = CollisionEngine(capture_threshold_dB=6).resolve([weak, strong])
    assert survivors == [strong]
    assert failed == [weak]


def test_equal_rssi_currently_fails_both_in_capture_mode():
    # Stable sorting preserves input order, but neither packet survives the tie.
    first = make_collision_packet("first", rssi=-100)
    second = make_collision_packet("second", rssi=-100)
    survivors, failed = CollisionEngine().resolve([second, first])
    assert survivors == []
    assert failed == [second, first]


def test_partial_bandwidth_overlap_scales_interferer_power():
    first = make_collision_packet("P1", bw_hz=125_000)
    second = make_collision_packet("P2", bw_hz=500_000)
    engine = CollisionEngine()
    survivors, failed = engine.resolve([first, second])
    assert survivors == [first]
    assert failed == [second]
    first_edge = next(
        row
        for row in engine.last_interference_edge_records
        if row["desired_packet_id"] == "P1"
    )
    assert first_edge["spectral_overlap_fraction"] == pytest.approx(0.25)


def test_capture_uses_only_direct_overlap_in_transitive_chain():
    # A and C share a connected busy period but never interfere directly.
    a = make_collision_packet("A", rx_start=0, rx_end=2, rssi=-80)
    b = make_collision_packet("B", rx_start=1, rx_end=3, rssi=-100)
    c = make_collision_packet("C", rx_start=2.5, rx_end=4, rssi=-90)
    survivors, failed = CollisionEngine().resolve([a, b, c])
    assert survivors == [a, c]
    assert failed == [b]


def test_low_sf_pool_priority_is_currently_b_then_a_fallback():
    # Records current behavior; this is not a claim that the behavior is ideal.
    env = simpy.Environment()
    pool = DemodulatorPool(env, 1, 1)
    primary = make_demod_packet("primary", sf=7)
    fallback = make_demod_packet("fallback", sf=7)
    assert pool.try_acquire(primary) and primary.demod_pool == "B"
    assert pool.try_acquire(fallback) and fallback.demod_pool == "A"


def test_high_sf_currently_uses_only_pool_a():
    env = simpy.Environment()
    pool = DemodulatorPool(env, 1, 10)
    first = make_demod_packet("P1", sf=11)
    second = make_demod_packet("P2", sf=11)
    assert pool.try_acquire(first) is True
    assert pool.try_acquire(second) is False
    assert second.demod_reason == "demod_busy"
    assert pool.used_b == 0


def test_duplicate_acquire_is_keyed_by_object_identity():
    env = simpy.Environment()
    pool = DemodulatorPool(env, 1, 2)
    first = make_demod_packet("same")
    same_id_other_object = make_demod_packet("same")
    assert pool.try_acquire(first) is True
    assert pool.try_acquire(first) is False
    assert first.demod_reason == "demod_duplicate_acquire"
    assert pool.try_acquire(same_id_other_object) is True


def test_double_release_logs_missing_release_without_negative_counter():
    env = simpy.Environment()
    pool = DemodulatorPool(env, 0, 1)
    packet = make_demod_packet(toa=0)
    pool.try_acquire(packet)
    pool.schedule_release(packet)
    pool.schedule_release(packet)
    env.run()
    assert pool.release_count == 1
    assert pool.used_b == 0
    assert pool.event_log[-1]["event"] == "RELEASE_MISSING"


def test_acquire_and_release_events_use_immutable_context_snapshot():
    env = simpy.Environment()
    pool = DemodulatorPool(env, 0, 1)
    packet = make_demod_packet("P1", sf=7, toa=0)
    pool.try_acquire(packet)
    acquire = pool.event_log[-1]
    packet.src_id = "MUTATED"
    packet.sf = 12
    packet.ch_id = 7
    pool.schedule_release(packet)
    env.run()
    release = pool.event_log[-1]
    assert acquire["src_id"] == release["src_id"] == "N1"
    assert acquire["sf"] == release["sf"] == 7
    assert acquire["ch_id"] == release["ch_id"] == 0


def test_release_deadline_is_captured_when_scheduled():
    env = simpy.Environment()
    pool = DemodulatorPool(env, 0, 1)
    packet = make_demod_packet(toa=1.0)
    pool.try_acquire(packet)
    pool.schedule_release(packet)
    packet.toa = 0.25
    env.run()
    assert packet.demod_release_time == pytest.approx(1.0, abs=1e-12)


def test_nan_rssi_is_rejected_before_collision_resolution():
    first = make_collision_packet("nan", rssi=math.nan)
    second = make_collision_packet("finite", rssi=-100)
    with pytest.raises(ValueError, match="non-finite rssi"):
        CollisionEngine().resolve([first, second])


def test_preexisting_collision_reason_is_not_cleared_when_packet_later_survives():
    # Records current mutation behavior; callers must provide fresh/reset packets.
    packet = make_collision_packet("P1", reason="collision")
    survivors, failed = CollisionEngine().resolve([packet])
    assert survivors == [packet]
    assert failed == []
    assert packet.reason == "collision"
