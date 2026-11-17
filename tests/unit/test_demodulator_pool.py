import pytest
import simpy

from loraphymac.runtime.demodulator_pool import DemodulatorPool
from loraphymac.runtime.packet import stable_packet_object_id
from tests.helpers.demodulator_fakes import RecordingStatistics, make_demod_packet


pytestmark = pytest.mark.unit


def make_pool(a=2, b=2, stats=None):
    env = simpy.Environment()
    return env, DemodulatorPool(env, a, b, receiver_id="R1", statistics=stats)


def test_initial_capacities_counters_and_summary():
    _, pool = make_pool(3, 4)
    summary = pool.get_summary()
    assert summary["capacity_a"] == 3
    assert summary["capacity_b"] == 4
    assert summary["capacity_total"] == 7
    for field in (
        "acquire_attempt_count", "acquire_success_count", "release_count",
        "demod_busy_count", "unsupported_sf_count", "peak_used_total",
        "used_a_at_end", "used_b_at_end", "active_mapping_count_at_end",
        "active_context_count_at_end",
    ):
        assert summary[field] == 0
    assert pool.packet_pool == {}
    assert pool.packet_context == {}
    assert pool.get_event_log() == []


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [(-1, -2, (0, 0)), (0, 0, (0, 0)), (2.9, 3.9, (2, 3)), ("2", "3", (2, 3))],
)
def test_capacity_inputs_are_clamped_or_converted(a, b, expected):
    _, pool = make_pool(a, b)
    assert (pool.capacity_a, pool.capacity_b) == expected


@pytest.mark.parametrize("bad", [None, "bad", object()])
def test_nonconvertible_capacity_is_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        make_pool(bad, 1)


@pytest.mark.parametrize(
    ("sf", "expected_pool"),
    [(5, "B"), (6, "B"), (7, "B"), (8, "B"), (9, "B"), (10, "B"), (11, "A"), (12, "A")],
)
def test_sf_mapping_and_primary_pool(sf, expected_pool):
    _, pool = make_pool()
    packet = make_demod_packet(sf=sf)
    assert pool.try_acquire(packet) is True
    assert packet.demod_pool == expected_pool
    assert pool.packet_pool[id(packet)] == expected_pool


@pytest.mark.parametrize("sf", [None, 4, 13, "bad"])
def test_unsupported_or_invalid_sf_is_rejected(sf):
    _, pool = make_pool()
    packet = make_demod_packet(sf=sf)
    assert pool.try_acquire(packet) is False
    assert packet.demod_reason == "demod_unsupported_sf"
    assert pool.unsupported_sf_count == 1
    assert pool.demod_busy_count == 0
    assert pool.event_log[-1]["reason"] == "demod_unsupported_sf"


def test_low_sf_falls_back_from_full_b_to_a():
    _, pool = make_pool(a=1, b=1)
    first = make_demod_packet("B", sf=7)
    fallback = make_demod_packet("A", sf=7)
    assert pool.try_acquire(first) is True
    assert pool.try_acquire(fallback) is True
    assert first.demod_pool == "B"
    assert fallback.demod_pool == "A"
    assert (pool.used_a, pool.used_b) == (1, 1)


def test_high_sf_has_no_pool_b_fallback():
    _, pool = make_pool(a=1, b=2)
    first = make_demod_packet("P1", sf=12)
    rejected = make_demod_packet("P2", sf=12)
    assert pool.try_acquire(first) is True
    assert pool.try_acquire(rejected) is False
    assert rejected.demod_reason == "demod_busy"
    assert pool.used_b == 0


def test_both_eligible_pools_full_produces_busy_reason_and_counter():
    _, pool = make_pool(a=1, b=1)
    # Keep strong references while allocations are active: the production key
    # is id(packet), not a semantic packet identifier.
    first = make_demod_packet("P1", sf=7)
    second = make_demod_packet("P2", sf=7)
    assert pool.try_acquire(first)
    assert pool.try_acquire(second)
    rejected = make_demod_packet("P3", sf=7)
    assert pool.try_acquire(rejected) is False
    assert rejected.demod_reason == "demod_busy"
    assert pool.demod_busy_count == 1
    assert pool.acquire_attempt_count == 3
    assert pool.acquire_success_count == 2


def test_duplicate_acquire_same_object_is_rejected_without_double_allocation():
    _, pool = make_pool()
    packet = make_demod_packet()
    assert pool.try_acquire(packet) is True
    before = (pool.used_a, pool.used_b)
    assert pool.try_acquire(packet) is False
    assert packet.demod_reason == "demod_duplicate_acquire"
    assert (pool.used_a, pool.used_b) == before
    assert pool.acquire_attempt_count == 2
    assert pool.acquire_success_count == 1


def test_same_semantic_packet_id_on_different_objects_allocates_twice():
    _, pool = make_pool(a=0, b=2)
    first = make_demod_packet("SAME")
    second = make_demod_packet("SAME")
    assert pool.try_acquire(first) is True
    assert pool.try_acquire(second) is True
    assert len(pool.packet_pool) == 2
    assert pool.used_b == 2


def test_acquire_metadata_and_statistics_callback():
    stats = RecordingStatistics()
    env, pool = make_pool(stats=stats)
    packet = make_demod_packet("P1", sf=7)
    env.run(until=2.5)
    assert pool.try_acquire(packet) is True
    event = pool.event_log[-1]
    assert stats.events == pool.event_log
    assert event["event"] == "ACQUIRE"
    assert event["receiver_id"] == "R1"
    assert event["pool"] == "B"
    assert event["sf"] == 7
    assert event["packet_object_id"] == stable_packet_object_id(packet)
    assert event["time"] == pytest.approx(2.5, abs=1e-12)
    assert event["used_b_before"] == 0
    assert event["used_b_after"] == 1


def test_release_occurs_after_toa_and_restores_capacity():
    env, pool = make_pool(a=0, b=1)
    packet = make_demod_packet(toa=0.25)
    assert pool.try_acquire(packet)
    pool.schedule_release(packet)
    env.run(until=0.249)
    assert pool.used_b == 1
    env.run()
    assert pool.used_b == 0
    assert packet.demod_release_time == pytest.approx(0.25, abs=1e-12)
    assert pool.release_count == 1
    assert pool.packet_pool == {}
    assert pool.packet_context == {}


def test_explicit_release_deadline_matches_physical_rx_end():
    env, pool = make_pool(a=0, b=1)
    env.run(until=5.0)
    packet = make_demod_packet(
        toa=9.0,
        rx_start=5.0,
        rx_end=5.25,
    )
    assert pool.try_acquire(packet)
    pool.schedule_release(packet, release_at=packet.rx_end)

    # Later packet mutation must not move the already scheduled RF deadline.
    packet.toa = 100.0
    packet.rx_end = 200.0
    env.run(until=5.249)
    assert pool.used_b == 1
    env.run()

    assert packet.demod_acquire_time == pytest.approx(5.0, abs=1e-12)
    assert packet.demod_release_time == pytest.approx(5.25, abs=1e-12)
    assert pool.used_b == 0


def test_release_after_fallback_uses_correct_pool():
    env, pool = make_pool(a=1, b=1)
    primary = make_demod_packet("B", sf=7, toa=0)
    fallback = make_demod_packet("A", sf=7, toa=0)
    pool.try_acquire(primary)
    pool.try_acquire(fallback)
    pool.schedule_release(fallback)
    env.run()
    assert (pool.used_a, pool.used_b) == (0, 1)
    assert pool.event_log[-1]["pool"] == "A"


def test_release_unknown_and_double_release_are_logged_without_negative_usage():
    env, pool = make_pool(a=0, b=1)
    packet = make_demod_packet(toa=0)
    pool.schedule_release(packet)
    env.run()
    assert pool.event_log[-1]["event"] == "RELEASE_MISSING"
    assert pool.event_log[-1]["reason"] == "demod_release_without_acquire"
    assert pool.release_count == 0
    assert (pool.used_a, pool.used_b) == (0, 0)

    packet2 = make_demod_packet("P2", toa=0)
    pool.try_acquire(packet2)
    pool.schedule_release(packet2)
    pool.schedule_release(packet2)
    env.run()
    assert pool.release_count == 1
    assert [event["event"] for event in pool.event_log[-2:]] == ["RELEASE", "RELEASE_MISSING"]
    assert pool.used_b == 0


def test_release_with_copy_does_not_release_original_object_allocation():
    env, pool = make_pool(a=0, b=1)
    original = make_demod_packet("P1", toa=0)
    copy = make_demod_packet("P1", toa=0)
    pool.try_acquire(original)
    pool.schedule_release(copy)
    env.run()
    assert pool.used_b == 1
    assert id(original) in pool.packet_pool
    assert pool.event_log[-1]["event"] == "RELEASE_MISSING"


@pytest.mark.parametrize("sf_sequence", [[7], [7, 8], [11, 12], [7, 7, 11]])
def test_acquire_release_sequences_do_not_leak_resources(sf_sequence):
    env, pool = make_pool(a=3, b=3)
    packets = [make_demod_packet(f"P{i}", sf=sf, toa=0) for i, sf in enumerate(sf_sequence)]
    assert all(pool.try_acquire(packet) for packet in packets)
    for packet in reversed(packets):
        pool.schedule_release(packet)
    env.run()
    summary = pool.get_summary()
    assert summary["used_a_at_end"] == 0
    assert summary["used_b_at_end"] == 0
    assert summary["active_mapping_count_at_end"] == 0
    assert summary["active_context_count_at_end"] == 0
    assert summary["acquire_success_count"] == len(packets)
    assert summary["release_count"] == len(packets)


def test_event_log_accessor_returns_new_list():
    _, pool = make_pool()
    pool.try_acquire(make_demod_packet())
    copied_log = pool.get_event_log()
    copied_log.clear()
    assert len(pool.event_log) == 1
