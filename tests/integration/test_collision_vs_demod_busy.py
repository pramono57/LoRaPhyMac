import pytest
import simpy

from loraphymac.mac.collision_engine import CollisionEngine
from loraphymac.runtime.demodulator_pool import DemodulatorPool
from tests.helpers.collision_fakes import make_collision_packet
from tests.helpers.demodulator_fakes import make_demod_packet


pytestmark = pytest.mark.integration


def test_rf_collision_reason_is_distinct_from_demodulator_busy():
    packets = [
        make_collision_packet("P1", rssi=-100),
        make_collision_packet("P2", rssi=-100),
    ]
    survivors, failed = CollisionEngine().resolve(packets)
    assert survivors == []
    assert all(packet.reason == "collision" for packet in failed)
    assert all(packet.reason != "demod_busy" for packet in failed)


def test_noncolliding_packet_can_fail_only_because_demodulator_is_full():
    env = simpy.Environment()
    pool = DemodulatorPool(env, pool_a_size=0, pool_b_size=1)
    occupying = make_demod_packet("occupying", sf=7)
    candidate = make_demod_packet("candidate", sf=7)
    assert pool.try_acquire(occupying) is True
    survivors, failed = CollisionEngine().resolve(
        [make_collision_packet("candidate", sf=7)]
    )
    assert len(survivors) == 1
    assert failed == []
    assert pool.try_acquire(candidate) is False
    assert candidate.demod_reason == "demod_busy"
    assert candidate.demod_reason != "collision"


def test_unsupported_sf_is_neither_collision_nor_busy():
    env = simpy.Environment()
    pool = DemodulatorPool(env, pool_a_size=1, pool_b_size=1)
    candidate = make_demod_packet("candidate", sf=13)
    assert pool.try_acquire(candidate) is False
    assert candidate.demod_reason == "demod_unsupported_sf"
    assert candidate.demod_reason not in ("collision", "demod_busy")
    assert pool.unsupported_sf_count == 1
    assert pool.demod_busy_count == 0

