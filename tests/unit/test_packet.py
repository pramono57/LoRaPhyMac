import pytest

from tests.helpers.packet_factory import make_packet
from loraphymac.runtime.packet import stable_packet_object_id


pytestmark = pytest.mark.unit


def test_constructor_minimum_valid_fields():
    packet = make_packet()
    assert packet.src_id == "N1"
    assert packet.current_tx == "N1"
    assert packet.prev_hop == "N1"
    assert packet.next_hop == "R1"
    assert packet.path == ["N1", "R1", "GW"]
    assert packet.duration == pytest.approx(0.1)


def test_path_properties_and_first_hop():
    packet = make_packet()
    assert packet.hop_count == 2
    assert packet.hop_index == 0
    assert packet.is_last_hop is False


def test_advance_hop_updates_routing_fields():
    packet = make_packet()
    packet.advance_hop()
    assert packet.prev_hop == "N1"
    assert packet.current_tx == "R1"
    assert packet.next_hop == "GW"
    assert packet.hop_index == 1
    assert packet.is_last_hop is False


def test_advance_to_last_hop_sets_next_hop_to_current_tx():
    packet = make_packet(hop_index=1, current_tx="R1", next_hop="GW")
    packet.advance_hop()
    assert packet.prev_hop == "R1"
    assert packet.current_tx == "GW"
    assert packet.next_hop == "GW"
    assert packet.is_last_hop is True


def test_advance_at_last_hop_is_no_op():
    packet = make_packet(
        hop_index=2, current_tx="GW", next_hop="GW", prev_hop="R1"
    )
    before = (packet.hop_index, packet.current_tx, packet.next_hop, packet.prev_hop)
    packet.advance_hop()
    assert (packet.hop_index, packet.current_tx, packet.next_hop, packet.prev_hop) == before


def test_latency_with_complete_timestamps():
    assert make_packet(birth_time=2.5, rx_time=7.0).latency == pytest.approx(4.5)


def test_latency_without_rx_time():
    assert make_packet(birth_time=2.5, rx_time=None).latency is None


def test_copy_is_distinct_and_path_is_independent():
    source = make_packet()
    copied = source.copy()
    assert copied is not source
    assert copied.path == source.path
    assert copied.path is not source.path
    copied.path.append("EXTRA")
    assert source.path == ["N1", "R1", "GW"]


def test_packet_object_fingerprint_is_stable_across_copies_and_runs():
    first = make_packet(packet_uid="UP:N1:7", fcnt=7, tx_start=3.0, tx_end=3.1)
    copied = first.copy()
    same_identity = make_packet(
        packet_uid="UP:N1:7", fcnt=7, tx_start=3.0, tx_end=3.1
    )
    later_transmission = make_packet(
        packet_uid="UP:N1:7", fcnt=7, tx_start=4.0, tx_end=4.1
    )

    assert stable_packet_object_id(first) == stable_packet_object_id(copied)
    assert stable_packet_object_id(first) == stable_packet_object_id(same_identity)
    assert stable_packet_object_id(first) != stable_packet_object_id(later_transmission)


def test_copy_preserves_frame_and_apc_scalar_metadata():
    source = make_packet(
        fopts=b"\x01\x02",
        fcnt=9,
        apc_old_ptx_dbm=20,
        apc_new_ptx_dbm=18,
        apc_uplink_src_id="N1",
        apc_uplink_hop_index=0,
    )
    copied = source.copy()
    assert copied.fopts == b"\x01\x02"
    assert copied.fcnt == 9
    assert copied.apc_old_ptx_dbm == 20
    assert copied.apc_new_ptx_dbm == 18
    assert copied.apc_uplink_src_id == "N1"
    assert copied.apc_uplink_hop_index == 0


def test_copy_resets_rx_collision_and_dynamic_demodulator_metadata():
    source = make_packet(
        rssi=-100,
        snr=4,
        margin=8,
        pathloss=120,
        feasible=True,
        rx_node="R1",
        distance_m=1000,
        rx_start=1.1,
        rx_end=1.2,
        rx_time=1.2,
        t_end=1.2,
        t_arrive=1.01,
        collided=True,
        captured=True,
        decoded=True,
        reason="collision",
    )
    source.phy_ready = True
    source.demod_drop = True
    copied = source.copy()
    for field in (
        "rssi", "snr", "margin", "pathloss", "feasible", "rx_node",
        "distance_m", "rx_start", "rx_end", "rx_time", "t_end", "reason",
    ):
        assert getattr(copied, field) is None
    assert copied.t_arrive == pytest.approx(1.01)
    assert copied.collided is False
    assert copied.captured is False
    assert copied.decoded is False
    # Dynamic demodulator fields are not copied into the new Packet at all.
    assert not hasattr(copied, "phy_ready")
    assert not hasattr(copied, "demod_drop")
    assert source.rssi == -100
    assert source.collided is True
    assert source.phy_ready is True
    assert source.demod_drop is True


def test_reset_can_clear_arrival_time_and_half_duplex_metadata():
    packet = make_packet(t_arrive=3.0)
    packet.rx_cancelled_by_tx = True
    packet.rx_cancelled_at = 2.0
    packet.rx_cancelled_by = "R1"
    packet.drop_time = 2.1
    packet.demod_admitted = True
    packet.demod_reason = None
    packet.demod_pool = "B"
    packet.demod_acquire_time = 1.0
    packet.demod_release_time = 2.0
    returned = packet.reset_rx_runtime_metadata(preserve_t_arrive=False)
    assert returned is packet
    assert packet.t_arrive is None
    assert packet.rx_cancelled_by_tx is False
    assert packet.rx_cancelled_at is None
    assert packet.rx_cancelled_by is None
    assert packet.drop_time is None
    for attr in (
        "demod_admitted", "demod_reason", "demod_pool",
        "demod_acquire_time", "demod_release_time",
    ):
        assert not hasattr(packet, attr)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"payload_size": 0}, "payload_size must be > 0"),
        ({"payload_size": -1}, "payload_size must be > 0"),
        ({"app_payload_size": -1}, "app_payload_size must be >= 0"),
        (
            {"payload_size": 12, "app_payload_size": 13},
            "app_payload_size cannot exceed payload_size",
        ),
        ({"fopts": bytes(16)}, "FOpts length must be <= 15"),
    ],
)
def test_constructor_rejects_invalid_frame_metadata(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_packet(**overrides)


def test_none_path_is_normalized_to_empty_and_is_last_hop():
    packet = make_packet(path=None)
    assert packet.path == []
    assert packet.hop_count == 0
    assert packet.is_last_hop is True
