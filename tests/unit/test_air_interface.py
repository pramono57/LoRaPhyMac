import math

import pytest
import simpy

from loraphymac.hardware.hardware import SX1276, SX1302
from loraphymac.phy.air_interface import AirInterface
from tests.helpers.packet_factory import make_packet
from tests.helpers.phy_fakes import FakeEndpoint, RecordingReceiver
from tests.helpers.reference_formulas import (
    reference_noise_floor_dbm,
    reference_rssi_dbm,
)


pytestmark = pytest.mark.unit


def make_air(feasible_margin_db=3):
    env = simpy.Environment()
    return env, AirInterface(env, feasible_margin_db=feasible_margin_db)


def test_registers_multiple_receivers_and_looks_up_instances():
    env, air = make_air()
    r1 = RecordingReceiver(env)
    gw = RecordingReceiver(env)
    air.register_rx(FakeEndpoint("R1", r1))
    air.register_rx(FakeEndpoint("GW", gw))
    assert air.rx_nodes_map == {"R1": r1, "GW": gw}


def test_duplicate_receiver_id_replaces_previous_receiver():
    env, air = make_air()
    first = RecordingReceiver(env)
    second = RecordingReceiver(env)
    air.register_rx(FakeEndpoint("R1", first))
    air.register_rx(FakeEndpoint("R1", second))
    assert air.rx_nodes_map == {"R1": second}


def test_transmit_broadcasts_independent_copies_to_all_other_receivers():
    env, air = make_air()
    r1 = RecordingReceiver(env)
    gw = RecordingReceiver(env)
    air.register_rx(FakeEndpoint("R1", r1))
    air.register_rx(FakeEndpoint("GW", gw))
    source = make_packet(next_hop="R1", rssi=-100, collided=True)
    air.transmit(source)
    env.run()
    assert len(r1.packets) == 1
    assert len(gw.packets) == 1
    received = r1.packets[0]
    interferer = gw.packets[0]
    assert received is not source
    assert interferer is not source
    assert interferer is not received
    assert received.path == source.path
    assert received.path is not source.path
    assert received.next_hop == "R1"
    assert received.rx_node == "R1"
    assert interferer.next_hop == "R1"
    assert interferer.rx_node == "GW"
    assert received.rssi is None
    assert received.collided is False
    assert source.rssi == -100
    assert source.collided is True


def test_transmitter_does_not_receive_its_own_rf_broadcast():
    env, air = make_air()
    transmitter_rx = RecordingReceiver(env)
    relay_rx = RecordingReceiver(env)
    air.register_rx(FakeEndpoint("N1", transmitter_rx))
    air.register_rx(FakeEndpoint("R1", relay_rx))

    air.transmit(make_packet(current_tx="N1", next_hop="R1"))
    env.run()

    assert transmitter_rx.packets == []
    assert len(relay_rx.packets) == 1


def test_broadcast_uses_receiver_specific_propagation_delay():
    env, air = make_air()
    transmitter_rx = RecordingReceiver(env)
    near_rx = RecordingReceiver(env)
    far_rx = RecordingReceiver(env)
    transmitter = FakeEndpoint("N1", transmitter_rx)
    near = FakeEndpoint("R1", near_rx)
    far = FakeEndpoint("R2", far_rx)
    transmitter.distance_to = lambda receiver: {
        "R1": 300.0,
        "R2": 600.0,
    }[receiver.id]
    air.register_rx(transmitter)
    air.register_rx(near)
    air.register_rx(far)

    air.transmit(
        make_packet(
            current_tx="N1",
            next_hop="R1",
            tx_start=0.0,
            tx_end=0.1,
            t_arrive=99.0,
        )
    )
    env.run()

    assert near_rx.arrival_times == pytest.approx([1e-6], abs=1e-12)
    assert far_rx.arrival_times == pytest.approx([2e-6], abs=1e-12)


@pytest.mark.parametrize("next_hop", ["UNKNOWN", None])
def test_unregistered_or_none_next_hop_is_silent_no_op(next_hop):
    env, air = make_air()
    source = make_packet(next_hop=next_hop)
    assert air.transmit(source) is None
    env.run()
    assert air.rx_nodes_map == {}
    assert source.next_hop == next_hop


@pytest.mark.parametrize("arrival", [0.0, 1e-6, 0.25, 10.0])
def test_delivery_waits_until_precomputed_packet_arrival_time(arrival):
    env, air = make_air()
    receiver = RecordingReceiver(env)
    air.register_rx(FakeEndpoint("R1", receiver))
    source = make_packet(t_arrive=arrival)
    air.transmit(source)
    if arrival > 0:
        env.run(until=arrival / 2)
        assert receiver.packets == []
    env.run()
    assert receiver.arrival_times == pytest.approx([arrival], abs=1e-12)
    assert source.t_arrive == arrival


def test_two_transmissions_keep_independent_arrival_times():
    env, air = make_air()
    receiver = RecordingReceiver(env)
    air.register_rx(FakeEndpoint("R1", receiver))
    air.transmit(make_packet(src_id="N1", t_arrive=2.0))
    air.transmit(make_packet(src_id="N2", t_arrive=1.0))
    env.run()
    assert receiver.arrival_times == pytest.approx([1.0, 2.0], abs=1e-12)
    assert [packet.src_id for packet in receiver.packets] == ["N2", "N1"]


@pytest.mark.parametrize(
    ("ptx", "gtx", "grx", "pathloss"),
    [(20, 2.15, 5.8, 120), (0, 0, 0, 0), (-10, -2, -3, 80)],
)
def test_rssi_matches_independent_link_arithmetic(ptx, gtx, grx, pathloss):
    _, air = make_air()
    expected = reference_rssi_dbm(ptx, gtx, grx, pathloss)
    assert air.compute_rssi(ptx, gtx, grx, pathloss) == pytest.approx(
        expected, abs=1e-9
    )


@pytest.mark.parametrize("bandwidth", [125_000, 250_000, 500_000])
@pytest.mark.parametrize("noise_figure", [3, 6])
def test_noise_floor_and_snr_match_independent_formula(bandwidth, noise_figure):
    _, air = make_air()
    expected_noise = reference_noise_floor_dbm(bandwidth, noise_figure)
    actual_noise = air.compute_noise_floor(bandwidth, noise_figure)
    assert actual_noise == pytest.approx(expected_noise, abs=1e-9)
    assert air.compute_snr(-120, bandwidth, noise_figure) == pytest.approx(
        -120 - expected_noise, abs=1e-9
    )


@pytest.mark.parametrize("hardware", [SX1276(), SX1302()])
@pytest.mark.parametrize("bandwidth", [125_000, 250_000, 500_000])
@pytest.mark.parametrize("sf", [7, 8, 9, 10, 11, 12])
def test_sensitivity_matches_hardware_reference_table(hardware, bandwidth, sf):
    _, air = make_air()
    expected = hardware.sensitivity[bandwidth // 1000][sf]
    assert air.compute_sensitivity(hardware, sf, bandwidth) == expected


@pytest.mark.parametrize(
    ("hardware", "sf", "bandwidth"),
    [
        (SX1276(), 6, 125_000),
        (SX1302(), 13, 125_000),
        (SX1276(), 7, 100_000),
        (SX1276(), 7, 0),
    ],
)
def test_sensitivity_rejects_unsupported_combinations(hardware, sf, bandwidth):
    _, air = make_air()
    with pytest.raises(ValueError):
        air.compute_sensitivity(hardware, sf, bandwidth)


def test_link_budget_matches_independent_arithmetic():
    _, air = make_air(feasible_margin_db=3)
    hw = SX1302()
    result = air.compute_link_budget(
        hw_rx=hw,
        sf=7,
        bw_hz=125_000,
        ptx=20,
        gtx=2,
        grx=5,
        pathloss=140,
    )
    expected_rssi = 20 + 2 + 5 - 140
    expected_sensitivity = hw.sensitivity[125][7]
    expected_noise = reference_noise_floor_dbm(125_000, hw.noise_figure)
    assert result["rssi"] == pytest.approx(expected_rssi, abs=1e-9)
    assert result["sensitivity"] == expected_sensitivity
    assert result["raw_margin"] == pytest.approx(
        expected_rssi - expected_sensitivity, abs=1e-9
    )
    assert result["margin"] == pytest.approx(
        expected_rssi - expected_sensitivity - 3, abs=1e-9
    )
    assert result["noise_floor"] == pytest.approx(expected_noise, abs=1e-9)
    assert result["snr"] == pytest.approx(expected_rssi - expected_noise, abs=1e-9)
    assert result["snr_min"] == hw.snr_min[7]


@pytest.mark.parametrize(
    ("offset_from_threshold", "expected"),
    [(-1e-9, False), (0.0, False), (1e-9, True)],
)
def test_feasibility_exact_rssi_margin_boundary(offset_from_threshold, expected):
    _, air = make_air(feasible_margin_db=3)
    hw = SX1302()
    sensitivity = hw.sensitivity[125][7]
    target_rssi = sensitivity + 3 + offset_from_threshold
    result = air.compute_link_budget(
        hw_rx=hw,
        sf=7,
        bw_hz=125_000,
        ptx=target_rssi,
        gtx=0,
        grx=0,
        pathloss=0,
    )
    assert result["margin"] == pytest.approx(offset_from_threshold, abs=1e-12)
    assert result["feasible"] is expected


def test_link_budget_accepts_per_call_feasible_margin_override():
    _, air = make_air(feasible_margin_db=10)
    result = air.compute_link_budget(
        hw_rx=SX1302(),
        sf=7,
        bw_hz=125_000,
        ptx=-125,
        gtx=0,
        grx=0,
        pathloss=0,
        feasible_margin_db=0,
    )
    assert result["raw_margin"] == 1
    assert result["margin"] == 1
    assert result["feasible"] is True


def test_data_rate_formula_is_finite_and_decreases_with_cr_index():
    _, air = make_air()
    rates = [air.compute_data_rate(7, 125_000, cr) for cr in range(1, 5)]
    assert all(math.isfinite(rate) and rate > 0 for rate in rates)
    assert rates == sorted(rates, reverse=True)
