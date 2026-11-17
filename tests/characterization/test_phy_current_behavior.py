"""Characterization tests record current PHY behavior, not an ideal model."""

import math

import numpy as np
import pytest
import simpy

from loraphymac.hardware.hardware import SX1302
from loraphymac.phy.air_interface import AirInterface
from loraphymac.phy.propagation import FSPL, ITM, OkumuraHata
from loraphymac.pyitm.itm import InputError
from tests.helpers.packet_factory import make_packet
from tests.helpers.phy_fakes import (
    FakeEndpoint,
    LinearPathlossModel,
    RecordingReceiver,
)


pytestmark = pytest.mark.characterization


def test_itm_zero_negative_and_sub_kilometer_distances_share_one_km_clamp():
    model = ITM()
    values = [model.pathloss(distance, 921.4, 2, 25, 90) for distance in (-100, 0, 500, 1000)]
    assert values == pytest.approx([values[0]] * len(values), abs=1e-6)


@pytest.mark.parametrize("invalid_height", [0, -1])
def test_itm_rejects_invalid_antenna_height_via_local_library(invalid_height):
    with pytest.raises(InputError):
        ITM().pathloss(1000, 921.4, invalid_height, 25, 90)


def test_itm_rejects_unknown_climate_via_local_library():
    with pytest.raises(InputError):
        ITM(radio_climate=99).pathloss(1000, 921.4, 2, 25, 90)


@pytest.mark.parametrize(
    ("distance", "frequency", "expected_kind"),
    [(0, 921.4, "negative_infinity"), (-1, 921.4, "nan"), (1000, 0, "negative_infinity"), (1000, -1, "nan")],
)
def test_fspl_invalid_inputs_currently_return_non_finite_values(
    distance, frequency, expected_kind
):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = FSPL().pathloss(distance, frequency)
    if expected_kind == "nan":
        assert math.isnan(result)
    else:
        assert result == -math.inf


def test_unknown_hata_environment_currently_falls_back_to_urban():
    expected = OkumuraHata("urban").pathloss(5000, 900, 50, 2)
    actual = OkumuraHata("unknown").pathloss(5000, 900, 50, 2)
    assert actual == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize(
    ("kwargs", "expected_kind"),
    [
        ({"d_m": 0, "freq_mhz": 900, "h_tx": 30, "h_rx": 2}, "negative_infinity"),
        ({"d_m": -1, "freq_mhz": 900, "h_tx": 30, "h_rx": 2}, "nan"),
        ({"d_m": 1000, "freq_mhz": 0, "h_tx": 30, "h_rx": 2}, "nan"),
        ({"d_m": 1000, "freq_mhz": 900, "h_tx": 0, "h_rx": 2}, "nan"),
    ],
)
def test_hata_invalid_inputs_currently_return_non_finite_values(kwargs, expected_kind):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = OkumuraHata().pathloss(**kwargs)
    if expected_kind == "nan":
        assert math.isnan(result)
    else:
        assert result == -math.inf


def test_air_interface_broadcasts_rf_bursts_beyond_directed_next_hop():
    env = simpy.Environment()
    air = AirInterface(env)
    selected = RecordingReceiver(env)
    other = RecordingReceiver(env)
    air.register_rx(FakeEndpoint("R1", selected))
    air.register_rx(FakeEndpoint("R2", other))
    air.transmit(make_packet(next_hop="R1"))
    env.run()
    assert len(selected.packets) == 1
    assert len(other.packets) == 1
    assert selected.packets[0].rx_node == "R1"
    assert other.packets[0].rx_node == "R2"


def test_air_interface_falls_back_to_tx_start_without_endpoint_geometry():
    env = simpy.Environment()
    air = AirInterface(env)
    receiver = RecordingReceiver(env)
    air.register_rx(FakeEndpoint("R1", receiver))
    packet = make_packet(t_arrive=None)
    packet.distance_m = 30_000_000
    air.transmit(packet)
    env.run()
    assert receiver.arrival_times == [packet.tx_start]


def test_feasibility_is_strictly_rssi_margin_based_at_zero_boundary():
    air = AirInterface(simpy.Environment(), feasible_margin_db=3)
    hw = SX1302()
    sensitivity = hw.sensitivity[125][7]
    result = air.compute_link_budget(
        hw_rx=hw,
        sf=7,
        bw_hz=125_000,
        ptx=sensitivity + 3,
        gtx=0,
        grx=0,
        pathloss=0,
    )
    assert result["margin"] == 0
    assert result["feasible"] is False


def test_feasibility_can_pass_even_when_diagnostic_snr_is_below_requirement():
    class ArtificialHardware:
        sensitivity = {125: {7: -200}}
        noise_figure = 100
        snr_min = {7: -7.5}

    result = AirInterface(simpy.Environment(), feasible_margin_db=0).compute_link_budget(
        hw_rx=ArtificialHardware(),
        sf=7,
        bw_hz=125_000,
        ptx=-190,
        gtx=0,
        grx=0,
        pathloss=0,
    )
    assert result["margin"] > 0
    assert result["snr"] < result["snr_min"]
    assert result["feasible"] is True


def test_noise_floor_invalid_bandwidth_current_behavior():
    air = AirInterface(simpy.Environment())
    with np.errstate(divide="ignore", invalid="ignore"):
        assert air.compute_noise_floor(0, 3) == -math.inf
        assert math.isnan(air.compute_noise_floor(-1, 3))


def test_max_distance_finds_rssi_feasibility_boundary():
    air = AirInterface(simpy.Environment())
    result = air.max_distance(
        model=LinearPathlossModel(intercept_db=80.0, db_per_meter=0.002),
        hw_tx=SX1302(),
        hw_rx=SX1302(),
        sf=7,
        bw_hz=125_000,
        ptx=20,
        gtx=0,
        grx=0,
        h_tx=25,
        h_rx=25,
        freq_mhz=921.4,
    )
    # SX1302 SF7/BW125 sensitivity=-126 dBm; with Mfeas=3 dB,
    # feasibility changes at pathloss=143 dB -> d=31,500 m.
    assert result == pytest.approx(31_500.0, abs=0.1)
