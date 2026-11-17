import math

import pytest

import loraphymac.phy.propagation as propagation
from loraphymac.phy.propagation import FSPL, ITM, OkumuraHata


pytestmark = pytest.mark.unit


def test_itm_forwards_all_parameters_and_converts_distance_to_km(monkeypatch):
    calls = []

    def fake_itm(*args):
        calls.append(args)
        return 123.456

    monkeypatch.setattr(propagation, "ITMAreadBLoss", fake_itm)
    model = ITM(
        EPS=15.5,
        SGM=0.01,
        EN0=333.0,
        radio_climate=3,
        pol=0,
        pctTime=0.6,
        pctLoc=0.4,
        pctConf=0.8,
    )
    result = model.pathloss(
        d_m=12_500,
        freq_mhz=922.2,
        h_tx=30,
        h_rx=2,
        terrain_delta_h=200,
    )
    assert result == pytest.approx(123.456)
    assert calls == [
        (
            0,       # ModVar
            200,     # deltaH
            30,      # transmitter height
            2,       # receiver height
            12.5,    # distance km
            1,       # transmitter site criteria
            0,       # receiver site criteria
            15.5,    # dielectric constant
            0.01,    # conductivity
            333.0,   # surface refractivity
            922.2,
            3,
            0,
            0.6,
            0.4,
            0.8,
        )
    ]


@pytest.mark.parametrize("distance_m", [0, -100, 500, 999.9, 1000])
def test_itm_forwards_sub_kilometer_distances_as_one_km(monkeypatch, distance_m):
    captured = []

    def fake_itm(*args):
        captured.append(args)
        return 100

    monkeypatch.setattr(propagation, "ITMAreadBLoss", fake_itm)
    ITM().pathloss(distance_m, 921.4, 2, 25, 90)
    assert captured[0][4] == 1.0


def test_itm_actual_output_is_finite_deterministic_and_changes_with_distance():
    model = ITM()
    close_a = model.pathloss(1_000, 921.4, 2, 25, 90)
    close_b = model.pathloss(1_000, 921.4, 2, 25, 90)
    far = model.pathloss(20_000, 921.4, 2, 25, 90)
    assert math.isfinite(close_a)
    assert close_a == pytest.approx(close_b, abs=1e-6)
    assert far > close_a


@pytest.mark.parametrize("delta_h", [0, 30, 90, 200, 500])
def test_itm_actual_outputs_remain_finite_across_selected_terrain(delta_h):
    assert math.isfinite(ITM().pathloss(10_000, 921.4, 2, 25, delta_h))


@pytest.mark.parametrize("height", [0.5, 2, 25, 100])
def test_itm_actual_outputs_remain_finite_across_valid_heights(height):
    assert math.isfinite(ITM().pathloss(10_000, 921.4, height, 25, 90))


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (FSPL(), {"d_m": 1_000, "freq_mhz": 921.4}),
        (
            OkumuraHata("urban"),
            {"d_m": 10_000, "freq_mhz": 900, "h_tx": 30, "h_rx": 2},
        ),
    ],
)
def test_analytic_propagation_models_are_deterministic(model, kwargs):
    first = model.pathloss(**kwargs)
    second = model.pathloss(**kwargs)
    assert math.isfinite(first)
    assert first == pytest.approx(second, abs=1e-12)

