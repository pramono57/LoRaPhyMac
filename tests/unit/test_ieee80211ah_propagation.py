import math

import pytest

from loraphymac.phy.propagation import FSPL, IEEE80211ah, ITM, OkumuraHata


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("distance_m", "frequency_mhz", "expected_loss_db"),
    [
        (770, 868.0, 131.501872797),
        (2_000, 868.0, 147.088549367),
        (1_000, 921.4, 136.314319668),
        (2_000, 921.4, 147.633047505),
    ],
)
def test_ieee80211ah_matches_equation_15_reference_values(
    distance_m, frequency_mhz, expected_loss_db
):
    result = IEEE80211ah().pathloss(distance_m, frequency_mhz)

    assert math.isfinite(result)
    assert result == pytest.approx(expected_loss_db, abs=1e-6)


def test_ieee80211ah_path_loss_increases_with_distance():
    model = IEEE80211ah()

    assert model.pathloss(2_000, 921.4) > model.pathloss(1_000, 921.4)


def test_ieee80211ah_path_loss_increases_with_frequency():
    model = IEEE80211ah()

    assert model.pathloss(1_000, 921.4) > model.pathloss(1_000, 868.0)


@pytest.mark.parametrize("distance_m", [0, -1, math.nan, math.inf, -math.inf])
def test_ieee80211ah_rejects_invalid_distance(distance_m):
    with pytest.raises(ValueError, match="distance_m"):
        IEEE80211ah().pathloss(distance_m, 921.4)


@pytest.mark.parametrize("frequency_mhz", [0, -1, math.nan, math.inf, -math.inf])
def test_ieee80211ah_rejects_invalid_frequency(frequency_mhz):
    with pytest.raises(ValueError, match="frequency_mhz"):
        IEEE80211ah().pathloss(1_000, frequency_mhz)


def test_ieee80211ah_ignores_height_and_terrain_arguments():
    model = IEEE80211ah()
    expected = model.pathloss(1_000, 921.4)

    assert model.pathloss(
        1_000,
        921.4,
        h_tx=2,
        h_rx=25,
        terrain_delta_h=20,
    ) == pytest.approx(expected, abs=1e-12)
    assert model.pathloss(
        1_000,
        921.4,
        h_tx=100,
        h_rx=0.5,
        terrain_delta_h=500,
    ) == pytest.approx(expected, abs=1e-12)


def test_existing_propagation_imports_remain_available():
    assert all(model is not None for model in (FSPL, ITM, OkumuraHata))
