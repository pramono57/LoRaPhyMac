import math

import pytest

from loraphymac.phy.propagation import OkumuraHata
from tests.helpers.reference_formulas import reference_hata_db


pytestmark = pytest.mark.scientific


@pytest.mark.parametrize("environment", ["urban", "suburban", "rural"])
@pytest.mark.parametrize(
    ("distance_m", "frequency_mhz", "base_height_m", "mobile_height_m"),
    [(1_000, 150, 30, 1.5), (5_000, 900, 50, 2), (10_000, 1_500, 100, 3)],
)
def test_hata_matches_independent_reference(
    environment, distance_m, frequency_mhz, base_height_m, mobile_height_m
):
    actual = OkumuraHata(environment).pathloss(
        distance_m, frequency_mhz, base_height_m, mobile_height_m
    )
    expected = reference_hata_db(
        distance_m,
        frequency_mhz,
        base_height_m,
        mobile_height_m,
        environment,
    )
    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, abs=1e-8)


def test_hata_selected_valid_domain_trends():
    model = OkumuraHata("urban")
    near = model.pathloss(1_000, 900, 30, 1.5)
    far = model.pathloss(10_000, 900, 30, 1.5)
    higher_base = model.pathloss(10_000, 900, 100, 1.5)
    higher_mobile = model.pathloss(10_000, 900, 30, 3)
    assert far > near
    assert higher_base < far
    assert higher_mobile < far


def test_hata_environment_corrections_are_distinct():
    values = {
        environment: OkumuraHata(environment).pathloss(5_000, 900, 50, 2)
        for environment in ("urban", "suburban", "rural")
    }
    assert values["urban"] > values["suburban"] > values["rural"]

