import math

import pytest

from loraphymac.phy.propagation import FSPL
from tests.helpers.reference_formulas import reference_fspl_db


pytestmark = pytest.mark.scientific


@pytest.mark.parametrize(
    ("distance_m", "frequency_mhz"),
    [(1_000, 433.0), (1_000, 921.4), (10_000, 921.4), (10_000, 2_400.0)],
)
def test_fspl_matches_independent_reference(distance_m, frequency_mhz):
    actual = FSPL().pathloss(distance_m, frequency_mhz)
    expected = reference_fspl_db(distance_m, frequency_mhz)
    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, abs=1e-9)


def test_fspl_doubling_distance_adds_six_db():
    model = FSPL()
    difference = model.pathloss(2_000, 921.4) - model.pathloss(1_000, 921.4)
    assert difference == pytest.approx(6.020599913279624, abs=1e-9)


def test_fspl_doubling_frequency_adds_six_db():
    model = FSPL()
    difference = model.pathloss(1_000, 1_842.8) - model.pathloss(1_000, 921.4)
    assert difference == pytest.approx(6.020599913279624, abs=1e-9)

