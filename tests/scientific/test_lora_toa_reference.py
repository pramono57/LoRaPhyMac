import pytest
import simpy

from loraphymac.phy.air_interface import AirInterface
from tests.helpers.reference_formulas import (
    reference_lora_payload_symbols,
    reference_lora_toa_s,
)


pytestmark = pytest.mark.scientific


def air():
    return AirInterface(simpy.Environment())


@pytest.mark.parametrize("sf", [7, 8, 9, 10, 11, 12])
@pytest.mark.parametrize("bandwidth", [125_000, 250_000, 500_000])
@pytest.mark.parametrize("payload", [0, 1, 4, 12, 50, 200])
def test_default_lora_toa_matches_independent_reference(sf, bandwidth, payload):
    actual = air().compute_toa(sf, bandwidth, payload)
    expected = reference_lora_toa_s(sf, bandwidth, payload)
    assert actual == pytest.approx(expected, abs=1e-12, rel=1e-10)


@pytest.mark.parametrize("cr_index", [1, 2, 3, 4])
@pytest.mark.parametrize("crc_enabled", [False, True])
@pytest.mark.parametrize("header_enabled", [False, True])
def test_toa_coding_rate_crc_and_header_modes(
    cr_index, crc_enabled, header_enabled
):
    kwargs = {
        "sf": 9,
        "bw_hz": 125_000,
        "payload_bytes": 50,
        "cr": cr_index,
        "crc_enabled": crc_enabled,
        "header_enabled": header_enabled,
    }
    actual = air().compute_toa(**kwargs)
    expected = reference_lora_toa_s(
        sf=9,
        bandwidth_hz=125_000,
        payload_bytes=50,
        cr_index=cr_index,
        crc_enabled=crc_enabled,
        header_enabled=header_enabled,
    )
    assert actual == pytest.approx(expected, abs=1e-12, rel=1e-10)


@pytest.mark.parametrize(
    ("sf", "bandwidth", "expected_ldro"),
    [(10, 125_000, 0), (11, 125_000, 1), (12, 125_000, 1), (11, 250_000, 0)],
)
def test_automatic_low_data_rate_optimization_boundary(sf, bandwidth, expected_ldro):
    interface = air()
    actual_symbols = interface.compute_payload_sym(sf, bandwidth, 50)
    expected_symbols = reference_lora_payload_symbols(
        sf, bandwidth, 50, low_data_rate_opt=expected_ldro
    )
    assert actual_symbols == expected_symbols


def test_known_simulator_data_up_frame_configuration():
    # Default application payload 50 B + documented LoRaPhyMAC overhead 12 B.
    payload_bytes = 62
    actual = air().compute_toa(
        sf=7,
        bw_hz=125_000,
        payload_bytes=payload_bytes,
        cr=1,
        preamble_len=8,
        crc_enabled=True,
        header_enabled=True,
        low_data_rate_opt=False,
    )
    expected = reference_lora_toa_s(
        sf=7,
        bandwidth_hz=125_000,
        payload_bytes=62,
        cr_index=1,
        preamble_len=8,
        crc_enabled=True,
        header_enabled=True,
        low_data_rate_opt=False,
    )
    assert actual == pytest.approx(expected, abs=1e-12, rel=1e-10)


@pytest.mark.parametrize("invalid_sf", [4, 13])
def test_payload_symbol_calculation_rejects_unsupported_sf(invalid_sf):
    with pytest.raises(ValueError):
        air().compute_payload_sym(invalid_sf, 125_000, 50)

