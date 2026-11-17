from types import SimpleNamespace

import pytest

from loraphymac.hardware.hardware import (
    ESP32,
    HardwareFactory,
    SX1276,
    SX1302,
    apply_hw_ptx_bounds,
    get_hw_ptx_bounds,
    validate_initial_ptx,
    validate_ptx,
)
from tests.helpers.reference_formulas import reference_energy_mj


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("name", "expected_type", "type_name"),
    [("sx1276", SX1276, "SX1276"), ("SX1302", SX1302, "SX1302"), ("esp32", ESP32, None)],
)
def test_hardware_factory(name, expected_type, type_name):
    hardware = HardwareFactory.get(name)
    assert isinstance(hardware, expected_type)
    if type_name is not None:
        assert hardware.type == type_name


@pytest.mark.parametrize("name", ["unknown", "", None])
def test_hardware_factory_rejects_unknown_or_invalid_name(name):
    with pytest.raises((AttributeError, ValueError)):
        HardwareFactory.get(name)


@pytest.mark.parametrize(
    ("hardware", "expected_bounds"),
    [(SX1276(), (5, 20)), (SX1302(), (12, 27))],
)
def test_ptx_bounds_match_complete_current_table(hardware, expected_bounds):
    assert get_hw_ptx_bounds(hardware) == expected_bounds
    assert sorted(hardware.tx_current) == list(range(expected_bounds[0], expected_bounds[1] + 1))


@pytest.mark.parametrize("hardware", [SX1276(), SX1302()])
def test_all_current_and_sensitivity_entries_are_non_negative_or_valid(hardware):
    assert all(current >= 0 for current in hardware.tx_current.values())
    assert hardware.rx_current >= 0
    assert hardware.idle_current >= 0
    for bandwidth_table in hardware.sensitivity.values():
        assert all(value < 0 for value in bandwidth_table.values())


def test_supported_bandwidth_units_and_supported_sf_ranges():
    node = SX1276()
    relay = SX1302()
    assert node.supported_bw == [125, 250, 500]
    assert relay.supported_bw_hz == [125_000, 250_000, 500_000]
    assert set(node.sensitivity[125]) == set(range(7, 13))
    assert set(relay.sensitivity[125]) == set(range(5, 13))


@pytest.mark.parametrize("hardware,valid_ptx", [(SX1276(), 5), (SX1276(), 20), (SX1302(), 12), (SX1302(), 27)])
def test_apply_bounds_and_validate_boundary_ptx(hardware, valid_ptx):
    entity = SimpleNamespace(id="E", hw=hardware)
    assert apply_hw_ptx_bounds(entity) == get_hw_ptx_bounds(hardware)
    assert validate_ptx(entity, valid_ptx) == valid_ptx
    assert validate_initial_ptx(entity, valid_ptx) == valid_ptx


@pytest.mark.parametrize(
    ("hardware", "bad_ptx"),
    [(SX1276(), 4), (SX1276(), 21), (SX1302(), 11), (SX1302(), 28)],
)
def test_validate_ptx_rejects_out_of_hardware_bounds(hardware, bad_ptx):
    entity = SimpleNamespace(id="E", hw=hardware)
    with pytest.raises(ValueError, match="outside hardware range"):
        validate_ptx(entity, bad_ptx)


def test_missing_exact_current_entry_is_rejected():
    hardware = SimpleNamespace(
        type="GAPPED",
        tx_power_min=1,
        tx_power_max=3,
        tx_current={1: 10, 3: 30},
    )
    with pytest.raises(ValueError, match="missing tx_current entries"):
        get_hw_ptx_bounds(hardware)


def test_validate_ptx_converts_float_to_integer_current_key():
    entity = SimpleNamespace(id="N1", hw=SX1276())
    assert validate_ptx(entity, 19.9) == 19


def test_sx1276_transition_values_and_current_lookup():
    hw = SX1276()
    assert hw.sleep_to_standby() == (hw.ts_osc, hw.standby_current)
    assert hw.standby_to_fs() == (hw.ts_fs, hw.fs_current)
    transition = hw.fs_to_tx(sf=7, bw_hz=125_000, tx_power=20)
    assert transition[0] >= 0
    assert transition[1] == hw.fs_current
    assert transition[2] == hw.tx_current[20]
    assert hw.fs_to_rx(125_000) == (hw.ts_re_table[125_000], hw.rx_current)
    assert hw.tx_to_rx(921.4, 921.6, 125_000)[0] >= 0


def test_sx1302_transition_values_and_current_lookup():
    hw = SX1302()
    assert hw.rx_to_tx(27) == (hw.ts_tx, hw.idle_current, hw.tx_current[27])
    assert hw.tx_to_rx() == (hw.ts_rx, hw.idle_current, hw.rx_current)
    with pytest.raises(ValueError):
        hw.rx_to_tx(None)
    with pytest.raises(KeyError):
        hw.rx_to_tx(11)


@pytest.mark.parametrize(
    ("method", "current"),
    [
        ("energy_active_modem", ESP32.active_modem_current),
        ("energy_active", ESP32.active_current),
        ("energy_sleep", ESP32.sleep_current),
        ("energy_deep_sleep", ESP32.deep_sleep_current),
    ],
)
def test_esp32_energy_matches_independent_formula(method, current):
    hw = ESP32()
    duration = 2.5
    expected = reference_energy_mj(hw.voltage, current, duration)
    assert getattr(hw, method)(duration) == pytest.approx(expected)


def test_esp32_compute_energy_and_unknown_state():
    hw = ESP32()
    assert hw.compute_energy("active", 1.0) == pytest.approx(hw.energy_active(1.0))
    assert hw.energy_wake_up() >= 0
    with pytest.raises(ValueError, match="Unknown ESP32 energy state"):
        hw.compute_energy("unknown", 1.0)

