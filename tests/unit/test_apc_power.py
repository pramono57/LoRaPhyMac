from types import SimpleNamespace

import pytest

from loraphymac.hardware.hardware import SX1276, SX1302, apply_hw_ptx_bounds
from loraphymac.runtime.apc_power import (
    APC_TX_INDEX_STEP_DBM,
    _get_supported_ptx_values,
    dbm_to_tx_power_index,
    get_max_tx_power_index,
    tx_power_index_to_dbm,
)


pytestmark = pytest.mark.unit

NODE_LEVELS = list(range(20, 4, -2))
RELAY_LEVELS = list(range(27, 11, -2))


def make_bounded_entity(hardware, entity_id="E"):
    entity = SimpleNamespace(id=entity_id, hw=hardware)
    apply_hw_ptx_bounds(entity)
    return entity


@pytest.mark.parametrize(
    ("hardware", "expected"),
    [(SX1276(), NODE_LEVELS), (SX1302(), RELAY_LEVELS)],
)
def test_supported_levels_follow_actual_hardware_table(hardware, expected):
    entity = make_bounded_entity(hardware)
    assert _get_supported_ptx_values(entity) == expected
    assert all(a - b == APC_TX_INDEX_STEP_DBM for a, b in zip(expected, expected[1:]))


@pytest.mark.parametrize("hardware,levels", [(SX1276(), NODE_LEVELS), (SX1302(), RELAY_LEVELS)])
def test_all_supported_levels_round_trip(hardware, levels):
    entity = make_bounded_entity(hardware)
    for expected_index, ptx in enumerate(levels):
        assert dbm_to_tx_power_index(entity, ptx) == expected_index
        assert tx_power_index_to_dbm(entity, expected_index) == ptx


@pytest.mark.parametrize("hardware,levels", [(SX1276(), NODE_LEVELS), (SX1302(), RELAY_LEVELS)])
def test_index_minimum_maximum_and_monotonicity(hardware, levels):
    entity = make_bounded_entity(hardware)
    assert tx_power_index_to_dbm(entity, 0) == levels[0]
    assert get_max_tx_power_index(entity) == len(levels) - 1
    assert tx_power_index_to_dbm(entity, len(levels) - 1) == levels[-1]
    decoded = [tx_power_index_to_dbm(entity, i) for i in range(len(levels))]
    assert decoded == sorted(decoded, reverse=True)


@pytest.mark.parametrize("bad_index", [-1, 8, 99])
def test_node_rejects_invalid_index(node_entity, bad_index):
    with pytest.raises(ValueError, match="invalid TxPowerIndex"):
        tx_power_index_to_dbm(node_entity, bad_index)


@pytest.mark.parametrize(
    ("input_dbm", "expected_index"),
    [(100, 0), (21, 0), (20, 0), (19, 0), (18, 1), (5, 7), (-100, 7)],
)
def test_node_dbm_conversion_clips_or_selects_nearest_level(
    node_entity, input_dbm, expected_index
):
    assert dbm_to_tx_power_index(node_entity, input_dbm) == expected_index


def test_midpoint_tie_selects_first_higher_power_level(node_entity):
    assert dbm_to_tx_power_index(node_entity, 17) == 1  # 18 dBm, before 16 dBm


@pytest.mark.parametrize(
    "entity",
    [
        None,
        SimpleNamespace(id="E"),
        SimpleNamespace(id="E", hw=SimpleNamespace()),
        SimpleNamespace(id="E", hw=SimpleNamespace(tx_current={})),
    ],
)
def test_missing_or_empty_hardware_table_is_rejected(entity):
    with pytest.raises(ValueError):
        _get_supported_ptx_values(entity)


@pytest.mark.parametrize(
    ("entity", "message"),
    [
        (SimpleNamespace(id="E", hw=SX1276(), ptx_max=20), "ptx_min"),
        (SimpleNamespace(id="E", hw=SX1276(), ptx_min=5), "ptx_max"),
    ],
)
def test_operational_ptx_bounds_are_required(entity, message):
    with pytest.raises(ValueError, match=message):
        _get_supported_ptx_values(entity)


@pytest.mark.parametrize(
    ("hardware", "ptx_min", "ptx_max", "expected"),
    [
        (SX1276(), 5, 17, [17, 15, 13, 11, 9, 7, 5]),
        (SX1302(), 12, 20, [20, 18, 16, 14, 12]),
    ],
)
def test_operational_ladder_is_anchored_at_eirp_limited_ptx_max(
    hardware, ptx_min, ptx_max, expected
):
    entity = SimpleNamespace(
        id="E",
        hw=hardware,
        ptx_min=ptx_min,
        ptx_max=ptx_max,
    )
    assert _get_supported_ptx_values(entity) == expected
    assert dbm_to_tx_power_index(entity, 1000) == 0
    assert dbm_to_tx_power_index(entity, -1000) == len(expected) - 1
