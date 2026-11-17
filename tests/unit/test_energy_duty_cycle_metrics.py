from __future__ import annotations

from types import SimpleNamespace

import pytest

from loraphymac.metrics.duty_cycle_metrics import (
    build_per_channel_duty_cycle_df,
    build_relay_duty_cycle_df,
)
from loraphymac.metrics.energy_metrics import (
    build_energy_breakdown_df,
    continuous_rx_energy_mj,
)


def _entity(entity_id: str, **overrides):
    values = {
        "id": entity_id,
        "hw": SimpleNamespace(
            tx_full_duplex=False,
            voltage=3.3,
            rx_current=10.0,
        ),
        "mcu": None,
        "include_mcu_energy": False,
        "rx_always_on": entity_id.startswith("R") or entity_id.startswith("GW"),
        "total_tx_time_s": 0.0,
        "total_rx_time_s": 0.0,
        "total_transition_time_s": 0.0,
        "energy_tx_mJ": 0.0,
        "energy_rx_mJ": 0.0,
        "energy_sleep_mJ": 0.0,
        "energy_idle_mJ": 0.0,
        "energy_transition_mJ": 0.0,
        "energy_cad_mJ": 0.0,
        "tx_airtime_per_channel": {},
        "tx_packet_count_per_channel": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _infer_entity_type(entity_id):
    if str(entity_id).startswith("GW"):
        return "gateway"
    if str(entity_id).startswith("R"):
        return "relay"
    return "node"


def test_continuous_rx_energy_preserves_half_duplex_and_mcu_formula():
    entity = _entity(
        "R1",
        total_tx_time_s=10.0,
        include_mcu_energy=True,
        mcu=SimpleNamespace(energy_active=lambda duration: duration * 5.0),
    )

    energy_mj, rx_time_s, tx_time_s = continuous_rx_energy_mj(entity, 100.0)

    assert rx_time_s == 90.0
    assert tx_time_s == 10.0
    assert energy_mj == pytest.approx(3.3 * 0.010 * 90.0 * 1000.0 + 450.0)


def test_continuous_rx_energy_preserves_full_duplex_horizon():
    hardware = SimpleNamespace(
        tx_full_duplex=True,
        voltage=3.3,
        rx_current=10.0,
    )
    entity = _entity("GW", hw=hardware, total_tx_time_s=10.0)

    energy_mj, rx_time_s, tx_time_s = continuous_rx_energy_mj(entity, 100.0)

    assert rx_time_s == 100.0
    assert tx_time_s == 10.0
    assert energy_mj == pytest.approx(3.3 * 0.010 * 100.0 * 1000.0)


def test_corrected_continuous_rx_conserves_exclusive_state_duration():
    entity = _entity(
        "R1",
        total_tx_time_s=10.0,
        total_transition_time_s=5.0,
    )

    energy_mj, rx_time_s, tx_time_s = continuous_rx_energy_mj(
        entity,
        100.0,
        "corrected",
    )

    assert rx_time_s == 85.0
    assert rx_time_s + tx_time_s + entity.total_transition_time_s == 100.0
    assert energy_mj == pytest.approx(3.3 * 0.010 * 85.0 * 1000.0)


def test_corrected_mode_removes_legacy_transition_overlap_from_rx():
    entity = _entity(
        "GW",
        total_tx_time_s=10.0,
        total_transition_time_s=5.0,
    )

    legacy_energy, legacy_rx_s, _ = continuous_rx_energy_mj(entity, 100.0, "legacy")
    corrected_energy, corrected_rx_s, _ = continuous_rx_energy_mj(
        entity,
        100.0,
        "corrected",
    )

    assert legacy_rx_s - corrected_rx_s == entity.total_transition_time_s
    assert legacy_energy - corrected_energy == pytest.approx(
        3.3 * 0.010 * entity.total_transition_time_s * 1000.0
    )


def test_zero_transition_is_identical_in_legacy_and_corrected_modes():
    entity = _entity("R1", total_tx_time_s=10.0, total_transition_time_s=0.0)

    assert continuous_rx_energy_mj(entity, 100.0, "corrected") == pytest.approx(
        continuous_rx_energy_mj(entity, 100.0, "legacy")
    )


def test_corrected_mode_rejects_nonexclusive_full_duplex_state_model():
    hardware = SimpleNamespace(
        tx_full_duplex=True,
        tx_phy_radio_count=1,
        voltage=3.3,
        rx_current=10.0,
    )
    entity = _entity("GW", hw=hardware, total_transition_time_s=1.0)

    with pytest.raises(ValueError, match="mutually exclusive"):
        continuous_rx_energy_mj(entity, 100.0, "corrected")


def test_energy_breakdown_preserves_node_energy_components_and_joule_conversion():
    node = _entity(
        "N1",
        total_tx_time_s=1.0,
        total_rx_time_s=2.0,
        energy_tx_mJ=100.0,
        energy_rx_mJ=200.0,
        energy_sleep_mJ=300.0,
        energy_idle_mJ=400.0,
        energy_transition_mJ=500.0,
        energy_cad_mJ=600.0,
        total_transition_time_s=0.25,
    )

    result = build_energy_breakdown_df(
        [node],
        {"run_id": "energy-8e"},
        _infer_entity_type,
        lambda entity: continuous_rx_energy_mj(entity, 100.0),
        lambda value: float(value) / 1000.0,
    ).iloc[0]

    assert result["run_id"] == "energy-8e"
    assert result["entity_type"] == "node"
    assert result["energy_accounting_mode"] == "legacy"
    assert result["tx_time_s"] == 1.0
    assert result["rx_time_s"] == 2.0
    assert result["total_transition_time_s"] == 0.25
    assert result["energy_tx_J"] == 0.1
    assert result["energy_rx_J"] == 0.2
    assert result["energy_sleep_J"] == 0.3
    assert result["energy_idle_J"] == 0.4
    assert result["energy_transition_J"] == 0.5
    assert result["energy_cad_J"] == 0.6
    assert result["total_energy_J"] == pytest.approx(2.1)


def test_corrected_energy_breakdown_total_uses_eq10_rx_duration():
    relay = _entity(
        "R1",
        total_tx_time_s=10.0,
        total_transition_time_s=5.0,
        energy_tx_mJ=100.0,
        energy_transition_mJ=50.0,
    )

    result = build_energy_breakdown_df(
        [relay],
        {"run_id": "corrected", "energy_accounting_mode": "corrected"},
        _infer_entity_type,
        lambda entity: continuous_rx_energy_mj(entity, 100.0, "corrected"),
        lambda value: float(value) / 1000.0,
    ).iloc[0]

    expected_rx_j = 3.3 * 0.010 * 85.0
    assert result["energy_accounting_mode"] == "corrected"
    assert result["rx_time_s"] == 85.0
    assert result["energy_rx_J"] == pytest.approx(expected_rx_j)
    assert result["total_energy_J"] == pytest.approx(0.1 + expected_rx_j + 0.05)


def test_relay_duty_cycle_preserves_limit_margin_and_status():
    relay = _entity("R1", total_tx_time_s=0.5)
    gateway = _entity("GW", total_tx_time_s=2.0)
    node = _entity("N1", total_tx_time_s=99.0)

    result = build_relay_duty_cycle_df(
        [node, relay, gateway],
        {
            "run_id": "duty-8e",
            "sim_time": 100.0,
            "relay_duty_cycle": 0.01,
            "relay_dc_mode": "per_channel",
        },
        _infer_entity_type,
    )

    assert result["entity_id"].tolist() == ["R1", "GW"]
    assert result["relay_duty_cycle_percent"].tolist() == [0.5, 2.0]
    assert result["duty_cycle_margin_percent"].tolist() == [0.5, -1.0]
    assert result["dc_status"].tolist() == ["OK", "VIOLATION"]


def test_per_channel_duty_cycle_is_sorted_and_keeps_unlimited_mode():
    relay = _entity(
        "R1",
        tx_airtime_per_channel={2: 2.0, 0: 0.5},
        tx_packet_count_per_channel={2: 4, 0: 1},
    )

    result = build_per_channel_duty_cycle_df(
        [relay],
        sim_time=None,
        run_metadata={
            "run_id": "channel-8e",
            "sim_time": 100.0,
            "relay_duty_cycle": None,
        },
        infer_entity_type=_infer_entity_type,
    )

    assert result["channel"].tolist() == [0, 2]
    assert result["tx_packet_count"].tolist() == [1, 4]
    assert result["per_channel_duty_cycle_percent"].tolist() == [0.5, 2.0]
    assert result["duty_cycle_margin_percent"].isna().all()
    assert result["dc_status"].tolist() == ["unlimited", "unlimited"]
