from __future__ import annotations

import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from loraphymac.config.defaults import build_config
from loraphymac.config.validation import validate_configuration
from loraphymac.hardware.hardware import SX1276, SX1302, apply_eirp_ptx_bounds
from loraphymac.metrics.apc_metrics import build_final_ptx_distribution_df
from loraphymac.runtime.apc_power import _get_supported_ptx_values
from loraphymac.runtime.islam_dbpc import (
    CalibrationIntegrityError,
    ISLAM_DBPC_POLICY_ID,
    load_frozen_calibration,
)
from loraphymac.runtime.tx_power_policy import apply_tx_power_policy


pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = (
    PROJECT_ROOT / "artifacts/experiments/apc_compare_model/calibration/islam_dbpc_itm0"
)
MANIFEST = CALIBRATION_DIR / "calibration_manifest.json"
MANIFEST_SHA256 = "a61f15f546c9480e9439dc87740f537a9db2f1c845e3654b5becec464bffe6bf"
CALIBRATION_ID = "islam_dbpc_itm90_margin0_fixed_sf7_v1"


def load_calibration(margin_db=0, manifest=MANIFEST, manifest_hash=MANIFEST_SHA256):
    return load_frozen_calibration(
        manifest,
        expected_manifest_sha256=manifest_hash,
        expected_calibration_id=CALIBRATION_ID,
        margin_db=margin_db,
    )


def entity(entity_id, hardware, x, *, gain, max_eirp):
    value = SimpleNamespace(
        id=entity_id,
        hw=hardware,
        loc=SimpleNamespace(x=float(x), y=0.0),
        h=2.0 if entity_id.startswith("N") else 25.0,
        gain=float(gain),
        default_ptx=999,
        fixed_ptx=999,
        ptx_mode="adaptive",
    )
    value.distance_to = lambda other: abs(value.loc.x - other.loc.x)
    apply_eirp_ptx_bounds(value, max_eirp, gain, 0.0)
    return value


def dbpc_config(**overrides):
    config = build_config(
        {
            "experiment_profile": "apc_compare_model_v1",
            "energy_accounting_mode": "corrected",
            "node_ptx_mode": "fixed",
            "relay_ptx_mode": "fixed",
            "tx_power_policy_id": ISLAM_DBPC_POLICY_ID,
            "tx_power_model_id": "islam_dbpc_itm0",
            "tx_power_calibration_id": CALIBRATION_ID,
            "tx_power_calibration_manifest": str(MANIFEST),
            "tx_power_calibration_manifest_sha256": MANIFEST_SHA256,
            "tx_power_calibration_margin_db": 0,
            **overrides,
        }
    )
    validate_configuration(config)
    return config


@pytest.mark.extended_research
def test_frozen_calibration_and_operational_ladders_are_exact():
    calibration = load_calibration()
    node = entity("N1", SX1276(), 0, gain=2.15, max_eirp=20.0)
    relay = entity("R1", SX1302(), 0, gain=5.8, max_eirp=26.0)

    assert calibration.ptx_ladders == {
        "node": (5, 7, 9, 11, 13, 15, 17),
        "relay": (12, 14, 16, 18, 20),
    }
    assert tuple(sorted(_get_supported_ptx_values(node))) == calibration.ptx_ladders["node"]
    assert tuple(sorted(_get_supported_ptx_values(relay))) == calibration.ptx_ladders["relay"]


@pytest.mark.extended_research
def test_algorithm_selects_minimum_strictly_feasible_power_and_is_monotonic():
    calibration = load_calibration()
    first_boundary = calibration.coverage_by_entity["node"][0][1]
    below = calibration.select(
        entity_id="N1", entity_type="node", parent_id="R1", distance_m=first_boundary - 1e-6
    )
    boundary = calibration.select(
        entity_id="N1", entity_type="node", parent_id="R1", distance_m=first_boundary
    )
    choices = [
        calibration.select(
            entity_id="N1", entity_type="node", parent_id="R1", distance_m=distance
        ).selected_ptx_dbm
        for distance in (1_000, 8_000, 10_000, 12_000, 17_000)
    ]

    assert below.selected_ptx_dbm == 5
    assert boundary.selected_ptx_dbm == 7
    assert choices == sorted(choices)


@pytest.mark.extended_research
def test_maximum_selection_is_distinct_from_outside_coverage():
    calibration = load_calibration()
    inside = calibration.select(
        entity_id="R1", entity_type="relay", parent_id="GW", distance_m=33_000
    )
    outside = calibration.select(
        entity_id="R1", entity_type="relay", parent_id="GW", distance_m=40_000
    )

    assert inside.selected_ptx_dbm == 20
    assert inside.selected_at_max_ptx is True
    assert inside.outside_estimated_coverage is False
    assert outside.selected_ptx_dbm == 20
    assert outside.selected_at_max_ptx is True
    assert outside.outside_estimated_coverage is True


@pytest.mark.extended_research
def test_manifest_and_protected_file_corruption_fail_closed(tmp_path):
    with pytest.raises(CalibrationIntegrityError, match="manifest SHA-256"):
        load_calibration(manifest_hash="0" * 64)

    copied = tmp_path / "calibration"
    shutil.copytree(CALIBRATION_DIR, copied)
    with (copied / "coverage_table_margin_0.csv").open("a", encoding="utf-8") as handle:
        handle.write("corrupt\n")
    with pytest.raises(CalibrationIntegrityError, match="file hash mismatch"):
        load_calibration(manifest=copied / "calibration_manifest.json")


@pytest.mark.parametrize(
    ("node_mode", "relay_mode"),
    [("fixed", "fixed"), ("adaptive", "fixed"), ("fixed", "adaptive"), ("adaptive", "adaptive")],
)
def test_legacy_dispatch_is_a_no_op_for_all_existing_modes(node_mode, relay_mode):
    node = entity("N1", SX1276(), 0, gain=2.15, max_eirp=20.0)
    relay = entity("R1", SX1302(), 10_000, gain=5.8, max_eirp=26.0)
    gateway = entity("GW", SX1302(), 20_000, gain=5.8, max_eirp=26.0)
    node.default_ptx, relay.default_ptx = 13, 18
    node.ptx_mode, relay.ptx_mode = node_mode, relay_mode
    before = (node.default_ptx, relay.default_ptx, node.ptx_mode, relay.ptx_mode)

    result = apply_tx_power_policy(
        config=build_config({"node_ptx_mode": node_mode, "relay_ptx_mode": relay_mode}),
        nodes=[node],
        relays=[relay],
        gateway=gateway,
        routing_map={"N1": "R1", "R1": "GW", "GW": None},
        node_map={"N1": node, "R1": relay, "GW": gateway},
    )

    assert result.policy_id == "legacy"
    assert result.selections == ()
    assert result.run_metadata == {}
    assert (node.default_ptx, relay.default_ptx, node.ptx_mode, relay.ptx_mode) == before


@pytest.mark.extended_research
def test_dbpc_applies_once_as_fixed_and_disables_feedback_without_rng():
    node = entity("N1", SX1276(), 0, gain=2.15, max_eirp=20.0)
    relay = entity("R1", SX1302(), 8_000, gain=5.8, max_eirp=26.0)
    gateway = entity("GW", SX1302(), 26_000, gain=5.8, max_eirp=26.0)
    random_state = random.getstate()
    result = apply_tx_power_policy(
        config=dbpc_config(),
        nodes=[node],
        relays=[relay],
        gateway=gateway,
        routing_map={"N1": "R1", "R1": "GW", "GW": None},
        node_map={"N1": node, "R1": relay, "GW": gateway},
    )

    assert len(result.selections) == 2
    assert node.default_ptx == node.fixed_ptx == 7
    assert relay.default_ptx == relay.fixed_ptx == 12
    assert node.ptx_mode == relay.ptx_mode == "fixed"
    assert node.tx_power_feedback_enabled is False
    assert relay.tx_power_feedback_enabled is False
    assert random.getstate() == random_state

    frame = build_final_ptx_distribution_df(
        [node, relay], result.run_metadata, lambda entity_id: entity_id[0]
    )
    assert set(frame["tx_power_model_id"]) == {"islam_dbpc_itm0"}
    assert set(frame["initial_ptx"]) == {7, 12}


def test_legacy_final_ptx_schema_is_unchanged():
    node = entity("N1", SX1276(), 0, gain=2.15, max_eirp=20.0)
    frame = build_final_ptx_distribution_df([node], {}, lambda _entity_id: "node")

    assert "tx_power_model_id" not in frame.columns
    assert "calibration_id" not in frame.columns


@pytest.mark.extended_research
def test_dbpc_rejects_runtime_configuration_not_used_by_calibration():
    with pytest.raises(CalibrationIntegrityError, match="node_ant_height"):
        apply_tx_power_policy(
            config=dbpc_config(node_ant_height=3.0),
            nodes=[],
            relays=[],
            gateway=entity("GW", SX1302(), 0, gain=5.8, max_eirp=26.0),
            routing_map={"GW": None},
            node_map={},
        )


def test_dbpc_policy_is_rejected_outside_comparison_profile():
    with pytest.raises(ValueError, match="apc_compare_model_v1"):
        dbpc_config(experiment_profile="general_multisf_multibw")
