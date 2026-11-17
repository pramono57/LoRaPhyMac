from __future__ import annotations

from types import SimpleNamespace

import pytest

from loraphymac.config.defaults import build_config
from loraphymac.config.validation import ConfigurationError, validate_configuration
from loraphymac.experiments.apc_compare_model.run import (
    DEFAULT_CONFIG,
    build_model_experiment_spec,
    load_comparison_definition,
)
from loraphymac.hardware.hardware import SX1276, SX1302, apply_eirp_ptx_bounds
from loraphymac.metrics.apc_metrics import build_final_ptx_distribution_df
from loraphymac.phy.channel_plan import AS923_2
from loraphymac.phy.propagation import ITM
from loraphymac.runtime.islam_dbpc import ISLAM_DBPC_POLICY_ID
from loraphymac.runtime.islam_dbpc_802 import (
    ISLAM_DBPC_802_ESTIMATOR_ID,
    ISLAM_DBPC_802_MODEL_ID,
    ISLAM_DBPC_802_POLICY_ID,
    IslamDBPC802Controller,
)
from loraphymac.runtime.tx_power_policy import apply_tx_power_policy
from loraphymac.simulation.builders import build_infrastructure


pytestmark = pytest.mark.unit


class RecordingEstimator:
    def __init__(self, path_loss_db):
        self.path_loss_db = float(path_loss_db)
        self.calls = []

    def pathloss(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self.path_loss_db


def entity(entity_id, hardware, x, *, gain, max_eirp):
    value = SimpleNamespace(
        id=entity_id,
        hw=hardware,
        loc=SimpleNamespace(x=float(x), y=0.0),
        h=2.0 if entity_id.startswith("N") else 25.0,
        gain=float(gain),
        ch_plan=AS923_2(),
        default_ptx=999,
        fixed_ptx=999,
        ptx_mode="adaptive",
    )
    value.distance_to = lambda other: abs(value.loc.x - other.loc.x)
    apply_eirp_ptx_bounds(value, max_eirp, gain, 0.0)
    return value


def dbpc_802_config(**overrides):
    config = build_config(
        {
            "experiment_profile": "apc_compare_model_v1",
            "energy_accounting_mode": "corrected",
            "node_ptx_mode": "fixed",
            "relay_ptx_mode": "fixed",
            "node_sf_list": [7],
            "relay_sf": 7,
            "routing_assumed_sf": 7,
            "node_bw": 125_000,
            "relay_bw_hz": 125_000,
            "tx_power_policy_id": ISLAM_DBPC_802_POLICY_ID,
            "tx_power_model_id": ISLAM_DBPC_802_MODEL_ID,
            "tx_power_calibration_margin_db": 0,
            **overrides,
        }
    )
    validate_configuration(config)
    return config


def topology(node_distance=1_000, relay_distance=18_000):
    node = entity("N1", SX1276(), 0, gain=2.15, max_eirp=20.0)
    relay = entity("R1", SX1302(), node_distance, gain=5.8, max_eirp=26.0)
    gateway = entity(
        "GW",
        SX1302(),
        node_distance + relay_distance,
        gain=5.8,
        max_eirp=26.0,
    )
    return node, relay, gateway


def apply_policy(config=None, *, node_distance=1_000, relay_distance=18_000):
    node, relay, gateway = topology(node_distance, relay_distance)
    result = apply_tx_power_policy(
        config=dbpc_802_config() if config is None else config,
        nodes=[node],
        relays=[relay],
        gateway=gateway,
        routing_map={"N1": "R1", "R1": "GW", "GW": None},
        node_map={"N1": node, "R1": relay, "GW": gateway},
    )
    return result, node, relay, gateway


def test_dbpc_802_calls_ieee_estimator_with_distance_and_configured_frequency_only():
    node, relay, _gateway = topology()
    estimator = RecordingEstimator(136.0)
    controller = IslamDBPC802Controller(ISLAM_DBPC_802_MODEL_ID, estimator)

    selection = controller.select(
        entity=node,
        entity_type="node",
        parent=relay,
        distance_m=1_000,
        frequency_mhz=921.4,
        sf=7,
        bandwidth_hz=125_000,
    )

    assert estimator.calls == [{"d_m": 1_000.0, "freq_mhz": 921.4}]
    assert selection.selected_ptx_dbm == 5


def test_dbpc_802_selects_minimum_feasible_power_and_maximum_fallback():
    result, node, relay, _gateway = apply_policy()

    assert [item.selected_ptx_dbm for item in result.selections] == [5, 20]
    assert node.ptx_min == 5 and node.ptx_max == 17
    assert relay.ptx_min == 12 and relay.ptx_max == 20
    assert result.selections[0].outside_estimated_coverage is False
    assert result.selections[1].outside_estimated_coverage is True
    assert node.default_ptx == node.fixed_ptx == 5
    assert relay.default_ptx == relay.fixed_ptx == 20


def test_dbpc_802_algorithm2_boundary_is_inclusive():
    node, relay, _gateway = topology()
    sensitivity = relay.hw.sensitivity[125][7]
    estimator = RecordingEstimator(9 + node.gain + relay.gain - sensitivity)
    selection = IslamDBPC802Controller(
        ISLAM_DBPC_802_MODEL_ID, estimator
    ).select(
        entity=node,
        entity_type="node",
        parent=relay,
        distance_m=1_000,
        frequency_mhz=921.4,
        sf=7,
        bandwidth_hz=125_000,
    )

    assert selection.estimated_rssi_dbm == pytest.approx(-126.0, abs=1e-12)
    assert selection.estimated_margin_db == pytest.approx(0.0, abs=1e-12)
    assert selection.selected_ptx_dbm == 9
    assert selection.estimated_feasible is True


@pytest.mark.parametrize("terrain_delta_h", [20, 90, 160])
def test_dbpc_802_is_independent_of_delta_h_for_fixed_geometry_and_route(
    terrain_delta_h,
):
    result, *_ = apply_policy(dbpc_802_config(terrain_delta_h=terrain_delta_h))

    assert [item.estimated_path_loss_db for item in result.selections] == pytest.approx(
        [136.31431966802407, 183.5125658599084]
    )
    assert [item.selected_ptx_dbm for item in result.selections] == [5, 20]


def test_dbpc_802_distance_and_frequency_affect_estimated_loss_without_itm_input():
    node, relay, _gateway = topology()
    controller = IslamDBPC802Controller(ISLAM_DBPC_802_MODEL_ID)
    near = controller.select(
        entity=node,
        entity_type="node",
        parent=relay,
        distance_m=1_000,
        frequency_mhz=868.0,
        sf=7,
        bandwidth_hz=125_000,
    )
    far = controller.select(
        entity=node,
        entity_type="node",
        parent=relay,
        distance_m=2_000,
        frequency_mhz=921.4,
        sf=7,
        bandwidth_hz=125_000,
    )

    assert far.estimated_path_loss_db > near.estimated_path_loss_db
    assert far.selected_ptx_dbm > near.selected_ptx_dbm


def test_dbpc_802_outputs_estimator_provenance_and_disables_feedback():
    result, node, relay, _gateway = apply_policy()
    frame = build_final_ptx_distribution_df(
        [node, relay], result.run_metadata, lambda entity_id: entity_id[0]
    )

    assert result.run_metadata["tx_power_estimator_id"] == ISLAM_DBPC_802_ESTIMATOR_ID
    assert result.run_metadata["tx_power_outside_coverage_entity_count"] == 1
    assert set(frame["estimator_id"]) == {ISLAM_DBPC_802_ESTIMATOR_ID}
    assert set(frame["estimator_frequency_mhz"]) == {921.4}
    assert set(frame["policy_margin_db"]) == {0.0}
    assert node.tx_power_feedback_enabled is False
    assert relay.tx_power_feedback_enabled is False


def test_dbpc_802_registry_is_additive_and_historical_plan_is_unchanged():
    definition = load_comparison_definition(DEFAULT_CONFIG)
    spec = build_model_experiment_spec(DEFAULT_CONFIG, model_id=ISLAM_DBPC_802_MODEL_ID)

    assert ISLAM_DBPC_802_MODEL_ID in definition.models
    assert definition.planned_execution_model_ids == ("islam_dbpc_itm0",)
    assert spec.base_config["tx_power_policy_id"] == ISLAM_DBPC_802_POLICY_ID
    assert spec.base_config["tx_power_calibration_margin_db"] == 0
    assert spec.base_config.get("tx_power_calibration_manifest") is None


def test_dbpc_802_rejects_calibration_artifacts_and_nonzero_margin():
    with pytest.raises(ConfigurationError, match="analytic"):
        dbpc_802_config(tx_power_calibration_id="not-allowed")
    with pytest.raises(ConfigurationError, match="must be 0 dB"):
        dbpc_802_config(tx_power_calibration_margin_db=6)


def test_dbpc_802_does_not_change_physical_itm_or_existing_policy_identity():
    infrastructure = build_infrastructure(dbpc_802_config())

    assert isinstance(infrastructure.model, ITM)
    assert ISLAM_DBPC_POLICY_ID == "islam_algorithm2_fixed_sf_ptx_v1"
