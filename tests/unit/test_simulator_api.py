from __future__ import annotations

import pandas as pd
import pytest

from loraphymac.config.validation import ConfigurationError
from loraphymac import (
    SIMULATOR_API_VERSION,
    SIMULATOR_VERSION,
    SimulationResult,
    Simulator,
    __version__,
    default_config,
    resolve_configuration,
    simulate,
)
from loraphymac import simulator as simulator_module


def test_public_api_returns_independent_validated_configurations():
    first = default_config()
    first["node_sf_list"].append(8)

    second = Simulator.default_config()
    resolved = resolve_configuration({"seed": 7, "node_num": 10})

    assert second["node_sf_list"] == [7]
    assert resolved["seed"] == 7
    assert resolved["node_num"] == 10
    assert __version__ == "1.0.0"
    assert SIMULATOR_VERSION == __version__
    assert Simulator.version == SIMULATOR_VERSION
    assert Simulator.api_version == SIMULATOR_API_VERSION


def test_public_api_rejects_unknown_configuration_before_execution():
    with pytest.raises(ConfigurationError, match="node_perid"):
        simulate({"node_perid": 60})
    with pytest.raises(ConfigurationError, match="mapping"):
        simulate([("seed", 7)])


def test_public_api_wraps_internal_runner_in_named_read_only_result(monkeypatch):
    ptx = pd.DataFrame([{"entity_id": "N1", "new_ptx": 10}])
    margin = pd.DataFrame([{"entity_id": "N1", "margin_db": 7.0}])
    packet_fate = pd.DataFrame([{"offered_count": 1, "delivered_count": 1}])
    observed = {}

    def fake_engine(config):
        observed.update(config)
        return (
            {"delivery_by_horizon_percent": 100.0},
            ptx,
            margin,
            {"packet_fate_summary_df": packet_fate},
        )

    monkeypatch.setattr(simulator_module, "_run_single_simulation", fake_engine)
    result = simulate({"run_id": "api-test", "seed": 11})

    assert isinstance(result, SimulationResult)
    assert result.run_id == "api-test"
    assert observed == dict(result.resolved_config)
    assert result.summary["delivery_by_horizon_percent"] == 100.0
    assert result.ptx_history.equals(ptx)
    assert result.margin_history.equals(margin)
    assert result.table("packet_fate_summary_df", copy=True).equals(packet_fate)
    assert result.table("packet_fate_summary_df", copy=True) is not packet_fate
    with pytest.raises(TypeError):
        result.summary["delivery_by_horizon_percent"] = 0.0

    legacy = result.to_legacy_tuple()
    assert legacy[0] == dict(result.summary)
    assert legacy[1] is ptx
    assert legacy[2] is margin
