"""Characterizes the intentional routing and runtime-feasibility contract."""

import inspect

import pandas as pd
import pytest

import loraphymac.engine as simulator
from loraphymac.routing.routing_engine import RoutingEngine
from loraphymac.simulation.builders import build_routing
from tests.helpers.routing_fakes import FakeRoutingEntity


pytestmark = pytest.mark.characterization


def test_invalid_high_score_link_is_selected_over_feasible_link_by_design():
    # Valid is diagnostic metadata; runtime RX performs admission control.
    frame = pd.DataFrame(
        [
            {"Node": "N1", "Target": "GW", "Score": -100, "Valid": True},
            {"Node": "N1", "Target": "R1", "Score": -90, "Valid": False},
        ]
    )
    assert RoutingEngine().build_routing_table(frame) == {"N1": "R1"}


def test_all_invalid_candidates_select_highest_score_to_preserve_offered_load():
    # An all-invalid node still gets a route and can contribute to PDR load.
    frame = pd.DataFrame(
        [
            {"Node": "N1", "Target": "GW", "Score": -100, "Valid": False},
            {"Node": "N1", "Target": "R1", "Score": -90, "Valid": False},
        ]
    )
    assert RoutingEngine().build_routing_table(frame) == {"N1": "R1"}


@pytest.mark.parametrize("valid_value", [None, "unknown"])
def test_missing_or_nonboolean_feasibility_is_intentionally_ignored(valid_value):
    frame = pd.DataFrame(
        [
            {"Node": "N1", "Target": "R1", "Score": 1, "Valid": valid_value},
            {"Node": "N1", "Target": "GW", "Score": 0, "Valid": True},
        ]
    )
    assert RoutingEngine().build_routing_table(frame)["N1"] == "R1"


def test_routing_table_does_not_require_valid_column():
    frame = pd.DataFrame([{"Node": "N1", "Target": "R1", "Score": 1}])
    assert RoutingEngine().build_routing_table(frame) == {"N1": "R1"}


@pytest.mark.parametrize(
    ("targets", "expected"),
    [(["GW", "R1"], "GW"), (["R1", "GW"], "R1"), (["R2", "R1", "GW"], "R2")],
)
def test_equal_score_tie_currently_uses_first_dataframe_row(targets, expected):
    # Stable outcome for a fixed DataFrame, but changing candidate order changes route.
    frame = pd.DataFrame(
        [{"Node": "N1", "Target": target, "Score": -100} for target in targets]
    )
    assert RoutingEngine().build_routing_table(frame)["N1"] == expected


def test_same_position_relay_order_currently_uses_input_order():
    gateway = FakeRoutingEntity("GW", 0)
    first = FakeRoutingEntity("R9", 10_000)
    second = FakeRoutingEntity("R1", 10_000)
    engine = RoutingEngine()
    assert engine.compute_relay_hops([first, second], gateway) == {
        "R9": 1, "R1": 2, "GW": 0,
    }
    assert engine.compute_relay_hops([second, first], gateway) == {
        "R1": 1, "R9": 2, "GW": 0,
    }


def test_relay_chain_is_built_even_without_link_feasibility_evidence():
    gateway = FakeRoutingEntity("GW", 0)
    relays = [FakeRoutingEntity("R1", 10_000), FakeRoutingEntity("R2", 20_000)]
    engine = RoutingEngine()
    hops = engine.compute_relay_hops(relays, gateway)
    assert engine.build_relay_chain(relays, hops, gateway) == {
        "R1": "GW", "R2": "R1", "GW": None,
    }


def test_gateway_is_forced_terminal_even_if_routing_table_contains_gateway_route():
    gateway = FakeRoutingEntity("GW", 0)
    result = RoutingEngine().build_routing_map([], {"GW": 0}, {"GW": "R1"}, gateway)
    assert result["GW"] is None


@pytest.mark.parametrize(
    "routing_map",
    [
        {"A": "A"},
        {"A": "B", "B": "A"},
        {"A": "B", "B": "C", "C": "A"},
    ],
)
def test_cycle_detection_fails_fast(routing_map):
    with pytest.raises(ValueError, match="Routing cycle"):
        RoutingEngine().get_full_path("A", routing_map)


def test_missing_route_fails_fast():
    with pytest.raises(KeyError, match="missing from routing_map"):
        RoutingEngine().get_full_path("N1", {})


def test_unknown_target_fails_fast():
    with pytest.raises(KeyError, match="Target UNKNOWN"):
        RoutingEngine().get_full_path("N1", {"N1": "UNKNOWN"})


def test_empty_link_dataframe_returns_empty_routing_table():
    assert RoutingEngine().build_routing_table(pd.DataFrame()) == {}


def test_duplicate_relay_id_fails_fast():
    gateway = FakeRoutingEntity("GW", 0)
    relays = [FakeRoutingEntity("R1", 20_000), FakeRoutingEntity("R1", 10_000)]
    with pytest.raises(ValueError, match="Relay IDs must be unique"):
        RoutingEngine().compute_relay_hops(relays, gateway)


def test_main_entrypoint_routes_with_node_max_ptx():
    entrypoint_source = inspect.getsource(simulator.run_single_simulation)
    routing_source = inspect.getsource(build_routing)
    assert "build_routing(cfg, infrastructure, topology)" in entrypoint_source
    assert 'sf=config["routing_assumed_sf"]' in routing_source
    assert "ptx=node.ptx_max" in routing_source
    assert 'ch_id=config["routing_ch_id"]' in routing_source
    assert "routing_assumed_ptx" not in simulator.DEFAULT_CONFIG
    assert simulator.DEFAULT_CONFIG["routing_hop_penalty_alpha"] == 2


def test_current_default_routing_and_runtime_phy_can_differ():
    # Routing uses node max PTx and an evaluation SF/channel; runtime supports
    # node-specific/adaptive values.
    cfg = simulator.DEFAULT_CONFIG
    assert cfg["routing_assumed_sf"] == 7
    assert "routing_assumed_ptx" not in cfg
    assert cfg["routing_ch_id"] == 0
    assert cfg["node_sf_list"] == [7]
    assert cfg["node_ptx_mode"] == "adaptive"
    assert cfg["relay_tx_ch_ids"] == [1]
