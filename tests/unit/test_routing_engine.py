import math

import pandas as pd
import pytest

from loraphymac.routing.routing_engine import RoutingEngine
from tests.helpers.routing_fakes import (
    DistancePathlossModel,
    FakeRoutingEntity,
    RecordingLinkBudgetAir,
    assert_terminal_route,
    make_topology,
)


pytestmark = pytest.mark.unit

LINK_COLUMNS = [
    "Node", "Target", "ChID", "Freq(MHz)", "Distance(m)",
    "Pathloss(dB)", "RSSI(dBm)", "SNR(dB)", "Margin(dB)",
    "Hop", "Score", "Valid",
]


@pytest.mark.parametrize(
    ("positions", "expected"),
    [
        ([], {"GW": 0}),
        ([10_000], {"R1": 1, "GW": 0}),
        ([20_000, 10_000], {"R1": 2, "R2": 1, "GW": 0}),
        ([30_000, 10_000, 20_000], {"R1": 3, "R2": 1, "R3": 2, "GW": 0}),
    ],
)
def test_compute_relay_hops_orders_by_absolute_x_distance(positions, expected):
    gateway = FakeRoutingEntity("GW", 0)
    relays = [FakeRoutingEntity(f"R{i}", x) for i, x in enumerate(positions, 1)]
    original = list(relays)
    assert RoutingEngine().compute_relay_hops(relays, gateway) == expected
    assert relays == original


def test_relay_hops_are_relative_to_gateway_x_not_origin():
    gateway = FakeRoutingEntity("GW", 100)
    near = FakeRoutingEntity("R1", 90)
    far = FakeRoutingEntity("R2", -100)
    assert RoutingEngine().compute_relay_hops([far, near], gateway) == {
        "R1": 1, "R2": 2, "GW": 0,
    }


@pytest.mark.parametrize("relay_count", [0, 1, 2, 3])
def test_relay_chain_points_each_hop_toward_gateway(relay_count):
    gateway = FakeRoutingEntity("GW", 0)
    relays = [FakeRoutingEntity(f"R{i}", i * 10_000) for i in range(1, relay_count + 1)]
    engine = RoutingEngine()
    hops = engine.compute_relay_hops(relays, gateway)
    chain = engine.build_relay_chain(relays, hops, gateway)
    assert chain["GW"] is None
    for i, relay in enumerate(relays, start=1):
        assert chain[relay.id] == ("GW" if i == 1 else f"R{i - 1}")
        assert chain[relay.id] != relay.id


def test_build_routing_map_combines_nodes_relays_and_terminal_gateway():
    node, relays, gateway = make_topology()
    engine = RoutingEngine()
    hops = engine.compute_relay_hops(relays, gateway)
    routing_map = engine.build_routing_map(relays, hops, {node.id: "R2"}, gateway)
    assert routing_map == {"N1": "R2", "R1": "GW", "R2": "R1", "GW": None}
    path = engine.get_full_path("N1", routing_map)
    assert path == ["N1", "R2", "R1", "GW"]
    assert_terminal_route(path)


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([{"Node": "N1", "Target": "GW", "Score": -100}], "GW"),
        (
            [
                {"Node": "N1", "Target": "R1", "Score": -110},
                {"Node": "N1", "Target": "GW", "Score": -100},
            ],
            "GW",
        ),
        (
            [
                {"Node": "N1", "Target": "R1", "Score": -90},
                {"Node": "N1", "Target": "GW", "Score": -100},
            ],
            "R1",
        ),
    ],
)
def test_routing_table_selects_highest_score(rows, expected):
    assert RoutingEngine().build_routing_table(pd.DataFrame(rows))["N1"] == expected


def test_routing_table_selects_one_target_per_node():
    frame = pd.DataFrame(
        [
            {"Node": "N1", "Target": "GW", "Score": -100},
            {"Node": "N1", "Target": "R1", "Score": -90},
            {"Node": "N2", "Target": "GW", "Score": -80},
            {"Node": "N2", "Target": "R1", "Score": -85},
        ]
    )
    assert RoutingEngine().build_routing_table(frame) == {"N1": "R1", "N2": "GW"}


@pytest.mark.parametrize("alpha", [0, 1, 2, 10])
def test_evaluate_links_applies_independent_rssi_minus_hop_penalty(alpha):
    node, relays, gateway = make_topology(relay_positions=(10_000, 20_000), node_x=30_000)
    engine = RoutingEngine(alpha)
    hops = engine.compute_relay_hops(relays, gateway)
    # Distances N1-R1=20km, N1-R2=10km, N1-GW=30km.
    model = DistancePathlossModel({20_000.0: 120, 10_000.0: 110, 30_000.0: 130})
    air = RecordingLinkBudgetAir(140)
    frame = engine.evaluate_links(
        node, relays, hops, gateway, model, air,
        sf=7, bw_hz=125_000, ptx=20, freq_mhz=921.4,
    )
    assert list(frame.columns) == LINK_COLUMNS
    for row in frame.to_dict("records"):
        assert row["Score"] == pytest.approx(
            row["RSSI(dBm)"] - alpha * row["Hop"], abs=1e-12
        )
        assert math.isfinite(row["Score"])


def test_hop_penalty_can_change_relay_to_gateway_choice():
    rows_alpha_zero = pd.DataFrame(
        [
            {"Node": "N1", "Target": "R2", "Score": -90},
            {"Node": "N1", "Target": "GW", "Score": -91},
        ]
    )
    assert RoutingEngine(0).build_routing_table(rows_alpha_zero)["N1"] == "R2"

    # Independent score: relay RSSI - alpha*hop = -90 - 2*2 = -94.
    rows_penalized = pd.DataFrame(
        [
            {"Node": "N1", "Target": "R2", "Score": -94},
            {"Node": "N1", "Target": "GW", "Score": -91},
        ]
    )
    assert RoutingEngine(2).build_routing_table(rows_penalized)["N1"] == "GW"


def test_evaluate_links_forwards_phy_parameters_and_channel_lookup():
    node, relays, gateway = make_topology(relay_positions=(10_000,), node_x=20_000)
    engine = RoutingEngine(2)
    hops = engine.compute_relay_hops(relays, gateway)
    model = DistancePathlossModel({10_000.0: 120, 20_000.0: 130})
    air = RecordingLinkBudgetAir()

    class ChannelPlan:
        def __init__(self):
            self.calls = []

        def get_freq(self, ch_id):
            self.calls.append(ch_id)
            return 922.8

    plan = ChannelPlan()
    frame = engine.evaluate_links(
        node, relays, hops, gateway, model, air,
        sf=12, bw_hz=250_000, ptx=17, ch_id=7, channel_plan=plan,
        terrain_delta_h=200,
    )
    assert plan.calls == [7]
    assert set(frame["ChID"]) == {7}
    assert set(frame["Freq(MHz)"]) == {922.8}
    assert all(call["sf"] == 12 for call in air.calls)
    assert all(call["bw_hz"] == 250_000 for call in air.calls)
    assert all(call["ptx"] == 17 for call in air.calls)
    assert all(call["terrain_delta_h"] == 200 for call in model.calls)


def test_evaluate_links_requires_frequency_or_channel_plan():
    node, relays, gateway = make_topology(relay_positions=(10_000,), node_x=20_000)
    with pytest.raises(ValueError, match="Provide freq_mhz"):
        RoutingEngine().evaluate_links(
            node, relays, {"R1": 1, "GW": 0}, gateway,
            DistancePathlossModel({}, default=120), RecordingLinkBudgetAir(),
            sf=7, bw_hz=125_000, ptx=20,
        )


@pytest.mark.parametrize(
    ("routing_map", "expected"),
    [
        ({"N1": "GW", "GW": None}, ["N1", "GW"]),
        ({"N1": "R1", "R1": "GW", "GW": None}, ["N1", "R1", "GW"]),
        (
            {"N1": "R2", "R2": "R1", "R1": "GW", "GW": None},
            ["N1", "R2", "R1", "GW"],
        ),
    ],
)
def test_get_full_path_for_terminal_routes(routing_map, expected):
    assert RoutingEngine().get_full_path("N1", routing_map) == expected


@pytest.mark.parametrize(
    ("routing_map", "exception", "match"),
    [
        ({}, KeyError, "Entity N1"),
        ({"N1": "UNKNOWN"}, KeyError, "Target UNKNOWN"),
        ({"N1": "R1", "R1": "N1"}, ValueError, "Routing cycle"),
    ],
)
def test_get_full_path_rejects_invalid_topology(routing_map, exception, match):
    with pytest.raises(exception, match=match):
        RoutingEngine().get_full_path("N1", routing_map)


def test_validate_linear_relay_order_accepts_r1_nearest_gateway():
    _, relays, gateway = make_topology(relay_positions=(10_000, 20_000))
    engine = RoutingEngine()
    hops = engine.compute_relay_hops(relays, gateway)
    assert engine.validate_linear_relay_order(relays, hops, gateway) is True


def test_validate_linear_relay_order_rejects_swapped_relay_labels():
    _, relays, gateway = make_topology(relay_positions=(20_000, 10_000))
    engine = RoutingEngine()
    hops = engine.compute_relay_hops(relays, gateway)
    with pytest.raises(ValueError, match="R1 nearest"):
        engine.validate_linear_relay_order(relays, hops, gateway)


def test_validate_linear_relay_order_rejects_equal_relay_distances():
    _, relays, gateway = make_topology(relay_positions=(10_000, 10_000))
    engine = RoutingEngine()
    hops = engine.compute_relay_hops(relays, gateway)
    with pytest.raises(ValueError, match="increase strictly"):
        engine.validate_linear_relay_order(relays, hops, gateway)


def test_validate_routing_map_accepts_topological_route_without_phy_filter():
    routing_map = {"N1": "R2", "R2": "R1", "R1": "GW", "GW": None}
    assert RoutingEngine().validate_routing_map(
        routing_map,
        node_ids=["N1"],
        relay_ids=["R1", "R2"],
        gateway_id="GW",
    ) is True


@pytest.mark.parametrize(
    ("routing_map", "match"),
    [
        ({"N1": "UNKNOWN", "R1": "GW", "GW": None}, "unknown target"),
        ({"N1": "R1", "R1": "N1", "GW": None}, "Routing cycle"),
        ({"N1": None, "R1": "GW", "GW": None}, "not at gateway"),
    ],
)
def test_validate_routing_map_rejects_broken_topology(routing_map, match):
    with pytest.raises(ValueError, match=match):
        RoutingEngine().validate_routing_map(
            routing_map,
            node_ids=["N1"],
            relay_ids=["R1"],
            gateway_id="GW",
        )


def test_evaluate_relay_chain_links_matches_routing_map_and_columns():
    node, relays, gateway = make_topology(relay_positions=(10_000, 20_000))
    engine = RoutingEngine(2)
    hops = engine.compute_relay_hops(relays, gateway)
    routing_map = engine.build_routing_map(relays, hops, {node.id: "R2"}, gateway)
    node_map = {entity.id: entity for entity in [node, *relays, gateway]}
    model = DistancePathlossModel({10_000.0: 120})
    frame = engine.evaluate_relay_chain_links(
        relays, routing_map, node_map, hops, gateway, model,
        RecordingLinkBudgetAir(), sf=7, bw_hz=125_000, ptx=27,
        freq_mhz=921.6, ch_id=1,
    )
    assert list(frame.columns) == LINK_COLUMNS
    assert frame[["Node", "Target"]].to_dict("records") == [
        {"Node": "R1", "Target": "GW"},
        {"Node": "R2", "Target": "R1"},
    ]
    assert frame["Score"].equals(frame["RSSI(dBm)"])


def test_empty_relay_chain_returns_schema_without_requiring_ptx():
    gateway = FakeRoutingEntity("GW", 0)
    frame = RoutingEngine().evaluate_relay_chain_links(
        [], {"GW": None}, {"GW": gateway}, {"GW": 0}, gateway,
        DistancePathlossModel({}, default=120), RecordingLinkBudgetAir(),
        sf=7, bw_hz=125_000, freq_mhz=921.6,
    )
    assert frame.empty
    assert list(frame.columns) == LINK_COLUMNS


def test_relay_chain_evaluation_uses_each_relay_max_ptx_by_default():
    node, relays, gateway = make_topology(relay_positions=(10_000, 20_000))
    relays[0].ptx_max = 18
    relays[1].ptx_max = 24
    engine = RoutingEngine()
    hops = engine.compute_relay_hops(relays, gateway)
    routing_map = engine.build_routing_map(relays, hops, {node.id: "R2"}, gateway)
    node_map = {entity.id: entity for entity in [node, *relays, gateway]}
    air = RecordingLinkBudgetAir()

    engine.evaluate_relay_chain_links(
        relays, routing_map, node_map, hops, gateway,
        DistancePathlossModel({}, default=120), air,
        sf=7, bw_hz=125_000, freq_mhz=921.6,
    )

    assert [call["ptx"] for call in air.calls] == [18, 24]


def test_relay_chain_link_rejects_unknown_target():
    relay = FakeRoutingEntity("R1", 10_000)
    gateway = FakeRoutingEntity("GW", 0)
    with pytest.raises(KeyError, match="is missing from node_map"):
        RoutingEngine().evaluate_relay_chain_links(
            [relay], {"R1": "MISSING", "GW": None}, {"R1": relay, "GW": gateway},
            {"R1": 1, "GW": 0}, gateway, DistancePathlossModel({}, default=120),
            RecordingLinkBudgetAir(), sf=7, bw_hz=125_000, ptx=27, freq_mhz=921.6,
        )


def test_routing_operations_do_not_mutate_entities_lists_or_model_map():
    node, relays, gateway = make_topology()
    relay_order = list(relays)
    locations = [(entity.loc.x, entity.loc.y) for entity in [node, *relays, gateway]]
    losses = {10_000.0: 110, 20_000.0: 120, 30_000.0: 130}
    model = DistancePathlossModel(losses)
    engine = RoutingEngine()
    hops = engine.compute_relay_hops(relays, gateway)
    engine.evaluate_links(
        node, relays, hops, gateway, model, RecordingLinkBudgetAir(),
        sf=7, bw_hz=125_000, ptx=20, freq_mhz=921.4,
    )
    assert relays == relay_order
    assert [(entity.loc.x, entity.loc.y) for entity in [node, *relays, gateway]] == locations
    assert model.losses_by_distance == losses
