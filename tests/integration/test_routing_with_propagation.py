import simpy
import pytest

from loraphymac.phy.air_interface import AirInterface
from loraphymac.phy.propagation import FSPL
from loraphymac.engine import run_single_simulation
from loraphymac.routing.routing_engine import RoutingEngine
from tests.helpers.routing_fakes import (
    DistancePathlossModel,
    FakeRoutingEntity,
    RecordingLinkBudgetAir,
    assert_terminal_route,
    make_topology,
)


pytestmark = pytest.mark.integration


def evaluate(engine, node, relays, gateway, model, air, ptx=20, sf=7):
    hops = engine.compute_relay_hops(relays, gateway)
    links = engine.evaluate_links(
        node, relays, hops, gateway, model, air,
        sf=sf, bw_hz=125_000, ptx=ptx, freq_mhz=921.4,
    )
    table = engine.build_routing_table(links)
    routing_map = engine.build_routing_map(relays, hops, table, gateway)
    return links, routing_map, engine.get_full_path(node.id, routing_map)


def test_controlled_pathloss_builds_two_relay_terminal_route():
    node, relays, gateway = make_topology(relay_positions=(10_000, 20_000), node_x=30_000)
    model = DistancePathlossModel({10_000.0: 100, 20_000.0: 130, 30_000.0: 150})
    links, routing_map, path = evaluate(
        RoutingEngine(2), node, relays, gateway, model, RecordingLinkBudgetAir(140)
    )
    assert routing_map == {"N1": "R2", "R1": "GW", "R2": "R1", "GW": None}
    assert path == ["N1", "R2", "R1", "GW"]
    assert_terminal_route(path)
    selected = links[links["Target"] == routing_map["N1"]]
    assert len(selected) == 1


def test_fspl_production_routing_is_deterministic_for_simple_topology():
    node = FakeRoutingEntity("N1", 10_000, height=2, gain=2.15)
    relay = FakeRoutingEntity("R1", 6_000, height=25, gain=5.8)
    gateway = FakeRoutingEntity("GW", 0, height=25, gain=5.8)

    def run_once():
        links, routing_map, path = evaluate(
            RoutingEngine(2), node, [relay], gateway, FSPL(),
            AirInterface(simpy.Environment(), feasible_margin_db=0),
        )
        return links[["Target", "Score", "Valid"]].to_dict("records"), routing_map, path

    first = run_once()
    second = run_once()
    assert first == second
    assert_terminal_route(first[2])


def test_hop_penalty_changes_selected_target_with_controlled_losses():
    node, relays, gateway = make_topology(relay_positions=(10_000, 20_000), node_x=30_000)
    # R2 RSSI is 1 dB stronger than GW, but R2 has hop=2.
    losses = {10_000.0: 119, 20_000.0: 140, 30_000.0: 120}
    _, map_alpha_zero, _ = evaluate(
        RoutingEngine(0), node, relays, gateway,
        DistancePathlossModel(losses), RecordingLinkBudgetAir(150),
    )
    _, map_alpha_two, _ = evaluate(
        RoutingEngine(2), node, relays, gateway,
        DistancePathlossModel(losses), RecordingLinkBudgetAir(150),
    )
    assert map_alpha_zero["N1"] == "R2"
    assert map_alpha_two["N1"] == "GW"


def test_higher_assumed_ptx_improves_feasibility_without_changing_equal_gain_ranking():
    node, relays, gateway = make_topology(relay_positions=(10_000,), node_x=20_000)
    model_low = DistancePathlossModel({10_000.0: 145, 20_000.0: 160})
    low_links, low_map, _ = evaluate(
        RoutingEngine(1), node, relays, gateway, model_low, RecordingLinkBudgetAir(150), ptx=10,
    )
    high_links, high_map, _ = evaluate(
        RoutingEngine(1), node, relays, gateway,
        DistancePathlossModel({10_000.0: 145, 20_000.0: 160}),
        RecordingLinkBudgetAir(150), ptx=20,
    )
    # Fake feasibility is pathloss-based, while score shifts equally with PTx.
    assert low_map == high_map
    assert (high_links["RSSI(dBm)"] - low_links["RSSI(dBm)"]).tolist() == [10, 10]


def test_receiver_sensitivity_can_change_valid_flag_with_production_air_interface():
    node = FakeRoutingEntity("N1", 10_000, gain=0)
    relay = FakeRoutingEntity("R1", 5_000, gain=0)
    gateway = FakeRoutingEntity("GW", 0, gain=0)
    # PTx 20 dBm and zero gains yield RSSI -127 dBm: below SX1302 SF7
    # sensitivity (-126) but above SF12 sensitivity (-137).
    model = DistancePathlossModel({5_000.0: 147, 10_000.0: 147})
    engine = RoutingEngine(0)
    hops = engine.compute_relay_hops([relay], gateway)
    links_sf7 = engine.evaluate_links(
        node, [relay], hops, gateway, model,
        AirInterface(simpy.Environment(), feasible_margin_db=0),
        sf=7, bw_hz=125_000, ptx=20, freq_mhz=921.4,
    )
    links_sf12 = engine.evaluate_links(
        node, [relay], hops, gateway,
        DistancePathlossModel({5_000.0: 147, 10_000.0: 147}),
        AirInterface(simpy.Environment(), feasible_margin_db=0),
        sf=12, bw_hz=125_000, ptx=20, freq_mhz=921.4,
    )
    assert links_sf7["Valid"].tolist() == [False, False]
    assert links_sf12["Valid"].tolist() == [True, True]


def test_no_relay_baseline_preserves_offered_load_and_runtime_feasibility_drop():
    summary, _, _, outputs = run_single_simulation({
        "sim_time": 2,
        "seed": 123,
        "node_num": 1,
        "relay_num": 0,
        "node_period": 1,
        "node_jitter": 0,
        "node_ptx_mode": "fixed",
        "relay_ptx_mode": "fixed",
        "node_rx_window_enabled": False,
        "plot_routing": False,
        "export_results": False,
        "export_debug": False,
        "export_internal": False,
    })

    selected = outputs["selected_next_hop_links_df"]
    assert selected[["Node", "Target"]].to_dict("records") == [
        {"Node": "N1", "Target": "GW"},
    ]
    assert selected["Valid"].tolist() == [False]
    assert outputs["relay_chain_links_df"].empty
    assert list(outputs["relay_chain_links_df"].columns) == [
        "Node", "Target", "ChID", "Freq(MHz)", "Distance(m)",
        "Pathloss(dB)", "RSSI(dBm)", "SNR(dB)", "Margin(dB)",
        "Hop", "Score", "Valid",
    ]
    assert outputs["routing_map_df"].to_dict("records") == [
        {"source": "N1", "next_hop": "GW"},
        {"source": "GW", "next_hop": None},
    ]

    # The invalid planning estimate does not suppress packet generation.
    # Runtime RX rejects it and the TX remains in the PDR denominator.
    assert summary["tx_packets"] == 1
    assert summary["rx_packets_gw"] == 0
    assert summary["drop_not_feasible_count"] == 1
    assert summary["pdr_percent"] == 0.0
