from collections import defaultdict

import pytest

from loraphymac.mac.collision_engine import CollisionEngine
from loraphymac.engine import (
    build_config,
    validate_experiment_profile,
)
from loraphymac.metrics.statistics import Statistics
from loraphymac.runtime.relay_rx import RelayRX
from tests.helpers.collision_fakes import make_collision_packet


pytestmark = pytest.mark.unit


def packet_ids(packets):
    return [packet.packet_id for packet in packets]


def relay_rx_window_registry():
    relay_rx = RelayRX.__new__(RelayRX)
    relay_rx.collision = CollisionEngine()
    relay_rx.demod_windows = defaultdict(list)
    relay_rx.demod_leader_busy = set()
    relay_rx.demod_window_sequence = 0
    return relay_rx


def test_rakibul_table4_is_versioned_and_directional():
    assert CollisionEngine.RAKIBUL_MATRIX_VERSION == (
        "rakibul_islam_2023_table4"
    )
    assert CollisionEngine.RAKIBUL_2023_SIR_MATRIX == {
        7: {7: 6.0, 8: -16.0, 9: -18.0, 10: -19.0, 11: -19.0, 12: -20.0},
        8: {7: -24.0, 8: 6.0, 9: -20.0, 10: -22.0, 11: -22.0, 12: -22.0},
        9: {7: -27.0, 8: -27.0, 9: 6.0, 10: -23.0, 11: -25.0, 12: -25.0},
        10: {7: -30.0, 8: -30.0, 9: -30.0, 10: 6.0, 11: -26.0, 12: -28.0},
        11: {7: -33.0, 8: -33.0, 9: -33.0, 10: -33.0, 11: 6.0, 12: -29.0},
        12: {7: -36.0, 8: -36.0, 9: -36.0, 10: -36.0, 11: -36.0, 12: 6.0},
    }


def test_same_logical_channel_without_spectral_overlap_survives():
    first = make_collision_packet("P1", ch_id=0, freq_mhz=923.2)
    second = make_collision_packet("P2", ch_id=0, freq_mhz=923.4)

    engine = CollisionEngine()
    survivors, failed = engine.resolve([first, second])

    assert packet_ids(survivors) == ["P1", "P2"]
    assert failed == []
    assert engine.last_interference_edge_records == []


def test_different_logical_channels_with_spectral_overlap_interfere():
    first = make_collision_packet("P1", ch_id=0, freq_mhz=923.2)
    second = make_collision_packet("P2", ch_id=7, freq_mhz=923.2)

    engine = CollisionEngine()
    survivors, failed = engine.resolve([first, second])

    assert survivors == []
    assert packet_ids(failed) == ["P1", "P2"]
    assert len(engine.last_interference_edge_records) == 2


def test_runtime_windows_keep_disjoint_spectra_independent():
    relay_rx = relay_rx_window_registry()
    first = make_collision_packet("P1", ch_id=0, freq_mhz=923.2)
    second = make_collision_packet("P2", ch_id=1, freq_mhz=923.4)

    first_key, _ = relay_rx._register_rf_window_packet(first)
    second_key, _ = relay_rx._register_rf_window_packet(second)

    assert first_key != second_key


def test_runtime_window_merges_multiple_components_when_wide_bw_bridges_them():
    relay_rx = relay_rx_window_registry()
    low = make_collision_packet("low", freq_mhz=923.1, bw_hz=125_000)
    high = make_collision_packet("high", freq_mhz=923.3, bw_hz=125_000)
    bridge = make_collision_packet(
        "bridge",
        freq_mhz=923.2,
        bw_hz=250_000,
    )

    low_key, _ = relay_rx._register_rf_window_packet(low)
    high_key, _ = relay_rx._register_rf_window_packet(high)
    bridge_key, merged = relay_rx._register_rf_window_packet(bridge)

    assert low_key != high_key
    assert bridge_key == low_key
    assert packet_ids(merged) == ["low", "high", "bridge"]
    assert relay_rx.demod_windows[high_key] == []


def test_partial_multi_bw_overlap_scales_interferer_power():
    narrow = make_collision_packet(
        "narrow",
        bw_hz=125_000,
        rssi=-100.0,
    )
    wide = make_collision_packet(
        "wide",
        bw_hz=500_000,
        rssi=-100.0,
    )

    engine = CollisionEngine()
    survivors, failed = engine.resolve([narrow, wide])
    narrow_edge = next(
        row
        for row in engine.last_interference_edge_records
        if row["desired_packet_id"] == "narrow"
    )

    assert packet_ids(survivors) == ["narrow"]
    assert packet_ids(failed) == ["wide"]
    assert narrow_edge["spectral_overlap_hz"] == pytest.approx(125_000.0)
    assert narrow_edge["spectral_overlap_fraction"] == pytest.approx(0.25)
    assert narrow_edge["effective_interferer_rssi_dbm"] == pytest.approx(
        -106.020599913,
        abs=1e-9,
    )
    assert narrow_edge["pair_sir_db"] == pytest.approx(
        6.020599913,
        abs=1e-9,
    )
    assert narrow_edge["pair_pass"] is True


def test_directional_cross_sf_matrix_changes_each_desired_packet_independently():
    matrix = {
        7: {7: 6.0, 8: 5.0},
        8: {7: -5.0, 8: 6.0},
    }
    sf7 = make_collision_packet("SF7", sf=7, rssi=-100.0)
    sf8 = make_collision_packet("SF8", sf=8, rssi=-100.0)
    engine = CollisionEngine(
        sf_interference_mode="sir_matrix",
        sir_matrix=matrix,
        sir_matrix_version="directionality_test_v1",
    )

    survivors, failed = engine.resolve([sf7, sf8])

    assert packet_ids(survivors) == ["SF8"]
    assert packet_ids(failed) == ["SF7"]
    edges = {
        row["desired_packet_id"]: row
        for row in engine.last_interference_edge_records
    }
    assert edges["SF7"]["required_sir_db"] == 5.0
    assert edges["SF7"]["pair_pass"] is False
    assert edges["SF8"]["required_sir_db"] == -5.0
    assert edges["SF8"]["pair_pass"] is True


def test_dominant_interferer_uses_smallest_margin_not_highest_raw_rssi():
    matrix = {
        7: {7: 6.0, 8: -10.0, 9: 12.0},
        8: {7: -100.0, 8: 6.0, 9: -100.0},
        9: {7: -100.0, 8: -100.0, 9: 6.0},
    }
    desired = make_collision_packet("D", sf=7, rssi=-80.0)
    stronger_raw = make_collision_packet("I8", sf=8, rssi=-85.0)
    weaker_raw = make_collision_packet("I9", sf=9, rssi=-90.0)
    engine = CollisionEngine(
        sf_interference_mode="sir_matrix",
        sir_matrix=matrix,
        sir_matrix_version="dominance_test_v1",
    )

    engine.resolve([desired, stronger_raw, weaker_raw])
    decision = next(
        row
        for row in engine.last_decision_records
        if row["packet_id"] == "D"
    )

    assert decision["strongest_interferer_packet_id"] == "I8"
    assert decision["dominant_interferer_packet_id"] == "I9"
    assert decision["min_pair_margin_db"] == pytest.approx(-2.0)
    assert decision["survived"] is False


def test_edge_statistics_are_schema_complete_and_conserve_directed_edges():
    first = make_collision_packet("P1", rssi=-90.0)
    second = make_collision_packet("P2", rssi=-100.0)
    stats = Statistics()
    engine = CollisionEngine()

    engine.resolve(
        [first, second],
        receiver_id="R1",
        event_time=2.0,
        statistics=stats,
    )
    decisions = stats.get_collision_event_log_df()
    edges = stats.get_collision_interference_edges_df()

    assert len(edges) == 2
    assert int(decisions["direct_interferer_count"].sum()) == len(edges)
    assert int(edges["dominant_interferer"].sum()) == 2
    assert set(
        [
            "spectral_overlap_hz",
            "spectral_overlap_fraction",
            "required_sir_db",
            "pair_sir_db",
            "pair_margin_db",
            "pair_pass",
        ]
    ).issubset(edges.columns)


def test_paper1_sf7_bw125_outcomes_match_ideal_legacy_profile():
    def run(sf_mode):
        packets = [
            make_collision_packet("A", rssi=-80.0),
            make_collision_packet("B", rssi=-90.0),
            make_collision_packet("C", rssi=-100.0),
        ]
        engine = CollisionEngine(sf_interference_mode=sf_mode)
        survivors, failed = engine.resolve(packets)
        return packet_ids(survivors), packet_ids(failed), [
            row["capture_delta_db"] for row in engine.last_decision_records
        ]

    legacy_result = run("ideal_orthogonal")
    matrix_result = run("sir_matrix")
    assert matrix_result == legacy_result


def test_general_profile_allows_multi_sf_and_multi_bw_configuration():
    cfg = build_config({
        "experiment_profile": "general_multisf_multibw",
        "node_sf_list": [7, 8, 10, 12],
        "node_bw": 250_000,
        "relay_sf": 12,
        "relay_bw_hz": 500_000,
    })
    validate_experiment_profile(cfg)


def test_paper1_profile_accepts_only_sf7_bw125_and_six_db_diagonal():
    cfg = build_config({"experiment_profile": "paper1_sf7_bw125"})
    validate_experiment_profile(cfg)


@pytest.mark.parametrize(
    "override",
    [
        {"node_sf_list": [7, 8]},
        {"node_bw": 250_000},
        {"relay_sf": 8},
        {"relay_bw_hz": 250_000},
        {"capture_threshold_db": 5.0, "collision_sf_interference_mode": "ideal_orthogonal"},
    ],
)
def test_paper1_profile_rejects_parameter_drift(override):
    cfg = build_config({
        "experiment_profile": "paper1_sf7_bw125",
        **override,
    })
    with pytest.raises(ValueError, match="paper1_sf7_bw125 profile violation"):
        validate_experiment_profile(cfg)


def test_custom_sir_matrix_requires_version():
    with pytest.raises(ValueError, match="requires.*sir_matrix_version"):
        CollisionEngine(
            sf_interference_mode="sir_matrix",
            sir_matrix={7: {7: 6.0}},
        )
