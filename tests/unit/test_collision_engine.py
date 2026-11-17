import pytest

from loraphymac.mac.collision_engine import CollisionEngine
from tests.helpers.collision_fakes import (
    assert_partition,
    make_collision_packet,
    reference_sinr_db,
)


pytestmark = pytest.mark.unit


def resolve(*packets, **engine_kwargs):
    engine = CollisionEngine(**engine_kwargs)
    survivors, failed = engine.resolve(list(packets))
    assert_partition(list(packets), survivors, failed)
    assert all(packet.reason == "collision" for packet in failed)
    assert all(packet.reason != "collision" for packet in survivors)
    return engine, survivors, failed


def test_empty_and_single_packet_partition():
    engine = CollisionEngine()
    assert engine.resolve([]) == ([], [])
    packet = make_collision_packet()
    survivors, failed = engine.resolve([packet])
    assert survivors == [packet]
    assert failed == []


@pytest.mark.parametrize(
    ("second_start", "second_end", "overlaps"),
    [
        (2.1, 3.0, False),
        (2.0, 3.0, False),
        (1.999999999, 3.0, True),
        (1.2, 1.8, True),
        (1.0, 2.0, True),
        (1.5, 1.500000001, True),
    ],
)
def test_time_overlap_boundaries(second_start, second_end, overlaps):
    first = make_collision_packet("P1", rssi=-100)
    second = make_collision_packet(
        "P2", rx_start=second_start, rx_end=second_end, rssi=-100
    )
    _, survivors, failed = resolve(first, second)
    assert len(failed) == (2 if overlaps else 0)
    assert len(survivors) == (0 if overlaps else 2)


@pytest.mark.parametrize(
    ("first_times", "second_times"),
    [((1.0, 1.0), (0.5, 1.5)), ((2.0, 1.0), (0.5, 3.0)), ((-2.0, -1.0), (-1.5, 0.0))],
)
def test_unusual_time_intervals_follow_overlap_predicate(first_times, second_times):
    first = make_collision_packet("P1", rx_start=first_times[0], rx_end=first_times[1])
    second = make_collision_packet("P2", rx_start=second_times[0], rx_end=second_times[1])
    expected = CollisionEngine._overlap(*first_times, *second_times)
    _, survivors, failed = resolve(first, second)
    assert (len(failed) > 0) is expected


def test_three_fully_overlapping_equal_packets_all_fail():
    packets = [make_collision_packet(f"P{i}", rssi=-100) for i in range(3)]
    _, survivors, failed = resolve(*packets)
    assert survivors == []
    assert len(failed) == 3


def test_transitive_overlap_builds_one_connected_group():
    a = make_collision_packet("A", rx_start=0, rx_end=2, rssi=-80)
    b = make_collision_packet("B", rx_start=1, rx_end=3, rssi=-100)
    c = make_collision_packet("C", rx_start=2.5, rx_end=4, rssi=-110)
    groups = CollisionEngine()._build_groups([a, b, c])
    assert len(groups) == 1
    assert {packet.packet_id for packet in groups[0]} == {"A", "B", "C"}


def test_pairwise_capture_does_not_treat_transitive_members_as_interferers():
    a = make_collision_packet("A", rx_start=0, rx_end=2, rssi=-80)
    b = make_collision_packet("B", rx_start=1, rx_end=3, rssi=-100)
    c = make_collision_packet("C", rx_start=2.5, rx_end=4, rssi=-90)
    engine = CollisionEngine(mode="capture", capture_threshold_dB=6)

    survivors, failed = engine.resolve(
        [a, b, c], receiver_id="R1", event_time=4.0
    )

    assert survivors == [a, c]
    assert failed == [b]
    assert a.captured is True
    assert b.captured is False
    assert c.captured is True

    records = {row["packet_id"]: row for row in engine.last_decision_records}
    assert records["A"]["direct_interferer_count"] == 1
    assert records["B"]["direct_interferer_count"] == 2
    assert records["C"]["direct_interferer_count"] == 1
    assert records["A"]["outcome"] == "CAPTURE_SURVIVOR"
    assert records["B"]["outcome"] == "COLLISION_DROP"
    assert records["C"]["outcome"] == "CAPTURE_SURVIVOR"
    assert records["C"]["strongest_interferer_packet_id"] == "B"
    assert records["C"]["capture_delta_db"] == pytest.approx(10.0)
    assert records["C"]["receiver_id"] == "R1"
    assert records["C"]["collision_model_version"] == "pairwise_spectral_sir_v3"


@pytest.mark.parametrize(
    ("ch2", "sf2", "collides"),
    [(0, 7, True), (1, 7, True), (0, 8, False), (1, 8, False)],
)
def test_channel_id_is_metadata_and_ideal_mode_filters_cross_sf(ch2, sf2, collides):
    first = make_collision_packet("P1")
    second = make_collision_packet("P2", ch_id=ch2, sf=sf2)
    _, survivors, failed = resolve(first, second)
    assert (len(failed) == 2) is collides
    assert len(survivors) == (0 if collides else 2)


def test_sf_orthogonality_can_be_disabled():
    first = make_collision_packet("P1", sf=7)
    second = make_collision_packet("P2", sf=12)
    _, survivors, failed = resolve(first, second, enforce_sf_match=False)
    assert survivors == []
    assert len(failed) == 2


@pytest.mark.parametrize(
    ("bandwidth", "expected_survivor_ids"),
    [
        (62_500, set()),
        (250_000, set()),
        (500_000, {"P1"}),
    ],
)
def test_bandwidth_scales_partial_overlap_power(bandwidth, expected_survivor_ids):
    first = make_collision_packet("P1", bw_hz=125_000)
    second = make_collision_packet("P2", bw_hz=bandwidth)
    _, survivors, failed = resolve(first, second)
    assert {packet.packet_id for packet in survivors} == expected_survivor_ids


def test_missing_or_non_numeric_channel_is_allowed_as_trace_metadata():
    packets = [
        make_collision_packet("P1", ch_id=None),
        make_collision_packet("P2", ch_id="logical-channel-A"),
    ]
    _, survivors, failed = resolve(*packets)
    assert survivors == []
    assert len(failed) == 2


def test_missing_bandwidth_is_rejected():
    with pytest.raises(ValueError, match="invalid bw_hz"):
        CollisionEngine().resolve([make_collision_packet(bw_hz=None)])


@pytest.mark.parametrize("invalid_sf", [None, "bad"])
def test_invalid_sf_is_rejected_during_grouping(invalid_sf):
    with pytest.raises((TypeError, ValueError)):
        CollisionEngine().resolve(
            [make_collision_packet("P1"), make_collision_packet("P2", sf=invalid_sf)]
        )


@pytest.mark.parametrize(
    ("delta_db", "expected_survivors"),
    [(5.999999999, 0), (6.0, 1), (6.000000001, 1), (0.0, 0)],
)
def test_capture_threshold_boundary(delta_db, expected_survivors):
    strong = make_collision_packet("strong", rssi=-100)
    weak = make_collision_packet("weak", rssi=-100 - delta_db)
    _, survivors, failed = resolve(strong, weak)
    assert len(survivors) == expected_survivors
    if survivors:
        assert survivors[0] is strong


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("strong_times", [(1.0, 2.0), (1.25, 2.25)])
def test_capture_winner_is_independent_of_input_and_arrival_order(reverse, strong_times):
    strong = make_collision_packet(
        "strong", rssi=-90, rx_start=strong_times[0], rx_end=strong_times[1]
    )
    weak = make_collision_packet("weak", rssi=-110, rx_start=1.0, rx_end=2.0)
    packets = [weak, strong] if reverse else [strong, weak]
    _, survivors, failed = resolve(*packets)
    assert survivors == [strong]
    assert failed == [weak]


def test_capture_three_packets_selects_strongest_against_second_strongest():
    strong = make_collision_packet("strong", rssi=-80)
    middle = make_collision_packet("middle", rssi=-90)
    weak = make_collision_packet("weak", rssi=-90)
    _, survivors, failed = resolve(weak, middle, strong)
    assert survivors == [strong]
    assert {packet.packet_id for packet in failed} == {"middle", "weak"}


def test_capture_tie_breaking_follows_stable_input_order_but_all_fail():
    first = make_collision_packet("first", rssi=-100)
    second = make_collision_packet("second", rssi=-100)
    _, survivors, failed = resolve(first, second)
    assert survivors == []
    assert failed == [first, second]


def test_sinr_matches_independent_linear_power_formula():
    target = make_collision_packet("target", rssi=-90)
    interferer = make_collision_packet("interferer", rssi=-100)
    engine = CollisionEngine(mode="sinr", noise_floor_dBm=-120, destructive=False)
    survivors, failed = engine.resolve([target, interferer])
    assert_partition([target, interferer], survivors, failed)
    expected = reference_sinr_db(-90, [-100], -120)
    assert target.sinr_dB == pytest.approx(expected, abs=1e-9)


def test_sinr_counts_two_overlapping_interferers_in_linear_domain():
    target = make_collision_packet("target", rssi=-80)
    interferers = [make_collision_packet(f"I{i}", rssi=-100) for i in range(2)]
    engine = CollisionEngine(mode="sinr", noise_floor_dBm=-120, destructive=False)
    survivors, failed = engine.resolve([target, *interferers])
    assert_partition([target, *interferers], survivors, failed)
    assert target.sinr_dB == pytest.approx(
        reference_sinr_db(-80, [-100, -100], -120), abs=1e-9
    )


def test_sinr_exact_required_threshold_passes():
    # No effective interference; choose signal so S/N is exactly SF7 threshold.
    target = make_collision_packet("target", rssi=-127.5)
    interferer = make_collision_packet("tiny", rssi=-999)
    engine = CollisionEngine(mode="sinr", noise_floor_dBm=-120, destructive=False)
    survivors, failed = engine.resolve([target, interferer])
    assert target.sinr_dB == pytest.approx(-7.5, abs=1e-9)
    assert target in survivors


@pytest.mark.parametrize("destructive,expected_count", [(True, 1), (False, 2)])
def test_sinr_destructive_option_controls_multiple_passers(destructive, expected_count):
    first = make_collision_packet("P1", rssi=-80)
    second = make_collision_packet("P2", rssi=-80, rx_start=1.9, rx_end=2.9)
    engine = CollisionEngine(mode="sinr", noise_floor_dBm=-120, destructive=destructive)
    survivors, failed = engine.resolve([first, second])
    assert_partition([first, second], survivors, failed)
    assert len(survivors) == expected_count


def test_timing_is_derived_from_arrival_end_when_rx_times_missing():
    packet = make_collision_packet("P1", rx_start=None, rx_end=None, t_arrive=3, t_end=4)
    CollisionEngine().resolve([packet])
    assert packet.rx_start == 3
    assert packet.rx_end == 4


def test_timing_falls_back_to_tx_start_plus_toa():
    packet = make_collision_packet(
        "P1", rx_start=None, rx_end=None, t_arrive=None, t_end=None, tx_start=3, toa=0.5
    )
    CollisionEngine().resolve([packet])
    assert packet.rx_start == 3
    assert packet.rx_end == pytest.approx(3.5, abs=1e-12)


def test_repeated_equivalent_inputs_produce_same_semantic_result():
    def run_once():
        packets = [
            make_collision_packet("strong", rssi=-90),
            make_collision_packet("weak", rssi=-110),
        ]
        survivors, failed = CollisionEngine().resolve(packets)
        return [p.packet_id for p in survivors], [p.packet_id for p in failed]

    assert run_once() == run_once() == (["strong"], ["weak"])
