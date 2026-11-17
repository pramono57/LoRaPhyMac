import pytest

from loraphymac.phy.channel_plan import AS923_2


pytestmark = pytest.mark.unit

EXPECTED_CHANNELS = [
    (0, 921.4),
    (1, 921.6),
    (2, 921.8),
    (3, 922.0),
    (4, 922.2),
    (5, 922.4),
    (6, 922.6),
    (7, 922.8),
]


def test_as923_2_channel_table_and_busy_metadata():
    plan = AS923_2()
    assert plan.num_channels() == 8
    assert [(row["id"], row["freq"]) for row in plan.channels] == EXPECTED_CHANNELS
    assert plan.channel_busy_until == {channel_id: 0.0 for channel_id, _ in EXPECTED_CHANNELS}


@pytest.mark.parametrize("channel_id,frequency", EXPECTED_CHANNELS)
def test_channel_lookup_both_directions(channel_id, frequency):
    plan = AS923_2()
    assert plan.get_freq(channel_id) == frequency
    assert plan.get_id(frequency) == channel_id


def test_first_last_node_and_backbone_channels():
    plan = AS923_2()
    assert plan.get_freq(0) == 921.4  # routing/default first node channel
    assert plan.get_freq(1) == 921.6  # default relay backbone channel
    assert plan.get_freq(7) == 922.8


@pytest.mark.parametrize("bad_id", [-1, 8, 999, None, "bad"])
def test_invalid_channel_id_is_rejected(bad_id):
    with pytest.raises((TypeError, ValueError)):
        AS923_2().get_freq(bad_id)


@pytest.mark.parametrize("bad_frequency", [0, 921.5, None, "bad"])
def test_invalid_frequency_is_rejected(bad_frequency):
    with pytest.raises((TypeError, ValueError)):
        AS923_2().get_id(bad_frequency)


def test_loraphy_metadata():
    plan = AS923_2("loraphy")
    assert plan.dr_table is None
    assert plan.rx1_offset_table is None
    assert plan.duty_cycle == 1.0
    assert plan.lbt_enabled is True
    assert plan.lbt_threshold == -85
    assert plan.get_rx1_params(0, 5, 0) is None
    assert plan.get_rx2_params() is None
    with pytest.raises(ValueError):
        plan.get_dr_params(0)


def test_lorawan_data_rate_rx_and_duty_cycle_metadata():
    plan = AS923_2("lorawan")
    assert plan.get_dr_params(0) == ("SF12", 125)
    assert plan.get_dr_params(6) == ("SF7", 250)
    assert plan.get_loraphymac_dr_params(5) == (7, 125_000)
    assert plan.get_loraphymac_dr_params(6) == (7, 250_000)
    assert plan.get_rx1_params(3, 5, 2) == (3, 3)
    assert plan.get_rx2_params() == (0, 2)
    assert plan.join_channels == [0, 1]
    assert plan.max_eirp == 16
    assert plan.duty_cycle == 0.01


@pytest.mark.parametrize(
    "uplink_dr,offset,expected_downlink_dr",
    [
        (0, 0, 0),
        (0, 5, 0),
        (0, 6, 1),
        (0, 7, 2),
        (3, 1, 2),
        (5, 5, 0),
        (5, 6, 6),
        (6, 5, 1),
    ],
)
def test_lorawan_rx1_mapping_without_downlink_dwell_time(
    uplink_dr, offset, expected_downlink_dr
):
    plan = AS923_2("lorawan", downlink_dwell_time=0)
    assert plan.get_rx1_params(3, uplink_dr, offset) == (3, expected_downlink_dr)


@pytest.mark.parametrize(
    "uplink_dr,offset,expected_downlink_dr",
    [
        (0, 0, 2),
        (3, 0, 3),
        (3, 1, 2),
        (5, 3, 2),
        (5, 6, 6),
    ],
)
def test_lorawan_rx1_mapping_with_downlink_dwell_time(
    uplink_dr, offset, expected_downlink_dr
):
    plan = AS923_2("lorawan", downlink_dwell_time=1)
    assert plan.get_rx1_params(3, uplink_dr, offset) == (3, expected_downlink_dr)


def test_lorawan_rx1_rejects_invalid_or_unsupported_parameters():
    plan = AS923_2("lorawan")
    with pytest.raises(ValueError, match="uplink data rate"):
        plan.get_rx1_params(0, 99, 0)
    with pytest.raises(ValueError, match="0..7"):
        plan.get_rx1_params(0, 5, 8)
    with pytest.raises(ValueError, match="unsupported downstream"):
        plan.get_rx1_params(0, 5, 7)
    with pytest.raises(ValueError, match="downlink_dwell_time"):
        AS923_2("lorawan", downlink_dwell_time=2)


def test_empty_channel_list_fails_lookups_without_changing_production_table():
    plan = AS923_2()
    plan.channels = []
    assert plan.num_channels() == 0
    with pytest.raises(ValueError):
        plan.get_freq(0)
    with pytest.raises(ValueError):
        plan.get_id(921.4)


def test_duplicate_channel_id_returns_first_match_on_isolated_instance():
    plan = AS923_2()
    plan.channels = [{"id": 0, "freq": 900.0}, {"id": 0, "freq": 901.0}]
    assert plan.get_freq(0) == 900.0

