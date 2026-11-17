from __future__ import annotations

import pandas as pd
import pytest

from loraphymac.experiments.common.multiseed_pipeline import atomic_write_csv
from loraphymac.metrics.instrumentation_metrics import (
    CHANNEL_OCCUPANCY_COLUMNS,
    build_channel_occupancy_df,
)


def _tx(time, toa, *, node="N1", channel=1, observed=None, uid=None):
    row = {
        "event": "TX",
        "node": node,
        "time": time,
        "ch": channel,
        "toa_s": toa,
        "observed_tx_airtime_s": toa if observed is None else observed,
    }
    if uid is not None:
        row["packet_uid"] = uid
    return row


def _occupancy(rows, horizon=10.0):
    return build_channel_occupancy_df(
        pd.DataFrame(rows), {"run_id": "unit", "sim_time": horizon}
    )


def test_busy_union_overlap_nested_adjacent_and_identical_intervals():
    overlap = _occupancy([
        _tx(0, 5, uid="a"),
        _tx(1, 3, node="N2", uid="b"),
        _tx(2, 1, node="N3", uid="c"),
    ]).iloc[0]
    assert overlap["sum_observed_tx_airtime_s"] == 9
    assert overlap["channel_busy_union_s"] == 5

    adjacent = _occupancy([
        _tx(0, 2, uid="a"),
        _tx(2, 3, node="N2", uid="b"),
    ]).iloc[0]
    assert adjacent["channel_busy_union_s"] == 5

    identical = _occupancy([
        _tx(2, 3, uid="a"),
        _tx(2, 3, node="N2", uid="b"),
    ]).iloc[0]
    assert identical["channel_busy_union_s"] == 3


def test_busy_union_is_separate_per_channel_and_has_valid_bounds():
    result = _occupancy([
        _tx(0, 4, channel=1, uid="a"),
        _tx(1, 4, channel=2, node="N2", uid="b"),
    ]).set_index("channel")
    assert result.loc[1, "channel_busy_union_s"] == 4
    assert result.loc[2, "channel_busy_union_s"] == 4
    assert (result["channel_busy_union_s"] <= result["sum_observed_tx_airtime_s"]).all()
    assert result["channel_busy_percent"].between(0, 100).all()


def test_empty_or_non_tx_input_returns_schema_without_inventing_channels():
    empty = build_channel_occupancy_df(pd.DataFrame(), {"sim_time": 10})
    no_tx = build_channel_occupancy_df(
        pd.DataFrame([{"event": "RX", "node": "R1"}]), {"sim_time": 10}
    )
    assert list(empty.columns) == CHANNEL_OCCUPANCY_COLUMNS
    assert list(no_tx.columns) == CHANNEL_OCCUPANCY_COLUMNS
    assert empty.empty and no_tx.empty


def test_horizon_clips_observed_tx_airtime_and_late_transmission():
    result = _occupancy([
        _tx(9, 5, observed=5, uid="cross"),
        _tx(11, 2, observed=2, channel=2, uid="late"),
    ]).set_index("channel")
    assert result.loc[1, "sum_offered_tx_airtime_s"] == 5
    assert result.loc[1, "sum_observed_tx_airtime_s"] == 1
    assert result.loc[1, "channel_busy_union_s"] == 1
    assert result.loc[2, "sum_observed_tx_airtime_s"] == 0
    assert result.loc[2, "channel_busy_percent"] == 0


def test_zero_horizon_is_schema_complete_and_invalid_horizons_reject():
    zero = _occupancy([_tx(0, 1, uid="a")], horizon=0)
    assert zero.empty
    assert list(zero.columns) == CHANNEL_OCCUPANCY_COLUMNS
    for horizon in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="sim_time"):
            _occupancy([], horizon=horizon)


def test_unsorted_timestamps_and_duplicate_physical_tx_are_rejected():
    with pytest.raises(ValueError, match="ordered"):
        _occupancy([_tx(2, 1, uid="later"), _tx(1, 1, uid="earlier")])
    duplicate = _tx(1, 1, uid="same")
    with pytest.raises(ValueError, match="duplicate physical TX"):
        _occupancy([duplicate, duplicate.copy()])


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_tx(-1, 1, uid="negative-time")], "timestamps"),
        ([_tx(0, -1, uid="negative-toa")], "non-negative"),
        ([_tx(0, 1, observed=2, uid="too-much")], "exceed nominal ToA"),
        ([_tx(0, 1, channel=-1, uid="bad-channel")], "channel IDs"),
        ([_tx(0, 1, channel=1.5, uid="fractional-channel")], "channel IDs"),
        ([_tx(0, float("nan"), uid="nan")], "null"),
        ([_tx(0, float("inf"), uid="infinite")], "finite numeric"),
        ([{"event": "TX", "node": "N1", "time": 0, "ch": 1}], "missing required columns"),
    ],
)
def test_malformed_tx_rows_fail_explicitly(rows, message):
    with pytest.raises(ValueError, match=message):
        _occupancy(rows)


def test_occupancy_table_csv_persistence_round_trip(tmp_path):
    frame = _occupancy([_tx(0, 2, uid="a"), _tx(1, 2, node="N2", uid="b")])
    target = tmp_path / "tables" / "channel_occupancy.csv"
    atomic_write_csv(target, frame)
    restored = pd.read_csv(target)
    pd.testing.assert_frame_equal(
        restored,
        frame,
        check_dtype=False,
        check_exact=True,
    )
