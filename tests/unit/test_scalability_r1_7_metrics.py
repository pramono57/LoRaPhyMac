from types import SimpleNamespace

import pandas as pd
import pytest

from loraphymac.metrics.instrumentation_metrics import (
    build_channel_occupancy_df,
    build_node_tx_duty_cycle_df,
)


def _events():
    return pd.DataFrame(
        [
            {"event": "TX", "node": "N1", "ch": 1, "time": 0.0, "toa_s": 3.0,
             "observed_tx_airtime_s": 3.0},
            {"event": "TX", "node": "N2", "ch": 1, "time": 1.0, "toa_s": 2.0,
             "observed_tx_airtime_s": 2.0},
            {"event": "TX", "node": "R1", "ch": 1, "time": 5.0, "toa_s": 1.0,
             "observed_tx_airtime_s": 1.0},
            {"event": "TX", "node": "N1", "ch": 2, "time": 9.0, "toa_s": 4.0,
             "observed_tx_airtime_s": 1.0},
        ]
    )


def test_node_tx_duty_cycle_includes_silent_nodes_and_excludes_relays():
    rows = build_node_tx_duty_cycle_df(
        _events(),
        [SimpleNamespace(id="N1"), SimpleNamespace(id="N2"),
         SimpleNamespace(id="N3"), SimpleNamespace(id="R1"),
         SimpleNamespace(id="GW")],
        {"run_id": "r1", "sim_time": 10.0},
    ).set_index("entity_id")

    assert set(rows.index) == {"N1", "N2", "N3"}
    assert rows.loc["N1", "observed_tx_airtime_s"] == pytest.approx(4.0)
    assert rows.loc["N1", "tx_duty_cycle_percent"] == pytest.approx(40.0)
    assert rows.loc["N3", "tx_duty_cycle_percent"] == pytest.approx(0.0)


def test_channel_occupancy_separates_sum_airtime_and_interval_union():
    rows = build_channel_occupancy_df(
        _events(), {"run_id": "r1", "sim_time": 10.0}
    ).set_index("channel")

    assert rows.loc[1, "sum_offered_tx_airtime_s"] == pytest.approx(6.0)
    assert rows.loc[1, "channel_busy_union_s"] == pytest.approx(4.0)
    assert rows.loc[1, "channel_busy_percent"] == pytest.approx(40.0)
    assert rows.loc[2, "sum_observed_tx_airtime_s"] == pytest.approx(1.0)
    assert rows.loc[2, "channel_busy_union_s"] == pytest.approx(1.0)


def test_channel_busy_time_does_not_double_count_overlapping_transmissions():
    events = _events().iloc[:2]
    row = build_channel_occupancy_df(
        events, {"sim_time": 10.0}
    ).iloc[0]
    assert row["sum_observed_tx_airtime_s"] == pytest.approx(5.0)
    assert row["channel_busy_union_s"] == pytest.approx(3.0)


def test_metrics_return_empty_when_horizon_is_not_positive():
    assert build_node_tx_duty_cycle_df(_events(), [], {"sim_time": 0.0}).empty
    assert build_channel_occupancy_df(_events(), {"sim_time": 0.0}).empty
