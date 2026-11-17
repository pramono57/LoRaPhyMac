import pandas as pd

from loraphymac.experiments.apc_load_sweep.profile_acceptance_pilot import (
    compare_scientific_table,
)


KEYS = ("entity_id", "channel")
TOLERANCES = {"total_airtime_s": 1e-12}


def _airtime_frame(airtime=(0.1, 0.2)):
    return pd.DataFrame({
        "entity_id": ["N1", "R1"],
        "channel": [1, 2],
        "tx_packet_count": [3, 4],
        "total_airtime_s": list(airtime),
        "airtime_consistent": [True, True],
    })


def _compare(left, right):
    return compare_scientific_table(
        left,
        right,
        key_columns=KEYS,
        absolute_tolerances=TOLERANCES,
    )


def test_scientific_equivalence_identical_tables_pass():
    frame = _airtime_frame()
    assert _compare(frame, frame.copy())["equivalent"]


def test_scientific_equivalence_airtime_roundoff_within_tolerance_passes():
    left = _airtime_frame()
    right = _airtime_frame((0.1 + 5e-13, 0.2))
    result = _compare(left, right)
    assert result["equivalent"]
    assert result["max_absolute_difference"]["total_airtime_s"] < 1e-12


def test_scientific_equivalence_airtime_difference_above_tolerance_fails():
    result = _compare(_airtime_frame(), _airtime_frame((0.1 + 2e-12, 0.2)))
    assert not result["equivalent"]
    assert "exceeds atol" in result["failures"][0]


def test_scientific_equivalence_missing_group_fails():
    result = _compare(_airtime_frame(), _airtime_frame().iloc[:1])
    assert not result["equivalent"]
    assert any("row-key mismatch" in failure for failure in result["failures"])


def test_scientific_equivalence_counter_change_fails_exactly():
    right = _airtime_frame()
    right.loc[0, "tx_packet_count"] += 1
    result = _compare(_airtime_frame(), right)
    assert not result["equivalent"]
    assert "exact field mismatch: tx_packet_count" in result["failures"]


def test_scientific_equivalence_packet_fate_change_fails_exactly():
    left = pd.DataFrame({"packet_uid": ["p1"], "terminal_state": ["DELIVERED"]})
    right = pd.DataFrame({"packet_uid": ["p1"], "terminal_state": ["DROPPED"]})
    result = compare_scientific_table(left, right, key_columns=("packet_uid",))
    assert not result["equivalent"]
    assert "exact field mismatch: terminal_state" in result["failures"]


def test_scientific_equivalence_collision_metric_change_fails_exactly():
    left = pd.DataFrame({"receiver_id": ["R1"], "collision_count": [7],
                         "avg_capture_delta_db": [3.0]})
    right = left.copy()
    right.loc[0, "avg_capture_delta_db"] += 1e-14
    result = compare_scientific_table(left, right, key_columns=("receiver_id",))
    assert not result["equivalent"]
    assert "exact field mismatch: avg_capture_delta_db" in result["failures"]


def test_scientific_equivalence_schema_change_fails():
    left = _airtime_frame()
    right = left.rename(columns={"total_airtime_s": "airtime_s"})
    result = _compare(left, right)
    assert not result["equivalent"]
    assert result["failures"] == ["schema/column order mismatch"]


def test_scientific_equivalence_integral_float_key_matches_numeric_value():
    summary = pd.DataFrame({"ptx_dbm": [17.0], "tx_packet_count": [5]})
    detailed = pd.DataFrame({"ptx_dbm": [17], "tx_packet_count": [5]})
    result = compare_scientific_table(
        summary, detailed, key_columns=("ptx_dbm",)
    )
    assert result["equivalent"]
    assert result["dtype_mismatches"]["ptx_dbm"] == {
        "summary": "float64", "detailed": "int64"
    }
