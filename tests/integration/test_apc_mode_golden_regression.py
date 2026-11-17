"""Versioned golden gate for the Paper 1 APC modes.

The v1 fixture preserves pre-causal-state values as historical provenance.
The active v2 fixture represents the implementation used in MAIN_W4. Its
APC counters distinguish logical decisions, physical attempts, retries,
applications, and actual PTx changes. There is no physical APC_ANS, so each
logical command reaches its two-retry bound even when the target applied it.
Repeated absolute-target application is idempotent but still consumes physical
airtime and energy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from loraphymac import simulate


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"
HISTORICAL_V1 = json.loads(
    (FIXTURE_DIR / "apc_golden_v1_e687367.json").read_text()
)
GOLDEN_V2 = json.loads(
    (FIXTURE_DIR / "apc_golden_v2_e5d58ea.json").read_text()
)
BASE_CONFIG = GOLDEN_V2["scenario_config"]
GOLDEN_MODES = GOLDEN_V2["metrics_by_mode"]


def test_historical_golden_v1_is_preserved_as_reference_only():
    assert HISTORICAL_V1["source_commit"] == (
        "e687367b8386145e9e6429161c2c1042b6f0ffd8"
    )
    assert HISTORICAL_V1["metrics_by_mode"]["AF"]["apc_tx"] == 8
    assert HISTORICAL_V1["metrics_by_mode"]["FA"]["apc_tx"] == 2
    assert HISTORICAL_V1["metrics_by_mode"]["AA"]["apc_tx"] == 10


@pytest.mark.parametrize("mode", ["FF", "AF", "FA", "AA"])
def test_apc_mode_matches_versioned_golden_result(mode):
    expected = GOLDEN_MODES[mode]
    result = simulate(
        {
            **BASE_CONFIG,
            "node_ptx_mode": expected["node_ptx_mode"],
            "relay_ptx_mode": expected["relay_ptx_mode"],
        }
    )
    summary = result.summary

    assert summary["tx_packets"] == expected["tx_packets"]
    assert summary["rx_packets_gw"] == expected["delivered_packets"]
    assert summary["packet_fate_delivered_count"] == expected["delivered_packets"]
    assert summary["apc_total_tx"] == expected["apc_physical_tx"]
    assert summary["apc_total_rx"] == expected["apc_physical_rx"]
    assert summary["apc_total_drop"] == 0

    for key in (
        "pdr_percent",
        "avg_e2e_delay_ms",
        "total_network_energy_J",
        "avg_final_node_ptx",
        "avg_final_relay_ptx",
    ):
        assert summary[key] == pytest.approx(expected[key], abs=1e-9)

    lifecycle = result.table("apc_feedback_lifecycle_df")
    ptx_history = result.table("ptx_history_df")
    topology = result.table("topology_parent_by_run_df")
    energy = result.table("energy_breakdown_df")
    airtime = result.table("tx_airtime_by_entity_mtype_ptx_channel_df")

    assert len(topology) == BASE_CONFIG["node_num"] + BASE_CONFIG["relay_num"] + 1
    assert "total_transition_time_s" in energy.columns
    assert airtime["airtime_consistent"].all()

    actual_ptx_changes = (
        int((ptx_history["old_ptx"] != ptx_history["new_ptx"]).sum())
        if not ptx_history.empty
        else 0
    )
    assert actual_ptx_changes == expected["ptx_actual_change_count"]

    if expected["apc_logical_commands"] == 0:
        assert lifecycle.empty
        assert result.table("apc_control_history_df").empty
        assert len(ptx_history) == 0
        return

    generated = lifecycle[lifecycle["event"] == "generated"]
    retries = lifecycle[lifecycle["event"] == "retry_generated"]
    sent = lifecycle[lifecycle["event"] == "sent"]
    received = lifecycle[lifecycle["event"] == "received"]
    applied = lifecycle[lifecycle["event"] == "applied"]
    exhausted = lifecycle[lifecycle["event"] == "retry_exhausted"]
    logical_ids = set(generated["feedback_id"])

    assert generated["feedback_id"].nunique() == expected["apc_logical_commands"]
    assert len(retries) == expected["apc_retry_count"]
    assert len(sent) == expected["apc_physical_tx"]
    assert len(received) == expected["apc_physical_rx"]
    assert len(applied) == expected["apc_apply_event_count"]
    assert len(exhausted) == expected["apc_logical_commands"]
    assert sent["packet_uid"].nunique() == expected["apc_physical_tx"]

    attempts_per_command = GOLDEN_V2["retry_contract"]["max_retries"] + 1
    for event_rows in (sent, received, applied):
        assert set(event_rows["feedback_id"]) == logical_ids
        assert event_rows.groupby("feedback_id").size().eq(attempts_per_command).all()
    assert set(retries["feedback_id"]) == logical_ids
    assert retries.groupby("feedback_id").size().eq(
        GOLDEN_V2["retry_contract"]["max_retries"]
    ).all()

    measured_control_airtime = float(sent["control_airtime_s"].sum())
    assert measured_control_airtime == pytest.approx(
        expected["control_airtime_s"], abs=1e-12
    )
    if expected["node_ptx_mode"] == "adaptive":
        assert not result.table("apc_control_history_df").empty


@pytest.mark.parametrize("mode", ["FF", "AA"])
def test_corrected_energy_accounting_changes_only_energy_outputs(mode):
    expected = GOLDEN_MODES[mode]
    mode_config = {
        **BASE_CONFIG,
        "node_ptx_mode": expected["node_ptx_mode"],
        "relay_ptx_mode": expected["relay_ptx_mode"],
    }
    legacy = simulate({**mode_config, "energy_accounting_mode": "legacy"})
    corrected = simulate({**mode_config, "energy_accounting_mode": "corrected"})

    for key in (
        "tx_packets",
        "rx_packets_gw",
        "pdr_percent",
        "avg_e2e_delay_ms",
        "apc_total_tx",
        "apc_total_rx",
        "apc_total_drop",
        "avg_final_node_ptx",
        "avg_final_relay_ptx",
    ):
        assert corrected.summary[key] == legacy.summary[key]

    for table_name in (
        "routing_table_df",
        "collision_summary_df",
        "ptx_history_df",
    ):
        legacy_table = legacy.table(table_name).drop(columns=["run_id"], errors="ignore")
        corrected_table = corrected.table(table_name).drop(columns=["run_id"], errors="ignore")
        assert_frame_equal(legacy_table, corrected_table, check_dtype=True)

    legacy_energy = legacy.table("energy_breakdown_df")
    corrected_energy = corrected.table("energy_breakdown_df")
    assert set(legacy_energy["energy_accounting_mode"]) == {"legacy"}
    assert set(corrected_energy["energy_accounting_mode"]) == {"corrected"}

    delta_j = (
        float(legacy.summary["total_network_energy_J"])
        - float(corrected.summary["total_network_energy_J"])
    )
    expected_delta_j = float(
        legacy_energy["energy_rx_J"].sum()
        - corrected_energy["energy_rx_J"].sum()
    )
    assert delta_j > 0.0
    # Core-summary energy is rounded to six decimal places; the entity table
    # retains the unrounded accounting delta.
    assert delta_j == pytest.approx(round(expected_delta_j, 6), abs=1e-12)
    print(
        f"STAGE2B1_ENERGY_DELTA mode={mode} "
        f"entity_delta_J={expected_delta_j:.12f} "
        f"summary_delta_J={delta_j:.12f}"
    )
