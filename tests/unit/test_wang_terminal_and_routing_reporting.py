from __future__ import annotations

import pandas as pd
import pytest

from loraphymac.experiments.apc_compare_model_terrain_sweep.postprocessing import (
    _assert_reporting_strategy,
    _wang_metrics,
    build_comparison_artifacts,
    reconcile_wang_command_lifecycle,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.routing_sensitivity_reporting import (
    add_reference_identity_anchor,
    build_r1_5_difference_in_differences,
    build_r1_5_paired_runs,
    build_route_change_table,
    summarize_r1_5_difference_in_differences,
    summarize_r1_5_pairs,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.run import (
    REVIEWER_R1_5_CONFIG,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.paired_routing_pilot import (
    _apc_offline_belief_diagnostics,
    _nonrouting_scientific_signature,
    _selected_link_quality,
)


pytestmark = pytest.mark.unit


def _historical_lifecycle(
    *, confirmed: int, applied_unconfirmed: int, not_applied: int, late_confirmed: int
) -> pd.DataFrame:
    rows = []
    command_number = 0

    def add(events: list[str]) -> None:
        nonlocal command_number
        command_id = f"C{command_number:03d}"
        command_number += 1
        rows.extend({"command_id": command_id, "event": event} for event in events)

    for index in range(confirmed):
        events = ["command_generated", "command_applied"]
        if index < late_confirmed:
            events.append("command_unconfirmed")
        events.append("command_confirmed_by_uplink")
        add(events)
    for _ in range(applied_unconfirmed):
        add(["command_generated", "command_applied", "command_unconfirmed"])
    for _ in range(not_applied):
        add(["command_generated", "command_unconfirmed", "command_not_applied"])
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("confirmed", "applied_unconfirmed", "not_applied", "late_confirmed"),
    [(37, 6, 19, 0), (55, 10, 37, 1)],
)
def test_wang_historical_terminal_accounting_is_disjoint(
    confirmed, applied_unconfirmed, not_applied, late_confirmed
):
    commands = _historical_lifecycle(
        confirmed=confirmed,
        applied_unconfirmed=applied_unconfirmed,
        not_applied=not_applied,
        late_confirmed=late_confirmed,
    )
    terminal = reconcile_wang_command_lifecycle(commands)
    counts = terminal["terminal_status"].value_counts()

    assert counts.to_dict() == {
        "APPLIED_CONFIRMED": confirmed,
        "NOT_APPLIED": not_applied,
        "APPLIED_UNCONFIRMED": applied_unconfirmed,
    }
    assert len(terminal) == confirmed + applied_unconfirmed + not_applied
    assert terminal["command_id"].is_unique


def test_wang_expiry_event_is_not_a_terminal_status():
    commands = _historical_lifecycle(
        confirmed=1, applied_unconfirmed=1, not_applied=1, late_confirmed=1
    )
    commands = pd.concat(
        [
            commands,
            pd.DataFrame(
                [
                    {"command_id": "C000", "event": "command_sent"},
                    {"command_id": "C000", "event": "command_sent"},
                    {"command_id": "C000", "event": "command_received"},
                ]
            ),
        ],
        ignore_index=True,
    )
    metrics = _wang_metrics(
        "wang_atp_sf7_a2only",
        pd.DataFrame(columns=["event", "report_id"]),
        pd.DataFrame(
            [
                {
                    "decision_id": "D1",
                    "window_index": 0,
                    "status": "decision_generated",
                    "reason": "complete",
                    "coverage_fraction": 1.0,
                }
            ]
        ),
        commands,
        pd.DataFrame(),
    )

    assert metrics["wang_commands_unconfirmed"] == 3
    assert metrics["wang_commands_unconfirmed_at_expiry_unique"] == 3
    assert metrics["wang_command_generated_events"] == 3
    assert metrics["wang_commands_generated_unique"] == 3
    assert metrics["wang_commands_applied_confirmed_unique"] == 1
    assert metrics["wang_commands_applied_unconfirmed_unique"] == 1
    assert metrics["wang_commands_not_applied_unique"] == 1
    assert metrics["wang_command_terminal_conservation_delta"] == 0
    assert metrics["wang_command_physical_hop_sent_events"] == 2
    assert metrics["wang_command_physical_hop_received_events"] == 1


def test_wang_terminal_reconciliation_rejects_contradictory_or_incomplete_ids():
    contradictory = pd.DataFrame(
        [
            {"command_id": "C1", "event": "command_generated"},
            {"command_id": "C1", "event": "command_applied"},
            {"command_id": "C1", "event": "command_not_applied"},
        ]
    )
    incomplete = pd.DataFrame(
        [{"command_id": "C2", "event": "command_generated"}]
    )
    with pytest.raises(ValueError, match="contradictory"):
        reconcile_wang_command_lifecycle(contradictory)
    with pytest.raises(ValueError, match="no observable final status"):
        reconcile_wang_command_lifecycle(incomplete)


def test_wang_unexpired_command_is_right_censored_at_simulation_horizon():
    commands = pd.DataFrame(
        [
            {
                "command_id": "C1",
                "event": "command_generated",
                "expiry_s": 3610.0,
            },
            {
                "command_id": "C1",
                "event": "command_sent",
                "expiry_s": 3610.0,
            },
            {
                "command_id": "C1",
                "event": "command_dropped",
                "expiry_s": 3610.0,
            },
        ]
    )

    terminal = reconcile_wang_command_lifecycle(
        commands, observation_end_s=3600.0
    )
    assert terminal.loc[0, "terminal_status"] == "CENSORED_AT_HORIZON"
    assert bool(terminal.loc[0, "censored_at_horizon"])

    with pytest.raises(ValueError, match="no observable final status"):
        reconcile_wang_command_lifecycle(commands, observation_end_s=3610.0)


def test_paired_physical_signature_and_selected_link_quality():
    fixed = {
        "scientific_config": {
            "terrain_delta_h": 160,
            "node_num": 100,
            "routing_strategy": "fixed_reference",
            "routing_reference_delta_h": 90,
            "routing_snapshot_manifest": "snapshot.json",
            "routing_snapshot_manifest_sha256": "abc",
        }
    }
    terrain = {
        "scientific_config": {
            "terrain_delta_h": 160,
            "node_num": 100,
            "routing_strategy": "terrain_specific",
        }
    }
    assert _nonrouting_scientific_signature(fixed) == (
        _nonrouting_scientific_signature(terrain)
    )

    quality = _selected_link_quality(
        pd.DataFrame(
            [
                {
                    "model_id": "ff",
                    "routing_strategy": "fixed_reference",
                    "terrain_delta_h": 160,
                    "seed": 30,
                    "parent_id": "GW",
                    "pathloss_db": 140.0,
                    "margin_db": 5.0,
                    "valid": True,
                },
                {
                    "model_id": "ff",
                    "routing_strategy": "fixed_reference",
                    "terrain_delta_h": 160,
                    "seed": 30,
                    "parent_id": "R1",
                    "pathloss_db": 155.0,
                    "margin_db": -1.0,
                    "valid": False,
                },
            ]
        )
    ).iloc[0]
    assert quality["selected_link_count"] == 2
    assert quality["physically_feasible_link_count"] == 1
    assert quality["physically_infeasible_link_count"] == 1
    assert quality["physically_feasible_link_ratio"] == 0.5


def test_apc_belief_mismatch_is_reconstructed_offline_only():
    feedback = pd.DataFrame(
        [
            {
                "event_sequence": 1,
                "event": "generated",
                "target_id": "N1",
                "decision_tx_power_index": 4,
                "applied_tx_power_index": None,
            },
            {
                "event_sequence": 2,
                "event": "applied",
                "target_id": "N1",
                "decision_tx_power_index": 4,
                "applied_tx_power_index": 4,
            },
            {
                "event_sequence": 3,
                "event": "generated",
                "target_id": "N2",
                "decision_tx_power_index": 6,
                "applied_tx_power_index": None,
            },
        ]
    )
    result = _apc_offline_belief_diagnostics(feedback, ["N1", "N2", "R1"])
    assert result["apc_offline_tpi_population"] == 3
    assert result["apc_offline_tpi_mismatch_count"] == 1
    assert result["apc_offline_tpi_mismatch_fraction"] == pytest.approx(1 / 3)
    assert result["apc_offline_tpi_mean_absolute_difference"] == 2.0


def test_wang_duplicate_raw_events_still_yield_one_final_status():
    commands = pd.DataFrame(
        [
            {"command_id": "C1", "event": "command_generated"},
            {"command_id": "C1", "event": "command_generated"},
            {"command_id": "C1", "event": "command_applied"},
            {"command_id": "C1", "event": "command_applied"},
            {"command_id": "C1", "event": "command_confirmed_by_uplink"},
            {"command_id": "C1", "event": "command_confirmed_by_uplink"},
        ]
    )

    terminal = reconcile_wang_command_lifecycle(commands)
    assert terminal.to_dict(orient="records")[0]["terminal_status"] == (
        "APPLIED_CONFIRMED"
    )
    assert len(terminal) == 1


def _paired_runs() -> pd.DataFrame:
    rows = []
    for model_index, model_id in enumerate(("ff", "aa")):
        for seed_index, seed in enumerate((30, 36, 42, 48, 56)):
            for strategy_index, strategy in enumerate(
                ("fixed_reference", "terrain_specific")
            ):
                delivered = 90 + model_index + seed_index + strategy_index
                rows.append(
                    {
                        "model_id": model_id,
                        "seed": seed,
                        "terrain_delta_h": 20,
                        "routing_strategy": strategy,
                        "offered_traffic_hash": f"traffic-{seed}",
                        "topology_hash": f"topology-{seed}",
                        "routing_hash": f"route-{strategy}-{seed}",
                        "offered_packets": 100,
                        "delivered_packets": delivered,
                        "pdr_percent": float(delivered),
                        "total_network_energy_J": 1000.0 + model_index * 10 + strategy_index,
                        "energy_per_delivered_packet_J": (1000.0 + strategy_index)
                        / delivered,
                        "avg_e2e_delay_ms": 50.0 + model_index + strategy_index,
                    }
                )
    return pd.DataFrame(rows)


def _route_changes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": model_id,
                "seed": seed,
                "terrain_delta_h": 20,
                "parent_change_ratio": 0.1,
                "full_route_change_ratio": 0.2,
            }
            for model_id in ("ff", "aa")
            for seed in (30, 36, 42, 48, 56)
        ]
    )


def test_r1_5_pairing_and_student_t_contract():
    paired = build_r1_5_paired_runs(_paired_runs(), _route_changes())
    statistics = summarize_r1_5_pairs(paired, expected_terrains=(20,))
    did = build_r1_5_difference_in_differences(paired)
    did_statistics = summarize_r1_5_difference_in_differences(
        did, expected_terrains=(20,)
    )

    assert len(paired) == 10
    assert (paired["difference_direction"] == "terrain_specific_minus_fixed_reference").all()
    assert not paired["routing_hash_equal"].any()
    assert set(statistics["n"]) == {5}
    assert set(statistics["degrees_of_freedom"]) == {4}
    assert set(statistics["ci_method"]) == {"student_t"}
    assert set(did["model_id"]) == {"aa"}
    assert len(did) == 5
    assert set(did_statistics["n"]) == {5}
    assert set(did_statistics["degrees_of_freedom"]) == {4}


def test_r1_5_pairing_rejects_workload_mismatch_and_missing_pair():
    mismatch = _paired_runs()
    mismatch.loc[
        (mismatch["model_id"] == "aa")
        & (mismatch["seed"] == 30)
        & (mismatch["routing_strategy"] == "terrain_specific"),
        "offered_traffic_hash",
    ] = "different"
    with pytest.raises(ValueError, match="offered_traffic_hash mismatch"):
        build_r1_5_paired_runs(mismatch, _route_changes())

    missing = _paired_runs().iloc[:-1]
    with pytest.raises(ValueError, match="exactly one terrain_specific"):
        build_r1_5_paired_runs(missing, _route_changes())


def test_delta90_anchor_requires_zero_route_change_evidence():
    fixed = _paired_runs().loc[
        lambda frame: frame["routing_strategy"] == "fixed_reference"
    ].copy()
    fixed["terrain_delta_h"] = 90
    evidence = _route_changes().copy()
    evidence["terrain_delta_h"] = 90
    evidence[["parent_change_ratio", "full_route_change_ratio"]] = 0.0

    anchored = add_reference_identity_anchor(fixed, evidence)
    assert set(anchored["routing_strategy"]) == {
        "fixed_reference",
        "terrain_specific",
    }
    assert set(
        anchored.loc[
            anchored["routing_strategy"] == "terrain_specific", "comparison_source"
        ]
    ) == {"fixed_reference_identity_anchor"}

    invalid = evidence.copy()
    invalid.loc[invalid.index[0], "parent_change_ratio"] = 0.1
    with pytest.raises(ValueError, match="not identical"):
        add_reference_identity_anchor(fixed, invalid)


def test_route_change_ratios_use_end_device_routes_and_identical_geometry():
    rows = []
    parents = {
        "fixed_reference": {"GW": None, "R1": "GW", "N1": "R1", "N2": "GW"},
        "terrain_specific": {"GW": None, "R1": "GW", "N1": "GW", "N2": "GW"},
    }
    for strategy, route in parents.items():
        for entity_id, parent_id in route.items():
            rows.append(
                {
                    "model_id": "aa",
                    "seed": 30,
                    "terrain_delta_h": 20,
                    "routing_strategy": strategy,
                    "entity_id": entity_id,
                    "entity_type": "gateway"
                    if entity_id == "GW"
                    else "relay"
                    if entity_id.startswith("R")
                    else "node",
                    "parent_id": parent_id,
                    "x_m": {"GW": 0, "R1": 1, "N1": 2, "N2": 3}[entity_id],
                    "y_m": 0,
                    "height_m": 1,
                }
            )
    changes = build_route_change_table(pd.DataFrame(rows)).iloc[0]

    assert changes["route_change_denominator"] == 2
    assert changes["parent_change_ratio"] == 0.5
    assert changes["full_route_change_ratio"] == 0.5


@pytest.mark.extended_research
def test_r1_5_postprocessing_is_isolated_from_main_w4_figures():
    fixed_only = pd.DataFrame({"routing_strategy": ["fixed_reference"]})
    mixed = pd.DataFrame(
        {"routing_strategy": ["fixed_reference", "terrain_specific"]}
    )
    _assert_reporting_strategy(fixed_only, "fixed_reference")
    with pytest.raises(ValueError, match="routing-strategy scope mismatch"):
        _assert_reporting_strategy(mixed, "fixed_reference")
    with pytest.raises(ValueError, match="dedicated paired-routing reporter"):
        build_comparison_artifacts(REVIEWER_R1_5_CONFIG)
