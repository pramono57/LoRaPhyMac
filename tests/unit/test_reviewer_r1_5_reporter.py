from __future__ import annotations

import math

import pandas as pd
import pytest

from loraphymac.experiments.apc_compare_model_terrain_sweep.contract import (
    APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS,
    MAIN_W4_MODELS,
    REVIEWER_R1_5_TERRAINS,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.reviewer_r1_5_reporter import (
    DIFFERENCE_DIRECTION,
    build_model_routing_interactions,
    build_pair_validation,
    build_parent_change_summary,
    build_paired_routing_per_run,
    physical_configuration_signature,
    summarize_model_routing_interactions,
    summarize_pairs,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.routing_sensitivity_reporting import (
    build_route_change_table,
)


pytestmark = pytest.mark.unit


def _runs() -> pd.DataFrame:
    rows = []
    for model_index, model_id in enumerate(MAIN_W4_MODELS):
        for terrain in REVIEWER_R1_5_TERRAINS:
            for seed_index, seed in enumerate(APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS):
                for strategy_index, strategy in enumerate(("fixed_reference", "terrain_specific")):
                    delta = strategy_index * (seed_index + 1)
                    rows.append(
                        {
                            "model_id": model_id,
                            "model_label": model_id,
                            "terrain_delta_h": terrain,
                            "seed": seed,
                            "routing_strategy": strategy,
                            "run_id": f"{model_id}-{terrain}-{seed}-{strategy}",
                            "offered_packets": 100,
                            "delivered_packets": 80 + delta,
                            "terminal_drop_packets": 20 - delta,
                            "censored_packets": 0,
                            "pdr_percent": 80.0 + delta,
                            "total_network_energy_J": 1000.0 + 10.0 * delta,
                            "energy_per_delivered_packet_J": (1000.0 + 10.0 * delta) / (80 + delta),
                            "avg_e2e_delay_ms": 250.0 + delta,
                            "throughput_bps": 10.0 * (80 + delta),
                            "not_feasible_packets": delta,
                            "collision_packets": 10 - delta,
                            "half_duplex_packets": 10,
                            "other_terminal_packets": 0,
                            "topology_hash": f"topology-{seed}",
                            "routing_hash": f"routing-{terrain}-{strategy}",
                            "offered_traffic_hash": f"traffic-{seed}",
                            "physical_configuration_signature": (
                                f"physical-{model_id}-{terrain}-{seed}"
                            ),
                            "route_construction_delta_h": (
                                90 if strategy == "fixed_reference" else terrain
                            ),
                            "physical_terrain_delta_h": terrain,
                            "model_index": model_index,
                        }
                    )
    return pd.DataFrame(rows)


def _route_changes() -> pd.DataFrame:
    rows = []
    for model_id in MAIN_W4_MODELS:
        for terrain in REVIEWER_R1_5_TERRAINS:
            for seed_index, seed in enumerate(APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS):
                changed = int(seed_index % 2 == 0)
                rows.append(
                    {
                        "model_id": model_id,
                        "terrain_delta_h": terrain,
                        "seed": seed,
                        "route_change_population": "end_device_nodes",
                        "route_change_denominator": 100,
                        "parent_change_count": changed,
                        "parent_change_ratio": changed / 100.0,
                        "full_route_change_count": changed,
                        "full_route_change_ratio": changed / 100.0,
                        "fixed_mean_hop_count": 1.5,
                        "terrain_specific_mean_hop_count": 1.5,
                    }
                )
    return pd.DataFrame(rows)


def test_physical_signature_ignores_only_routing_treatment_and_profile():
    fixed = {
        "experiment_profile": "main_w4",
        "routing_strategy": "fixed_reference",
        "routing_reference_delta_h": 90,
        "routing_snapshot_manifest": "snapshot.json",
        "routing_snapshot_manifest_sha256": "abc",
        "terrain_delta_h": 160,
        "seed": 30,
        "node_num": 100,
        "node_app_payload_size_bytes": 50,
    }
    terrain = {
        "experiment_profile": "reviewer_r1_5",
        "routing_strategy": "terrain_specific",
        "routing_reference_delta_h": None,
        "routing_snapshot_manifest": None,
        "routing_snapshot_manifest_sha256": None,
        "terrain_delta_h": 160,
        "seed": 30,
        "node_num": 100,
        "node_app_payload_size_bytes": 50,
    }
    assert physical_configuration_signature(fixed) == (physical_configuration_signature(terrain))
    terrain["node_app_payload_size_bytes"] = 49
    assert physical_configuration_signature(fixed) != (physical_configuration_signature(terrain))


def test_exact_75_pairs_and_differences_are_terrain_minus_fixed():
    runs = _runs()
    validation = build_pair_validation(runs)
    paired = build_paired_routing_per_run(runs, _route_changes())

    assert len(validation) == 75
    assert validation["validation_pass"].all()
    assert len(paired) == 75
    assert set(paired["difference_direction"]) == {DIFFERENCE_DIRECTION}
    seed_56 = paired.loc[paired["seed"] == 56]
    assert (seed_56["delivered_packets_difference"] == 5).all()
    assert (seed_56["pdr_difference_pp"] == 5.0).all()
    assert (seed_56["total_energy_difference_J"] == 50.0).all()


def test_offered_traffic_mismatch_is_not_relaxed():
    runs = _runs()
    target = (
        (runs["model_id"] == "aa")
        & (runs["terrain_delta_h"] == 20)
        & (runs["seed"] == 30)
        & (runs["routing_strategy"] == "terrain_specific")
    )
    runs.loc[target, "offered_traffic_hash"] = "different"
    validation = build_pair_validation(runs)
    failed = validation.loc[
        (validation["model_id"] == "aa")
        & (validation["terrain_delta_h"] == 20)
        & (validation["seed"] == 30)
    ].iloc[0]

    assert not bool(failed["validation_pass"])
    assert "offered_traffic_hash_equal" in failed["failed_checks"]


def test_paired_student_t_statistics_use_five_seeds_and_sample_sd():
    paired = build_paired_routing_per_run(_runs(), _route_changes())
    statistics = summarize_pairs(paired)
    row = statistics.loc[
        (statistics["model_id"] == "aa")
        & (statistics["terrain_delta_h"] == 20)
        & (statistics["metric"] == "pdr_difference_pp")
    ].iloc[0]
    expected = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    t_critical_df4 = 2.7764451051977987
    expected_sem = expected.std(ddof=1) / math.sqrt(5)

    assert row["n"] == 5
    assert row["degrees_of_freedom"] == 4
    assert row["mean"] == pytest.approx(expected.mean())
    assert row["std"] == pytest.approx(expected.std(ddof=1))
    assert row["ci_lower"] == pytest.approx(expected.mean() - t_critical_df4 * expected_sem)
    assert row["ci_upper"] == pytest.approx(expected.mean() + t_critical_df4 * expected_sem)


def test_parent_summary_uses_five_seeds_not_repeated_model_rows():
    route_changes = _route_changes()
    summary = build_parent_change_summary(route_changes)
    assert set(summary["n"]) == {5}
    row = summary.loc[
        (summary["terrain_delta_h"] == 20) & (summary["metric"] == "parent_change_ratio")
    ].iloc[0]
    assert row["mean"] == pytest.approx(0.006)

    broken = route_changes.copy()
    broken.loc[
        (broken["model_id"] == "aa") & (broken["terrain_delta_h"] == 20) & (broken["seed"] == 30),
        "parent_change_count",
    ] = 2
    with pytest.raises(ValueError, match="differ across power models"):
        build_parent_change_summary(broken)


def test_model_routing_interactions_are_paired_against_ff():
    paired = build_paired_routing_per_run(_runs(), _route_changes())
    interactions = build_model_routing_interactions(paired)
    statistics = summarize_model_routing_interactions(interactions)

    assert len(interactions) == 60
    assert set(interactions["reference_model_id"]) == {"ff"}
    assert len(statistics) == 60
    assert set(statistics["n"]) == {5}
    assert (statistics["mean"] == 0.0).all()


def test_unchanged_parent_map_has_zero_parent_and_full_route_change():
    rows = []
    parents = {"GW": None, "R1": "GW", "N1": "R1", "N2": "GW"}
    for strategy in ("fixed_reference", "terrain_specific"):
        for entity_id, parent_id in parents.items():
            rows.append(
                {
                    "model_id": "aa",
                    "seed": 30,
                    "terrain_delta_h": 20,
                    "routing_strategy": strategy,
                    "entity_id": entity_id,
                    "entity_type": (
                        "gateway"
                        if entity_id == "GW"
                        else "relay"
                        if entity_id.startswith("R")
                        else "node"
                    ),
                    "parent_id": parent_id,
                    "x_m": {"GW": 0, "R1": 1, "N1": 2, "N2": 3}[entity_id],
                    "y_m": 0,
                    "height_m": 1,
                }
            )
    result = build_route_change_table(pd.DataFrame(rows)).iloc[0]
    assert result["parent_change_count"] == 0
    assert result["full_route_change_count"] == 0
