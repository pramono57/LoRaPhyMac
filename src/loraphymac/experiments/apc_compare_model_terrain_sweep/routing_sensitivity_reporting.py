"""Strict paired-data contract for Reviewer R1.5 routing sensitivity."""

from __future__ import annotations

import math
from typing import Any, Sequence

import pandas as pd

from loraphymac.experiments.apc_compare_model_terrain_sweep.contract import (
    APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS,
    REVIEWER_R1_5_COMPARISON_TERRAINS,
)
from loraphymac.experiments.common.multiseed_pipeline import aggregate_paper_grade
from loraphymac.routing.snapshot import full_routes


FIXED_REFERENCE = "fixed_reference"
TERRAIN_SPECIFIC = "terrain_specific"
PAIR_KEYS = ("model_id", "seed", "terrain_delta_h")
ROUTING_EFFECT_METRICS = (
    "pdr_difference_pp",
    "total_energy_difference_J",
    "energy_per_delivered_difference_J_per_packet",
    "avg_e2e_delay_difference_ms",
    "parent_change_ratio",
    "full_route_change_ratio",
)
DID_METRICS = tuple(f"did_{metric}" for metric in ROUTING_EFFECT_METRICS[:4])


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _one_strategy_row(group: pd.DataFrame, strategy: str) -> pd.Series:
    selected = group.loc[group["routing_strategy"].astype(str) == strategy]
    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one {strategy} row per model/seed/terrain pair"
        )
    return selected.iloc[0]


def build_route_change_table(topology: pd.DataFrame) -> pd.DataFrame:
    """Compare end-device parent and complete route under the two strategies."""
    required = (
        *PAIR_KEYS,
        "routing_strategy",
        "entity_id",
        "entity_type",
        "parent_id",
        "x_m",
        "y_m",
        "height_m",
    )
    _require_columns(topology, required, "routing topology")
    rows: list[dict[str, Any]] = []
    for keys, group in topology.groupby(list(PAIR_KEYS), sort=True, dropna=False):
        strategy_frames: dict[str, pd.DataFrame] = {}
        for strategy in (FIXED_REFERENCE, TERRAIN_SPECIFIC):
            selected = group.loc[
                group["routing_strategy"].astype(str) == strategy
            ].copy()
            if selected.empty or selected["entity_id"].astype(str).duplicated().any():
                raise ValueError(
                    f"Missing or duplicate topology entities for {keys}/{strategy}"
                )
            selected["entity_id"] = selected["entity_id"].astype(str)
            strategy_frames[strategy] = selected.set_index("entity_id").sort_index()

        fixed = strategy_frames[FIXED_REFERENCE]
        terrain = strategy_frames[TERRAIN_SPECIFIC]
        if not fixed.index.equals(terrain.index):
            raise ValueError(f"Entity identity mismatch for routing pair {keys}")
        for column in ("entity_type", "x_m", "y_m", "height_m"):
            left = fixed[column].fillna("<NA>").astype(str)
            right = terrain[column].fillna("<NA>").astype(str)
            if not left.equals(right):
                raise ValueError(
                    f"Geometry/identity mismatch in {column} for routing pair {keys}"
                )

        def route_map(frame: pd.DataFrame) -> dict[str, str | None]:
            return {
                entity_id: (
                    None if pd.isna(parent_id) else str(parent_id)
                )
                for entity_id, parent_id in frame["parent_id"].items()
            }

        fixed_map = route_map(fixed)
        terrain_map = route_map(terrain)
        fixed_routes = full_routes(fixed_map)
        terrain_routes = full_routes(terrain_map)
        node_ids = fixed.index[fixed["entity_type"].astype(str) == "node"].tolist()
        if not node_ids:
            raise ValueError(f"No end-device nodes in routing pair {keys}")
        parent_changes = sum(
            fixed_map[node_id] != terrain_map[node_id] for node_id in node_ids
        )
        route_changes = sum(
            fixed_routes[node_id] != terrain_routes[node_id] for node_id in node_ids
        )
        rows.append(
            {
                **dict(zip(PAIR_KEYS, keys)),
                "route_change_population": "end_device_nodes",
                "route_change_denominator": len(node_ids),
                "parent_change_count": parent_changes,
                "parent_change_ratio": parent_changes / len(node_ids),
                "full_route_change_count": route_changes,
                "full_route_change_ratio": route_changes / len(node_ids),
                "fixed_mean_hop_count": sum(
                    len(fixed_routes[node_id]) - 1 for node_id in node_ids
                )
                / len(node_ids),
                "terrain_specific_mean_hop_count": sum(
                    len(terrain_routes[node_id]) - 1 for node_id in node_ids
                )
                / len(node_ids),
            }
        )
    return pd.DataFrame(rows)


def add_reference_identity_anchor(
    runs: pd.DataFrame,
    route_changes: pd.DataFrame,
    *,
    reference_delta_h: int = 90,
) -> pd.DataFrame:
    """Reuse the fixed-reference result at delta_h=90 only with route proof.

    This does not invent a second simulation result.  It marks the same MAIN_W4
    run as the comparison anchor after independently constructed routing maps
    prove zero parent and full-route changes at the reference terrain.
    """
    _require_columns(runs, (*PAIR_KEYS, "routing_strategy"), "routing runs")
    _require_columns(
        route_changes,
        (*PAIR_KEYS, "parent_change_ratio", "full_route_change_ratio"),
        "route changes",
    )
    result = runs.copy()
    if "comparison_source" not in result.columns:
        result["comparison_source"] = "executed_simulation"
    reference = result.loc[
        pd.to_numeric(result["terrain_delta_h"], errors="raise").astype(int)
        == reference_delta_h
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in reference.groupby(list(PAIR_KEYS), sort=True, dropna=False):
        fixed = _one_strategy_row(group, FIXED_REFERENCE)
        terrain = group.loc[
            group["routing_strategy"].astype(str) == TERRAIN_SPECIFIC
        ]
        if len(terrain) == 1:
            continue
        if len(terrain) > 1:
            raise ValueError(f"Duplicate terrain-specific reference rows for {keys}")
        evidence = route_changes
        for column, value in zip(PAIR_KEYS, keys):
            evidence = evidence.loc[evidence[column] == value]
        if len(evidence) != 1:
            raise ValueError(f"Missing unique delta_h=90 route evidence for {keys}")
        evidence_row = evidence.iloc[0]
        if (
            float(evidence_row["parent_change_ratio"]) != 0.0
            or float(evidence_row["full_route_change_ratio"]) != 0.0
        ):
            raise ValueError(f"delta_h=90 routing strategies are not identical for {keys}")
        anchor = fixed.to_dict()
        anchor["routing_strategy"] = TERRAIN_SPECIFIC
        anchor["comparison_source"] = "fixed_reference_identity_anchor"
        rows.append(anchor)
    if rows:
        result = pd.concat([result, pd.DataFrame(rows)], ignore_index=True)
    return result


def build_r1_5_paired_runs(
    runs: pd.DataFrame,
    route_changes: pd.DataFrame,
) -> pd.DataFrame:
    """Build TERRAIN_SPECIFIC minus FIXED_REFERENCE paired run differences."""
    required = (
        *PAIR_KEYS,
        "routing_strategy",
        "offered_traffic_hash",
        "topology_hash",
        "routing_hash",
        "offered_packets",
        "delivered_packets",
        "pdr_percent",
        "total_network_energy_J",
        "energy_per_delivered_packet_J",
        "avg_e2e_delay_ms",
    )
    _require_columns(runs, required, "routing-sensitivity runs")
    if runs.duplicated([*PAIR_KEYS, "routing_strategy"]).any():
        raise ValueError("Duplicate routing-sensitivity model/seed/terrain/strategy row")

    rows: list[dict[str, Any]] = []
    for keys, group in runs.groupby(list(PAIR_KEYS), sort=True, dropna=False):
        fixed = _one_strategy_row(group, FIXED_REFERENCE)
        terrain = _one_strategy_row(group, TERRAIN_SPECIFIC)
        for column in ("offered_traffic_hash", "topology_hash", "offered_packets"):
            if fixed[column] != terrain[column]:
                raise ValueError(f"Paired routing {column} mismatch for {keys}")

        def difference(column: str) -> float:
            left = float(terrain[column])
            right = float(fixed[column])
            return left - right if math.isfinite(left) and math.isfinite(right) else math.nan

        rows.append(
            {
                **dict(zip(PAIR_KEYS, keys)),
                "difference_direction": "terrain_specific_minus_fixed_reference",
                "offered_traffic_hash": fixed["offered_traffic_hash"],
                "topology_hash": fixed["topology_hash"],
                "offered_packets": int(fixed["offered_packets"]),
                "fixed_delivered_packets": int(fixed["delivered_packets"]),
                "terrain_specific_delivered_packets": int(
                    terrain["delivered_packets"]
                ),
                "fixed_routing_hash": fixed["routing_hash"],
                "terrain_specific_routing_hash": terrain["routing_hash"],
                "routing_hash_equal": fixed["routing_hash"] == terrain["routing_hash"],
                "fixed_result_source": fixed.get(
                    "comparison_source", "executed_simulation"
                ),
                "terrain_specific_result_source": terrain.get(
                    "comparison_source", "executed_simulation"
                ),
                "pdr_difference_pp": difference("pdr_percent"),
                "total_energy_difference_J": difference("total_network_energy_J"),
                "energy_per_delivered_difference_J_per_packet": difference(
                    "energy_per_delivered_packet_J"
                ),
                "avg_e2e_delay_difference_ms": difference("avg_e2e_delay_ms"),
            }
        )
    paired = pd.DataFrame(rows)
    _require_columns(route_changes, (*PAIR_KEYS, "parent_change_ratio", "full_route_change_ratio"), "route changes")
    if route_changes.duplicated(list(PAIR_KEYS)).any():
        raise ValueError("Duplicate route-change model/seed/terrain row")
    merged = paired.merge(
        route_changes,
        on=list(PAIR_KEYS),
        how="left",
        validate="one_to_one",
    )
    if merged[["parent_change_ratio", "full_route_change_ratio"]].isna().any().any():
        raise ValueError("Missing route-change evidence for one or more paired runs")
    return merged


def summarize_r1_5_pairs(
    paired: pd.DataFrame,
    *,
    expected_seeds: Sequence[int] = APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS,
    expected_terrains: Sequence[int] = REVIEWER_R1_5_COMPARISON_TERRAINS,
) -> pd.DataFrame:
    """Return paired Student-t 95% CIs, requiring the predeclared five seeds."""
    expected = set(map(int, expected_seeds))
    observed_terrains = set(
        pd.to_numeric(paired["terrain_delta_h"], errors="raise").astype(int)
    )
    if observed_terrains != set(map(int, expected_terrains)):
        raise ValueError("Incomplete Reviewer R1.5 comparison terrains")
    for keys, group in paired.groupby(["model_id", "terrain_delta_h"], sort=True):
        seeds = pd.to_numeric(group["seed"], errors="raise").astype(int)
        if set(seeds) != expected or seeds.duplicated().any():
            raise ValueError(f"Incomplete paired seeds for {keys}")
        if group[list(ROUTING_EFFECT_METRICS)].isna().any().any():
            raise ValueError(f"Incomplete paired metric values for {keys}")
    result = aggregate_paper_grade(
        paired,
        group_columns=["model_id", "terrain_delta_h"],
        metric_columns=ROUTING_EFFECT_METRICS,
        confidence_level=0.95,
    )
    result.insert(result.columns.get_loc("mean"), "degrees_of_freedom", result["n"] - 1)
    return result


def build_r1_5_difference_in_differences(paired: pd.DataFrame) -> pd.DataFrame:
    """Subtract FF's routing effect from each adaptive/baseline model effect."""
    effect_metrics = ROUTING_EFFECT_METRICS[:4]
    index = ["seed", "terrain_delta_h"]
    ff = (
        paired.loc[paired["model_id"].astype(str) == "ff"]
        .set_index(index)
        .sort_index()
    )
    if ff.empty or ff.index.duplicated().any():
        raise ValueError("Exactly one FF routing pair is required per seed/terrain")
    rows: list[dict[str, Any]] = []
    for model_id, model in paired.loc[
        paired["model_id"].astype(str) != "ff"
    ].groupby("model_id", sort=True):
        selected = model.set_index(index).sort_index()
        if selected.index.duplicated().any() or not selected.index.equals(ff.index):
            raise ValueError(f"Unmatched FF difference-in-differences cells for {model_id}")
        for seed, terrain in selected.index:
            row: dict[str, Any] = {
                "model_id": model_id,
                "reference_model_id": "ff",
                "seed": seed,
                "terrain_delta_h": terrain,
            }
            for metric in effect_metrics:
                row[f"did_{metric}"] = selected.at[(seed, terrain), metric] - ff.at[
                    (seed, terrain), metric
                ]
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_r1_5_difference_in_differences(
    differences: pd.DataFrame,
    *,
    expected_seeds: Sequence[int] = APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS,
    expected_terrains: Sequence[int] = REVIEWER_R1_5_COMPARISON_TERRAINS,
) -> pd.DataFrame:
    """Return paired five-seed Student-t CIs for model-by-routing effects."""
    expected = set(map(int, expected_seeds))
    observed_terrains = set(
        pd.to_numeric(differences["terrain_delta_h"], errors="raise").astype(int)
    )
    if observed_terrains != set(map(int, expected_terrains)):
        raise ValueError("Incomplete Reviewer R1.5 interaction terrains")
    for keys, group in differences.groupby(
        ["model_id", "terrain_delta_h"], sort=True
    ):
        seeds = pd.to_numeric(group["seed"], errors="raise").astype(int)
        if set(seeds) != expected or seeds.duplicated().any():
            raise ValueError(f"Incomplete difference-in-differences seeds for {keys}")
        if group[list(DID_METRICS)].isna().any().any():
            raise ValueError(
                f"Incomplete difference-in-differences metric values for {keys}"
            )
    result = aggregate_paper_grade(
        differences,
        group_columns=["model_id", "reference_model_id", "terrain_delta_h"],
        metric_columns=DID_METRICS,
        confidence_level=0.95,
    )
    result.insert(result.columns.get_loc("mean"), "degrees_of_freedom", result["n"] - 1)
    return result


__all__ = [
    "ROUTING_EFFECT_METRICS",
    "add_reference_identity_anchor",
    "summarize_r1_5_difference_in_differences",
    "build_r1_5_difference_in_differences",
    "build_r1_5_paired_runs",
    "build_route_change_table",
    "summarize_r1_5_pairs",
]
