"""Reconstruct and compare routing for the frozen 300-run experiment grid.

This utility stops after topology generation and route selection.  It does not
register runtime processes or advance the SimPy environment, so it is a route
verification pass rather than a new simulation experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from loraphymac.experiments.apc_terrain_sweep.run import DEFAULT_CONFIG
from loraphymac.experiments.common.multiseed_pipeline import (
    build_experiment_grid,
    load_experiment_spec,
)
from loraphymac.metrics.instrumentation_metrics import (
    TOPOLOGY_PARENT_COLUMNS,
    build_topology_parent_by_run_df,
)
from loraphymac.simulation.builders import (
    build_infrastructure,
    build_routing,
    build_topology,
)
from loraphymac.simulator import resolve_configuration


def reconstruct_topology_parent_grid(config_path: str | Path) -> pd.DataFrame:
    """Rebuild route selection for every frozen grid row without simulation."""
    spec = load_experiment_spec(config_path)
    frames: list[pd.DataFrame] = []
    for run in build_experiment_grid(spec):
        config = resolve_configuration(run["scientific_config"])
        infrastructure = build_infrastructure(config)
        topology = build_topology(config, infrastructure)
        routing = build_routing(config, infrastructure, topology)
        metadata = {
            "run_id": run["run_id"],
            "seed": run["seed"],
            "node_ptx_mode": config["node_ptx_mode"],
            "relay_ptx_mode": config["relay_ptx_mode"],
            "terrain_delta_h": config["terrain_delta_h"],
        }
        frame = build_topology_parent_by_run_df(routing, metadata)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=TOPOLOGY_PARENT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def mode_parent_inconsistencies(parent_df: pd.DataFrame) -> pd.DataFrame:
    """Return parent mismatches across FF/AF/FA/AA for one seed and terrain."""
    nodes = parent_df[parent_df["entity_type"] == "node"].copy()
    group = ["seed", "terrain_delta_h", "entity_id"]
    rows = []
    for keys, frame in nodes.groupby(group, dropna=False, sort=True):
        parent_count = frame["parent_id"].nunique(dropna=False)
        mode_count = frame["mode"].nunique(dropna=False)
        if parent_count != 1 or mode_count != 4:
            rows.append(
                {
                    "seed": keys[0],
                    "terrain_delta_h": keys[1],
                    "entity_id": keys[2],
                    "mode_count": mode_count,
                    "parent_count": parent_count,
                    "mode_parent_pairs": ";".join(
                        f"{row.mode}:{row.parent_id}"
                        for row in frame[["mode", "parent_id"]]
                        .drop_duplicates()
                        .sort_values("mode")
                        .itertuples(index=False)
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "seed",
            "terrain_delta_h",
            "entity_id",
            "mode_count",
            "parent_count",
            "mode_parent_pairs",
        ],
    )


def build_parent_change_summary(
    parent_df: pd.DataFrame,
    *,
    reference_delta_h: int = 90,
) -> pd.DataFrame:
    """Summarize node-parent changes versus 90 m and across the full sweep."""
    nodes = parent_df[
        (parent_df["entity_type"] == "node") & (parent_df["mode"] == "FF")
    ][["seed", "terrain_delta_h", "entity_id", "parent_id"]].copy()
    rows: list[dict] = []

    for seed, seed_df in nodes.groupby("seed", sort=True):
        reference = seed_df[
            seed_df["terrain_delta_h"] == reference_delta_h
        ].set_index("entity_id")["parent_id"]
        if reference.empty:
            raise ValueError(
                f"Reference terrain_delta_h={reference_delta_h} missing for seed {seed}"
            )

        for terrain, terrain_df in seed_df.groupby("terrain_delta_h", sort=True):
            current = terrain_df.set_index("entity_id")["parent_id"]
            aligned = pd.concat(
                [reference.rename("reference"), current.rename("current")],
                axis=1,
                join="inner",
            )
            changed = int((aligned["reference"] != aligned["current"]).sum())
            total = int(len(aligned))
            rows.append(
                {
                    "comparison_type": "versus_reference",
                    "seed": int(seed),
                    "reference_delta_h": reference_delta_h,
                    "terrain_delta_h": int(terrain),
                    "changed_node_count": changed,
                    "compared_node_count": total,
                    "changed_node_percent": 100.0 * changed / total,
                }
            )

        terrains = sorted(seed_df["terrain_delta_h"].unique())
        for previous_terrain, terrain in zip(terrains, terrains[1:]):
            previous = seed_df[
                seed_df["terrain_delta_h"] == previous_terrain
            ].set_index("entity_id")["parent_id"]
            current = seed_df[
                seed_df["terrain_delta_h"] == terrain
            ].set_index("entity_id")["parent_id"]
            aligned = pd.concat(
                [previous.rename("previous"), current.rename("current")],
                axis=1,
                join="inner",
            )
            changed = int((aligned["previous"] != aligned["current"]).sum())
            total = int(len(aligned))
            rows.append(
                {
                    "comparison_type": "adjacent_delta_h",
                    "seed": int(seed),
                    "reference_delta_h": int(previous_terrain),
                    "terrain_delta_h": int(terrain),
                    "changed_node_count": changed,
                    "compared_node_count": total,
                    "changed_node_percent": 100.0 * changed / total,
                }
            )

        parent_counts = seed_df.groupby("entity_id")["parent_id"].nunique(
            dropna=False
        )
        any_changed = int((parent_counts > 1).sum())
        total_nodes = int(len(parent_counts))
        rows.append(
            {
                "comparison_type": "any_change_across_sweep",
                "seed": int(seed),
                "reference_delta_h": reference_delta_h,
                "terrain_delta_h": pd.NA,
                "changed_node_count": any_changed,
                "compared_node_count": total_nodes,
                "changed_node_percent": 100.0 * any_changed / total_nodes,
            }
        )

    summary = pd.DataFrame(rows)
    detailed = summary[summary["comparison_type"] == "versus_reference"]
    aggregate = (
        detailed.groupby(["comparison_type", "terrain_delta_h"], as_index=False)
        .agg(
            changed_node_count=("changed_node_count", "sum"),
            compared_node_count=("compared_node_count", "sum"),
        )
    )
    aggregate["seed"] = "ALL"
    aggregate["reference_delta_h"] = reference_delta_h
    aggregate["changed_node_percent"] = (
        100.0
        * aggregate["changed_node_count"]
        / aggregate["compared_node_count"]
    )
    aggregate = aggregate.reindex(columns=summary.columns)

    adjacent = summary[summary["comparison_type"] == "adjacent_delta_h"]
    adjacent_aggregate = (
        adjacent.groupby(
            ["comparison_type", "reference_delta_h", "terrain_delta_h"],
            as_index=False,
        )
        .agg(
            changed_node_count=("changed_node_count", "sum"),
            compared_node_count=("compared_node_count", "sum"),
        )
    )
    adjacent_aggregate["seed"] = "ALL"
    adjacent_aggregate["changed_node_percent"] = (
        100.0
        * adjacent_aggregate["changed_node_count"]
        / adjacent_aggregate["compared_node_count"]
    )
    adjacent_aggregate = adjacent_aggregate.reindex(columns=summary.columns)

    adjacent_all = pd.DataFrame(
        [
            {
                "comparison_type": "adjacent_all_transitions",
                "seed": "ALL",
                "reference_delta_h": pd.NA,
                "terrain_delta_h": pd.NA,
                "changed_node_count": int(adjacent["changed_node_count"].sum()),
                "compared_node_count": int(adjacent["compared_node_count"].sum()),
                "changed_node_percent": (
                    100.0
                    * adjacent["changed_node_count"].sum()
                    / adjacent["compared_node_count"].sum()
                ),
            }
        ]
    )

    sweep = summary[summary["comparison_type"] == "any_change_across_sweep"]
    sweep_aggregate = pd.DataFrame(
        [
            {
                "comparison_type": "any_change_across_sweep",
                "seed": "ALL",
                "reference_delta_h": reference_delta_h,
                "terrain_delta_h": pd.NA,
                "changed_node_count": int(sweep["changed_node_count"].sum()),
                "compared_node_count": int(sweep["compared_node_count"].sum()),
                "changed_node_percent": (
                    100.0
                    * sweep["changed_node_count"].sum()
                    / sweep["compared_node_count"].sum()
                ),
            }
        ]
    )
    return pd.concat(
        [
            summary,
            aggregate,
            adjacent_aggregate,
            adjacent_all,
            sweep_aggregate,
        ],
        ignore_index=True,
    )


def run_routing_audit(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Write routing evidence to a new verification directory."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    parent_df = reconstruct_topology_parent_grid(config_path)
    inconsistencies = mode_parent_inconsistencies(parent_df)
    changes = build_parent_change_summary(parent_df)

    parent_df.to_csv(output / "topology_parent_by_run.csv", index=False)
    inconsistencies.to_csv(output / "mode_parent_inconsistencies.csv", index=False)
    changes.to_csv(output / "parent_change_summary.csv", index=False)

    sweep_all = changes[
        (changes["comparison_type"] == "any_change_across_sweep")
        & (changes["seed"].astype(str) == "ALL")
    ].iloc[0]
    adjacent_all = changes[
        changes["comparison_type"] == "adjacent_all_transitions"
    ].iloc[0]
    result = {
        "route_rows": int(len(parent_df)),
        "run_count": int(parent_df["run_id"].nunique()),
        "node_route_rows": int((parent_df["entity_type"] == "node").sum()),
        "mode_parent_inconsistency_count": int(len(inconsistencies)),
        "node_seed_pairs_with_any_parent_change": int(
            sweep_all["changed_node_count"]
        ),
        "node_seed_pair_count": int(sweep_all["compared_node_count"]),
        "node_seed_pairs_with_any_parent_change_percent": float(
            sweep_all["changed_node_percent"]
        ),
        "adjacent_parent_change_event_count": int(
            adjacent_all["changed_node_count"]
        ),
        "adjacent_parent_comparison_count": int(
            adjacent_all["compared_node_count"]
        ),
        "adjacent_parent_change_event_percent": float(
            adjacent_all["changed_node_percent"]
        ),
        "reference_delta_h": 90,
        "method": "topology_and_routing_reconstruction_only_no_event_simulation",
    }
    (output / "routing_verification_summary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct APC terrain-sweep routes without simulation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_routing_audit(args.config, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_parent_change_summary",
    "mode_parent_inconsistencies",
    "reconstruct_topology_parent_grid",
    "run_routing_audit",
]
