"""Preflight and development reporting for the W3C paired-routing pilot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from loraphymac.experiments.apc_compare_model_terrain_sweep.contract import (
    PAIRED_ROUTING_PILOT_MODELS,
    PAIRED_ROUTING_PILOT_PROFILE,
    PAIRED_ROUTING_PILOT_RUN_COUNT,
    PAIRED_ROUTING_PILOT_STRATEGIES,
    PAIRED_ROUTING_PILOT_TERRAINS,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.postprocessing import (
    _read_csv,
    _read_json,
    _run_metrics,
    _validate_matched_cells,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.routing_sensitivity_reporting import (
    build_r1_5_paired_runs,
    build_route_change_table,
)
from loraphymac.experiments.common.multiseed_pipeline import (
    atomic_write_csv,
    atomic_write_json,
    build_experiment_grid,
    content_hash,
    file_sha256,
    utc_now,
)
from loraphymac.experiments.common.outputs import ExperimentArtifactPaths
from loraphymac.routing.snapshot import full_routes
from loraphymac.simulation.builders import (
    build_infrastructure,
    build_routing,
    build_topology,
)
from loraphymac.simulator import resolve_configuration


ROUTING_CONFIG_KEYS = {
    "routing_strategy",
    "routing_reference_delta_h",
    "routing_snapshot_manifest",
    "routing_snapshot_manifest_sha256",
}


def _runner_api():
    from loraphymac.experiments.apc_compare_model_terrain_sweep.run import (
        PAIRED_ROUTING_PILOT_CONFIG,
        build_model_specs,
        load_sweep_definition,
        planned_grid,
    )

    return (
        PAIRED_ROUTING_PILOT_CONFIG,
        build_model_specs,
        load_sweep_definition,
        planned_grid,
    )


def _model_and_strategy(spec: Any) -> tuple[str, str]:
    return (
        str(spec.base_config["tx_power_model_id"]),
        str(spec.base_config["routing_strategy"]),
    )


def _packet_outcome_hash(packet_fates: pd.DataFrame) -> str:
    columns = (
        "src_id",
        "fcnt",
        "final_status",
        "terminal_reason",
        "delivered_time",
        "e2e_delay_s",
    )
    missing = set(columns).difference(packet_fates.columns)
    if missing:
        raise ValueError(f"packet_fates missing outcome fields: {sorted(missing)}")
    frame = packet_fates.loc[:, columns].sort_values(
        ["src_id", "fcnt"], kind="stable"
    )
    records = []
    for row in frame.itertuples(index=False):
        records.append(
            {
                "src_id": str(row.src_id),
                "fcnt": int(row.fcnt),
                "final_status": str(row.final_status),
                "terminal_reason": None
                if pd.isna(row.terminal_reason)
                else str(row.terminal_reason),
                "delivered_time_ns": None
                if pd.isna(row.delivered_time)
                else round(float(row.delivered_time) * 1_000_000_000),
                "e2e_delay_ns": None
                if pd.isna(row.e2e_delay_s)
                else round(float(row.e2e_delay_s) * 1_000_000_000),
            }
        )
    return content_hash(records, length=64)


def _nonrouting_scientific_signature(resolved: dict[str, Any]) -> str:
    config = dict(resolved["scientific_config"])
    for key in ROUTING_CONFIG_KEYS:
        config.pop(key, None)
    return content_hash(config, length=64)


def _selected_link_quality(topology: pd.DataFrame) -> pd.DataFrame:
    links = topology.loc[topology["parent_id"].notna()].copy()
    links["pathloss_db"] = pd.to_numeric(links["pathloss_db"], errors="raise")
    links["margin_db"] = pd.to_numeric(links["margin_db"], errors="raise")
    links["physically_feasible"] = (
        links["valid"].astype(str).str.strip().str.lower().eq("true")
    )
    keys = ["model_id", "routing_strategy", "terrain_delta_h", "seed"]
    rows = []
    for values, group in links.groupby(keys, sort=True):
        feasible = int(group["physically_feasible"].sum())
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "selected_link_count": len(group),
                "physically_feasible_link_count": feasible,
                "physically_infeasible_link_count": len(group) - feasible,
                "physically_feasible_link_ratio": feasible / len(group),
                "physical_itm_pathloss_mean_db": float(group["pathloss_db"].mean()),
                "physical_itm_pathloss_max_db": float(group["pathloss_db"].max()),
                "physical_itm_margin_mean_db": float(group["margin_db"].mean()),
                "physical_itm_margin_min_db": float(group["margin_db"].min()),
            }
        )
    return pd.DataFrame(rows)


def _apc_offline_belief_diagnostics(
    feedback: pd.DataFrame, active_entity_ids: list[str]
) -> dict[str, Any]:
    actual = {entity_id: 0 for entity_id in active_entity_ids}
    estimated = actual.copy()
    if not feedback.empty:
        ordered = feedback.sort_values("event_sequence", kind="stable")
        for row in ordered.itertuples(index=False):
            target_id = str(row.target_id)
            if target_id not in actual:
                continue
            if str(row.event) == "generated" and not pd.isna(
                row.decision_tx_power_index
            ):
                estimated[target_id] = int(row.decision_tx_power_index)
            if str(row.event) == "applied" and not pd.isna(
                row.applied_tx_power_index
            ):
                actual[target_id] = int(row.applied_tx_power_index)
    absolute = [abs(estimated[key] - actual[key]) for key in active_entity_ids]
    mismatch_count = sum(value != 0 for value in absolute)
    population = len(active_entity_ids)
    return {
        "apc_offline_tpi_population": population,
        "apc_offline_tpi_mismatch_count": mismatch_count,
        "apc_offline_tpi_mismatch_fraction": (
            mismatch_count / population if population else float("nan")
        ),
        "apc_offline_tpi_mean_absolute_difference": (
            sum(absolute) / population if population else float("nan")
        ),
        "apc_offline_tpi_mismatch_scope": (
            "postprocessing_only_from_serialized_lifecycle_no_runtime_feedback"
        ),
    }


def preflight_paired_routing_pilot(
    config: str | Path | None = None,
    *,
    require_new_output: bool = True,
) -> dict[str, Any]:
    """Build routes only and fail before traffic if pairing is not causal."""
    default, build_specs, load_definition, planned = _runner_api()
    config_path = Path(config or default).resolve()
    definition = load_definition(config_path)
    if definition.experiment_profile != PAIRED_ROUTING_PILOT_PROFILE:
        raise ValueError("Not the registered paired-routing pilot profile")
    if require_new_output and definition.experiment_dir.exists():
        raise FileExistsError(
            f"Pilot output namespace already exists: {definition.experiment_dir}"
        )
    grid = planned(config_path)
    if len(grid) != PAIRED_ROUTING_PILOT_RUN_COUNT or len(
        {
            (
                row["model_id"],
                row["terrain_delta_h"],
                row["seed"],
                row["routing_strategy"],
            )
            for row in grid
        }
    ) != PAIRED_ROUTING_PILOT_RUN_COUNT:
        raise ValueError("Paired-routing pilot must contain exactly 30 unique cells")

    records: list[dict[str, Any]] = []
    specs = build_specs(config_path)
    for spec in specs.values():
        model_id, strategy = _model_and_strategy(spec)
        for run in build_experiment_grid(spec):
            resolved = resolve_configuration(run["scientific_config"])
            if resolved["application_scheduling_mode"] != "independent_fifo_v1":
                raise ValueError("Application scheduling is not independent")
            infrastructure = build_infrastructure(resolved)
            topology = build_topology(resolved, infrastructure)
            routing = build_routing(resolved, infrastructure, topology)
            records.append(
                {
                    "model_id": model_id,
                    "routing_strategy": strategy,
                    "terrain_delta_h": int(resolved["terrain_delta_h"]),
                    "seed": int(resolved["seed"]),
                    "topology_hash": routing.topology_sha256,
                    "routing_hash": routing.selected_parent_routing_sha256,
                    "route_construction_delta_h": routing.route_construction_delta_h,
                    "physical_terrain_delta_h": routing.physical_terrain_delta_h,
                    "routing_snapshot_id": routing.routing_snapshot_id,
                    "routing_snapshot_sha256": routing.routing_snapshot_sha256,
                    "full_routes_hash": content_hash(
                        {
                            key: list(value)
                            for key, value in full_routes(routing.routing_map).items()
                        },
                        length=64,
                    ),
                }
            )
    evidence = pd.DataFrame(records)
    checks: dict[str, bool] = {
        "exact_30_cells": len(evidence) == PAIRED_ROUTING_PILOT_RUN_COUNT,
        "one_topology": evidence["topology_hash"].nunique() == 1,
        "physical_delta_matches_scenario": (
            evidence["physical_terrain_delta_h"].astype(int)
            == evidence["terrain_delta_h"].astype(int)
        ).all(),
        "fixed_reference_built_at_90": (
            evidence.loc[
                evidence["routing_strategy"] == "fixed_reference",
                "route_construction_delta_h",
            ].astype(int)
            == 90
        ).all(),
        "terrain_specific_built_at_scenario": (
            evidence.loc[
                evidence["routing_strategy"] == "terrain_specific",
                "route_construction_delta_h",
            ].astype(int)
            == evidence.loc[
                evidence["routing_strategy"] == "terrain_specific",
                "terrain_delta_h",
            ].astype(int)
        ).all(),
        "fixed_routing_constant_across_terrain": evidence.loc[
            evidence["routing_strategy"] == "fixed_reference", "routing_hash"
        ].nunique()
        == 1,
        "routing_common_across_models_per_strategy_terrain": (
            evidence.groupby(["routing_strategy", "terrain_delta_h"])[
                "routing_hash"
            ].nunique()
            == 1
        ).all(),
        "delta90_parent_identity": evidence.loc[
            evidence["terrain_delta_h"] == 90
        ]["routing_hash"].nunique()
        == 1,
        "delta90_full_route_identity": evidence.loc[
            evidence["terrain_delta_h"] == 90
        ]["full_routes_hash"].nunique()
        == 1,
        "fixed_snapshot_never_fallback": evidence.loc[
            evidence["routing_strategy"] == "fixed_reference",
            "routing_snapshot_id",
        ].notna().all(),
        "terrain_specific_has_no_snapshot": evidence.loc[
            evidence["routing_strategy"] == "terrain_specific",
            "routing_snapshot_id",
        ].isna().all(),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError(f"Paired-routing preflight failed: {failed}")
    return {
        "status": "PASS",
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "experiment_dir": str(definition.experiment_dir),
        "planned_run_count": len(grid),
        "checks": checks,
        "route_evidence": records,
    }


def _diagnostics_for_run(
    model_id: str,
    strategy: str,
    terrain: int,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    metrics, validation = _run_metrics(
        model_id=model_id,
        run_dir=run_dir,
        expected_seed=30,
        expected_terrain=terrain,
        expected_routing_strategy=strategy,
    )
    tables = run_dir / "tables"
    fates = _read_csv(tables / "packet_fates.csv")
    topology = _read_csv(tables / "topology_parent_by_run.csv")
    topology.insert(0, "model_id", model_id)
    topology["routing_strategy"] = strategy
    validation["packet_outcome_hash"] = _packet_outcome_hash(fates)
    resolved = dict(_read_json(run_dir / "resolved_config.json"))
    validation["nonrouting_scientific_input_signature"] = (
        _nonrouting_scientific_signature(resolved)
    )
    core = _read_csv(run_dir / "core_summary.csv").iloc[0]
    metrics["served_node_count"] = int(core["served_node_count"])
    metrics["unserved_node_count"] = int(core["node_num"]) - int(
        core["served_node_count"]
    )
    final_ptx = _read_csv(tables / "final_ptx_distribution.csv")
    active = final_ptx.loc[final_ptx["entity_type"].isin(["node", "relay"])]
    diagnostics: dict[str, Any] = {
        "model_id": model_id,
        "routing_strategy": strategy,
        "terrain_delta_h": terrain,
        "seed": 30,
        "mean_final_ptx_dbm": float(active["final_ptx"].mean()),
        "selected_at_max_ptx_count": int(
            active.get("selected_at_max_ptx", pd.Series(dtype=float))
            .eq(True)
            .sum()
        ),
        "outside_estimated_coverage_count": int(
            active.get("outside_estimated_coverage", pd.Series(dtype=float))
            .eq(True)
            .sum()
        ),
    }
    if model_id == "aa":
        feedback = _read_csv(tables / "apc_feedback_lifecycle.csv", allow_empty=True)
        counts = feedback["event"].astype(str).value_counts() if not feedback.empty else {}
        diagnostics.update(
            {
                "apc_unique_commands": int(
                    feedback.loc[
                        feedback["event"].astype(str) == "generated", "feedback_id"
                    ].nunique()
                ),
                "apc_physical_sends": int(counts.get("sent", 0)),
                "apc_retry_generated": int(counts.get("retry_generated", 0)),
                "apc_retry_exhausted": int(counts.get("retry_exhausted", 0)),
                "apc_controller_state_scope": "controller_local_no_physical_ack",
                **_apc_offline_belief_diagnostics(
                    feedback, active["entity_id"].astype(str).tolist()
                ),
            }
        )
    if model_id == "islam_dbpc_802" and not active.empty:
        diagnostics.update(
            {
                "estimator_id": str(active["estimator_id"].dropna().iloc[0]),
                "policy_margin_db": float(active["policy_margin_db"].dropna().iloc[0]),
                "runtime_adaptation_count": 0,
            }
        )
    return metrics, validation, topology, diagnostics


def build_paired_pilot_artifacts(config: str | Path | None = None) -> dict[str, Any]:
    """Validate completed runs and write development-only paired tables."""
    default, build_specs, load_definition, _ = _runner_api()
    config_path = Path(config or default).resolve()
    definition = load_definition(config_path)
    parent = ExperimentArtifactPaths(definition.experiment_dir).ensure()
    parent_status = _read_json(parent.status)
    if parent_status.get("status") != "PASS":
        raise ValueError("Paired-routing pilot is incomplete")

    metric_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    topology_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for spec in build_specs(config_path).values():
        model_id, strategy = _model_and_strategy(spec)
        child = ExperimentArtifactPaths.from_spec(spec)
        child_status = _read_json(child.status)
        if child_status.get("status") != "PASS":
            raise ValueError(f"Incomplete child {strategy}/{model_id}")
        for run_dir in sorted(path for path in child.runs.iterdir() if path.is_dir()):
            resolved = _read_json(run_dir / "resolved_config.json")
            terrain = int(resolved["scientific_config"]["terrain_delta_h"])
            metrics, validation, topology, diagnostics = _diagnostics_for_run(
                model_id, strategy, terrain, run_dir
            )
            metric_rows.append(metrics)
            validation_rows.append(validation)
            topology_frames.append(topology)
            diagnostic_rows.append(diagnostics)

    per_run = pd.DataFrame(metric_rows)
    validation = pd.DataFrame(validation_rows)
    within = []
    for strategy in PAIRED_ROUTING_PILOT_STRATEGIES:
        selected = validation.loc[validation["routing_strategy"] == strategy]
        within.append(_validate_matched_cells(selected, PAIRED_ROUTING_PILOT_MODELS))
    validation = pd.concat(within, ignore_index=True)

    paired_checks: list[dict[str, Any]] = []
    for keys, group in validation.groupby(
        ["model_id", "terrain_delta_h", "seed"], sort=True
    ):
        if set(group["routing_strategy"]) != set(PAIRED_ROUTING_PILOT_STRATEGIES):
            raise ValueError(f"Missing paired strategy for {keys}")
        fixed = group.loc[group["routing_strategy"] == "fixed_reference"].iloc[0]
        terrain = group.loc[group["routing_strategy"] == "terrain_specific"].iloc[0]
        row = {
            "model_id": keys[0],
            "terrain_delta_h": keys[1],
            "seed": keys[2],
            "matched_topology": fixed["topology_hash"] == terrain["topology_hash"],
            "matched_offered_traffic": fixed["offered_traffic_hash"]
            == terrain["offered_traffic_hash"],
            "matched_nonrouting_scientific_input": fixed[
                "nonrouting_scientific_input_signature"
            ]
            == terrain["nonrouting_scientific_input_signature"],
            "delta90_routing_equal": True
            if int(keys[1]) != 90
            else fixed["routing_hash"] == terrain["routing_hash"],
            "delta90_packet_outcomes_equal": True
            if int(keys[1]) != 90
            else fixed["packet_outcome_hash"] == terrain["packet_outcome_hash"],
        }
        row["paired_validation_pass"] = all(
            value for key, value in row.items() if key.startswith("matched_") or key.startswith("delta90_")
        )
        paired_checks.append(row)
    paired_validation = pd.DataFrame(paired_checks)
    if not validation["validation_pass"].all() or not paired_validation[
        "paired_validation_pass"
    ].all():
        raise ValueError("Paired workload/topology validation failed")

    topology = pd.concat(topology_frames, ignore_index=True)
    route_changes = build_route_change_table(topology)
    selected_link_quality = _selected_link_quality(topology)
    hashes = validation[
        [
            "model_id",
            "routing_strategy",
            "terrain_delta_h",
            "seed",
            "offered_traffic_hash",
            "topology_hash",
            "routing_hash",
        ]
    ]
    paired_input = per_run.merge(
        hashes,
        on=["model_id", "routing_strategy", "terrain_delta_h", "seed"],
        validate="one_to_one",
    )
    paired = build_r1_5_paired_runs(paired_input, route_changes)

    output_tables = {
        "per_run_metrics.csv": per_run,
        "validation_summary.csv": validation,
        "paired_validation.csv": paired_validation,
        "paired_routing_metrics.csv": paired,
        "route_change_metrics.csv": route_changes,
        "selected_link_quality.csv": selected_link_quality,
        "model_specific_diagnostics.csv": pd.DataFrame(diagnostic_rows),
        "fixed_reference_diagnostics.csv": per_run.loc[
            per_run["routing_strategy"] == "fixed_reference"
        ],
    }
    for filename, frame in output_tables.items():
        atomic_write_csv(parent.tables / filename, frame)
    manifest = {
        "manifest_version": "w3c_paired_routing_pilot_analysis_v1",
        "created_at": utc_now(),
        "development_only": True,
        "single_seed_no_statistical_significance": True,
        "validated_run_count": len(per_run),
        "model_ids": list(PAIRED_ROUTING_PILOT_MODELS),
        "routing_strategies": list(PAIRED_ROUTING_PILOT_STRATEGIES),
        "terrains": list(PAIRED_ROUTING_PILOT_TERRAINS),
        "difference_direction": "terrain_specific_minus_fixed_reference",
        "wang_terminal_status_contract": (
            "generated = applied_confirmed + applied_unconfirmed + "
            "not_applied + censored_at_horizon; the three terminal outcomes "
            "alone conserve terminal-eligible commands"
        ),
        "analysis_source_files": {
            "src/loraphymac/experiments/apc_compare_model_terrain_sweep/"
            "paired_routing_pilot.py": file_sha256(Path(__file__)),
            "src/loraphymac/experiments/apc_compare_model_terrain_sweep/"
            "postprocessing.py": file_sha256(Path(__file__).with_name("postprocessing.py")),
            "src/loraphymac/experiments/apc_compare_model_terrain_sweep/"
            "routing_sensitivity_reporting.py": file_sha256(
                Path(__file__).with_name("routing_sensitivity_reporting.py")
            ),
        },
        "outputs": {
            filename: file_sha256(parent.tables / filename)
            for filename in output_tables
        },
    }
    atomic_write_json(definition.experiment_dir / "analysis_manifest.json", manifest)
    return {
        "status": "PASS",
        "validated_run_count": len(per_run),
        "tables": [str(parent.tables / name) for name in output_tables],
        "analysis_manifest": str(definition.experiment_dir / "analysis_manifest.json"),
    }


__all__ = ["build_paired_pilot_artifacts", "preflight_paired_routing_pilot"]
