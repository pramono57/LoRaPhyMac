"""Dedicated reporting-only comparison for Reviewer R1.5 routing sensitivity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from loraphymac.experiments.apc_compare_model_terrain_sweep.contract import (
    APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS,
    MAIN_W4_MODELS,
    REVIEWER_R1_5_TERRAINS,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.postprocessing import (
    FIGURE_SIZE,
    FIGURE_STYLE,
    MODEL_STYLES,
    OFFERED_TRAFFIC_HASH_VERSION,
    PLOT_MODEL_ORDER,
    _offered_traffic_hash,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.routing_sensitivity_reporting import (
    FIXED_REFERENCE,
    TERRAIN_SPECIFIC,
    build_route_change_table,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.run import MODEL_LABELS
from loraphymac.experiments.common.multiseed_pipeline import (
    aggregate_paper_grade,
    atomic_write_csv,
    atomic_write_json,
    content_hash,
    file_sha256,
    utc_now,
)


DIFFERENCE_DIRECTION = "terrain_specific_minus_fixed_reference"
REPORTER_VERSION = "reviewer_r1_5_paired_routing_reporter_v1"
ROUTING_TREATMENT_KEYS = frozenset(
    {
        "experiment_profile",
        "routing_strategy",
        "routing_reference_delta_h",
        "routing_snapshot_manifest",
        "routing_snapshot_manifest_sha256",
    }
)
PAIR_KEYS = ("model_id", "terrain_delta_h", "seed")
STATISTIC_METRICS = (
    "delivered_packets_difference",
    "pdr_difference_pp",
    "total_energy_difference_J",
    "energy_per_delivered_difference_J_per_packet",
    "avg_e2e_delay_difference_ms",
    "throughput_difference_bps",
    "terminal_drop_difference_packets",
    "censored_difference_packets",
    "not_feasible_difference_packets",
    "collision_difference_packets",
    "half_duplex_difference_packets",
    "parent_change_ratio",
    "full_route_change_ratio",
)
INTERACTION_METRICS = (
    "pdr_difference_pp",
    "total_energy_difference_J",
    "energy_per_delivered_difference_J_per_packet",
    "avg_e2e_delay_difference_ms",
    "throughput_difference_bps",
)
FIGURE_METRICS = (
    (
        "pdr_difference_pp",
        "PDR difference (percentage points)",
        "reviewer_r1_5_figure_pdr_difference",
    ),
    (
        "total_energy_difference_J",
        "Total network energy difference (J)",
        "reviewer_r1_5_figure_energy_difference",
    ),
    (
        "avg_e2e_delay_difference_ms",
        "Average end-to-end delay difference (ms)",
        "reviewer_r1_5_figure_delay_difference",
    ),
)


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _single_row(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"{path}: expected exactly one row, got {len(frame)}")
    return frame.iloc[0]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def physical_configuration_signature(
    scientific_config: Mapping[str, Any],
) -> str:
    """Hash all scientific inputs except declared routing/provenance treatment."""
    normalized = {
        key: value for key, value in scientific_config.items() if key not in ROUTING_TREATMENT_KEYS
    }
    return content_hash(normalized, length=64)


def _terminal_reason_counts(packet_fates: pd.DataFrame) -> dict[str, int]:
    terminal = packet_fates.loc[packet_fates["final_status"].astype(str) == "TERMINAL_DROP"]
    reasons = terminal["terminal_reason"].fillna("").astype(str)
    not_feasible = int((reasons == "not_feasible").sum())
    collision = int((reasons == "collision").sum())
    half_duplex = int(reasons.str.startswith("half_duplex").sum())
    return {
        "not_feasible_packets": not_feasible,
        "collision_packets": collision,
        "half_duplex_packets": half_duplex,
        "other_terminal_packets": int(len(terminal)) - not_feasible - collision - half_duplex,
    }


def _run_record(
    *,
    model_id: str,
    strategy: str,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, list[Path]]:
    paths = [
        run_dir / "status.json",
        run_dir / "resolved_config.json",
        run_dir / "core_summary.csv",
        run_dir / "tables" / "packet_fates.csv",
        run_dir / "tables" / "topology_parent_by_run.csv",
        run_dir / "tables" / "energy_breakdown.csv",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Reviewer R1.5 input files: {missing}")

    status = _read_json(paths[0])
    resolved = _read_json(paths[1])
    core = _single_row(paths[2])
    fates = pd.read_csv(paths[3])
    topology = pd.read_csv(paths[4])
    energy = pd.read_csv(paths[5])
    scientific = dict(resolved["scientific_config"])
    terrain = int(scientific["terrain_delta_h"])
    seed = int(scientific["seed"])

    offered = int(core["packet_fate_offered_count"])
    delivered = int(core["packet_fate_delivered_count"])
    terminal = int(core["packet_fate_terminal_drop_count"])
    censored = int(core["packet_fate_censored_count"])
    delivered_rows = fates.loc[fates["final_status"].astype(str) == "DELIVERED"]
    recomputed_delay = (
        float(pd.to_numeric(delivered_rows["e2e_delay_ms"], errors="raise").mean())
        if delivered
        else math.nan
    )
    energy_total = float(pd.to_numeric(energy["total_energy_J"], errors="raise").sum())
    expected_epp = energy_total / delivered if delivered else math.nan
    expected_pdr = 100.0 * delivered / offered if offered else math.nan
    expected_throughput = (
        delivered * int(core["node_app_payload_size_bytes"]) * 8.0 / float(core["sim_time"])
    )
    checks = {
        "status_pass": status.get("status") == "PASS",
        "model_identity_match": str(core["tx_power_model_id"]) == model_id,
        "routing_strategy_match": str(scientific["routing_strategy"]) == strategy,
        "packet_conservation": offered == delivered + terminal + censored,
        "packet_fate_rows_match": len(fates) == offered,
        "delivered_rows_match": len(delivered_rows) == delivered,
        "energy_accounting_corrected": (
            str(core["energy_accounting_mode"]) == "corrected"
            and set(energy["energy_accounting_mode"].astype(str)) == {"corrected"}
        ),
        "energy_conservation": math.isclose(
            float(core["total_network_energy_J"]),
            energy_total,
            rel_tol=1e-9,
            abs_tol=1e-5,
        ),
        "pdr_recomputed": math.isclose(
            float(core["delivery_by_horizon_percent"]),
            expected_pdr,
            rel_tol=1e-9,
            abs_tol=1e-5,
        ),
        "delay_recomputed": delivered == 0
        or math.isclose(
            float(core["avg_e2e_delay_ms"]),
            recomputed_delay,
            rel_tol=1e-9,
            abs_tol=1e-3,
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"{run_dir}: input validation failed: {failed}")

    topology = topology.copy()
    topology.insert(0, "model_id", model_id)
    topology["routing_strategy"] = strategy
    topology["terrain_delta_h"] = terrain
    topology["seed"] = seed
    fate_counts = _terminal_reason_counts(fates)
    record = {
        "model_id": model_id,
        "model_label": MODEL_LABELS[model_id],
        "terrain_delta_h": terrain,
        "seed": seed,
        "routing_strategy": strategy,
        "run_id": str(core["run_id"]),
        "run_dir": str(run_dir),
        "offered_packets": offered,
        "delivered_packets": delivered,
        "terminal_drop_packets": terminal,
        "censored_packets": censored,
        "pdr_percent": expected_pdr,
        "total_network_energy_J": energy_total,
        "energy_per_delivered_packet_J": expected_epp,
        "avg_e2e_delay_ms": recomputed_delay,
        "throughput_bps": expected_throughput,
        "topology_hash": str(core["topology_hash"]),
        "routing_hash": str(core["selected_parent_routing_hash"]),
        "offered_traffic_hash": _offered_traffic_hash(fates),
        "physical_configuration_signature": physical_configuration_signature(scientific),
        "route_construction_delta_h": core["route_construction_delta_h"],
        "physical_terrain_delta_h": int(core["physical_terrain_delta_h"]),
        **fate_counts,
    }
    validation = {
        "model_id": model_id,
        "terrain_delta_h": terrain,
        "seed": seed,
        "routing_strategy": strategy,
        "run_id": str(core["run_id"]),
        **checks,
        "run_validation_pass": True,
        "failed_checks": "",
    }
    return record, validation, topology, paths


def _validate_parent_experiment(
    root: Path,
    *,
    expected_run_count: int,
    expected_models: Sequence[str],
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[Path]]:
    manifest_path = root / "experiment_manifest.json"
    status_path = root / "experiment_status.json"
    if not manifest_path.is_file() or not status_path.is_file():
        raise FileNotFoundError(f"Missing parent manifest/status in {root}")
    manifest = _read_json(manifest_path)
    status = _read_json(status_path)
    if status.get("status") != "PASS":
        raise ValueError(f"{root}: parent status is not PASS")
    if int(status.get("passed_run_count", -1)) != expected_run_count:
        raise ValueError(f"{root}: unexpected passed-run count")
    if int(manifest.get("planned_run_count", -1)) != expected_run_count:
        raise ValueError(f"{root}: unexpected planned-run count")
    child_summaries = status.get("child_summaries", {})
    if set(child_summaries) != set(expected_models):
        raise ValueError(f"{root}: unexpected model set")
    if any(summary.get("status") != "PASS" for summary in child_summaries.values()):
        raise ValueError(f"{root}: one or more child pipelines are incomplete")
    return manifest, status, [manifest_path, status_path]


def _collect_runs(
    root: Path,
    *,
    strategy: str,
    selected_terrains: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    records: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    topology_frames: list[pd.DataFrame] = []
    input_paths: list[Path] = []
    terrain_set = set(map(int, selected_terrains))
    for model_id in MAIN_W4_MODELS:
        model_root = root / "models" / model_id
        for metadata_name in ("experiment_manifest.json", "experiment_status.json"):
            metadata_path = model_root / metadata_name
            if not metadata_path.is_file():
                raise FileNotFoundError(metadata_path)
            input_paths.append(metadata_path)
        for run_dir in sorted(path for path in (model_root / "runs").iterdir() if path.is_dir()):
            resolved = _read_json(run_dir / "resolved_config.json")
            terrain = int(resolved["scientific_config"]["terrain_delta_h"])
            if terrain not in terrain_set:
                continue
            record, validation, topology, paths = _run_record(
                model_id=model_id,
                strategy=strategy,
                run_dir=run_dir,
            )
            records.append(record)
            validations.append(validation)
            topology_frames.append(topology)
            input_paths.extend(paths)
    return (
        pd.DataFrame(records),
        pd.DataFrame(validations),
        pd.concat(topology_frames, ignore_index=True),
        input_paths,
    )


def build_pair_validation(runs: pd.DataFrame) -> pd.DataFrame:
    """Validate physical/workload equality without requiring equal routing hashes."""
    rows: list[dict[str, Any]] = []
    for keys, group in runs.groupby(list(PAIR_KEYS), sort=True):
        indexed = group.set_index("routing_strategy")
        has_both = set(indexed.index) == {FIXED_REFERENCE, TERRAIN_SPECIFIC}
        row: dict[str, Any] = dict(zip(PAIR_KEYS, keys, strict=True))
        row["both_strategies_present"] = has_both and len(group) == 2
        if not row["both_strategies_present"]:
            row["validation_pass"] = False
            row["failed_checks"] = "both_strategies_present"
            rows.append(row)
            continue
        fixed = indexed.loc[FIXED_REFERENCE]
        terrain = indexed.loc[TERRAIN_SPECIFIC]
        checks = {
            "topology_hash_equal": fixed["topology_hash"] == terrain["topology_hash"],
            "offered_traffic_hash_equal": fixed["offered_traffic_hash"]
            == terrain["offered_traffic_hash"],
            "offered_packets_equal": fixed["offered_packets"] == terrain["offered_packets"],
            "physical_configuration_equal": fixed["physical_configuration_signature"]
            == terrain["physical_configuration_signature"],
            "physical_terrain_equal": int(fixed["physical_terrain_delta_h"])
            == int(terrain["physical_terrain_delta_h"])
            == int(keys[1]),
            "fixed_built_at_reference_90": int(fixed["route_construction_delta_h"]) == 90,
            "terrain_specific_built_at_scenario": int(terrain["route_construction_delta_h"])
            == int(keys[1]),
        }
        failed = sorted(name for name, passed in checks.items() if not passed)
        row.update(
            {
                "fixed_reference_run_id": fixed["run_id"],
                "terrain_specific_run_id": terrain["run_id"],
                "fixed_reference_routing_hash": fixed["routing_hash"],
                "terrain_specific_routing_hash": terrain["routing_hash"],
                "routing_hash_equal": fixed["routing_hash"] == terrain["routing_hash"],
                "offered_traffic_hash": fixed["offered_traffic_hash"],
                **checks,
                "validation_pass": not failed,
                "failed_checks": ";".join(failed),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_paired_routing_per_run(
    runs: pd.DataFrame,
    route_changes: pd.DataFrame,
) -> pd.DataFrame:
    """Return one TERRAIN_SPECIFIC minus FIXED_REFERENCE row per paired run."""
    value_columns = {
        "delivered_packets": "delivered_packets_difference",
        "pdr_percent": "pdr_difference_pp",
        "total_network_energy_J": "total_energy_difference_J",
        "energy_per_delivered_packet_J": ("energy_per_delivered_difference_J_per_packet"),
        "avg_e2e_delay_ms": "avg_e2e_delay_difference_ms",
        "throughput_bps": "throughput_difference_bps",
        "terminal_drop_packets": "terminal_drop_difference_packets",
        "censored_packets": "censored_difference_packets",
        "not_feasible_packets": "not_feasible_difference_packets",
        "collision_packets": "collision_difference_packets",
        "half_duplex_packets": "half_duplex_difference_packets",
    }
    rows: list[dict[str, Any]] = []
    for keys, group in runs.groupby(list(PAIR_KEYS), sort=True):
        indexed = group.set_index("routing_strategy")
        if set(indexed.index) != {FIXED_REFERENCE, TERRAIN_SPECIFIC} or len(group) != 2:
            raise ValueError(f"Incomplete routing pair: {keys}")
        fixed = indexed.loc[FIXED_REFERENCE]
        terrain = indexed.loc[TERRAIN_SPECIFIC]
        row: dict[str, Any] = {
            **dict(zip(PAIR_KEYS, keys, strict=True)),
            "model_label": fixed["model_label"],
            "difference_direction": DIFFERENCE_DIRECTION,
            "offered_packets": int(fixed["offered_packets"]),
            "offered_traffic_hash": fixed["offered_traffic_hash"],
            "topology_hash": fixed["topology_hash"],
            "fixed_reference_run_id": fixed["run_id"],
            "terrain_specific_run_id": terrain["run_id"],
            "fixed_reference_routing_hash": fixed["routing_hash"],
            "terrain_specific_routing_hash": terrain["routing_hash"],
            "routing_hash_equal": fixed["routing_hash"] == terrain["routing_hash"],
        }
        for source_column, difference_column in value_columns.items():
            fixed_value = float(fixed[source_column])
            terrain_value = float(terrain[source_column])
            row[f"fixed_reference_{source_column}"] = fixed_value
            row[f"terrain_specific_{source_column}"] = terrain_value
            row[difference_column] = (
                terrain_value - fixed_value
                if math.isfinite(fixed_value) and math.isfinite(terrain_value)
                else math.nan
            )
        rows.append(row)
    paired = pd.DataFrame(rows)
    merged = paired.merge(
        route_changes,
        on=list(PAIR_KEYS),
        how="left",
        validate="one_to_one",
    )
    route_columns = [
        "parent_change_count",
        "parent_change_ratio",
        "full_route_change_count",
        "full_route_change_ratio",
    ]
    if merged[route_columns].isna().any().any():
        raise ValueError("Missing route-change evidence for paired rows")
    return merged.sort_values(list(PAIR_KEYS), kind="stable").reset_index(drop=True)


def summarize_pairs(paired: pd.DataFrame) -> pd.DataFrame:
    expected_seeds = set(map(int, APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS))
    expected_terrains = set(map(int, REVIEWER_R1_5_TERRAINS))
    if set(pd.to_numeric(paired["terrain_delta_h"]).astype(int)) != expected_terrains:
        raise ValueError("Incomplete Reviewer R1.5 terrain set")
    for keys, group in paired.groupby(["model_id", "terrain_delta_h"], sort=True):
        seeds = pd.to_numeric(group["seed"], errors="raise").astype(int)
        if set(seeds) != expected_seeds or seeds.duplicated().any():
            raise ValueError(f"Incomplete five-seed pairing for {keys}")
        if group[list(STATISTIC_METRICS)].isna().any().any():
            raise ValueError(f"Missing paired metric for {keys}")
    result = aggregate_paper_grade(
        paired,
        group_columns=["model_id", "terrain_delta_h"],
        metric_columns=STATISTIC_METRICS,
        confidence_level=0.95,
    )
    result.insert(result.columns.get_loc("mean"), "degrees_of_freedom", result["n"] - 1)
    return result


def build_model_routing_interactions(paired: pd.DataFrame) -> pd.DataFrame:
    index = ["terrain_delta_h", "seed"]
    ff = paired.loc[paired["model_id"] == "ff"].set_index(index).sort_index()
    if ff.empty or ff.index.duplicated().any():
        raise ValueError("Exactly one FF effect is required per terrain/seed")
    rows: list[dict[str, Any]] = []
    for model_id, model in paired.loc[paired["model_id"] != "ff"].groupby("model_id", sort=True):
        selected = model.set_index(index).sort_index()
        if not selected.index.equals(ff.index):
            raise ValueError(f"Unmatched FF interaction cells for {model_id}")
        for terrain, seed in selected.index:
            row: dict[str, Any] = {
                "model_id": model_id,
                "reference_model_id": "ff",
                "terrain_delta_h": terrain,
                "seed": seed,
                "interaction_direction": "routing_effect_model_minus_routing_effect_ff",
            }
            for metric in INTERACTION_METRICS:
                row[f"interaction_{metric}"] = float(selected.at[(terrain, seed), metric]) - float(
                    ff.at[(terrain, seed), metric]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_model_routing_interactions(
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    metrics = tuple(f"interaction_{metric}" for metric in INTERACTION_METRICS)
    result = aggregate_paper_grade(
        interactions,
        group_columns=["model_id", "reference_model_id", "terrain_delta_h"],
        metric_columns=metrics,
        confidence_level=0.95,
    )
    result.insert(result.columns.get_loc("mean"), "degrees_of_freedom", result["n"] - 1)
    return result


def build_parent_change_summary(route_changes: pd.DataFrame) -> pd.DataFrame:
    """Summarize five seeds once, after proving model-invariant routing evidence."""
    fields = (
        "parent_change_count",
        "parent_change_ratio",
        "full_route_change_count",
        "full_route_change_ratio",
    )
    for keys, group in route_changes.groupby(["terrain_delta_h", "seed"]):
        if any(group[field].nunique(dropna=False) != 1 for field in fields):
            raise ValueError(f"Routing changes differ across power models for {keys}")
    unique = route_changes.drop_duplicates(["terrain_delta_h", "seed"])
    summary = aggregate_paper_grade(
        unique,
        group_columns=["terrain_delta_h"],
        metric_columns=fields,
        confidence_level=0.95,
    )
    summary.insert(summary.columns.get_loc("mean"), "degrees_of_freedom", summary["n"] - 1)
    return summary


def _plot_metric(statistics: pd.DataFrame, metric: str, ylabel: str, output: Path) -> list[Path]:
    selected = statistics.loc[statistics["metric"] == metric]
    outputs = [output.with_suffix(".png"), output.with_suffix(".svg")]
    with plt.rc_context(FIGURE_STYLE):
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        for model_id in PLOT_MODEL_ORDER:
            frame = selected.loc[selected["model_id"] == model_id].sort_values("terrain_delta_h")
            if frame.empty:
                continue
            style = MODEL_STYLES[model_id]
            x = frame["terrain_delta_h"].to_numpy(dtype=float)
            mean = frame["mean"].to_numpy(dtype=float)
            lower = frame["ci_lower"].to_numpy(dtype=float)
            upper = frame["ci_upper"].to_numpy(dtype=float)
            axis.plot(
                x,
                mean,
                label=MODEL_LABELS[model_id],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=7,
            )
            axis.fill_between(x, lower, upper, color=style["color"], alpha=0.06)
        axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
        axis.set_xticks(list(REVIEWER_R1_5_TERRAINS))
        axis.set_xlabel(r"Terrain irregularity, $\Delta h$ (m)")
        axis.set_ylabel(ylabel)
        axis.grid(True)
        axis.legend(ncols=2)
        figure.tight_layout()
        figure.savefig(outputs[0], dpi=300, bbox_inches="tight", facecolor="white")
        figure.savefig(outputs[1], bbox_inches="tight", facecolor="white")
        plt.close(figure)
    return outputs


def _plot_route_changes(summary: pd.DataFrame, output: Path) -> list[Path]:
    outputs = [output.with_suffix(".png"), output.with_suffix(".svg")]
    styles = {
        "parent_change_ratio": ("Parent changed", "tab:blue", "o", "-"),
        "full_route_change_ratio": ("Full route changed", "tab:orange", "s", "--"),
    }
    with plt.rc_context(FIGURE_STYLE):
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        for metric, (label, color, marker, linestyle) in styles.items():
            frame = summary.loc[summary["metric"] == metric].sort_values("terrain_delta_h")
            x = frame["terrain_delta_h"].to_numpy(dtype=float)
            mean = 100.0 * frame["mean"].to_numpy(dtype=float)
            lower = 100.0 * frame["ci_lower"].to_numpy(dtype=float)
            upper = 100.0 * frame["ci_upper"].to_numpy(dtype=float)
            axis.plot(
                x,
                mean,
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=7,
            )
            axis.fill_between(x, lower, upper, color=color, alpha=0.06)
        axis.set_xticks(list(REVIEWER_R1_5_TERRAINS))
        axis.set_xlabel(r"Terrain irregularity, $\Delta h$ (m)")
        axis.set_ylabel("Changed end-device routes (%)")
        axis.grid(True)
        axis.legend()
        figure.tight_layout()
        figure.savefig(outputs[0], dpi=300, bbox_inches="tight", facecolor="white")
        figure.savefig(outputs[1], bbox_inches="tight", facecolor="white")
        plt.close(figure)
    return outputs


def _effect_text(row: pd.Series, subject: str) -> str:
    mean = float(row["mean"])
    lower = float(row["ci_lower"])
    upper = float(row["ci_upper"])
    if lower > 0:
        verdict = "lebih tinggi secara statistik"
    elif upper < 0:
        verdict = "lebih rendah secara statistik"
    else:
        verdict = "tidak berbeda secara statistik"
    return f"{subject}: mean {mean:.3f}, 95% CI [{lower:.3f}, {upper:.3f}], {verdict}."


def _interpretation_report(
    statistics: pd.DataFrame,
    parent_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Laporan Reviewer R1.5 — Sensitivitas Routing",
        "",
        "Selisih selalu didefinisikan sebagai TERRAIN_SPECIFIC − FIXED_REFERENCE. "
        "Setiap estimasi menggunakan lima seed berpasangan dan Student-t 95% CI "
        "dengan df=4.",
        "",
        "## Perubahan routing",
        "",
    ]
    for terrain in REVIEWER_R1_5_TERRAINS:
        p = parent_summary.loc[
            (parent_summary["terrain_delta_h"] == terrain)
            & (parent_summary["metric"] == "parent_change_ratio")
        ].iloc[0]
        r = parent_summary.loc[
            (parent_summary["terrain_delta_h"] == terrain)
            & (parent_summary["metric"] == "full_route_change_ratio")
        ].iloc[0]
        lines.append(
            f"- Δh={terrain} m: parent berubah {100 * float(p['mean']):.2f}% "
            f"dan full-route berubah {100 * float(r['mean']):.2f}% rata-rata."
        )
    lines.extend(["", "## Dampak per model", ""])
    for model_id in MAIN_W4_MODELS:
        lines.append(f"### {MODEL_LABELS[model_id]}")
        lines.append("")
        for terrain in REVIEWER_R1_5_TERRAINS:
            selected = statistics.loc[
                (statistics["model_id"] == model_id) & (statistics["terrain_delta_h"] == terrain)
            ]
            pdr = selected.loc[selected["metric"] == "pdr_difference_pp"].iloc[0]
            energy = selected.loc[selected["metric"] == "total_energy_difference_J"].iloc[0]
            delay = selected.loc[selected["metric"] == "avg_e2e_delay_difference_ms"].iloc[0]
            lines.append(f"- Δh={terrain} m:")
            lines.append(f"  - {_effect_text(pdr, 'PDR (percentage points)')}")
            lines.append(f"  - {_effect_text(energy, 'Total energy (J)')}")
            lines.append(f"  - {_effect_text(delay, 'Delay delivered-packet (ms)')}")
        lines.append("")
    lines.extend(
        [
            "## Batas interpretasi",
            "",
            "- Delay hanya dihitung dari paket delivered dan harus dibaca bersama PDR.",
            "- Efek routing dapat berinteraksi dengan power-control dan kondisi link; "
            "seluruh perubahan performa tidak boleh diatribusikan pada routing secara "
            "terpisah dari interaksi tersebut.",
            "- Unit inferensi adalah seed berpasangan (n=5), bukan paket individual.",
        ]
    )
    return "\n".join(lines) + "\n"


def _input_fingerprint(paths: Sequence[Path], roots: Sequence[Path]) -> tuple[str, int]:
    unique = sorted({path.resolve() for path in paths}, key=str)
    records = []
    for path in unique:
        owner = next((root for root in roots if _is_relative_to(path, root)), None)
        if owner is None:
            raise ValueError(f"Input file outside declared roots: {path}")
        records.append(
            {
                "root": str(owner),
                "relative_path": str(path.relative_to(owner)),
                "sha256": file_sha256(path),
            }
        )
    return content_hash(records, length=64), len(records)


def build_reviewer_r1_5_report(
    *,
    fixed_root: str | Path,
    terrain_specific_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    fixed_root = Path(fixed_root).resolve()
    terrain_root = Path(terrain_specific_root).resolve()
    output = Path(output_dir).resolve()
    if not fixed_root.is_dir() or not terrain_root.is_dir():
        raise FileNotFoundError("Both production input roots must exist")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    if _is_relative_to(output, fixed_root) or _is_relative_to(output, terrain_root):
        raise ValueError("Reviewer output must be outside both production inputs")

    _, _, fixed_parent_paths = _validate_parent_experiment(
        fixed_root,
        expected_run_count=375,
        expected_models=MAIN_W4_MODELS,
    )
    _, _, terrain_parent_paths = _validate_parent_experiment(
        terrain_root,
        expected_run_count=75,
        expected_models=MAIN_W4_MODELS,
    )
    fixed, fixed_validation, fixed_topology, fixed_paths = _collect_runs(
        fixed_root,
        strategy=FIXED_REFERENCE,
        selected_terrains=REVIEWER_R1_5_TERRAINS,
    )
    terrain, terrain_validation, terrain_topology, terrain_paths = _collect_runs(
        terrain_root,
        strategy=TERRAIN_SPECIFIC,
        selected_terrains=REVIEWER_R1_5_TERRAINS,
    )
    runs = pd.concat([fixed, terrain], ignore_index=True)
    run_validation = pd.concat([fixed_validation, terrain_validation], ignore_index=True)
    expected_pairs = (
        len(MAIN_W4_MODELS)
        * len(REVIEWER_R1_5_TERRAINS)
        * len(APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS)
    )
    if len(fixed) != expected_pairs or len(terrain) != expected_pairs:
        raise ValueError("Expected exactly 75 selected runs from each strategy")
    if not run_validation["run_validation_pass"].all():
        raise ValueError("One or more source runs failed reporter validation")

    validation = build_pair_validation(runs)
    if len(validation) != expected_pairs or not validation["validation_pass"].all():
        failures = validation.loc[
            ~validation["validation_pass"], list(PAIR_KEYS) + ["failed_checks"]
        ]
        raise ValueError(f"Paired input validation failed: {failures.to_dict('records')}")

    topology = pd.concat([fixed_topology, terrain_topology], ignore_index=True)
    route_changes = build_route_change_table(topology)
    paired = build_paired_routing_per_run(runs, route_changes)
    if len(paired) != expected_pairs:
        raise ValueError("Expected exactly 75 paired routing rows")
    statistics = summarize_pairs(paired)
    parent_summary = build_parent_change_summary(route_changes)
    interactions = build_model_routing_interactions(paired)
    interaction_statistics = summarize_model_routing_interactions(interactions)

    output.mkdir(parents=True, exist_ok=True)
    table_dir = output / "tables"
    figure_dir = output / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "paired_routing_per_run.csv": paired,
        "paired_routing_statistics.csv": statistics,
        "parent_change_summary.csv": parent_summary,
        "validation_summary.csv": validation,
        "model_routing_interaction_per_run.csv": interactions,
        "model_routing_interaction_statistics.csv": interaction_statistics,
    }
    for name, frame in tables.items():
        atomic_write_csv(table_dir / name, frame)

    figures: list[Path] = []
    for metric, ylabel, stem in FIGURE_METRICS:
        figures.extend(_plot_metric(statistics, metric, ylabel, figure_dir / stem))
    figures.extend(
        _plot_route_changes(
            parent_summary,
            figure_dir / "reviewer_r1_5_figure_routing_changes",
        )
    )
    report_path = output / "laporan_interpretasi_r1_5.md"
    report_path.write_text(_interpretation_report(statistics, parent_summary), encoding="utf-8")

    all_input_paths = fixed_parent_paths + terrain_parent_paths + fixed_paths + terrain_paths
    source_fingerprint, source_file_count = _input_fingerprint(
        all_input_paths, [fixed_root, terrain_root]
    )
    output_paths = [table_dir / name for name in tables] + figures + [report_path]
    manifest = {
        "manifest_version": REPORTER_VERSION,
        "status": "PASS",
        "created_at": utc_now(),
        "analysis_scope": "reviewer_r1_5_paired_routing_only",
        "input_mode": "read_only_existing_simulation_artifacts",
        "fixed_reference_root": str(fixed_root),
        "terrain_specific_root": str(terrain_root),
        "difference_direction": DIFFERENCE_DIRECTION,
        "model_ids": list(MAIN_W4_MODELS),
        "terrains": list(REVIEWER_R1_5_TERRAINS),
        "seeds": list(APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS),
        "pair_count": len(paired),
        "validation": {
            "selected_run_count": len(runs),
            "selected_runs_pass": bool(run_validation["run_validation_pass"].all()),
            "paired_validation_count": len(validation),
            "paired_validation_pass": bool(validation["validation_pass"].all()),
        },
        "statistics": {
            "unit_of_inference": "paired_seed",
            "n_per_model_terrain": 5,
            "sample_standard_deviation_ddof": 1,
            "confidence_level": 0.95,
            "ci_method": "paired Student-t",
            "degrees_of_freedom": 4,
        },
        "metric_definitions": {
            "pdr_percent": "100 * delivered / offered",
            "energy_per_delivered_packet_J": (
                "corrected total network energy / delivered packets per run"
            ),
            "avg_e2e_delay_ms": "mean over delivered packets only",
            "parent_change_ratio": "changed end-device parents / end devices",
            "full_route_change_ratio": "changed end-device full routes / end devices",
        },
        "offered_traffic_hash_version": OFFERED_TRAFFIC_HASH_VERSION,
        "source_input_file_count": source_file_count,
        "source_input_fingerprint_sha256": source_fingerprint,
        "reporter_source": str(Path(__file__).resolve()),
        "reporter_source_sha256": file_sha256(Path(__file__)),
        "outputs": [{"path": str(path), "sha256": file_sha256(path)} for path in output_paths],
    }
    manifest_path = output / "analysis_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {
        "status": "PASS",
        "pair_count": len(paired),
        "output_dir": str(output),
        "tables": [str(table_dir / name) for name in tables],
        "figures": [str(path) for path in figures],
        "analysis_manifest": str(manifest_path),
        "interpretation_report": str(report_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-root", required=True, type=Path)
    parser.add_argument("--terrain-specific-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_reviewer_r1_5_report(
        fixed_root=args.fixed_root,
        terrain_specific_root=args.terrain_specific_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DIFFERENCE_DIRECTION",
    "REPORTER_VERSION",
    "STATISTIC_METRICS",
    "build_model_routing_interactions",
    "build_pair_validation",
    "build_parent_change_summary",
    "build_paired_routing_per_run",
    "build_reviewer_r1_5_report",
    "physical_configuration_signature",
    "summarize_model_routing_interactions",
    "summarize_pairs",
]
