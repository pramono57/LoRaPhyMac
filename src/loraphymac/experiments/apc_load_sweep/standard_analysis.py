"""Deterministic postprocessing for one Sweep Standard APC Load cohort."""

from __future__ import annotations

import json
import argparse
import itertools
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from loraphymac.experiments.common.multiseed_pipeline import (
    _read_verified_scientific_tables,
    aggregate_paper_grade,
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
)
from loraphymac.experiments.common.analysis_publication import AnalysisPublisher


ANALYSIS_VERSION = "apc_load_sweep_analysis_v1"
METRICS = (
    "pdr_percent",
    "energy_per_delivered_packet_J",
    "total_network_energy_J",
    "avg_e2e_delay_ms",
)


def build_load_sweep_analysis(cohort_dir: str | Path) -> dict[str, Any]:
    """Build cohort-owned summaries and figures; source runs remain immutable."""
    root = Path(cohort_dir).expanduser().resolve()
    manifest_path = root / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output_layout_version") != "sweep_cohort_v1":
        raise ValueError("Load analysis requires a Sweep Standard cohort")
    sweep_profile = str(
        manifest.get("sweep_profile")
        or manifest.get("spec", {}).get("run_annotations", {}).get("sweep_profile", "production")
    )
    if sweep_profile not in {"production", "pilot"}:
        raise ValueError(f"Unsupported Load sweep profile in cohort manifest: {sweep_profile}")
    expected_experiment_profile = (
        "apc_load_sweep_pilot_v1" if sweep_profile == "pilot"
        else "apc_load_sweep_standard_v1"
    )
    if manifest.get("spec", {}).get("base_config", {}).get("experiment_profile") not in {
        None, expected_experiment_profile
    }:
        raise ValueError("Selected sweep profile conflicts with the experiment configuration")
    status = json.loads((root / "experiment_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE" or status.get("seed_grid_complete") is not True:
        raise ValueError("Load analysis requires completed runs and aggregation")

    core_path = root / "aggregate" / "all_runs_core_summary.csv"
    if not core_path.is_file():
        raise ValueError(f"Missing all-run core summary: {core_path}")
    core = pd.read_csv(core_path)
    groups = ["apc_mode", "node_num", "node_period"]
    required = {*groups, "run_id", "seed", "node_ptx_mode", *METRICS}
    if not required.issubset(core.columns):
        raise ValueError(f"Core summary lacks columns: {sorted(required - set(core.columns))}")
    expected_run_ids = manifest.get("run_ids")
    if not isinstance(expected_run_ids, list) or not expected_run_ids:
        raise ValueError("Cohort manifest must contain a non-empty run_ids list")
    expected_run_ids = [str(value) for value in expected_run_ids]
    if any(not value for value in expected_run_ids) or len(set(expected_run_ids)) != len(
        expected_run_ids
    ):
        raise ValueError("Cohort manifest contains empty or duplicate run IDs")
    expected_count = manifest.get("expected_run_count")
    if expected_count is not None and int(expected_count) != len(expected_run_ids):
        raise ValueError("Cohort expected_run_count conflicts with its run_ids")
    if sweep_profile == "pilot" and len(expected_run_ids) != 2:
        raise ValueError("Pilot analysis requires exactly two distinct run IDs")
    if sweep_profile == "production" and len(expected_run_ids) != 120:
        raise ValueError("Production Load analysis requires exactly 120 run IDs")
    run_root = root / "runs"
    actual_run_dirs = (
        {
            path.name
            for path in run_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        if run_root.is_dir()
        else set()
    )
    if actual_run_dirs != set(expected_run_ids):
        raise ValueError(
            "Run directory IDs do not match cohort manifest; "
            f"missing={sorted(set(expected_run_ids) - actual_run_dirs)[:10]}, "
            f"unexpected={sorted(actual_run_dirs - set(expected_run_ids))[:10]}"
        )
    if core["run_id"].isna().any() or core["run_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Core summary contains an empty run_id")
    core["run_id"] = core["run_id"].astype(str)
    if core["run_id"].duplicated().any():
        duplicate_ids = sorted(core.loc[core["run_id"].duplicated(), "run_id"].unique())
        raise ValueError(f"Core summary contains duplicate run IDs: {duplicate_ids[:10]}")
    expected_ids = set(expected_run_ids)
    actual_ids = set(core["run_id"])
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"Core run IDs do not match cohort manifest; missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )

    grid = manifest.get("parameter_grid", {})
    factors = grid.get("factors")
    seeds = grid.get("seeds")
    if not isinstance(factors, dict) or not factors or not isinstance(seeds, list) or not seeds:
        raise ValueError("Cohort manifest lacks configured factors or seeds")
    expected_cells = {
        (str(mode), int(nodes), int(period), int(seed))
        for mode, nodes, period, seed in itertools.product(
            factors.get("node_ptx_mode", []),
            factors.get("node_num", []),
            factors.get("node_period", []),
            seeds,
        )
    }
    expected_factors = (
        {"node_num": [50], "node_period": [90], "node_ptx_mode": ["adaptive", "fixed"]}
        if sweep_profile == "pilot"
        else {
            "node_num": [50, 100, 150, 200],
            "node_period": [30, 60, 90],
            "node_ptx_mode": ["adaptive", "fixed"],
        }
    )
    expected_seeds = [42] if sweep_profile == "pilot" else [30, 36, 42, 48, 56]
    if {key: list(values) for key, values in factors.items()} != expected_factors or list(seeds) != expected_seeds:
        raise ValueError(f"{sweep_profile.title()} Load analysis grid does not match its profile contract")
    if not expected_cells or len(expected_cells) != len(expected_run_ids):
        raise ValueError(
            "Configured load grid does not have a one-to-one mapping to manifest run IDs"
        )
    if sweep_profile == "pilot" and expected_cells != {
        ("adaptive", 50, 90, 42), ("fixed", 50, 90, 42)
    }:
        raise ValueError("Pilot Load analysis requires AA and FF at 50 nodes, 90 seconds, seed 42")
    id_cells: dict[str, tuple[str, int, int, int]] = {}
    for run_id in expected_run_ids:
        run_dir = root / "runs" / run_id
        config_path = run_dir / "resolved_config.json"
        if not config_path.is_file():
            raise ValueError(f"Missing resolved configuration for expected run {run_id}")
        resolved = json.loads(config_path.read_text(encoding="utf-8"))
        scientific = resolved.get("scientific_config", {})
        run_factors = resolved.get("factors", {})
        if resolved.get("run_id") != run_id:
            raise ValueError(f"Resolved configuration run_id mismatch for {run_id}")
        node_mode = run_factors.get("node_ptx_mode")
        relay_mode = scientific.get("relay_ptx_mode")
        expected_pair = (
            "adaptive" if node_mode == "adaptive" else "fixed" if node_mode == "fixed" else None
        )
        if expected_pair is None or relay_mode != expected_pair:
            raise ValueError(f"Invalid AA/FF mode pairing for run {run_id}")
        cell = (
            str(node_mode),
            int(run_factors["node_num"]),
            int(run_factors["node_period"]),
            int(resolved["seed"]),
        )
        if cell not in expected_cells:
            raise ValueError(f"Resolved configuration is outside the configured grid: {run_id}")
        if cell in id_cells.values():
            raise ValueError(f"Duplicate configured condition/seed for run {run_id}")
        id_cells[run_id] = cell
    if set(id_cells.values()) != expected_cells:
        raise ValueError("Resolved run configurations do not cover the configured grid exactly")

    expected_mode = core["node_ptx_mode"].map({"adaptive": "AA", "fixed": "FF"})
    if expected_mode is None or not expected_mode.eq(core["apc_mode"]).all():
        raise ValueError("Core APC mode conflicts with configured node/relay modes")
    core_cells = []
    for row in core.itertuples(index=False):
        run_id = str(row.run_id)
        cell = id_cells[run_id]
        observed = (cell[0], int(row.node_num), int(row.node_period), int(row.seed))
        if observed != cell:
            raise ValueError(f"Core condition/seed conflicts with resolved config for {run_id}")
        if str(row.node_ptx_mode) != cell[0]:
            raise ValueError(f"Core node mode conflicts with resolved config for {run_id}")
        core_cells.append(
            (str(row.node_ptx_mode), int(row.node_num), int(row.node_period), int(row.seed))
        )
    if len(set(core_cells)) != len(expected_cells):
        raise ValueError("Core summary contains duplicate condition/seed combinations")

    for metric in METRICS:
        raw_values = core[metric]
        values = pd.to_numeric(core[metric], errors="coerce")
        if (raw_values.notna() & values.isna()).any():
            raise ValueError(f"Metric {metric} contains non-numeric values")
        core[metric] = values
        if metric == "energy_per_delivered_packet_J":
            invalid = (~values.map(math.isfinite) & ~values.isna()) | values.lt(0)
            invalid |= core["pdr_percent"].gt(0) & values.isna()
        elif metric == "avg_e2e_delay_ms":
            invalid = (~values.map(math.isfinite) & ~values.isna()) | values.lt(0)
            invalid |= core["pdr_percent"].gt(0) & values.isna()
        else:
            invalid = ~values.map(math.isfinite) | values.lt(0)
            if metric == "pdr_percent":
                invalid |= values.gt(100)
        if invalid.any():
            bad_ids = core.loc[invalid, "run_id"].astype(str).tolist()
            raise ValueError(f"Metric {metric} has invalid values for runs: {bad_ids[:10]}")
    summary = aggregate_paper_grade(
        core,
        group_columns=groups,
        metric_columns=METRICS,
        confidence_level=0.95,
    )
    # Preserve the exact run-level diagnostic inputs used by future analyses.
    run_dirs = [root / "runs" / run_id for run_id in expected_run_ids]
    for run_dir in run_dirs:
        _read_verified_scientific_tables(run_dir)
    inputs = [
        {
            "path": core_path.relative_to(root).as_posix(),
            "size_bytes": core_path.stat().st_size,
            "sha256": file_sha256(core_path),
        }
    ] + [
        {
            "path": f"runs/{run_id}/scientific_output_manifest.json",
            "size_bytes": (root / "runs" / run_id / "scientific_output_manifest.json")
            .stat()
            .st_size,
            "sha256": file_sha256(root / "runs" / run_id / "scientific_output_manifest.json"),
        }
        for run_id in expected_run_ids
    ]
    generator = Path(__file__).resolve()
    analysis = {
        "analysis_id": f"{manifest['cohort_id']}-analysis-v1",
        "analysis_version": ANALYSIS_VERSION,
        "sweep_profile": sweep_profile,
        "evidence_class": "non-production integration evidence" if sweep_profile == "pilot" else "production experiment",
        "is_non_production_pilot": sweep_profile == "pilot",
        "source_cohort_id": manifest["cohort_id"],
        "status": "COMPLETE",
        "inputs": inputs,
        "generator": {
            "path": generator.relative_to(Path(__file__).resolve().parents[4]).as_posix(),
            "size_bytes": generator.stat().st_size,
            "sha256": file_sha256(generator),
        },
        "metrics": {
            "pdr_percent": "per-run delivered packets / offered packets * 100",
            "energy_per_delivered_packet_J": "per-run corrected network energy / delivered packets",
            "total_network_energy_J": "per-run sum of corrected network energy",
            "avg_e2e_delay_ms": "mean end-to-end delay among delivered packets",
        },
        "grouping_dimensions": groups,
        "seed_aggregation": {
            "n": "number of seed-level run observations",
            "mean": "arithmetic mean across seeds",
            "std": "sample standard deviation (ddof=1)",
            "ci": "two-sided Student-t 95% confidence interval, df=n-1",
        },
        "omitted_statistics": (
            ["sample standard deviation and 95% Student-t confidence intervals are unavailable for n=1 seed per condition"]
            if sweep_profile == "pilot"
            else []
        ),
        "included_model_variants": ["AA", "FF"],
    }
    publisher = AnalysisPublisher(root)
    try:
        reused = publisher.find_reusable(analysis)
        if reused is not None:
            existing_manifest = manifest.get("analysis_manifest_path")
            result_path = Path(reused["analysis_manifest"]).relative_to(root).as_posix()
            if manifest.get("analysis_status") != "COMPLETE" or existing_manifest != result_path:
                manifest["analysis_status"] = "COMPLETE"
                manifest["analysis_manifest_path"] = result_path
                atomic_write_json(manifest_path, manifest)
            return reused
        tables_dir = publisher.tables_dir
        figures_dir = publisher.figures_dir
        output_paths = [tables_dir / "load_performance_summary.csv"]
        atomic_write_csv(output_paths[0], summary)
        for source_name, output_name in (
            ("node_tx_duty_cycle.csv", "load_node_duty_cycle_per_run.csv"),
            ("channel_occupancy.csv", "load_channel_occupancy_per_run.csv"),
            ("apc_feedback_lifecycle.csv", "load_feedback_lifecycle_per_run.csv"),
        ):
            frames = []
            for run_dir in run_dirs:
                source = run_dir / "tables" / source_name
                if not source.is_file():
                    raise ValueError(f"Missing required per-run analysis table: {source}")
                frames.append(pd.read_csv(source))
            target = tables_dir / output_name
            atomic_write_csv(target, pd.concat(frames, ignore_index=True, sort=False))
            output_paths.append(target)

        figure_paths: list[Path] = []
        for metric, ylabel, stem in (
            ("pdr_percent", "Packet delivery ratio (%)", "pdr_vs_nodes_by_period_mode"),
            (
                "energy_per_delivered_packet_J",
                "Energy per delivered packet (J/packet)",
                "energy_per_delivered_packet_vs_nodes",
            ),
        ):
            selected = summary.loc[summary["metric"].eq(metric)]
            fig, axis = plt.subplots(figsize=(7.0, 4.5))
            for (mode, period), part in selected.groupby(
                ["apc_mode", "node_period"], sort=True, dropna=False
            ):
                part = part.sort_values("node_num", kind="stable")
                if part["n"].ge(2).all():
                    axis.errorbar(
                        part["node_num"],
                        part["mean"],
                        yerr=[
                            part["mean"] - part["ci_lower"],
                            part["ci_upper"] - part["mean"],
                        ],
                        marker="o",
                        linestyle="-" if mode == "AA" else "--",
                        capsize=2,
                        label=f"{mode}, {period} s",
                    )
                else:
                    axis.plot(
                        part["node_num"], part["mean"],
                        marker="o", linestyle="-" if mode == "AA" else "--",
                        label=f"{mode}, {period} s (CI unavailable, n=1)",
                    )
            axis.set_xlabel("Number of nodes")
            axis.set_ylabel(ylabel)
            axis.legend(frameon=False, ncol=2)
            fig.tight_layout()
            target = figures_dir / f"{stem}.png"
            fig.savefig(target, dpi=300, metadata={"Software": "LoRaPhyMAC Sweep Standard v1"})
            plt.close(fig)
            figure_paths.append(target)

        all_outputs = [*output_paths, *figure_paths]
        analysis["tables"] = [path.relative_to(root).as_posix() for path in output_paths]
        analysis["figures"] = [path.relative_to(root).as_posix() for path in figure_paths]
        analysis["outputs"] = [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in all_outputs
        ]
        result = publisher.publish(analysis, all_outputs)
        result_path = Path(result["analysis_manifest"]).relative_to(root).as_posix()
        if (
            manifest.get("analysis_status") != "COMPLETE"
            or manifest.get("analysis_manifest_path") != result_path
        ):
            manifest["analysis_status"] = "COMPLETE"
            manifest["analysis_manifest_path"] = result_path
            atomic_write_json(manifest_path, manifest)
        return result
    finally:
        publisher.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort_dir", type=Path)
    print(json.dumps(build_load_sweep_analysis(parser.parse_args(argv).cohort_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
