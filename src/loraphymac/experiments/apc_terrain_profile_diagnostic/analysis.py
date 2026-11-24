"""Immutable analysis publication for the paired ten-run terrain diagnostic."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from loraphymac.experiments.apc_terrain_profile_diagnostic.contract import (
    APC_TERRAIN_PROFILE_DIAGNOSTIC_PROFILE,
    APC_TERRAIN_PROFILE_DIAGNOSTIC_SEEDS,
    APC_TERRAIN_PROFILE_DIAGNOSTIC_TERRAIN,
    diagnostic_config_violations,
)
from loraphymac.experiments.common.analysis_publication import AnalysisPublisher
from loraphymac.experiments.common.multiseed_pipeline import (
    _read_verified_scientific_tables,
    aggregate_paper_grade,
    atomic_write_json,
    file_sha256,
)
from loraphymac.experiments.common.outputs import ExperimentArtifactPaths
from loraphymac.experiments.apc_terrain_sweep.postprocessing import (
    _read_json,
    _validate_completed_sweep,
)


ANALYSIS_VERSION = "apc_terrain_profile_diagnostic_analysis_v1"
METRIC_UNITS = {
    "delivery_by_horizon_percent": "%",
    "terminal_loss_percent": "%",
    "censored_fraction_percent": "%",
    "served_node_ratio_95_percent": "%",
    "avg_e2e_delay_ms": "ms",
    "total_node_energy_J": "J",
    "total_relay_energy_J": "J",
    "total_gateway_energy_J": "J",
    "total_network_energy_J": "J",
    "total_constrained_energy_J": "J",
    "energy_per_delivered_packet_J": "J/packet",
    "energy_per_delivered_packet_constrained_J": "J/packet",
    "avg_final_node_ptx": "dBm",
    "avg_final_relay_ptx": "dBm",
    "avg_final_node_decision_margin": "dB",
    "avg_final_node_apc_rx_margin": "dB",
    "avg_final_relay_decision_margin": "dB",
    "avg_final_relay_apc_rx_margin": "dB",
    "apc_total_tx": "count",
    "apc_total_rx": "count",
    "apc_total_drop": "count",
    "apc_apply_success_percent": "%",
    "apc_down_node_success_percent": "%",
    "apc_down_relay_success_percent": "%",
    "ptx_update_count": "count",
    "packet_fate_conservation_delta": "count",
    "collision_outcome_conservation_delta": "count",
}


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _load_runs(paths: ExperimentArtifactPaths, manifest: dict[str, Any]) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    inputs.append(_record(paths.status, paths.experiment_dir))
    run_ids = manifest["run_ids"]
    for run_id in run_ids:
        run_dir = paths.runs / run_id
        resolved_path = run_dir / "resolved_config.json"
        status_path = run_dir / "status.json"
        scientific_manifest = run_dir / "scientific_output_manifest.json"
        verified = _read_verified_scientific_tables(run_dir)
        core_record = verified.get("core_summary_df")
        core_path = core_record[1] if core_record else run_dir / "core_summary.csv"
        if not core_path.is_file():
            raise ValueError(f"Missing per-run core summary for {run_id}")
        resolved = _read_json(resolved_path, "resolved run configuration")
        violations = diagnostic_config_violations(resolved.get("scientific_config", {}))
        if violations:
            raise ValueError(f"Invalid diagnostic scientific config for {run_id}: {violations}")
        frame = pd.read_csv(core_path)
        if len(frame) != 1:
            raise ValueError(f"Per-run core summary must have exactly one row: {run_id}")
        row = frame.iloc[0].to_dict()
        row.update(
            {
                "run_id": run_id,
                "seed": int(resolved["seed"]),
                "node_ptx_mode": resolved["factors"]["node_ptx_mode"],
                "relay_ptx_mode": resolved["factors"]["relay_ptx_mode"],
            }
        )
        if row["node_ptx_mode"] != row["relay_ptx_mode"]:
            raise ValueError(f"Expected paired FF/AA run, got mixed mode: {run_id}")
        rows.append(row)
        for path in (resolved_path, status_path, scientific_manifest, core_path):
            inputs.append(_record(path, paths.experiment_dir))
    core = pd.DataFrame(rows)
    if core["run_id"].duplicated().any():
        raise ValueError("Duplicate per-run core summary IDs")
    if set(core["seed"].astype(int)) != set(APC_TERRAIN_PROFILE_DIAGNOSTIC_SEEDS):
        raise ValueError("Diagnostic analysis does not contain the five declared seeds")
    if len(core) != 2 * len(APC_TERRAIN_PROFILE_DIAGNOSTIC_SEEDS):
        raise ValueError("Diagnostic analysis requires exactly ten run summaries")
    return core, inputs


def _paired_rows(core: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    metrics = [
        name for name in METRIC_UNITS
        if name in core.columns and pd.api.types.is_numeric_dtype(core[name])
    ]
    if not metrics:
        raise ValueError("No contracted diagnostic metrics are available in core summaries")
    for metric in metrics:
        raw = core[metric]
        values = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & values.isna()).any():
            raise ValueError(f"Metric {metric} contains non-numeric values")
        invalid = values.notna() & ~values.map(math.isfinite)
        if METRIC_UNITS[metric] == "%":
            invalid |= values.notna() & ~values.between(0.0, 100.0)
        elif METRIC_UNITS[metric] in {"ms", "J", "J/packet", "count"}:
            invalid |= values.notna() & values.lt(0.0)
        if metric in {"packet_fate_conservation_delta", "collision_outcome_conservation_delta"}:
            invalid |= values.notna() & values.ne(0.0)
        if invalid.any():
            raise ValueError(f"Metric {metric} violates its finite/unit/invariant bounds")
    records = []
    for seed in APC_TERRAIN_PROFILE_DIAGNOSTIC_SEEDS:
        by_mode = core.loc[core["seed"].astype(int) == seed].set_index("node_ptx_mode")
        if set(by_mode.index) != {"fixed", "adaptive"} or len(by_mode) != 2:
            raise ValueError(f"Seed {seed} does not contain exactly one FF and one AA run")
        ff = by_mode.loc["fixed"]
        aa = by_mode.loc["adaptive"]
        for metric in metrics:
            ff_value = pd.to_numeric(pd.Series([ff[metric]]), errors="coerce").iloc[0]
            aa_value = pd.to_numeric(pd.Series([aa[metric]]), errors="coerce").iloc[0]
            if pd.isna(ff_value) or pd.isna(aa_value):
                continue
            records.append(
                {
                    "seed": int(seed),
                    "metric": metric,
                    "unit": METRIC_UNITS[metric],
                    "ff_value": float(ff_value),
                    "aa_value": float(aa_value),
                    "aa_minus_ff": float(aa_value - ff_value),
                    "ff_run_id": str(ff["run_id"]),
                    "aa_run_id": str(aa["run_id"]),
                }
            )
    paired = pd.DataFrame(records)
    if paired.empty:
        raise ValueError("No finite paired metric observations are available")
    summaries = []
    for metric, subset in paired.groupby("metric", sort=True):
        stats = aggregate_paper_grade(subset[["aa_minus_ff"]], [], ["aa_minus_ff"])
        summaries.append({"metric_name": metric, "unit": METRIC_UNITS[metric], **stats.iloc[0].to_dict()})
    return paired, pd.DataFrame(summaries), metrics


def build_diagnostic_analysis(*, experiment_dir: str | Path) -> dict[str, Any]:
    """Validate a complete ten-run diagnostic and publish immutable analysis outputs."""
    paths = ExperimentArtifactPaths.for_experiment(Path(experiment_dir).expanduser().resolve())
    if not paths.is_sweep_standard:
        raise ValueError("Terrain diagnostic analysis requires sweep_cohort_v1 layout")
    _validate_completed_sweep(paths)
    manifest_path = paths.manifest
    cohort = _read_json(manifest_path, "terrain diagnostic cohort manifest")
    if cohort.get("spec", {}).get("base_config", {}).get("experiment_profile") != APC_TERRAIN_PROFILE_DIAGNOSTIC_PROFILE:
        raise ValueError("Terrain diagnostic analysis received a different experiment profile")
    core, inputs = _load_runs(paths, cohort)
    aggregate_path = paths.aggregate / "all_runs_core_summary.csv"
    if not aggregate_path.is_file():
        raise ValueError(f"Missing cohort all-run aggregate: {aggregate_path}")
    aggregate = pd.read_csv(aggregate_path)
    if "run_id" not in aggregate or aggregate["run_id"].duplicated().any():
        raise ValueError("Cohort aggregate has missing or duplicate run IDs")
    if set(aggregate["run_id"].astype(str)) != set(cohort["run_ids"]):
        raise ValueError("Cohort aggregate run IDs do not match the ten-run manifest")
    for metric in METRIC_UNITS:
        if metric in core.columns and metric in aggregate.columns:
            lhs = core.set_index("run_id")[metric].sort_index()
            rhs = aggregate.assign(run_id=aggregate["run_id"].astype(str)).set_index("run_id")[metric].sort_index()
            pd.testing.assert_series_equal(lhs.astype(float), rhs.astype(float), check_names=False, check_dtype=False)
    inputs.append(_record(aggregate_path, paths.experiment_dir))
    paired, summary, metrics = _paired_rows(core)

    source_manifest = dict(cohort)
    source_manifest.pop("analysis_status", None)
    source_manifest.pop("analysis_manifest_path", None)
    source_manifest_hash = __import__("hashlib").sha256(
        json.dumps(source_manifest, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    generator = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "COMPLETE",
        "source_cohort_id": cohort["cohort_id"],
        "experiment_profile": APC_TERRAIN_PROFILE_DIAGNOSTIC_PROFILE,
        "evidence_class": "bounded diagnostic integration evidence; not production findings",
        "run_ids": list(cohort["run_ids"]),
        "run_matrix": cohort["parameter_grid"]["run_matrix"],
        "terrain_delta_h_m": APC_TERRAIN_PROFILE_DIAGNOSTIC_TERRAIN,
        "seeds": list(APC_TERRAIN_PROFILE_DIAGNOSTIC_SEEDS),
        "comparison": "AA minus FF, paired by seed",
        "metrics": {name: {"unit": METRIC_UNITS[name]} for name in metrics},
        "aggregation": {
            "paired_difference": "AA - FF within each seed",
            "summary": "mean, sample standard deviation (ddof=1), and two-sided Student-t 95% CI across available paired seed differences",
            "confidence_level": 0.95,
            "interpretation": "descriptive only; no significance claim",
        },
        "source_commit": cohort.get("git_head"),
        "source_fingerprint_sha256": cohort.get("source_fingerprint_sha256"),
        "source_experiment_manifest_content_sha256": source_manifest_hash,
        "generator": {
            "path": "src/loraphymac/experiments/apc_terrain_profile_diagnostic/analysis.py",
            "size_bytes": generator.stat().st_size,
            "sha256": file_sha256(generator),
        },
        "inputs": inputs,
        "omitted_statistics": [],
    }
    for row in summary.to_dict(orient="records"):
        if row.get("n", 0) < 2:
            manifest["omitted_statistics"].append(
                f"{row['metric_name']}: sample SD and 95% CI unavailable for n={row.get('n')}"
            )

    publisher = AnalysisPublisher(paths.experiment_dir)
    try:
        reusable = publisher.find_reusable(manifest)
        if reusable is not None:
            relative_manifest = Path(reusable["analysis_manifest"]).relative_to(paths.experiment_dir).as_posix()
            if cohort.get("analysis_status") != "COMPLETE" or cohort.get("analysis_manifest_path") != relative_manifest:
                cohort["analysis_status"] = "COMPLETE"
                cohort["analysis_manifest_path"] = relative_manifest
                atomic_write_json(manifest_path, cohort)
            return {"status": "COMPLETE", **reusable}

        paired_path = publisher.tables_dir / "terrain_diagnostic_paired_seed_differences.csv"
        summary_path = publisher.tables_dir / "terrain_diagnostic_paired_difference_summary.csv"
        paired.to_csv(paired_path, index=False, float_format="%.12g", lineterminator="\n")
        summary.to_csv(summary_path, index=False, float_format="%.12g", lineterminator="\n")
        figure_paths = []
        for metric in ("delivery_by_horizon_percent", "avg_e2e_delay_ms", "total_network_energy_J"):
            subset = paired.loc[paired["metric"] == metric]
            if subset.empty:
                continue
            figure, axis = plt.subplots(figsize=(7, 4))
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.scatter(subset["seed"], subset["aa_minus_ff"], color="#145da0", zorder=3)
            axis.plot(subset["seed"], subset["aa_minus_ff"], color="#145da0", alpha=0.55)
            axis.set_xlabel("Seed")
            axis.set_ylabel(f"AA − FF ({METRIC_UNITS[metric]})")
            axis.set_title(f"Terrain diagnostic paired difference: {metric}")
            axis.grid(axis="y", alpha=0.25)
            figure.tight_layout()
            target = publisher.figures_dir / f"terrain_diagnostic_{metric}_paired_difference.png"
            figure.savefig(target, dpi=160, metadata={"Software": "LoRaPhyMAC diagnostic analysis"})
            plt.close(figure)
            figure_paths.append(target)
        outputs = [paired_path, summary_path, *figure_paths]
        manifest["tables"] = [path.relative_to(paths.experiment_dir).as_posix() for path in (paired_path, summary_path)]
        manifest["figures"] = [path.relative_to(paths.experiment_dir).as_posix() for path in figure_paths]
        manifest["outputs"] = [
            {
                "path": path.relative_to(paths.experiment_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in outputs
        ]
        result = publisher.publish(manifest, outputs)
        relative_manifest = Path(result["analysis_manifest"]).relative_to(paths.experiment_dir).as_posix()
        cohort["analysis_status"] = "COMPLETE"
        cohort["analysis_manifest_path"] = relative_manifest
        atomic_write_json(manifest_path, cohort)
        return result
    except Exception as exc:
        if cohort.get("analysis_status") != "COMPLETE":
            cohort["analysis_status"] = "FAILED"
            cohort["analysis_error"] = f"{type(exc).__name__}: {exc}"
            atomic_write_json(manifest_path, cohort)
        raise
    finally:
        publisher.close()


__all__ = ["ANALYSIS_VERSION", "METRIC_UNITS", "build_diagnostic_analysis"]
