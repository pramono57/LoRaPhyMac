"""Reporting-only analysis for the isolated 120-run APC load sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from loraphymac.experiments.apc_load_sweep.contract import (
    APC_LOAD_SWEEP_FULL_FACTORIAL_BATCH,
    APC_LOAD_SWEEP_SEEDS,
)
from loraphymac.experiments.apc_load_sweep.postprocessing import (
    _plot_metric,
    _plot_relay_duty,
)
from loraphymac.experiments.common.multiseed_pipeline import atomic_write_csv, atomic_write_json
from loraphymac.experiments.common.outputs import resolve_master_table_path
from loraphymac.paths import repository_root


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = repository_root(EXPERIMENT_DIR)
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/experiments/apc_load_sweep/full_factorial_v1"
OUTPUT_DIR = ARTIFACT_ROOT / "analysis"
EXPERIMENT_IDS = {
    "ff": "apc_load_sweep_ff_full_factorial_v1",
    "aa": "apc_load_sweep_aa_full_factorial_v1",
}
PAIR_KEYS = ["node_num", "node_period", "seed"]
MODE_KEYS = {"node_ptx_mode", "relay_ptx_mode"}
SCHEDULE_COLUMNS = ["packet_uid", "src_id", "fcnt", "birth_time", "offered_time"]
METRICS = (
    "pdr_percent",
    "total_network_energy_J",
    "energy_per_delivered_packet_J",
    "avg_e2e_delay_ms",
    "throughput_bps",
)


class IncompleteFullFactorialError(ValueError):
    """Raised when the new APC load-sweep grid is incomplete or unpaired."""


def _energy_per_delivered_packet(total_energy: pd.Series, delivered_count: pd.Series) -> pd.Series:
    energy = pd.to_numeric(total_energy, errors="raise")
    delivered = pd.to_numeric(delivered_count, errors="raise")
    if energy.lt(0).any() or delivered.lt(0).any():
        raise ValueError("Energy and delivered-packet counts must be non-negative")
    return energy / delivered.replace(0, np.nan)


def expected_grid() -> set[tuple[str, int, int, int]]:
    return {
        (model, nodes, period, seed)
        for model in ("ff", "aa")
        for nodes in APC_LOAD_SWEEP_FULL_FACTORIAL_BATCH["node_num"]
        for period in APC_LOAD_SWEEP_FULL_FACTORIAL_BATCH["node_period"]
        for seed in APC_LOAD_SWEEP_SEEDS
    }


def validate_full_factorial_rows(frame: pd.DataFrame) -> None:
    keys = ["model_id", *PAIR_KEYS]
    required = {*keys, "run_id", "stage1_status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Core summary missing columns: {sorted(missing)}")
    if pd.to_numeric(frame["node_period"], errors="raise").eq(15).any():
        raise ValueError("15-second runs are prohibited in the full-factorial dataset")
    if frame.duplicated(keys).any() or frame["run_id"].duplicated().any():
        raise ValueError("Duplicate model/scenario/seed or run_id detected")
    actual = {
        (str(row.model_id), int(row.node_num), int(row.node_period), int(row.seed))
        for row in frame.itertuples()
    }
    expected = expected_grid()
    if actual != expected or len(frame) != 120:
        raise IncompleteFullFactorialError(
            f"Expected 120 unique runs; actual={len(frame)}, "
            f"missing={len(expected - actual)}, extra={len(actual - expected)}"
        )
    if not frame["stage1_status"].astype(str).eq("PASS").all():
        raise ValueError("All 120 run rows must have stage1_status=PASS")


def _read_schedule(run_dir: Path) -> tuple[pd.DataFrame, str]:
    path = run_dir / "tables/packet_fates.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Compact offered-packet fate table missing: {path}")
    schedule = pd.read_csv(path, usecols=SCHEDULE_COLUMNS).sort_values(
        ["src_id", "fcnt", "packet_uid"], kind="stable"
    ).reset_index(drop=True)
    digest = hashlib.sha256(schedule.to_csv(index=False).encode("utf-8")).hexdigest()
    return schedule, digest


def _read_run_metadata(experiment_id: str, run_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    run_dir = ARTIFACT_ROOT / experiment_id / "runs" / run_id
    status_path = run_dir / "status.json"
    config_path = run_dir / "resolved_config.json"
    if not status_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Run provenance incomplete: {run_dir}")
    status, config_record = json.loads(status_path.read_text()), json.loads(config_path.read_text())
    if status.get("status") != "PASS":
        raise ValueError(f"Run is not PASS: {run_id}")
    return run_dir, status, config_record


def _check_pair_inputs(core: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ff_rows = core.loc[core.model_id.eq("ff")].set_index(PAIR_KEYS)
    aa_rows = core.loc[core.model_id.eq("aa")].set_index(PAIR_KEYS)
    if len(ff_rows) != 60 or len(aa_rows) != 60 or not ff_rows.index.equals(aa_rows.index):
        raise IncompleteFullFactorialError("FF and AA are not a complete 60-pair grid")
    for key in ff_rows.index:
        ff, aa = ff_rows.loc[key], aa_rows.loc[key]
        if str(ff.source_fingerprint) != str(aa.source_fingerprint):
            raise ValueError(f"Source fingerprint differs for pair {key}")
        ff_id, aa_id = str(ff.run_id), str(aa.run_id)
        ff_dir, _, ff_record = _read_run_metadata(str(ff.experiment_id), ff_id)
        aa_dir, _, aa_record = _read_run_metadata(str(aa.experiment_id), aa_id)
        ff_cfg = ff_record.get("resolved_simulator_config", {})
        aa_cfg = aa_record.get("resolved_simulator_config", {})
        comparable_ff = {k: v for k, v in ff_cfg.items() if k not in MODE_KEYS | {"run_id"}}
        comparable_aa = {k: v for k, v in aa_cfg.items() if k not in MODE_KEYS | {"run_id"}}
        if comparable_ff != comparable_aa:
            raise ValueError(f"Resolved physical/application configs differ for pair {key}")
        ff_schedule, ff_hash = _read_schedule(ff_dir)
        aa_schedule, aa_hash = _read_schedule(aa_dir)
        if not ff_schedule.equals(aa_schedule):
            raise ValueError(f"Offered traffic differs for pair {key}")
        ff_topology = _routing_provenance(ff_dir)
        aa_topology = _routing_provenance(aa_dir)
        if ff_topology[0] != aa_topology[0]:
            raise ValueError(f"Topology differs for pair {key}")
        if ff_topology[1] != aa_topology[1]:
            raise ValueError(f"Routing differs for pair {key}")
        rows.append({
            **dict(zip(PAIR_KEYS, map(int, key))),
            "run_id_ff": ff_id, "run_id_aa": aa_id,
            "offered_count": len(ff_schedule),
            "offered_schedule_sha256": ff_hash,
            "topology_sha256": ff_topology[0],
            "routing_sha256": ff_topology[1],
            "physical_config_match": True, "offered_schedule_match": True,
            "topology_match": True, "routing_match": True, "pair_status": "PASS",
            "source_fingerprint_match": True,
        })
    return pd.DataFrame(rows)


def _routing_provenance(run_dir: Path) -> tuple[str, str]:
    path = run_dir / "tables/topology_parent_by_run.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Topology/routing provenance table missing: {path}")
    table = pd.read_csv(path)
    if table.empty:
        raise ValueError(f"Topology/routing provenance table is empty: {path}")
    topology = set(table["topology_sha256"].dropna().astype(str))
    routing = set(table["selected_parent_routing_sha256"].dropna().astype(str))
    if len(topology) != 1 or len(routing) != 1:
        raise ValueError(f"Inconsistent topology/routing hashes in {path}")
    return next(iter(topology)), next(iter(routing))


def _stats(frame: pd.DataFrame, group_keys: list[str], metric: str) -> pd.DataFrame:
    result = []
    for key, group in frame.groupby(group_keys, sort=True):
        values = pd.to_numeric(group[metric], errors="coerce").dropna()
        n = len(values)
        mean = float(values.mean()) if n else np.nan
        sd = float(values.std(ddof=1)) if n > 1 else np.nan
        sem = sd / np.sqrt(n) if n > 1 else np.nan
        critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else np.nan
        result.append({
            **dict(zip(group_keys, key if isinstance(key, tuple) else (key,))),
            "metric": metric, "n": n, "mean": mean, "sample_sd": sd, "sem": sem,
            "ci95_lower": mean - critical * sem if n > 1 else np.nan,
            "ci95_upper": mean + critical * sem if n > 1 else np.nan,
            "ci_method": "two-sided Student-t across five seeds; df=4",
        })
    return pd.DataFrame(result)


def _paired(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        wide = frame.pivot(index=PAIR_KEYS, columns="model_id", values=metric)
        delta = wide["aa"] - wide["ff"]
        for (nodes, period), group in delta.groupby(level=["node_num", "node_period"]):
            values = group.dropna().to_numpy(dtype=float)
            n = len(values)
            mean = float(values.mean()) if n else np.nan
            sd = float(values.std(ddof=1)) if n > 1 else np.nan
            sem = sd / np.sqrt(n) if n > 1 else np.nan
            half = float(student_t.ppf(0.975, n - 1)) * sem if n > 1 else np.nan
            rows.append({"node_num": nodes, "node_period": period, "metric": metric,
                         "difference_direction": "AA - FF", "n": n,
                         "mean_difference": mean, "sample_sd": sd, "sem": sem,
                         "ci95_lower": mean-half if n > 1 else np.nan,
                         "ci95_upper": mean+half if n > 1 else np.nan})
    return pd.DataFrame(rows)


def run_postprocessing(output_dir: str | Path = OUTPUT_DIR) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    frames = []
    input_files = []
    manifests = {}
    for model, experiment_id in EXPERIMENT_IDS.items():
        experiment_dir = ARTIFACT_ROOT / experiment_id
        manifest_path = experiment_dir / "experiment_manifest.json"
        core_path = resolve_master_table_path(experiment_dir, "all_runs_core_summary.csv")
        if not manifest_path.is_file() or not core_path.is_file():
            raise FileNotFoundError(f"Completed {model} experiment artifacts not found under {experiment_dir}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("spec", {}).get("base_config", {}).get("output_profile") != "summary":
            raise ValueError(f"{model} run manifest does not use output_profile=summary")
        core = pd.read_csv(core_path)
        if set(core["experiment_id"].astype(str)) != {experiment_id} or set(core["model_id"].astype(str)) != {model}:
            raise ValueError(f"Wrong experiment/model rows in {core_path}")
        if not core["run_id"].is_unique:
            raise ValueError(f"Duplicate run IDs in {core_path}")
        pdr_percent = pd.to_numeric(core["pdr_percent"], errors="raise")
        if not pdr_percent.between(0, 100).all():
            raise ValueError(f"PDR percentages must be within [0, 100] in {core_path}")
        core["source_fingerprint"] = hashlib.sha256(json.dumps(manifest.get("source_fingerprint"), sort_keys=True).encode()).hexdigest()
        core["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        core["experiment_profile"] = manifest.get("spec", {}).get("base_config", {}).get("experiment_profile")
        core["stage1_status"] = "PASS"
        core["config_hash"] = None
        core["source_commit"] = manifest.get("source_commit")
        core["topology_sha256"] = None
        core["selected_parent_routing_sha256"] = None
        core["energy_per_delivered_packet_J"] = _energy_per_delivered_packet(
            core["total_network_energy_J"], core["packet_fate_delivered_count"]
        )
        # Throughput is application goodput: delivered 50-byte payload bits / 3600 s.
        core["throughput_bps"] = pd.to_numeric(core["packet_fate_delivered_count"], errors="raise") * 50 * 8 / 3600.0
        for row in core.itertuples():
            run_dir, status, record = _read_run_metadata(experiment_id, str(row.run_id))
            if status.get("status") != "PASS":
                raise ValueError(f"Non-PASS status for {row.run_id}")
            resolved = record.get("resolved_simulator_config", {})
            if resolved.get("output_profile") != "summary" or resolved.get("energy_accounting_mode") != "corrected":
                raise ValueError(f"Unexpected output/energy profile for {row.run_id}")
            schedule, digest = _read_schedule(run_dir)
            core.loc[core.run_id.eq(row.run_id), "config_hash"] = record.get("config_hash")
            topology_hash, route_hash = _routing_provenance(run_dir)
            core.loc[core.run_id.eq(row.run_id), "topology_sha256"] = topology_hash
            core.loc[core.run_id.eq(row.run_id), "selected_parent_routing_sha256"] = route_hash
            core.loc[core.run_id.eq(row.run_id), "offered_packet_count"] = len(schedule)
            core.loc[core.run_id.eq(row.run_id), "offered_schedule_sha256"] = digest
        manifests[model] = manifest
        input_files += [manifest_path, core_path]
        frames.append(core)
    per_run = pd.concat(frames, ignore_index=True)
    validate_full_factorial_rows(per_run)
    pairs = _check_pair_inputs(per_run)
    stat_keys = ["model_id", "node_num", "node_period"]
    stats = pd.concat([_stats(per_run, stat_keys, metric) for metric in METRICS], ignore_index=True)
    pair_stats = _paired(per_run)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir()
    atomic_write_csv(output_dir / "combined_dataset.csv", per_run)
    atomic_write_csv(output_dir / "comparison_summary.csv", stats)
    atomic_write_csv(output_dir / "paired_statistics.csv", pair_stats)
    atomic_write_csv(output_dir / "paired_validation.csv", pairs)
    duty_rows, energy_rows, occupancy_rows, apc_rows = [], [], [], []
    for row in per_run.itertuples():
        run_dir = ARTIFACT_ROOT / str(row.experiment_id) / "runs" / str(row.run_id)
        for filename, target in (("relay_duty_cycle.csv", duty_rows),
                                 ("node_tx_duty_cycle.csv", duty_rows),
                                 ("energy_breakdown.csv", energy_rows),
                                 ("channel_occupancy.csv", occupancy_rows),
                                 ("apc_feedback_lifecycle.csv", apc_rows)):
            path = run_dir / "tables" / filename
            if path.is_file() and path.stat().st_size:
                table = pd.read_csv(path)
                if not table.empty:
                    target.append(table)
    for filename, rows in (("relay_and_node_duty.csv", duty_rows), ("energy_breakdown.csv", energy_rows),
                           ("channel_occupancy.csv", occupancy_rows), ("apc_feedback_lifecycle.csv", apc_rows)):
        atomic_write_csv(output_dir / filename, pd.concat(rows, ignore_index=True) if rows else pd.DataFrame())
    figures = []
    for metric, label, stem in (("pdr_percent", "Network PDR (%)", "figure_01_pdr_vs_nodes"),
                                ("energy_per_delivered_packet_J", "Energy per delivered packet (J/packet)", "figure_02_energy_per_delivered_packet_vs_nodes")):
        figures.extend(_plot_metric(stats, metric, label, figures_dir / stem))
    relay_rows = []
    for row in per_run.itertuples():
        path = ARTIFACT_ROOT / str(row.experiment_id) / "runs" / str(row.run_id) / "tables/relay_duty_cycle.csv"
        relay = pd.read_csv(path)
        relay = relay.loc[relay.entity_type.eq("relay")]
        for col, val in zip(("model_id", "node_num", "node_period", "seed"),
                            (row.model_id, row.node_num, row.node_period, row.seed)):
            relay[col] = val
        relay_rows.append(relay)
    relay_frame = pd.concat(relay_rows, ignore_index=True)
    relay_stats = _stats(relay_frame, ["model_id", "node_num", "node_period", "entity_id"], "relay_duty_cycle_percent")
    atomic_write_csv(output_dir / "relay_duty_statistics.csv", relay_stats)
    figures.extend(_plot_relay_duty(relay_stats.rename(columns={"entity_id": "entity_id"}), figures_dir / "figure_03_relay_duty_vs_nodes"))
    manifest = {"status": "PASS", "analysis_version": "apc_load_sweep_full_factorial_v1",
                "validated_run_count": 120, "paired_comparison_count": 60,
                "scenarios": APC_LOAD_SWEEP_FULL_FACTORIAL_BATCH,
                "seeds": list(APC_LOAD_SWEEP_SEEDS), "models": ["ff", "aa"],
                "output_profile": "summary", "energy_accounting_mode": "corrected",
                "statistics": "mean, sample SD, SEM, two-sided Student-t 95% CI across seeds (df=4)",
                "throughput_definition": "delivered 50-byte application payload bits / 3600 s",
                "input_manifests": {model: hashlib.sha256((ARTIFACT_ROOT / eid / "experiment_manifest.json").read_bytes()).hexdigest()
                                    for model, eid in EXPERIMENT_IDS.items()},
                "figures": [str(path.relative_to(output_dir)) for path in figures]}
    atomic_write_json(output_dir / "analysis_manifest.json", manifest)
    return {"status": "PASS", "validated_runs": 120, "validated_pairs": len(pairs),
            "analysis_dir": str(output_dir), "figures": [str(p) for p in figures]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reporting-only APC full-factorial analysis")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args(argv)
    print(json.dumps(run_postprocessing(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
