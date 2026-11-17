"""User-invoked paired Summary/Detailed acceptance pilot and validator.

This module never runs unless explicitly invoked with ``--run``. The pilot is
three workload scenarios × FF/AA × two output profiles (12 short validation runs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from loraphymac.experiments.common.multiseed_pipeline import (
    build_experiment_grid,
    load_experiment_spec,
)
from loraphymac.config.defaults import build_config
from loraphymac.experiments.apc_load_sweep.contract import (
    APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_PILOT_PROFILE,
)
from loraphymac.simulator import resolve_configuration


EXPERIMENT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = EXPERIMENT_DIR / "configs"
PILOT_DIR = Path(__file__).resolve().parents[4] / "artifacts/experiments/apc_load_sweep/output_profile_acceptance_v3"
SCENARIOS = ((50, 90), (150, 60), (200, 30))
SEED = 30
EQUIVALENCE_TABLES = (
    "core_summary_df", "energy_breakdown_df", "packet_fate_summary_df",
    "packet_fate_df", "relay_duty_cycle_df", "per_channel_duty_cycle_df",
    "collision_summary_df", "node_tx_duty_cycle_df", "channel_occupancy_df",
    "apc_feedback_lifecycle_df", "topology_parent_by_run_df",
    "tx_airtime_by_entity_mtype_ptx_channel_df", "frame_summary_df",
    "frame_mtype_summary_df", "frame_hop_event_summary_df",
    "frame_drop_reason_by_hop_df", "final_ptx_distribution_df",
    "routing_map_df", "routing_table_df",
)
AIRTIME_TABLE = "tx_airtime_by_entity_mtype_ptx_channel_df"
AIRTIME_SHEET = "R19_tx_airtime_by_entity_mtype_"
AIRTIME_KEY_COLUMNS = (
    "entity_id", "entity_type", "traffic_class", "mtype", "mtype_name",
    "ptx_dbm", "channel", "sf", "bw_hz",
)
AIRTIME_TOLERANCES_S = {
    "total_airtime_s": 1e-12,
    "entity_grouped_airtime_s": 1e-12,
    "entity_recorded_total_tx_time_s": 1e-12,
    "airtime_consistency_delta_s": 1e-12,
}


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _table_hash(frame: pd.DataFrame, run_id: str | None = None) -> str:
    # Per-profile run IDs are deliberately unique. Remove the metadata column
    # and canonicalize their use inside lifecycle identifiers before comparing.
    comparable = frame.drop(columns=["run_id"], errors="ignore").copy()
    if run_id:
        for column in comparable.select_dtypes(include=["object", "string"]).columns:
            comparable[column] = comparable[column].map(
                lambda value: value.replace(run_id, "<RUN_ID>")
                if isinstance(value, str)
                else value
            )
    data = comparable.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _scientific_config_hash(config: dict[str, Any]) -> str:
    """Hash resolved scientific inputs, excluding run/output-only metadata."""
    config_for_hash = dict(config)
    for key in (
        "output_profile", "output_dir", "run_id", "export_summary",
        "export_results", "export_debug", "export_internal", "plot_routing",
        "print_summary", "print_detail", "print_debug", "print_internal", "verbose",
    ):
        config_for_hash.pop(key, None)
    resolved = resolve_configuration(build_config(config_for_hash))
    encoded = json.dumps(resolved, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def compare_scientific_table(
    summary: pd.DataFrame,
    detailed: pd.DataFrame,
    *,
    key_columns: tuple[str, ...],
    absolute_tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare keyed tables exactly except explicitly tolerated numeric columns."""
    tolerances = absolute_tolerances or {}
    failures: list[str] = []
    if list(summary.columns) != list(detailed.columns):
        return {"equivalent": False, "failures": ["schema/column order mismatch"]}
    if any(column not in summary.columns for column in key_columns):
        return {"equivalent": False, "failures": ["missing comparison key column"]}
    if summary[list(key_columns)].isna().any().any() or detailed[list(key_columns)].isna().any().any():
        return {"equivalent": False, "failures": ["null comparison key"]}
    if summary.duplicated(list(key_columns)).any() or detailed.duplicated(list(key_columns)).any():
        return {"equivalent": False, "failures": ["duplicate comparison key"]}
    dtype_mismatches = {
        column: {"summary": str(summary[column].dtype), "detailed": str(detailed[column].dtype)}
        for column in summary.columns
        if summary[column].dtype != detailed[column].dtype
    }

    merged = summary.merge(
        detailed,
        on=list(key_columns),
        how="outer",
        suffixes=("_summary", "_detailed"),
        indicator=True,
        sort=False,
    )
    if not merged["_merge"].eq("both").all():
        missing_summary = int(merged["_merge"].eq("right_only").sum())
        missing_detailed = int(merged["_merge"].eq("left_only").sum())
        failures.append(
            f"row-key mismatch: missing_summary={missing_summary}, "
            f"missing_detailed={missing_detailed}"
        )

    max_absolute: dict[str, float] = {}
    max_relative: dict[str, float] = {}
    for column in (name for name in summary.columns if name not in key_columns):
        left = merged[f"{column}_summary"]
        right = merged[f"{column}_detailed"]
        if column in tolerances:
            try:
                left_num = pd.to_numeric(left, errors="raise").astype(float)
                right_num = pd.to_numeric(right, errors="raise").astype(float)
            except (TypeError, ValueError):
                failures.append(f"non-numeric tolerated field: {column}")
                continue
            both_null = left_num.isna() & right_num.isna()
            abs_diff = (left_num - right_num).abs()
            abs_diff = abs_diff.mask(both_null, 0.0)
            relative = abs_diff / pd.concat([left_num.abs(), right_num.abs()], axis=1).max(axis=1)
            relative = relative.mask(both_null | (abs_diff.eq(0.0)), 0.0).fillna(float("inf"))
            max_absolute[column] = float(abs_diff.max()) if len(abs_diff) else 0.0
            max_relative[column] = float(relative.max()) if len(relative) else 0.0
            if not (both_null | abs_diff.le(float(tolerances[column]))).all():
                failures.append(
                    f"{column} exceeds atol={tolerances[column]:g}s "
                    f"(max_abs={max_absolute[column]:.17g}s, "
                    f"max_rel={max_relative[column]:.17g})"
                )
        else:
            equal = left.eq(right) | (left.isna() & right.isna())
            if not equal.all():
                failures.append(f"exact field mismatch: {column}")
    return {
        "equivalent": not failures,
        "failures": failures,
        "dtype_mismatches": dtype_mismatches,
        "max_absolute_difference": max_absolute,
        "max_relative_difference": max_relative,
        "absolute_tolerances": tolerances,
        "matched_row_count": int(merged["_merge"].eq("both").sum()),
    }


def _load_airtime_table(run_dir: Path) -> pd.DataFrame:
    workbooks = sorted(run_dir.glob("results_*.xlsx"))
    if len(workbooks) != 1:
        raise ValueError(f"Expected one results workbook in {run_dir}, found {len(workbooks)}")
    return pd.read_excel(workbooks[0], sheet_name=AIRTIME_SHEET).drop(
        columns=["run_id"], errors="ignore"
    )


def load_existing_pilot_results(output_dir: Path) -> pd.DataFrame:
    """Load completed per-run JSON records for reporting-only validation."""
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted(output_dir.glob("*/acceptance_result.json"))]
    return pd.DataFrame(records)


def _pilot_run_id(model: str, nodes: int, period: int, seed: int, profile: str) -> str:
    return f"profile_acceptance_v3_{model}_n{nodes}_p{period}_s{seed}_{profile}"


def run_worker(config_path: Path, result_path: Path) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    from loraphymac.simulation.runner import run_single_simulation

    started = perf_counter()
    core, _ptx, _margin, tables = run_single_simulation(payload["config"])
    elapsed = perf_counter() - started
    missing = set(EQUIVALENCE_TABLES) - set(tables)
    if missing:
        raise ValueError(f"Required comparison tables missing: {sorted(missing)}")
    run_id = str(payload["config"]["run_id"])
    hashes = {
        name: _table_hash(tables[name], run_id=run_id)
        for name in EQUIVALENCE_TABLES
    }
    schedule = tables["packet_fate_df"][
        ["packet_uid", "src_id", "fcnt", "birth_time", "offered_time"]
    ].sort_values(["src_id", "fcnt", "packet_uid"], kind="stable").reset_index(drop=True)
    topology = tables["topology_parent_by_run_df"]
    topology_hashes = set(topology["topology_sha256"].dropna().astype(str))
    route_hashes = set(topology["selected_parent_routing_sha256"].dropna().astype(str))
    if len(topology_hashes) != 1 or len(route_hashes) != 1:
        raise ValueError("Topology/routing hashes are missing or inconsistent")
    config_sha = _scientific_config_hash(payload["config"])
    out_dir = Path(str(payload["config"]["output_dir"]))
    files = [p for p in out_dir.rglob("*") if p.is_file()]
    result = {
        "run_id": payload["config"]["run_id"],
        "model_id": payload["model_id"],
        "node_num": payload["node_num"],
        "node_period": payload["node_period"],
        "seed": SEED,
        "output_profile": payload["output_profile"],
        "runtime_s": elapsed,
        "output_bytes": sum(path.stat().st_size for path in files),
        "file_count": len(files),
        "peak_rss_bytes": _peak_rss_bytes(),
        "table_sha256": hashes,
        "offered_schedule_sha256": _table_hash(schedule),
        "topology_sha256": next(iter(topology_hashes)),
        "routing_sha256": next(iter(route_hashes)),
        "physical_config_sha256": config_sha,
        "core": dict(core),
    }
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def _worker(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-config", type=Path, required=True)
    parser.add_argument("--worker-result", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_worker(args.worker_config, args.worker_result)
    print(json.dumps(result))
    return 0


def run_pilot(output_dir: Path = PILOT_DIR) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty pilot directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    spec_by_model = {
        model: load_experiment_spec(CONFIG_DIR / f"{model}_full_factorial_v1.json")
        for model in ("ff", "aa")
    }
    worker = [sys.executable, "-m", "loraphymac.experiments.apc_load_sweep.profile_acceptance_pilot"]
    with tempfile.TemporaryDirectory(prefix="loraphymac-profile-pilot-") as temp_name:
        temp_dir = Path(temp_name)
        for nodes, period in SCENARIOS:
            for model, spec in spec_by_model.items():
                run = next(
                    item for item in build_experiment_grid(spec)
                    if item["seed"] == SEED
                    and item["factors"] == {"node_num": nodes, "node_period": period}
                )
                for profile in ("summary", "detailed"):
                    run_id = _pilot_run_id(model, nodes, period, SEED, profile)
                    run_dir = output_dir / run_id
                    config = dict(run["scientific_config"])
                    config.update({
                        "experiment_profile": APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_PILOT_PROFILE,
                        "output_profile": profile,
                        "run_id": run_id,
                        "output_dir": str(run_dir),
                        "export_summary": True,
                        "export_results": True,
                        "export_debug": True,
                        "plot_routing": False,
                        "print_summary": False,
                        "print_detail": False,
                        "print_debug": False,
                        "verbose": False,
                    })
                    config_path = temp_dir / f"{run_id}.json"
                    result_path = run_dir / "acceptance_result.json"
                    config_path.write_text(json.dumps({"config": config, "model_id": model,
                                                       "node_num": nodes, "node_period": period,
                                                       "output_profile": profile}, indent=2))
                    subprocess.run(
                        [*worker, "--worker-config", str(config_path),
                         "--worker-result", str(result_path)], check=True,
                        stdout=subprocess.DEVNULL,
                    )
                    rows.append(json.loads(result_path.read_text()))
    results = pd.DataFrame(rows)
    comparisons = validate_pilot_results(results, output_dir=output_dir)
    pd.DataFrame([{k: v for k, v in row.items() if k != "table_sha256" and k != "core"}
                  for row in rows]).to_csv(output_dir / "pilot_measurements.csv", index=False)
    comparisons.to_csv(output_dir / "pilot_equivalence.csv", index=False)
    return {"status": "PASS", "pilot_runs": len(rows), "paired_scenarios": len(comparisons),
            "output_dir": str(output_dir)}


def validate_pilot_results(
    results: pd.DataFrame, *, output_dir: Path | None = None
) -> pd.DataFrame:
    expected = {
        (model, nodes, period, profile)
        for model in ("ff", "aa") for nodes, period in SCENARIOS
        for profile in ("summary", "detailed")
    }
    actual = {
        (str(r.model_id), int(r.node_num), int(r.node_period), str(r.output_profile))
        for r in results.itertuples()
    }
    if actual != expected or len(results) != 12:
        raise ValueError(f"Pilot must contain 12 unique runs; missing={len(expected-actual)}")
    rows = []
    for (model, nodes, period), group in results.groupby(["model_id", "node_num", "node_period"]):
        profiles = {str(row.output_profile): row for row in group.itertuples()}
        summary, detailed = profiles["summary"], profiles["detailed"]
        for field in (
            "offered_schedule_sha256", "topology_sha256", "routing_sha256",
            "physical_config_sha256",
        ):
            if getattr(summary, field) != getattr(detailed, field):
                raise ValueError(f"Paired input {field} differs for {model}/{nodes}/{period}")
        hash_mismatches = [
            name for name in EQUIVALENCE_TABLES
            if summary.table_sha256[name] != detailed.table_sha256[name]
        ]
        airtime_comparison = None
        for name in hash_mismatches:
            if name != AIRTIME_TABLE or output_dir is None:
                raise ValueError(
                    f"Scientific outputs differ for {model}/{nodes}/{period}: "
                    f"{hash_mismatches}"
                )
            airtime_comparison = compare_scientific_table(
                _load_airtime_table(output_dir / str(summary.run_id)),
                _load_airtime_table(output_dir / str(detailed.run_id)),
                key_columns=AIRTIME_KEY_COLUMNS,
                absolute_tolerances=AIRTIME_TOLERANCES_S,
            )
            if not airtime_comparison["equivalent"]:
                raise ValueError(
                    f"Scientific outputs differ for {model}/{nodes}/{period}: "
                    f"{airtime_comparison['failures']}"
                )
        rows.append({"model_id": model, "node_num": nodes, "node_period": period,
                     "seed": SEED, "equivalence": "PASS",
                     "hash_mismatch_tables": ";".join(hash_mismatches),
                     "max_airtime_abs_difference_s": max(
                         (airtime_comparison or {}).get("max_absolute_difference", {}).values(),
                         default=0.0,
                     ),
                     "max_airtime_relative_difference": max(
                         (airtime_comparison or {}).get("max_relative_difference", {}).values(),
                         default=0.0,
                     ),
                     "summary_runtime_s": summary.runtime_s,
                     "detailed_runtime_s": detailed.runtime_s,
                     "summary_output_bytes": summary.output_bytes,
                     "detailed_output_bytes": detailed.output_bytes,
                     "summary_file_count": summary.file_count,
                     "detailed_file_count": detailed.file_count,
                     "summary_peak_rss_bytes": summary.peak_rss_bytes,
                     "detailed_peak_rss_bytes": detailed.peak_rss_bytes})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--worker-config" in args:
        return _worker(args)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly execute the 12-run validation pilot")
    parser.add_argument(
        "--validate-existing", action="store_true",
        help="Validate completed pilot artifacts without running simulations",
    )
    parser.add_argument("--output-dir", type=Path, default=PILOT_DIR)
    parsed = parser.parse_args(args)
    if parsed.validate_existing:
        results = load_existing_pilot_results(parsed.output_dir.expanduser().resolve())
        comparisons = validate_pilot_results(
            results, output_dir=parsed.output_dir.expanduser().resolve()
        )
        print(json.dumps({
            "status": "PASS",
            "pilot_runs": int(len(results)),
            "paired_scenarios": int(len(comparisons)),
            "output_dir": str(parsed.output_dir.expanduser().resolve()),
            "comparisons": comparisons.to_dict(orient="records"),
        }, indent=2, default=str))
        return 0
    if not parsed.run:
        parser.error("No simulations are run by default; pass --run only when ready")
    print(json.dumps(run_pilot(parsed.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
