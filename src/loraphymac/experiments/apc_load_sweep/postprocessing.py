"""Paired-data validation, summary statistics, and figures for APC load sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from scipy.stats import t as student_t

from loraphymac.experiments.common.single_run_diagnostics.figures import FIGURE_STYLE
from loraphymac.experiments.common.multiseed_pipeline import (
    atomic_write_csv,
    atomic_write_json,
    build_experiment_grid,
    file_sha256,
    load_experiment_spec,
    _source_fingerprint,
    utc_now,
)
from loraphymac.experiments.common.outputs import resolve_master_table_path
from loraphymac.paths import repository_root

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = repository_root(EXPERIMENT_DIR)
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/experiments/apc_load_sweep"
RUN_GROUP_IDS = {
    "ff": ("apc_load_sweep_ff_wide_nodes_90s", "apc_load_sweep_ff_node150_periods"),
    "aa": ("apc_load_sweep_aa_wide_nodes_90s", "apc_load_sweep_aa_node150_periods"),
}
RUN_CONFIG_FILES = {
    "ff": ("ff_wide_nodes_90s.json", "ff_node150_periods.json"),
    "aa": ("aa_wide_nodes_90s.json", "aa_node150_periods.json"),
}
LEGACY_EXPERIMENT_IDS = {"ff": "apc_load_sweep_ff_r01", "aa": "apc_load_sweep_aa_r01"}
NODE_COUNTS = (50, 100, 150, 200)
PERIODS_S = (30, 60, 90)
SEEDS = (30, 36, 42, 48, 56)
HISTORICAL_SCENARIOS = {
    (50, 30), (50, 60), (100, 30), (100, 60), (200, 30), (200, 60)
}
NEW_SCENARIOS = {
    (50, 90), (100, 90), (150, 30), (150, 60), (150, 90), (200, 90)
}
COMBINED_SCENARIOS = HISTORICAL_SCENARIOS | NEW_SCENARIOS
MODEL_ORDER = ("aa", "ff")
PERIOD_COLORS = {30: "#d62728", 60: "#1764ab", 90: "#218739"}
METRICS = ("pdr_percent", "energy_per_delivered_packet_J")
PAIR_KEYS = ["node_num", "node_period", "seed"]
SCHEDULE_COLUMNS = ["packet_uid", "src_id", "fcnt", "birth_time", "offered_time"]
COMMON_CONFIG_COLUMNS = [
    # node_num and node_period are pairing keys; node_period_s duplicates node_period.
    "sim_time", "terrain_delta_h", "node_jitter_s",
    "relay_num", "node_bw", "relay_bw_hz", "node_app_payload_size_bytes",
    "node_sf_list", "relay_sf", "energy_accounting_mode", "collision_mode",
    "collision_sf_interference_mode", "node_rx_window_enabled",
    "application_scheduling_mode", "application_rng_policy",
    "application_queue_policy", "relay_duty_cycle",
]
SOURCE_CHANGE_CLASSES = {
    "src/loraphymac/experiments/apc_load_sweep/contract.py": "experiment contract/configuration validation only",
    "src/loraphymac/experiments/apc_load_sweep/maintenance.py": "experiment audit/maintenance only",
    "src/loraphymac/experiments/apc_load_sweep/postprocessing.py": "reporting/postprocessing only",
    "src/loraphymac/experiments/common/contract_registry.py": "experiment orchestration/contract registration only",
    "src/loraphymac/simulation/output_collection.py": "output instrumentation only",
}
SOURCE_COMMIT_KEYS = ("source_commit", "git_commit", "git_head", "commit_sha")
MODEL_MODE_KEYS = {"node_ptx_mode", "relay_ptx_mode"}
VARIABLE_CONFIG_KEYS = MODEL_MODE_KEYS | {
    "node_num", "node_period", "seed", "run_id", "experiment_profile",
    "output_profile",
}


class IncompleteDatasetError(ValueError):
    """Raised when the combined historical/new dataset is not complete."""


def _expected_new_runs(model_id: str) -> dict[str, tuple[int, int, int]]:
    expected: dict[str, tuple[int, int, int]] = {}
    for config_name in RUN_CONFIG_FILES[model_id]:
        spec = load_experiment_spec(EXPERIMENT_DIR / "configs" / config_name)
        if spec.run_annotations.get("model_id") != model_id:
            raise ValueError(f"Wrong model annotation in {config_name}")
        for run in build_experiment_grid(spec):
            factors = run["factors"]
            expected[run["run_id"]] = (
                int(factors["node_num"]), int(factors["node_period"]), int(run["seed"])
            )
    return expected


def _normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in VARIABLE_CONFIG_KEYS}


def _source_fingerprint_digest(fingerprint: dict[str, Any]) -> str:
    canonical = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _manifest_for(experiment_id: str) -> tuple[Path, dict[str, Any]]:
    path = _experiment_dir(experiment_id) / "experiment_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Experiment provenance manifest is missing: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _removed_15s_audit() -> tuple[Path, dict[str, Any]]:
    path = ARTIFACT_ROOT / "audit/removed_15s_runs.json"
    if not path.is_file():
        raise FileNotFoundError(f"15-second run removal audit is missing: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "DELETED_AFTER_HASH_AUDIT":
        raise ValueError("15-second removal audit is not finalized")
    if audit.get("run_count") != 30 or audit.get("file_count") != 540:
        raise ValueError("15-second removal audit count does not match the verified inventory")
    if any((PROJECT_ROOT / row["run_directory"]).exists() for row in audit.get("runs", [])):
        raise ValueError("An audited 15-second raw run directory still exists")
    return path, audit


def _validate_source_compatibility(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    historical_ids = set(LEGACY_EXPERIMENT_IDS.values())
    new_ids = {eid for group in RUN_GROUP_IDS.values() for eid in group}
    historical = [manifests[x].get("source_fingerprint") for x in sorted(historical_ids)]
    new = [manifests[x].get("source_fingerprint") for x in sorted(new_ids)]
    if any(not isinstance(fp, dict) or not isinstance(fp.get("files"), dict) for fp in historical + new):
        raise ValueError("Source fingerprint is missing or malformed in an experiment manifest")
    if any(fp != historical[0] for fp in historical[1:]):
        raise ValueError("Historical FF/AA runs do not share one source fingerprint")
    if any(fp != new[0] for fp in new[1:]):
        raise ValueError("New FF/AA batches do not share one source fingerprint")
    current_fingerprint = _source_fingerprint(PROJECT_ROOT)
    if new[0] != current_fingerprint:
        raise ValueError("Current source differs from the source used for the new runs")
    old_files = historical[0]["files"]
    new_files = new[0]["files"]
    changed = sorted(
        path for path in set(old_files) | set(new_files)
        if old_files.get(path) != new_files.get(path)
    )
    unreviewed = sorted(set(changed) - set(SOURCE_CHANGE_CLASSES))
    if unreviewed:
        raise ValueError(
            "Historical/new source differs in unreviewed files: " + ", ".join(unreviewed)
        )
    # Ensure that the reviewed differences are exactly the current audited source,
    # not a later unreviewed edit to the same filename.
    for path in changed:
        source_path = PROJECT_ROOT / path
        if not source_path.is_file() or file_sha256(source_path) != new_files.get(path):
            raise ValueError(f"Reviewed source differs from the new run fingerprint: {path}")
    return {
        "status": "COMPATIBLE_NON_SIMULATION_SOURCE_CHANGES",
        "changed_file_count": len(changed),
        "changed_files": [
            {"path": path, "classification": SOURCE_CHANGE_CLASSES[path]}
            for path in changed
        ],
        "historical_fingerprint_sha256": _source_fingerprint_digest(historical[0]),
        "new_fingerprint_sha256": _source_fingerprint_digest(new[0]),
        "source_commit_available": all(
            any(manifests[eid].get(key) for key in SOURCE_COMMIT_KEYS)
            for eid in historical_ids | new_ids
        ),
    }


def validate_combined_dataset(frame: pd.DataFrame) -> None:
    """Validate exact 120-run union, status, per-cell seed/model coverage and provenance."""
    required = {
        "run_id", "experiment_id", "experiment_profile", "model_id", "node_num",
        "node_period", "seed", "terrain_delta_h", "source_commit",
        "source_fingerprint", "config_hash", "source_cohort", "stage1_status",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Combined dataset lacks required provenance: {sorted(missing)}")
    if not frame["stage1_status"].astype(str).eq("PASS").all():
        raise ValueError("All 120 combined runs must have PASS status")
    if frame["node_period"].astype(int).eq(15).any():
        raise ValueError("15-second runs are prohibited in the combined dataset")
    key_columns = ["node_num", "node_period", "seed", "model_id"]
    if frame.duplicated(key_columns).any() or frame["run_id"].duplicated().any():
        raise ValueError("Duplicate run or composite scenario/model/seed key")
    expected = {
        (model, nodes, period, seed)
        for model in MODEL_ORDER
        for nodes, period in COMBINED_SCENARIOS
        for seed in SEEDS
    }
    actual = {
        (str(row.model_id), int(row.node_num), int(row.node_period), int(row.seed))
        for row in frame.itertuples()
    }
    if actual != expected:
        missing_keys, extra_keys = expected - actual, actual - expected
        raise IncompleteDatasetError(
            f"Grid gabungan tidak lengkap: missing={len(missing_keys)}, unexpected={len(extra_keys)}"
        )
    if len(frame) != 120:
        raise IncompleteDatasetError(f"Dataset final harus 120 run; ditemukan {len(frame)}")
    if frame["source_fingerprint"].isna().any() or frame["config_hash"].isna().any():
        raise ValueError("Source fingerprint and resolved config hash must be present per run")
    for _, row in frame.iterrows():
        expected_cohort = (
            "historical" if (int(row.node_num), int(row.node_period)) in HISTORICAL_SCENARIOS
            else "new"
        )
        if row.source_cohort != expected_cohort:
            raise ValueError(f"Incorrect source_cohort for run {row.run_id}")


def _experiment_dir(experiment_id: str) -> Path:
    return ARTIFACT_ROOT / experiment_id


def _load_model_core(model_id: str) -> tuple[pd.DataFrame, set[str]]:
    """Load retained historical rows and any available new rows for one model."""
    legacy_path = resolve_master_table_path(
        _experiment_dir(LEGACY_EXPERIMENT_IDS[model_id]),
        "all_runs_core_summary.csv",
    )
    if not legacy_path.is_file():
        raise FileNotFoundError(f"Historical core summary is missing: {legacy_path}")
    legacy_all = pd.read_csv(legacy_path)
    all_periods = pd.to_numeric(legacy_all["node_period"], errors="raise").astype(int)
    excluded_15s = legacy_all.loc[all_periods.eq(15)].copy()
    _, audit = _removed_15s_audit()
    expected_removed = {
        row["run_id"] for row in audit["runs"] if row["model_id"] == model_id
    }
    if set(excluded_15s["run_id"].astype(str)) != expected_removed or len(expected_removed) != 15:
        raise ValueError(f"Historical 15-second summary rows do not match removal audit for {model_id}")
    legacy = legacy_all.loc[~all_periods.eq(15)].copy()
    if len(legacy) != 30 or not legacy["stage1_status"].astype(str).eq("PASS").all():
        raise ValueError(f"Expected 30 retained PASS historical rows for {model_id}")
    expected_historical = {
        (nodes, period, seed)
        for nodes, period in HISTORICAL_SCENARIOS
        for seed in SEEDS
    }
    actual_historical = {
        (int(row.node_num), int(row.node_period), int(row.seed))
        for row in legacy.itertuples()
    }
    if actual_historical != expected_historical or legacy["run_id"].duplicated().any():
        raise ValueError(f"Historical {model_id} matrix has missing or duplicate rows")

    expected_new: dict[str, tuple[int, int, int]] = {}
    batches = []
    for experiment_id, config_name in zip(
        RUN_GROUP_IDS[model_id], RUN_CONFIG_FILES[model_id], strict=True
    ):
        spec = load_experiment_spec(EXPERIMENT_DIR / "configs" / config_name)
        expected_group = {}
        for run in build_experiment_grid(spec):
            factors = run["factors"]
            expected_group[run["run_id"]] = (
                int(factors["node_num"]), int(factors["node_period"]), int(run["seed"])
            )
        expected_new.update(expected_group)
        path = resolve_master_table_path(
            _experiment_dir(experiment_id), "all_runs_core_summary.csv"
        )
        if path.is_file():
            batch = pd.read_csv(path)
            if batch["run_id"].duplicated().any():
                raise ValueError(f"Duplicate run_id in {path}")
            if set(batch["experiment_id"].astype(str)) != {experiment_id}:
                raise ValueError(f"Experiment ID mismatch in {path}")
            batches.append(batch)
    new = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()
    observed_new_ids = set(new.get("run_id", pd.Series(dtype=str)).astype(str))
    missing_new_ids = set(expected_new) - observed_new_ids
    if not new.empty:
        unexpected = observed_new_ids - set(expected_new)
        if unexpected:
            raise ValueError(f"Unexpected run IDs in new {model_id} batches: {sorted(unexpected)[:5]}")
        if set(new["model_id"].astype(str)) != {model_id}:
            raise ValueError(f"New batch contains a wrong model for {model_id}")
        if not new["stage1_status"].astype(str).eq("PASS").all():
            raise ValueError(f"New {model_id} batch contains a non-PASS run")
        for row in new.itertuples():
            expected_key = expected_new[str(row.run_id)]
            actual_key = (int(row.node_num), int(row.node_period), int(row.seed))
            if actual_key != expected_key:
                raise ValueError(f"Resolved grid mismatch for run {row.run_id}")

    frame = pd.concat([legacy, new], ignore_index=True)
    if frame["run_id"].duplicated().any():
        raise ValueError(f"Duplicate run IDs in historical/new {model_id} data")
    if set(legacy["model_id"].astype(str)) != {model_id}:
        raise ValueError(f"Historical core summary contains a wrong model for {model_id}")
    return frame, missing_new_ids


def _attach_run_provenance(
    core: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[Path]]:
    manifests: dict[str, dict[str, Any]] = {}
    manifest_paths: dict[str, Path] = {}
    resolved_configs: dict[str, dict[str, Any]] = {}
    rows = []
    for run in core.itertuples():
        experiment_id, run_id = str(run.experiment_id), str(run.run_id)
        manifest_path, manifest = _manifest_for(experiment_id)
        manifests[experiment_id], manifest_paths[experiment_id] = manifest, manifest_path
        resolved_path = _experiment_dir(experiment_id) / "runs" / run_id / "resolved_config.json"
        status_path = resolved_path.parent / "status.json"
        if not resolved_path.is_file() or not status_path.is_file():
            raise ValueError(f"Run provenance files are missing for {run_id}")
        resolved_record = json.loads(resolved_path.read_text(encoding="utf-8"))
        config = resolved_record.get("resolved_simulator_config", {})
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if str(resolved_record.get("experiment_id")) != experiment_id:
            raise ValueError(f"Resolved experiment ID mismatch for {run_id}")
        if status.get("status") != "PASS" or str(run.stage1_status) != "PASS":
            raise ValueError(f"Run {run_id} does not have PASS status in both sources")
        for key, expected_value in (
            ("run_id", run_id), ("node_num", int(run.node_num)),
            ("node_period", int(run.node_period)), ("seed", int(run.seed)),
        ):
            actual_value = resolved_record.get(key, config.get(key))
            if str(actual_value) != str(expected_value):
                raise ValueError(f"Resolved configuration mismatch for {run_id}: {key}")
        if str(config.get("experiment_profile")) != str(run.experiment_profile):
            raise ValueError(f"Experiment profile mismatch for run {run_id}")
        if str(config.get("terrain_delta_h")) != str(run.terrain_delta_h):
            raise ValueError(f"Terrain provenance mismatch for run {run_id}")
        fingerprint = manifest.get("source_fingerprint")
        if not isinstance(fingerprint, dict) or not isinstance(fingerprint.get("files"), dict):
            raise ValueError(f"Missing source fingerprint for {experiment_id}")
        spec = manifest.get("spec", {})
        if spec.get("experiment_id") != experiment_id:
            raise ValueError(f"Manifest experiment ID mismatch for {experiment_id}")
        if spec.get("run_annotations", {}).get("model_id") != str(run.model_id):
            raise ValueError(f"Manifest model annotation mismatch for {experiment_id}")
        source_commit = next(
            (manifest[key] for key in SOURCE_COMMIT_KEYS if manifest.get(key)), None
        )
        record = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "resolved_experiment_profile": config.get("experiment_profile"),
            "model_id": str(run.model_id),
            "node_num": int(run.node_num),
            "node_period": int(run.node_period),
            "seed": int(run.seed),
            "terrain_delta_h": config.get("terrain_delta_h"),
            "source_commit": source_commit,
            "source_commit_status": "RECORDED" if source_commit else "NOT_RECORDED_IN_MANIFEST",
            "source_fingerprint": _source_fingerprint_digest(fingerprint),
            "source_fingerprint_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "config_hash": resolved_record.get("config_hash"),
            "resolved_config_file_sha256": file_sha256(resolved_path),
            "resolved_config_path": str(resolved_path.relative_to(PROJECT_ROOT)),
            "source_cohort": "historical" if experiment_id in LEGACY_EXPERIMENT_IDS.values() else "new",
            "stage1_status": str(run.stage1_status),
        }
        rows.append(record)
        resolved_configs[run_id] = config
    provenance = pd.DataFrame(rows)
    enriched = core.merge(
        provenance,
        on=["run_id", "experiment_id", "model_id", "node_num", "node_period", "seed", "stage1_status"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_provenance"),
    )
    return enriched, manifests, resolved_configs, list(manifest_paths.values())


def _validate_cross_cohort_configs(resolved_configs: dict[str, dict[str, Any]]) -> None:
    reference = None
    for run_id, config in resolved_configs.items():
        current = json.dumps(_normalized_config(config), sort_keys=True, separators=(",", ":"))
        if reference is None:
            reference = current
        elif current != reference:
            raise ValueError(f"Incompatible resolved simulation configuration at run {run_id}")


def _load_combined_core() -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    frames = []
    missing_new: set[str] = set()
    for model_id in MODEL_ORDER:
        model_frame, model_missing = _load_model_core(model_id)
        frames.append(model_frame)
        missing_new.update(model_missing)
    if missing_new:
        examples = ", ".join(sorted(missing_new)[:6])
        raise IncompleteDatasetError(
            f"Dataset final belum lengkap: {len(missing_new)} dari 60 run baru belum tersedia "
            f"(historis dipertahankan 60; target total 120). Contoh run yang kurang: {examples}"
        )

    core = pd.concat(frames, ignore_index=True)
    required = {
        "run_id", "experiment_id", "model_id", "seed", "node_num", "node_period",
        "stage1_status", "topology_sha256", "selected_parent_routing_sha256",
        *COMMON_CONFIG_COLUMNS, "pdr_percent", "packet_fate_delivered_count",
        "energy_per_delivered_packet_J", "total_network_energy_J", "avg_e2e_delay_ms",
    }
    missing_columns = required - set(core.columns)
    if missing_columns:
        raise ValueError(f"Core run summaries lack required columns: {sorted(missing_columns)}")
    core["node_num"] = pd.to_numeric(core["node_num"], errors="raise").astype(int)
    core["node_period"] = pd.to_numeric(core["node_period"], errors="raise").astype(int)
    core["seed"] = pd.to_numeric(core["seed"], errors="raise").astype(int)
    core, manifests, resolved_configs, manifest_paths = _attach_run_provenance(core)
    compatibility = _validate_source_compatibility(manifests)
    _validate_cross_cohort_configs(resolved_configs)
    validate_combined_dataset(core)
    return core, compatibility, manifest_paths


def _schedule_frame(experiment_id: str, run_id: str) -> pd.DataFrame:
    path = _experiment_dir(experiment_id) / "runs" / run_id / "tables/packet_fates.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=SCHEDULE_COLUMNS)
    return frame.sort_values(
        ["src_id", "fcnt", "packet_uid"], kind="stable"
    ).reset_index(drop=True)


def _read_resolved_simulator_config(experiment_id: str, run_id: str) -> dict[str, Any]:
    path = _experiment_dir(experiment_id) / "runs" / run_id / "resolved_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Resolved configuration is missing: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    config = record.get("resolved_simulator_config")
    if not isinstance(config, dict):
        raise ValueError(f"Malformed resolved simulator configuration: {path}")
    return config


def _same_value(left: Any, right: Any) -> bool:
    return bool(pd.isna(left) and pd.isna(right)) or left == right


def validate_pairs(core: pd.DataFrame) -> pd.DataFrame:
    """Require paired geometry/routing, shared physics, and offered schedules."""
    frames = {model: core.loc[core["model_id"].eq(model)].copy() for model in MODEL_ORDER}
    ff = frames["ff"].rename(
        columns={c: f"{c}_ff" for c in frames["ff"].columns if c not in PAIR_KEYS}
    )
    aa = frames["aa"].rename(
        columns={c: f"{c}_aa" for c in frames["aa"].columns if c not in PAIR_KEYS}
    )
    pairs = ff.merge(aa, on=PAIR_KEYS, how="outer", validate="one_to_one", indicator=True)
    if len(pairs) != 60 or not pairs["_merge"].eq("both").all():
        raise ValueError("FF/AA grid is not a complete one-to-one 60-cell combined pairing")

    rows = []
    for _, pair in pairs.iterrows():
        key = {name: int(pair[name]) for name in PAIR_KEYS}
        config_match = all(
            _same_value(pair[f"{name}_ff"], pair[f"{name}_aa"])
            for name in COMMON_CONFIG_COLUMNS
        )
        ff_config = _read_resolved_simulator_config(
            str(pair["experiment_id_ff"]), str(pair["run_id_ff"])
        )
        aa_config = _read_resolved_simulator_config(
            str(pair["experiment_id_aa"]), str(pair["run_id_aa"])
        )
        full_ff = {k: v for k, v in ff_config.items() if k not in MODEL_MODE_KEYS | {"run_id"}}
        full_aa = {k: v for k, v in aa_config.items() if k not in MODEL_MODE_KEYS | {"run_id"}}
        config_match = config_match and full_ff == full_aa
        topology_match = bool(
            pair["topology_sha256_ff"]
            and pair["topology_sha256_ff"] == pair["topology_sha256_aa"]
        )
        routing_match = bool(
            pair["selected_parent_routing_sha256_ff"]
            and pair["selected_parent_routing_sha256_ff"]
            == pair["selected_parent_routing_sha256_aa"]
        )
        ff_schedule = _schedule_frame(
            str(pair["experiment_id_ff"]), str(pair["run_id_ff"])
        )
        aa_schedule = _schedule_frame(
            str(pair["experiment_id_aa"]), str(pair["run_id_aa"])
        )
        schedule_match = ff_schedule.equals(aa_schedule)
        ff_hash = hashlib.sha256(ff_schedule.to_csv(index=False).encode()).hexdigest()
        aa_hash = hashlib.sha256(aa_schedule.to_csv(index=False).encode()).hexdigest()
        checks = (config_match, topology_match, routing_match, schedule_match)
        rows.append(
            {
                **key,
                "run_id_ff": pair["run_id_ff"],
                "run_id_aa": pair["run_id_aa"],
                "common_physical_config_match": config_match,
                "topology_match": topology_match,
                "routing_match": routing_match,
                "offered_packet_count_ff": len(ff_schedule),
                "offered_packet_count_aa": len(aa_schedule),
                "offered_schedule_match": schedule_match,
                "offered_schedule_sha256_ff": ff_hash,
                "offered_schedule_sha256_aa": aa_hash,
                "pair_status": "PASS" if all(checks) else "FAIL",
            }
        )
    validation = pd.DataFrame(rows)
    if not validation["pair_status"].eq("PASS").all():
        failed = validation.loc[validation["pair_status"].ne("PASS"), PAIR_KEYS]
        raise ValueError(f"Paired input validation failed: {failed.to_dict(orient='records')[:5]}")
    return validation


def _collect_run_tables(core: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    relay_rows, node_rows, occupancy_rows = [], [], []
    for _, run in core.iterrows():
        model_id, run_id = str(run["model_id"]), str(run["run_id"])
        table_dir = (
            _experiment_dir(str(run["experiment_id"])) / "runs" / run_id / "tables"
        )
        relay = pd.read_csv(table_dir / "relay_duty_cycle.csv")
        relay = relay.loc[relay["entity_type"].eq("relay")].copy()
        if set(relay["entity_id"].astype(str)) != {"R1", "R2"}:
            raise ValueError(f"{run_id} must include relay R1 and R2")
        tables = [
            (relay, relay_rows),
            (pd.read_csv(table_dir / "node_tx_duty_cycle.csv"), node_rows),
            (pd.read_csv(table_dir / "channel_occupancy.csv"), occupancy_rows),
        ]
        for table, target in tables:
            for column in ("model_id", *PAIR_KEYS):
                table[column] = model_id if column == "model_id" else run[column]
            target.append(table)
    return tuple(pd.concat(rows, ignore_index=True) for rows in (relay_rows, node_rows, occupancy_rows))


def _summary_statistics(
    frame: pd.DataFrame, group_columns: list[str], metric: str
) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = pd.to_numeric(group[metric], errors="coerce").dropna()
        n = len(values)
        mean = float(values.mean()) if n else np.nan
        sd = float(values.std(ddof=1)) if n > 1 else np.nan
        sem = sd / np.sqrt(n) if n > 1 else np.nan
        critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else np.nan
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "metric": metric,
                "n": n,
                "mean": mean,
                "sample_sd": sd,
                "ci95_lower": mean - critical * sem if n > 1 else np.nan,
                "ci95_upper": mean + critical * sem if n > 1 else np.nan,
                "ci_method": "two-sided Student-t across seeds; df=n-1",
            }
        )
    return pd.DataFrame(rows)


def _paired_comparison_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        ("pdr_percent", "PDR (percentage points)"),
        ("total_network_energy_J", "Total network energy (J)"),
        ("energy_per_delivered_packet_J", "Energy per delivered packet (J/packet)"),
        ("avg_e2e_delay_ms", "Average end-to-end delay (ms)"),
    )
    rows = []
    for (node_num, node_period), cell in per_run.groupby(
        ["node_num", "node_period"], sort=True
    ):
        wide = cell.pivot(index="seed", columns="model_id")
        for metric, label in metrics:
            if metric not in wide.columns.get_level_values(0):
                continue
            differences = (wide[metric]["aa"] - wide[metric]["ff"]).dropna()
            n = len(differences)
            mean = float(differences.mean()) if n else np.nan
            sd = float(differences.std(ddof=1)) if n > 1 else np.nan
            critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else np.nan
            half_width = critical * sd / np.sqrt(n) if n > 1 else np.nan
            rows.append(
                {
                    "node_num": int(node_num),
                    "node_period_s": int(node_period),
                    "metric": label,
                    "difference_direction": "AA - FF",
                    "n_paired_seeds": n,
                    "mean_difference": mean,
                    "sample_sd": sd,
                    "ci95_lower": mean - half_width if n > 1 else np.nan,
                    "ci95_upper": mean + half_width if n > 1 else np.nan,
                    "ci_method": "paired Student-t 95%; df=n-1",
                }
            )
    return pd.DataFrame(rows)


def _write_interpretation_report(
    per_run: pd.DataFrame, comparison: pd.DataFrame, output_path: Path
) -> None:
    scenario_rows = []
    for (nodes, period), group in per_run.groupby(
        ["node_num", "node_period"], sort=True
    ):
        aa = group.loc[group["model_id"].eq("aa")]
        ff = group.loc[group["model_id"].eq("ff")]
        scenario_rows.append(
            f"| {int(nodes)} | {int(period)} | {len(aa)} | "
            f"{aa['pdr_percent'].mean():.3f} | {ff['pdr_percent'].mean():.3f} | "
            f"{aa['energy_per_delivered_packet_J'].mean():.6f} | "
            f"{ff['energy_per_delivered_packet_J'].mean():.6f} |"
        )
    report = [
        "# Ringkasan APC Load Sweep",
        "",
        f"Dataset final memuat {len(per_run)} run dan {len(per_run) // 2} pasangan FF–AA.",
        "Perbandingan mencakup enam skenario historis dan enam skenario baru; masing-masing memakai lima seed.",
        "",
        "| Node | Interval (s) | Seed/model | PDR AA (%) | PDR FF (%) | Energi/paket AA (J) | Energi/paket FF (J) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        *scenario_rows,
        "",
        "Selisih berpasangan dihitung sebagai AA − FF per seed; interval kepercayaan "
        "95% menggunakan Student-t pada seed, bukan paket individual.",
        "Energi per paket dihitung per run dari energi jaringan terkoreksi dibagi "
        "jumlah paket terkirim; run tanpa paket terkirim dilaporkan sebagai nilai kosong.",
        "Delay adalah rata-rata paket yang berhasil terkirim dan bersifat kondisional "
        "pada keberhasilan delivery; delay lebih rendah tidak ditafsirkan sebagai "
        "perbaikan kualitas bila PDR menurun.",
        "Duty cycle relay menggunakan TX airtime teramati dibagi horizon simulasi; "
        "metrik ini bukan pengukuran kepatuhan regulasi.",
        "",
        f"Jumlah baris uji beda berpasangan: {len(comparison)}.",
    ]
    output_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def _figure_style() -> dict[str, Any]:
    return {
        **FIGURE_STYLE,
        "axes.grid": True,
        "grid.color": "#b0b0b0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }


def _resolved_font_path() -> str:
    prop = font_manager.FontProperties(family="Times New Roman")
    return str(Path(font_manager.findfont(prop, fallback_to_default=False)).resolve())


def _save_figure(figure: Any, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = [base.with_suffix(".png"), base.with_suffix(".svg")]
    figure.savefig(paths[0], dpi=300, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    plt.close(figure)
    return paths


def _plot_metric(stats: pd.DataFrame, metric: str, ylabel: str, base: Path) -> list[Path]:
    selected = stats.loc[stats["metric"].eq(metric)].copy()
    if selected.empty:
        raise ValueError(f"No aggregated rows found for metric={metric!r}")
    expected_nodes = set(NODE_COUNTS)
    for period in PERIODS_S:
        for model in MODEL_ORDER:
            part = selected.loc[
                selected["node_period"].eq(period) & selected["model_id"].eq(model)
            ]
            if (
                len(part) != len(NODE_COUNTS)
                or set(pd.to_numeric(part["node_num"], errors="raise")) != expected_nodes
                or part["node_num"].duplicated().any()
            ):
                raise ValueError(
                    f"Expected one {metric} mean for each node count {NODE_COUNTS}; "
                    f"got {model}/{period}s"
                )
            if "n" in part and not pd.to_numeric(part["n"], errors="raise").eq(5).all():
                raise ValueError(f"{metric} confidence intervals require five seed replicates")
            values = pd.to_numeric(part["mean"], errors="raise")
            bounds = part[["ci95_lower", "ci95_upper"]].apply(pd.to_numeric, errors="raise")
            if not np.isfinite(values).all() or not np.isfinite(bounds.to_numpy()).all():
                raise ValueError(f"Non-finite aggregate values for {model}/{period}s/{metric}")
            if metric == "pdr_percent" and not values.between(0, 100).all():
                raise ValueError("PDR percentage means must be within [0, 100]")

    with plt.rc_context(_figure_style()):
        figure, axis = plt.subplots(figsize=(8, 5))
        for period in PERIODS_S:
            for model in MODEL_ORDER:
                part = selected.loc[
                    selected["node_period"].eq(period) & selected["model_id"].eq(model)
                ].sort_values("node_num")
                line = "-" if model == "aa" else "--"
                axis.errorbar(
                    part["node_num"], part["mean"],
                    yerr=[part["mean"] - part["ci95_lower"],
                          part["ci95_upper"] - part["mean"]],
                    color=PERIOD_COLORS[period], linestyle=line,
                    marker="o" if model == "aa" else "s",
                    markerfacecolor=PERIOD_COLORS[period] if model == "aa" else "white",
                    markeredgewidth=1.2, linewidth=2.0, markersize=7, capsize=2,
                    elinewidth=1.0, capthick=1.0,
                    label=f"{period} s ({model.upper()})",
                )
        axis.set_xlabel("Number of nodes")
        axis.set_ylabel(ylabel)
        axis.set_xticks(NODE_COUNTS)
        if metric == "pdr_percent":
            axis.set_ylim(55, 90)
        # Match the terrain-sweep figure's framed in-axes legend and paired
        # series keys. Choose the least-obstructive corner for each metric.
        handles, labels = axis.get_legend_handles_labels()
        order = [0, 2, 4, 1, 3, 5]
        legend_loc = "lower left" if metric == "pdr_percent" else "upper right"
        axis.legend(
            [handles[index] for index in order],
            [labels[index] for index in order],
            ncol=2,
            loc=legend_loc,
            frameon=True,
            framealpha=0.85,
            fancybox=True,
        )
        figure.tight_layout()
        return _save_figure(figure, base)


def _plot_relay_duty(stats: pd.DataFrame, base: Path) -> list[Path]:
    with plt.rc_context(_figure_style()):
        figure, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        for axis, relay_id in zip(axes, ("R1", "R2")):
            selected = stats.loc[stats["entity_id"].eq(relay_id)]
            for period in PERIODS_S:
                for model in MODEL_ORDER:
                    part = selected.loc[
                        selected["node_period"].eq(period) & selected["model_id"].eq(model)
                    ].sort_values("node_num")
                    axis.errorbar(
                        part["node_num"], part["mean"],
                        yerr=[part["mean"] - part["ci95_lower"],
                              part["ci95_upper"] - part["mean"]],
                        color=PERIOD_COLORS[period],
                        linestyle="-" if model == "aa" else "--", marker="o",
                        markerfacecolor=PERIOD_COLORS[period] if model == "aa" else "white",
                        markeredgewidth=1.2, linewidth=1.6, markersize=5, capsize=2,
                    )
            axis.set_title(f"Relay {relay_id}")
            axis.set_xlabel("Number of nodes")
            axis.set_xticks(NODE_COUNTS)
        axes[0].set_ylabel("Relay TX duty-cycle utilization (%)")
        handles = [
            Line2D(
                [0], [0], color=PERIOD_COLORS[period],
                linestyle="-" if model == "AA" else "--", marker="o",
                markerfacecolor=PERIOD_COLORS[period] if model == "AA" else "white",
                label=f"{period} s ({model})",
            )
            for period in PERIODS_S for model in ("AA", "FF")
        ]
        figure.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.98))
        figure.tight_layout(rect=(0, 0, 1, 0.87))
        return _save_figure(figure, base)


def run_postprocessing(output_dir: str | Path | None = None) -> dict[str, Any]:
    """Validate the complete historical/new paired matrix before writing outputs."""
    output_dir = Path(output_dir) if output_dir else ARTIFACT_ROOT / "analysis"
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    removal_audit_path, removal_audit = _removed_15s_audit()
    cores, source_compatibility, source_manifest_paths = _load_combined_core()
    pairs = validate_pairs(cores)
    schedule_hash_by_run: dict[str, str] = {}
    for pair in pairs.itertuples():
        schedule_hash_by_run[str(pair.run_id_ff)] = str(pair.offered_schedule_sha256_ff)
        schedule_hash_by_run[str(pair.run_id_aa)] = str(pair.offered_schedule_sha256_aa)
    cores["offered_schedule_sha256"] = cores["run_id"].map(schedule_hash_by_run)
    if cores["offered_schedule_sha256"].isna().any():
        raise ValueError("One or more runs lack a matched offered-schedule fingerprint")

    per_run = cores[
        [
            "run_id", "experiment_id", "resolved_experiment_profile", "model_id",
            "node_num", "node_period", "seed", "terrain_delta_h", "source_commit",
            "source_fingerprint", "config_hash", "source_cohort", "stage1_status",
            "resolved_config_file_sha256", "resolved_config_path",
            "source_fingerprint_manifest", "offered_schedule_sha256",
            "topology_sha256", "selected_parent_routing_sha256", "pdr_percent",
            "energy_per_delivered_packet_J", "total_network_energy_J",
            "packet_fate_delivered_count", "avg_e2e_delay_ms",
        ]
    ].rename(columns={"resolved_experiment_profile": "experiment_profile"}).copy()
    validate_combined_dataset(per_run)
    delivered = pd.to_numeric(per_run["packet_fate_delivered_count"], errors="raise")
    per_run.loc[delivered.eq(0), "energy_per_delivered_packet_J"] = np.nan

    font_path = _resolved_font_path()
    relay_runs, node_duty, channel_occupancy = _collect_run_tables(cores)
    metric_stats = pd.concat(
        [
            _summary_statistics(
                per_run, ["model_id", "node_period", "node_num"], metric
            )
            for metric in METRICS
        ],
        ignore_index=True,
    )
    relay_stats = _summary_statistics(
        relay_runs,
        ["model_id", "node_period", "node_num", "entity_id"],
        "relay_duty_cycle_percent",
    )
    if not metric_stats.loc[metric_stats["metric"].eq("pdr_percent"), "n"].eq(5).all():
        raise ValueError("PDR requires all five seed observations in every final cell")
    if not relay_stats["n"].eq(5).all():
        raise ValueError("Relay duty-cycle statistics have cells with fewer than five seeds")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir, reports_dir = output_dir / "figures", output_dir / "reports"
    figures_dir.mkdir()
    reports_dir.mkdir()
    atomic_write_csv(output_dir / "combined_dataset.csv", per_run)
    paired_statistics = _paired_comparison_summary(per_run)
    atomic_write_csv(output_dir / "comparison_summary.csv", metric_stats)
    atomic_write_csv(output_dir / "paired_statistics.csv", paired_statistics)
    atomic_write_csv(output_dir / "paired_run_validation.csv", pairs)
    atomic_write_csv(output_dir / "statistics_mean_sd_ci95.csv", metric_stats)
    atomic_write_csv(output_dir / "relay_duty_per_run.csv", relay_runs)
    atomic_write_csv(output_dir / "relay_duty_statistics.csv", relay_stats)
    atomic_write_csv(output_dir / "node_tx_duty_cycle.csv", node_duty)
    atomic_write_csv(output_dir / "channel_occupancy.csv", channel_occupancy)

    figures = []
    for metric, ylabel, stem in (
        ("pdr_percent", "Network packet delivery ratio (%)", "figure_01_network_pdr_vs_nodes"),
        (
            "energy_per_delivered_packet_J",
            "Network energy per delivered packet (J/packet)",
            "figure_02_network_energy_per_delivered_packet_vs_nodes",
        ),
    ):
        selected = metric_stats.loc[metric_stats["metric"].eq(metric)]
        figures.extend(_plot_metric(selected, metric, ylabel, figures_dir / stem))
    figures.extend(
        _plot_relay_duty(relay_stats, figures_dir / "figure_03_relay_duty_cycle_vs_nodes")
    )
    _write_interpretation_report(
        per_run, paired_statistics, reports_dir / "interpretation_id.md"
    )
    input_paths = [
        resolve_master_table_path(
            _experiment_dir(experiment_id), "all_runs_core_summary.csv"
        )
        for experiment_id in (*LEGACY_EXPERIMENT_IDS.values(), *(x for g in RUN_GROUP_IDS.values() for x in g))
    ]
    input_paths += source_manifest_paths
    input_paths += [
        _experiment_dir(str(row.experiment_id)) / "runs" / str(row.run_id) / "resolved_config.json"
        for row in per_run.itertuples()
    ]
    input_paths += [EXPERIMENT_DIR / "configs" / name for names in RUN_CONFIG_FILES.values() for name in names]
    input_paths.append(removal_audit_path)
    input_paths = sorted(set(input_paths))
    manifest = {
        "analysis_version": "apc_load_sweep_combined_reporting_v2",
        "created_at": utc_now(),
        "status": "PASS",
        "experiment_name": "apc_load_sweep",
        "source_commit_available": source_compatibility["source_commit_available"],
        "source_compatibility": source_compatibility,
        "models": list(MODEL_ORDER),
        "scenarios": [
            {"node_num": nodes, "node_period_s": period}
            for nodes, period in sorted(COMBINED_SCENARIOS)
        ],
        "application_periods_s": list(PERIODS_S),
        "seeds": list(SEEDS),
        "validated_runs": len(per_run),
        "validated_pairs": len(pairs),
        "historical_run_count": int(per_run["source_cohort"].eq("historical").sum()),
        "new_run_count": int(per_run["source_cohort"].eq("new").sum()),
        "excluded_15s_run_count": int(removal_audit["run_count"]),
        "removed_15s_runs_audit": "../audit/removed_15s_runs.json",
        "statistics": "mean, sample SD, two-sided Student-t 95% CI across five seeds (df=4)",
        "pairing_checks": [
            "common physical configuration", "topology hash", "routing hash",
            "packet ID and FCnt", "birth and offered timestamps",
        ],
        "metric_definitions": {
            "energy_per_delivered_packet_J": "per-run corrected network energy divided by delivered packets",
            "relay_duty_cycle_percent": "observed relay TX airtime / simulation horizon; not a compliance claim",
            "channel_busy_union_s": "union of simulated TX intervals per channel; not receiver-sensed occupancy",
            "avg_e2e_delay_ms": "average delay among successfully delivered packets; conditional metric",
        },
        "font_family": "Times New Roman",
        "font_path": font_path,
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in input_paths
        },
        "figure_files": [str(path.relative_to(output_dir)) for path in figures],
        "report_files": ["reports/interpretation_id.md"],
    }
    atomic_write_json(output_dir / "analysis_manifest.json", manifest)
    return {
        "status": "PASS",
        "analysis_dir": str(output_dir),
        "validated_runs": len(per_run),
        "validated_pairs": len(pairs),
        "figures": [str(path) for path in figures],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reporting-only APC load-sweep analysis")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    print(json.dumps(run_postprocessing(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
