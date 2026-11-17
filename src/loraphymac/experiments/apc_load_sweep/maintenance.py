"""Auditable pruning and plan-manifest helpers for APC Load Sweep."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loraphymac.experiments.common.multiseed_pipeline import (
    atomic_write_json,
    build_experiment_grid,
    file_sha256,
    load_experiment_spec,
)
from loraphymac.paths import repository_root


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = repository_root(EXPERIMENT_DIR)
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/experiments/apc_load_sweep"
AUDIT_PATH = ARTIFACT_ROOT / "audit/removed_15s_runs.json"
ACTIVE_MANIFEST_PATH = ARTIFACT_ROOT / "active_manifest.json"
LEGACY_IDS = {"ff": "apc_load_sweep_ff_r01", "aa": "apc_load_sweep_aa_r01"}
FINAL_CONFIGS = (
    "ff_wide_nodes_90s.json",
    "ff_node150_periods.json",
    "aa_wide_nodes_90s.json",
    "aa_node150_periods.json",
)
EXPECTED_SCENARIOS = {(50, 90), (100, 90), (150, 30), (150, 60), (150, 90), (200, 90)}
EXPECTED_SEEDS = (30, 36, 42, 48, 56)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def inventory_15s_runs(artifact_root: Path = ARTIFACT_ROOT) -> list[dict[str, Any]]:
    """Find only run directories whose resolved simulator config says 15 s."""
    records = []
    for model_id, experiment_id in LEGACY_IDS.items():
        runs_dir = artifact_root / experiment_id / "runs"
        for run_dir in sorted(runs_dir.iterdir() if runs_dir.exists() else []):
            config_path = run_dir / "resolved_config.json"
            status_path = run_dir / "status.json"
            if not config_path.is_file():
                continue
            resolved = json.loads(config_path.read_text(encoding="utf-8"))
            simulator = resolved.get("resolved_simulator_config", {})
            period = simulator.get("node_period", simulator.get("node_period_s"))
            if period != 15:
                continue
            if not status_path.is_file():
                raise ValueError(f"15-second run has no status file: {run_dir}")
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") != "PASS":
                raise ValueError(f"15-second run is not PASS; refusing prune: {run_dir}")
            if run_dir.is_symlink() or not run_dir.resolve().is_relative_to(runs_dir.resolve()):
                raise ValueError(f"Unsafe run directory path: {run_dir}")
            files = []
            for path in sorted(run_dir.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"Symlink in run output; refusing prune: {path}")
                if path.is_file():
                    files.append(
                        {
                            "path": path.relative_to(PROJECT_ROOT).as_posix(),
                            "size_bytes": path.stat().st_size,
                            "sha256": file_sha256(path),
                        }
                    )
            records.append(
                {
                    "model_id": model_id,
                    "experiment_id": experiment_id,
                    "run_id": resolved["run_id"],
                    "run_directory": run_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "status": status["status"],
                    "seed": int(resolved["seed"]),
                    "parameters": {
                        "node_num": int(simulator["node_num"]),
                        "node_period_s": int(period),
                        "terrain_delta_h": simulator.get("terrain_delta_h"),
                        "sim_time_s": simulator.get("sim_time"),
                        "energy_accounting_mode": simulator.get("energy_accounting_mode"),
                    },
                    "file_count": len(files),
                    "files": files,
                }
            )
    if len(records) != 30:
        raise ValueError(f"Expected exactly 30 resolved 15-second runs; found {len(records)}")
    expected = {
        (model, nodes, seed)
        for model in ("ff", "aa")
        for nodes in (50, 100, 200)
        for seed in EXPECTED_SEEDS
    }
    actual = {
        (r["model_id"], r["parameters"]["node_num"], r["seed"])
        for r in records
    }
    if actual != expected or len(actual) != len(records):
        raise ValueError("Resolved 15-second run inventory differs from the audited grid")
    return records


def _analysis_inventory(artifact_root: Path) -> list[dict[str, Any]]:
    analysis_dir = artifact_root / "analysis_r01"
    inventory = []
    for path in sorted(analysis_dir.rglob("*")) if analysis_dir.exists() else []:
        if path.is_file():
            inventory.append(
                {
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": file_sha256(path),
                    "disposition": "PRESERVE_MIXED_INTERVAL_ANALYSIS",
                    "reason": "Existing tables/figures combine 15, 30 and 60 second intervals; selective deletion would require rewriting historical analysis.",
                }
            )
    return inventory


def audit_and_prune_15s_runs(
    *, artifact_root: Path = ARTIFACT_ROOT, apply: bool = False
) -> dict[str, Any]:
    """Write the full hash audit first; remove only the verified 15-second run dirs."""
    audit_path = artifact_root / "audit/removed_15s_runs.json"
    records = inventory_15s_runs(artifact_root)
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "DELETED_AFTER_HASH_AUDIT":
            raise FileExistsError(f"15-second runs already pruned; preserving audit: {audit_path}")
        if audit.get("runs") != records:
            raise ValueError("Existing audit does not exactly match current 15-second run files/hashes")
    else:
        audit = {
            "schema_version": "apc_load_sweep_pruned_runs_v1",
            "created_at": _utc_now(),
            "scope": "Only raw per-run directories with resolved node_period=15 seconds.",
            "status": "AUDITED_PENDING_DELETE" if apply else "DRY_RUN_NOT_DELETED",
            "run_count": len(records),
            "file_count": sum(r["file_count"] for r in records),
            "bytes_total": sum(f["size_bytes"] for r in records for f in r["files"]),
            "runs": records,
            "analysis_outputs": {
                "removed": [],
                "preserved_mixed": _analysis_inventory(artifact_root),
            },
            "historical_manifests": "Unmodified in the legacy r01 experiment directories.",
        }
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(audit_path, audit)
    if not apply:
        return audit

    audit["status"] = "AUDITED_PENDING_DELETE"
    atomic_write_json(audit_path, audit)
    verified = json.loads(audit_path.read_text(encoding="utf-8"))
    if verified.get("run_count") != 30 or verified.get("file_count") != audit["file_count"]:
        raise ValueError("Pre-deletion audit read-back verification failed")
    for run in records:
        run_dir = PROJECT_ROOT / run["run_directory"]
        for item in run["files"]:
            file_path = PROJECT_ROOT / item["path"]
            if not file_path.is_file() or file_sha256(file_path) != item["sha256"]:
                raise ValueError(f"File changed after audit; refusing deletion: {file_path}")
        shutil.rmtree(run_dir)
    if any((PROJECT_ROOT / r["run_directory"]).exists() for r in records):
        raise RuntimeError("One or more audited run directories remain after deletion")
    audit["status"] = "DELETED_AFTER_HASH_AUDIT"
    audit["deleted_at"] = _utc_now()
    atomic_write_json(audit_path, audit)
    return audit


def build_active_manifest(artifact_root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    """Build the six-cell active plan without modifying historical manifests."""
    active_manifest_path = artifact_root / "active_manifest.json"
    audit_path = artifact_root / "audit/removed_15s_runs.json"
    if active_manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite active manifest: {active_manifest_path}")
    if not audit_path.is_file():
        raise FileNotFoundError(f"15-second run audit is required first: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "DELETED_AFTER_HASH_AUDIT":
        raise ValueError("Active manifest requires completed 15-second pruning audit")
    planned_runs = []
    config_hashes = {}
    for config_name in FINAL_CONFIGS:
        config_path = EXPERIMENT_DIR / "configs" / config_name
        config_hashes[str(config_path.relative_to(PROJECT_ROOT))] = file_sha256(config_path)
        spec = load_experiment_spec(config_path)
        for run in build_experiment_grid(spec):
            nodes = int(run["scientific_config"]["node_num"])
            period = int(run["scientific_config"]["node_period"])
            if (nodes, period) not in EXPECTED_SCENARIOS:
                raise ValueError(f"Unexpected final scenario in {config_name}: {(nodes, period)}")
            planned_runs.append(
                {
                    "model_id": spec.run_annotations["model_id"],
                    "experiment_id": spec.experiment_id,
                    "run_id": run["run_id"],
                    "node_num": nodes,
                    "node_period_s": period,
                    "seed": int(run["seed"]),
                }
            )
    keys = [(r["model_id"], r["node_num"], r["node_period_s"], r["seed"]) for r in planned_runs]
    expected = {
        (model, nodes, period, seed)
        for model in ("ff", "aa")
        for nodes, period in EXPECTED_SCENARIOS
        for seed in EXPECTED_SEEDS
    }
    if len(planned_runs) != 60 or len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Final run plan must contain exactly 60 unique model/scenario/seed cells")

    retained = []
    for model, experiment_id in LEGACY_IDS.items():
        for run_dir in sorted((artifact_root / experiment_id / "runs").glob("*")):
            config_path, status_path = run_dir / "resolved_config.json", run_dir / "status.json"
            if not config_path.is_file() or not status_path.is_file():
                continue
            resolved = json.loads(config_path.read_text(encoding="utf-8"))
            simulator = resolved.get("resolved_simulator_config", {})
            if simulator.get("node_period", simulator.get("node_period_s")) == 15:
                continue
            if json.loads(status_path.read_text(encoding="utf-8")).get("status") != "PASS":
                raise ValueError(f"Retained legacy run is not PASS: {run_dir}")
            retained.append(
                {
                    "model_id": model,
                    "run_id": resolved["run_id"],
                    "node_num": int(simulator["node_num"]),
                    "node_period_s": int(simulator.get("node_period", simulator.get("node_period_s"))),
                    "seed": int(resolved["seed"]),
                    "analysis_disposition": "RETAINED_HISTORICAL_OUTSIDE_FINAL_MATRIX",
                }
            )
    if len(retained) != 60:
        raise ValueError(f"Expected 60 retained legacy runs; found {len(retained)}")
    active = {
        "schema_version": "apc_load_sweep_active_manifest_v1",
        "experiment_name": "apc_load_sweep",
        "created_at": _utc_now(),
        "status": "READY_TO_RUN",
        "final_scenarios": [
            {"node_num": nodes, "node_period_s": period}
            for nodes, period in sorted(EXPECTED_SCENARIOS)
        ],
        "seeds": list(EXPECTED_SEEDS),
        "models": ["ff", "aa"],
        "expected_new_run_count": 60,
        "expected_paired_cell_count": 30,
        "preserved_legacy_run_count": len(retained),
        "preserved_legacy_runs": retained,
        "planned_analysis_run_ids": planned_runs,
        "removed_run_audit": "audit/removed_15s_runs.json",
        "historical_manifests": [
            f"{experiment_id}/experiment_manifest.json"
            for experiment_id in LEGACY_IDS.values()
        ],
        "configs": config_hashes,
        "output_namespace": str(artifact_root.relative_to(PROJECT_ROOT)),
        "analysis_output": "analysis/",
        "git_head": subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_worktree_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"], text=True
            ).strip()
        ),
    }
    atomic_write_json(active_manifest_path, active)
    return active


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit/prune retired 15-second APC load-sweep runs")
    parser.add_argument("action", choices=("audit", "prune", "write-active-manifest"))
    args = parser.parse_args()
    if args.action == "audit":
        result = audit_and_prune_15s_runs(apply=False)
    elif args.action == "prune":
        result = audit_and_prune_15s_runs(apply=True)
    else:
        result = build_active_manifest()
    print(json.dumps({k: v for k, v in result.items() if k not in {"runs", "planned_analysis_run_ids", "preserved_legacy_runs"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
