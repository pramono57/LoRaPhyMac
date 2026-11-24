from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from loraphymac.experiments.apc_terrain_sweep.postprocessing import (
    _validate_completed_sweep,
)
from loraphymac.experiments.apc_terrain_profile_diagnostic.analysis import (
    build_diagnostic_analysis,
)
from loraphymac.experiments.common.multiseed_pipeline import (
    build_experiment_grid,
    file_sha256,
    load_experiment_spec,
    validate_seed_completeness,
)
from loraphymac.experiments.common.outputs import ExperimentArtifactPaths
from loraphymac.experiments.common.scientific_output_contract import (
    output_contract_for,
    preflight_output_violations,
    required_table_keys,
)
from loraphymac.experiments.common.contract_registry import experiment_run_tables
from loraphymac.experiments.common.scientific_output_manager import ScientificOutputManager


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "src/loraphymac/experiments/apc_terrain_sweep/configs/diagnostic_v1.json"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _diagnostic_cohort(tmp_path: Path) -> ExperimentArtifactPaths:
    spec = load_experiment_spec(CONFIG)
    grid = build_experiment_grid(spec)
    cohort_id = "terrain_sweep_v1_r01_20261124"
    root = tmp_path / "APC_TERRAIN_SWEEP" / cohort_id
    paths = ExperimentArtifactPaths(root, "sweep_cohort_v1")
    paths.runs.mkdir(parents=True)
    paths.aggregate.mkdir()
    run_statuses = {}
    all_rows = []
    for index, run in enumerate(grid):
        run_id = run["run_id"].replace(spec.experiment_id, cohort_id, 1)
        config = dict(run["scientific_config"])
        # Cohort identity is incorporated in the pipeline-generated run ID, not the science config.
        new_id = run_id
        run_dir = paths.runs / new_id
        table_path = run_dir / "core_summary.csv"
        table_path.parent.mkdir(parents=True)
        seed = int(run["seed"])
        metrics = {
            "delivery_by_horizon_percent": 80.0 + index,
            "terminal_loss_percent": 20.0 - index,
            "censored_fraction_percent": 0.0,
            "served_node_ratio_95_percent": 70.0 + index,
            "avg_e2e_delay_ms": 100.0 + 2 * index,
            "total_node_energy_J": 1.0 + index,
            "total_relay_energy_J": 2.0 + index,
            "total_gateway_energy_J": 3.0 + index,
            "total_network_energy_J": 6.0 + 3 * index,
            "total_constrained_energy_J": 6.0 + 3 * index,
            "energy_per_delivered_packet_J": 0.1 + index / 100,
            "energy_per_delivered_packet_constrained_J": 0.1 + index / 100,
            "avg_final_node_ptx": 10.0 + index,
            "avg_final_relay_ptx": 20.0 + index,
            "avg_final_node_decision_margin": 3.0,
            "avg_final_node_apc_rx_margin": 3.0,
            "avg_final_relay_decision_margin": 3.0,
            "avg_final_relay_apc_rx_margin": 3.0,
            "apc_total_tx": 1 + index,
            "apc_total_rx": 1 + index,
            "apc_total_drop": 0,
            "apc_apply_success_percent": 100.0,
            "apc_down_node_success_percent": 100.0,
            "apc_down_relay_success_percent": 100.0,
            "ptx_update_count": 1 + index,
            "packet_fate_conservation_delta": 0,
            "collision_outcome_conservation_delta": 0,
        }
        row = {"run_id": new_id, "seed": seed, **metrics}
        pd.DataFrame([row]).to_csv(table_path, index=False)
        factors = dict(run["factors"])
        resolved = {
            "run_id": new_id,
            "seed": seed,
            "factors": factors,
            "scientific_config": {**config, "seed": seed},
        }
        _write_json(run_dir / "resolved_config.json", resolved)
        _write_json(run_dir / "status.json", {"run_id": new_id, "status": "PASS"})
        table_record = {
            "path": "core_summary.csv",
            "size_bytes": table_path.stat().st_size,
            "sha256": file_sha256(table_path),
        }
        _write_json(
            run_dir / "scientific_output_manifest.json",
            {
                "status": "COMPLETED",
                "tables": {"core_summary_df": table_record},
                "artifacts": {},
                "compatibility_artifacts": {},
                "required_table_keys": ["core_summary_df"],
                "required_artifacts": ["core_summary.csv"],
            },
        )
        all_rows.append(row)
        run_statuses[new_id] = "PASS"
    pd.DataFrame(all_rows).to_csv(paths.aggregate / "core_metrics_long.csv", index=False)
    pd.DataFrame(all_rows).to_csv(paths.aggregate / "all_runs_core_summary.csv", index=False)
    _write_json(
        paths.status,
        {
            "status": "COMPLETE",
            "seed_grid_complete": True,
            "passed_run_count_on_disk": len(grid),
            "requested_run_count": len(grid),
            "experiment_id": cohort_id,
        },
    )
    _write_json(
        paths.manifest,
        {
            "output_layout_version": "sweep_cohort_v1",
            "cohort_id": cohort_id,
            "git_head": "test-source-head",
            "source_fingerprint_sha256": "test-source-fingerprint",
            "expected_run_count": len(grid),
            "run_ids": list(run_statuses),
            "run_statuses": run_statuses,
            "parameter_grid": {
                "factors": {key: list(value) for key, value in spec.factors.items()},
                "seeds": list(spec.seeds),
                "run_matrix": [dict(row) for row in spec.run_matrix],
            },
            "spec": spec.serializable(),
        },
    )
    # Run IDs change with cohort ID, and the selected grid hash must track that identity.
    manifest = json.loads(paths.manifest.read_text())
    serialized = manifest["spec"]
    serialized["experiment_id"] = cohort_id
    manifest["spec"] = serialized
    _write_json(paths.manifest, manifest)
    return paths


def test_diagnostic_validator_accepts_exact_explicit_matrix_and_hashes(tmp_path):
    paths = _diagnostic_cohort(tmp_path)
    status = _validate_completed_sweep(paths)
    assert status["passed_run_count_on_disk"] == 10


def test_seed_completeness_uses_explicit_sparse_run_matrix_without_cartesian_expansion():
    spec = load_experiment_spec(CONFIG)
    grid = build_experiment_grid(spec)
    frame = pd.DataFrame(
        [
            {"run_id": run["run_id"], "seed": run["seed"], **run["factors"]}
            for run in grid
        ]
    )
    report = validate_seed_completeness(
        frame,
        ["node_ptx_mode", "relay_ptx_mode", "terrain_delta_h"],
        spec.seeds,
        expected_run_matrix=spec.run_matrix,
    )
    assert len(report) == 2
    assert report["complete"].all()
    assert set(report["expected_seed_count"]) == {5}

    incomplete = validate_seed_completeness(
        frame.iloc[:-1],
        ["node_ptx_mode", "relay_ptx_mode", "terrain_delta_h"],
        spec.seeds,
        expected_run_matrix=spec.run_matrix,
    )
    assert incomplete["complete"].sum() == 1
    assert (~incomplete["complete"]).sum() == 1


@pytest.mark.parametrize(
    "profile",
    [
        "apc_terrain_profile_diagnostic_v1",
        "apc_terrain_profile_full_sweep_v1",
        "apc_terrain_channel_occupancy_pilot_v1",
    ],
)
def test_terrain_diagnostic_profiles_have_matching_scientific_output_contracts(profile, tmp_path):
    config = {"experiment_profile": profile, "persistence_profile": "scientific"}
    contract = output_contract_for(config)
    assert contract is not None
    assert contract.required_additional_tables == frozenset(experiment_run_tables(config))
    assert preflight_output_violations(config) == []
    tables = {key: pd.DataFrame() for key in required_table_keys(config)}
    selection = ScientificOutputManager(tmp_path / profile).select_tables(config, tables)
    assert selection.manager_writes is True
    assert set(selection.table_keys) == set(required_table_keys(config))


def test_diagnostic_validator_rejects_matrix_missing_row(tmp_path):
    paths = _diagnostic_cohort(tmp_path)
    manifest = json.loads(paths.manifest.read_text())
    manifest["parameter_grid"]["run_matrix"].pop()
    _write_json(paths.manifest, manifest)
    with pytest.raises(ValueError, match="exact ten-row"):
        _validate_completed_sweep(paths)


def test_diagnostic_validator_rejects_failed_status_and_corrupt_table(tmp_path):
    paths = _diagnostic_cohort(tmp_path)
    run_id = paths.runs.iterdir().__next__().name
    _write_json(paths.runs / run_id / "status.json", {"run_id": run_id, "status": "FAILED"})
    with pytest.raises(ValueError, match="not complete and passing"):
        _validate_completed_sweep(paths)

    paths = _diagnostic_cohort(tmp_path / "corrupt")
    run_id = paths.runs.iterdir().__next__().name
    (paths.runs / run_id / "core_summary.csv").write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _validate_completed_sweep(paths)


def test_diagnostic_analysis_publishes_hashed_tables_figures_and_reuses_outputs(tmp_path):
    paths = _diagnostic_cohort(tmp_path)
    first = build_diagnostic_analysis(experiment_dir=paths.experiment_dir)
    assert first["status"] == "COMPLETE"
    manifest_path = paths.analysis_manifest
    analysis = json.loads(manifest_path.read_text())
    assert analysis["status"] == "COMPLETE"
    assert analysis["evidence_class"].startswith("bounded diagnostic")
    assert analysis["aggregation"]["interpretation"] == "descriptive only; no significance claim"
    assert len(analysis["run_ids"]) == 10
    assert len(analysis["outputs"]) == 5
    for record in analysis["outputs"]:
        output = paths.experiment_dir / record["path"]
        assert output.is_file()
        assert output.stat().st_size == record["size_bytes"]
        assert file_sha256(output) == record["sha256"]
    summary = pd.read_csv(paths.experiment_dir / analysis["tables"][1])
    assert set(summary["metric_name"]) >= {
        "delivery_by_horizon_percent",
        "avg_e2e_delay_ms",
        "total_network_energy_J",
    }
    assert (summary["n"] == 5).all()
    before = {
        record["path"]: (
            file_sha256(paths.experiment_dir / record["path"]),
            (paths.experiment_dir / record["path"]).stat().st_mtime_ns,
        )
        for record in analysis["outputs"]
    }
    second = build_diagnostic_analysis(experiment_dir=paths.experiment_dir)
    assert second["status"] == "COMPLETE"
    assert second["reused"] is True
    after = {
        record["path"]: (
            file_sha256(paths.experiment_dir / record["path"]),
            (paths.experiment_dir / record["path"]).stat().st_mtime_ns,
        )
        for record in analysis["outputs"]
    }
    assert before == after
    cohort = json.loads(paths.manifest.read_text())
    assert cohort["analysis_status"] == "COMPLETE"
    assert cohort["analysis_manifest_path"] == "analysis/analysis_manifest.json"
