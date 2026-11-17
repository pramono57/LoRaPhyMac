import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from loraphymac.experiments.common import multiseed_pipeline as pipeline
from loraphymac.experiments.common.outputs import ExperimentArtifactPaths
from loraphymac import SimulationResult


def make_spec(tmp_path, seeds=(11, 22, 33)):
    return pipeline.ExperimentSpec(
        experiment_id="unit-p2",
        output_dir=tmp_path,
        base_config={"sim_time": 10, "node_period": 60},
        factors={"node_num": [10, 20]},
        seeds=list(seeds),
        confidence_level=0.95,
    )


def test_scientific_master_policy_is_opt_in_and_validated(tmp_path):
    payload = {
        "experiment_id": "scientific-master-policy",
        "output_dir": str(tmp_path / "out"),
        "base_config": {
            "sim_time": 10,
            "persistence_profile": "scientific",
        },
        "factors": {"node_num": [10]},
        "seeds": [1],
        "scientific_master_policy": {
            "materialization": "deferred",
            "table_keys": ["core_summary_df", "relay_bottleneck_metrics_df"],
            "compression": "inherit",
        },
    }
    path = tmp_path / "scientific-master.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    spec = pipeline.load_experiment_spec(path)
    assert spec.scientific_master_policy == payload["scientific_master_policy"]

    payload.pop("scientific_master_policy")
    path.write_text(json.dumps(payload), encoding="utf-8")
    legacy = pipeline.load_experiment_spec(path)
    assert legacy.scientific_master_policy is None
    assert legacy.master_table_mode == "full"


@pytest.mark.parametrize(
    ("persistence", "policy", "message"),
    [
        (None, {"materialization": "deferred", "table_keys": ["core_summary_df", "relay_bottleneck_metrics_df"], "compression": "inherit"}, "requires persistence_profile"),
        ("scientific", {"materialization": "deferred", "table_keys": ["core_summary_df"], "compression": "inherit"}, "must include core_summary_df and relay_bottleneck_metrics_df"),
        ("scientific", {"materialization": "later", "table_keys": ["core_summary_df", "relay_bottleneck_metrics_df"], "compression": "inherit"}, "must be deferred or on_complete"),
    ],
)
def test_scientific_master_policy_rejects_invalid_combinations(
    tmp_path, persistence, policy, message
):
    payload = {
        "experiment_id": "invalid-scientific-master",
        "output_dir": str(tmp_path / "out"),
        "base_config": {"sim_time": 10},
        "factors": {"node_num": [10]},
        "seeds": [1],
        "scientific_master_policy": policy,
    }
    if persistence is not None:
        payload["base_config"]["persistence_profile"] = persistence
    path = tmp_path / "invalid-scientific-master.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        pipeline.load_experiment_spec(path)


def test_scientific_master_frames_select_only_requested_canonical_tables(tmp_path):
    run_dir = tmp_path / "runs" / "run-1"
    tables = run_dir / "tables"
    tables.mkdir(parents=True)
    frames = {
        "core_summary_df": pd.DataFrame([{"run_id": "run-1", "pdr_percent": 90.0}]),
        "relay_bottleneck_metrics_df": pd.DataFrame(columns=["run_id", "entity_id", "queue_mean"]),
        "relay_duty_cycle_df": pd.DataFrame([{"run_id": "run-1", "duty": 0.2}]),
    }
    records = {}
    required_artifacts = []
    for key, frame in frames.items():
        filename = f"{key.removesuffix('_df')}.csv"
        path = tables / filename
        frame.to_csv(path, index=False)
        relative = f"tables/{filename}"
        record = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": pipeline.file_sha256(path),
        }
        records[key] = record
        required_artifacts.append(relative)
    (run_dir / "scientific_output_manifest.json").write_text(
        json.dumps({
            "status": "COMPLETED",
            "tables": records,
            "artifacts": {},
            "compatibility_artifacts": {},
            "required_table_keys": list(records),
            "required_artifacts": sorted(required_artifacts),
        }),
        encoding="utf-8",
    )
    result = pipeline._scientific_master_frames(
        tmp_path,
        [run_dir],
        ["core_summary_df", "relay_bottleneck_metrics_df"],
    )
    assert set(result) == {
        "all_runs_core_summary.csv",
        "per_run_relay_bottleneck_metrics.csv",
    }
    assert result["all_runs_core_summary.csv"].loc[0, "pdr_percent"] == 90.0
    assert list(result["per_run_relay_bottleneck_metrics.csv"].columns) == [
        "run_id", "entity_id", "queue_mean"
    ]
    assert result["per_run_relay_bottleneck_metrics.csv"].empty
    (tables / "relay_bottleneck_metrics.csv").write_text("tampered\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        pipeline._scientific_master_frames(
            tmp_path,
            [run_dir],
            ["core_summary_df", "relay_bottleneck_metrics_df"],
        )


def test_manual_scientific_master_rebuild_rejects_incomplete_expected_matrix(tmp_path):
    spec = pipeline.ExperimentSpec(
        experiment_id="deferred-cohort",
        output_dir=tmp_path / "artifacts",
        base_config={"sim_time": 10, "persistence_profile": "scientific"},
        factors={"node_num": [10]},
        seeds=[1, 2],
        scientific_master_policy={
            "materialization": "deferred",
            "table_keys": ["core_summary_df", "relay_bottleneck_metrics_df"],
            "compression": "inherit",
        },
    )
    experiment_dir = pipeline.initialize_experiment(spec, tmp_path)
    expected = pipeline.build_experiment_grid(spec)
    partial_run = ExperimentArtifactPaths(experiment_dir).runs / expected[0]["run_id"]
    partial_run.mkdir(parents=True)
    with pytest.raises(ValueError, match="Expected run matrix mismatch"):
        pipeline.rebuild_scientific_master_tables(experiment_dir, project_root=tmp_path)
    assert not ExperimentArtifactPaths(experiment_dir).scientific_master.exists()


def test_selective_master_accepts_contract_not_applicable_wang_table(tmp_path):
    from loraphymac.experiments.apc_compare_model_terrain_sweep.contract import (
        APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
    )

    run_dir = tmp_path / "runs" / "ff-run"
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True)
    frames = {
        "core_summary_df": pd.DataFrame([{"run_id": "ff-run", "pdr_percent": 80.0}]),
        "relay_bottleneck_metrics_df": pd.DataFrame(columns=["run_id", "entity_id"]),
    }
    records = {}
    paths = []
    for key, frame in frames.items():
        filename = f"{key.removesuffix('_df')}.csv"
        path = tables_dir / filename
        frame.to_csv(path, index=False)
        relative = f"tables/{filename}"
        paths.append(relative)
        records[key] = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": pipeline.file_sha256(path),
        }
    (run_dir / "scientific_output_manifest.json").write_text(
        json.dumps({
            "status": "COMPLETED",
            "tables": records,
            "artifacts": {},
            "compatibility_artifacts": {},
            "required_table_keys": list(records),
            "required_artifacts": sorted(paths),
        }),
        encoding="utf-8",
    )
    config = {
        "experiment_profile": APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
        "node_ptx_mode": "fixed",
    }
    result = pipeline._scientific_master_frames(
        tmp_path,
        [run_dir],
        ["core_summary_df", "relay_bottleneck_metrics_df", "wang_snr_report_lifecycle_df"],
        run_configs={"ff-run": config},
    )
    wang = result["per_run_wang_snr_report_lifecycle.csv"]
    assert wang.empty
    assert "report_id" in wang.columns


def test_scientific_master_publish_is_gzip_atomic_and_idempotent(tmp_path, monkeypatch):
    source_fingerprint = {"algorithm": "sha256", "files": {"src/x.py": "abc"}}
    monkeypatch.setattr(pipeline, "_source_fingerprint", lambda _root: source_fingerprint)
    monkeypatch.setattr(
        pipeline,
        "collect_complete_source_provenance",
        lambda _root: {"dependency_scope_complete": True, "digest": "source"},
    )
    monkeypatch.setattr(
        pipeline.ScientificOutputManager,
        "validate_completed",
        lambda *_args, **_kwargs: SimpleNamespace(
            safe_to_skip=True,
            reasons=(),
            manifest={"config_fingerprint_sha256": "config"},
        ),
    )
    spec = pipeline.ExperimentSpec(
        experiment_id="complete-cohort",
        output_dir=tmp_path / "artifacts",
        base_config={
            "sim_time": 10,
            "persistence_profile": "scientific",
            "gzip_scientific_tables": True,
        },
        factors={"node_num": [10]},
        seeds=[1],
        scientific_master_policy={
            "materialization": "deferred",
            "table_keys": ["core_summary_df", "relay_bottleneck_metrics_df"],
            "compression": "inherit",
        },
    )
    experiment_dir = pipeline.initialize_experiment(spec, tmp_path)
    run = pipeline.build_experiment_grid(spec)[0]
    run_dir = ExperimentArtifactPaths(experiment_dir).runs / run["run_id"]
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True)
    run_frames = {
        "core_summary_df": pd.DataFrame([{"run_id": run["run_id"], "pdr_percent": 80.0}]),
        "relay_bottleneck_metrics_df": pd.DataFrame(columns=["run_id", "entity_id"]),
    }
    table_records = {}
    required_artifacts = []
    for key, frame in run_frames.items():
        filename = f"{key.removesuffix('_df')}.csv"
        artifact = tables_dir / filename
        frame.to_csv(artifact, index=False)
        relative = f"tables/{filename}"
        required_artifacts.append(relative)
        table_records[key] = {
            "path": relative,
            "size_bytes": artifact.stat().st_size,
            "sha256": pipeline.file_sha256(artifact),
        }
    (run_dir / "scientific_output_manifest.json").write_text(
        json.dumps({
            "status": "COMPLETED",
            "tables": table_records,
            "artifacts": {},
            "compatibility_artifacts": {},
            "required_table_keys": list(table_records),
            "required_artifacts": sorted(required_artifacts),
        }),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    paths = ExperimentArtifactPaths(experiment_dir)
    paths.status.write_text(json.dumps({"status": "RUNS_COMPLETE"}), encoding="utf-8")

    atomic_csv = pipeline.atomic_write_csv
    monkeypatch.setattr(
        pipeline,
        "atomic_write_csv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected write failure")),
    )
    with pytest.raises(OSError, match="injected write failure"):
        pipeline.rebuild_scientific_master_tables(experiment_dir, project_root=tmp_path)
    assert not paths.scientific_master.exists()
    monkeypatch.setattr(pipeline, "atomic_write_csv", atomic_csv)

    first = pipeline.rebuild_scientific_master_tables(experiment_dir, project_root=tmp_path)
    master_manifest_path = paths.scientific_master / "aggregation_manifest.json"
    master_manifest = json.loads(master_manifest_path.read_text(encoding="utf-8"))
    assert set(master_manifest["files"]) == {
        "all_runs_core_summary.csv",
        "per_run_relay_bottleneck_metrics.csv",
    }
    core_record = master_manifest["files"]["all_runs_core_summary.csv"]
    assert core_record["path"].endswith(".csv.gz")
    assert first["all_runs_core_summary.csv"].loc[0, "pdr_percent"] == 80.0
    assert json.loads(paths.status.read_text(encoding="utf-8"))["cohort_finalized"] is True
    before = pipeline.file_sha256(paths.scientific_master / core_record["path"])
    second = pipeline.rebuild_scientific_master_tables(experiment_dir, project_root=tmp_path)
    assert pipeline.file_sha256(paths.scientific_master / core_record["path"]) == before
    pd.testing.assert_frame_equal(
        first["all_runs_core_summary.csv"], second["all_runs_core_summary.csv"]
    )
    core_path = paths.scientific_master / core_record["path"]
    core_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Finalized scientific master checksum mismatch"):
        pipeline.rebuild_scientific_master_tables(experiment_dir, project_root=tmp_path)
    experiment_manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    experiment_manifest["source_fingerprint"] = {"tampered": True}
    paths.manifest.write_text(json.dumps(experiment_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source fingerprint differs"):
        pipeline.rebuild_scientific_master_tables(experiment_dir, project_root=tmp_path)


@pytest.mark.parametrize(
    ("materialization", "expected_rebuild_calls", "expected_status"),
    [
        ("deferred", 0, "RUNS_COMPLETE"),
        ("on_complete", 1, "PASS"),
    ],
)
def test_run_experiment_obeys_scientific_master_materialization_policy(
    tmp_path, monkeypatch, materialization, expected_rebuild_calls, expected_status
):
    spec = pipeline.ExperimentSpec(
        experiment_id=f"policy-{materialization}",
        output_dir=tmp_path,
        base_config={"sim_time": 10, "persistence_profile": "scientific"},
        factors={"node_num": [10]},
        seeds=[1],
        scientific_master_policy={
            "materialization": materialization,
            "table_keys": ["core_summary_df", "relay_bottleneck_metrics_df"],
            "compression": "csv",
        },
    )
    experiment_dir = tmp_path / spec.experiment_id
    ExperimentArtifactPaths(experiment_dir).ensure()
    monkeypatch.setattr(pipeline, "initialize_experiment", lambda *_args: experiment_dir)
    monkeypatch.setattr(pipeline, "execute_one_run", lambda *_args, **_kwargs: "PASS")
    expected_run = pipeline.build_experiment_grid(spec)[0]
    monkeypatch.setattr(
        pipeline,
        "_read_passed_run_dirs",
        lambda _directory: [ExperimentArtifactPaths(experiment_dir).runs / expected_run["run_id"]],
    )
    monkeypatch.setattr(
        pipeline,
        "_scientific_master_frames",
        lambda *_args, **_kwargs: {
            "all_runs_core_summary.csv": pd.DataFrame([{"pdr_percent": 80.0}]),
            "per_run_relay_bottleneck_metrics.csv": pd.DataFrame(),
        },
    )
    rebuild_calls = []
    monkeypatch.setattr(
        pipeline,
        "rebuild_scientific_master_tables",
        lambda *_args, **_kwargs: rebuild_calls.append("called") or {},
    )
    monkeypatch.setattr(
        pipeline,
        "aggregate_experiment",
        lambda *_args, **_kwargs: {
            "seed_completeness": pd.DataFrame({"complete": [True]}),
            "core_metrics": pd.DataFrame(),
            "relay_bottleneck_metrics": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(pipeline, "simulate", lambda *_args, **_kwargs: pytest.fail("simulation called"))
    result = pipeline.run_experiment(
        spec, tmp_path, show_progress=False, resume=False
    )
    assert len(rebuild_calls) == expected_rebuild_calls
    assert result["status"] == expected_status
    assert result["cohort_finalized"] is (materialization == "on_complete")
    assert result["aggregation_status"] == (
        "FINALIZED" if materialization == "on_complete" else "PENDING"
    )


def test_grid_is_deterministic_unique_and_contains_full_seed_cross_product(tmp_path):
    spec = make_spec(tmp_path)
    first = pipeline.build_experiment_grid(spec)
    second = pipeline.build_experiment_grid(spec)

    assert first == second
    assert len(first) == 6
    assert len({row["run_id"] for row in first}) == 6
    assert {row["seed"] for row in first} == {11, 22, 33}
    assert "scientific_master_policy" not in spec.serializable()


def test_spec_rejects_unknown_and_pipeline_controlled_config_keys(tmp_path):
    config = {
        "experiment_id": "invalid",
        "output_dir": str(tmp_path / "out"),
        "base_config": {"node_perid": 60},
        "factors": {"node_num": [10]},
        "seeds": [1, 2],
    }
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown simulator keys"):
        pipeline.load_experiment_spec(path)

    config["base_config"] = {"run_id": "manual"}
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="pipeline-controlled keys"):
        pipeline.load_experiment_spec(path)


def test_spec_rejects_unknown_top_level_key_and_scalar_factor(tmp_path):
    config = {
        "experiment_id": "invalid-schema",
        "output_dir": str(tmp_path / "out"),
        "base_config": {"sim_time": 10},
        "factors": {"node_num": [10]},
        "seeds": [1],
        "confindence_level": 0.95,
    }
    path = tmp_path / "invalid-schema.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        pipeline.load_experiment_spec(path)

    config.pop("confindence_level")
    config["factors"] = {"node_num": 10}
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array"):
        pipeline.load_experiment_spec(path)


def test_spec_validates_master_table_mode(tmp_path):
    config = {
        "experiment_id": "master-mode",
        "output_dir": str(tmp_path / "out"),
        "base_config": {"sim_time": 10},
        "factors": {"node_num": [10]},
        "seeds": [1],
        "master_table_mode": "aggregate_inputs",
    }
    path = tmp_path / "master-mode.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    assert (
        pipeline.load_experiment_spec(path).master_table_mode
        == "aggregate_inputs"
    )

    config["master_table_mode"] = "summary-ish"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="master_table_mode"):
        pipeline.load_experiment_spec(path)


@pytest.mark.parametrize("invalid_seed", [True, 1.5, "1", -1, 2**32])
def test_spec_rejects_non_reproducible_seed_values(tmp_path, invalid_seed):
    config = {
        "experiment_id": "invalid-seed",
        "output_dir": str(tmp_path / "out"),
        "base_config": {"sim_time": 10},
        "factors": {"node_num": [10]},
        "seeds": [invalid_seed],
    }
    path = tmp_path / "invalid-seed.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="seed"):
        pipeline.load_experiment_spec(path)


def test_grid_rejects_invalid_resolved_factor_combination(tmp_path):
    spec = pipeline.ExperimentSpec(
        experiment_id="invalid-cell",
        output_dir=tmp_path,
        base_config={"sim_time": 10},
        factors={"node_period": [0]},
        seeds=[7],
    )

    with pytest.raises(ValueError, match="node_period"):
        pipeline.build_experiment_grid(spec)


def test_seed_completeness_detects_missing_and_duplicate_seed_cells():
    frame = pd.DataFrame(
        {
            "run_id": ["a", "b", "c", "d", "e"],
            "node_num": [10, 10, 20, 20, 20],
            "seed": [1, 2, 1, 1, 3],
        }
    )
    report = pipeline.validate_seed_completeness(
        frame, ["node_num"], expected_seeds=[1, 2, 3]
    ).set_index("node_num")

    assert not bool(report.loc[10, "complete"])
    assert report.loc[10, "missing_seeds"] == "3"
    assert not bool(report.loc[20, "complete"])
    assert report.loc[20, "missing_seeds"] == "2"
    assert report.loc[20, "duplicate_seed_count"] == 1


def test_paper_grade_aggregation_reports_student_t_ci_and_quantiles():
    frame = pd.DataFrame(
        {
            "node_num": [10, 10, 10],
            "metric_value": [1.0, 2.0, 3.0],
        }
    )
    result = pipeline.aggregate_paper_grade(
        frame,
        group_columns=["node_num"],
        metric_columns=["metric_value"],
        confidence_level=0.95,
    ).iloc[0]

    assert result["n"] == 3
    assert result["mean"] == pytest.approx(2.0)
    assert result["std"] == pytest.approx(1.0)
    assert result["median"] == pytest.approx(2.0)
    assert result["ci_method"] == "student_t"
    assert result["ci_lower"] == pytest.approx(-0.4841, abs=1e-3)
    assert result["ci_upper"] == pytest.approx(4.4841, abs=1e-3)


def test_run_is_atomic_resumable_and_preserves_relay_evidence(tmp_path, monkeypatch):
    spec = pipeline.ExperimentSpec(
        experiment_id="resume-p2",
        output_dir=tmp_path,
        base_config={"sim_time": 10, "node_period": 60},
        factors={"node_num": [10]},
        seeds=[7],
    )
    calls = []

    def fake_simulator(config):
        calls.append(config["run_id"])
        core = {
            "run_id": config["run_id"],
            "delivery_by_horizon_percent": 80.0,
            "terminal_loss_percent": 10.0,
            "censored_fraction_percent": 10.0,
            "packet_fate_conservation_delta": 0,
        }
        outputs = {
            "packet_fate_summary_df": pd.DataFrame(
                [
                    {
                        "offered_count": 10,
                        "delivered_count": 8,
                        "terminal_drop_count": 1,
                        "censored_count": 1,
                        "packet_fate_conservation_delta": 0,
                    }
                ]
            ),
            "relay_tx_queue_metrics_df": pd.DataFrame(
                [
                    {
                        "entity_id": "R1",
                        "entity_type": "relay",
                        "accepted_count": 9,
                        "completed_count": 8,
                        "unfinished_end_count": 1,
                        "avg_fifo_wait_ms": 12.0,
                    }
                ]
            ),
            "relay_rx_stats_df": pd.DataFrame(
                [
                    {
                        "relay": "R1",
                        "rx_arrival_count": 9,
                        "collision": 1,
                    }
                ]
            ),
            "topology_parent_by_run_df": pd.DataFrame(
                [{"entity_id": "N1", "parent_id": "R1"}]
            ),
            "apc_feedback_lifecycle_df": pd.DataFrame(
                [{"feedback_id": "feedback-1", "event": "applied"}]
            ),
            "tx_airtime_by_entity_mtype_ptx_channel_df": pd.DataFrame(
                [{"entity_id": "N1", "traffic_class": "uplink"}]
            ),
        }
        return SimulationResult(
            resolved_config=config,
            summary=core,
            tables=outputs,
        )

    monkeypatch.setattr(pipeline, "simulate", fake_simulator)
    project_root = Path(__file__).resolve().parents[2]
    experiment_dir = pipeline.initialize_experiment(spec, project_root)
    run = pipeline.build_experiment_grid(spec)[0]

    assert pipeline.execute_one_run(spec, experiment_dir, run) == "PASS"
    assert pipeline.execute_one_run(spec, experiment_dir, run) == "SKIPPED"
    assert calls == [run["run_id"]]

    run_tables = experiment_dir / "runs" / run["run_id"] / "tables"
    assert pd.read_csv(run_tables / "topology_parent_by_run.csv").loc[
        0, "parent_id"
    ] == "R1"
    assert pd.read_csv(run_tables / "apc_feedback_lifecycle.csv").loc[
        0, "event"
    ] == "applied"
    assert pd.read_csv(
        run_tables / "tx_airtime_by_entity_mtype_ptx_channel.csv"
    ).loc[0, "traffic_class"] == "uplink"

    master = pipeline.rebuild_master_tables(experiment_dir)
    relay = master["per_run_relay_bottleneck_metrics.csv"]
    assert relay.loc[0, "entity_id"] == "R1"
    assert relay.loc[0, "queue_accepted_count"] == 9
    assert relay.loc[0, "rx_collision"] == 1

    aggregated = pipeline.aggregate_experiment(spec, experiment_dir, master)
    assert bool(aggregated["seed_completeness"].loc[0, "complete"])


def test_managed_persistence_uses_atomic_layout_and_never_implicitly_skips(tmp_path, monkeypatch):
    from loraphymac.experiments.apc_load_sweep.contract import (
        APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE,
    )
    from loraphymac.experiments.common.scientific_output_contract import required_table_keys
    import loraphymac.experiments.common.scientific_output_manager as output_manager

    spec = pipeline.ExperimentSpec(
        experiment_id="managed-output-test",
        output_dir=tmp_path,
        base_config={"experiment_profile": APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE},
        factors={"node_num": [10]},
        seeds=[30],
    )
    run = {
        "run_id": "managed-run-30",
        "seed": 30,
        "factors": {"node_num": 10},
        "scientific_config": {"persistence_profile": "scientific"},
    }
    config = {
        "experiment_profile": APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE,
        "output_profile": "summary",
        "persistence_profile": "scientific",
        "node_ptx_mode": "fixed",
        "relay_ptx_mode": "fixed",
        "tx_power_model_id": "ff",
        "run_id": run["run_id"],
        "seed": 30,
    }
    monkeypatch.setattr(pipeline, "_resolved_simulator_config", lambda _: config)
    monkeypatch.setattr(pipeline, "validate_run_output_contract", lambda *_: None)
    monkeypatch.setattr(output_manager, "validate_run_output_contract", lambda *_: None)
    outputs = {
        key: pd.DataFrame({"value": [1]}) for key in required_table_keys(config)
    }
    from loraphymac.experiments.common.scientific_output_contract import output_contract_for

    contract = output_contract_for(config)
    for key, columns in contract.table_schemas.items():
        outputs[key] = pd.DataFrame(columns=columns)
    from loraphymac.metrics.apc_metrics import PTX_HISTORY_COLUMNS
    from loraphymac.metrics.instrumentation_metrics import APC_FEEDBACK_LIFECYCLE_COLUMNS

    outputs["ptx_history_df"] = pd.DataFrame(columns=PTX_HISTORY_COLUMNS)
    outputs["apc_feedback_lifecycle_df"] = pd.DataFrame(
        columns=APC_FEEDBACK_LIFECYCLE_COLUMNS
    )
    outputs.update(
        {
            "core_summary_df": pd.DataFrame([{"pdr_percent": 80.0}]),
            "relay_tx_queue_metrics_df": pd.DataFrame(columns=["value"]),
            "relay_rx_stats_df": pd.DataFrame(columns=["value"]),
            "relay_duty_cycle_df": pd.DataFrame(columns=["value"]),
            "demodulator_pool_summary_df": pd.DataFrame(columns=["value"]),
            "collision_summary_df": pd.DataFrame(
                columns=contract.table_schemas["collision_summary_df"]
            ),
        }
    )
    calls = []

    def fake_simulator(_):
        calls.append(1)
        return SimulationResult(
            resolved_config=config,
            summary={"pdr_percent": 80.0},
            tables=outputs,
        )

    monkeypatch.setattr(pipeline, "simulate", fake_simulator)
    experiment_dir = pipeline.initialize_experiment(spec, tmp_path)

    assert pipeline.execute_one_run(spec, experiment_dir, run) == "PASS"
    run_dir = experiment_dir / "runs" / run["run_id"]
    assert json.loads((run_dir / "status.json").read_text())["status"] == "PASS"
    assert pd.read_csv(run_dir / "core_summary.csv").loc[0, "pdr_percent"] == 80.0
    assert (run_dir / "scientific_output_manifest.json").is_file()
    master = pipeline.rebuild_master_tables(
        experiment_dir,
        APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE,
        persistence_profile="scientific",
    )
    assert len(master["all_runs_core_summary.csv"]) == 1
    assert "per_run_ptx_history.csv" in master
    assert master["per_run_ptx_history.csv"].empty
    assert set(PTX_HISTORY_COLUMNS) <= set(master["per_run_ptx_history.csv"].columns)
    assert master["per_run_apc_feedback_lifecycle.csv"].empty
    assert set(APC_FEEDBACK_LIFECYCLE_COLUMNS) <= set(
        master["per_run_apc_feedback_lifecycle.csv"].columns
    )

    # Completed manager output is immutable; runner does not infer safe-to-skip.
    assert pipeline.execute_one_run(spec, experiment_dir, run) == "FAILED"
    assert calls == [1]


def test_aggregate_inputs_master_mode_avoids_detail_table_duplication(tmp_path):
    experiment_dir = tmp_path / "lean-master"
    run_dir = experiment_dir / "runs" / "run-1"
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text('{"status":"PASS"}', encoding="utf-8")
    pd.DataFrame([{"run_id": "run-1", "seed": 7}]).to_csv(
        run_dir / "core_summary.csv", index=False
    )
    pd.DataFrame([{"run_id": "run-1", "packet_id": "p1"}]).to_csv(
        tables_dir / "packet_fates.csv", index=False
    )
    pd.DataFrame([{"run_id": "run-1", "entity_id": "R1"}]).to_csv(
        tables_dir / "relay_bottleneck_metrics.csv", index=False
    )

    master = pipeline.rebuild_master_tables(
        experiment_dir,
        master_table_mode="aggregate_inputs",
    )

    assert set(master) == {
        "all_runs_core_summary.csv",
        "per_run_relay_bottleneck_metrics.csv",
    }
    assert not (experiment_dir / "master" / "per_run_packet_fates.csv").exists()


def test_scientific_master_preserves_contract_schema_for_zero_byte_empty_table(tmp_path):
    from loraphymac.experiments.apc_load_sweep.contract import (
        APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE,
    )
    from loraphymac.metrics.apc_metrics import PTX_HISTORY_COLUMNS

    experiment_dir = tmp_path / "scientific-empty-master"
    run_dir = experiment_dir / "runs" / "run-1"
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text('{"status":"PASS"}', encoding="utf-8")
    pd.DataFrame([{"run_id": "run-1", "seed": 30}]).to_csv(
        run_dir / "core_summary.csv", index=False
    )
    resolved = run_dir / "resolved_config.json"
    resolved.write_text("{}", encoding="utf-8")
    empty_table = tables_dir / "ptx_history.csv.gz"
    with gzip.open(empty_table, "wb") as stream:
        stream.write(b"")
    def record(path, relative, kind):
        return {
            "path": relative,
            "kind": kind,
            "size_bytes": path.stat().st_size,
            "sha256": pipeline.file_sha256(path),
        }

    table_record = record(empty_table, "tables/ptx_history.csv.gz", "table")
    artifact_record = record(resolved, "resolved_config.json", "resolved_config")
    core_record = record(
        run_dir / "core_summary.csv", "core_summary.csv", "pipeline_compatibility_table"
    )
    status_record = record(
        run_dir / "status.json", "status.json", "pipeline_compatibility_status"
    )
    required_artifacts = sorted(
        [table_record["path"], artifact_record["path"], core_record["path"], status_record["path"]]
    )
    manifest = {
        "status": "COMPLETED",
        "run_id": "run-1",
        "required_table_keys": ["ptx_history_df"],
        "gzip_table_keys": ["ptx_history_df"],
        "required_artifacts": required_artifacts,
        "tables": {"ptx_history_df": table_record},
        "artifacts": {"resolved_config.json": artifact_record},
        "compatibility_artifacts": {
            "core_summary.csv": core_record,
            "status.json": status_record,
        },
    }
    (run_dir / "scientific_output_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    master = pipeline.rebuild_master_tables(
        experiment_dir,
        experiment_profile=APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE,
        persistence_profile="scientific",
    )
    ptx = master["per_run_ptx_history.csv.gz"]
    assert ptx.empty
    assert list(ptx.columns) == list(PTX_HISTORY_COLUMNS)
    master_path = experiment_dir / "master" / "per_run_ptx_history.csv.gz"
    assert pd.read_csv(master_path).empty
    with gzip.open(master_path, "rt", encoding="utf-8") as stream:
        assert stream.read().strip() == ",".join(PTX_HISTORY_COLUMNS)


def test_scientific_master_fails_when_manifested_table_is_missing(tmp_path):
    experiment_dir = tmp_path / "scientific-missing-master"
    run_dir = experiment_dir / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text('{"status":"PASS"}', encoding="utf-8")
    (run_dir / "core_summary.csv").write_text("run_id\nrun-1\n", encoding="utf-8")
    (run_dir / "resolved_config.json").write_text("{}", encoding="utf-8")
    status_path = run_dir / "status.json"
    config_path = run_dir / "resolved_config.json"
    core_path = run_dir / "core_summary.csv"
    missing_record = {
        "path": "tables/packet_fates.csv", "size_bytes": 1, "sha256": "0" * 64
    }
    config_record = {
        "path": "resolved_config.json", "size_bytes": config_path.stat().st_size,
        "sha256": pipeline.file_sha256(config_path),
    }
    core_record = {
        "path": "core_summary.csv", "size_bytes": core_path.stat().st_size,
        "sha256": pipeline.file_sha256(core_path),
    }
    status_record = {
        "path": "status.json", "size_bytes": status_path.stat().st_size,
        "sha256": pipeline.file_sha256(status_path),
    }
    (run_dir / "scientific_output_manifest.json").write_text(
        json.dumps({
            "status": "COMPLETED",
            "run_id": "run-1",
            "required_table_keys": ["packet_fate_df"],
            "required_artifacts": sorted([
                missing_record["path"], config_record["path"],
                core_record["path"], status_record["path"],
            ]),
            "tables": {"packet_fate_df": missing_record},
            "artifacts": {"resolved_config.json": config_record},
            "compatibility_artifacts": {
                "core_summary.csv": core_record, "status.json": status_record,
            },
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Scientific artifact is missing"):
        pipeline.rebuild_master_tables(
            experiment_dir,
            experiment_profile="apc_load_sweep_full_factorial_v1",
            persistence_profile="scientific",
        )


def test_scientific_master_rejects_manifest_checksum_mismatch(tmp_path):
    experiment_dir = tmp_path / "scientific-checksum-master"
    run_dir = experiment_dir / "runs" / "run-1"
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text('{"status":"PASS"}', encoding="utf-8")
    (run_dir / "core_summary.csv").write_text("run_id\nrun-1\n", encoding="utf-8")
    (run_dir / "resolved_config.json").write_text("{}", encoding="utf-8")
    table_path = tables_dir / "ptx_history.csv"
    table_path.write_text("run_id,time\n", encoding="utf-8")
    records = {
        "ptx_history_df": {
            "path": "tables/ptx_history.csv", "size_bytes": table_path.stat().st_size,
            "sha256": "0" * 64,
        },
        "resolved_config.json": {
            "path": "resolved_config.json",
            "size_bytes": (run_dir / "resolved_config.json").stat().st_size,
            "sha256": pipeline.file_sha256(run_dir / "resolved_config.json"),
        },
        "core_summary.csv": {
            "path": "core_summary.csv", "size_bytes": (run_dir / "core_summary.csv").stat().st_size,
            "sha256": pipeline.file_sha256(run_dir / "core_summary.csv"),
        },
        "status.json": {
            "path": "status.json", "size_bytes": (run_dir / "status.json").stat().st_size,
            "sha256": pipeline.file_sha256(run_dir / "status.json"),
        },
    }
    (run_dir / "scientific_output_manifest.json").write_text(
        json.dumps({
            "status": "COMPLETED",
            "required_table_keys": ["ptx_history_df"],
            "required_artifacts": sorted(record["path"] for record in records.values()),
            "tables": {"ptx_history_df": records["ptx_history_df"]},
            "artifacts": {"resolved_config.json": records["resolved_config.json"]},
            "compatibility_artifacts": {
                "core_summary.csv": records["core_summary.csv"],
                "status.json": records["status.json"],
            },
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        pipeline.rebuild_master_tables(
            experiment_dir,
            experiment_profile="apc_load_sweep_full_factorial_v1",
            persistence_profile="scientific",
        )


def test_legacy_master_keeps_skipping_empty_inputs(tmp_path):
    experiment_dir = tmp_path / "legacy-empty-master"
    run_dir = experiment_dir / "runs" / "run-1"
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text('{"status":"PASS"}', encoding="utf-8")
    pd.DataFrame([{"run_id": "run-1", "seed": 30}]).to_csv(
        run_dir / "core_summary.csv", index=False
    )
    pd.DataFrame(columns=["packet_uid", "src_id"]).to_csv(
        tables_dir / "packet_fates.csv", index=False
    )
    master = pipeline.rebuild_master_tables(experiment_dir)
    assert master["per_run_packet_fates.csv"].empty
    assert master["per_run_packet_fates.csv"].columns.empty


def test_source_fingerprint_covers_nested_scientific_modules(tmp_path):
    source_dir = tmp_path / "src" / "loraphymac" / "routing"
    source_dir.mkdir(parents=True)
    source = source_dir / "routing_engine.py"
    source.write_text("MODEL_VERSION = 1\n", encoding="utf-8")
    first = pipeline._source_fingerprint(tmp_path)

    source.write_text("MODEL_VERSION = 2\n", encoding="utf-8")
    second = pipeline._source_fingerprint(tmp_path)

    assert "src/loraphymac/routing/routing_engine.py" in first["files"]
    assert first != second


def test_incomplete_grid_persists_experiment_level_failure_status(tmp_path, monkeypatch):
    spec = pipeline.ExperimentSpec(
        experiment_id="incomplete-p2",
        output_dir=tmp_path,
        base_config={"sim_time": 10, "node_period": 60},
        factors={"node_num": [10]},
        seeds=[7],
    )
    monkeypatch.setattr(pipeline, "execute_one_run", lambda *args, **kwargs: "FAILED")

    with pytest.raises(ValueError, match="Seed grid is incomplete"):
        pipeline.run_experiment(spec, Path(__file__).resolve().parents[2])

    status_path = tmp_path / spec.experiment_id / "experiment_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "INCOMPLETE"
    assert status["status_counts"]["FAILED"] == 1
    assert status["aggregation_error_type"] == "ValueError"


def test_run_experiment_prints_compact_live_progress(tmp_path, monkeypatch, capsys):
    spec = pipeline.ExperimentSpec(
        experiment_id="progress-apc",
        output_dir=tmp_path,
        base_config={"sim_time": 10, "node_period": 60},
        factors={
            "node_ptx_mode": ["adaptive"],
            "relay_ptx_mode": ["fixed"],
            "terrain_delta_h": [90],
        },
        seeds=[7],
    )
    monkeypatch.setattr(pipeline, "execute_one_run", lambda *args, **kwargs: "PASS")
    monkeypatch.setattr(pipeline, "rebuild_master_tables", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        pipeline,
        "aggregate_experiment",
        lambda *args, **kwargs: {
            "seed_completeness": pd.DataFrame({"complete": [True]})
        },
    )
    monkeypatch.setattr(pipeline, "_read_passed_run_dirs", lambda *args: [tmp_path])

    pipeline.run_experiment(spec, Path(__file__).resolve().parents[2])

    output = capsys.readouterr().out
    assert "[PIPELINE] experiment=progress-apc | runs=1 | resume=True" in output
    assert "[001/001] RUNNING | mode=AF | delta_h=90 | seed=7" in output
    assert "[001/001] PASS | elapsed=" in output
    assert "[PIPELINE] AGGREGATING | processed=1/1" in output
    assert "[PIPELINE] PASS | passed_on_disk=1/1 | seed_grid_complete=True" in output


def test_run_experiment_can_suppress_progress(tmp_path, monkeypatch, capsys):
    spec = pipeline.ExperimentSpec(
        experiment_id="quiet-progress",
        output_dir=tmp_path,
        base_config={"sim_time": 10, "node_period": 60},
        factors={"node_num": [10]},
        seeds=[7],
    )
    monkeypatch.setattr(pipeline, "execute_one_run", lambda *args, **kwargs: "PASS")
    monkeypatch.setattr(pipeline, "rebuild_master_tables", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        pipeline,
        "aggregate_experiment",
        lambda *args, **kwargs: {
            "seed_completeness": pd.DataFrame({"complete": [True]})
        },
    )
    monkeypatch.setattr(pipeline, "_read_passed_run_dirs", lambda *args: [tmp_path])

    pipeline.run_experiment(
        spec,
        Path(__file__).resolve().parents[2],
        show_progress=False,
    )

    assert capsys.readouterr().out == ""
