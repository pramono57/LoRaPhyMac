from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from loraphymac.experiments.apc_load_sweep.standard_analysis import (
    build_load_sweep_analysis,
)
import loraphymac.experiments.apc_load_sweep.standard_analysis as load_analysis
from loraphymac.experiments.common.analysis_publication import AnalysisPublisher
from loraphymac.experiments.common.multiseed_pipeline import (
    ExperimentSpec,
    build_experiment_grid,
    initialize_experiment,
    load_experiment_spec,
    run_experiment,
)
from loraphymac.experiments.common.outputs import (
    ExperimentArtifactPaths,
    SWEEP_OUTPUT_LAYOUT_VERSION,
    resolve_master_table_path,
)
from loraphymac.experiments.common.sweep_standard import reserve_sweep_cohort, write_cohort_readme
from loraphymac.experiments.apc_load_sweep.run import DEFAULT_CONFIG as LOAD_STANDARD_CONFIG


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cohort_reservations_are_unique_and_no_overwrite(tmp_path):
    spec = ExperimentSpec(
        experiment_id="demo",
        output_dir=tmp_path,
        base_config={},
        factors={"node_num": [10]},
        seeds=[1],
    )
    instant = datetime(2026, 11, 23, 12, 0, tzinfo=ZoneInfo("Asia/Jakarta"))

    def reserve():
        return reserve_sweep_cohort(
            spec, project_root=tmp_path, scenario="load_sweep", now=instant
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _: reserve(), range(2)))
    assert {first.experiment_id, second.experiment_id} == {
        "load_sweep_v1_r01_20261123",
        "load_sweep_v1_r02_20261123",
    }
    assert first.execution_id != second.execution_id
    initialized = initialize_experiment(first, tmp_path)
    paths = ExperimentArtifactPaths.for_experiment(initialized)
    assert paths.is_sweep_standard
    assert paths.runs.is_dir()
    assert paths.aggregate.is_dir()
    assert (initialized / "analysis" / "figures").is_dir()
    assert (initialized / "analysis" / "tables").is_dir()
    assert not (initialized / "master").exists()
    assert not (initialized / "README.md").exists()
    assert json.loads(paths.manifest.read_text())["status"] == "INITIALIZED"
    try:
        initialize_experiment(first, tmp_path)
    except (FileExistsError, ValueError):
        pass
    else:
        raise AssertionError("an initialized cohort must never be reopened")
    write_cohort_readme(
        initialized, spec=first, run_count=1, seed_list=[1], source_fingerprint_hash="synthetic"
    )
    assert (initialized / "README.md").is_file()
    assert "runs/<run_id>/" in (initialized / "README.md").read_text()


def test_load_standard_grid_has_aa_and_ff_in_one_120_run_cohort():
    from loraphymac.experiments.common.multiseed_pipeline import load_experiment_spec

    spec = load_experiment_spec(LOAD_STANDARD_CONFIG)
    grid = build_experiment_grid(spec)
    assert len(grid) == 120
    assert {row["factors"]["node_ptx_mode"] for row in grid} == {"adaptive", "fixed"}
    assert {row["scientific_config"]["relay_ptx_mode"] for row in grid} == {
        "adaptive",
        "fixed",
    }
    assert {
        ("AA" if row["scientific_config"]["node_ptx_mode"] == "adaptive" else "FF")
        for row in grid
    } == {"AA", "FF"}
    assert len({(row["factors"]["node_num"], row["factors"]["node_period"], row["seed"])
                for row in grid if row["factors"]["node_ptx_mode"] == "adaptive"}) == 60


def test_terrain_and_feedback_grids_are_config_derived_and_unique():
    from loraphymac.experiments.apc_terrain_sweep.run import DEFAULT_CONFIG as TERRAIN_CONFIG
    from loraphymac.experiments.apc_feedback_robustness_sweep.run import DEFAULT_CONFIG as FEEDBACK_CONFIG
    from loraphymac.experiments.common.multiseed_pipeline import load_experiment_spec

    terrain = build_experiment_grid(load_experiment_spec(TERRAIN_CONFIG))
    terrain_cells = {
        (row["factors"]["node_ptx_mode"], row["factors"]["relay_ptx_mode"],
         row["factors"]["terrain_delta_h"], row["seed"])
        for row in terrain
    }
    assert len(terrain) == len(terrain_cells) == 300
    feedback = build_experiment_grid(load_experiment_spec(FEEDBACK_CONFIG))
    feedback_cells = {
        (row["factors"]["node_period"],
         row["factors"]["feedback_additional_loss_probability"], row["seed"])
        for row in feedback
    }
    assert len(feedback) == len(feedback_cells) == 30


def test_load_dry_run_reports_standard_layout_without_reserving(tmp_path, capsys):
    from loraphymac.experiments.common.cli import run_standard_cli

    assert run_standard_cli(
        description="load standard dry-run",
        project_root=tmp_path,
        scenario="load_sweep",
        default_config=LOAD_STANDARD_CONFIG,
        argv=["--dry-run", "--quiet"],
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["output_layout_version"] == SWEEP_OUTPUT_LAYOUT_VERSION
    assert report["run_count"] == 120
    assert report["cohort_reserved"] is False
    assert "rNN_" in report["proposed_cohort_id"]
    assert str(tmp_path / "artifacts" / "experiments" / "APC_LOAD_SWEEP") in report["experiment_dir"]
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.extended_research
def test_compare_standard_grid_keeps_all_five_models_in_one_cohort():
    from loraphymac.experiments.apc_compare_model_terrain_sweep.run import (
        STANDARD_CONFIG as COMPARE_STANDARD_CONFIG,
        _build_standard_cohort_spec,
        build_model_specs,
        load_sweep_definition,
    )

    definition = load_sweep_definition(COMPARE_STANDARD_CONFIG)
    specs = build_model_specs(COMPARE_STANDARD_CONFIG)
    cohort_spec = _build_standard_cohort_spec(definition, specs)
    grid = build_experiment_grid(cohort_spec)
    assert len(grid) == 375
    by_model = pd.Series([run["factors"]["model_id"] for run in grid]).value_counts()
    assert set(by_model.to_dict().values()) == {75}
    assert set(by_model.index) == set(definition.models)
    assert definition.analysis_models == (
        "ff",
        "aa",
        "islam_dbpc_itm6",
        "wang_atp_sf7_a2only",
    )
    refs = cohort_spec.external_references
    assert len(refs["w4_snapshot_files"]) == 5
    assert all(len(row["files"]) == 3 for row in refs["w4_snapshot_files"])
    dbpc = next(value for key, value in refs.items() if key.startswith("dbpc_calibration_manifest_"))
    assert len(dbpc["referenced_files"]) == 5


@pytest.mark.unreleased_standard
def test_compare_standard_figure_stems_are_descriptive_without_renaming_legacy():
    from loraphymac.experiments.apc_compare_model_terrain_sweep.postprocessing import (
        FIGURES as LEGACY_COMPARE_FIGURES,
        STANDARD_FIGURE_STEMS,
    )

    assert set(STANDARD_FIGURE_STEMS.values()) == {
        "pdr_vs_terrain",
        "energy_per_delivered_packet_vs_terrain",
        "avg_e2e_delay_vs_terrain",
    }
    assert all(not value.startswith("figure_") for value in STANDARD_FIGURE_STEMS.values())
    assert {row[2] for row in LEGACY_COMPARE_FIGURES} == {
        "figure_08_pdr_vs_terrain",
        "figure_09_energy_per_delivered_packet_vs_terrain",
        "figure_10_avg_e2e_delay_vs_terrain",
    }


def test_standard_path_resolver_reads_aggregate_and_preserves_legacy_master(tmp_path):
    new = ExperimentArtifactPaths(
        tmp_path / "new", SWEEP_OUTPUT_LAYOUT_VERSION
    ).ensure()
    (new.experiment_dir / "experiment_manifest.json").write_text(
        json.dumps({"output_layout_version": SWEEP_OUTPUT_LAYOUT_VERSION})
    )
    summary = new.aggregate / "all_runs_core_summary.csv"
    summary.write_text("run_id\nr1\n")
    assert resolve_master_table_path(new.experiment_dir, summary.name) == summary
    assert not new.master.joinpath("scientific_master").exists()

    legacy = ExperimentArtifactPaths(tmp_path / "legacy").ensure()
    legacy_summary = legacy.master / "all_runs_core_summary.csv"
    legacy_summary.write_text("run_id\nr1\n")
    assert resolve_master_table_path(legacy.experiment_dir, legacy_summary.name) == legacy_summary


def _make_synthetic_load_cohort(root: Path, monkeypatch=None) -> list[str]:
    (root / "aggregate").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    rows = []
    run_ids = []
    for mode in ("AA", "FF"):
        node_mode = "adaptive" if mode == "AA" else "fixed"
        for nodes in (50, 100, 150, 200):
            for period in (30, 60, 90):
                for seed in (30, 36, 42, 48, 56):
                    run_id = f"synthetic-{mode}-{nodes}-{period}-{seed}"
                    run_ids.append(run_id)
                    run_dir = root / "runs" / run_id
                    run_dir.mkdir()
                    (run_dir / "tables").mkdir()
                    for table_name in (
                        "node_tx_duty_cycle.csv",
                        "channel_occupancy.csv",
                        "apc_feedback_lifecycle.csv",
                    ):
                        (run_dir / "tables" / table_name).write_text(
                            "run_id,value\n" + run_id + ",1\n"
                        )
                    (run_dir / "resolved_config.json").write_text(json.dumps({
                        "run_id": run_id,
                        "seed": seed,
                        "factors": {
                            "node_ptx_mode": node_mode,
                            "node_num": nodes,
                            "node_period": period,
                        },
                        "scientific_config": {"relay_ptx_mode": node_mode},
                    }))
                    (run_dir / "scientific_output_manifest.json").write_text(
                        json.dumps({"run_id": run_id, "outputs": []})
                    )
                    rows.append(
                        {
                            "run_id": run_id,
                            "apc_mode": mode,
                            "node_ptx_mode": node_mode,
                            "node_num": nodes,
                            "node_period": period,
                            "seed": seed,
                            "pdr_percent": 90 + seed / 100,
                            "energy_per_delivered_packet_J": 1.0 + nodes / 100,
                            "total_network_energy_J": 10.0 + nodes,
                            "avg_e2e_delay_ms": 200.0 + seed,
                        }
                    )
    core = root / "aggregate" / "all_runs_core_summary.csv"
    pd.DataFrame(rows).to_csv(core, index=False)
    (root / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "output_layout_version": SWEEP_OUTPUT_LAYOUT_VERSION,
                "cohort_id": "synthetic_v1_r01_20261123",
                "run_ids": run_ids,
                "expected_run_count": len(run_ids),
                "parameter_grid": {
                    "factors": {"node_ptx_mode": ["adaptive", "fixed"], "node_num": [50, 100, 150, 200], "node_period": [30, 60, 90]},
                    "seeds": [30, 36, 42, 48, 56],
                },
            }
        )
    )
    (root / "experiment_status.json").write_text(
        json.dumps({"status": "COMPLETE", "seed_grid_complete": True})
    )

    if monkeypatch is not None:
        monkeypatch.setattr(load_analysis, "_read_verified_scientific_tables", lambda _path: {})
    return run_ids


def test_synthetic_load_analysis_is_immutable_and_repeat_keeps_timestamps(tmp_path, monkeypatch):
    root = tmp_path / "cohort"
    _make_synthetic_load_cohort(root, monkeypatch)
    first = build_load_sweep_analysis(root)
    first_manifest = Path(first["analysis_manifest"])
    output_paths = [root / item["path"] for item in json.loads(first_manifest.read_text())["outputs"]]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in [first_manifest, *output_paths]}
    second = build_load_sweep_analysis(root)
    after = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in [first_manifest, *output_paths]}
    assert first["status"] == second["status"] == "COMPLETE"
    assert second["reused"] is True
    assert before == after
    assert len(list((root / "analysis" / "figures").glob("*.png"))) == 2


def test_load_analysis_rejects_bad_run_ids_grid_and_metrics(tmp_path, monkeypatch):
    root = tmp_path / "cohort"
    _make_synthetic_load_cohort(root, monkeypatch)
    core_path = root / "aggregate" / "all_runs_core_summary.csv"
    core = pd.read_csv(core_path)
    core.loc[1, "run_id"] = core.loc[0, "run_id"]
    core.to_csv(core_path, index=False)
    with pytest.raises(ValueError, match="duplicate run IDs"):
        build_load_sweep_analysis(root)


def test_load_analysis_rejects_missing_and_unexpected_run_ids(tmp_path, monkeypatch):
    root = tmp_path / "cohort"
    _make_synthetic_load_cohort(root, monkeypatch)
    core_path = root / "aggregate" / "all_runs_core_summary.csv"
    core = pd.read_csv(core_path)
    core.iloc[:-1].to_csv(core_path, index=False)
    with pytest.raises(ValueError, match="run IDs do not match"):
        build_load_sweep_analysis(root)
    core.iloc[0:1].assign(run_id="unexpected").to_csv(core_path, index=False)
    with pytest.raises(ValueError, match="run IDs do not match"):
        build_load_sweep_analysis(root)


def test_load_analysis_rejects_duplicate_condition_seed_and_bad_metrics(tmp_path, monkeypatch):
    root = tmp_path / "cohort"
    _make_synthetic_load_cohort(root, monkeypatch)
    core_path = root / "aggregate" / "all_runs_core_summary.csv"
    core = pd.read_csv(core_path)
    duplicate_id = core.loc[1, "run_id"]
    config_path = root / "runs" / duplicate_id / "resolved_config.json"
    first_id = core.loc[0, "run_id"]
    first_config = json.loads((root / "runs" / first_id / "resolved_config.json").read_text())
    second_config = json.loads(config_path.read_text())
    second_config["seed"] = first_config["seed"]
    second_config["factors"] = first_config["factors"]
    second_config["scientific_config"] = first_config["scientific_config"]
    config_path.write_text(json.dumps(second_config))
    with pytest.raises(ValueError, match="Duplicate configured condition/seed"):
        build_load_sweep_analysis(root)


def test_load_analysis_rejects_missing_metric_and_invalid_metric_values(tmp_path, monkeypatch):
    root = tmp_path / "cohort"
    _make_synthetic_load_cohort(root, monkeypatch)
    core_path = root / "aggregate" / "all_runs_core_summary.csv"
    core = pd.read_csv(core_path)
    core.drop(columns=["avg_e2e_delay_ms"]).to_csv(core_path, index=False)
    with pytest.raises(ValueError, match="lacks columns"):
        build_load_sweep_analysis(root)
    root = tmp_path / "cohort-metric"
    _make_synthetic_load_cohort(root, monkeypatch)
    core_path = root / "aggregate" / "all_runs_core_summary.csv"
    core = pd.read_csv(core_path)
    core.loc[0, "pdr_percent"] = 101
    core.to_csv(core_path, index=False)
    with pytest.raises(ValueError, match="invalid values"):
        build_load_sweep_analysis(root)


def test_missing_configuration_fails_before_cohort_reservation(tmp_path):
    missing = tmp_path / "missing.json"
    spec = ExperimentSpec(
        experiment_id="demo", output_dir=tmp_path, base_config={},
        factors={"node_num": [10]}, seeds=[1], input_config_path=missing,
    )
    with pytest.raises(FileNotFoundError):
        reserve_sweep_cohort(spec, project_root=tmp_path, scenario="load_sweep")
    assert not (tmp_path / "artifacts" / "experiments").exists()

    external = ExperimentSpec(
        experiment_id="demo", output_dir=tmp_path, base_config={},
        factors={"node_num": [10]}, seeds=[1],
        external_references={"required": {"path": "missing-reference.csv", "sha256": "deadbeef"}},
    )
    with pytest.raises(FileNotFoundError, match="external reference"):
        reserve_sweep_cohort(external, project_root=tmp_path, scenario="load_sweep")
    assert not (tmp_path / "artifacts" / "experiments").exists()


def test_failure_after_reservation_is_machine_readable_and_not_reused(tmp_path, monkeypatch):
    from loraphymac.experiments.common import multiseed_pipeline

    base = load_experiment_spec(LOAD_STANDARD_CONFIG)
    spec = reserve_sweep_cohort(
        base, project_root=tmp_path, scenario="load_sweep",
        now=datetime(2026, 11, 23, tzinfo=ZoneInfo("Asia/Jakarta")),
    )

    def fail_source_fingerprint(_root):
        raise RuntimeError("synthetic initialization failure")

    monkeypatch.setattr(multiseed_pipeline, "_source_fingerprint", fail_source_fingerprint)
    with pytest.raises(RuntimeError, match="synthetic initialization failure"):
        run_experiment(spec, tmp_path, show_progress=False)
    paths = ExperimentArtifactPaths.from_spec(spec)
    assert json.loads(paths.manifest.read_text())["status"] == "FAILED"
    assert json.loads(paths.status.read_text())["status"] == "FAILED"
    assert json.loads(Path(spec.cohort_reservation_path).read_text())["status"] == "FAILED"
    rerun = reserve_sweep_cohort(
        base, project_root=tmp_path, scenario="load_sweep",
        now=datetime(2026, 11, 23, tzinfo=ZoneInfo("Asia/Jakarta")),
    )
    assert rerun.experiment_id == "load_sweep_v1_r02_20261123"


def test_analysis_publisher_versions_revisions_without_changing_completed_files(tmp_path):
    root = tmp_path / "cohort"
    analysis = root / "analysis"
    (analysis / "tables").mkdir(parents=True)
    (analysis / "figures").mkdir()
    (analysis / "analysis_manifest.json").write_text(json.dumps({"status": "PENDING"}))
    first = AnalysisPublisher(root)
    candidate = {"analysis_id": "cohort-analysis-v1", "analysis_version": "v1",
                 "source_cohort_id": "cohort", "inputs": [{"sha256": "input-a"}],
                 "generator": {"sha256": "script-a"}}
    staged = first.tables_dir / "table.csv"
    staged.write_text("x\n1\n")
    manifest = {**candidate, "tables": [staged.relative_to(root).as_posix()],
                "outputs": [{"path": staged.relative_to(root).as_posix(),
                             "size_bytes": staged.stat().st_size,
                             "sha256": _sha256(staged)}]}
    one = first.publish(manifest, [staged])
    first.close()
    canonical = Path(one["analysis_manifest"])
    old_output = root / json.loads(canonical.read_text())["outputs"][0]["path"]
    old_bytes, old_mtime = old_output.read_bytes(), old_output.stat().st_mtime_ns
    second = AnalysisPublisher(root)
    assert second.find_reusable(candidate)["reused"] is True
    second.close()
    assert (old_output.read_bytes(), old_output.stat().st_mtime_ns) == (old_bytes, old_mtime)

    revised = AnalysisPublisher(root)
    candidate["inputs"] = [{"sha256": "input-b"}]
    staged2 = revised.tables_dir / "table.csv"
    staged2.write_text("x\n2\n")
    manifest2 = {**candidate, "tables": [staged2.relative_to(root).as_posix()],
                 "outputs": [{"path": staged2.relative_to(root).as_posix(),
                              "size_bytes": staged2.stat().st_size,
                              "sha256": _sha256(staged2)}]}
    two = revised.publish(manifest2, [staged2])
    revised.close()
    assert two["analysis_revision"] == "analysis-v2"
    assert "analysis/versions/analysis-v2/" in two["tables"][0]
    assert (old_output.read_bytes(), old_output.stat().st_mtime_ns) == (old_bytes, old_mtime)


def test_analysis_publisher_rejects_corrupt_input_hash(tmp_path):
    root = tmp_path / "cohort"
    (root / "analysis").mkdir(parents=True)
    source = root / "aggregate.csv"
    source.write_text("x\n1\n")
    publisher = AnalysisPublisher(root)
    with pytest.raises(Exception, match="SHA-256 mismatch"):
        publisher.find_reusable({"inputs": [{"path": "aggregate.csv", "sha256": "bad"}]})
    publisher.close()


def test_analysis_publisher_interruption_keeps_previous_completion_unchanged(tmp_path, monkeypatch):
    import loraphymac.experiments.common.analysis_publication as publication

    root = tmp_path / "cohort"
    (root / "analysis" / "tables").mkdir(parents=True)
    (root / "analysis" / "figures").mkdir()
    (root / "analysis" / "analysis_manifest.json").write_text(json.dumps({"status": "PENDING"}))
    publisher = AnalysisPublisher(root)
    staged = publisher.tables_dir / "result.csv"
    staged.write_text("v\n1\n")
    initial = {"analysis_version": "v1", "source_cohort_id": "cohort",
               "inputs": [{"sha256": "input-1"}], "generator": {"sha256": "script-1"},
               "tables": [staged.relative_to(root).as_posix()],
               "outputs": [{"path": staged.relative_to(root).as_posix(),
                            "size_bytes": staged.stat().st_size, "sha256": _sha256(staged)}]}
    completed = publisher.publish(initial, [staged])
    publisher.close()
    canonical = Path(completed["analysis_manifest"])
    old_manifest = canonical.read_bytes()
    old_output = root / json.loads(old_manifest)["outputs"][0]["path"]
    old_output_bytes = old_output.read_bytes()

    revised = AnalysisPublisher(root)
    stage2 = revised.tables_dir / "result.csv"
    stage2.write_text("v\n2\n")
    revision = {**initial, "inputs": [{"sha256": "input-2"}],
                "tables": [stage2.relative_to(root).as_posix()],
                "outputs": [{"path": stage2.relative_to(root).as_posix(),
                             "size_bytes": stage2.stat().st_size, "sha256": _sha256(stage2)}]}
    def interrupted_link(_src, _dst):
        raise OSError("synthetic interrupted publish")
    monkeypatch.setattr(publication.os, "link", interrupted_link)
    with pytest.raises(OSError, match="interrupted publish"):
        revised.publish(revision, [stage2])
    revised.close()
    assert canonical.read_bytes() == old_manifest
    assert old_output.read_bytes() == old_output_bytes
    failed_manifests = list((root / "analysis" / "versions").glob("analysis-v*/analysis_manifest.json"))
    assert len(failed_manifests) == 1
    assert json.loads(failed_manifests[0].read_text())["status"] == "FAILED"

def test_active_reference_config_unchanged_by_grid_construction():
    config = Path(__file__).resolve().parents[2] / (
        "src/loraphymac/experiments/apc_terrain_sweep/configs/production_v1.json"
    )
    before = _sha256(config)
    from loraphymac.experiments.common.multiseed_pipeline import load_experiment_spec

    build_experiment_grid(load_experiment_spec(
        Path(__file__).resolve().parents[2]
        / "src/loraphymac/experiments/apc_terrain_sweep/configs/production_v1.json"
    ))
    assert _sha256(config) == before
