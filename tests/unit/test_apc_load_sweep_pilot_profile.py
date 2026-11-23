from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from loraphymac.experiments.apc_load_sweep.contract import (
    APC_LOAD_SWEEP_PILOT_PROFILE,
    apc_load_sweep_pilot_config_violations,
)
from loraphymac.experiments.apc_load_sweep.run import DEFAULT_CONFIG
from loraphymac.experiments.apc_load_sweep.standard_analysis import build_load_sweep_analysis
from loraphymac.experiments.common.contract_registry import validate_experiment_spec_contract
from loraphymac.experiments.common.multiseed_pipeline import (
    build_experiment_grid,
    load_experiment_spec,
)
from loraphymac.experiments.common.outputs import SWEEP_OUTPUT_LAYOUT_VERSION
from loraphymac.experiments.common.sweep_standard import standard_manifest_fields, write_cohort_readme
from loraphymac.experiments.common import cli


def _pilot_spec():
    spec = load_experiment_spec(DEFAULT_CONFIG)
    return cli._select_sweep_profile(spec, "pilot", DEFAULT_CONFIG)


def test_production_contract_remains_120_and_rejects_two_run_grid():
    spec = load_experiment_spec(DEFAULT_CONFIG)
    assert len(build_experiment_grid(spec)) == 120
    reduced = replace(
        spec,
        factors={
            "node_num": [50],
            "node_period": [90],
            "node_ptx_mode": ["adaptive", "fixed"],
        },
        seeds=[42],
    )
    with pytest.raises(ValueError, match="Standard load factors"):
        validate_experiment_spec_contract(reduced)


def test_pilot_profile_is_exact_two_run_grid_and_rejects_bad_parameters():
    spec = _pilot_spec()
    grid = build_experiment_grid(spec)
    assert spec.base_config["experiment_profile"] == APC_LOAD_SWEEP_PILOT_PROFILE
    assert spec.base_config["sim_time"] == 300
    assert len(grid) == 2
    assert {run["factors"]["node_ptx_mode"] for run in grid} == {"adaptive", "fixed"}
    assert len({run["run_id"] for run in grid}) == 2
    assert all(run["seed"] == 42 for run in grid)
    for run in grid:
        config = dict(run["scientific_config"])
        assert apc_load_sweep_pilot_config_violations(config) == []
    bad = dict(grid[0]["scientific_config"], sim_time=301)
    assert "pilot sim_time must be 300 seconds" in apc_load_sweep_pilot_config_violations(bad)
    with pytest.raises(ValueError, match="unexpected parameter-grid"):
        cli._select_sweep_profile(
            replace(load_experiment_spec(DEFAULT_CONFIG), factors={"node_num": [99]}),
            "pilot",
            DEFAULT_CONFIG,
        )
    with pytest.raises(ValueError, match="requires explicit --profile pilot"):
        cli._select_sweep_profile(
            replace(spec, base_config={**spec.base_config, "experiment_profile": APC_LOAD_SWEEP_PILOT_PROFILE}),
            "production",
            DEFAULT_CONFIG,
        )


def test_pilot_dry_run_uses_external_root_without_reserving(tmp_path, capsys):
    pilot_root = tmp_path / "pilot-output"
    result = cli.run_standard_cli(
        description="pilot dry run",
        project_root=tmp_path,
        scenario="load_sweep",
        default_config=DEFAULT_CONFIG,
        argv=["--profile", "pilot", "--output-root", str(pilot_root), "--dry-run", "--quiet"],
    )
    report = json.loads(capsys.readouterr().out)
    assert result == 0
    assert report["run_count"] == 2
    assert report["sweep_profile"] == "pilot"
    assert [
        (run["factors"]["node_num"], run["factors"]["node_period"],
         run["factors"]["node_ptx_mode"], run["seed"], run["scientific_config"]["sim_time"])
        for run in report["runs"]
    ] == [
        (50, 90, "adaptive", 42, 300),
        (50, 90, "fixed", 42, 300),
    ]
    assert report["cohort_reserved"] is False
    assert "rNN_" in report["proposed_cohort_id"]
    assert str(pilot_root / "APC_LOAD_SWEEP") in report["experiment_dir"]
    assert not pilot_root.exists()
    assert not (tmp_path / "artifacts").exists()
    with pytest.raises(ValueError, match="outside repository artifacts"):
        cli.run_standard_cli(
            description="pilot dry run",
            project_root=tmp_path,
            scenario="load_sweep",
            default_config=DEFAULT_CONFIG,
            argv=["--profile", "pilot", "--output-root", str(tmp_path / "artifacts"), "--dry-run"],
        )
    with pytest.raises(ValueError, match="requires --output-root"):
        cli.run_standard_cli(
            description="pilot dry run",
            project_root=tmp_path,
            scenario="load_sweep",
            default_config=DEFAULT_CONFIG,
            argv=["--profile", "pilot", "--dry-run", "--quiet"],
        )


def test_production_and_pilot_dry_runs_share_the_same_execution_path(tmp_path, monkeypatch, capsys):
    original = cli.run_experiment
    observed = []

    def capture(spec, **kwargs):
        observed.append((spec.base_config["experiment_profile"], kwargs["dry_run"]))
        return original(spec, **kwargs)

    monkeypatch.setattr(cli, "run_experiment", capture)
    cli.run_standard_cli(
        description="production dry run", project_root=tmp_path, scenario="load_sweep",
        default_config=DEFAULT_CONFIG, argv=["--dry-run", "--quiet"],
    )
    capsys.readouterr()
    cli.run_standard_cli(
        description="pilot dry run", project_root=tmp_path, scenario="load_sweep",
        default_config=DEFAULT_CONFIG,
        argv=["--profile", "pilot", "--output-root", str(tmp_path / "pilot"), "--dry-run", "--quiet"],
    )
    capsys.readouterr()
    assert observed == [
        ("apc_load_sweep_standard_v1", True),
        (APC_LOAD_SWEEP_PILOT_PROFILE, True),
    ]


def test_pilot_profile_is_explicit_in_manifest_and_readme(tmp_path):
    spec = _pilot_spec()
    grid = build_experiment_grid(spec)
    cohort = tmp_path / "load_sweep_v1_r01_20261123"
    cohort.mkdir()
    fields = standard_manifest_fields(
        spec,
        expected_run_count=2,
        run_ids=[run["run_id"] for run in grid],
        git_head=None,
    )
    assert fields["sweep_profile"] == "pilot"
    assert fields["is_non_production_pilot"] is True
    (cohort / "experiment_manifest.json").write_text(json.dumps(fields))
    write_cohort_readme(
        cohort, spec=spec, run_count=2, seed_list=[42], source_fingerprint_hash="fixture"
    )
    readme = (cohort / "README.md").read_text()
    assert "Selected sweep profile: `pilot`" in readme
    assert "Non-production integration evidence" in readme


def test_pilot_analysis_reports_n1_statistics_unavailable_and_is_immutable(tmp_path, monkeypatch):
    root = tmp_path / "load_sweep_v1_r01_20261123"
    (root / "aggregate").mkdir(parents=True)
    (root / "runs").mkdir()
    rows = []
    run_ids = []
    for mode, node_mode in (("AA", "adaptive"), ("FF", "fixed")):
        run_id = f"pilot-{mode}-s42"
        run_ids.append(run_id)
        run_dir = root / "runs" / run_id
        tables = run_dir / "tables"
        tables.mkdir(parents=True)
        (run_dir / "resolved_config.json").write_text(json.dumps({
            "run_id": run_id,
            "seed": 42,
            "factors": {"node_ptx_mode": node_mode, "node_num": 50, "node_period": 90},
            "scientific_config": {"relay_ptx_mode": node_mode},
            "sweep_profile": "pilot",
        }))
        (run_dir / "scientific_output_manifest.json").write_text("{}")
        for name in ("node_tx_duty_cycle.csv", "channel_occupancy.csv", "apc_feedback_lifecycle.csv"):
            (tables / name).write_text("run_id,value\n" + run_id + ",1\n")
        rows.append({
            "run_id": run_id, "apc_mode": mode, "node_ptx_mode": node_mode,
            "node_num": 50, "node_period": 90, "seed": 42,
            "pdr_percent": 90.0, "energy_per_delivered_packet_J": 1.2,
            "total_network_energy_J": 100.0, "avg_e2e_delay_ms": 250.0,
        })
    pd.DataFrame(rows).to_csv(root / "aggregate" / "all_runs_core_summary.csv", index=False)
    profile_spec = {
        "base_config": {"experiment_profile": APC_LOAD_SWEEP_PILOT_PROFILE},
        "run_annotations": {"sweep_profile": "pilot"},
    }
    (root / "experiment_manifest.json").write_text(json.dumps({
        "cohort_id": root.name,
        "output_layout_version": SWEEP_OUTPUT_LAYOUT_VERSION,
        "sweep_profile": "pilot",
        "is_non_production_pilot": True,
        "spec": profile_spec,
        "run_ids": run_ids,
        "expected_run_count": 2,
        "parameter_grid": {
            "factors": {"node_num": [50], "node_period": [90], "node_ptx_mode": ["adaptive", "fixed"]},
            "seeds": [42],
        },
    }))
    (root / "experiment_status.json").write_text(json.dumps({"status": "COMPLETE", "seed_grid_complete": True}))
    monkeypatch.setattr(
        "loraphymac.experiments.apc_load_sweep.standard_analysis._read_verified_scientific_tables",
        lambda _path: {},
    )
    result = build_load_sweep_analysis(root)
    manifest_path = Path(result["analysis_manifest"])
    analysis_manifest = json.loads(manifest_path.read_text())
    assert analysis_manifest["sweep_profile"] == "pilot"
    assert analysis_manifest["is_non_production_pilot"] is True
    assert analysis_manifest["omitted_statistics"]
    summary = pd.read_csv(root / analysis_manifest["tables"][0])
    assert set(summary["n"]) == {1}
    assert set(summary["ci_method"]) == {"not_estimable_n_lt_2"}
    assert summary["std"].isna().all() and summary["ci_lower"].isna().all()
    completed = [manifest_path, *(root / output["path"] for output in analysis_manifest["outputs"])]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in completed}
    repeated = build_load_sweep_analysis(root)
    assert repeated["reused"] is True
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in completed}
