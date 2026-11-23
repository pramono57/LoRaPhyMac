from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import loraphymac.experiments.apc_terrain_sweep.postprocessing as terrain_analysis
from loraphymac.experiments.common.multiseed_pipeline import build_experiment_grid
from loraphymac.experiments.apc_feedback_robustness_sweep.reporting import (
    CohortCheck,
    RunBundle,
)
from loraphymac.experiments.common.outputs import SWEEP_OUTPUT_LAYOUT_VERSION


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_cohort(root: Path, run_ids: list[str], factors: dict, seeds: list[int]) -> None:
    (root / "runs").mkdir(parents=True)
    (root / "aggregate").mkdir()
    (root / "analysis" / "figures").mkdir(parents=True)
    (root / "analysis" / "tables").mkdir()
    (root / "aggregate" / "core_metrics_long.csv").write_text("metric,value\npdr_percent,90\n")
    (root / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "cohort_id": root.name,
                "output_layout_version": SWEEP_OUTPUT_LAYOUT_VERSION,
                "run_ids": run_ids,
                "expected_run_count": len(run_ids),
                "parameter_grid": {"factors": factors, "seeds": seeds},
                "analysis_status": "PENDING",
            }
        )
    )
    (root / "experiment_status.json").write_text(
        json.dumps(
            {
                "experiment_id": root.name,
                "status": "COMPLETE",
                "seed_grid_complete": True,
                "requested_run_count": len(run_ids),
                "passed_run_count_on_disk": len(run_ids),
            }
        )
    )


def _write_run(root: Path, run_id: str, factors: dict, scientific: dict, seed: int) -> Path:
    run_dir = root / "runs" / run_id
    run_dir.mkdir()
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "seed": seed,
                "factors": factors,
                "scientific_config": scientific,
            }
        )
    )
    tables = run_dir / "tables"
    tables.mkdir()
    core = tables / "core_summary.csv"
    core.write_text(f"run_id,seed\n{run_id},{seed}\n")
    manifest = run_dir / "scientific_output_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "COMPLETE",
                "outputs": [
                    {
                        "path": "tables/core_summary.csv",
                        "size_bytes": core.stat().st_size,
                        "sha256": _sha256(core),
                    }
                ],
            }
        )
    )
    return manifest


@pytest.mark.unreleased_standard
def test_synthetic_terrain_analysis_publishes_manifested_outputs(tmp_path, monkeypatch):
    root = tmp_path / "terrain_v1_r01_20261123"
    modes = [
        ("adaptive", "adaptive"),
        ("adaptive", "fixed"),
        ("fixed", "adaptive"),
        ("fixed", "fixed"),
    ]
    run_ids = [f"terrain-{i}" for i in range(len(modes))]
    _make_cohort(
        root,
        run_ids,
        {
            "node_ptx_mode": ["adaptive", "fixed"],
            "relay_ptx_mode": ["adaptive", "fixed"],
            "terrain_delta_h": [90],
        },
        [30],
    )
    for run_id, (node_mode, relay_mode) in zip(run_ids, modes, strict=True):
        _write_run(
            root,
            run_id,
            {
                "node_ptx_mode": node_mode,
                "relay_ptx_mode": relay_mode,
                "terrain_delta_h": 90,
            },
            {
                "node_ptx_mode": node_mode,
                "relay_ptx_mode": relay_mode,
                "terrain_delta_h": 90,
                "seed": 30,
            },
            30,
        )

    def make_table(_experiment, *, publication_dir, **_kwargs):
        publication_dir.mkdir(parents=True, exist_ok=True)
        output = publication_dir / "terrain_summary.csv"
        output.write_text("mode,value\nAA,1\n")
        return {"terrain_summary": output}

    def make_figures(_experiment, *, publication_dir, **_kwargs):
        publication_dir.mkdir(parents=True, exist_ok=True)
        output = publication_dir / "terrain_profile.png"
        output.write_bytes(b"synthetic PNG fixture")
        return [output]

    monkeypatch.setattr(terrain_analysis, "_read_verified_scientific_tables", lambda _path: {})
    monkeypatch.setattr(terrain_analysis, "build_publication_tables", make_table)
    monkeypatch.setattr(terrain_analysis, "generate_all_figures", make_figures)
    result = terrain_analysis.build_terrain_sweep_artifacts(experiment_dir=root)
    manifest = json.loads(Path(result["analysis_manifest"]).read_text())
    assert manifest["status"] == "COMPLETE"
    assert len(manifest["artifacts"]) == 2
    for item in manifest["artifacts"]:
        output = root / item["path"]
        assert output.is_file() and _sha256(output) == item["sha256"]
    assert result["analysis_manifest"].endswith("analysis/analysis_manifest.json")
    immutable = [
        Path(result["analysis_manifest"]),
        *(root / item["path"] for item in manifest["artifacts"]),
    ]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in immutable}
    repeated = terrain_analysis.build_terrain_sweep_artifacts(experiment_dir=root)
    assert repeated["reused"] is True
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in immutable}


@pytest.mark.extended_research
def test_synthetic_compare_analysis_keeps_five_model_cohort_and_filters_publication(
    tmp_path, monkeypatch
):
    from loraphymac.experiments.apc_compare_model_terrain_sweep import (
        postprocessing as compare_analysis,
    )

    from loraphymac.experiments.apc_compare_model_terrain_sweep.run import (
        STANDARD_CONFIG,
        _build_standard_cohort_spec,
        build_model_specs,
        load_sweep_definition,
        model_label,
    )

    definition = load_sweep_definition(STANDARD_CONFIG)
    cohort_spec = _build_standard_cohort_spec(
        definition, build_model_specs(STANDARD_CONFIG)
    )
    cohort_spec = replace(
        cohort_spec,
        experiment_id="compare_model_sweep_v1_r01_20261123",
        output_dir=tmp_path / "artifacts" / "experiments" / "APC_COMPARE_MODEL_SWEEP",
    )
    grid = build_experiment_grid(cohort_spec)
    root = cohort_spec.output_dir / cohort_spec.experiment_id
    run_ids = [run["run_id"] for run in grid]
    _make_cohort(root, run_ids, {}, [])
    (root / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "cohort_id": cohort_spec.experiment_id,
                "output_layout_version": SWEEP_OUTPUT_LAYOUT_VERSION,
                "run_ids": run_ids,
                "expected_run_count": len(run_ids),
                "parameter_grid": {
                    "factors": {},
                    "seeds": [],
                    "run_matrix": cohort_spec.run_matrix,
                },
                "analysis_status": "PENDING",
            }
        )
    )
    rows = {}
    for run in grid:
        model_id = str(run["factors"]["model_id"])
        terrain = int(run["factors"]["terrain_delta_h"])
        seed = int(run["seed"])
        _write_run(root, run["run_id"], run["factors"], run["scientific_config"], seed)
        rows[run["run_id"]] = (model_id, terrain, seed)
    monkeypatch.setattr(compare_analysis, "_read_verified_scientific_tables", lambda _path: {})

    def fake_metrics(
        *, model_id, run_dir, expected_seed, expected_terrain, expected_routing_strategy
    ):
        expected_model, terrain, seed = rows[run_dir.name]
        assert (expected_model, terrain, seed) == (model_id, expected_terrain, expected_seed)
        return (
            {
                "model_id": model_id,
                "model_label": model_label(model_id, definition.experiment_profile),
                "terrain_delta_h": terrain,
                "seed": seed,
                "routing_strategy": expected_routing_strategy,
                "pdr_percent": 90.0 + seed / 100,
                "energy_per_delivered_packet_J": 1.0 + terrain / 1000,
                "avg_e2e_delay_ms": 200.0 + seed,
                "total_network_energy_J": 100.0 + seed,
            },
            {"model_id": model_id, "terrain_delta_h": terrain, "seed": seed},
        )

    monkeypatch.setattr(compare_analysis, "_run_metrics", fake_metrics)
    monkeypatch.setattr(
        compare_analysis,
        "_validate_matched_cells",
        lambda frame, _models: frame.assign(validation_pass=True, failed_checks=""),
    )
    result = compare_analysis.build_comparison_artifacts(STANDARD_CONFIG, experiment_dir=root)
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert len(manifest["inputs"]) == 375
    assert manifest["model_ids"] == list(definition.models)
    assert len(manifest["included_model_variants"]) == 4
    assert result["validated_run_count"] == 300
    assert {Path(item).name for item in result["figures"]} >= {
        "pdr_vs_terrain.png",
        "energy_per_delivered_packet_vs_terrain.png",
        "avg_e2e_delay_vs_terrain.png",
    }
    assert len({run["factors"]["model_id"] for run in grid}) == 5
    immutable = [
        Path(result["manifest"]),
        *(Path(path) for path in result["tables"] + result["figures"]),
    ]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in immutable}
    repeated = compare_analysis.build_comparison_artifacts(STANDARD_CONFIG, experiment_dir=root)
    assert repeated["reused"] is True
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in immutable}


@pytest.mark.unreleased_standard
def test_synthetic_feedback_analysis_publishes_complete_outputs(tmp_path, monkeypatch):
    from loraphymac.experiments.apc_feedback_robustness_sweep import (
        standard_analysis as feedback_analysis,
    )

    root = tmp_path / "feedback_v1_r01_20261123"
    run_ids = [f"feedback-{index:02d}" for index in range(30)]
    factors = {
        "node_period": [30, 90],
        "feedback_additional_loss_probability": [0.0, 0.1, 0.3],
    }
    _make_cohort(root, run_ids, factors, [30, 36, 42, 48, 56])
    bundles = []
    for index, run_id in enumerate(run_ids):
        period = (30, 90)[index // 15]
        loss = (0.0, 0.1, 0.3)[(index // 5) % 3]
        seed = (30, 36, 42, 48, 56)[index % 5]
        source_manifest = _write_run(
            root,
            run_id,
            {},
            {
                "node_period": period,
                "feedback_additional_loss_probability": loss,
                "seed": seed,
            },
            seed,
        )
        bundles.append(
            RunBundle(
                run_id=run_id,
                config={
                    "node_period": period,
                    "feedback_additional_loss_probability": loss,
                    "seed": seed,
                },
                core=pd.DataFrame(),
                tables={},
                source_fingerprint="synthetic",
                config_fingerprint="synthetic",
                manifest_sha256=_sha256(source_manifest),
            )
        )
    check = CohortCheck(30, 30, True, (), (), (), ())
    frames = {
        "scenario_summary.csv": pd.DataFrame(
            [
                {
                    "node_period": period,
                    "feedback_additional_loss_probability": loss,
                    "metric": "packet_fate_terminal_drop_count",
                    "mean": 2.0 + loss,
                    "ci95_lower": 1.0 + loss,
                    "ci95_upper": 3.0 + loss,
                }
                for period in (30, 90)
                for loss in (0.0, 0.1, 0.3)
            ]
        ),
        "paired_loss_effects.csv": pd.DataFrame([{"period": 30, "effect": 0.1}]),
        "feedback_metrics_by_run.csv": pd.DataFrame([{"run_id": run_ids[0], "value": 1.0}]),
        "run_metrics.csv": pd.DataFrame([{"run_id": run_ids[0], "value": 1.0}]),
    }
    monkeypatch.setattr(feedback_analysis, "load_cohort", lambda _root: (bundles, check))
    monkeypatch.setattr(
        feedback_analysis, "aggregate_complete_cohort", lambda _bundles, _check: frames
    )
    result = feedback_analysis.build_feedback_robustness_analysis(root)
    manifest = json.loads(Path(result["analysis_manifest"]).read_text())
    assert manifest["status"] == "COMPLETE"
    assert len(manifest["inputs"]) == 30
    assert len(manifest["outputs"]) == 5
    for item in manifest["outputs"]:
        output = root / item["path"]
        assert output.is_file() and _sha256(output) == item["sha256"]
    immutable = [
        Path(result["analysis_manifest"]),
        *(root / item["path"] for item in manifest["outputs"]),
    ]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in immutable}
    repeated = feedback_analysis.build_feedback_robustness_analysis(root)
    assert repeated["reused"] is True
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in immutable}
