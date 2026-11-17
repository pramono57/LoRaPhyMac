from __future__ import annotations

import json
from pathlib import Path

import pytest

from loraphymac.experiments.common.multiseed_pipeline import file_sha256
from loraphymac.experiments.common.outputs import ExperimentArtifactPaths
from loraphymac.publications.paper_rapc_multihop_lora import run as publication_cli
from loraphymac.publications.paper_rapc_multihop_lora import workflow
from loraphymac.publications.provenance import (
    PUBLICATION_PROVENANCE_VERSION,
    PublicationProvenanceError,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _publication_fixture(tmp_path: Path) -> tuple[Path, ExperimentArtifactPaths, Path]:
    source = ExperimentArtifactPaths(tmp_path / "experiments" / "paper1_apc_v1").ensure()
    publication_dir = tmp_path / "publications" / "paper1_apc_v1"
    versions = {
        "pipeline_version": "multiseed_pipeline_v4",
        "simulator_api_version": "simulator_api_v1",
        "output_layout_version": "paper_artifacts_v1",
    }
    manifest = {
        **versions,
        "experiment_contract_version": "paper1_apc_contract_v1.0",
        "spec_hash": "4172e451a34ed789",
        "source_fingerprint": {
            "algorithm": "sha256",
            "files": {"src/loraphymac/simulator.py": "a" * 64},
        },
        "spec": {
            "experiment_id": "paper1_apc_v1",
            "base_config": {"experiment_profile": "paper1_apc_v1"},
            "factors": {
                "node_ptx_mode": ["fixed", "adaptive"],
                "relay_ptx_mode": ["fixed", "adaptive"],
                "terrain_delta_h": list(range(20, 161, 10)),
            },
            "seeds": [30, 36, 42, 48, 56],
        },
    }
    status = {
        **versions,
        "experiment_id": "paper1_apc_v1",
        "status": "PASS",
        "requested_run_count": 300,
        "passed_run_count_on_disk": 300,
        "seed_grid_complete": True,
        "status_counts": {"PASS": 300, "FAILED": 0, "SKIPPED": 0},
    }
    _write_json(source.manifest, manifest)
    _write_json(source.status, status)
    (source.aggregate / "core_metrics_long.csv").write_text(
        "metric,mean\ndelivery_by_horizon_percent,80.0\n",
        encoding="utf-8",
    )
    config = tmp_path / "publication.json"
    _write_json(
        config,
        {
            "publication_id": "paper_rapc_multihop_lora",
            "source_experiment_dir": str(source.experiment_dir),
            "output_dir": str(publication_dir),
            "options": {"comparison_terrain_delta_h": 90},
        },
    )
    return config, source, publication_dir


def test_publication_workflow_records_source_and_generated_artifact_provenance(
    tmp_path, monkeypatch
):
    config, source, publication_dir = _publication_fixture(tmp_path)
    protected_sources = (source.manifest, source.status, source.aggregate / "core_metrics_long.csv")
    source_hashes_before = {path: file_sha256(path) for path in protected_sources}

    def fake_tables(*, config):
        table = publication_dir / "tables" / "table_09.csv"
        table.parent.mkdir(parents=True, exist_ok=True)
        table.write_text("Mode,PDR\nAA,80.0\n", encoding="utf-8")
        return {"comparison": table, "table_09": table}

    def fake_figures(*, config):
        figure_dir = publication_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        png = figure_dir / "figure_01.png"
        pdf = figure_dir / "figure_01.pdf"
        png.write_bytes(b"png-evidence")
        pdf.write_bytes(b"pdf-evidence")
        return [png, pdf]

    monkeypatch.setattr(workflow, "build_publication_tables", fake_tables)
    monkeypatch.setattr(workflow, "generate_all_figures", fake_figures)

    summary = workflow.run_publication(config)

    assert summary["status"] == "PASS"
    assert summary["table_count"] == 1
    assert summary["figure_count"] == 2
    assert summary["artifact_count"] == 3
    manifest_path = publication_dir / "publication_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["publication_provenance_version"] == PUBLICATION_PROVENANCE_VERSION
    assert manifest["generator"]["loraphymac_version"] == "1.0.0"
    assert manifest["source_experiment"]["experiment_id"] == "paper1_apc_v1"
    assert manifest["source_experiment"]["requested_run_count"] == 300
    assert manifest["source_experiment"]["passed_run_count_on_disk"] == 300
    assert manifest["source_experiment"]["status_counts"]["FAILED"] == 0
    assert manifest["source_experiment"]["seed_grid_complete"] is True
    assert manifest["source_experiment"]["simulator_version"] is None
    assert manifest["source_experiment"]["spec_hash"] == "4172e451a34ed789"
    assert manifest["publication_config"]["sha256"] == file_sha256(config)
    assert manifest["artifact_counts"] == {"tables": 1, "figures": 2, "total": 3}
    for artifact in manifest["artifacts"]:
        output = publication_dir / artifact["path"]
        assert artifact["sha256"] == file_sha256(output)
        assert artifact["size_bytes"] == output.stat().st_size
    assert source_hashes_before == {path: file_sha256(path) for path in protected_sources}


def test_multi_source_publication_assembles_terrain_and_closed_loop_artifacts(tmp_path):
    config, terrain, publication_dir = _publication_fixture(tmp_path)
    terrain_records = []
    for relative in workflow.TERRAIN_SOURCE_TO_PUBLICATION:
        path = terrain.experiment_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"terrain:{relative}".encode())
        terrain_records.append({"path": relative, "sha256": file_sha256(path)})
    _write_json(
        terrain.experiment_dir / "analysis_manifest.json",
        {
            "status": "PASS",
            "experiment_id": "paper1_apc_v1",
            "core_metrics_sha256": file_sha256(terrain.aggregate / "core_metrics_long.csv"),
            "artifacts": terrain_records,
        },
    )

    closed = tmp_path / "experiments" / "apc_closed_loop_dh90_s42"
    _write_json(
        closed / "verification_report.json",
        {
            "status": "PASS",
            "experiment_profile": "apc_closed_loop_verification_v1",
            "experiment_contract_version": "apc_closed_loop_contract_v1.0",
            "run_id": "apc_closed_loop_dh90_s42",
            "seed": 42,
            "terrain_delta_h": 90.0,
            "simulator_version": "1.0.0",
            "simulator_api_version": "simulator_api_v1",
            "resolved_config_hash": "frozen",
            "checks": {
                "closed_loop_contract": True,
                "stage1_status": "PASS",
                "packet_fate_conservation_delta": 0,
                "collision_outcome_conservation_delta": 0,
            },
        },
    )
    for relative in workflow.CLOSED_LOOP_SOURCE_TO_PUBLICATION:
        path = closed / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"closed:{relative}".encode())

    _write_json(
        config,
        {
            "publication_id": "paper_rapc_multihop_lora",
            "sources": {
                "terrain_sweep": str(terrain.experiment_dir),
                "closed_loop_verification": str(closed),
            },
            "output_dir": str(publication_dir),
            "options": {"comparison_terrain_delta_h": 90},
        },
    )

    summary = workflow.run_publication(config)

    assert summary["status"] == "PASS"
    assert summary["table_count"] == 4
    assert summary["figure_count"] == 20
    manifest = json.loads((publication_dir / "publication_manifest.json").read_text())
    assert set(manifest["source_experiments"]) == {
        "terrain_sweep",
        "closed_loop_verification",
    }
    assert {artifact["source_role"] for artifact in manifest["artifacts"]} == {
        "terrain_sweep",
        "closed_loop_verification",
    }


def test_check_only_validates_without_writing_publication_files(tmp_path, monkeypatch):
    config, _, publication_dir = _publication_fixture(tmp_path)

    def unexpected_builder(**kwargs):
        raise AssertionError("artifact builder must not run during --check-only")

    monkeypatch.setattr(workflow, "build_publication_tables", unexpected_builder)
    monkeypatch.setattr(workflow, "generate_all_figures", unexpected_builder)

    summary = workflow.run_publication(config, check_only=True)

    assert summary["status"] == "READY"
    assert summary["check_only"] is True
    assert not publication_dir.exists()


def test_source_simulator_release_must_match_when_new_manifests_record_it(tmp_path):
    config, source, _ = _publication_fixture(tmp_path)
    manifest = json.loads(source.manifest.read_text(encoding="utf-8"))
    status = json.loads(source.status.read_text(encoding="utf-8"))
    manifest["simulator_version"] = "1.0.0"
    status["simulator_version"] = "1.0.1"
    _write_json(source.manifest, manifest)
    _write_json(source.status, status)

    with pytest.raises(
        PublicationProvenanceError,
        match="simulator_version differs",
    ):
        workflow.run_publication(config, check_only=True)


def test_incomplete_source_is_rejected_before_publication(tmp_path):
    config, source, publication_dir = _publication_fixture(tmp_path)
    status = json.loads(source.status.read_text(encoding="utf-8"))
    status["seed_grid_complete"] = False
    _write_json(source.status, status)

    with pytest.raises(PublicationProvenanceError, match="seed grid is incomplete"):
        workflow.run_publication(config)
    assert not publication_dir.exists()


def test_failed_regeneration_removes_stale_publication_manifest(tmp_path, monkeypatch):
    config, _, publication_dir = _publication_fixture(tmp_path)
    publication_dir.mkdir(parents=True)
    stale_manifest = publication_dir / "publication_manifest.json"
    stale_manifest.write_text('{"status": "PASS"}', encoding="utf-8")

    def fail_tables(**kwargs):
        raise ValueError("synthetic publication failure")

    monkeypatch.setattr(workflow, "build_publication_tables", fail_tables)

    with pytest.raises(ValueError, match="synthetic publication failure"):
        workflow.run_publication(config)
    assert not stale_manifest.exists()


def test_paper1_publication_cli_check_only_reports_ready(tmp_path, capsys):
    config, _, publication_dir = _publication_fixture(tmp_path)

    exit_code = publication_cli.main(["--config", str(config), "--check-only"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out)["status"] == "READY"
    assert captured.err == ""
    assert not publication_dir.exists()


def test_paper1_publication_cli_returns_failure_for_invalid_source(tmp_path, capsys):
    config, source, _ = _publication_fixture(tmp_path)
    source.status.unlink()

    exit_code = publication_cli.main(["--config", str(config), "--check-only"])

    captured = capsys.readouterr()
    assert exit_code == 1
    error = json.loads(captured.err)
    assert error["status"] == "FAILED"
    assert "experiment status" in error["error"]
