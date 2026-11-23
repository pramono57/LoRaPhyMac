from __future__ import annotations

import json
from pathlib import Path

import pytest

from loraphymac.experiments.apc_load_sweep import run
from loraphymac.experiments.common.multiseed_pipeline import load_experiment_spec


CONFIG_DIR = Path(__file__).resolve().parents[2] / "src/loraphymac/experiments/apc_load_sweep/configs"


def _legacy_config(tmp_path: Path, *, invalid: bool = False) -> Path:
    source = json.loads((CONFIG_DIR / "ff_full_factorial_v1.json").read_text())
    source["experiment_id"] = "legacy_cli_dispatch_fixture"
    source["output_dir"] = str(tmp_path / "legacy-output")
    if invalid:
        source["factors"]["node_num"] = [999]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


def test_default_cli_runs_standard_120_run_dry_run_without_reservation(capsys):
    result = run.main(["--dry-run", "--quiet"])
    report = json.loads(capsys.readouterr().out)
    assert result == 0
    assert report["run_count"] == 120
    assert report["sweep_profile"] == "production"
    assert report["cohort_reserved"] is False
    assert not Path(report["experiment_dir"]).exists()


def test_explicit_standard_configuration_uses_standard_lifecycle(monkeypatch, tmp_path):
    calls = []

    def standard(**kwargs):
        calls.append(("standard", kwargs["argv"]))
        return 0

    def legacy(**kwargs):
        calls.append(("legacy", kwargs["argv"]))
        return 1

    monkeypatch.setattr(run, "run_standard_cli", standard)
    monkeypatch.setattr(run, "run_cli", legacy)
    config = CONFIG_DIR / "standard_v1.json"
    assert run.main([str(config), "--dry-run"]) == 0
    assert calls == [("standard", [str(config), "--dry-run"])]


def test_pilot_output_root_is_not_mistaken_for_config(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        run,
        "run_standard_cli",
        lambda **kwargs: calls.append(kwargs["argv"]) or 0,
    )
    monkeypatch.setattr(
        run,
        "run_cli",
        lambda **kwargs: pytest.fail("pilot profile was routed to legacy CLI"),
    )
    argv = ["--profile", "pilot", "--output-root", str(tmp_path / "pilot"), "--dry-run"]
    assert run.main(argv) == 0
    assert calls == [argv]


def test_valid_legacy_config_uses_legacy_lifecycle(monkeypatch, tmp_path):
    calls = []
    config = _legacy_config(tmp_path)
    monkeypatch.setattr(
        run,
        "run_cli",
        lambda **kwargs: calls.append(("legacy", kwargs["argv"])) or 0,
    )
    monkeypatch.setattr(
        run,
        "run_standard_cli",
        lambda **kwargs: calls.append(("standard", kwargs["argv"])) or 1,
    )
    assert run.main([str(config), "--dry-run"]) == 0
    assert calls == [("legacy", [str(config), "--dry-run"])]


def test_legacy_dry_run_preserves_grid_and_creates_no_output(tmp_path, capsys):
    config = _legacy_config(tmp_path)
    assert run.main([str(config), "--dry-run", "--quiet"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["run_count"] == 60
    assert report["output_layout_version"] == "paper_artifacts_v1"
    assert not (tmp_path / "legacy-output" / "legacy_cli_dispatch_fixture").exists()


def test_invalid_legacy_config_is_rejected_without_output(tmp_path):
    config = _legacy_config(tmp_path, invalid=True)
    with pytest.raises(ValueError, match="Full-factorial factors must be exactly"):
        run.main([str(config), "--dry-run", "--quiet"])
    assert not (tmp_path / "legacy-output" / "legacy_cli_dispatch_fixture").exists()


def test_invalid_standard_config_never_falls_back_to_legacy(monkeypatch, tmp_path):
    config = json.loads((CONFIG_DIR / "standard_v1.json").read_text())
    config["base_config"]["experiment_profile"] = "apc_load_sweep_standard_v1"
    config["factors"]["node_num"] = [999]
    path = tmp_path / "invalid-standard.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    def unexpected_legacy(**kwargs):
        raise AssertionError("invalid Standard configuration was routed to legacy")

    monkeypatch.setattr(run, "run_cli", unexpected_legacy)
    with pytest.raises(ValueError, match="Standard load factors"):
        run.main([str(path), "--dry-run"])


def test_explicit_pilot_rejects_legacy_configuration(tmp_path):
    config = _legacy_config(tmp_path)
    with pytest.raises(ValueError, match="only be selected from an APC load configuration"):
        run.main(
            [
                str(config),
                "--profile",
                "pilot",
                "--output-root",
                str(tmp_path / "isolated-pilot"),
                "--dry-run",
            ]
        )


def test_explicit_production_does_not_silently_route_legacy(tmp_path):
    config = _legacy_config(tmp_path)
    with pytest.raises(ValueError, match="Production load profile requires"):
        run.main([str(config), "--profile", "production", "--dry-run"])


def test_standard_layout_contract_remains_sweep_cohort_v1():
    spec = load_experiment_spec(CONFIG_DIR / "standard_v1.json")
    assert spec.output_layout_version == "sweep_cohort_v1"
