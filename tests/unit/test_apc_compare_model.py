from pathlib import Path

import pytest

from loraphymac.experiments.apc_compare_model.run import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    build_model_experiment_spec,
    load_comparison_definition,
    load_model_definition,
    planned_grid,
)


pytestmark = pytest.mark.unit


@pytest.mark.extended_research
def test_comparison_config_defines_separate_corrected_75_cell_dbpc_grid():
    comparison = load_comparison_definition(DEFAULT_CONFIG)
    spec = build_model_experiment_spec(DEFAULT_CONFIG, model_id="islam_dbpc_itm0")
    grid = planned_grid(DEFAULT_CONFIG, model_id="islam_dbpc_itm0")

    assert comparison.planned_execution_model_ids == ("islam_dbpc_itm0",)
    assert len(grid) == 75
    assert spec.base_config["energy_accounting_mode"] == "corrected"
    assert spec.base_config["node_ptx_mode"] == "fixed"
    assert spec.base_config["relay_ptx_mode"] == "fixed"
    assert spec.output_dir == PROJECT_ROOT / "artifacts/experiments/apc_compare_model"
    assert "apc_terrain_sweep" not in str(spec.output_dir)


@pytest.mark.extended_research
def test_model_identity_and_calibration_are_recorded_in_scientific_config():
    spec = build_model_experiment_spec(DEFAULT_CONFIG, model_id="islam_dbpc_itm0")

    assert spec.base_config["tx_power_model_id"] == "islam_dbpc_itm0"
    assert spec.base_config["tx_power_policy_id"] == "islam_algorithm2_fixed_sf_ptx_v1"
    assert spec.base_config["tx_power_calibration_id"].endswith("fixed_sf7_v1")
    assert len(spec.base_config["tx_power_calibration_manifest_sha256"]) == 64


@pytest.mark.extended_research
def test_itm6_sensitivity_uses_the_same_immutable_bundle():
    primary = build_model_experiment_spec(DEFAULT_CONFIG, model_id="islam_dbpc_itm0")
    sensitivity = build_model_experiment_spec(DEFAULT_CONFIG, model_id="islam_dbpc_itm6")

    assert sensitivity.base_config["tx_power_calibration_margin_db"] == 6
    assert sensitivity.base_config["tx_power_calibration_id"] == primary.base_config[
        "tx_power_calibration_id"
    ]
    assert sensitivity.base_config["tx_power_calibration_manifest_sha256"] == (
        primary.base_config["tx_power_calibration_manifest_sha256"]
    )


def test_rapc_model_routes_to_legacy_aa_without_dbpc_calibration():
    comparison = load_comparison_definition(DEFAULT_CONFIG)
    model = load_model_definition(comparison.models["rapc"])
    spec = build_model_experiment_spec(DEFAULT_CONFIG, model_id="rapc")

    assert model.policy_id == "legacy"
    assert spec.base_config["node_ptx_mode"] == "adaptive"
    assert spec.base_config["relay_ptx_mode"] == "adaptive"
    assert "tx_power_calibration_id" not in spec.base_config


def test_unknown_model_id_fails_explicitly():
    with pytest.raises(ValueError, match="Unknown model ID"):
        build_model_experiment_spec(DEFAULT_CONFIG, model_id="unknown")


def test_model_files_are_under_the_generic_experiment_folder():
    comparison = load_comparison_definition(DEFAULT_CONFIG)
    assert all(
        Path(path).is_relative_to(DEFAULT_CONFIG.parent) for path in comparison.models.values()
    )
