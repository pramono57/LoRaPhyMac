import json
from pathlib import Path

import pandas as pd
import pytest

from loraphymac.experiments.apc_terrain_profile_diagnostic.contract import (
    APC_TERRAIN_PROFILE_DIAGNOSTIC_PROFILE,
    APC_TERRAIN_PROFILE_DIAGNOSTIC_RUN_COUNT,
    APC_TERRAIN_PROFILE_DIAGNOSTIC_VERSION,
    diagnostic_run_tables,
    validate_diagnostic_outputs,
)
from loraphymac.experiments.apc_terrain_sweep.contract import (
    APC_TERRAIN_SWEEP_EXPECTED_RUN_COUNT,
    APC_TERRAIN_SWEEP_PROFILE,
    APC_TERRAIN_SWEEP_CONTRACT_VERSION,
)
from loraphymac.experiments.common.contract_registry import (
    experiment_config_violations,
    experiment_contract_version,
)
from loraphymac.experiments.common.multiseed_pipeline import (
    build_experiment_grid,
    load_experiment_spec,
)


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_CONFIG = (
    ROOT
    / "src/loraphymac/experiments/apc_terrain_sweep/configs/diagnostic_v1.json"
)
OFFICIAL_TERRAIN_CONFIG = (
    ROOT / "src/loraphymac/experiments/apc_terrain_sweep/configs/production_v1.json"
)


def test_official_terrain_cartesian_grid_remains_300_runs():
    spec = load_experiment_spec(OFFICIAL_TERRAIN_CONFIG)
    runs = build_experiment_grid(spec)
    assert spec.run_matrix is None
    assert len(runs) == APC_TERRAIN_SWEEP_EXPECTED_RUN_COUNT == 300
    assert len({run["run_id"] for run in runs}) == 300
    assert experiment_contract_version(APC_TERRAIN_SWEEP_PROFILE) == (
        APC_TERRAIN_SWEEP_CONTRACT_VERSION
    )


def test_explicit_diagnostic_matrix_resolves_exactly_ten_runs():
    spec = load_experiment_spec(DIAGNOSTIC_CONFIG)
    runs = build_experiment_grid(spec)
    assert len(runs) == APC_TERRAIN_PROFILE_DIAGNOSTIC_RUN_COUNT == 10
    assert len({run["run_id"] for run in runs}) == 10
    observed = {
        (
            run["scientific_config"]["node_ptx_mode"],
            run["scientific_config"]["relay_ptx_mode"],
            run["scientific_config"]["terrain_delta_h"],
            run["seed"],
        )
        for run in runs
    }
    expected = {
        (mode, mode, 90, seed)
        for mode in ("fixed", "adaptive")
        for seed in (30, 36, 42, 48, 56)
    }
    assert observed == expected
    for run in runs:
        config = run["scientific_config"]
        assert config["experiment_profile"] == APC_TERRAIN_PROFILE_DIAGNOSTIC_PROFILE
        assert config["node_rx_collision_enabled"] is True
        assert config["application_scheduling_mode"] == "independent_fifo_v1"
        assert config["application_rng_policy"] == "sha256_seed_node_v1"
        assert config["application_queue_policy"] == "unbounded_fifo_v1"
        assert config["application_horizon_policy"] == "right_censor_waiting_v1"


def test_diagnostic_contract_accepts_conditional_empty_tables():
    spec = load_experiment_spec(DIAGNOSTIC_CONFIG)
    config = build_experiment_grid(spec)[0]["scientific_config"]
    output_keys = set(diagnostic_run_tables(config))
    from loraphymac.experiments.apc_terrain_sweep.contract import (
        PAPER1_APC_REQUIRED_OUTPUT_TABLES,
    )

    outputs = {
        key: pd.DataFrame()
        for key in output_keys | set(PAPER1_APC_REQUIRED_OUTPUT_TABLES)
    }
    summary = {
        key: 0
        for key in (
            "packet_fate_conservation_delta",
            "collision_outcome_conservation_delta",
        )
    }
    summary["stage1_status"] = "PASS"
    from loraphymac.experiments.apc_terrain_sweep.contract import (
        PAPER1_APC_REQUIRED_SUMMARY_FIELDS,
    )

    for key in PAPER1_APC_REQUIRED_SUMMARY_FIELDS:
        summary.setdefault(key, 0)
    validate_diagnostic_outputs(config, summary, outputs)


def test_explicit_matrix_rejects_missing_combination(tmp_path):
    raw = json.loads(DIAGNOSTIC_CONFIG.read_text())
    raw["run_matrix"].pop()
    config_path = tmp_path / "missing-row.json"
    config_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="exactly 10 rows|exactly FF and AA"):
        load_experiment_spec(config_path)


def test_explicit_matrix_rejects_cross_mode_combination(tmp_path):
    raw = json.loads(DIAGNOSTIC_CONFIG.read_text())
    raw["run_matrix"][0]["relay_ptx_mode"] = "adaptive"
    config_path = tmp_path / "cross-mode.json"
    config_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="exactly FF and AA"):
        load_experiment_spec(config_path)


def test_diagnostic_contract_rejects_wrong_scheduling_and_official_is_unchanged(
    tmp_path,
):
    raw = json.loads(DIAGNOSTIC_CONFIG.read_text())
    raw["base_config"]["application_rng_policy"] = "legacy_global_random_v1"
    config_path = tmp_path / "_invalid_test.json"
    config_path.write_text(json.dumps(raw))
    try:
        with pytest.raises(ValueError, match="application_rng_policy"):
            load_experiment_spec(config_path)
    finally:
        config_path.unlink(missing_ok=True)

    assert experiment_contract_version(APC_TERRAIN_SWEEP_PROFILE) == (
        APC_TERRAIN_SWEEP_CONTRACT_VERSION
    )
    invalid_official = {
        "experiment_profile": APC_TERRAIN_SWEEP_PROFILE,
        "application_scheduling_mode": "independent_fifo_v1",
        "application_rng_policy": "sha256_seed_node_v1",
        "application_queue_policy": "unbounded_fifo_v1",
        "application_horizon_policy": "right_censor_waiting_v1",
        "node_rx_collision_enabled": True,
    }
    violations = experiment_config_violations(APC_TERRAIN_SWEEP_PROFILE, invalid_official)
    assert violations
    assert APC_TERRAIN_PROFILE_DIAGNOSTIC_VERSION.endswith("v1.0")
