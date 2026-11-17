import json
from pathlib import Path

import pandas as pd
import pytest

from loraphymac.experiments.apc_terrain_profile_diagnostic.contract import (
    APC_TERRAIN_PROFILE_DIAGNOSTIC_RUN_COUNT,
)
from loraphymac.experiments.apc_terrain_profile_diagnostic.full_sweep_contract import (
    APC_TERRAIN_PROFILE_FULL_SWEEP_PROFILE,
    APC_TERRAIN_PROFILE_FULL_SWEEP_RUN_COUNT,
    full_sweep_run_tables,
    validate_full_sweep_outputs,
)
from loraphymac.experiments.apc_terrain_sweep.contract import (
    APC_TERRAIN_SWEEP_CONTRACT_VERSION,
    APC_TERRAIN_SWEEP_EXPECTED_RUN_COUNT,
    APC_TERRAIN_SWEEP_PROFILE,
    APC_TERRAIN_SWEEP_SEEDS,
    APC_TERRAIN_SWEEP_TERRAIN_DELTA_H_VALUES,
    PAPER1_APC_REQUIRED_OUTPUT_TABLES,
    PAPER1_APC_REQUIRED_SUMMARY_FIELDS,
)
from loraphymac.experiments.common.contract_registry import (
    experiment_contract_version,
    experiment_run_tables,
)
from loraphymac.experiments.common.multiseed_pipeline import (
    build_experiment_grid,
    load_experiment_spec,
)


ROOT = Path(__file__).resolve().parents[2]
FULL_CONFIG = (
    ROOT
    / "artifacts/diagnostic_runs_20261111/configs/"
    "apc_terrain_profile_full_sweep_v1.json"
)
OFFICIAL_CONFIG = (
    ROOT / "src/loraphymac/experiments/apc_terrain_sweep/configs/production_v1.json"
)
DIAGNOSTIC_CONFIG = (
    ROOT
    / "artifacts/diagnostic_runs_20261111/configs/"
    "apc_terrain_profile_diagnostic_v1.json"
)


@pytest.mark.extended_research
def test_full_sweep_matrix_has_300_unique_resolved_runs():
    spec = load_experiment_spec(FULL_CONFIG)
    runs = build_experiment_grid(spec)
    assert len(runs) == APC_TERRAIN_PROFILE_FULL_SWEEP_RUN_COUNT == 300
    assert len({row["run_id"] for row in runs}) == 300
    counts = {}
    observed = set()
    for row in runs:
        config = row["scientific_config"]
        mode = (config["node_ptx_mode"], config["relay_ptx_mode"])
        counts[mode] = counts.get(mode, 0) + 1
        observed.add((*mode, config["terrain_delta_h"], row["seed"]))
        assert config["experiment_profile"] == APC_TERRAIN_PROFILE_FULL_SWEEP_PROFILE
        assert config["node_rx_collision_enabled"] is True
        assert config["application_scheduling_mode"] == "independent_fifo_v1"
        assert config["application_rng_policy"] == "sha256_seed_node_v1"
        assert config["application_queue_policy"] == "unbounded_fifo_v1"
        assert config["application_horizon_policy"] == "right_censor_waiting_v1"
    assert counts == {
        ("fixed", "fixed"): 75,
        ("fixed", "adaptive"): 75,
        ("adaptive", "fixed"): 75,
        ("adaptive", "adaptive"): 75,
    }
    assert observed == {
        (node, relay, terrain, seed)
        for node, relay in (
            ("fixed", "fixed"),
            ("fixed", "adaptive"),
            ("adaptive", "fixed"),
            ("adaptive", "adaptive"),
        )
        for terrain in APC_TERRAIN_SWEEP_TERRAIN_DELTA_H_VALUES
        for seed in APC_TERRAIN_SWEEP_SEEDS
    }


@pytest.mark.extended_research
def test_official_cartesian_and_ten_run_diagnostic_contracts_are_unchanged():
    official = build_experiment_grid(load_experiment_spec(OFFICIAL_CONFIG))
    diagnostic = build_experiment_grid(load_experiment_spec(DIAGNOSTIC_CONFIG))
    assert len(official) == APC_TERRAIN_SWEEP_EXPECTED_RUN_COUNT == 300
    assert len(diagnostic) == APC_TERRAIN_PROFILE_DIAGNOSTIC_RUN_COUNT == 10
    assert experiment_contract_version(APC_TERRAIN_SWEEP_PROFILE) == (
        APC_TERRAIN_SWEEP_CONTRACT_VERSION
    )


@pytest.mark.extended_research
def test_full_sweep_contract_rejects_missing_duplicate_or_wrong_policy(tmp_path):
    raw = json.loads(FULL_CONFIG.read_text())
    raw["run_matrix"].pop()
    missing_path = tmp_path / "missing.json"
    missing_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="300 rows|missing, duplicating"):
        load_experiment_spec(missing_path)

    raw = json.loads(FULL_CONFIG.read_text())
    raw["run_matrix"][1] = dict(raw["run_matrix"][0])
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="duplicate rows|missing, duplicating"):
        load_experiment_spec(duplicate_path)

    raw = json.loads(FULL_CONFIG.read_text())
    raw["base_config"]["application_queue_policy"] = "none"
    wrong_policy = tmp_path / "wrong-policy.json"
    wrong_policy.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="application_queue_policy"):
        load_experiment_spec(wrong_policy)


@pytest.mark.extended_research
def test_full_sweep_output_contract_requires_real_tables_and_conservation():
    spec = load_experiment_spec(FULL_CONFIG)
    config = build_experiment_grid(spec)[0]["scientific_config"]
    table_keys = set(full_sweep_run_tables(config)) | set(PAPER1_APC_REQUIRED_OUTPUT_TABLES)
    outputs = {key: pd.DataFrame() for key in table_keys}
    outputs["node_tx_duty_cycle_df"] = pd.DataFrame([
        {
            "run_id": "unit",
            "entity_id": f"N{index}",
            "entity_type": "node",
            "sim_time_s": 3600.0,
            "observed_tx_airtime_s": 1.0,
            "tx_duty_fraction": 1.0 / 3600.0,
            "tx_duty_cycle_percent": 100.0 / 3600.0,
            "measurement_scope": "physical_tx_airtime_over_simulation_horizon",
        }
        for index in range(int(config["node_num"]))
    ])
    outputs["channel_occupancy_df"] = pd.DataFrame([{
        "run_id": "unit",
        "channel": 1,
        "sim_time_s": 3600.0,
        "sum_offered_tx_airtime_s": 3.0,
        "sum_observed_tx_airtime_s": 3.0,
        "channel_busy_union_s": 2.0,
        "channel_busy_fraction": 2.0 / 3600.0,
        "channel_busy_percent": 200.0 / 3600.0,
        "occupancy_method": "union_of_simulated_tx_intervals",
    }])
    summary = {key: 0 for key in PAPER1_APC_REQUIRED_SUMMARY_FIELDS}
    summary["stage1_status"] = "PASS"
    validate_full_sweep_outputs(config, summary, outputs)

    outputs.pop("channel_occupancy_df")
    with pytest.raises(ValueError, match="missing evidence"):
        validate_full_sweep_outputs(config, summary, outputs)


@pytest.mark.extended_research
def test_full_sweep_run_tables_are_profile_specific():
    full_config = build_experiment_grid(load_experiment_spec(FULL_CONFIG))[0][
        "scientific_config"
    ]
    diagnostic_config = build_experiment_grid(
        load_experiment_spec(DIAGNOSTIC_CONFIG)
    )[0]["scientific_config"]
    assert {
        "channel_occupancy_df": "channel_occupancy.csv",
        "node_tx_duty_cycle_df": "node_tx_duty_cycle.csv",
    }.items() <= experiment_run_tables(full_config).items()
    assert "channel_occupancy_df" not in experiment_run_tables(diagnostic_config)


@pytest.mark.extended_research
def test_full_sweep_validator_rejects_invalid_occupancy_values():
    spec = load_experiment_spec(FULL_CONFIG)
    config = build_experiment_grid(spec)[0]["scientific_config"]
    table_keys = set(full_sweep_run_tables(config)) | set(PAPER1_APC_REQUIRED_OUTPUT_TABLES)
    outputs = {key: pd.DataFrame() for key in table_keys}
    outputs["node_tx_duty_cycle_df"] = pd.DataFrame([
        {
            "run_id": "unit", "entity_id": f"N{index}", "entity_type": "node",
            "sim_time_s": 3600.0, "observed_tx_airtime_s": 1.0,
            "tx_duty_fraction": 1.0 / 3600.0,
            "tx_duty_cycle_percent": 100.0 / 3600.0,
            "measurement_scope": "physical_tx_airtime_over_simulation_horizon",
        }
        for index in range(int(config["node_num"]))
    ])
    outputs["channel_occupancy_df"] = pd.DataFrame([{
        "run_id": "unit", "channel": 1, "sim_time_s": 3600.0,
        "sum_offered_tx_airtime_s": 1.0,
        "sum_observed_tx_airtime_s": 1.0,
        "channel_busy_union_s": 1.1,
        "channel_busy_fraction": 1.1 / 3600.0,
        "channel_busy_percent": 110.0 / 3600.0,
        "occupancy_method": "union_of_simulated_tx_intervals",
    }])
    summary = {key: 0 for key in PAPER1_APC_REQUIRED_SUMMARY_FIELDS}
    summary["stage1_status"] = "PASS"
    with pytest.raises(ValueError, match="airtime is inconsistent"):
        validate_full_sweep_outputs(config, summary, outputs)


@pytest.mark.extended_research
def test_full_sweep_validator_requires_one_duty_row_per_node():
    spec = load_experiment_spec(FULL_CONFIG)
    config = build_experiment_grid(spec)[0]["scientific_config"]
    table_keys = set(full_sweep_run_tables(config)) | set(PAPER1_APC_REQUIRED_OUTPUT_TABLES)
    outputs = {key: pd.DataFrame() for key in table_keys}
    outputs["node_tx_duty_cycle_df"] = pd.DataFrame([
        {
            "run_id": "unit", "entity_id": f"N{index}", "entity_type": "node",
            "sim_time_s": 3600.0, "observed_tx_airtime_s": 1.0,
            "tx_duty_fraction": 1.0 / 3600.0,
            "tx_duty_cycle_percent": 100.0 / 3600.0,
            "measurement_scope": "physical_tx_airtime_over_simulation_horizon",
        }
        for index in range(int(config["node_num"]) - 1)
    ])
    outputs["channel_occupancy_df"] = pd.DataFrame([{
        "run_id": "unit", "channel": 1, "sim_time_s": 3600.0,
        "sum_offered_tx_airtime_s": 3.0, "sum_observed_tx_airtime_s": 3.0,
        "channel_busy_union_s": 2.0, "channel_busy_fraction": 2.0 / 3600.0,
        "channel_busy_percent": 200.0 / 3600.0,
        "occupancy_method": "union_of_simulated_tx_intervals",
    }])
    summary = {key: 0 for key in PAPER1_APC_REQUIRED_SUMMARY_FIELDS}
    summary["stage1_status"] = "PASS"
    with pytest.raises(ValueError, match="one row per configured end node"):
        validate_full_sweep_outputs(config, summary, outputs)
