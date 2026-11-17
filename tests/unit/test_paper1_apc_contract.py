from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pandas as pd
import pytest

from loraphymac.config.validation import ConfigurationError, validate_configuration
from loraphymac.experiments.apc_terrain_sweep.contract import (
    APC_TERRAIN_SWEEP_CONTRACT_VERSION,
    APC_TERRAIN_SWEEP_PROFILE,
    PAPER1_APC_COMPARISON_TERRAIN_DELTA_H,
    PAPER1_APC_CONTRACT_VERSION,
    PAPER1_APC_E1_ONLY_CONFIG_KEYS,
    PAPER1_APC_LEGACY_DEFAULT_CONFIG,
    PAPER1_APC_LOCKED_CONFIG,
    PAPER1_APC_OPERATIONAL_CONFIG_KEYS,
    PAPER1_APC_PROFILE,
    PAPER1_APC_VARIABLE_CONFIG_KEYS,
    paper1_apc_run_tables,
    validate_paper1_apc_outputs,
)
from loraphymac.experiments.common.multiseed_pipeline import (
    ExperimentSpec,
    build_experiment_grid,
    load_experiment_spec,
)
from loraphymac.experiments.common import multiseed_pipeline as pipeline
from loraphymac.engine import DEFAULT_CONFIG, build_config
from loraphymac import SimulationResult


def _paper1_config(**overrides):
    config = build_config(
        {
            **dict(PAPER1_APC_LOCKED_CONFIG),
            "experiment_profile": PAPER1_APC_PROFILE,
            **overrides,
        }
    )
    validate_configuration(config, known_keys=DEFAULT_CONFIG)
    return config


def test_contract_classifies_every_simulator_configuration_key():
    classified = (
        set(PAPER1_APC_LOCKED_CONFIG)
        | set(PAPER1_APC_LEGACY_DEFAULT_CONFIG)
        | set(PAPER1_APC_E1_ONLY_CONFIG_KEYS)
        | set(PAPER1_APC_VARIABLE_CONFIG_KEYS)
        | set(PAPER1_APC_OPERATIONAL_CONFIG_KEYS)
        | {"experiment_profile"}
    )
    assert classified == set(DEFAULT_CONFIG)


def test_canonical_paper1_configuration_and_all_four_apc_modes_are_valid():
    for node_mode, relay_mode in (
        ("fixed", "fixed"),
        ("adaptive", "fixed"),
        ("fixed", "adaptive"),
        ("adaptive", "adaptive"),
    ):
        _paper1_config(node_ptx_mode=node_mode, relay_ptx_mode=relay_mode)


@pytest.mark.parametrize(
    ("override", "key"),
    [
        ({"node_num": 60}, "node_num"),
        ({"area_x": 50_000}, "area_x"),
        ({"sim_time": 1_800}, "sim_time"),
        ({"node_sf_list": [8]}, "node_sf_list"),
        ({"apc_fopts_size": 3}, "apc_fopts_size"),
        ({"node_target_margin_db": 5.0}, "node_target_margin_db"),
        ({"relay_duty_cycle": 0.01}, "relay_duty_cycle"),
        ({"include_mcu_energy": True}, "include_mcu_energy"),
        ({"energy_accounting_mode": "corrected"}, "energy_accounting_mode"),
        ({"node_rx_window_duration_s": 2.0}, "node_rx_window_duration_s"),
        ({"terrain_delta_h": 19}, "terrain_delta_h"),
        ({"seed": 43}, "seed"),
    ],
)
def test_paper1_profile_rejects_contract_drift(override, key):
    with pytest.raises(ConfigurationError, match=key):
        _paper1_config(**override)


def test_versioned_apc_terrain_sweep_spec_builds_complete_reproducible_grid():
    project_root = Path(__file__).resolve().parents[2]
    spec = load_experiment_spec(
        project_root
        / "src/loraphymac/experiments/apc_terrain_sweep/configs/production_v1.json"
    )
    grid = build_experiment_grid(spec)

    assert len(grid) == 300
    assert len({run["run_id"] for run in grid}) == 300
    assert {
        (run["factors"]["node_ptx_mode"], run["factors"]["relay_ptx_mode"]) for run in grid
    } == {
        ("fixed", "fixed"),
        ("adaptive", "fixed"),
        ("fixed", "adaptive"),
        ("adaptive", "adaptive"),
    }
    assert {run["scientific_config"]["experiment_profile"] for run in grid} == {
        APC_TERRAIN_SWEEP_PROFILE
    }


def test_paper1_grid_contract_rejects_incomplete_factor_design(tmp_path):
    spec = ExperimentSpec(
        experiment_id="p1-incomplete",
        output_dir=tmp_path,
        base_config={"experiment_profile": PAPER1_APC_PROFILE},
        factors={"terrain_delta_h": [90]},
        seeds=[30, 36, 42, 48, 56],
    )

    with pytest.raises(ValueError, match="Paper 1 APC"):
        build_experiment_grid(spec)


def test_paper1_table_policy_keeps_histories_only_at_comparison_terrain():
    comparison = _paper1_config(terrain_delta_h=PAPER1_APC_COMPARISON_TERRAIN_DELTA_H)
    sweep = _paper1_config(terrain_delta_h=20)

    assert "ptx_history_df" in paper1_apc_run_tables(comparison)
    assert "margin_history_df" in paper1_apc_run_tables(comparison)
    assert "ptx_history_df" not in paper1_apc_run_tables(sweep)
    assert "margin_history_df" not in paper1_apc_run_tables(sweep)
    assert "energy_breakdown_df" in paper1_apc_run_tables(sweep)
    assert pipeline.master_table_names_for_profile(PAPER1_APC_PROFILE)["ptx_history.csv"] == (
        "per_run_ptx_history.csv"
    )


def _valid_fixed_outputs(contract_version=PAPER1_APC_CONTRACT_VERSION):
    summary = {
        "delivery_by_horizon_percent": 90.0,
        "terminal_loss_percent": 10.0,
        "censored_fraction_percent": 0.0,
        "avg_e2e_delay_ms": 100.0,
        "total_node_energy_J": 10.0,
        "total_relay_energy_J": 5.0,
        "total_gateway_energy_J": 20.0,
        "total_network_energy_J": 35.0,
        "total_constrained_energy_J": 15.0,
        "energy_per_delivered_packet_J": 1.0,
        "energy_per_delivered_packet_constrained_J": 0.5,
        "avg_final_node_ptx": 17.0,
        "avg_final_relay_ptx": 20.0,
        "apc_total_tx": 0,
        "apc_total_rx": 0,
        "apc_total_drop": 0,
        "ptx_update_count": 0,
        "stage1_status": "PASS",
        "packet_fate_conservation_delta": 0,
        "collision_outcome_conservation_delta": 0,
    }
    required_tables = {
        "energy_breakdown_df",
        "final_ptx_distribution_df",
        "frame_summary_df",
        "run_metadata_df",
        "ptx_history_df",
        "margin_history_df",
        "apc_control_history_df",
        "packet_fate_summary_df",
    }
    outputs = {name: pd.DataFrame() for name in required_tables}
    outputs["ptx_history_df"] = pd.DataFrame(
        columns=["entity_type", "old_ptx", "new_ptx", "apply_source"]
    )
    outputs["run_metadata_df"] = pd.DataFrame(
        [{"experiment_contract_version": contract_version}]
    )
    outputs["frame_summary_df"] = pd.DataFrame([{"stage1_status": "PASS"}])
    return summary, outputs


def test_paper1_output_contract_accepts_consistent_fixed_baseline():
    summary, outputs = _valid_fixed_outputs()
    validate_paper1_apc_outputs(
        _paper1_config(node_ptx_mode="fixed", relay_ptx_mode="fixed"),
        summary,
        outputs,
    )


def test_paper1_output_contract_rejects_apc_accounting_mismatch():
    summary, outputs = _valid_fixed_outputs()
    summary.update({"apc_total_tx": 1, "apc_total_rx": 1})

    with pytest.raises(ValueError, match="fixed/fixed|ptx_update_count"):
        validate_paper1_apc_outputs(
            _paper1_config(node_ptx_mode="fixed", relay_ptx_mode="fixed"),
            summary,
            outputs,
        )


def test_pipeline_retains_paper1_tables_according_to_contract(tmp_path, monkeypatch):
    project_root = Path(__file__).resolve().parents[2]
    spec = replace(
        load_experiment_spec(
            project_root
            / "src/loraphymac/experiments/apc_terrain_sweep/configs/production_v1.json"
        ),
        output_dir=tmp_path,
    )
    grid = build_experiment_grid(spec)
    fixed_runs = [
        run
        for run in grid
        if run["factors"]["node_ptx_mode"] == "fixed"
        and run["factors"]["relay_ptx_mode"] == "fixed"
        and run["seed"] == 30
    ]

    def fake_simulator(config):
        summary, outputs = _valid_fixed_outputs(APC_TERRAIN_SWEEP_CONTRACT_VERSION)
        summary["run_id"] = config["run_id"]
        return SimulationResult(
            resolved_config=config,
            summary=summary,
            tables=outputs,
        )

    monkeypatch.setattr(pipeline, "simulate", fake_simulator)
    experiment_dir = tmp_path / "paper1_apc_v1"

    sweep_run = next(run for run in fixed_runs if run["factors"]["terrain_delta_h"] == 20)
    assert pipeline.execute_one_run(spec, experiment_dir, sweep_run) == "PASS"
    sweep_tables = experiment_dir / "runs" / sweep_run["run_id"] / "tables"
    assert (sweep_tables / "energy_breakdown.csv").exists()
    assert not (sweep_tables / "ptx_history.csv").exists()

    comparison_run = next(
        run
        for run in fixed_runs
        if run["factors"]["terrain_delta_h"] == PAPER1_APC_COMPARISON_TERRAIN_DELTA_H
    )
    assert pipeline.execute_one_run(spec, experiment_dir, comparison_run) == "PASS"
    comparison_tables = experiment_dir / "runs" / comparison_run["run_id"] / "tables"
    assert (comparison_tables / "energy_breakdown.csv").exists()
    assert (comparison_tables / "ptx_history.csv").exists()
    assert (comparison_tables / "margin_history.csv").exists()
