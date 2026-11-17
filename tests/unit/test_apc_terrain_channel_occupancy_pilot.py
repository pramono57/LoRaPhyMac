import json
from pathlib import Path

import pytest

from loraphymac.experiments.apc_terrain_profile_diagnostic.channel_occupancy_pilot_contract import (
    APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_PROFILE,
    APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_RUN_COUNT,
    APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_VERSION,
)
from loraphymac.experiments.apc_terrain_profile_diagnostic.full_sweep_contract import (
    APC_TERRAIN_PROFILE_FULL_SWEEP_PROFILE,
    APC_TERRAIN_PROFILE_FULL_SWEEP_RUN_COUNT,
)
from loraphymac.experiments.common.contract_registry import (
    experiment_contract_version,
    experiment_run_tables,
)
from loraphymac.experiments.common.multiseed_pipeline import (
    build_experiment_grid,
    load_experiment_spec,
    run_experiment,
)
from loraphymac.simulation.output_collection import collects_node_channel_duty


ROOT = Path(__file__).resolve().parents[2]
PILOT_CONFIG = (
    ROOT
    / "artifacts/diagnostic_runs_20261112/configs/"
    "apc_terrain_channel_occupancy_pilot_r01.json"
)
FULL_SWEEP_CONFIG = (
    ROOT
    / "artifacts/diagnostic_runs_20261111/configs/"
    "apc_terrain_profile_full_sweep_v1.json"
)


@pytest.mark.extended_research
def test_channel_occupancy_pilot_resolves_four_runs_and_dry_runs_only():
    spec = load_experiment_spec(PILOT_CONFIG)
    runs = build_experiment_grid(spec)
    dry_run = run_experiment(spec, ROOT, dry_run=True, show_progress=False)

    assert len(runs) == APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_RUN_COUNT == 4
    assert len({run["run_id"] for run in runs}) == 4
    assert dry_run["run_count"] == 4
    assert Path(dry_run["experiment_dir"]) == (
        ROOT
        / "artifacts/latest_runner_rerun_20261110/APC_TERRAIN_SWEEP/"
        "run_261112_channel_occupancy_pilot_r01"
    )
    assert not Path(dry_run["experiment_dir"]).exists()

    observed = set()
    for run in runs:
        config = run["scientific_config"]
        observed.add(
            (
                config["node_ptx_mode"],
                config["relay_ptx_mode"],
                config["terrain_delta_h"],
                run["seed"],
                config["output_profile"],
            )
        )
        assert config["experiment_profile"] == (
            APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_PROFILE
        )
        assert config["node_rx_collision_enabled"] is True
        assert config["application_scheduling_mode"] == "independent_fifo_v1"
        assert config["application_rng_policy"] == "sha256_seed_node_v1"
        assert config["application_queue_policy"] == "unbounded_fifo_v1"
        assert config["application_horizon_policy"] == "right_censor_waiting_v1"
        tables = experiment_run_tables(config)
        assert tables["channel_occupancy_df"] == "channel_occupancy.csv"
        assert tables["node_tx_duty_cycle_df"] == "node_tx_duty_cycle.csv"
        assert collects_node_channel_duty(config["experiment_profile"])

    assert observed == {
        (mode, mode, 90, 42, profile)
        for mode in ("fixed", "adaptive")
        for profile in ("summary", "detailed")
    }


@pytest.mark.extended_research
def test_pilot_contract_is_separate_from_unchanged_300_run_contract():
    assert experiment_contract_version(APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_PROFILE) == (
        APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_VERSION
    )
    assert experiment_contract_version(APC_TERRAIN_PROFILE_FULL_SWEEP_PROFILE) == (
        "apc_terrain_profile_full_sweep_contract_v1.0"
    )
    assert APC_TERRAIN_CHANNEL_OCCUPANCY_PILOT_RUN_COUNT == 4
    assert APC_TERRAIN_PROFILE_FULL_SWEEP_RUN_COUNT == 300
    assert len(build_experiment_grid(load_experiment_spec(FULL_SWEEP_CONFIG))) == 300


@pytest.mark.extended_research
def test_pilot_contract_rejects_missing_matrix_row(tmp_path):
    raw = json.loads(PILOT_CONFIG.read_text(encoding="utf-8"))
    raw["run_matrix"].pop()
    config_path = tmp_path / "missing-pilot-row.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly four rows|four pilot runs"):
        load_experiment_spec(config_path)
