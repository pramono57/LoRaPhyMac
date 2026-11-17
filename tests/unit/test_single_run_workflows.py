from __future__ import annotations

import json

import pandas as pd
import pytest

from loraphymac import SimulationResult, resolve_configuration
from loraphymac.config.validation import ConfigurationError
from loraphymac.experiments.apc_closed_loop_verification.contract import (
    APC_CLOSED_LOOP_CONTRACT_VERSION,
    APC_CLOSED_LOOP_PROFILE,
    APCClosedLoopContractError,
    validate_apc_closed_loop_outputs,
)
from loraphymac.experiments.apc_closed_loop_verification import workflow
from loraphymac.experiments.apc_closed_loop_verification.workflow import (
    default_apc_closed_loop_config,
    run_apc_closed_loop_verification,
)
from loraphymac.experiments.common.contract_registry import (
    experiment_contract_version,
)
from loraphymac.experiments.single_run_exploration.workflow import (
    default_exploration_config,
)


def _valid_result(config) -> SimulationResult:
    ptx_history = pd.DataFrame(
        [
            {
                "entity_id": "N1",
                "entity_type": "node",
                "time": 61.0,
                "old_ptx": 17,
                "new_ptx": 15,
            },
            {
                "entity_id": "R1",
                "entity_type": "relay",
                "time": 2.0,
                "old_ptx": 20,
                "new_ptx": 18,
            },
        ]
    )
    margin_history = pd.DataFrame(
        [
            {
                "entity_id": "N1",
                "entity_type": "node",
                "time": 60.0,
                "margin": 10.0,
            },
            {
                "entity_id": "R1",
                "entity_type": "relay",
                "time": 1.0,
                "margin": 9.0,
            },
        ]
    )
    summary = {
        "delivery_by_horizon_percent": 90.0,
        "terminal_loss_percent": 8.0,
        "censored_fraction_percent": 2.0,
        "avg_e2e_delay_ms": 250.0,
        "total_network_energy_J": 1000.0,
        "avg_final_node_ptx": 15.0,
        "avg_final_relay_ptx": 18.0,
        "apc_total_tx": 3,
        "apc_total_rx": 2,
        "apc_total_drop": 1,
        "ptx_update_count": 2,
        "stage1_status": "PASS",
        "packet_fate_conservation_delta": 0,
        "collision_outcome_conservation_delta": 0,
    }
    tables = {
        "energy_breakdown_df": pd.DataFrame(),
        "final_ptx_distribution_df": pd.DataFrame(),
        "ptx_history_df": ptx_history,
        "margin_history_df": margin_history,
        "apc_control_history_df": pd.DataFrame([{"event": "APC_APPLY"}]),
        "frame_summary_df": pd.DataFrame([{"stage1_status": "PASS"}]),
        "packet_fate_summary_df": pd.DataFrame([{"delivered_count": 90}]),
        "run_metadata_df": pd.DataFrame(
            [
                {
                    "experiment_contract_version": (
                        APC_CLOSED_LOOP_CONTRACT_VERSION
                    )
                }
            ]
        ),
        "event_log_df": pd.DataFrame(),
    }
    return SimulationResult(
        resolved_config=config,
        summary=summary,
        tables=tables,
    )


def test_exploration_profile_keeps_scientific_parameters_open():
    config = default_exploration_config()
    config.update(
        {
            "node_num": 25,
            "sim_time": 300,
            "relay_tx_ch_ids": [1, 2, 3, 4],
            "export_summary": False,
            "export_results": False,
            "export_debug": False,
            "plot_routing": False,
        }
    )

    resolved = resolve_configuration(config)

    assert resolved["experiment_profile"] == "general_multisf_multibw"
    assert resolved["node_num"] == 25
    assert resolved["relay_tx_ch_ids"] == [1, 2, 3, 4]
    assert experiment_contract_version(resolved["experiment_profile"]) is None


def test_closed_loop_profile_rejects_scientific_drift_before_execution():
    config = default_apc_closed_loop_config()
    config["node_num"] = 99

    with pytest.raises(ConfigurationError, match="node_num must remain 100"):
        resolve_configuration(config)


def test_closed_loop_contract_rejects_missing_relay_feedback():
    config = resolve_configuration(default_apc_closed_loop_config())
    result = _valid_result(config)
    result.tables["ptx_history_df"].drop(
        result.tables["ptx_history_df"].index[-1],
        inplace=True,
    )

    with pytest.raises(
        APCClosedLoopContractError,
        match="PTx-history row count|relay closed-loop",
    ):
        validate_apc_closed_loop_outputs(
            result.resolved_config,
            result.summary,
            result.tables,
        )


def test_closed_loop_workflow_writes_pass_report_last(tmp_path, monkeypatch):
    config = default_apc_closed_loop_config(output_dir=tmp_path)

    def fake_simulate(resolved):
        return _valid_result(resolve_configuration(resolved))

    monkeypatch.setattr(workflow, "simulate", fake_simulate)

    result = run_apc_closed_loop_verification(config)
    report_path = tmp_path / "verification_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.resolved_config["experiment_profile"] == APC_CLOSED_LOOP_PROFILE
    assert report["status"] == "PASS"
    assert report["experiment_contract_version"] == APC_CLOSED_LOOP_CONTRACT_VERSION
    assert report["checks"]["node_ptx_update_count"] == 1
    assert report["checks"]["relay_ptx_update_count"] == 1
