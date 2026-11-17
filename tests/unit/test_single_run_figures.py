from __future__ import annotations

from pathlib import Path

import pandas as pd

from loraphymac import SimulationResult
from loraphymac.experiments.apc_closed_loop_verification.contract import (
    APC_CLOSED_LOOP_PROFILE,
    apc_closed_loop_config_violations,
)
from loraphymac.experiments.apc_closed_loop_verification.workflow import (
    APC_CLOSED_LOOP_ARTIFACT_ROOT,
    apc_closed_loop_output_dir,
    default_apc_closed_loop_config,
)
from loraphymac.experiments.common.single_run_diagnostics import (
    FIGURE_FONT_SIZE,
    FIGURE_STYLE,
    build_apc_decision_validation_table,
    build_data_up_down_debug_table,
    generate_single_run_figures,
    generate_single_run_tables,
    map_power_updates_to_next_transmission,
    select_representative_nodes,
)
from loraphymac.experiments.single_run_exploration.workflow import (
    EXPLORATION_ARTIFACT_ROOT,
    default_exploration_config,
    exploration_output_dir,
)


def test_exploration_configuration_is_flexible_and_multichannel(tmp_path):
    config = default_exploration_config(output_dir=tmp_path)

    assert config["run_id"] == "exploration_001"
    assert config["experiment_profile"] == "general_multisf_multibw"
    assert config["sim_time"] == 3600
    assert config["node_num"] == 100
    assert config["relay_tx_ch_ids"] == [1, 2, 3, 4]
    assert config["node_ptx_mode"] == "adaptive"
    assert config["relay_ptx_mode"] == "adaptive"
    assert config["output_dir"] == str(tmp_path)
    assert config["print_debug"] is False


def test_apc_closed_loop_configuration_is_one_frozen_paper1_point(tmp_path):
    config = default_apc_closed_loop_config(output_dir=tmp_path)

    assert config["run_id"] == "apc_closed_loop_dh90_s42"
    assert config["experiment_profile"] == APC_CLOSED_LOOP_PROFILE
    assert config["sim_time"] == 3600
    assert config["node_num"] == 100
    assert config["seed"] == 42
    assert config["terrain_delta_h"] == 90
    assert config["relay_tx_ch_ids"] == [1]
    assert config["node_ptx_mode"] == "adaptive"
    assert config["relay_ptx_mode"] == "adaptive"
    assert config["output_dir"] == str(tmp_path)
    assert apc_closed_loop_config_violations(config) == []


def test_two_single_run_workflows_have_separate_artifact_namespaces():
    exploration = default_exploration_config()
    verification = default_apc_closed_loop_config()

    assert exploration_output_dir(exploration["run_id"]) == (
        EXPLORATION_ARTIFACT_ROOT / "exploration_001"
    )
    assert apc_closed_loop_output_dir(verification["run_id"]) == (
        APC_CLOSED_LOOP_ARTIFACT_ROOT / "apc_closed_loop_dh90_s42"
    )
    assert Path(exploration["output_dir"]).is_relative_to(EXPLORATION_ARTIFACT_ROOT)
    assert Path(verification["output_dir"]).is_relative_to(APC_CLOSED_LOOP_ARTIFACT_ROOT)


def _result(tmp_path: Path) -> SimulationResult:
    ptx = pd.DataFrame(
        [
            {
                "entity_id": "N1",
                "entity_type": "node",
                "time": 10,
                "old_ptx": 17,
                "new_ptx": 14,
                "old_tx_power_index": 0,
                "new_tx_power_index": 1,
                "decision_margin": 12.0,
                "decision_rssi": -114.0,
                "apc_rx_margin": 7.0,
                "feedback_from": "R1",
                "apply_source": "physical_data_down",
            },
            {
                "entity_id": "N1",
                "entity_type": "node",
                "time": 11,
                "old_ptx": 14,
                "new_ptx": 12,
                "old_tx_power_index": 1,
                "new_tx_power_index": 2,
                "decision_margin": 7.0,
                "decision_rssi": -119.0,
                "apc_rx_margin": 6.5,
                "feedback_from": "R1",
                "apply_source": "physical_data_down",
            },
            {
                "entity_id": "N2",
                "entity_type": "node",
                "time": 12,
                "old_ptx": 17,
                "new_ptx": 16,
                "old_tx_power_index": 0,
                "new_tx_power_index": 1,
                "decision_margin": 6.2,
                "decision_rssi": -119.8,
                "apc_rx_margin": 6.0,
                "feedback_from": "R2",
                "apply_source": "physical_data_down",
            },
            {
                "entity_id": "R1",
                "entity_type": "relay",
                "time": 2,
                "old_ptx": 20,
                "new_ptx": 12,
                "old_tx_power_index": 0,
                "new_tx_power_index": 4,
                "decision_margin": 16.0,
                "decision_rssi": -110.0,
                "apc_rx_margin": 16.0,
                "feedback_from": "GW",
                "apply_source": "physical_data_down_fwd",
            },
        ]
    )
    margin = pd.DataFrame(
        [
            {"entity_id": "N1", "entity_type": "node", "time": 10, "margin": 12.0},
            {"entity_id": "N1", "entity_type": "node", "time": 11, "margin": 7.0},
            {"entity_id": "N2", "entity_type": "node", "time": 12, "margin": 6.2},
            {"entity_id": "R1", "entity_type": "relay", "time": 2, "margin": 8.0},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event": "TX",
                "node": "N1",
                "src": "N1",
                "next_hop": "R1",
                "mtype": 1,
                "time": 20,
                "ch": 0,
                "reason": None,
            },
            {
                "event": "TX",
                "node": "N2",
                "src": "N2",
                "next_hop": "R2",
                "mtype": 1,
                "time": 22,
                "ch": 1,
                "reason": None,
            },
            {
                "event": "TX",
                "node": "R1",
                "src": "N1",
                "next_hop": "GW",
                "mtype": 2,
                "time": 5,
                "ch": 1,
                "reason": None,
            },
            {
                "event": "TX",
                "node": "R1",
                "src": "R1",
                "next_hop": "N1",
                "mtype": 3,
                "time": 15,
                "ch": 0,
                "reason": None,
            },
            {
                "event": "RX",
                "node": "N1",
                "src": "R1",
                "next_hop": "N1",
                "mtype": 3,
                "time": 15.1,
                "ch": 0,
                "reason": None,
            },
            {
                "event": "TX",
                "node": "R2",
                "src": "R2",
                "next_hop": "N2",
                "mtype": 3,
                "time": 16,
                "ch": 1,
                "reason": None,
            },
            {
                "event": "DROP",
                "node": "N2",
                "src": "R2",
                "next_hop": "N2",
                "mtype": 3,
                "time": 16.1,
                "ch": 1,
                "reason": "collision",
            },
        ]
    )
    return SimulationResult(
        resolved_config={
            "run_id": "single-test",
            "output_dir": str(tmp_path),
            "node_num": 2,
            "node_target_margin_db": 6.0,
            "relay_target_margin_db": 6.0,
        },
        summary={},
        tables={
            "ptx_history_df": ptx,
            "margin_history_df": margin,
            "event_log_df": events,
        },
    )


def test_representative_nodes_are_deterministic_and_unique(tmp_path):
    result = _result(tmp_path)

    selected = select_representative_nodes(
        result.ptx_history,
        result.margin_history,
        target_margin_db=6.0,
    )

    assert selected == ["N1", "N2"]


def test_representative_nodes_do_not_fill_a_fourth_node_without_legacy_criterion():
    ptx = pd.DataFrame(
        [
            {"entity_id": "N24", "entity_type": "node", "time": 1, "old_ptx": 17, "new_ptx": 5},
            {"entity_id": "N1", "entity_type": "node", "time": 1, "old_ptx": 17, "new_ptx": 11},
            {"entity_id": "N95", "entity_type": "node", "time": 1, "old_ptx": 17, "new_ptx": 11},
            {"entity_id": "N10", "entity_type": "node", "time": 1, "old_ptx": 17, "new_ptx": 7},
        ]
    )
    margin = pd.DataFrame(
        [
            {"entity_id": "N24", "entity_type": "node", "time": 1, "margin": 15.0},
            {"entity_id": "N1", "entity_type": "node", "time": 1, "margin": 6.6},
            {"entity_id": "N95", "entity_type": "node", "time": 1, "margin": 6.01},
            {"entity_id": "N10", "entity_type": "node", "time": 1, "margin": 6.4},
        ]
    )

    selected = select_representative_nodes(ptx, margin, target_margin_db=6.0)

    assert selected == ["N24", "N1", "N95"]
    assert "N10" not in selected


def test_single_run_figures_restore_legacy_times_new_roman_18_style():
    assert FIGURE_FONT_SIZE == 18
    assert FIGURE_STYLE["font.family"] == "Times New Roman"
    for key in (
        "font.size",
        "axes.labelsize",
        "axes.titlesize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
        "legend.title_fontsize",
        "figure.titlesize",
    ):
        assert FIGURE_STYLE[key] == 18


def test_multiple_updates_before_one_tx_use_latest_power(tmp_path):
    result = _result(tmp_path)

    used = map_power_updates_to_next_transmission(
        result.ptx_history,
        result.table("event_log_df"),
        entity_ids=["N1"],
        entity_type="node",
        max_time_s=30,
    )

    assert used[["time", "ptx_dbm"]].to_dict("records") == [
        {"time": 0.0, "ptx_dbm": 17.0},
        {"time": 20.0, "ptx_dbm": 12.0},
    ]


def test_single_run_builder_restores_four_diagnostic_figure_groups(tmp_path):
    result = _result(tmp_path)

    generated = generate_single_run_figures(result, formats=("png", "pdf"))

    assert set(generated) == {
        "node_margin_convergence",
        "node_ptx_next_tx",
        "relay_margin_convergence",
        "relay_ptx_next_tx",
    }
    assert all({path.suffix for path in paths} == {".png", ".pdf"} for paths in generated.values())
    assert all(path.is_file() for paths in generated.values() for path in paths)
    assert all(
        path.parent == tmp_path / "figures" for paths in generated.values() for path in paths
    )


def test_single_run_builds_legacy_table_iv_and_data_up_down_debug_table(tmp_path):
    result = _result(tmp_path)

    decision = build_apc_decision_validation_table(result.ptx_history)
    data_up_down = build_data_up_down_debug_table(
        result.table("event_log_df"),
        node_ids=("N1", "N2"),
    )

    assert set(decision["Target"]) == {"N1", "N2", "R1"}
    assert set(decision["APC decision"]) == {"Decrease PTx"}
    assert set(decision["Feedback frame"]) == {"DATA_DOWN", "DATA_DOWN_FWD"}

    debug = data_up_down.set_index("node")
    assert debug.loc["N1", "data_up_ch"] == [0]
    assert debug.loc["N1", "data_down_rx_count"] == 1
    assert bool(debug.loc["N1", "channel_match"])
    assert debug.loc["N2", "data_down_drop_count"] == 1
    assert debug.loc["N2", "drop_reason_summary"] == "collision=1"

    outputs = generate_single_run_tables(result)
    assert set(outputs) == {"apc_decision_validation", "data_up_down_debug"}
    assert all(path.parent == tmp_path / "tables" for path in outputs.values())
    assert all(path.is_file() for path in outputs.values())
    assert len(pd.read_csv(outputs["data_up_down_debug"])) == 2
