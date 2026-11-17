from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import loraphymac.simulation.output_collection as output_collection
from loraphymac.simulation.context import RoutingArtifacts
from loraphymac.experiments.apc_terrain_profile_diagnostic.full_sweep_contract import (
    APC_TERRAIN_PROFILE_FULL_SWEEP_PROFILE,
)
from loraphymac.simulation.exporting import export_simulation_outputs
from loraphymac.simulation.output_collection import (
    collects_node_channel_duty,
    collect_simulation_outputs,
)
from loraphymac.simulation.reporting import _display_table


def _frame(**values):
    return pd.DataFrame([values])


def _routing() -> RoutingArtifacts:
    selected = _frame(Node="N1", Target="GW", Valid=True)
    return RoutingArtifacts(
        relay_hops={"R1": 1, "GW": 0},
        link_df=selected.copy(),
        routing_table={"N1": "GW"},
        selected_next_hop_links_df=selected,
        routing_map={"N1": "GW", "R1": "GW", "GW": None},
        all_nodes=["N1", "R1", "GW"],
        node_map={"N1": "N1", "R1": "R1", "GW": "GW"},
        relay_chain_links_df=_frame(Node="R1", Target="GW", Valid=True),
        visual_links_df=selected.copy(),
    )


def test_v2_profiles_enable_node_duty_and_channel_occupancy_collection():
    assert collects_node_channel_duty("apc_compare_model_terrain_sweep_v2")
    assert collects_node_channel_duty(
        "apc_compare_model_terrain_sweep_v2_five_model_pilot"
    )
    assert not collects_node_channel_duty("apc_compare_model_terrain_sweep_v1")
    assert collects_node_channel_duty(APC_TERRAIN_PROFILE_FULL_SWEEP_PROFILE)
    assert not collects_node_channel_duty("apc_terrain_profile_diagnostic_v1")
    assert not collects_node_channel_duty("apc_terrain_sweep_v1")


class FakeStatistics:
    def __init__(self):
        self.calls = []
        self.ptx = _frame(entity_id="N1", new_ptx=10)
        self.margin = _frame(entity_id="N1", margin_db=7.0)
        self.apc = _frame(entity_id="N1", status="applied")
        self.failures = _frame(reason="collision")

    def get_summary_bundle(self, nodes):
        self.calls.append(("summary", nodes))
        return {
            "01_core_summary": _frame(tx_packets=2, rx_packets_gw=1),
            "02_run_metadata": _frame(run_id="unit"),
            "03_frame_summary": _frame(apc_total_tx=1, apc_total_rx=1),
            "04_hop_summary": pd.DataFrame(),
            "05_frame_mtype_summary": pd.DataFrame(),
            "06_frame_hop_event_summary": pd.DataFrame(),
            "07_frame_drop_reason_by_hop": pd.DataFrame(),
            "08_relay_duty_cycle_summary": pd.DataFrame(),
            "09_tx_queue_waiting_summary": pd.DataFrame(),
            "10_demodulator_pool_summary": pd.DataFrame(),
            "11_collision_summary": pd.DataFrame(),
            "12_packet_fate_summary": pd.DataFrame(),
        }

    def get_result_bundle(self, nodes, **routing_tables):
        self.calls.append(("result", nodes, routing_tables))
        return {
            "R01_energy_breakdown": pd.DataFrame(),
            "R02_node_rx_matrix": _frame(node_id="N1", gateway=1),
            "R03_per_channel_duty_cycle": pd.DataFrame(),
            "R04_final_ptx_distribution": pd.DataFrame(),
            "R05_relay_tx_mgc_queue": pd.DataFrame(),
            "R06_relay_tx_queue_metrics": pd.DataFrame(),
            "R07_tx_queue_traffic_class": pd.DataFrame(),
            "R11_ptx_history": self.ptx,
            "R12_margin_history": self.margin,
            "R13_apc_control_history": self.apc,
            "R14_delivered_packets": pd.DataFrame(),
            "R15_packet_failures": self.failures,
            "R16_relay_rx_statistics": pd.DataFrame(),
            "R17_rx_channel_usage": pd.DataFrame(),
        }

    def get_debug_bundle(self):
        self.calls.append(("debug", None))
        return {
            "D01_event_log": pd.DataFrame(),
            "D02_tx_queue_waiting_time": pd.DataFrame(),
            "D03_tx_queue_lifecycle": pd.DataFrame(),
            "D04_demodulator_events": pd.DataFrame(),
            "D05_collision_events": pd.DataFrame(),
            "D06_collision_interference_edges": pd.DataFrame(),
            "D07_packet_fates": pd.DataFrame(),
        }

    def get_packet_fate_df(self):
        self.calls.append(("packet_fates", None))
        return pd.DataFrame(columns=["packet_uid", "fcnt", "birth_time", "offered_time"])

    def get_summary_node_tx_duty_cycle_df(self, nodes):
        return pd.DataFrame({"node_id": nodes, "tx_duty_cycle": [0.1] * len(nodes)})

    def get_summary_channel_occupancy_df(self):
        return _frame(channel=0, channel_busy_fraction=0.2)


@pytest.mark.parametrize(
    "profile",
    ["apc_compare_model_terrain_sweep_v2", APC_TERRAIN_PROFILE_FULL_SWEEP_PROFILE],
)
def test_registered_profiles_emit_duty_and_occupancy_in_both_profiles(
    profile, monkeypatch
):
    routing = _routing()
    summary_stats = FakeStatistics()
    summary_stats.output_profile = "summary"
    summary_stats.run_metadata = {
        "experiment_profile": profile
    }
    summary_outputs = collect_simulation_outputs(
        statistics=summary_stats, routing=routing
    )
    assert not summary_outputs.tables["node_tx_duty_cycle_df"].empty
    assert not summary_outputs.tables["channel_occupancy_df"].empty

    detailed_stats = FakeStatistics()
    detailed_stats.output_profile = "detailed"
    detailed_stats.run_metadata = {
        "experiment_profile": profile
    }
    monkeypatch.setattr(
        output_collection,
        "build_node_tx_duty_cycle_df",
        lambda *_: _frame(node_id="N1", tx_duty_cycle=0.1),
    )
    monkeypatch.setattr(
        output_collection,
        "build_channel_occupancy_df",
        lambda *_: _frame(channel=0, channel_busy_fraction=0.2),
    )
    detailed_outputs = collect_simulation_outputs(
        statistics=detailed_stats, routing=routing
    )
    assert not detailed_outputs.tables["node_tx_duty_cycle_df"].empty
    assert not detailed_outputs.tables["channel_occupancy_df"].empty


def test_collection_preserves_bundle_order_and_legacy_contract():
    statistics = FakeStatistics()
    routing = _routing()

    outputs = collect_simulation_outputs(statistics=statistics, routing=routing)

    assert [call[0] for call in statistics.calls] == [
        "summary",
        "result",
        "debug",
    ]
    assert outputs.ptx_history is statistics.ptx
    assert outputs.margin_history is statistics.margin
    assert outputs.tables["packet_failure_df"] is statistics.failures


def test_summary_profile_skips_debug_bundle_construction():
    statistics = FakeStatistics()
    statistics.output_profile = "summary"

    outputs = collect_simulation_outputs(statistics=statistics, routing=_routing())

    assert [call[0] for call in statistics.calls] == [
        "summary", "result", "packet_fates"
    ]
    assert outputs.debug_bundle == {}
    assert outputs.tables["event_log_df"].empty
    assert outputs.tables["routing_table_df"].to_dict("records") == [
        {"source": "N1", "next_hop": "GW"}
    ]
    assert outputs.master_summary["drop_collision_count"] == 1
    assert outputs.master_summary["feasible_node_count"] == 1
    summary, ptx_history, margin_history, tables = outputs.to_legacy_tuple()
    assert summary is outputs.master_summary
    assert ptx_history is outputs.ptx_history
    assert margin_history is outputs.margin_history
    assert tables is outputs.tables


class FakeExporter:
    def __init__(self):
        self.calls = []

    def export_bundle_to_excel(self, bundle, filepath):
        self.calls.append((bundle, filepath))


def _export_config(output_dir: Path, **overrides):
    config = {
        "output_dir": output_dir,
        "export_summary": False,
        "export_results": False,
        "export_debug": False,
        "print_summary": False,
        "verbose": False,
        "node_num": 8,
        "relay_num": 2,
        "terrain_delta_h": 90,
        "node_ptx_mode": "adaptive",
        "relay_ptx_mode": "fixed",
        "node_bw": 125000,
        "node_ptx_init": None,
        "sim_time": 120,
    }
    config.update(overrides)
    return config


def test_disabled_export_has_no_filesystem_or_statistics_side_effect(tmp_path):
    output_dir = tmp_path / "not-created"
    statistics = FakeExporter()
    outputs = collect_simulation_outputs(statistics=FakeStatistics(), routing=_routing())

    result = export_simulation_outputs(
        config=_export_config(output_dir),
        run_id="disabled",
        statistics=statistics,
        outputs=outputs,
    )

    assert result == {}
    assert not output_dir.exists()
    assert statistics.calls == []


def test_enabled_exports_use_separate_stable_workbook_names(tmp_path):
    statistics = FakeExporter()
    outputs = collect_simulation_outputs(statistics=FakeStatistics(), routing=_routing())

    result = export_simulation_outputs(
        config=_export_config(
            tmp_path,
            export_summary=True,
            export_results=True,
            export_debug=True,
        ),
        run_id="run-7",
        statistics=statistics,
        outputs=outputs,
    )

    assert set(result) == {"summary", "results", "debug"}
    assert Path(result["summary"]).name.startswith("summary_run_run-7_")
    assert Path(result["results"]).name.startswith("results_run_run-7_")
    assert Path(result["debug"]).name.startswith("debug_run_run-7_")
    assert statistics.calls == [
        (outputs.summary_bundle, result["summary"]),
        (outputs.result_bundle, result["results"]),
        (outputs.debug_bundle, result["debug"]),
    ]


def test_large_debug_table_uses_five_head_and_five_tail_rows(capsys):
    _display_table(pd.DataFrame({"value": range(20)}), max_rows=10)

    output = capsys.readouterr().out
    assert " 0\n" in output
    assert " 4\n" in output
    assert "..\n" in output
    assert "15\n" in output
    assert "19\n" in output
    assert " 5\n" not in output
    assert "preview 10 of 20 rows" in output
