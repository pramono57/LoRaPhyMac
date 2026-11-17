from __future__ import annotations

import pandas as pd
import pytest

from loraphymac.metrics import statistics as statistics_module
from loraphymac.metrics.statistics import Statistics


def test_statistics_summary_bundle_is_a_facade(monkeypatch):
    statistics = Statistics()
    nodes = [object()]
    expected = {"summary": pd.DataFrame([{"value": 1}])}
    calls = []

    def build_summary(facade, received_nodes):
        calls.append((facade, received_nodes))
        return expected

    monkeypatch.setattr(statistics_module, "_build_summary_bundle", build_summary)

    assert statistics.get_summary_bundle(nodes) is expected
    assert calls == [(statistics, nodes)]


def test_statistics_result_bundle_is_a_facade(monkeypatch):
    statistics = Statistics()
    nodes = [object()]
    selected = pd.DataFrame([{"source": "N1", "next_hop": "GW"}])
    relay_chain = pd.DataFrame([{"source": "R1", "next_hop": "GW"}])
    visual = selected.copy()
    expected = {"result": pd.DataFrame([{"value": 1}])}
    calls = []

    def build_result(facade, *args):
        calls.append((facade, args))
        return expected

    monkeypatch.setattr(statistics_module, "_build_result_bundle", build_result)

    result = statistics.get_result_bundle(nodes, selected, relay_chain, visual)

    assert result is expected
    assert calls == [(statistics, (nodes, selected, relay_chain, visual))]


def test_statistics_debug_bundle_keeps_legacy_signature(monkeypatch):
    statistics = Statistics()
    expected = {"debug": pd.DataFrame([{"value": 1}])}
    calls = []

    def build_debug(facade):
        calls.append(facade)
        return expected

    monkeypatch.setattr(statistics_module, "_build_debug_bundle", build_debug)

    result = statistics.get_debug_bundle(
        nodes=[object()],
        selected_next_hop_links_df=pd.DataFrame(),
    )

    assert result is expected
    assert calls == [statistics]


def test_statistics_packet_tables_are_facade_methods(monkeypatch):
    statistics = Statistics()
    fate = pd.DataFrame([{"packet_uid": "UP:N1:1"}])
    fate_summary = pd.DataFrame([{"offered_count": 1}])
    delivered = pd.DataFrame([{"packet_uid": "UP:N1:1"}])
    failures = pd.DataFrame([{"packet_uid": "UP:N1:2"}])

    monkeypatch.setattr(
        statistics_module,
        "_build_packet_fate_df",
        lambda packet_fates: fate,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_packet_fate_summary_df",
        lambda frame, metadata, conflict_count: fate_summary,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_delivered_packet_df",
        lambda records: delivered,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_packet_failure_df",
        lambda event_log: failures,
    )

    assert statistics.get_packet_fate_df() is fate
    assert statistics.get_packet_fate_summary_df() is fate_summary
    assert statistics.get_delivered_packet_df() is delivered
    assert statistics.get_packet_failure_df() is failures


def test_statistics_collision_and_demodulator_tables_are_facade_methods(monkeypatch):
    statistics = Statistics()
    demod_events = pd.DataFrame([{"event": "ACQUIRE"}])
    demod_summary = pd.DataFrame([{"acquire_success_count": 1}])
    collision_events = pd.DataFrame([{"outcome": "CAPTURE_SURVIVOR"}])
    collision_edges = pd.DataFrame([{"pair_pass": True}])
    collision_summary = pd.DataFrame([{"capture_success_count": 1}])

    monkeypatch.setattr(
        statistics_module,
        "_build_demodulator_event_log_df",
        lambda records: demod_events,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_demodulator_pool_summary_df",
        lambda frame: demod_summary,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_collision_event_log_df",
        lambda records: collision_events,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_collision_interference_edges_df",
        lambda records: collision_edges,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_collision_summary_df",
        lambda frame: collision_summary,
    )

    assert statistics.get_demodulator_event_log_df() is demod_events
    assert statistics.get_demodulator_pool_summary_df() is demod_summary
    assert statistics.get_collision_event_log_df() is collision_events
    assert statistics.get_collision_interference_edges_df() is collision_edges
    assert statistics.get_collision_summary_df() is collision_summary


def test_statistics_receiver_tables_are_facade_methods(monkeypatch):
    statistics = Statistics()
    relay = pd.DataFrame([{"relay": "R1"}])
    channel = pd.DataFrame([{"relay": "R1", "channel": 0}])
    node = pd.DataFrame([{"node": "N1"}])
    hop = pd.DataFrame([{"relay_count": 1}])

    monkeypatch.setattr(
        statistics_module,
        "_build_relay_rx_stats_df",
        lambda counters: relay,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_rx_channel_usage_df",
        lambda counts, durations: channel,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_node_rx_matrix_df",
        lambda counters: node,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_hop_summary_df",
        lambda frame: hop,
    )

    assert statistics.get_relay_rx_stats_df() is relay
    assert statistics.get_rx_channel_usage_df() is channel
    assert statistics.get_node_rx_matrix_df() is node
    assert statistics.get_hop_summary_df() is hop


def test_statistics_energy_and_duty_cycle_tables_are_facade_methods(monkeypatch):
    statistics = Statistics()
    entity = object()
    energy = pd.DataFrame([{"total_energy_J": 1.0}])
    relay_duty = pd.DataFrame([{"relay_duty_cycle_percent": 1.0}])
    channel_duty = pd.DataFrame([{"per_channel_duty_cycle_percent": 1.0}])
    continuous = (1000.0, 10.0, 1.0)

    monkeypatch.setattr(
        statistics_module,
        "_continuous_rx_energy_mj",
        lambda node, sim_time, accounting_mode: continuous,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_energy_breakdown_df",
        lambda nodes, metadata, infer, rx_energy, convert: energy,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_relay_duty_cycle_df",
        lambda nodes, metadata, infer: relay_duty,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_per_channel_duty_cycle_df",
        lambda nodes, sim_time, metadata, infer: channel_duty,
    )

    assert statistics._continuous_rx_energy_mJ(entity) is continuous
    assert statistics.get_energy_breakdown_df([entity]) is energy
    assert statistics.get_relay_duty_cycle_df([entity]) is relay_duty
    assert statistics.get_per_channel_duty_cycle_df([entity]) is channel_duty


def test_statistics_queue_tables_are_facade_methods(monkeypatch):
    statistics = Statistics()
    statistics.run_metadata["sim_time"] = 10.0
    entity = object()
    lifecycle = pd.DataFrame([{"queue_visit_id": "q1"}])
    waiting = pd.DataFrame([{"relay": "R1"}])
    waiting_summary = pd.DataFrame([{"relay": "R1", "count": 1}])
    queue_v2 = pd.DataFrame([{"entity_id": "R1", "queue_metrics_version": "2.0"}])
    traffic_class = pd.DataFrame([{"traffic_class": "DATA_UP_FWD"}])

    monkeypatch.setattr(
        statistics_module,
        "_build_tx_queue_lifecycle_df",
        lambda records: lifecycle,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_tx_queue_waiting_time_df",
        lambda waiting_times: waiting,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_tx_queue_waiting_time_summary_df",
        lambda *args: waiting_summary,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_relay_tx_queue_metrics_df",
        lambda *args: queue_v2,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_tx_queue_traffic_class_metrics_df",
        lambda *args: traffic_class,
    )

    assert statistics.get_tx_queue_lifecycle_df() is lifecycle
    assert statistics.get_tx_queue_waiting_time_df() is waiting
    assert statistics.get_tx_queue_waiting_time_summary_df() is waiting_summary
    assert statistics.get_relay_tx_queue_metrics_df([entity]) is queue_v2
    assert statistics.get_relay_tx_mgc_queue_metrics_df([entity]) is queue_v2
    assert statistics.get_tx_queue_traffic_class_metrics_df() is traffic_class


def test_statistics_frame_tables_are_facade_methods(monkeypatch):
    statistics = Statistics()
    statistics.run_metadata["run_id"] = "frame-facade"
    statistics.ptx_history.append({"new_ptx": 10})
    normalized = pd.DataFrame([{"event": "TX", "mtype": 1}])
    delivered = pd.DataFrame([{"packet_uid": "UP:N1:1"}])
    summary = pd.DataFrame([{"stage1_status": "PASS"}])
    mtype = pd.DataFrame([{"mtype_name": "DATA_UP"}])
    hop = pd.DataFrame([{"hop_count": 1}])
    drops = pd.DataFrame([{"reason": "collision"}])
    summary_calls = []

    monkeypatch.setattr(
        statistics_module,
        "_normalize_frame_event_df",
        lambda event_log: normalized,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_delivered_packet_df",
        lambda records: delivered,
    )

    def build_summary(frame_events, metadata, ptx_update_count, delivered_count):
        summary_calls.append(
            (frame_events, metadata, ptx_update_count, delivered_count)
        )
        return summary

    monkeypatch.setattr(statistics_module, "_build_frame_summary_df", build_summary)
    monkeypatch.setattr(
        statistics_module,
        "_build_frame_mtype_summary_df",
        lambda frame_events: mtype,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_frame_hop_event_summary_df",
        lambda frame_events: hop,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_frame_drop_reason_by_hop_df",
        lambda frame_events: drops,
    )

    assert statistics._get_frame_event_df() is normalized
    assert statistics.get_frame_summary_df() is summary
    assert summary_calls == [(normalized, statistics.run_metadata, 1, 1)]
    assert statistics.get_frame_mtype_summary_df() is mtype
    assert statistics.get_frame_hop_event_summary_df() is hop
    assert statistics.get_frame_drop_reason_by_hop_df() is drops


def test_statistics_core_summary_is_a_facade_method(monkeypatch):
    statistics = Statistics()
    nodes = [object()]
    expected = pd.DataFrame([{"run_id": "core-facade"}])
    calls = []

    def build_core(facade, received_nodes):
        calls.append((facade, received_nodes))
        return expected

    monkeypatch.setattr(statistics_module, "_build_core_summary_df", build_core)

    assert statistics.get_core_summary_df(nodes) is expected
    assert calls == [(statistics, nodes)]


def test_statistics_apc_ptx_tables_are_facade_methods(monkeypatch):
    statistics = Statistics()
    nodes = [object()]
    ptx = pd.DataFrame([{"new_ptx": 10.0}])
    control = pd.DataFrame([{"event": "APC_REQ_RX"}])
    margin = pd.DataFrame([{"margin": 7.0}])
    final = pd.DataFrame([{"entity_id": "N1", "final_ptx": 10.0}])
    final_calls = []

    monkeypatch.setattr(
        statistics_module,
        "_build_ptx_history_df",
        lambda records: ptx,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_apc_control_history_df",
        lambda records: control,
    )
    monkeypatch.setattr(
        statistics_module,
        "_build_margin_history_df",
        lambda records: margin,
    )

    def build_final(received_nodes, metadata, infer_entity_type):
        final_calls.append((received_nodes, metadata, infer_entity_type))
        return final

    monkeypatch.setattr(
        statistics_module,
        "_build_final_ptx_distribution_df",
        build_final,
    )

    assert statistics.get_ptx_history_df() is ptx
    assert statistics.get_apc_control_history_df() is control
    assert statistics.get_margin_history_df() is margin
    assert statistics.get_final_ptx_distribution_df(nodes) is final
    assert final_calls == [(nodes, statistics.run_metadata, statistics._infer_entity_type)]


def test_statistics_views_preserve_column_order_and_ignore_missing_columns():
    statistics = Statistics()
    source = pd.DataFrame(
        [
            {
                "unrelated": 99,
                "avg_e2e_delay_ms": 12.5,
                "run_id": "run-1",
                "pdr_percent": 80.0,
            }
        ]
    )

    actual = statistics.view_core_summary(source)

    assert list(actual.columns) == ["run_id", "pdr_percent", "avg_e2e_delay_ms"]
    assert actual.to_dict("records") == [
        {"run_id": "run-1", "pdr_percent": 80.0, "avg_e2e_delay_ms": 12.5}
    ]


@pytest.mark.filterwarnings("ignore:datetime.datetime.utcnow.*:DeprecationWarning")
def test_statistics_excel_facade_writes_placeholder_for_empty_bundle(tmp_path):
    output = tmp_path / "empty.xlsx"

    Statistics().export_bundle_to_excel({"empty": pd.DataFrame()}, output)

    assert pd.ExcelFile(output).sheet_names == ["empty"]
    assert pd.read_excel(output).to_dict("records") == [
        {"message": "No data available for this output mode."}
    ]
