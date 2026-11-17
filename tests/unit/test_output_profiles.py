from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from loraphymac.experiments.apc_load_sweep.profile_acceptance_pilot import _table_hash
from loraphymac.config.defaults import build_config
from loraphymac.config.validation import validate_configuration
from loraphymac.mac.frame_defs import MType
from loraphymac.metrics.instrumentation_metrics import (
    build_tx_airtime_by_entity_mtype_ptx_channel_df,
    build_channel_occupancy_df,
    build_node_tx_duty_cycle_df,
)
from loraphymac.metrics.collision_metrics import build_collision_summary_df
from loraphymac.metrics.statistics import Statistics
from loraphymac.runtime.packet import Packet
from loraphymac.simulation.runner import run_single_simulation


def _packet(*, uid, start, toa, entity="N1", channel=1, mtype=MType.DATA_UP):
    return Packet(
        src_id=entity,
        current_tx=entity,
        next_hop="R1",
        path=[entity, "R1"],
        hop_index=0,
        ch_id=channel,
        freq_mhz=921.4,
        sf=7,
        bw_hz=125_000,
        ptx_dbm=17.0,
        payload_size=50,
        toa=toa,
        tx_start=start,
        tx_end=start + toa,
        packet_uid=uid,
        fcnt=int(uid[-1]),
        mtype=int(mtype),
    )


def test_output_profile_default_is_detailed_and_invalid_value_rejected():
    assert build_config()["output_profile"] == "detailed"
    config = build_config({"output_profile": "compact"})
    with pytest.raises(ValueError, match="output_profile"):
        validate_configuration(config, known_keys=None)


def test_summary_streaming_airtime_matches_event_interval_union_exactly():
    stats = Statistics(output_profile="summary")
    stats.set_run_metadata(run_id="unit", sim_time=10.0)
    packets = [
        (_packet(uid="N1:1", start=1.0, toa=3.0), "N1"),
        (_packet(uid="N2:2", start=2.0, toa=3.0, entity="N2"), "N2"),
        (_packet(uid="R1:3", start=5.0, toa=2.0, entity="R1", mtype=MType.DATA_UP_FWD), "R1"),
        (_packet(uid="N1:4", start=9.0, toa=3.0), "N1"),
    ]
    for packet, entity in packets:
        stats.record_tx(packet, entity)

    detailed_log = pd.DataFrame(
        [
            {
                "event": "TX", "node": entity, "time": packet.tx_start,
                "toa_s": packet.toa,
                "observed_tx_airtime_s": min(packet.toa, 10.0 - packet.tx_start),
                "ch": packet.ch_id,
            }
            for packet, entity in packets
        ]
    )
    expected_occupancy = build_channel_occupancy_df(
        detailed_log, {"run_id": "unit", "sim_time": 10.0}
    )
    observed_occupancy = stats.get_summary_channel_occupancy_df()
    pd.testing.assert_frame_equal(
        observed_occupancy,
        expected_occupancy,
        check_dtype=False,
        check_exact=True,
    )
    assert observed_occupancy["channel_busy_union_s"].max() <= 10.0
    assert observed_occupancy["sum_observed_tx_airtime_s"].sum() > (
        observed_occupancy["channel_busy_union_s"].sum()
    )

    expected_duty = build_node_tx_duty_cycle_df(
        detailed_log,
        [type("Entity", (), {"id": "N1"})(), type("Entity", (), {"id": "N2"})()],
        {"run_id": "unit", "sim_time": 10.0},
    )
    observed_duty = stats.get_summary_node_tx_duty_cycle_df(
        [type("Entity", (), {"id": "N1"})(), type("Entity", (), {"id": "N2"})()]
    )
    pd.testing.assert_frame_equal(observed_duty, expected_duty, check_exact=True)
    assert stats.to_dataframe().empty


@pytest.mark.parametrize("include_apc_retries", [False, True], ids=["ff", "aa-retries"])
def test_summary_detailed_airtime_and_occupancy_match_for_50_node_like_records(
    include_apc_retries,
):
    sim_time = 3600.0
    stats = Statistics(output_profile="summary")
    stats.set_run_metadata(run_id="summary", sim_time=sim_time)
    events = []
    airtime_by_entity = {}
    start = 0.0
    packet_index = 0

    def add_tx(entity_id, mtype, channel):
        nonlocal start, packet_index
        toa = 0.731 + (packet_index % 7) * 0.137
        uid = f"{entity_id}:{packet_index}"
        packet = _packet(
            uid=uid,
            start=start,
            toa=toa,
            entity=entity_id,
            channel=channel,
            mtype=mtype,
        )
        stats.record_tx(packet, entity_id)
        observed = min(toa, sim_time - start)
        airtime_by_entity[entity_id] = airtime_by_entity.get(entity_id, 0.0) + observed
        events.append({
            "event": "TX", "node": entity_id, "time": start, "ch": channel,
            "toa_s": toa, "observed_tx_airtime_s": observed,
            "mtype": int(mtype), "ptx": packet.ptx_dbm, "sf": packet.sf,
            "bw_hz": packet.bw_hz,
        })
        start += 0.19  # overlapping transmissions across transmitters are intentional.
        packet_index += 1

    for node in range(50):
        for attempt in range(20):
            add_tx(f"N{node}", MType.DATA_UP, attempt % 8)
    if include_apc_retries:
        for channel in (0, 1, 2):
            add_tx("R1", MType.DATA_DOWN, channel)

    detailed_log = pd.DataFrame(events)
    entities = [
        SimpleNamespace(id=entity_id, total_tx_time_s=airtime)
        for entity_id, airtime in airtime_by_entity.items()
    ]
    summary_airtime = stats.get_tx_airtime_by_entity_mtype_ptx_channel_df(entities)
    detailed_airtime = build_tx_airtime_by_entity_mtype_ptx_channel_df(
        events, entities, {"run_id": "detailed"}
    )
    pd.testing.assert_frame_equal(
        summary_airtime.drop(columns="run_id"),
        detailed_airtime.drop(columns="run_id"),
        check_dtype=False,
        check_exact=True,
    )
    assert summary_airtime["ptx_dbm"].dtype == detailed_airtime["ptx_dbm"].dtype
    assert _table_hash(summary_airtime) == _table_hash(detailed_airtime)

    summary_occupancy = stats.get_summary_channel_occupancy_df()
    detailed_occupancy = build_channel_occupancy_df(
        detailed_log, {"run_id": "detailed", "sim_time": sim_time}
    )
    pd.testing.assert_frame_equal(
        summary_occupancy.drop(columns="run_id"),
        detailed_occupancy.drop(columns="run_id"),
        check_dtype=False,
        check_exact=True,
    )
    assert detailed_occupancy["channel_busy_union_s"].le(sim_time).all()
    if include_apc_retries:
        control = summary_airtime.loc[
            summary_airtime["traffic_class"].eq("control_feedback")
        ]
        assert int(control["tx_packet_count"].sum()) == 3
    else:
        assert not summary_airtime["traffic_class"].eq("control_feedback").any()


def test_collision_summary_means_are_exact_under_event_reordering():
    records = pd.DataFrame([
        {"collision_model_version": "v1", "spectral_model_version": "s1",
         "collision_mode": "capture", "sf_interference_mode": "sir_matrix",
         "sir_matrix_version": "matrix", "resolution_id": i,
         "receiver_id": "R1", "hop_index": 0, "mtype": 1, "ch_id": 2,
         "sf": 7, "outcome": "CAPTURE_SURVIVOR", "survived": True,
         "direct_interferer_count": 1, "contender_count": 2,
         "capture_delta_db": value, "min_pair_margin_db": value / 3,
         "max_pair_overlap_s": abs(value) / 10}
        for i, value in enumerate((1e16, 1.0, -1e16), start=1)
    ])
    forward = build_collision_summary_df(records)
    reverse = build_collision_summary_df(records.iloc[::-1].reset_index(drop=True))
    pd.testing.assert_frame_equal(forward, reverse, check_exact=True)


def test_summary_and_detailed_paired_smoke_have_exact_core_metrics():
    common = {
        "node_num": 4,
        "relay_num": 1,
        "sim_time": 120,
        "node_period": 20,
        "node_jitter": 0,
        "seed": 31,
        "energy_accounting_mode": "corrected",
        "plot_routing": False,
        "print_summary": False,
        "verbose": False,
        "export_summary": False,
        "export_results": False,
        "export_debug": False,
    }
    summary = run_single_simulation({**common, "output_profile": "summary"})
    summary_python_rng = random.getstate()
    summary_numpy_rng = np.random.get_state()
    detailed = run_single_simulation({**common, "output_profile": "detailed"})
    detailed_python_rng = random.getstate()
    detailed_numpy_rng = np.random.get_state()

    assert summary_python_rng == detailed_python_rng
    assert summary_numpy_rng[0] == detailed_numpy_rng[0]
    np.testing.assert_array_equal(summary_numpy_rng[1], detailed_numpy_rng[1])
    assert summary_numpy_rng[2:] == detailed_numpy_rng[2:]
    assert summary[0] == detailed[0]
    pd.testing.assert_frame_equal(summary[1], detailed[1], check_exact=True)
    pd.testing.assert_frame_equal(summary[2], detailed[2], check_exact=True)
    assert summary[3]["event_log_df"].empty
    assert not detailed[3]["event_log_df"].empty
    schedule_columns = ["packet_uid", "fcnt", "birth_time", "offered_time"]
    pd.testing.assert_frame_equal(
        summary[3]["packet_fate_df"][schedule_columns],
        detailed[3]["packet_fate_df"][schedule_columns],
        check_exact=True,
    )
    for table in (
        "core_summary_df", "energy_breakdown_df", "node_rx_matrix_df",
        "hop_summary_df", "frame_summary_df", "frame_mtype_summary_df",
        "frame_hop_event_summary_df", "frame_drop_reason_by_hop_df",
        "relay_duty_cycle_df", "per_channel_duty_cycle_df",
        "tx_queue_waiting_time_summary_df", "relay_tx_queue_metrics_df",
        "tx_queue_traffic_class_metrics_df", "final_ptx_distribution_df",
        "relay_rx_stats_df", "rx_channel_usage_df", "collision_summary_df",
        "demodulator_pool_summary_df", "packet_fate_summary_df",
        "apc_feedback_lifecycle_df", "wang_snr_report_lifecycle_df",
        "wang_power_command_lifecycle_df", "routing_table_df",
        "routing_map_df", "topology_parent_by_run_df",
        "tx_airtime_by_entity_mtype_ptx_channel_df", "ptx_history_df",
        "margin_history_df", "apc_control_history_df",
    ):
        pd.testing.assert_frame_equal(
            summary[3][table], detailed[3][table],
            check_dtype=False, check_exact=True,
            obj=table,
        )
