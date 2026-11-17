from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from loraphymac.experiments.apc_terrain_sweep.routing_audit import (
    build_parent_change_summary,
    mode_parent_inconsistencies,
)
from loraphymac.mac.frame_defs import MType
from loraphymac.metrics.instrumentation_metrics import (
    APC_FEEDBACK_LIFECYCLE_COLUMNS,
    TOPOLOGY_PARENT_COLUMNS,
    TX_AIRTIME_COLUMNS,
    build_apc_feedback_lifecycle_df,
    build_topology_parent_by_run_df,
    build_tx_airtime_by_entity_mtype_ptx_channel_df,
)
from loraphymac.metrics.statistics import Statistics
from loraphymac.runtime.node_rx import NodeRX
from loraphymac.runtime.packet import Packet
from loraphymac.simulation.context import RoutingArtifacts


def _entity(entity_id, x, parent_airtime=0.0):
    return SimpleNamespace(
        id=entity_id,
        loc=SimpleNamespace(x=x, y=0.0),
        h=2.0,
        total_tx_time_s=parent_airtime,
    )


def test_topology_parent_export_contains_route_selection_evidence():
    node = _entity("N1", 100.0)
    relay = _entity("R1", 50.0)
    gateway = _entity("GW", 0.0)
    selected = pd.DataFrame(
        [
            {
                "Node": "N1",
                "Target": "R1",
                "ChID": 0,
                "Freq(MHz)": 923.2,
                "Distance(m)": 50.0,
                "Pathloss(dB)": 110.0,
                "RSSI(dBm)": -90.0,
                "SNR(dB)": 8.0,
                "Margin(dB)": 12.0,
                "Hop": 1,
                "Score": 10.0,
                "Valid": True,
            }
        ]
    )
    relay_link = selected.assign(Node="R1", Target="GW")
    routing = RoutingArtifacts(
        relay_hops={"R1": 1, "GW": 0},
        link_df=pd.concat(
            [selected, selected.assign(Target="GW")],
            ignore_index=True,
        ),
        routing_table={"N1": "R1"},
        selected_next_hop_links_df=selected,
        routing_map={"N1": "R1", "R1": "GW", "GW": None},
        all_nodes=[node, relay, gateway],
        node_map={"N1": node, "R1": relay, "GW": gateway},
        relay_chain_links_df=relay_link,
        visual_links_df=pd.concat([selected, relay_link], ignore_index=True),
    )

    result = build_topology_parent_by_run_df(
        routing,
        {
            "run_id": "route-1",
            "seed": 30,
            "node_ptx_mode": "fixed",
            "relay_ptx_mode": "adaptive",
            "terrain_delta_h": 90,
        },
    )

    assert result.columns.tolist() == TOPOLOGY_PARENT_COLUMNS
    assert result["parent_id"].tolist() == ["R1", "GW", None]
    assert result.iloc[0]["candidate_count"] == 2
    assert result.iloc[0]["selection_rule"] == "highest_score"
    assert result.iloc[0]["margin_db"] == 12.0


def test_airtime_export_groups_traffic_and_reconciles_entity_total():
    records = [
        {
            "event": "TX",
            "node": "N1",
            "mtype": MType.DATA_UP,
            "ptx": 17,
            "ch": 0,
            "sf": 7,
            "bw_hz": 125000,
            "observed_tx_airtime_s": 0.1,
        },
        {
            "event": "TX",
            "node": "N1",
            "mtype": MType.DATA_UP,
            "ptx": 17,
            "ch": 0,
            "sf": 7,
            "bw_hz": 125000,
            "observed_tx_airtime_s": 0.2,
        },
        {
            "event": "TX",
            "node": "R1",
            "mtype": MType.DATA_DOWN_FWD,
            "ptx": 20,
            "ch": 1,
            "sf": 7,
            "bw_hz": 125000,
            "observed_tx_airtime_s": 0.4,
        },
    ]

    result = build_tx_airtime_by_entity_mtype_ptx_channel_df(
        records,
        [_entity("N1", 0.0, 0.3), _entity("R1", 0.0, 0.4)],
        {"run_id": "airtime-1"},
    )

    assert result.columns.tolist() == TX_AIRTIME_COLUMNS
    assert result["traffic_class"].tolist() == ["uplink", "control_feedback"]
    assert result["tx_packet_count"].tolist() == [2, 1]
    assert result["total_airtime_s"].tolist() == pytest.approx([0.3, 0.4])
    assert result["airtime_consistent"].all()


def _feedback_packet():
    return Packet(
        src_id="N1",
        current_tx="R1",
        next_hop="N1",
        path=["R1", "N1"],
        hop_index=0,
        ch_id=0,
        freq_mhz=923.2,
        sf=7,
        bw_hz=125000,
        ptx_dbm=20,
        payload_size=17,
        toa=0.1,
        tx_start=2.0,
        tx_end=2.1,
        mtype=MType.DATA_DOWN,
        apc_info={
            "feedback_id": "run-1:apc:000001",
            "decision_time": 1.0,
            "retry_timeout_s": 5.0,
            "src_id": "N1",
            "hop_index": 0,
            "context": {
                "controller_id": "R1",
                "controller_type": "relay",
                "target_id": "N1",
                "target_type": "node",
                "control_scope": "node_apc",
                "old_tx_power_index": 0,
                "tx_power_index": 3,
            },
        },
    )


def test_feedback_lifecycle_records_observed_events_and_age():
    statistics = Statistics()
    statistics.set_run_metadata(run_id="run-1", sim_time=10.0)
    packet = _feedback_packet()
    statistics.record_apc_feedback_event(
        event="generated",
        time=1.0,
        apc_info=packet.apc_info,
        feedback_id="run-1:apc:000001",
        mtype=MType.DATA_DOWN,
    )
    statistics.record_tx(packet, "R1")
    packet.rx_time = 2.1
    packet.decoded = True
    statistics.record_rx(packet, "N1")
    statistics.record_apc_feedback_event(
        event="applied",
        time=2.1,
        packet=packet,
        applied_tx_power_index=3,
        stale_tpi=False,
    )

    result = statistics.get_apc_feedback_lifecycle_df()

    assert result.columns.tolist() == APC_FEEDBACK_LIFECYCLE_COLUMNS
    assert result["event"].tolist() == ["generated", "sent", "received", "applied"]
    assert result["feedback_age_s"].tolist() == [0.0, 1.0, 1.1, 1.1]
    assert bool(result.iloc[-1]["stale_tpi"]) is False
    assert result["feedback_id"].nunique() == 1
    assert build_apc_feedback_lifecycle_df([], {}).columns.tolist() == (
        APC_FEEDBACK_LIFECYCLE_COLUMNS
    )


def test_node_rx_does_not_silently_swallow_logging_failures():
    class FailingStatistics:
        def record_node_rx(self, *_args, **_kwargs):
            raise RuntimeError("logging failed")

    receiver = NodeRX(
        env=SimpleNamespace(now=3.0),
        node=SimpleNamespace(
            id="N1",
            rx_windows=[],
            is_transmitting=False,
            tx_busy_until=0.0,
        ),
        air_interface=None,
        model=None,
        terrain_delta_h=90,
        statistics=FailingStatistics(),
    )

    with pytest.raises(RuntimeError, match="logging failed"):
        receiver._drop(_feedback_packet(), "test_drop")


def test_feedback_collision_is_logged_only_when_observed():
    statistics = Statistics()
    statistics.set_run_metadata(run_id="run-collision", sim_time=10.0)
    packet = _feedback_packet()
    packet.collided = True
    packet.drop_time = 3.0

    statistics.record_drop(packet, reason="collision", node_id="N1")

    event = statistics.get_apc_feedback_lifecycle_df().iloc[0]
    assert event["event"] == "collided"
    assert event["reason"] == "collision"
    assert bool(event["collided"]) is True


def test_parent_change_and_mode_consistency_summaries():
    rows = []
    for mode in ("FF", "AF", "FA", "AA"):
        for terrain, parent in ((80, "R1"), (90, "GW")):
            rows.append(
                {
                    "seed": 30,
                    "terrain_delta_h": terrain,
                    "entity_id": "N1",
                    "entity_type": "node",
                    "parent_id": parent,
                    "mode": mode,
                }
            )
    parent_df = pd.DataFrame(rows)

    assert mode_parent_inconsistencies(parent_df).empty
    changes = build_parent_change_summary(parent_df)
    versus_80 = changes[
        (changes["comparison_type"] == "versus_reference")
        & (changes["seed"].astype(str) == "30")
        & (changes["terrain_delta_h"] == 80)
    ].iloc[0]
    assert versus_80["changed_node_percent"] == 100.0
