from types import SimpleNamespace

import pytest
import simpy

from loraphymac.mac.collision_engine import CollisionEngine
from loraphymac.metrics.statistics import Statistics
from loraphymac.runtime.demodulator_pool import DemodulatorPool
from loraphymac.runtime.packet import Packet
from loraphymac.runtime.relay_rx import RelayRX


pytestmark = pytest.mark.integration


class FixedPropagation:
    def pathloss(self, **kwargs):
        return 100.0


class FixedAirInterface:
    def compute_link_budget(self, **kwargs):
        return {
            "feasible": True,
            "rssi": -100.0,
            "snr": 10.0,
            "margin": 20.0,
        }


def make_packet(src_id, next_hop="R1"):
    packet = Packet(
        src_id=src_id,
        current_tx=src_id,
        next_hop=next_hop,
        path=[src_id, "R1"],
        hop_index=0,
        ch_id=0,
        freq_mhz=923.2,
        sf=7,
        bw_hz=125_000,
        ptx_dbm=14.0,
        payload_size=20,
        app_payload_size=8,
        toa=1.0,
        tx_start=0.0,
        tx_end=1.0,
        fcnt=1,
    )
    packet.t_arrive = 0.0
    return packet


def test_demodulator_occupies_rx_interval_and_collision_is_terminal_cause():
    env = simpy.Environment()
    stats = Statistics()
    tx1 = SimpleNamespace(id="N1", h=2.0, gain=0.0)
    tx2 = SimpleNamespace(id="N2", h=2.0, gain=0.0)
    tx1.distance_to = lambda other: 100.0
    tx2.distance_to = lambda other: 100.0
    relay = SimpleNamespace(
        id="R1",
        h=10.0,
        gain=0.0,
        hw=SimpleNamespace(tx_full_duplex=True),
        network_nodes_map={"N1": tx1, "N2": tx2},
    )
    demod = DemodulatorPool(
        env,
        pool_a_size=0,
        pool_b_size=1,
        receiver_id="R1",
        statistics=stats,
    )
    receiver = RelayRX(
        env=env,
        relay=relay,
        air_interface=FixedAirInterface(),
        model=FixedPropagation(),
        terrain_delta_h=0.0,
        collision_engine=CollisionEngine(mode="capture"),
        demodulator_pool=demod,
        relay_tx=SimpleNamespace(),
        statistics=stats,
    )

    receiver.on_arrival(make_packet("N1"))
    receiver.on_arrival(make_packet("N2"))
    env.run()

    demod_events = stats.get_demodulator_event_log_df()
    acquired = demod_events[demod_events["event"] == "ACQUIRE"]
    rejected = demod_events[demod_events["event"] == "REJECT"]
    released = demod_events[demod_events["event"] == "RELEASE"]
    assert len(acquired) == len(rejected) == len(released) == 1
    assert acquired.iloc[0]["time"] == pytest.approx(acquired.iloc[0]["rx_start"])
    assert rejected.iloc[0]["time"] == pytest.approx(rejected.iloc[0]["rx_start"])
    assert released.iloc[0]["time"] == pytest.approx(released.iloc[0]["rx_end"])

    collision_events = stats.get_collision_event_log_df()
    assert len(collision_events) == 2
    assert set(collision_events["outcome"]) == {"COLLISION_DROP"}
    assert set(collision_events["demod_admitted"].astype(bool)) == {False, True}
    assert set(collision_events["reason"]) == {"collision"}

    relay_stats = stats.relay_stats["R1"]
    assert relay_stats["rx_collision"] == 2
    assert relay_stats["demodulator_drop_count"] == 0

    summary = stats.get_collision_summary_df()
    assert int(summary["decision_count"].sum()) == 2
    assert int(summary["collision_opportunity_count"].sum()) == 2
    assert int(summary["collision_drop_count"].sum()) == 2
    assert int(summary["outcome_conservation_delta"].sum()) == 0


def test_non_target_rf_burst_interferes_without_creating_duplicate_packet_fate():
    env = simpy.Environment()
    stats = Statistics()
    tx1 = SimpleNamespace(id="N1", h=2.0, gain=0.0)
    tx2 = SimpleNamespace(id="N2", h=2.0, gain=0.0)
    tx1.distance_to = lambda other: 100.0
    tx2.distance_to = lambda other: 100.0
    relay = SimpleNamespace(
        id="R1",
        h=10.0,
        gain=0.0,
        hw=SimpleNamespace(tx_full_duplex=True),
        network_nodes_map={"N1": tx1, "N2": tx2},
    )
    receiver = RelayRX(
        env=env,
        relay=relay,
        air_interface=FixedAirInterface(),
        model=FixedPropagation(),
        terrain_delta_h=0.0,
        collision_engine=CollisionEngine(mode="capture"),
        demodulator_pool=DemodulatorPool(
            env,
            pool_a_size=0,
            pool_b_size=8,
            receiver_id="R1",
            statistics=stats,
        ),
        relay_tx=SimpleNamespace(),
        statistics=stats,
    )

    target = make_packet("N1", next_hop="R1")
    interferer = make_packet("N2", next_hop="R2")
    receiver.on_arrival(target)
    receiver.on_arrival(interferer)
    env.run()

    decisions = stats.get_collision_event_log_df()
    assert len(decisions) == 2
    assert set(decisions["receiver_id"]) == {"R1"}
    assert stats.relay_stats["R1"]["rx_collision"] == 1
    drops = [row for row in stats.records if row.get("event") == "DROP"]
    assert len(drops) == 1
    assert drops[0]["src"] == "N1"
