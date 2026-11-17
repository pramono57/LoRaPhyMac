from types import SimpleNamespace

import pytest
import simpy

from loraphymac.domain.node import Node, NodeState
from loraphymac.hardware.hardware import SX1276, SX1302
from loraphymac.phy.channel_plan import AS923_2
from loraphymac.runtime.packet import Packet
from loraphymac.runtime.relay_tx import RelayTX


pytestmark = pytest.mark.scientific


class RecordingAir:
    def __init__(self):
        self.packets = []

    def transmit(self, packet):
        self.packets.append(packet)


def make_node(env):
    return Node(
        node_id="N1",
        loc=SimpleNamespace(x=0.0, y=0.0),
        height=2.0,
        radio_hw=SX1276(),
        mcu_hw=None,
        gain_dbi=0.0,
        channel_plan=AS923_2(),
        air_interface=RecordingAir(),
        routing_engine=None,
        collision_engine=None,
        include_mcu_energy=False,
    )


def make_tx_packet(toa=2.0):
    return Packet(
        src_id="N1",
        current_tx="N1",
        next_hop="R1",
        path=["N1", "R1"],
        hop_index=0,
        ch_id=0,
        freq_mhz=923.2,
        sf=7,
        bw_hz=125_000,
        ptx_dbm=20,
        payload_size=20,
        app_payload_size=8,
        toa=toa,
        tx_start=0.0,
        tx_end=toa,
    )


def test_sleep_energy_is_clipped_at_observation_horizon():
    env = simpy.Environment()
    env.simulation_end = 1.0
    node = make_node(env)

    env.process(node.sleep_event(env, 20.0))
    env.run(until=env.simulation_end)

    expected_mj = node.hw.voltage * (node.hw.sleep_current / 1000.0) * 1.0 * 1000.0
    assert node.total_sleep_time_s == pytest.approx(1.0)
    assert node.energy_sleep_mJ == pytest.approx(expected_mj)


def test_node_tx_energy_is_clipped_at_observation_horizon():
    env = simpy.Environment()
    env.simulation_end = 1.0
    node = make_node(env)
    node.state = NodeState.FS
    packet = make_tx_packet(toa=2.0)

    env.process(
        node.tx_event(
            env=env,
            packet=packet,
            model=None,
            nexthop=SimpleNamespace(id="R1"),
            terrain_delta_h=90,
        )
    )
    env.run(until=env.simulation_end)

    observed_tx_s = env.simulation_end - packet.tx_start
    expected_mj = (
        node.hw.voltage
        * (node.hw.tx_current[20] / 1000.0)
        * observed_tx_s
        * 1000.0
    )
    assert 0.0 < observed_tx_s < packet.toa
    assert node.total_tx_time_s == pytest.approx(observed_tx_s)
    assert node.energy_tx_mJ == pytest.approx(expected_mj)


def test_open_rx_window_accounts_only_time_inside_horizon():
    env = simpy.Environment()
    env.simulation_end = 1.0
    node = make_node(env)

    env.process(
        node.open_rx_window(
            env=env,
            rx_start_s=0.0,
            rx_end_s=5.0,
            ch_id=0,
            bw_hz=125_000,
        )
    )
    env.run(until=env.simulation_end)

    expected_mj = node.hw.voltage * (node.hw.rx_current / 1000.0) * 1.0 * 1000.0
    assert node.total_rx_time_s == pytest.approx(1.0)
    assert node.energy_rx_mJ == pytest.approx(expected_mj)


def test_rx_window_early_close_removes_preaccounted_future_energy():
    env = simpy.Environment()
    env.simulation_end = 1.0
    node = make_node(env)

    env.process(
        node.open_rx_window(
            env=env,
            rx_start_s=0.0,
            rx_end_s=5.0,
            ch_id=0,
            bw_hz=125_000,
        )
    )

    def close_early():
        yield env.timeout(0.25)
        node.close_rx_window(env, reason="TEST_DOWNLINK")

    env.process(close_early())
    env.run(until=env.simulation_end)

    expected_mj = node.hw.voltage * (node.hw.rx_current / 1000.0) * 0.25 * 1000.0
    assert node.total_rx_time_s == pytest.approx(0.25)
    assert node.energy_rx_mJ == pytest.approx(expected_mj)


def test_rx_window_extension_adds_energy_once():
    env = simpy.Environment()
    env.simulation_end = 1.0
    node = make_node(env)

    env.process(
        node.open_rx_window(
            env=env,
            rx_start_s=0.0,
            rx_end_s=0.2,
            ch_id=0,
            bw_hz=125_000,
        )
    )

    def extend_window():
        yield env.timeout(0.1)
        assert node.extend_rx_window_until(env, 0.4, reason="TEST_PREAMBLE")

    env.process(extend_window())
    env.run(until=env.simulation_end)

    expected_mj = node.hw.voltage * (node.hw.rx_current / 1000.0) * 0.4 * 1000.0
    assert node.total_rx_time_s == pytest.approx(0.4)
    assert node.energy_rx_mJ == pytest.approx(expected_mj)


def test_relay_tx_energy_and_airtime_are_clipped_at_horizon():
    env = simpy.Environment()
    env.simulation_end = 1.0
    env.run(until=0.75)
    relay = SimpleNamespace(
        id="R1",
        hw=SX1302(),
        include_mcu_energy=False,
        mcu=None,
        energy_tx_mJ=0.0,
        total_tx_time_s=0.0,
        tx_airtime_per_channel={},
        tx_packet_count_per_channel={},
    )
    relay_tx = RelayTX.__new__(RelayTX)
    relay_tx.env = env
    relay_tx.relay = relay
    packet = make_tx_packet(toa=2.0)
    packet.current_tx = "R1"
    packet.ptx_dbm = 20

    relay_tx._record_airtime(packet)
    relay_tx._record_energy(packet)

    observed_tx_s = 0.25
    expected_mj = (
        relay.hw.voltage
        * (relay.hw.tx_current[20] / 1000.0)
        * observed_tx_s
        * 1000.0
    )
    assert relay.total_tx_time_s == pytest.approx(observed_tx_s)
    assert relay.tx_airtime_per_channel[0] == pytest.approx(observed_tx_s)
    assert relay.energy_tx_mJ == pytest.approx(expected_mj)
