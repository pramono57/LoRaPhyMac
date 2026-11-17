from types import SimpleNamespace

import pytest
import simpy

from loraphymac.config.defaults import DEFAULT_CONFIG, build_config
from loraphymac.config.validation import ConfigurationError, validate_configuration
from loraphymac.mac.collision_engine import CollisionEngine
from loraphymac.mac.frame_defs import MType, encode_fctrl
from loraphymac.metrics.statistics import Statistics
from loraphymac.phy.air_interface import AirInterface
from loraphymac.runtime.demodulator_pool import DemodulatorPool
from loraphymac.runtime.node_rx import NodeRX
from loraphymac.runtime.packet import Packet
from loraphymac.runtime.apc_commands import encode_apc_req
from loraphymac.runtime.wang_power_commands import (
    WangPowerCommand,
    encode_wang_power_command,
)


class DistancePathloss:
    def __init__(self):
        self.distances = []

    def pathloss(self, *, d_m, **_kwargs):
        self.distances.append(float(d_m))
        return float(d_m)


class DeterministicAir(AirInterface):
    def compute_link_budget(self, *, ptx, pathloss, **_kwargs):
        rssi = float(ptx) - float(pathloss)
        return {
            "rssi": rssi,
            "snr": rssi + 120.0,
            "margin": rssi + 123.0,
            "feasible": rssi > -123.0,
        }


class ApplyingCoordinator:
    def __init__(self):
        self.calls = 0

    def apply_at_node(self, node, _packet):
        self.calls += 1
        node.default_ptx = 5
        node.wang_last_applied_command_seq = 1
        return SimpleNamespace(applied=True, reason="absolute_target_applied")


def _entity(entity_id, distance_m, *, ptx=20):
    return SimpleNamespace(
        id=entity_id,
        h=10.0,
        gain=0.0,
        default_ptx=ptx,
        hw=SimpleNamespace(tx_current={value: 1.0 for value in range(5, 21)}),
        distance_to=lambda _target: float(distance_m),
    )


def _packet(
    *,
    current_tx,
    next_hop,
    mtype,
    ptx,
    start=10.0,
    freq=921.4,
    ch=0,
    sf=7,
    command=False,
):
    fopts = b""
    dst_id = next_hop
    if command:
        fopts = encode_wang_power_command(
            WangPowerCommand(
                target_id="N-A",
                command_seq=1,
                window_index=0,
                expected_ptx_dbm=17,
                expected_epoch=0,
                target_ptx_dbm=5,
                issued_at_s=10.0,
                expiry_s=20.0,
            )
        )
        dst_id = "N-A"
    return Packet(
        src_id=current_tx,
        current_tx=current_tx,
        prev_hop=current_tx,
        next_hop=next_hop,
        path=[current_tx, next_hop],
        hop_index=0,
        ch_id=ch,
        freq_mhz=freq,
        sf=sf,
        bw_hz=125000,
        ptx_dbm=ptx,
        payload_size=23,
        toa=0.1,
        tx_start=start,
        tx_end=start + 0.1,
        packet_uid=f"{current_tx}:{start}:{next_hop}",
        mtype=mtype,
        dst_id=dst_id,
        fopts=fopts,
    )


def _fixture(
    *,
    enabled=True,
    engine=None,
    desired_distance=100,
    interferer_distance=97,
    initial_time=10.0,
):
    env = simpy.Environment(initial_time=initial_time)
    stats = Statistics()
    air = DeterministicAir(env)
    model = DistancePathloss()
    coordinator = ApplyingCoordinator()
    gateway = _entity("GW", 200)
    gateway.wang_command_coordinator = coordinator
    relay_1 = _entity("R1", desired_distance)
    relay_2 = _entity("R2", 150)
    node_b = _entity("N-B", interferer_distance, ptx=17)
    node_a = _entity("N-A", 0, ptx=17)
    node_a.is_transmitting = False
    node_a.tx_busy_until = 0.0
    node_a.rx_window_enabled = True
    node_a.wang_snr_reporting_enabled = True
    node_a.wang_last_applied_command_seq = 0
    node_a.rx_windows = [
        {
            "start": 10.0,
            "end": 11.0,
            "ch_id": 0,
            "freq_mhz": 921.4,
            "bw_hz": 125000,
            "sf": 7,
            "closed": False,
        }
    ]
    node_a.extend_rx_window_until = lambda **_kwargs: True
    node_a.close_rx_window = lambda **_kwargs: True
    entities = {
        item.id: item
        for item in (node_a, node_b, relay_1, relay_2, gateway)
    }
    node_a.network_nodes_map = entities
    collision = engine or CollisionEngine(
        mode="capture",
        capture_threshold_dB=6.0,
        sf_interference_mode="ideal_orthogonal",
    )
    demod = DemodulatorPool(
        env,
        pool_a_size=0,
        pool_b_size=1,
        receiver_id=node_a.id,
        statistics=stats,
    )
    node_a.rx = NodeRX(
        env,
        node_a,
        air,
        model,
        90,
        stats,
        collision_engine=(collision if enabled else None),
        demodulator_pool=(demod if enabled else None),
        collision_enabled=enabled,
    )
    air.register_rx(node_a)
    air.rx_endpoints_map.update(entities)
    return env, air, node_a, coordinator, model, stats


def _transmit_pair(air, desired, interferer):
    air.transmit(desired)
    air.transmit(interferer)


def test_collision_gate_defaults_off_and_requires_rx1():
    assert DEFAULT_CONFIG["node_rx_collision_enabled"] is False
    config = build_config(
        {
            "node_rx_collision_enabled": True,
            "node_rx_window_enabled": False,
        }
    )
    with pytest.raises(ConfigurationError, match="requires node_rx_window_enabled"):
        validate_configuration(config, known_keys=DEFAULT_CONFIG)


def test_unrelated_node_uplink_collides_and_cannot_apply_wang_command():
    env, air, node, coordinator, model, stats = _fixture()
    desired = _packet(
        current_tx="R1", next_hop="N-A", mtype=MType.DATA_DOWN, ptx=20, command=True
    )
    interferer = _packet(
        current_tx="N-B", next_hop="R2", mtype=MType.DATA_UP, ptx=17
    )
    _transmit_pair(air, desired, interferer)
    env.run(until=10.2)

    assert node.default_ptx == 17 and coordinator.calls == 0
    assert {100.0, 97.0}.issubset(set(model.distances))
    drops = [row for row in stats.records if row.get("event") == "DROP"]
    assert len(drops) == 1 and drops[0]["reason"] == "collision"


def test_collided_rapc_feedback_cannot_change_node_power():
    env, air, node, _, _, stats = _fixture()
    node.apc_tx_power_index = 0
    desired = _packet(
        current_tx="R1", next_hop="N-A", mtype=MType.DATA_DOWN, ptx=20
    )
    desired.fopts = encode_apc_req(dr=5, tx_power_index=6)
    desired.fctrl = encode_fctrl(apc=True, fopts_len=len(desired.fopts))
    interferer = _packet(
        current_tx="N-B", next_hop="R2", mtype=MType.DATA_UP, ptx=17
    )
    _transmit_pair(air, desired, interferer)
    env.run(until=10.2)

    assert node.default_ptx == 17 and node.apc_tx_power_index == 0
    assert not stats.ptx_history


def test_capture_uses_received_power_at_target_node_not_destination():
    env, air, node, coordinator, _, _ = _fixture(
        desired_distance=80, interferer_distance=107
    )
    desired = _packet(
        current_tx="R1", next_hop="N-A", mtype=MType.DATA_DOWN, ptx=20, command=True
    )
    interferer = _packet(
        current_tx="N-B", next_hop="R2", mtype=MType.DATA_UP, ptx=17
    )
    _transmit_pair(air, desired, interferer)
    env.run(until=10.2)

    assert node.default_ptx == 5 and coordinator.calls == 1


def test_late_unrelated_uplink_is_included_before_downlink_finalization():
    env, air, node, coordinator, _, _ = _fixture()
    desired = _packet(
        current_tx="R1", next_hop="N-A", mtype=MType.DATA_DOWN, ptx=20, command=True
    )
    interferer = _packet(
        current_tx="N-B",
        next_hop="R2",
        mtype=MType.DATA_UP,
        ptx=17,
        start=10.05,
    )
    _transmit_pair(air, desired, interferer)
    env.run(until=10.25)

    assert node.default_ptx == 17 and coordinator.calls == 0


def test_burst_beginning_before_rx1_is_retained_when_it_physically_overlaps():
    env, air, node, coordinator, _, _ = _fixture(initial_time=9.95)
    desired = _packet(
        current_tx="R1", next_hop="N-A", mtype=MType.DATA_DOWN, ptx=20, command=True
    )
    interferer = _packet(
        current_tx="N-B",
        next_hop="R2",
        mtype=MType.DATA_UP,
        ptx=17,
        start=9.95,
    )
    _transmit_pair(air, desired, interferer)
    env.run(until=10.2)

    assert node.default_ptx == 17 and coordinator.calls == 0


@pytest.mark.parametrize(
    ("current_tx", "next_hop", "mtype"),
    [
        ("R2", "GW", MType.DATA_UP_FWD),
        ("GW", "N-B", MType.DATA_DOWN),
    ],
)
def test_unrelated_relay_and_gateway_bursts_are_physical_interferers(
    current_tx, next_hop, mtype
):
    env, air, node, coordinator, _, _ = _fixture()
    node.network_nodes_map[current_tx].distance_to = lambda _target: 100.0
    desired = _packet(
        current_tx="R1", next_hop="N-A", mtype=MType.DATA_DOWN, ptx=20, command=True
    )
    interferer = _packet(
        current_tx=current_tx,
        next_hop=next_hop,
        mtype=mtype,
        ptx=20,
    )
    _transmit_pair(air, desired, interferer)
    env.run(until=10.2)

    assert node.default_ptx == 17 and coordinator.calls == 0


@pytest.mark.parametrize(
    ("freq", "ch", "sf"),
    [(922.4, 5, 7), (921.4, 0, 8)],
)
def test_noninterfering_frequency_or_ideal_orthogonal_sf_does_not_collide(
    freq, ch, sf
):
    env, air, node, coordinator, _, _ = _fixture()
    desired = _packet(
        current_tx="R1", next_hop="N-A", mtype=MType.DATA_DOWN, ptx=20, command=True
    )
    interferer = _packet(
        current_tx="N-B",
        next_hop="R2",
        mtype=MType.DATA_UP,
        ptx=17,
        freq=freq,
        ch=ch,
        sf=sf,
    )
    _transmit_pair(air, desired, interferer)
    env.run(until=10.2)

    assert node.default_ptx == 5 and coordinator.calls == 1


def test_cross_sf_follows_existing_directional_sir_matrix():
    engine = CollisionEngine(sf_interference_mode="sir_matrix")
    env, air, node, coordinator, _, _ = _fixture(engine=engine)
    desired = _packet(
        current_tx="R1", next_hop="N-A", mtype=MType.DATA_DOWN, ptx=20, command=True
    )
    interferer = _packet(
        current_tx="N-B", next_hop="R2", mtype=MType.DATA_UP, ptx=17, sf=8
    )
    _transmit_pair(air, desired, interferer)
    env.run(until=10.2)

    # SF7 needs -16 dB SIR against SF8 in the configured shared matrix.
    assert node.default_ptx == 5 and coordinator.calls == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [("freq_mhz", 921.6), ("sf", 8), ("bw_hz", 250000)],
)
def test_desired_rx1_tuning_mismatch_prevents_decode(field, value):
    env, air, node, coordinator, _, stats = _fixture()
    desired = _packet(
        current_tx="R1", next_hop="N-A", mtype=MType.DATA_DOWN, ptx=20, command=True
    )
    setattr(desired, field, value)
    air.transmit(desired)
    env.run(until=10.2)

    assert node.default_ptx == 17 and coordinator.calls == 0
    assert any(
        row.get("reason") == "node_rx_window_closed_or_tuning_mismatch"
        for row in stats.records
    )


def test_gate_off_preserves_destination_filtered_legacy_behavior():
    env, air, node, coordinator, _, _ = _fixture(enabled=False)
    desired = _packet(
        current_tx="R1", next_hop="N-A", mtype=MType.DATA_DOWN, ptx=20, command=True
    )
    interferer = _packet(
        current_tx="N-B", next_hop="R2", mtype=MType.DATA_UP, ptx=17
    )
    _transmit_pair(air, desired, interferer)
    env.run(until=10.2)

    assert node.default_ptx == 5 and coordinator.calls == 1


def test_half_duplex_still_rejects_desired_downlink():
    env, air, node, coordinator, _, stats = _fixture()
    node.is_transmitting = True
    desired = _packet(
        current_tx="R1", next_hop="N-A", mtype=MType.DATA_DOWN, ptx=20, command=True
    )
    air.transmit(desired)
    env.run(until=10.2)

    assert node.default_ptx == 17 and coordinator.calls == 0
    assert any(row.get("reason") == "node_half_duplex_tx_busy" for row in stats.records)
