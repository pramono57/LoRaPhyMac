from types import SimpleNamespace

import pytest
import simpy

from loraphymac.mac.frame_defs import CID, FPort, MType, encode_fctrl, frame_size_bytes
from loraphymac.metrics.instrumentation_metrics import (
    WANG_POWER_COMMAND_LIFECYCLE_COLUMNS,
)
from loraphymac.metrics.statistics import Statistics
from loraphymac.phy.air_interface import AirInterface
from loraphymac.runtime.node_rx import NodeRX
from loraphymac.runtime.packet import Packet
from loraphymac.runtime.wang_power_commands import (
    WANG_POWER_COMMAND_FOPTS_BYTES,
    WangGatewayCommandCoordinator,
    WangPowerCommand,
    WangPowerCommandError,
    apply_wang_power_command,
    decode_wang_power_command,
    encode_wang_power_command,
)
from loraphymac.runtime.wang_sink_decision import WangActiveTransmitter
from loraphymac.runtime.wang_snr_reporting import (
    WangGatewayReceivedReport,
    WangSNRReport,
)


class FakeStats:
    def __init__(self):
        self.events = []
        self.ptx_updates = []

    def record_wang_power_command_event(self, **event):
        self.events.append(event)

    def record_ptx_update(self, **event):
        self.ptx_updates.append(event)


class FakeChannelPlan:
    def get_freq(self, channel):
        return 921.4 + int(channel) * 0.2


class FakeRelayTX:
    def __init__(self):
        self.items = []
        self.relay_tx_ch_ids = [1]
        self.channel_plan = FakeChannelPlan()
        self.relay_sf = 7
        self.relay_bw_hz = 125000

    def put(self, packet):
        self.items.append(packet)
        return True


def entity(entity_id, ptx):
    return SimpleNamespace(
        id=entity_id,
        default_ptx=ptx,
        fixed_ptx=ptx,
        wang_last_applied_command_seq=0,
        rx1_delay_s=1.0,
        hw=SimpleNamespace(tx_current={value: 1.0 for value in range(5, 21)}),
    )


def command(*, target="N1", seq=1, expected=17, epoch=0, target_ptx=5):
    return WangPowerCommand(
        target_id=target,
        command_seq=seq,
        window_index=0,
        expected_ptx_dbm=expected,
        expected_epoch=epoch,
        target_ptx_dbm=target_ptx,
        issued_at_s=310.0,
        expiry_s=610.0,
    )


def packet_for(value, *, next_hop=None, path=None, hop_index=0):
    raw = encode_wang_power_command(value)
    route = list(path or ["GW", value.target_id])
    receiver = next_hop or route[hop_index + 1]
    return Packet(
        src_id="GW",
        current_tx=route[hop_index],
        prev_hop=route[hop_index],
        next_hop=receiver,
        path=route,
        hop_index=hop_index,
        ch_id=1,
        freq_mhz=921.6,
        sf=7,
        bw_hz=125000,
        ptx_dbm=20,
        payload_size=frame_size_bytes(0, len(raw)),
        toa=0.05,
        tx_start=310.0,
        tx_end=310.05,
        packet_uid=value.command_id,
        mtype=(MType.DATA_DOWN_FWD if receiver.startswith("R") else MType.DATA_DOWN),
        dst_id=value.target_id,
        hop_limit=len(route) - 1,
        fctrl=encode_fctrl(fopts_len=len(raw)),
        fopts=raw,
        fport=FPort.MAC_CONTROL,
    )


def coordinator(*, route, active):
    env = simpy.Environment(initial_time=310.0)
    gateway = entity("GW", 20)
    nodes = {item.entity_id: entity(item.entity_id, item.expected_ptx_dbm) for item in active}
    all_entities = {"GW": gateway, **nodes}
    for item in active:
        if item.entity_type == "relay":
            all_entities[item.entity_id] = nodes[item.entity_id]
    gateway.network_nodes_map = all_entities
    queues = {item: FakeRelayTX() for item in {"GW", *route.values()} if item}
    queues.setdefault("GW", FakeRelayTX())
    stats = FakeStats()
    result = WangGatewayCommandCoordinator(
        env=env,
        gateway=gateway,
        active_transmitters=active,
        routing_map=route,
        relay_tx=queues,
        air_interface=AirInterface(env),
        statistics=stats,
    )
    for item in all_entities.values():
        item.network_nodes_map = all_entities
    return env, result, all_entities, queues, stats


def decision(target_id, current, target):
    return SimpleNamespace(
        status="decision_generated",
        window_index=0,
        entity_decisions=(
            SimpleNamespace(
                entity_id=target_id,
                current_ptx_dbm=current,
                target_ptx_dbm=target,
            ),
        ),
    )


def test_command_wire_is_eleven_bytes_and_round_trips_from_physical_fields():
    value = command()
    raw = encode_wang_power_command(value)
    assert len(raw) == WANG_POWER_COMMAND_FOPTS_BYTES == 11
    pkt = packet_for(value)
    assert pkt.payload_size == 23
    assert decode_wang_power_command(pkt) == value


@pytest.mark.parametrize("seq", [0, 256])
def test_command_sequence_has_no_zero_or_wrap(seq):
    with pytest.raises(WangPowerCommandError):
        encode_wang_power_command(command(seq=seq))


def test_nstep_five_absolute_reduction_applies_once_and_never_twice():
    node = entity("N1", 17)
    pkt = packet_for(command(target_ptx=5))
    first = apply_wang_power_command(node, pkt, now_s=311.0)
    second = apply_wang_power_command(node, pkt, now_s=312.0)
    assert first.applied and first.old_ptx_dbm == 17 and first.new_ptx_dbm == 5
    assert node.default_ptx == 5 and node.wang_last_applied_command_seq == 1
    assert not second.applied and second.reason == "duplicate_command"
    assert node.default_ptx == 5


@pytest.mark.parametrize(
    ("value", "node_id", "now", "reason"),
    [
        (command(target="N2"), "N1", 311.0, "wrong_target"),
        (command(epoch=1), "N1", 311.0, "expected_epoch_mismatch"),
        (command(expected=16), "N1", 311.0, "expected_ptx_mismatch"),
        (command(target_ptx=20), "N1", 311.0, "ptx_increase_forbidden"),
        (command(), "N1", 611.0, "command_expired"),
    ],
)
def test_invalid_commands_do_not_mutate_power(value, node_id, now, reason):
    node = entity(node_id, 17)
    result = apply_wang_power_command(node, packet_for(value), now_s=now)
    assert not result.applied and result.reason == reason
    assert node.default_ptx == 17 and node.wang_last_applied_command_seq == 0


def test_direct_gateway_node_waits_for_real_target_uplink_rx1():
    active = (WangActiveTransmitter("N1", "node", 17),)
    env, runtime, entities, queues, _ = coordinator(
        route={"N1": "GW", "GW": None}, active=active
    )
    runtime.handle_window_decision(decision("N1", 17, 5))
    assert queues["GW"].items == []
    assert entities["N1"].default_ptx == 17
    uplink = SimpleNamespace(
        mtype=MType.DATA_UP,
        prev_hop="N1",
        packet_uid="N1-UP-1",
        tx_end=310.0,
        ch_id=0,
        freq_mhz=921.4,
        sf=7,
        bw_hz=125000,
    )
    runtime.observe_successful_uplink(entities["GW"], uplink)
    env.run(until=311.01)
    queued = queues["GW"].items[0]
    assert queued.path == ["GW", "N1"]
    assert queued.mtype == MType.DATA_DOWN and queued.sf == 7
    assert entities["N1"].default_ptx == 17


@pytest.mark.parametrize(
    ("route", "relay_path"),
    [
        ({"N1": "R1", "R1": "GW", "GW": None}, ["R1"]),
        (
            {"N1": "R1", "R1": "R2", "R2": "GW", "GW": None},
            ["R2", "R1"],
        ),
    ],
)
def test_one_and_two_relay_commands_follow_reverse_route_then_wait_rx1(
    route, relay_path
):
    active = (WangActiveTransmitter("N1", "node", 17),)
    env, runtime, entities, queues, _ = coordinator(route=route, active=active)
    for relay_id in relay_path:
        entities.setdefault(relay_id, entity(relay_id, 20))
        entities[relay_id].network_nodes_map = entities
    entities["GW"].network_nodes_map = entities
    runtime.handle_window_decision(decision("N1", 17, 5))
    pkt = queues["GW"].items.pop()
    assert pkt.path == ["GW", *relay_path, "N1"]
    for relay_id in relay_path:
        runtime.receive_at_relay(entities[relay_id], pkt)
        if relay_id != relay_path[-1]:
            pkt = queues[relay_id].items.pop()
    uplink = SimpleNamespace(
        mtype=MType.DATA_UP,
        prev_hop="N1",
        packet_uid="N1-UP-1",
        tx_end=310.0,
        ch_id=0,
        freq_mhz=921.4,
        sf=7,
        bw_hz=125000,
    )
    runtime.observe_successful_uplink(entities[relay_path[-1]], uplink)
    # Forwarding reuses and mutates the same Packet in production.  RX1 must
    # remain bound to the immutable first-hop context captured above.
    uplink.ch_id = 1
    uplink.freq_mhz = 921.6
    uplink.tx_end = 999.0
    env.run(until=311.01)
    final = queues[relay_path[-1]].items[0]
    assert final.next_hop == "N1" and final.mtype == MType.DATA_DOWN
    assert final.ch_id == 0 and final.freq_mhz == pytest.approx(921.4)
    assert entities["N1"].default_ptx == 17


def test_relay_target_applies_only_after_final_physical_receive():
    active = (WangActiveTransmitter("R1", "relay", 20),)
    route = {"R1": "R2", "R2": "GW", "GW": None}
    _, runtime, entities, queues, stats = coordinator(route=route, active=active)
    entities["R2"] = entity("R2", 20)
    entities["R1"].network_nodes_map = entities
    entities["R2"].network_nodes_map = entities
    runtime.handle_window_decision(decision("R1", 20, 14))
    assert entities["R1"].default_ptx == 20
    pkt = queues["GW"].items.pop()
    runtime.receive_at_relay(entities["R2"], pkt)
    pkt = queues["R2"].items.pop()
    runtime.receive_at_relay(entities["R1"], pkt)
    assert entities["R1"].default_ptx == 14
    assert entities["R1"].fixed_ptx == 14
    assert len(stats.ptx_updates) == 1


def test_lost_command_does_not_mutate_or_confirm_gateway_state():
    active = (WangActiveTransmitter("R1", "relay", 20),)
    _, runtime, entities, queues, _ = coordinator(
        route={"R1": "GW", "GW": None}, active=active
    )
    runtime.handle_window_decision(decision("R1", 20, 14))
    assert len(queues["GW"].items) == 1
    assert entities["R1"].default_ptx == 20
    assert runtime.states["R1"].last_confirmed_ptx_dbm == 20
    assert runtime.states["R1"].pending_command is not None


def received_report(*, seq, ptx, observation):
    report = WangSNRReport(
        packet_uid="P1",
        hop_index=0,
        tx_id="N1",
        rx_id="GW",
        snr_db=1.0,
        observed_ptx_dbm=ptx,
        observation_time_s=observation,
        last_applied_command_seq=seq,
    )
    return WangGatewayReceivedReport(report=report, gateway_arrival_time_s=observation + 1)


def test_only_post_command_transmitter_declaration_confirms_pending_state():
    active = (WangActiveTransmitter("N1", "node", 17),)
    _, runtime, _, _, stats = coordinator(
        route={"N1": "GW", "GW": None}, active=active
    )
    runtime.handle_window_decision(decision("N1", 17, 5))
    assert runtime.reconcile_report(received_report(seq=1, ptx=5, observation=309)) is None
    assert runtime.states["N1"].pending_command is not None
    confirmed = runtime.reconcile_report(
        received_report(seq=1, ptx=5, observation=311)
    )
    assert confirmed is not None
    assert runtime.states["N1"].last_confirmed_ptx_dbm == 5
    assert runtime.states["N1"].last_confirmed_epoch == 1
    assert any(
        row["event"] == "command_confirmed_by_uplink" for row in stats.events
    )


def test_pre_command_observation_arriving_after_expiry_cannot_resolve_pending():
    active = (WangActiveTransmitter("N1", "node", 17),)
    env, runtime, _, _, _ = coordinator(
        route={"N1": "GW", "GW": None}, active=active
    )
    runtime.handle_window_decision(decision("N1", 17, 5))
    env.run(until=610.01)
    received = received_report(seq=0, ptx=17, observation=309)
    received = WangGatewayReceivedReport(
        report=received.report,
        gateway_arrival_time_s=611.0,
    )
    assert runtime.reconcile_report(received) is None
    state = runtime.states["N1"]
    assert state.pending_command is not None
    assert state.pending_status == "expired_unconfirmed_application_unknown"


def test_pre_expiry_old_state_remains_unconfirmed_after_late_arrival():
    active = (WangActiveTransmitter("N1", "node", 17),)
    env, runtime, _, _, _ = coordinator(
        route={"N1": "GW", "GW": None}, active=active
    )
    runtime.handle_window_decision(decision("N1", 17, 5))
    env.run(until=610.01)
    received = received_report(seq=0, ptx=17, observation=609)
    received = WangGatewayReceivedReport(
        report=received.report,
        gateway_arrival_time_s=611.0,
    )
    assert runtime.reconcile_report(received) is None
    assert runtime.states["N1"].pending_command is not None


def test_post_expiry_old_state_proves_not_applied_and_allows_new_sequence():
    active = (WangActiveTransmitter("N1", "node", 17),)
    env, runtime, _, _, stats = coordinator(
        route={"N1": "GW", "GW": None}, active=active
    )
    runtime.handle_window_decision(decision("N1", 17, 11))
    env.run(until=610.01)
    assert runtime.reconcile_report(
        received_report(seq=0, ptx=17, observation=611)
    ) is None
    state = runtime.states["N1"]
    assert state.pending_command is None
    assert state.status == "not_applied_confirmed_by_uplink"
    assert state.last_confirmed_ptx_dbm == 17
    assert state.last_confirmed_epoch == 0
    assert any(row["event"] == "command_not_applied" for row in stats.events)

    runtime.handle_window_decision(decision("N1", 17, 11))
    assert state.pending_command is not None
    assert state.pending_command.command_seq == 2
    assert state.pending_command.expected_ptx_dbm == 17
    assert state.pending_command.expected_epoch == 0


def test_invalid_arrival_before_observation_cannot_resolve_pending():
    active = (WangActiveTransmitter("N1", "node", 17),)
    _, runtime, _, _, _ = coordinator(
        route={"N1": "GW", "GW": None}, active=active
    )
    runtime.handle_window_decision(decision("N1", 17, 5))
    received = received_report(seq=1, ptx=5, observation=311)
    received = WangGatewayReceivedReport(
        report=received.report,
        gateway_arrival_time_s=310.5,
    )
    assert runtime.reconcile_report(received) is None
    assert runtime.states["N1"].pending_command is not None


def test_each_node_uplink_supplies_fresh_rx1_channel_context():
    active = (WangActiveTransmitter("N1", "node", 17),)
    env, runtime, entities, queues, _ = coordinator(
        route={"N1": "GW", "GW": None}, active=active
    )
    runtime.handle_window_decision(decision("N1", 17, 11))
    first = SimpleNamespace(
        mtype=MType.DATA_UP,
        prev_hop="N1",
        packet_uid="N1-UP-1",
        tx_end=310.0,
        ch_id=0,
        freq_mhz=921.4,
        sf=7,
        bw_hz=125000,
    )
    runtime.observe_successful_uplink(entities["GW"], first)
    env.run(until=311.01)
    assert queues["GW"].items[-1].ch_id == 0
    runtime.reconcile_report(received_report(seq=1, ptx=11, observation=311))

    runtime.handle_window_decision(decision("N1", 11, 5))
    second = SimpleNamespace(
        mtype=MType.DATA_UP,
        prev_hop="N1",
        packet_uid="N1-UP-2",
        tx_end=312.0,
        ch_id=2,
        freq_mhz=921.8,
        sf=7,
        bw_hz=125000,
    )
    runtime.observe_successful_uplink(entities["GW"], second)
    env.run(until=313.01)
    assert queues["GW"].items[-1].ch_id == 2
    assert queues["GW"].items[-1].freq_mhz == pytest.approx(921.8)


def test_expired_rx1_context_cannot_release_stale_command():
    active = (WangActiveTransmitter("N1", "node", 17),)
    env, runtime, entities, queues, stats = coordinator(
        route={"N1": "GW", "GW": None}, active=active
    )
    runtime.handle_window_decision(decision("N1", 17, 5))
    stale = SimpleNamespace(
        mtype=MType.DATA_UP,
        prev_hop="N1",
        packet_uid="N1-UP-AFTER-EXPIRY",
        tx_end=611.0,
        ch_id=3,
        freq_mhz=922.0,
        sf=7,
        bw_hz=125000,
    )
    runtime.observe_successful_uplink(entities["GW"], stale)
    env.run(until=612.01)
    assert queues["GW"].items == []
    assert entities["N1"].default_ptx == 17
    assert any(
        row["event"] == "command_expired"
        and row["reason"] == "expired_before_target_rx1"
        for row in stats.events
    )


def test_per_target_sequence_exhaustion_is_explicit_and_does_not_wrap():
    active = (WangActiveTransmitter("N1", "node", 17),)
    _, runtime, _, _, stats = coordinator(
        route={"N1": "GW", "GW": None}, active=active
    )
    runtime.states["N1"].next_command_seq = 256
    runtime.handle_window_decision(decision("N1", 17, 5))
    assert runtime.states["N1"].pending_command is None
    assert runtime.states["N1"].status == "sequence_exhausted"
    assert any(
        row["event"] == "command_rejected"
        and "command_sequence_exhausted" in row["reason"]
        for row in stats.events
    )


class LinkBudgetAir:
    def compute_link_budget(self, **_kwargs):
        return {"rssi": -80.0, "snr": 5.0, "margin": 20.0, "feasible": True}


class PathlossModel:
    def pathloss(self, **_kwargs):
        return 100.0


def node_rx_fixture(*, window_open):
    env = simpy.Environment(initial_time=310.0)
    stats = Statistics()
    gateway = entity("GW", 20)
    node = entity("N1", 17)
    node.id = "N1"
    node.is_transmitting = False
    node.tx_busy_until = 0.0
    node.rx_window_enabled = True
    node.rx_windows = (
        [
            {
                "start": 310.0,
                "end": 311.0,
                "ch_id": 1,
                "bw_hz": 125000,
                "closed": False,
            }
        ]
        if window_open
        else []
    )
    node.gain = 2.15
    node.h = 2.0
    node.extend_rx_window_until = lambda **_kwargs: None
    node.close_rx_window = lambda **_kwargs: None
    gateway.gain = 5.8
    gateway.h = 25.0
    gateway.distance_to = lambda _target: 100.0
    active = (WangActiveTransmitter("N1", "node", 17),)
    queues = {"GW": FakeRelayTX()}
    runtime = WangGatewayCommandCoordinator(
        env=env,
        gateway=gateway,
        active_transmitters=active,
        routing_map={"N1": "GW", "GW": None},
        relay_tx=queues,
        air_interface=AirInterface(env),
        statistics=stats,
    )
    gateway.wang_command_coordinator = runtime
    gateway.wang_snr_reporting_enabled = True
    node.wang_snr_reporting_enabled = True
    node.network_nodes_map = {"N1": node, "GW": gateway}
    gateway.network_nodes_map = node.network_nodes_map
    rx = NodeRX(env, node, LinkBudgetAir(), PathlossModel(), 90, stats)
    value = command()
    pkt = packet_for(value)
    pkt.t_arrive = 310.0
    pkt.t_end = 310.05
    return env, node, rx, pkt, stats


def test_node_changes_power_only_after_successful_physical_rx1_decode():
    env, node, rx, pkt, _ = node_rx_fixture(window_open=True)
    rx.on_arrival(pkt)
    env.run(until=310.1)
    assert node.default_ptx == 5
    assert node.wang_last_applied_command_seq == 1


def test_closed_rx1_drops_physical_command_without_power_mutation():
    env, node, rx, pkt, stats = node_rx_fixture(window_open=False)
    rx.on_arrival(pkt)
    env.run(until=310.1)
    assert node.default_ptx == 17
    assert node.wang_last_applied_command_seq == 0
    assert any(row.get("reason") == "node_rx_window_closed" for row in stats.records)


def test_command_lifecycle_export_is_separate_and_schema_complete():
    stats = Statistics()
    stats.set_run_metadata(run_id="wang-command-test")
    value = command()
    pkt = packet_for(value)
    stats.record_wang_power_command_event(
        event="command_generated",
        time=310.0,
        packet=pkt,
        command=value,
    )
    stats.record_tx(pkt, "GW")
    result = stats.get_wang_power_command_lifecycle_df()
    frame = stats.to_dataframe()
    assert result.columns.tolist() == WANG_POWER_COMMAND_LIFECYCLE_COLUMNS
    assert result["event"].tolist() == ["command_generated", "command_sent"]
    assert result["fopts_len"].tolist() == [11, 11]
    assert result["control_airtime_s"].notna().all()
    assert frame.loc[0, "control_cid"] == int(CID.WANG_POWER_COMMAND)
    assert stats.get_frame_summary_df().loc[0, "apc_total_tx"] == 0
