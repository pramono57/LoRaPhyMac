from types import SimpleNamespace

import pytest
import simpy

from loraphymac.mac.frame_defs import FPort, MType, encode_fctrl
from loraphymac.metrics.statistics import Statistics
from loraphymac.runtime.apc_commands import encode_apc_req
from loraphymac.runtime.apc_controller_state import APCControllerLinkState
from loraphymac.runtime.node_rx import NodeRX
from loraphymac.runtime.packet import Packet
from loraphymac.runtime.relay_rx import RelayRX


pytestmark = pytest.mark.unit


class FakeStats:
    def __init__(self):
        self.events = []
        self.sequence = 0

    def allocate_apc_feedback_id(self):
        self.sequence += 1
        return f"test:apc:{self.sequence:06d}"

    def record_apc_feedback_event(self, **kwargs):
        self.events.append(dict(kwargs))


class FixedAir:
    @staticmethod
    def compute_toa(**_kwargs):
        return 0.02

    @staticmethod
    def compute_link_budget(**_kwargs):
        return {"feasible": True, "rssi": -90.0, "snr": 12.0, "margin": 18.0}


class FixedPathloss:
    @staticmethod
    def pathloss(**_kwargs):
        return 100.0


def adaptive_entity(entity_id="N1"):
    return SimpleNamespace(
        id=entity_id,
        ptx_mode="adaptive",
        fixed_ptx=17,
        default_ptx=17,
        apc_tx_power_index=0,
        ptx_min=5,
        ptx_max=17,
        target_margin_db=6.0,
        hw=SimpleNamespace(tx_current={value: 0.01 for value in range(5, 18, 2)}),
    )


def uplink_packet(*, margin=10.0, apc=True):
    packet = SimpleNamespace(
        mtype=MType.DATA_UP,
        fctrl=encode_fctrl(apc=apc),
        margin=float(margin),
        rssi=-90.0,
        snr=10.0,
        ch_id=1,
        freq_mhz=921.4,
        sf=7,
        bw_hz=125_000,
        src_id="N1",
        prev_hop="N1",
        next_hop="R1",
        hop_index=0,
    )
    return packet


def controller_fixture():
    env = simpy.Environment()
    stats = FakeStats()
    relay = SimpleNamespace(
        id="R1",
        apc_downlink_mode="physical_all",
        apc_retry_timeout_s=5.0,
        apc_max_retries=2,
    )
    receiver = object.__new__(RelayRX)
    receiver.env = env
    receiver.relay = relay
    receiver.stats = stats
    receiver.apc_controller_link_states = {}
    sent = []

    def fake_send(**kwargs):
        sent.append(kwargs)
        yield env.timeout(0)

    receiver._send_physical_apc_downlink = fake_send
    return env, receiver, stats, sent


def test_controller_state_timeout_and_bounded_retry_never_confirm_delivery():
    state = APCControllerLinkState.from_preprovisioned_state(
        controller_id="R1",
        transmitter_id="N1",
        initial_tx_power_index=0,
        max_tx_power_index=6,
        target_margin_db=6.0,
    )
    state.begin_unconfirmed_command(
        feedback_id="F1", target_tx_power_index=2, now_s=0.0
    )

    assert state.confirmed_tx_power_index == 0
    assert state.estimated_tx_power_index == 2
    assert state.estimate_provenance == "unconfirmed_command_assumption"
    assert state.retry_status(now_s=4.9, retry_timeout_s=5, max_retries=2) == "waiting"
    assert state.retry_status(now_s=5, retry_timeout_s=5, max_retries=2) == "retry"
    state.mark_retry(now_s=5)
    assert state.retry_status(now_s=10, retry_timeout_s=5, max_retries=2) == "retry"
    state.mark_retry(now_s=10)
    assert state.retry_status(now_s=15, retry_timeout_s=5, max_retries=2) == "exhausted"
    state.mark_retry_exhausted()

    assert state.pending_command is None
    assert state.confirmed_tx_power_index == 0
    assert state.estimated_tx_power_index == 2
    assert state.estimate_provenance == "unconfirmed_retry_exhausted"


def test_controller_decision_is_invariant_to_private_remote_runtime_state():
    outputs = []
    for actual_index, actual_ptx in ((0, 17), (6, 5)):
        env, receiver, _, sent = controller_fixture()
        transmitter = adaptive_entity()
        transmitter.apc_tx_power_index = actual_index
        transmitter.default_ptx = actual_ptx

        receiver._apply_adaptive_ptx_feedback(
            transmitter, uplink_packet(margin=8.0)
        )
        env.run()
        state = receiver.apc_controller_link_states[("N1", "R1")]
        outputs.append(
            (
                state.pending_command.target_tx_power_index,
                state.estimated_tx_power_index,
                sent[0]["apc_info"]["context"]["tx_power_index"],
            )
        )

    assert outputs == [(1, 1, 1), (1, 1, 1)]


def test_delivered_or_lost_outcome_cannot_clear_controller_pending_state():
    state = APCControllerLinkState.from_preprovisioned_state(
        controller_id="R1",
        transmitter_id="N1",
        initial_tx_power_index=0,
        max_tx_power_index=6,
        target_margin_db=6,
    )
    state.begin_unconfirmed_command(
        feedback_id="F1", target_tx_power_index=2, now_s=0
    )
    transmitter = adaptive_entity()

    # Simulate either local target application or loss by changing only private
    # transmitter state. Neither outcome is physical evidence at the controller.
    transmitter.apc_tx_power_index = 2
    transmitter.default_ptx = 13
    assert state.pending_command is not None
    assert state.confirmed_tx_power_index == 0
    transmitter.apc_tx_power_index = 0
    transmitter.default_ptx = 17
    assert state.pending_command is not None
    assert state.confirmed_tx_power_index == 0


def test_controller_retries_same_absolute_target_then_records_exhaustion():
    env, receiver, stats, sent = controller_fixture()
    transmitter = adaptive_entity()
    packet = uplink_packet(margin=10.0)

    receiver._apply_adaptive_ptx_feedback(transmitter, packet)
    env.run(until=0.1)
    initial_target = sent[-1]["apc_info"]["context"]["tx_power_index"]
    feedback_id = sent[-1]["apc_info"]["feedback_id"]

    for deadline in (5.1, 10.1):
        env.run(until=deadline)
        receiver._apply_adaptive_ptx_feedback(transmitter, packet)
        env.run(until=deadline + 0.01)
        assert sent[-1]["apc_info"]["context"]["tx_power_index"] == initial_target
        assert sent[-1]["apc_info"]["feedback_id"] == feedback_id

    env.run(until=15.2)
    receiver._apply_adaptive_ptx_feedback(transmitter, packet)

    state = receiver.apc_controller_link_states[("N1", "R1")]
    assert len(sent) == 3
    assert state.pending_command is None
    assert state.estimate_provenance == "unconfirmed_retry_exhausted"
    assert stats.events[-1]["event"] == "retry_exhausted"


def test_shadow_log_cannot_update_controller_belief():
    state = APCControllerLinkState.from_preprovisioned_state(
        controller_id="R1",
        transmitter_id="N1",
        initial_tx_power_index=0,
        max_tx_power_index=6,
        target_margin_db=6,
    )
    state.begin_unconfirmed_command(
        feedback_id="F1", target_tx_power_index=2, now_s=0
    )
    before = (
        state.confirmed_tx_power_index,
        state.estimated_tx_power_index,
        state.pending_command,
    )
    stats = Statistics()
    stats.record_apc_shadow(
        time=1.0,
        event="APC_ANS_SHADOW",
        sender_id="N1",
        sender_type="Node",
        receiver_id="R1",
        receiver_type="Relay",
        controller_id="R1",
        controller_type="Relay",
        target_id="N1",
        target_type="Node",
        control_direction="uplink",
        control_scope="shadow_only",
        old_ptx=13,
        new_ptx=13,
        status=0x1F,
    )
    after = (
        state.confirmed_tx_power_index,
        state.estimated_tx_power_index,
        state.pending_command,
    )
    assert after == before


def test_fixed_and_wang_gates_do_not_create_rapc_controller_state():
    env, receiver, _, sent = controller_fixture()
    fixed = adaptive_entity()
    fixed.ptx_mode = "fixed"
    receiver._apply_adaptive_ptx_feedback(fixed, uplink_packet())

    wang = adaptive_entity("N2")
    receiver._apply_adaptive_ptx_feedback(wang, uplink_packet(apc=False))
    env.run()

    assert receiver.apc_controller_link_states == {}
    assert sent == []


def rapc_downlink(*, target_index=2, now=0.0):
    fopts = encode_apc_req(dr=0, tx_power_index=target_index)
    return Packet(
        src_id="GW",
        current_tx="GW",
        prev_hop="GW",
        next_hop="N1",
        path=["GW", "N1"],
        hop_index=0,
        ch_id=1,
        freq_mhz=921.4,
        sf=7,
        bw_hz=125_000,
        ptx_dbm=20,
        payload_size=16,
        toa=0.02,
        tx_start=now,
        tx_end=now + 0.02,
        packet_uid=f"APC:{now}",
        app_payload_size=0,
        mtype=MType.DATA_DOWN,
        dst_id="N1",
        hop_limit=1,
        fctrl=encode_fctrl(apc=True, fopts_len=len(fopts)),
        fcnt=1,
        fopts=fopts,
        fport=FPort.MAC_CONTROL,
        birth_time=now,
    )


def node_rx_fixture(*, window_open=True):
    env = simpy.Environment()
    stats = Statistics()
    gateway = SimpleNamespace(id="GW", gain=0.0, h=20.0)
    node = adaptive_entity()
    node.gain = 0.0
    node.h = 2.0
    node.rx_window_enabled = True
    node.rx_windows = (
        [{"start": 0.0, "end": 10.0, "ch_id": 1, "bw_hz": 125_000}]
        if window_open
        else []
    )
    node.is_transmitting = False
    node.tx_busy_until = 0.0
    node.extend_rx_window_until = lambda **_kwargs: None
    node.close_rx_window = lambda **_kwargs: None
    node.apc_pending_tx_index = 4
    node.apc_pending_ack = True
    node.apc_pending_from = "R1"
    node.apc_pending_time = -1.0
    gateway.distance_to = lambda _target: 100.0
    node.network_nodes_map = {"N1": node, "GW": gateway}
    receiver = NodeRX(env, node, FixedAir(), FixedPathloss(), 90, stats)
    return env, node, receiver, stats


def test_physical_apc_req_applies_locally_without_clearing_controller_fields():
    env, node, receiver, _ = node_rx_fixture(window_open=True)
    packet = rapc_downlink(target_index=2)
    packet.t_arrive = 0.0
    packet.t_end = 0.02
    receiver.on_arrival(packet)
    env.run(until=0.1)

    assert node.apc_tx_power_index == 2
    assert node.default_ptx == 13
    assert node.apc_pending_tx_index == 4
    assert node.apc_pending_ack is True


def test_lost_apc_req_does_not_mutate_local_power_or_pending_fields():
    env, node, receiver, stats = node_rx_fixture(window_open=False)
    packet = rapc_downlink(target_index=2)
    receiver.on_arrival(packet)
    env.run(until=0.1)

    assert node.apc_tx_power_index == 0
    assert node.default_ptx == 17
    assert node.apc_pending_tx_index == 4
    assert node.apc_pending_ack is True
    assert any(row.get("reason") == "node_rx_window_closed" for row in stats.records)


def test_delayed_apc_req_application_does_not_confirm_controller_state():
    state = APCControllerLinkState.from_preprovisioned_state(
        controller_id="R1",
        transmitter_id="N1",
        initial_tx_power_index=0,
        max_tx_power_index=6,
        target_margin_db=6,
    )
    state.begin_unconfirmed_command(
        feedback_id="F1", target_tx_power_index=2, now_s=0
    )
    state.mark_retry(now_s=5)

    env, node, receiver, _ = node_rx_fixture(window_open=True)
    env.run(until=6.0)
    delayed = rapc_downlink(target_index=2, now=6.0)
    delayed.t_arrive = 6.0
    delayed.t_end = 6.02
    receiver.on_arrival(delayed)
    env.run(until=6.1)

    assert node.apc_tx_power_index == 2
    assert state.pending_command is not None
    assert state.pending_command.retry_count == 1
    assert state.confirmed_tx_power_index == 0
    assert state.estimate_provenance == "unconfirmed_command_assumption"


def test_duplicate_absolute_tpi_application_is_idempotent():
    env, node, receiver, _ = node_rx_fixture(window_open=True)
    first = rapc_downlink(target_index=2, now=0.0)
    first.t_arrive = 0.0
    first.t_end = 0.02
    receiver.on_arrival(first)
    env.run(until=0.1)

    second = rapc_downlink(target_index=2, now=0.1)
    second.t_arrive = 0.1
    second.t_end = 0.12
    receiver.on_arrival(second)
    env.run(until=0.2)

    assert node.apc_tx_power_index == 2
    assert node.default_ptx == 13


def test_apc_req_remains_four_fopts_bytes_and_sixteen_physical_bytes():
    packet = rapc_downlink(target_index=3)
    assert len(packet.fopts) == 4
    assert packet.payload_size == 16
