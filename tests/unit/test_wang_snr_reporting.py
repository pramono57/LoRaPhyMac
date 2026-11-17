from __future__ import annotations

from types import SimpleNamespace

import pytest
import simpy

from loraphymac.mac.frame_defs import (
    MType,
    decode_fctrl,
    encode_fctrl,
    frame_size_bytes,
)
from loraphymac.metrics.statistics import Statistics
from loraphymac.phy.air_interface import AirInterface
from loraphymac.runtime.packet import Packet
from loraphymac.runtime.node_tx import NodeTX
from loraphymac.runtime.relay_rx import RelayRX
from loraphymac.runtime.relay_tx import RelayTX
from loraphymac.runtime.wang_snr_reporting import (
    WANG_NO_COMMAND_APPLIED_SEQ,
    WANG_SNR_REPORT_SIZE_BYTES,
    WANG_TX_STATE_HEADER_SIZE_BYTES,
    WangSNRReportCapacityError,
    WangSNRReportError,
    WangTransmitterState,
    append_report,
    declare_transmitter_state,
    decode_reports,
    decode_uplink_envelope,
    encode_report,
    encode_transmitter_state,
    freeze_wang_routing_map,
    reconstruct_physical_uplink_identity,
    report_from_successful_rx,
)


pytestmark = pytest.mark.unit


def packet(
    path,
    *,
    hop_index=0,
    snr_db=6.25,
    ptx_dbm=17,
    declared_ptx_dbm=None,
    command_seq=0,
    wang=True,
):
    current_tx = path[hop_index]
    fopts = b""
    if wang:
        fopts = encode_transmitter_state(
            WangTransmitterState(
                actual_ptx_dbm=(
                    ptx_dbm if declared_ptx_dbm is None else declared_ptx_dbm
                ),
                last_applied_command_seq=command_seq,
            )
        )
    pkt = Packet(
        src_id=path[0],
        current_tx=current_tx,
        next_hop=path[hop_index + 1],
        path=list(path),
        hop_index=hop_index,
        ch_id=0,
        freq_mhz=921.4,
        sf=7,
        bw_hz=125_000,
        ptx_dbm=ptx_dbm,
        payload_size=frame_size_bytes(50, len(fopts)),
        app_payload_size=50,
        toa=0.1,
        tx_start=0.0,
        tx_end=0.1,
        packet_uid=f"UP:{path[0]}:1",
        mtype=MType.DATA_UP if hop_index == 0 else MType.DATA_UP_FWD,
        dst_id=path[-1],
        hop_limit=len(path) - 1,
        fcnt=1,
        fopts=fopts,
        fctrl=encode_fctrl(fopts_len=len(fopts)),
    )
    pkt.decoded = True
    pkt.feasible = True
    pkt.collided = False
    pkt.is_intended_receiver = True
    pkt.snr = snr_db
    pkt.test_routing_map = {
        path[index]: path[index + 1] for index in range(len(path) - 1)
    }
    pkt.test_routing_map[path[-1]] = None
    return pkt


def receiver(
    entity_id,
    stats,
    *,
    now=10.0,
    enabled=True,
    routing_map=None,
):
    rx = object.__new__(RelayRX)
    rx.env = SimpleNamespace(now=now)
    rx.relay = SimpleNamespace(
        id=entity_id,
        wang_snr_reporting_enabled=enabled,
        wang_routing_map=routing_map,
    )
    rx.stats = stats
    return rx


def receive(rx, pkt, *, snr_db):
    pkt.decoded = True
    pkt.collided = False
    pkt.feasible = True
    pkt.is_intended_receiver = True
    pkt.snr = snr_db
    rx.relay.wang_routing_map = pkt.test_routing_map
    rx._handle_wang_snr_success(pkt)


def make_report(pkt, receiver_id, observation_time_s):
    return report_from_successful_rx(
        pkt,
        receiver_id,
        observation_time_s,
        routing_map=pkt.test_routing_map,
    )


def add_report(pkt, report):
    append_report(pkt, report, routing_map=pkt.test_routing_map)


def decode_envelope(pkt, *, receiver_id=None):
    return decode_uplink_envelope(
        pkt,
        routing_map=pkt.test_routing_map,
        receiver_id=receiver_id,
    )


def carried_reports(pkt, *, receiver_id=None):
    return decode_reports(
        pkt,
        routing_map=pkt.test_routing_map,
        receiver_id=receiver_id,
    )


def advance(pkt, *, ptx_dbm=20, command_seq=0):
    pkt.advance_hop()
    pkt.mtype = MType.DATA_UP_FWD
    declare_transmitter_state(
        pkt,
        actual_ptx_dbm=ptx_dbm,
        last_applied_command_seq=command_seq,
    )


def test_physical_envelope_round_trip_reconstructs_link_and_provenance():
    pkt = packet(
        ["N1", "R1", "GW"],
        snr_db=-4.501,
        declared_ptx_dbm=17,
        command_seq=4,
    )
    pkt.ptx_dbm = 5  # Simulator metadata must not be used as Wang evidence.
    report = make_report(pkt, "R1", 12.39)

    encoded = encode_report(report)
    add_report(pkt, report)
    advance(pkt)
    state, decoded_reports = decode_envelope(pkt, receiver_id="GW")
    decoded = decoded_reports[0]

    assert len(encoded) == WANG_SNR_REPORT_SIZE_BYTES == 6
    assert WANG_TX_STATE_HEADER_SIZE_BYTES == 3
    assert (state.actual_ptx_dbm, state.ptx_epoch) == (20, 0)
    assert (decoded.tx_id, decoded.rx_id) == ("N1", "R1")
    assert decoded.packet_uid == "UP:N1:1"
    assert decoded.snr_db == pytest.approx(-4.51)
    assert decoded.observation_time_s == pytest.approx(12.3)
    assert decoded.observed_ptx_dbm == 17
    assert decoded.ptx_epoch == 4
    assert decode_fctrl(pkt.fctrl)["fopts_len"] == 9


def test_initial_transmitter_state_uses_explicit_no_command_sentinel():
    pkt = packet(["N1", "GW"], declared_ptx_dbm=17)

    state, reports = decode_envelope(pkt, receiver_id="GW")

    assert state.actual_ptx_dbm == 17
    assert state.ptx_epoch == WANG_NO_COMMAND_APPLIED_SEQ == 0
    assert state.last_applied_command_seq == 0
    assert reports == []
    assert len(pkt.fopts) == 3


def test_node_tx_physically_encodes_its_own_initial_state():
    env = simpy.Environment()
    captured = []

    def tx_event(**kwargs):
        captured.append(kwargs["packet"])
        yield env.timeout(1.0)

    node = SimpleNamespace(
        id="N1",
        default_sf=7,
        default_bw_hz=125_000,
        default_ptx=17,
        ptx_mode="fixed",
        fcnt=0,
        wang_snr_reporting_enabled=True,
        wang_last_applied_command_seq=0,
        network_nodes_map={"GW": SimpleNamespace(id="GW")},
        select_channel=lambda _now: (0, 921.4),
        tx_event=tx_event,
    )
    tx = NodeTX(
        env=env,
        node=node,
        routing_map={"N1": "GW", "GW": None},
        air_interface=AirInterface(env),
        model=SimpleNamespace(),
        payload_size=50,
        period=60,
        initial_delay=0,
        jitter=0,
    )

    env.process(tx.run())
    env.run(until=0.01)

    state, reports = decode_uplink_envelope(captured[0])
    assert (state.actual_ptx_dbm, state.ptx_epoch) == (17, 0)
    assert reports == []
    assert captured[0].payload_size == frame_size_bytes(50, 3)
    assert node.default_ptx == 17


def test_relay_forward_replaces_only_current_transmitter_state():
    pkt = packet(["N1", "R1", "GW"])
    add_report(pkt, make_report(pkt, "R1", 1.0))
    pkt.advance_hop()

    tx = object.__new__(RelayTX)
    tx.relay = SimpleNamespace(
        id="R1",
        ptx_mode="fixed",
        fixed_ptx=20,
        wang_snr_reporting_enabled=True,
        wang_last_applied_command_seq=9,
    )
    tx.frame_enabled = True
    tx._apply_common_relay_phy = lambda packet, selected_ptx: packet

    prepared = tx._prepare_uplink_forward_packet(pkt)

    state, reports = decode_envelope(prepared, receiver_id="GW")
    assert (state.actual_ptx_dbm, state.ptx_epoch) == (20, 9)
    assert [(item.tx_id, item.rx_id, item.ptx_epoch) for item in reports] == [
        ("N1", "R1", 0)
    ]


def test_quantization_is_conservative_around_future_three_db_threshold():
    above = packet(["N1", "R1", "GW"], snr_db=-4.499)
    below = packet(["N1", "R1", "GW"], snr_db=-4.501)
    add_report(above, make_report(above, "R1", 1.09))
    add_report(below, make_report(below, "R1", 1.09))
    advance(above)
    advance(below)

    assert carried_reports(above)[0].snr_db == pytest.approx(-4.50)
    assert carried_reports(below)[0].snr_db == pytest.approx(-4.51)
    assert carried_reports(above)[0].snr_db <= -4.499
    assert carried_reports(below)[0].snr_db <= -4.501


@pytest.mark.parametrize(
    ("decoded", "collided", "feasible"),
    [(False, False, True), (True, True, True), (True, False, False)],
)
def test_failed_or_collided_reception_generates_no_report(decoded, collided, feasible):
    stats = Statistics()
    pkt = packet(["N1", "R1", "GW"])
    pkt.decoded = decoded
    pkt.collided = collided
    pkt.feasible = feasible

    receiver(
        "R1",
        stats,
        routing_map=pkt.test_routing_map,
    )._handle_wang_snr_success(pkt)

    assert len(pkt.fopts) == WANG_TX_STATE_HEADER_SIZE_BYTES
    assert not any(
        row["event"] == "generated" for row in stats.wang_snr_report_lifecycle
    )


def test_direct_route_gateway_uses_only_its_successful_local_observation():
    stats = Statistics()
    pkt = packet(["N1", "GW"])
    gw = receiver("GW", stats)

    receive(gw, pkt, snr_db=4.25)

    assert len(gw.relay.wang_received_snr_reports) == 1
    assert len(gw.relay.wang_received_snr_observations) == 1
    assert (
        gw.relay.wang_received_snr_observations[0].gateway_arrival_time_s
        == pytest.approx(10.0)
    )
    report = gw.relay.wang_received_snr_reports[0]
    assert (report.tx_id, report.rx_id, report.snr_db) == ("N1", "GW", 4.25)
    assert len(pkt.fopts) == 3
    received_row = next(
        row
        for row in stats.wang_snr_report_lifecycle
        if row["event"] == "received"
    )
    assert received_row["ptx_epoch"] == 0
    assert received_row["last_applied_command_seq"] == 0
    assert received_row["ptx_provenance"] == "transmitter_declared_frame"


def test_one_relay_route_delivers_intermediate_plus_gateway_observation():
    stats = Statistics()
    pkt = packet(["N1", "R1", "GW"])
    receive(receiver("R1", stats, now=10.0), pkt, snr_db=3.0)
    assert len(pkt.fopts) == 9
    advance(pkt, command_seq=7)
    gw = receiver("GW", stats, now=20.0)

    receive(gw, pkt, snr_db=5.0)

    assert [(r.tx_id, r.rx_id) for r in gw.relay.wang_received_snr_reports] == [
        ("N1", "R1"),
        ("R1", "GW"),
    ]
    assert [r.observed_ptx_dbm for r in gw.relay.wang_received_snr_reports] == [
        17,
        20,
    ]
    assert [r.ptx_epoch for r in gw.relay.wang_received_snr_reports] == [0, 7]


def test_two_relay_route_fits_two_wire_reports_and_delivers_three_observations():
    stats = Statistics()
    pkt = packet(["N1", "R2", "R1", "GW"])
    receive(receiver("R2", stats, now=10.0), pkt, snr_db=1.0)
    advance(pkt)
    receive(receiver("R1", stats, now=20.0), pkt, snr_db=2.0)
    assert len(pkt.fopts) == 15
    advance(pkt)
    gw = receiver("GW", stats, now=30.0)

    receive(gw, pkt, snr_db=3.0)

    assert [(r.tx_id, r.rx_id) for r in gw.relay.wang_received_snr_reports] == [
        ("N1", "R2"),
        ("R2", "R1"),
        ("R1", "GW"),
    ]


def test_physical_identity_rejects_inconsistent_or_ambiguous_routes():
    pkt = packet(["N1", "R1", "GW"])
    pkt.next_hop = "GW"
    with pytest.raises(WangSNRReportError, match="NextHop"):
        reconstruct_physical_uplink_identity(
            pkt,
            routing_map=pkt.test_routing_map,
            receiver_id="R1",
        )

    with pytest.raises(WangSNRReportError, match="not provisioned"):
        freeze_wang_routing_map({"N1": "R1", "R1": "GW"})

    with pytest.raises(WangSNRReportError, match="ambiguous"):
        freeze_wang_routing_map(
            {1: "GW", "1": "R1", "R1": None, "GW": None}
        )


def test_duplicate_gateway_delivery_is_suppressed_by_packet_hop_receiver_identity():
    stats = Statistics()
    pkt = packet(["N1", "GW"])
    gw = receiver("GW", stats)

    receive(gw, pkt, snr_db=2.0)
    receive(gw, pkt, snr_db=2.0)

    assert len(gw.relay.wang_received_snr_reports) == 1
    events = [row["event"] for row in stats.wang_snr_report_lifecycle]
    assert events.count("received") == 1
    assert events.count("duplicate") == 1


def test_fopts_capacity_fails_closed_and_unknown_existing_option_is_invalid():
    pkt = packet(["N1", "R1", "R2", "R3", "GW"])
    for receiver_id in ("R1", "R2"):
        add_report(pkt, make_report(pkt, receiver_id, 1.0))
        advance(pkt)
    third = make_report(pkt, "R3", 2.0)
    before = (pkt.fopts, pkt.payload_size, pkt.fctrl)
    with pytest.raises(WangSNRReportCapacityError):
        add_report(pkt, third)
    assert (pkt.fopts, pkt.payload_size, pkt.fctrl) == before

    bad = packet(["N2", "R1", "GW"])
    bad.fopts = b"\x01\x00\x00"
    with pytest.raises(WangSNRReportError, match="unexpected"):
        add_report(bad, make_report(bad, "R1", 1.0))


def test_added_physical_bytes_increase_toa_and_relay_tx_energy():
    air = AirInterface(SimpleNamespace(now=0.0))
    pkt = packet(["N1", "R1", "GW"])
    old_size = pkt.payload_size
    old_toa = air.compute_toa(pkt.sf, pkt.bw_hz, old_size)
    add_report(pkt, make_report(pkt, "R1", 1.0))
    new_toa = air.compute_toa(pkt.sf, pkt.bw_hz, pkt.payload_size)

    assert old_size == frame_size_bytes(50, 0) + 3
    assert pkt.app_payload_size == 50
    assert pkt.payload_size == old_size + 6
    assert new_toa > old_toa

    tx = object.__new__(RelayTX)
    tx.env = SimpleNamespace(now=0.0)
    tx.relay = SimpleNamespace(
        id="R1",
        hw=SimpleNamespace(tx_current={20: 120.0}, voltage=5.0),
        energy_tx_mJ=0.0,
        include_mcu_energy=False,
    )
    pkt.ptx_dbm = 20
    pkt.toa = new_toa
    tx._record_energy(pkt)
    assert tx.relay.energy_tx_mJ == pytest.approx(5.0 * 0.120 * new_toa * 1000.0)
    assert tx.relay.energy_tx_mJ > 5.0 * 0.120 * old_toa * 1000.0


def test_physical_forward_and_loss_lifecycle_never_fabricates_gateway_receive():
    stats = Statistics()
    pkt = packet(["N1", "R1", "GW"])
    receive(receiver("R1", stats), pkt, snr_db=3.0)
    advance(pkt)
    pkt.tx_start = 12.0
    stats.record_tx(pkt, node_id="R1")
    pkt.drop_time = 12.1
    stats.record_drop(pkt, reason="collision", node_id="GW")

    events = [row["event"] for row in stats.wang_snr_report_lifecycle]
    assert events == ["generated", "forwarded", "lost"]
    assert "received" not in events


def test_wang_disabled_baseline_does_not_change_frame_or_create_lifecycle():
    stats = Statistics()
    pkt = packet(["N1", "R1", "GW"], wang=False)
    before = (pkt.fopts, pkt.fctrl, pkt.payload_size, pkt.ptx_dbm)

    receiver("R1", stats, enabled=False)._handle_wang_snr_success(pkt)

    assert (pkt.fopts, pkt.fctrl, pkt.payload_size, pkt.ptx_dbm) == before
    assert stats.wang_snr_report_lifecycle == []
