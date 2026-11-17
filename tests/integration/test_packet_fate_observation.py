from loraphymac.mac.frame_defs import MType
from loraphymac.metrics.statistics import Statistics
from loraphymac.runtime.packet import Packet


def make_uplink(uid, fcnt):
    return Packet(
        src_id="N1",
        current_tx="N1",
        next_hop="R1",
        path=["N1", "R1", "GW"],
        hop_index=0,
        ch_id=0,
        freq_mhz=921.4,
        sf=7,
        bw_hz=125_000,
        ptx_dbm=17,
        payload_size=62,
        toa=0.1,
        tx_start=float(fcnt),
        tx_end=float(fcnt) + 0.1,
        packet_uid=uid,
        mtype=MType.DATA_UP,
        fcnt=fcnt,
        birth_time=float(fcnt),
    )


def test_packet_uid_survives_air_copy_and_forwarding_metadata_cleanup():
    packet = make_uplink("UP:N1:1", 1)
    copied = packet.copy()
    copied.advance_hop()
    copied.reset_rx_runtime_metadata(preserve_t_arrive=False)

    assert copied.packet_uid == "UP:N1:1"


def test_finite_horizon_fate_conservation_separates_loss_and_censoring():
    stats = Statistics()
    stats.set_run_metadata(run_id="fate-test", sim_time=20.0)
    delivered = make_uplink("UP:N1:1", 1)
    dropped = make_uplink("UP:N1:2", 2)
    waiting = make_uplink("UP:N1:3", 3)
    receiving = make_uplink("UP:N1:4", 4)

    for packet in (delivered, dropped, waiting, receiving):
        stats.record_tx(packet, node_id="N1")

    stats.record_packet_fate_state(
        delivered, "DELIVERED", entity_id="GW", time=5.0
    )
    dropped.drop_time = 6.0
    stats.record_drop(dropped, reason="collision", node_id="R1")
    stats.record_packet_fate_state(
        waiting, "QUEUE_WAITING", entity_id="R1", time=7.0
    )
    stats.record_packet_fate_state(
        receiving, "RX_ACTIVE", entity_id="R2", time=8.0
    )
    stats.finalize_packet_observation(20.0)

    summary = stats.get_packet_fate_summary_df().iloc[0]
    assert summary["offered_count"] == 4
    assert summary["delivered_count"] == 1
    assert summary["terminal_drop_count"] == 1
    assert summary["censored_waiting_count"] == 1
    assert summary["censored_rx_count"] == 1
    assert summary["packet_fate_conservation_delta"] == 0
    assert summary["terminal_conflict_count"] == 0

