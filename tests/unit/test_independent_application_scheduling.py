from __future__ import annotations

from types import SimpleNamespace

import pytest
import simpy

from loraphymac.metrics.statistics import Statistics
from loraphymac.phy.air_interface import AirInterface
from loraphymac.runtime.node_tx import (
    INDEPENDENT_APPLICATION_SCHEDULING_MODE,
    LEGACY_APPLICATION_SCHEDULING_MODE,
    NodeTX,
    stable_application_rng_seed,
)
from loraphymac.runtime.wang_snr_reporting import decode_uplink_envelope


pytestmark = pytest.mark.unit


class _Node:
    def __init__(
        self,
        env,
        *,
        model_id="ff",
        fixed_tx_s=None,
        pre_tx_s=0.0,
        rx1_s=0.0,
    ):
        self.id = "N1"
        self.default_sf = 7
        self.default_bw_hz = 125_000
        self.default_ptx = 17
        self.ptx_mode = "fixed"
        self.fcnt = 0
        self.wang_snr_reporting_enabled = model_id == "wang_atp_sf7_a2only"
        self.wang_last_applied_command_seq = 0
        self.network_nodes_map = {"GW": SimpleNamespace(id="GW")}
        self.rx_window_enabled = rx1_s > 0
        self.rx1_delay_s = 0.0
        self.rx_window_duration_s = rx1_s
        self.stats = Statistics()
        self.stats.set_run_metadata(run_id=model_id, sim_time=10.0)
        self.captured = []
        self.fixed_tx_s = fixed_tx_s
        self.pre_tx_s = pre_tx_s

    def select_channel(self, _now):
        return 0, 921.4

    def sleep_event(self, env, duration_s):
        yield env.timeout(duration_s)

    def tx_event(self, env, packet, **_kwargs):
        if self.pre_tx_s > 0:
            yield env.timeout(self.pre_tx_s)
        packet.tx_start = float(env.now)
        duration = packet.toa if self.fixed_tx_s is None else self.fixed_tx_s
        packet.tx_end = packet.tx_start + duration
        self.captured.append(packet)
        self.stats.record_tx(packet, node_id=self.id)
        self.stats.record_node_tx(packet.src_id)
        yield env.timeout(duration)

    def open_rx_window(self, env, **_kwargs):
        yield env.timeout(self.rx_window_duration_s)


def _build_tx(
    *,
    model_id="ff",
    mode=INDEPENDENT_APPLICATION_SCHEDULING_MODE,
    period=1.0,
    jitter=0.0,
    fixed_tx_s=None,
    pre_tx_s=0.0,
    rx1_s=0.0,
):
    env = simpy.Environment()
    node = _Node(
        env,
        model_id=model_id,
        fixed_tx_s=fixed_tx_s,
        pre_tx_s=pre_tx_s,
        rx1_s=rx1_s,
    )
    tx = NodeTX(
        env=env,
        node=node,
        routing_map={"N1": "GW", "GW": None},
        air_interface=AirInterface(env),
        model=SimpleNamespace(),
        payload_size=50,
        period=period,
        initial_delay=0.0,
        jitter=jitter,
        application_scheduling_mode=mode,
        application_rng_seed=(
            stable_application_rng_seed(30, node.id)
            if mode == INDEPENDENT_APPLICATION_SCHEDULING_MODE
            else None
        ),
    )
    env.process(tx.run())
    return env, node, tx


def _offered_signature(node):
    fate = node.stats.get_packet_fate_df()
    return list(
        fate.loc[:, ["packet_uid", "fcnt", "birth_time", "offered_time"]]
        .itertuples(index=False, name=None)
    )


def test_four_models_share_exogenous_schedule_while_wang_keeps_real_toa():
    results = {}
    for model_id in ("ff", "aa", "islam_dbpc_itm6", "wang_atp_sf7_a2only"):
        env, node, _tx = _build_tx(model_id=model_id, period=0.05)
        env.run(until=0.5)
        results[model_id] = (_offered_signature(node), node.captured)

    signatures = [value[0] for value in results.values()]
    assert all(signature == signatures[0] for signature in signatures[1:])
    ff_packets = results["ff"][1]
    wang_packets = results["wang_atp_sf7_a2only"][1]
    assert wang_packets[0].payload_size > ff_packets[0].payload_size
    assert wang_packets[0].toa > ff_packets[0].toa
    assert wang_packets[1].tx_start > ff_packets[1].tx_start


def test_application_generation_continues_while_radio_busy_and_fifo_waits():
    env, node, tx = _build_tx(period=1.0, fixed_tx_s=3.0)
    env.run(until=3.1)
    record = tx.finalize_observation(3.1)
    node.stats.finalize_packet_observation(3.1)

    assert tx._application_generated_count == 4
    assert record["max_backlog_packets"] >= 2
    assert record["max_queue_waiting_time_s"] >= 2.0
    fate = node.stats.get_packet_fate_df()
    assert (fate["final_status"] == "CENSORED_WAITING").sum() == 2
    assert not fate.loc[
        fate["final_status"] == "CENSORED_WAITING", "terminal_reason"
    ].notna().any()
    summary = node.stats.get_packet_fate_summary_df().iloc[0]
    assert summary["packet_fate_conservation_delta"] == 0


def test_offer_is_registered_once_and_queue_state_is_exportable():
    env, node, tx = _build_tx(period=1.0, fixed_tx_s=0.2)
    env.run(until=2.1)
    tx.finalize_observation(2.1)

    assert len(node.stats.packet_fates) == tx._application_generated_count == 3
    queue = node.stats.get_application_queue_metrics_df()
    assert len(queue) == 1
    assert queue.iloc[0]["queue_policy"] == "unbounded_fifo_v1"


def test_horizon_distinguishes_queue_in_service_before_physical_tx():
    env, node, tx = _build_tx(period=1.0, pre_tx_s=0.5)
    env.run(until=0.1)
    record = tx.finalize_observation(0.1)
    node.stats.finalize_packet_observation(0.1)

    fate = node.stats.get_packet_fate_df().iloc[0]
    assert fate["final_status"] == "CENSORED_IN_SERVICE"
    assert record["waiting_at_horizon_count"] == 0
    assert record["in_service_at_horizon_count"] == 1


def test_queued_packet_uses_current_wang_ptx_and_epoch_at_physical_tx():
    env, node, _tx = _build_tx(
        model_id="wang_atp_sf7_a2only",
        period=0.02,
    )

    def update_power_before_second_tx():
        yield env.timeout(0.06)
        node.default_ptx = 11
        node.wang_last_applied_command_seq = 7

    env.process(update_power_before_second_tx())
    env.run(until=0.35)

    first, second = node.captured[:2]
    first_state, _ = decode_uplink_envelope(first)
    second_state, _ = decode_uplink_envelope(second)
    assert (first_state.actual_ptx_dbm, first_state.ptx_epoch) == (17, 0)
    assert (second_state.actual_ptx_dbm, second_state.ptx_epoch) == (11, 7)
    assert [first.fcnt, second.fcnt] == [0, 1]
    assert [first.packet_uid, second.packet_uid] == ["UP:N1:0", "UP:N1:1"]


def test_rx1_blocks_consumer_but_not_independent_producer():
    env, node, tx = _build_tx(period=0.05, rx1_s=0.2)
    env.run(until=0.5)

    assert tx._application_generated_count > len(node.captured)
    assert tx._application_max_backlog > 0


def test_legacy_mode_retains_post_tx_done_timing():
    env, node, _tx = _build_tx(
        mode=LEGACY_APPLICATION_SCHEDULING_MODE,
        period=1.0,
        fixed_tx_s=0.2,
    )
    env.run(until=2.5)

    assert [packet.tx_start for packet in node.captured] == pytest.approx(
        [0.0, 1.2, 2.4]
    )
    assert [packet.birth_time for packet in node.captured] == pytest.approx(
        [0.0, 1.2, 2.4]
    )


def test_stable_application_seed_is_repeatable_and_node_specific():
    assert stable_application_rng_seed(30, "N1") == stable_application_rng_seed(
        30, "N1"
    )
    assert stable_application_rng_seed(30, "N1") != stable_application_rng_seed(
        30, "N2"
    )
