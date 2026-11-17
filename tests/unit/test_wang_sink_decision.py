from __future__ import annotations

from types import SimpleNamespace

import pytest
import simpy

from loraphymac.config.defaults import DEFAULT_CONFIG, build_config
from loraphymac.config.validation import ConfigurationError, validate_configuration
from loraphymac.mac.frame_defs import MType, encode_fctrl, frame_size_bytes
from loraphymac.metrics.statistics import Statistics
from loraphymac.runtime.packet import Packet
from loraphymac.runtime.relay_rx import RelayRX
from loraphymac.runtime.wang_sink_decision import (
    WangActiveTransmitter,
    WangGatewaySinkDecisionEngine,
    build_active_transmitter_population,
    install_wang_gateway_scheduler,
    last_closable_window_index,
    wang_runtime_enabled,
)
from loraphymac.runtime.wang_snr_reporting import (
    WangGatewayReceivedReport,
    WangSNRReport,
    WangTransmitterState,
    encode_transmitter_state,
    report_from_successful_rx,
)


pytestmark = pytest.mark.unit


def active(entity_id="N1", entity_type="node", ptx=17, epoch=0):
    return WangActiveTransmitter(entity_id, entity_type, ptx, epoch)


def received(
    *,
    packet_id,
    tx_id="N1",
    rx_id="GW",
    hop=0,
    snr=-4.5,
    ptx=17,
    epoch=0,
    observed=100.0,
    arrived=101.0,
):
    report = WangSNRReport(
        packet_uid=packet_id,
        hop_index=hop,
        tx_id=tx_id,
        rx_id=rx_id,
        snr_db=snr,
        observed_ptx_dbm=ptx,
        observation_time_s=observed,
        last_applied_command_seq=epoch,
    )
    return WangGatewayReceivedReport(report, arrived)


def rejection_map(record):
    return dict(record.rejection_counts)


def test_complete_window_creates_one_immutable_record_with_per_entity_decisions():
    engine = WangGatewaySinkDecisionEngine(
        [active("N1", "node", 17), active("R1", "relay", 20)]
    )
    engine.accept_observation(received(packet_id="UP:N1:1", snr=-4.5))
    engine.accept_observation(
        received(
            packet_id="UP:N2:1",
            tx_id="R1",
            rx_id="GW",
            hop=1,
            snr=-1.5,
            ptx=20,
        )
    )

    result = engine.close_window(0, decision_time_s=310.0)

    assert result.status == "decision_generated"
    assert result.controller_id == "GW"
    assert result.coverage_fraction == 1.0
    assert result.decision_cutoff_s == 310.0
    assert len(result.entity_decisions) == 2
    assert result.source_report_ids == (
        "UP:N1:1:wang-snr:0:N1->GW",
        "UP:N2:1:wang-snr:1:R1->GW",
    )


def test_missing_transmitter_is_explicit_no_decision_and_counts_frequency():
    engine = WangGatewaySinkDecisionEngine([active("N1"), active("N2")])
    engine.accept_observation(received(packet_id="UP:N1:1"))

    result = engine.close_window(0, decision_time_s=310.0)

    assert result.status == "no_decision"
    assert result.coverage_fraction == 0.5
    assert result.missing_transmitter_ids == ("N2",)
    assert result.entity_decisions == ()
    assert engine.no_decision_count == 1


def test_report_arriving_after_cutoff_is_never_used():
    engine = WangGatewaySinkDecisionEngine([active()])
    engine.accept_observation(
        received(packet_id="UP:N1:1", observed=299.9, arrived=310.01)
    )

    result = engine.close_window(0, decision_time_s=311.0)

    assert result.status == "no_decision"
    assert rejection_map(result)["arrival_after_decision_cutoff"] == 1


def test_duplicate_invalid_future_and_old_epoch_reports_are_rejected():
    engine = WangGatewaySinkDecisionEngine([active()])
    valid = received(packet_id="UP:N1:1")
    assert engine.accept_observation(valid)
    assert not engine.accept_observation(valid)
    engine.accept_observation(received(packet_id="UP:N1:2", snr=float("nan")))
    engine.accept_observation(
        received(packet_id="UP:N1:3", observed=105.0, arrived=104.0)
    )
    engine.accept_observation(received(packet_id="UP:N1:4", ptx=14))
    engine.accept_observation(received(packet_id="UP:N1:5", epoch=1))

    result = engine.close_window(0, decision_time_s=310.0)
    rejected = rejection_map(result)

    assert result.status == "decision_generated"
    assert rejected == {
        "duplicate_report": 1,
        "future_dated_observation": 1,
        "invalid_snr": 1,
        "outdated_ptx_epoch": 1,
        "unexpected_declared_ptx": 1,
    }
    assert result.entity_decisions[0].report_count == 1


def test_precommand_observation_cannot_confirm_a_future_power_epoch():
    engine = WangGatewaySinkDecisionEngine([active(epoch=1)])
    engine.accept_observation(received(packet_id="UP:N1:old", epoch=0))

    result = engine.close_window(0, decision_time_s=310.0)

    assert result.status == "no_decision"
    assert rejection_map(result) == {"outdated_ptx_epoch": 1}


def test_half_open_window_boundaries_are_non_overlapping():
    engine = WangGatewaySinkDecisionEngine([active()])
    engine.accept_observation(
        received(packet_id="UP:N1:1", observed=299.9, arrived=300.0)
    )
    engine.accept_observation(
        received(packet_id="UP:N1:2", observed=300.0, arrived=300.1)
    )

    first = engine.close_window(0, decision_time_s=310.0)
    second = engine.close_window(1, decision_time_s=610.0)

    assert first.entity_decisions[0].source_report_ids == (
        "UP:N1:1:wang-snr:0:N1->GW",
    )
    assert second.entity_decisions[0].source_report_ids == (
        "UP:N1:2:wang-snr:0:N1->GW",
    )


def test_stale_wrong_window_and_malformed_receiver_are_rejected():
    engine = WangGatewaySinkDecisionEngine([active()])
    engine.accept_observation(
        received(packet_id="UP:N1:old", observed=299.0, arrived=320.0)
    )
    engine.accept_observation(
        received(
            packet_id="UP:N1:bad",
            rx_id="",
            observed=400.0,
            arrived=401.0,
        )
    )

    result = engine.close_window(1, decision_time_s=610.0)

    assert result.status == "no_decision"
    assert rejection_map(result) == {
        "malformed_receiver_id": 1,
        "wrong_window_stale": 1,
    }


def test_transmitter_means_receive_equal_weight_in_one_network_decision():
    engine = WangGatewaySinkDecisionEngine(
        [active("N1", "node", 17), active("R1", "relay", 20)]
    )
    for index, snr in enumerate((-4.0, -2.0), start=1):
        engine.accept_observation(
            received(packet_id=f"UP:N1:{index}", tx_id="N1", snr=snr)
        )
    for index in range(10):
        engine.accept_observation(
            received(
                packet_id=f"UP:N2:{index}",
                tx_id="R1",
                rx_id="GW",
                hop=1,
                snr=10.0,
                ptx=20,
            )
        )

    result = engine.close_window(0, decision_time_s=310.0)
    decisions = {item.entity_id: item for item in result.entity_decisions}

    assert decisions["N1"].transmitter_mean_snr_db == pytest.approx(-3.0)
    assert decisions["N1"].report_count == 2
    assert decisions["R1"].transmitter_mean_snr_db == pytest.approx(10.0)
    assert decisions["R1"].report_count == 10
    assert result.network_snr_avg_db == pytest.approx(3.5)
    assert result.network_snr_mad_db == pytest.approx(6.5)
    assert result.network_margin_db == pytest.approx(4.5)
    assert result.network_reduction_steps == 1
    assert {item.network_reduction_steps for item in result.entity_decisions} == {1}
    assert decisions["N1"].target_ptx_dbm == 14
    assert decisions["R1"].target_ptx_dbm == 17


def test_common_nstep_allows_different_hardware_clamped_targets():
    engine = WangGatewaySinkDecisionEngine(
        [active("N1", "node", 8, epoch=2), active("R1", "relay", 14, epoch=2)]
    )
    engine.accept_observation(
        received(
            packet_id="UP:N1:1", tx_id="N1", snr=20.0, ptx=8, epoch=2
        )
    )
    engine.accept_observation(
        received(
            packet_id="UP:N2:1",
            tx_id="R1",
            rx_id="GW",
            hop=1,
            snr=20.0,
            ptx=14,
            epoch=2,
        )
    )

    result = engine.close_window(0, decision_time_s=310.0)
    decisions = {item.entity_id: item for item in result.entity_decisions}

    assert result.network_reduction_steps == 9
    assert {item.network_reduction_steps for item in result.entity_decisions} == {9}
    assert decisions["N1"].target_ptx_dbm == 5
    assert decisions["R1"].target_ptx_dbm == 12


@pytest.mark.parametrize(
    ("snr", "margin", "steps", "target"),
    [
        (-4.5, 3.0, 1, 14),
        (-7.5, 0.0, 0, 17),
        (-10.0, -2.5, 0, 17),
    ],
)
def test_positive_zero_and_negative_margin_use_existing_w1_controller(
    snr, margin, steps, target
):
    engine = WangGatewaySinkDecisionEngine([active()])
    engine.accept_observation(received(packet_id="UP:N1:1", snr=snr))

    result = engine.close_window(0, decision_time_s=310.0)
    decision = result.entity_decisions[0]

    assert result.network_margin_db == pytest.approx(margin)
    assert result.network_reduction_steps == steps
    assert decision.network_reduction_steps == steps
    assert decision.target_ptx_dbm == target


@pytest.mark.parametrize(
    ("spec", "snr", "target"),
    [
        (active("N1", "node", 17), 100.0, 5),
        (active("R1", "relay", 20), 100.0, 12),
    ],
)
def test_multi_step_reduction_clamps_to_device_specific_bounds(spec, snr, target):
    engine = WangGatewaySinkDecisionEngine([spec])
    engine.accept_observation(
        received(
            packet_id="UP:N1:1",
            tx_id=spec.entity_id,
            hop=0,
            snr=snr,
            ptx=spec.expected_ptx_dbm,
        )
    )

    result = engine.close_window(0, decision_time_s=310.0)
    decision = result.entity_decisions[0]

    assert decision.target_ptx_dbm == target
    assert result.network_reduction_steps > 1
    assert decision.network_reduction_steps == result.network_reduction_steps


def test_window_processing_is_idempotent_and_does_not_mutate_actual_ptx():
    radio = SimpleNamespace(id="N1", default_ptx=17)
    engine = WangGatewaySinkDecisionEngine([active(radio.id, "node", radio.default_ptx)])
    engine.accept_observation(received(packet_id="UP:N1:1", snr=10.0))

    first = engine.close_window(0, decision_time_s=310.0)
    second = engine.close_window(0, decision_time_s=999.0)

    assert first is second
    assert len(engine.decisions) == 1
    assert first.entity_decisions[0].target_ptx_dbm == 5
    assert radio.default_ptx == 17


def uplink_packet():
    fopts = encode_transmitter_state(WangTransmitterState(17, 0))
    pkt = Packet(
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
        payload_size=frame_size_bytes(50, len(fopts)),
        app_payload_size=50,
        toa=0.1,
        tx_start=0.0,
        tx_end=0.1,
        packet_uid="UP:N1:1",
        mtype=MType.DATA_UP,
        dst_id="GW",
        hop_limit=2,
        fcnt=1,
        fopts=fopts,
        fctrl=encode_fctrl(fopts_len=len(fopts)),
    )
    pkt.decoded = True
    pkt.feasible = True
    pkt.collided = False
    pkt.is_intended_receiver = True
    pkt.snr = 1.0
    return pkt


def test_intermediate_relay_never_calls_sink_decision_engine():
    calls = []
    fake_engine = SimpleNamespace(accept_observation=lambda item: calls.append(item))
    rx = object.__new__(RelayRX)
    rx.env = SimpleNamespace(now=10.0)
    rx.relay = SimpleNamespace(
        id="R1",
        wang_snr_reporting_enabled=True,
        wang_sink_decision_engine=fake_engine,
        wang_routing_map={"N1": "R1", "R1": "GW", "GW": None},
    )
    rx.stats = Statistics()

    rx._handle_wang_snr_success(uplink_packet())

    assert calls == []


def test_sink_engine_cannot_be_instantiated_for_intermediate_relay():
    with pytest.raises(ValueError, match="main gateway"):
        WangGatewaySinkDecisionEngine([active()], gateway_id="R1")


def test_active_population_comes_from_routes_and_excludes_unused_relays():
    nodes = [
        SimpleNamespace(id="N1", ptx_max=17, default_ptx=17),
        SimpleNamespace(id="N2", ptx_max=17, default_ptx=17),
    ]
    relays = [
        SimpleNamespace(id="R1", ptx_max=20, default_ptx=20),
        SimpleNamespace(id="R2", ptx_max=20, default_ptx=20),
    ]
    routing_map = {"N1": "R1", "N2": "GW", "R1": "GW", "R2": "GW"}

    population = build_active_transmitter_population(nodes, relays, routing_map)

    assert [(item.entity_id, item.expected_ptx_dbm) for item in population] == [
        ("N1", 17),
        ("N2", 17),
        ("R1", 20),
    ]


def test_active_population_uses_resolved_hardware_bound_not_wang_constant():
    node = SimpleNamespace(id="N1", ptx_max=16, default_ptx=16)

    population = build_active_transmitter_population(
        [node], [], {"N1": "GW"}
    )

    assert population[0].expected_ptx_dbm == 16


def test_exact_model_identity_is_the_only_runtime_gate():
    assert wang_runtime_enabled({"tx_power_model_id": "wang_atp_sf7_a2only"})
    for model_id in (None, "rapc", "fixed", "islam_dbpc_itm6"):
        assert not wang_runtime_enabled({"tx_power_model_id": model_id})


def test_wang_configuration_gate_is_fixed_sf7_corrected_and_legacy_is_unchanged():
    wang = build_config(
        {
            "tx_power_model_id": "wang_atp_sf7_a2only",
            "node_ptx_mode": "fixed",
            "relay_ptx_mode": "fixed",
            "energy_accounting_mode": "corrected",
        }
    )
    validate_configuration(wang, known_keys=DEFAULT_CONFIG)
    validate_configuration(build_config(), known_keys=DEFAULT_CONFIG)


@pytest.mark.parametrize(
    "override",
    [
        {"node_ptx_mode": "adaptive"},
        {"energy_accounting_mode": "legacy"},
        {"node_sf_list": [8]},
        {"node_ptx_init": 17},
    ],
)
def test_invalid_wang_runtime_contract_is_rejected(override):
    config = build_config(
        {
            "tx_power_model_id": "wang_atp_sf7_a2only",
            "node_ptx_mode": "fixed",
            "relay_ptx_mode": "fixed",
            "energy_accounting_mode": "corrected",
            **override,
        }
    )
    with pytest.raises(ConfigurationError):
        validate_configuration(config, known_keys=DEFAULT_CONFIG)


def test_gateway_scheduler_includes_same_time_cutoff_delivery_once():
    env = simpy.Environment()
    gateway = SimpleNamespace(id="GW")
    node = SimpleNamespace(id="N1", ptx_max=17, default_ptx=17)
    engine = install_wang_gateway_scheduler(
        env=env,
        gateway=gateway,
        nodes=[node],
        relays=[],
        routing_map={"N1": "GW", "GW": None},
        simulation_horizon_s=620.0,
    )

    def deliver_at_cutoff():
        yield env.timeout(310.0)
        engine.accept_observation(
            received(
                packet_id="UP:N1:1",
                snr=-4.5,
                observed=299.9,
                arrived=310.0,
            )
        )

    env.process(deliver_at_cutoff())
    env.run(until=311.0)

    assert len(engine.decisions) == 1
    assert engine.decisions[0].status == "decision_generated"
    assert engine.decisions[0].network_reduction_steps == 1
    assert gateway.wang_sink_decision_engine is engine
    assert node.wang_snr_reporting_enabled is True
    assert node.wang_last_applied_command_seq == 0
    assert gateway.wang_routing_map is node.wang_routing_map
    with pytest.raises(TypeError):
        node.wang_routing_map["N1"] = "R9"


def test_private_path_and_packet_uid_cannot_change_sink_decision():
    routing_map = {"N1": "GW", "GW": None}
    reports = []
    for private_path, private_uid in (
        (["N1", "GW"], "PRIVATE-A"),
        (["N1", "R9", "GW"], "PRIVATE-B"),
    ):
        pkt = uplink_packet()
        pkt.next_hop = "GW"
        pkt.hop_limit = 1
        pkt.path = private_path
        pkt.packet_uid = private_uid
        reports.append(
            report_from_successful_rx(
                pkt,
                "GW",
                100.0,
                routing_map=routing_map,
            )
        )

    decisions = []
    for report in reports:
        engine = WangGatewaySinkDecisionEngine([active()])
        assert engine.accept_observation(
            WangGatewayReceivedReport(report, gateway_arrival_time_s=101.0)
        )
        decisions.append(engine.close_window(0, decision_time_s=310.0))

    assert reports[0] == reports[1]
    assert decisions[0].network_margin_db == decisions[1].network_margin_db
    assert decisions[0].network_reduction_steps == decisions[1].network_reduction_steps


def test_numeric_horizon_excludes_cutoff_at_or_after_stop_time():
    assert last_closable_window_index(300.0) == -1
    assert last_closable_window_index(310.0) == -1
    assert last_closable_window_index(311.0) == 0
    assert last_closable_window_index(3600.0) == 10


def test_disabled_gateway_gate_does_not_feed_engine_or_change_ptx():
    calls = []
    radio = SimpleNamespace(
        id="GW",
        wang_snr_reporting_enabled=False,
        wang_sink_decision_engine=SimpleNamespace(
            accept_observation=lambda item: calls.append(item)
        ),
        default_ptx=20,
    )
    rx = object.__new__(RelayRX)
    rx.env = SimpleNamespace(now=10.0)
    rx.relay = radio
    rx.stats = Statistics()
    pkt = uplink_packet()
    pkt.path = ["N1", "GW"]
    pkt.next_hop = "GW"

    rx._handle_wang_snr_success(pkt)

    assert calls == []
    assert radio.default_ptx == 20
