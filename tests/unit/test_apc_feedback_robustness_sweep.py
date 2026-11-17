from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from loraphymac.experiments.apc_feedback_robustness_sweep.contract import (
    APC_FEEDBACK_ROBUSTNESS_PROFILE,
    APC_FEEDBACK_ROBUSTNESS_RUN_COUNT,
    APC_FEEDBACK_ROBUSTNESS_RUN_TABLES,
    TABLE_SCHEMAS,
    feedback_robustness_config_violations,
    validate_feedback_robustness_outputs,
)
from loraphymac.experiments.apc_feedback_robustness_sweep.feedback_loss import (
    additional_feedback_erasure,
    feedback_loss_uniform,
    is_apc_feedback_frame,
)
from loraphymac.experiments.common.multiseed_pipeline import (
    build_experiment_grid,
    load_experiment_spec,
)
from loraphymac.experiments.common.contract_registry import (
    experiment_run_tables,
    experiment_table_filenames,
)
from loraphymac.mac.frame_defs import MType, encode_fctrl
from loraphymac.metrics.statistics import Statistics


def test_approved_configuration_builds_exact_30_run_grid():
    spec = load_experiment_spec(
        "src/loraphymac/experiments/apc_feedback_robustness_sweep/configs/aa_feedback_robustness_v1.json"
    )
    runs = build_experiment_grid(spec)
    assert len(runs) == APC_FEEDBACK_ROBUSTNESS_RUN_COUNT == 30
    assert len({row["run_id"] for row in runs}) == 30
    assert len({row["scientific_config"]["feedback_additional_loss_probability"] for row in runs}) == 3
    assert {row["scientific_config"]["node_period"] for row in runs} == {30, 90}
    assert {row["seed"] for row in runs} == {30, 36, 42, 48, 56}
    assert all(row["scientific_config"]["experiment_profile"] == APC_FEEDBACK_ROBUSTNESS_PROFILE for row in runs)
    assert set(experiment_run_tables(runs[0]["scientific_config"])) == set(APC_FEEDBACK_ROBUSTNESS_RUN_TABLES)
    assert set(experiment_table_filenames(APC_FEEDBACK_ROBUSTNESS_PROFILE)) == set(APC_FEEDBACK_ROBUSTNESS_RUN_TABLES.values())


def test_profile_rejects_unapproved_probability_and_non_adaptive_mode():
    base = {
        "experiment_profile": APC_FEEDBACK_ROBUSTNESS_PROFILE,
        "feedback_additional_loss_probability": 0.2,
        "node_period": 30,
        "node_num": 100,
        "relay_num": 2,
        "terrain_delta_h": 90,
        "sim_time": 3600,
        "seed": 30,
        "node_ptx_mode": "fixed",
        "relay_ptx_mode": "adaptive",
    }
    errors = feedback_robustness_config_violations(base)
    assert any("probability" in error for error in errors)
    assert any("adaptive" in error for error in errors)


def test_feedback_loss_is_deterministic_common_random_and_attempt_specific():
    shared = dict(seed=42, controller_id="GW", target_id="N7", decision_ordinal="000013", attempt_number=1)
    u1 = feedback_loss_uniform(**shared)
    u2 = feedback_loss_uniform(**shared)
    assert u1 == u2
    assert feedback_loss_uniform(**{**shared, "attempt_number": 2}) != u1
    erased_10, sampled_10 = additional_feedback_erasure(
        0.1, seed=42, controller_id="GW", target_id="N7", feedback_id="r:apc:000013", attempt_number=1
    )
    erased_30, sampled_30 = additional_feedback_erasure(
        0.3, seed=42, controller_id="GW", target_id="N7", feedback_id="x:apc:000013", attempt_number=1
    )
    assert sampled_10 == sampled_30 == u1
    assert not erased_10 or erased_30
    assert additional_feedback_erasure(
        0.0, seed=42, controller_id="GW", target_id="N7", feedback_id="r:apc:000013", attempt_number=1
    ) == (False, None)


def test_feedback_loss_probability_thresholds_are_exact_and_rng_neutral(monkeypatch):
    def forbidden_rng(*_args, **_kwargs):
        raise AssertionError("feedback loss must not consume global RNG")

    monkeypatch.setattr(random, "random", forbidden_rng)
    monkeypatch.setattr(np.random, "random", forbidden_rng)
    for ordinal in range(1, 256):
        common = dict(
            seed=42, controller_id="GW", target_id="N7",
            feedback_id=f"r:apc:{ordinal:06d}", attempt_number=1,
        )
        uniform = feedback_loss_uniform(
            seed=42, controller_id="GW", target_id="N7",
            decision_ordinal=f"{ordinal:06d}", attempt_number=1,
        )
        erased_10, sample_10 = additional_feedback_erasure(0.1, **common)
        erased_30, sample_30 = additional_feedback_erasure(0.3, **common)
        assert sample_10 == sample_30 == uniform
        assert erased_10 is (uniform < 0.1)
        assert erased_30 is (uniform < 0.3)
        assert not erased_10 or erased_30


def test_only_marked_apc_frame_for_matching_target_is_eligible():
    packet = SimpleNamespace(
        mtype=int(MType.DATA_DOWN), fctrl=encode_fctrl(apc=True),
        apc_info={"feedback_id": "r:apc:000001", "context": {"target_id": "N7", "controller_id": "GW"}},
    )
    assert is_apc_feedback_frame(packet, "N7")
    assert not is_apc_feedback_frame(packet, "N8")
    packet.fctrl = encode_fctrl(apc=False)
    assert not is_apc_feedback_frame(packet, "N7")
    packet.fctrl = encode_fctrl(apc=True)
    packet.mtype = int(MType.DATA_UP)
    assert not is_apc_feedback_frame(packet, "N7")


def test_statistics_keeps_synthetic_erasure_out_of_packet_fate():
    stats = Statistics(output_profile="summary")
    stats.set_run_metadata(
        run_id="robust-run", experiment_profile=APC_FEEDBACK_ROBUSTNESS_PROFILE,
        seed=42, sim_time=3600, feedback_additional_loss_probability=1.0,
    )
    stats.register_feedback_tpi_transmitter("N7", "node", initial_tpi_index=5)
    apc_info = {
        "feedback_id": "robust-run:apc:000001", "decision_time": 10.0,
        "context": {
            "controller_id": "GW", "target_id": "N7", "target_type": "node",
            "control_scope": "node_apc", "tx_power_index": 5,
        },
    }
    stats.record_apc_feedback_event(
        event="generated", time=10.0, apc_info=apc_info,
        feedback_id=apc_info["feedback_id"], mtype=int(MType.DATA_DOWN),
    )
    packet = SimpleNamespace(
        mtype=int(MType.DATA_DOWN), fctrl=encode_fctrl(apc=True), next_hop="N7",
        apc_info=apc_info, ch_id=1, toa=0.2, ptx_dbm=14.0, tx_start=11.0,
        tx_end=11.2, src_id="GW", current_tx="GW", dst_id="N7", hop_index=0,
        hop_limit=5, path=["GW", "N7"], sf=7, bw_hz=125000, fcnt=1,
        fport=1, fopts=b"abcd", tx_mode="APC_DOWN", packet_uid="ctrl-1",
    )
    stats.record_tx(packet, "GW")
    # AirInterface delivers a copied packet object; tracking by object identity
    # loses the physical TX attempt at the receiver.
    received_packet = copy.copy(packet)
    assert received_packet is not packet
    assert stats.evaluate_additional_feedback_loss(received_packet, "N7", 11.2)
    assert not stats.packet_fates
    tables = stats.get_feedback_robustness_tables()
    assert len(tables["feedback_attempt_metrics_df"]) == 1
    row = tables["feedback_attempt_metrics_df"].iloc[0]
    assert row["native_reception_status"] == "success"
    assert row["synthetic_erasure_decision"]
    assert row["logical_delivery_status"] == "synthetic_erased"
    assert tables["feedback_transaction_metrics_df"].iloc[0]["synthetic_erasures"] == 1

    # A physical data TX after the desired version exists, but before its
    # application, is included in the stale numerator and denominator.
    data_packet = SimpleNamespace(
        mtype=int(MType.DATA_UP), fctrl=0, next_hop="R1", apc_info=None,
        ch_id=1, toa=0.3, ptx_dbm=14.0, tx_start=12.0, tx_end=12.3,
        src_id="N7", current_tx="N7", dst_id="R1", hop_index=0,
        hop_limit=5, path=["N7", "R1"], sf=7, bw_hz=125000,
        fcnt=2, fport=1, fopts=b"", tx_mode="RELAY", packet_uid="data-1",
    )
    snapshot = stats.record_tx(data_packet, "N7")
    tables = stats.get_feedback_robustness_tables()
    stale = tables["stale_tpi_transmission_summary_df"].loc[
        tables["stale_tpi_transmission_summary_df"]["control_scope_id"].notna()
    ].iloc[0]
    assert stale["eligible_data_tx_count"] == 1
    assert stale["stale_tpi_tx_count"] == 1
    assert snapshot["stale_tpi_status"] == "STALE"
    airtime = tables["apc_control_airtime_summary_df"].iloc[0]
    assert airtime["total_observed_tx_count"] == 2
    assert airtime["apc_control_airtime_percent"] == 40.0


def test_application_latency_and_margin_first_entry_are_separate_and_censored():
    stats = Statistics(output_profile="summary")
    stats.set_run_metadata(
        run_id="latency-run", experiment_profile=APC_FEEDBACK_ROBUSTNESS_PROFILE,
        seed=30, sim_time=3600, feedback_additional_loss_probability=0.0,
    )
    apc_info = {
        "feedback_id": "latency-run:apc:000001", "decision_time": 10.0,
        "context": {
            "controller_id": "GW", "target_id": "N9", "target_type": "node",
            "control_scope": "node_apc", "tx_power_index": 6,
        },
    }
    stats.record_apc_feedback_event(
        event="generated", time=10.0, apc_info=apc_info,
        feedback_id=apc_info["feedback_id"], mtype=int(MType.DATA_DOWN),
    )
    stats.record_apc_feedback_event(
        event="applied", time=15.0, apc_info=apc_info,
        feedback_id=apc_info["feedback_id"], applied_tx_power_index=6,
    )
    stats.record_margin_sample(
        entity_id="N9", entity_type="node", margin=5.0, decoded=True,
        time=16.0,
    )
    transaction = stats.get_feedback_robustness_tables()[
        "feedback_transaction_metrics_df"
    ].iloc[0]
    convergence = stats.get_feedback_robustness_tables()[
        "apc_convergence_metrics_df"
    ].iloc[0]
    assert transaction["application_latency_s"] == 5.0
    assert convergence["margin_first_entry_latency_s"] == 6.0
    assert convergence["observation_status"] == "first_entry_proxy"


def _data_packet(*, uid, transmitter, mtype, tx_time, ptx=14.0):
    return SimpleNamespace(
        mtype=int(mtype), packet_uid=uid, fctrl=0, next_hop="GW",
        apc_info=None, ch_id=1, toa=0.25, ptx_dbm=ptx,
        tx_start=tx_time, tx_end=tx_time + 0.25, src_id=transmitter,
        current_tx=transmitter, dst_id="GW", hop_index=0, hop_limit=5,
        path=[transmitter, "GW"], sf=7, bw_hz=125000, fcnt=1,
        fport=1, fopts=b"", tx_mode="DATA",
    )


def _robust_stats(run_id="stale-test"):
    stats = Statistics(output_profile="summary")
    stats.set_run_metadata(
        run_id=run_id, experiment_profile=APC_FEEDBACK_ROBUSTNESS_PROFILE,
        seed=42, sim_time=3600,
    )
    stats.register_feedback_tpi_transmitter("N7", "node", initial_tpi_index=5)
    stats.register_feedback_tpi_transmitter("R1", "relay", initial_tpi_index=5)
    return stats


def _decision(stats, target, *, ordinal, time, tpi=5, scope=None):
    role = "relay" if target.startswith("R") else "node"
    context = {
        "controller_id": "GW", "target_id": target, "target_type": role,
        "control_scope": scope or f"{role}_apc", "tx_power_index": tpi,
    }
    info = {
        "feedback_id": f"{stats.run_metadata['run_id']}:{target}:apc:{ordinal:06d}",
        "decision_time": time, "context": context,
    }
    stats.record_apc_feedback_event(
        event="generated", time=time, apc_info=info,
        feedback_id=info["feedback_id"], mtype=int(MType.DATA_DOWN),
    )
    return info


def _apply(stats, info, *, time):
    stats.record_apc_feedback_event(
        event="applied", time=time, apc_info=info,
        feedback_id=info["feedback_id"],
        applied_tx_power_index=info["context"]["tx_power_index"],
    )


def test_physical_data_tx_classifies_predecision_stale_current_and_applied_at_tx_time():
    stats = _robust_stats()
    pre = stats.record_tx(
        _data_packet(uid="logical-1", transmitter="N7", mtype=MType.DATA_UP, tx_time=1),
        "N7",
    )
    assert pre["stale_tpi_status"] == "PRE-DECISION"

    first = _decision(stats, "N7", ordinal=1, time=2, tpi=5)
    _apply(stats, first, time=3)
    current = stats.record_tx(
        _data_packet(uid="logical-2", transmitter="N7", mtype=MType.DATA_UP, tx_time=4),
        "N7",
    )
    assert current["stale_tpi_status"] == "CURRENT"
    assert current["applied_tpi_version"] == current["latest_decided_tpi_version"] == 1

    pending = _decision(stats, "N7", ordinal=2, time=5, tpi=6)
    stale = stats.record_tx(
        _data_packet(uid="logical-3", transmitter="N7", mtype=MType.DATA_UP, tx_time=6),
        "N7",
    )
    assert stale["stale_tpi_status"] == "STALE"
    _apply(stats, pending, time=7)
    applied = stats.record_tx(
        _data_packet(uid="logical-3", transmitter="N7", mtype=MType.DATA_UP, tx_time=8),
        "N7",
    )
    assert applied["stale_tpi_status"] == "CURRENT"
    assert applied["applied_tpi_version"] == applied["latest_decided_tpi_version"] == 2

    rows = stats.get_feedback_robustness_tables()[
        "stale_tpi_transmission_summary_df"
    ]
    assert rows.total_data_physical_tx_count.sum() == 4
    assert rows.pre_decision_tx_count.sum() == 1
    assert rows.post_decision_data_tx_count.sum() == 3
    assert rows.eligible_data_tx_count.sum() == 3
    assert rows.stale_tpi_tx_count.sum() == 1
    assert rows.current_tpi_tx_count.sum() == 2
    assert rows.unclassified_tx_count.sum() == 0
    assert rows.stale_tpi_tx_count.sum() / rows.eligible_data_tx_count.sum() == pytest.approx(1 / 3)


def test_baseline_registration_is_opt_in_and_seeds_initial_applied_version():
    stats = _robust_stats()
    stats.register_feedback_tpi_transmitter("N8", "node", initial_tpi_index=4)
    assert stats.feedback_tpi_registered_transmitters["N8"]["initial_tpi_index"] == 4
    assert stats.feedback_tpi_registered_transmitters["N8"]["transmitter_role"] == "node"

    baseline = Statistics(output_profile="summary")
    baseline.set_run_metadata(run_id="general", experiment_profile="general_multisf_multibw")
    baseline.register_feedback_tpi_transmitter("N8", "node", initial_tpi_index=4)
    assert baseline.feedback_tpi_registered_transmitters == {}


def test_snapshot_uses_tx_time_state_and_covers_relay_forwarding():
    stats = _robust_stats()
    info = _decision(stats, "R1", ordinal=1, time=1, tpi=5)
    # The packet may have waited in a queue since before this decision.
    _apply(stats, info, time=2)
    relay_snapshot = stats.record_tx(
        _data_packet(uid="forwarded-logical", transmitter="R1", mtype=MType.DATA_UP_FWD, tx_time=3),
        "R1",
    )
    assert relay_snapshot["transmitter_role"] == "relay"
    assert relay_snapshot["mtype"] == "DATA_UP_FWD"
    assert relay_snapshot["tx_timestamp_s"] == 3
    assert relay_snapshot["actual_ptx_dbm"] == 14.0
    assert relay_snapshot["stale_tpi_status"] == "CURRENT"


def test_physical_identity_distinguishes_retries_and_deduplicates_callback():
    stats = _robust_stats()
    info = _decision(stats, "N7", ordinal=1, time=1)
    _apply(stats, info, time=2)
    packet = _data_packet(uid="same-logical-packet", transmitter="N7", mtype=MType.DATA_UP, tx_time=3)
    first = stats.record_tx(packet, "N7")
    duplicate = stats.record_tx(packet, "N7")
    assert duplicate["physical_tx_id"] == first["physical_tx_id"]
    assert packet.physical_tx_id == first["physical_tx_id"]

    retry = _data_packet(uid="same-logical-packet", transmitter="N7", mtype=MType.DATA_UP, tx_time=4)
    second = stats.record_tx(retry, "N7")
    assert second["physical_tx_id"] != first["physical_tx_id"]
    assert second["packet_uid"] == first["packet_uid"]
    row = stats.get_feedback_robustness_tables()["stale_tpi_transmission_summary_df"].iloc[0]
    assert row.total_data_physical_tx_count == 2


def test_missing_and_inconsistent_versions_are_unclassified_and_counted():
    stats = _robust_stats()
    info = _decision(stats, "N7", ordinal=1, time=1)
    # No application event: post-decision state exists but applied version is missing.
    missing_applied = stats.record_tx(
        _data_packet(uid="missing-applied", transmitter="N7", mtype=MType.DATA_UP, tx_time=2),
        "N7",
    )
    assert missing_applied["stale_tpi_status"] == "STALE"  # known initial version 0

    stats.feedback_tpi_applied_by_scope["GW|N7|node_apc"] = None
    missing = stats.record_tx(
        _data_packet(uid="missing-state", transmitter="N7", mtype=MType.DATA_UP, tx_time=3),
        "N7",
    )
    assert missing["stale_tpi_status"] == "UNCLASSIFIED"
    stats.feedback_tpi_applied_by_scope["GW|N7|node_apc"] = 3
    invalid = stats.record_tx(
        _data_packet(uid="invalid-state", transmitter="N7", mtype=MType.DATA_UP, tx_time=4),
        "N7",
    )
    assert invalid["stale_tpi_status"] == "UNCLASSIFIED"
    stale = stats.get_feedback_robustness_tables()[
        "stale_tpi_transmission_summary_df"
    ].iloc[0]
    assert stale.unclassified_tx_count == 2
    assert stale.version_invariant_violation_count == 1
    assert info["context"]["tpi_version"] == 1


def test_zero_tx_schema_and_packet_copy_preserve_logical_and_physical_ids():
    from loraphymac.runtime.packet import Packet

    packet = Packet(
        src_id="N7", current_tx="N7", next_hop="R1", path=["N7", "R1"],
        hop_index=0, ch_id=1, freq_mhz=915.0, sf=7, bw_hz=125000,
        ptx_dbm=14.0, payload_size=10, toa=0.2, tx_start=1.0, tx_end=1.2,
        packet_uid="logical-packet", physical_tx_id="run:data-tx:00000001",
    )
    copied = packet.copy()
    assert copied.packet_uid == packet.packet_uid
    assert copied.physical_tx_id == packet.physical_tx_id

    stats = _robust_stats()
    from loraphymac.experiments.apc_feedback_robustness_sweep.contract import STALE_TPI_COLUMNS
    table = stats.get_feedback_robustness_tables()["stale_tpi_transmission_summary_df"]
    assert tuple(table.columns) == STALE_TPI_COLUMNS
    assert set(table["transmitter_role"]) == {"node", "relay"}
    assert table["total_data_physical_tx_count"].sum() == 0
    assert table["stale_tpi_transmission_fraction"].isna().all()


def test_stale_tpi_instrumentation_does_not_consume_rng_or_change_packet_fate():
    def forbidden(*_args, **_kwargs):
        raise AssertionError("stale-TPI instrumentation must not consume RNG")

    stats = _robust_stats()
    packet = _data_packet(uid="neutral", transmitter="N7", mtype=MType.DATA_UP, tx_time=1)
    before_packet = (packet.toa, packet.ptx_dbm, packet.tx_start, packet.tx_end)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(random, "random", forbidden)
    monkeypatch.setattr(np.random, "random", forbidden)
    try:
        stats.record_tx(packet, "N7")
    finally:
        monkeypatch.undo()
    assert (packet.toa, packet.ptx_dbm, packet.tx_start, packet.tx_end) == before_packet
    baseline = Statistics(output_profile="summary")
    baseline.set_run_metadata(run_id="stale-test", sim_time=3600)
    baseline_packet = _data_packet(
        uid="neutral", transmitter="N7", mtype=MType.DATA_UP, tx_time=1
    )
    baseline.record_tx(baseline_packet, "N7")
    assert stats.packet_fates == baseline.packet_fates
    assert dict(stats.summary_node_tx_airtime) == dict(baseline.summary_node_tx_airtime)
    assert dict(stats.summary_tx_airtime) == dict(baseline.summary_tx_airtime)
    assert stats.packet_fates["neutral"]["latest_state"] == "TX_IN_FLIGHT"
    assert stats.summary_node_tx_airtime["N7"][0] == packet.toa
    assert stats.feedback_tpi_data_tx_counters


def test_generated_feedback_retry_and_duplicate_generation_keep_one_decision_version():
    stats = _robust_stats()
    info = _decision(stats, "N7", ordinal=1, time=1, tpi=5)
    assert stats.feedback_tpi_latest_decided_by_scope["GW|N7|node_apc"] == 1
    stats.record_apc_feedback_event(
        event="retry_generated", time=2, apc_info=info,
        feedback_id=info["feedback_id"], mtype=int(MType.DATA_DOWN),
    )
    stats.record_apc_feedback_event(
        event="generated", time=3, apc_info=info,
        feedback_id=info["feedback_id"], mtype=int(MType.DATA_DOWN),
    )
    assert stats.feedback_tpi_latest_decided_by_scope["GW|N7|node_apc"] == 1
    same_value_new_command = _decision(stats, "N7", ordinal=2, time=4, tpi=5)
    assert stats.feedback_tpi_latest_decided_by_scope["GW|N7|node_apc"] == 2
    assert same_value_new_command["context"]["tpi_version"] == 2


def test_independent_transmitter_scopes_do_not_share_versions():
    stats = _robust_stats()
    node_info = _decision(stats, "N7", ordinal=1, time=1, tpi=5)
    relay_info = _decision(stats, "R1", ordinal=1, time=1, tpi=5)
    _apply(stats, node_info, time=2)
    _apply(stats, relay_info, time=2)
    _decision(stats, "N7", ordinal=2, time=3, tpi=6)

    node = stats.record_tx(
        _data_packet(uid="node", transmitter="N7", mtype=MType.DATA_UP, tx_time=4),
        "N7",
    )
    relay = stats.record_tx(
        _data_packet(uid="relay", transmitter="R1", mtype=MType.DATA_UP_FWD, tx_time=4),
        "R1",
    )
    assert node["control_scope_id"] == "GW|N7|node_apc"
    assert node["stale_tpi_status"] == "STALE"
    assert relay["control_scope_id"] == "GW|R1|relay_apc"
    assert relay["stale_tpi_status"] == "CURRENT"


def test_same_timestamp_classification_follows_observed_callback_order():
    stats = _robust_stats()
    info = _decision(stats, "N7", ordinal=1, time=10, tpi=6)
    before_apply = stats.record_tx(
        _data_packet(uid="same-time-before", transmitter="N7", mtype=MType.DATA_UP, tx_time=10),
        "N7",
    )
    _apply(stats, info, time=10)
    after_apply = stats.record_tx(
        _data_packet(uid="same-time-after", transmitter="N7", mtype=MType.DATA_UP, tx_time=10),
        "N7",
    )
    assert before_apply["stale_tpi_status"] == "STALE"
    assert after_apply["stale_tpi_status"] == "CURRENT"
    assert before_apply["event_sequence"] < after_apply["event_sequence"]


def test_missing_latest_decision_is_unclassified_not_current():
    stats = _robust_stats()
    info = _decision(stats, "N7", ordinal=1, time=1)
    _apply(stats, info, time=2)
    stats.feedback_tpi_latest_decided_by_scope.pop("GW|N7|node_apc")
    snapshot = stats.record_tx(
        _data_packet(uid="missing-latest", transmitter="N7", mtype=MType.DATA_UP, tx_time=3),
        "N7",
    )
    assert snapshot["stale_tpi_status"] == "UNCLASSIFIED"
    assert pd.isna(snapshot["latest_decided_tpi_version"])


def test_stale_table_csv_roundtrip_keeps_stable_schema(tmp_path):
    from loraphymac.experiments.apc_feedback_robustness_sweep.contract import STALE_TPI_COLUMNS

    stats = _robust_stats()
    info = _decision(stats, "N7", ordinal=1, time=1)
    _apply(stats, info, time=2)
    stats.record_tx(
        _data_packet(uid="csv-roundtrip", transmitter="N7", mtype=MType.DATA_UP, tx_time=3),
        "N7",
    )
    table = stats.get_feedback_robustness_tables()["stale_tpi_transmission_summary_df"]
    path = tmp_path / "stale_tpi_by_transmitter.csv"
    table.to_csv(path, index=False)
    loaded = pd.read_csv(path)
    assert tuple(loaded.columns) == STALE_TPI_COLUMNS
    assert loaded.loc[0, "transmitter_id"] == "N7"
    assert loaded.loc[0, "current_tpi_tx_count"] == 1


def test_robustness_output_tables_keep_stable_empty_schemas():
    stats = Statistics(output_profile="summary")
    stats.set_run_metadata(
        run_id="empty", experiment_profile=APC_FEEDBACK_ROBUSTNESS_PROFILE,
        seed=30, sim_time=3600, feedback_additional_loss_probability=0.0,
    )
    tables = stats.get_feedback_robustness_tables()
    assert tuple(tables["feedback_attempt_metrics_df"].columns)
    assert tables["feedback_attempt_metrics_df"].empty
    assert len(tables["apc_control_airtime_summary_df"]) == 1
    assert tables["apc_control_airtime_summary_df"].iloc[0]["apc_control_airtime_percent"] == 0.0


def test_scientific_output_contract_accepts_empty_lifecycle_schemas_and_rejects_missing_table():
    spec = load_experiment_spec(
        "src/loraphymac/experiments/apc_feedback_robustness_sweep/configs/aa_feedback_robustness_v1.json"
    )
    config = {
        **spec.base_config, "node_num": 100, "node_period": 30,
        "feedback_additional_loss_probability": 0.0, "seed": 30,
    }
    outputs = {key: pd.DataFrame(columns=TABLE_SCHEMAS[key]) for key in APC_FEEDBACK_ROBUSTNESS_RUN_TABLES}
    outputs["apc_control_airtime_summary_df"] = pd.DataFrame([{
        "run_id": "validation", "apc_control_tx_count": 0,
        "apc_control_airtime_s": 0.0, "total_observed_tx_count": 0,
        "total_observed_tx_airtime_s": 0.0, "apc_control_airtime_percent": 0.0,
        "by_transmitter_role_json": "{}", "by_mtype_json": "{}",
    }], columns=TABLE_SCHEMAS["apc_control_airtime_summary_df"])
    summary = {
        "pdr_percent": 0.0, "packet_fate_delivered_count": 0,
        "avg_e2e_delay_ms": 0.0, "total_network_energy_J": 1.0,
        "energy_per_delivered_packet_J": 0.0,
        "packet_fate_conservation_delta": 0,
        "collision_outcome_conservation_delta": 0,
    }
    validate_feedback_robustness_outputs(config, summary, outputs)
    outputs.pop("feedback_attempt_metrics_df")
    with pytest.raises(ValueError, match="tables"):
        validate_feedback_robustness_outputs(config, summary, outputs)
