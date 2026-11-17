import pytest

from loraphymac import SIMULATOR_VERSION
from loraphymac.engine import run_single_simulation


BASE_CONFIG = {
    "seed": 777,
    "node_num": 20,
    "relay_num": 2,
    "node_jitter": 0,
    "node_ptx_mode": "fixed",
    "relay_ptx_mode": "fixed",
    "node_rx_window_enabled": False,
    "plot_routing": False,
    "export_results": False,
    "export_debug": False,
    "export_internal": False,
}


def test_unbounded_q1_preserves_fixed_seed_pdr_delay_and_event_counts():
    config = dict(BASE_CONFIG)
    config.update(
        {
            "sim_time": 60,
            "node_period": 5,
            "relay_duty_cycle": None,
            "relay_tx_waiting_queue_capacity_packets": None,
        }
    )

    summary, _, _, outputs = run_single_simulation(config)

    # Regression oracle after receiver-wide RF broadcast and receiver-specific
    # propagation timing were introduced.
    assert summary["tx_packets"] == 179
    assert summary["rx_packets_gw"] == 133
    assert summary["pdr_percent"] == pytest.approx(74.302)
    assert summary["avg_e2e_delay_ms"] == pytest.approx(278.704)
    assert summary["drop_not_feasible_count"] == 0
    assert summary["drop_collision_count"] == 0
    assert summary["drop_half_duplex_count"] == 46
    assert summary["drop_demod_busy_count"] == 0
    assert len(outputs["event_log_df"]) == 658
    edge_df = outputs["collision_interference_edges_df"]
    assert {
        "spectral_overlap_hz",
        "spectral_overlap_fraction",
        "required_sir_db",
        "pair_margin_db",
        "dominant_interferer",
    }.issubset(edge_df.columns)
    metadata = outputs["run_metadata_df"].iloc[0]
    assert metadata["experiment_profile"] == "general_multisf_multibw"
    assert metadata["collision_model_version"] == "pairwise_spectral_sir_v3"
    assert metadata["collision_sf_interference_mode"] == "sir_matrix"
    assert metadata["collision_sir_matrix_version"] == ("rakibul_islam_2023_table4")

    metrics = outputs["relay_tx_queue_metrics_df"]
    assert (metrics["attempt_conservation_delta"] == 0).all()
    assert (metrics["accepted_conservation_delta"] == 0).all()
    assert summary["queue_metrics_version"] == "2.0"
    assert summary["simulator_version"] == SIMULATOR_VERSION


def test_duty_cycle_and_half_duplex_waits_are_separate_and_conserved():
    config = dict(BASE_CONFIG)
    config.update(
        {
            "sim_time": 20,
            "seed": 42,
            "node_period": 2,
            "relay_duty_cycle": 0.01,
        }
    )

    _, _, _, outputs = run_single_simulation(config)
    lifecycle = outputs["tx_queue_lifecycle_df"]
    completed = lifecycle[lifecycle["status"] == "COMPLETED"].copy()
    assert not completed.empty

    pre_tx_sum = (
        completed["fifo_wait_s"]
        + completed["duty_wait_s"]
        + completed["half_duplex_wait_s"]
        + completed["rx_to_tx_transition_s"]
    )
    sojourn_sum = pre_tx_sum + completed["airtime_s"] + completed["tx_to_rx_transition_s"]
    assert completed["time_to_tx_start_s"].to_numpy() == pytest.approx(
        pre_tx_sum.to_numpy(),
        abs=1e-12,
    )
    assert completed["system_sojourn_s"].to_numpy() == pytest.approx(
        sojourn_sum.to_numpy(),
        abs=1e-12,
    )

    relay_metrics = outputs["relay_tx_queue_metrics_df"]
    relay_metrics = relay_metrics[relay_metrics["entity_type"] == "relay"]
    assert (relay_metrics["avg_duty_wait_ms"] > 0).all()
    assert (relay_metrics["worker_occupied_fraction"] <= 1.0).all()
    assert (
        relay_metrics["effective_offered_load_estimate"]
        > relay_metrics["offered_airtime_load_ratio"]
    ).all()
    assert (relay_metrics["accepted_conservation_delta"] == 0).all()
    assert set(lifecycle["status"]).issuperset(
        {
            "COMPLETED",
            "CENSORED_WAITING",
            "CENSORED_IN_SERVICE",
        }
    )


def test_finite_capacity_records_overflow_as_explicit_pdr_drop():
    config = dict(BASE_CONFIG)
    config.update(
        {
            "sim_time": 20,
            "seed": 42,
            "node_period": 1,
            "relay_duty_cycle": 0.01,
            "relay_tx_waiting_queue_capacity_packets": 2,
        }
    )

    summary, _, _, outputs = run_single_simulation(config)
    metrics = outputs["relay_tx_queue_metrics_df"]
    relay_metrics = metrics[metrics["entity_type"] == "relay"]
    failures = outputs["packet_failure_df"]

    overflow_count = int(relay_metrics["overflow_drop_count"].sum())
    failure_count = int((failures["reason"] == "forwarding_queue_overflow").sum())
    assert overflow_count > 0
    assert failure_count == overflow_count
    assert summary["drop_forwarding_queue_overflow_count"] == overflow_count
    assert (relay_metrics["max_waiting_queue_length"] <= 2).all()
    assert (relay_metrics["attempt_conservation_delta"] == 0).all()
    assert (relay_metrics["accepted_conservation_delta"] == 0).all()
    assert summary["pdr_percent"] < 74.302
