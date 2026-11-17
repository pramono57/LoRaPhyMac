"""Canonical simulator defaults and configuration merge helper."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict

from loraphymac.config.validation import validate_config_keys


DEFAULT_CONFIG: Dict[str, object] = {
    # =====================================================
    # Simulation
    # =====================================================
    "area_x": 45000, # 50000,
    "area_y": 15000,
    "sim_time": 3600, # 1jam
    "seed": 42,
    "experiment_profile": "general_multisf_multibw",

    # =====================================================
    # PHY Score Cluster
    # =====================================================
    "routing_assumed_sf": 7,
    "routing_ch_id": 0,
    "terrain_delta_h": 90,
    "routing_hop_penalty_alpha": 2,
    # Historical behavior: rebuild one route from the scenario terrain before
    # traffic, then keep it fixed for the run. ``fixed_reference`` is an
    # explicit opt-in that requires a prebuilt, hash-pinned snapshot.
    "routing_strategy": "terrain_specific",
    "routing_reference_delta_h": None,
    "routing_snapshot_manifest": None,
    "routing_snapshot_manifest_sha256": None,

    # =====================================================
    # Air Interface metadata/control placeholder
    # =====================================================
    "feasible_margin_db": 0, #
    "delta_snr": 3,

    # Receiver-scoped collision/capture model used by Paper P1/P2.
    "collision_mode": "capture",
    "capture_threshold_db": 6.0,
    # Deprecated compatibility input; explicit mode below is authoritative.
    "collision_enforce_sf_match": False,
    "collision_sf_interference_mode": "sir_matrix",
    "collision_sir_matrix_preset": "rakibul_islam_2023_table4",
    "collision_sir_matrix": None,
    "collision_sir_matrix_version": None,
    "collision_noise_floor_dbm": -120.0,
    "collision_destructive": True,

    # =====================================================
    # REGULATORY EIRP LIMIT
    # Indonesia LPWAN 920-923 MHz
    # Permenkomdigi No. 2 Tahun 2025
    # =====================================================
    "node_max_eirp_dbm": 20.0,
    "relay_max_eirp_dbm": 26.0,
    "gw_max_eirp_dbm": 26.0,

    # Feeder / cable / connector loss
    # 0 dB = loss ignored in current simulation model
    "node_feeder_loss_db": 0.0,
    "relay_feeder_loss_db": 0.0,
    "gw_feeder_loss_db": 0.0,



    # =====================================================
    # LoRaPhyMAC Frame Annotation
    # =====================================================
    "frame_enabled": True,
    "frame_validation_enabled": False,
    "frame_version": 0,
    "frame_default_ack_req": False,

    # Application scheduling. The legacy mode deliberately preserves the
    # historical post-TX_DONE traffic clock. The independent FIFO mode is an
    # explicit opt-in used by the four-model V2 comparison only.
    "application_scheduling_mode": "legacy_post_tx_done_v1",
    "application_rng_policy": "legacy_global_random_v1",
    "application_queue_policy": "none",
    "application_horizon_policy": "physical_tx_offer_v1",
    "application_initial_delay_max_s": 30.0,


    # =====================================================
    # APC DATA_DOWN physical control
    # =====================================================
    "apc_downlink_mode": "physical_all",   #  "physical_all"

    # APC_REQ length carried in FOpts.
    # Total APC frame:
    # 12-byte fixed overhead + 4-byte FOpts = 16 bytes.
    "apc_fopts_size": 4,
    "apc_downlink_delay_s": 1.0,  # delay before the controller sends DATA_DOWN
    "apc_retry_timeout_s": 5.0,

    # Optional APC downlink PTx override
    # None = use relay/gateway forwarding PTx
    "apc_downlink_ptx_dbm": None,
    "apc_downlink_node_ptx_dbm": None,
    "apc_downlink_relay_ptx_dbm": None,

    # =====================================================
    # End Device physical downlink RX window
    # =====================================================
    "node_rx_window_enabled": True,
    # Opt-in shared receiver-wide PHY collision evaluation for single-channel
    # NodeRX. False preserves historical FF/AA/DBPC experiment semantics.
    "node_rx_collision_enabled": False,
    "node_rx1_delay_s": 1.0,
    "node_rx_window_duration_s": 1,


    # =====================================================
    # Output control
    # =====================================================
    "output_dir": "artifacts/experiments/single_run_exploration/default",
    # Missing profile configurations resolve to detailed to preserve the
    # historical recording and output behavior.
    "output_profile": "detailed",
    "export_summary": False,
    "export_results": False,
    "export_debug": False,
    "export_internal": False,

    "print_summary": False,
    "print_detail": False,
    "print_debug": False,
    "print_internal": False,
    "debug_preview_rows": 10,

    "plot_routing": False,
    "include_mcu_energy": False,
    # ``legacy`` preserves the 300-run Paper 1 energy accounting.
    # ``corrected`` applies manuscript Eq. (10) to mutually exclusive
    # half-duplex relay/gateway TX, transition, and continuous-RX durations.
    "energy_accounting_mode": "legacy",
    "verbose": False,

    # Optional pre-traffic PTx policy. The legacy policy is a no-op dispatcher
    # and preserves FF/AF/FA/AA exactly. Calibration fields are required only
    # for the independent Islam-DBPC experiment.
    "tx_power_policy_id": "legacy",
    "tx_power_model_id": None,
    "tx_power_calibration_id": None,
    "tx_power_calibration_manifest": None,
    "tx_power_calibration_manifest_sha256": None,
    "tx_power_calibration_margin_db": 0,

    # =====================================================
    # Node configuration
    # =====================================================
    "node_num": 100,  # 45,
    "node_sf_list": [7],
    "node_bw": 125000,
    "node_ptx_init": None,  # None = start from maximum regulatory-compliant PTx
    "node_payload_size": 50,
    "node_period": 60,  # Base interval between transmissions (s)
    "node_jitter": 2,  # Random variation around the period (±s)
    "node_ant_height": 2 ,
    "node_ant_gain": 2.15,

    # Node PTX mode: fixed / adaptive
    "node_ptx_mode": "adaptive",
    "node_target_margin_db": 6.0,


    # =====================================================
    # Relay configuration
    # =====================================================
    "relay_num": 2,  # 2,
    "relay_sf": 7,
    "relay_bw_hz": 125000,
    "relay_ptx_init": None,  # None = start from maximum regulatory-compliant PTx
    "relay_ant_height": 25,
    "relay_ant_gain": 5.8,
    "relay_tx_ch_ids": [1],  # TX channels; for example, [1, 2, 3, 4]

    # RelayTX forwarding queue. None preserves the historical unbounded FIFO.
    # Finite values count waiting packets only; in-service packets are tracked
    # separately. Overflow is non-blocking so RelayRX is never stalled.
    "relay_tx_waiting_queue_capacity_packets": None,
    "gateway_tx_waiting_queue_capacity_packets": None,
    "tx_queue_overflow_policy": "drop_tail",

    # Relay duty cycle:
    # 0.01 -> regulatory 1%
    # 0.10 -> relaxed 10%
    # None -> unlimited
    "relay_duty_cycle": None,

    # =====================================================
    # Relay half-duplex scheduling
    # =====================================================

    # Maximum polling interval for active RX sessions.
    # This is not a limit on total waiting time.
    "rx_idle_poll_interval_ms": 2.0,

    # Maximum time RelayTX gives full priority to RX.
    # Once this limit is reached, tx_pending rejects new RX sessions,
    # while already-active RX sessions are still allowed to finish.
    "max_rx_idle_wait_ms": 3000.0,


    # Relay PTX mode: fixed / adaptive
    "relay_ptx_mode": "adaptive",
    "relay_target_margin_db": 6.0,


    # =====================================================
    # Gateway configuration
    # =====================================================
    "gw_ant_height": 25,
    "gw_ant_gain": 5.8,

    # =====================================================
    # Batch/manual identity
    # =====================================================
    "run_id": None,
}


# =========================================================
# HELPER: MERGE CONFIG
# =========================================================
def build_config(config: Dict[str, object] | None = None) -> Dict[str, object]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if config is not None:
        validate_config_keys(config, DEFAULT_CONFIG)
        cfg.update(deepcopy(dict(config)))
        # Before the three-workbook output contract, ``export_results`` wrote
        # the summary workbook. Preserve that observable behavior for callers
        # that have not adopted the explicit ``export_summary`` switch yet.
        if "export_summary" not in config and bool(config.get("export_results", False)):
            cfg["export_summary"] = True
        if (
            "collision_enforce_sf_match" in config
            and "collision_sf_interference_mode" not in config
        ):
            cfg["collision_sf_interference_mode"] = (
                "ideal_orthogonal"
                if bool(config["collision_enforce_sf_match"])
                else "common_threshold"
            )
    return cfg
