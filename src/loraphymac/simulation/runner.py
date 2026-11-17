"""Internal event runner for a single LoRaPhyMAC simulation.

Use :mod:`loraphymac.simulator` from application and experiment code. The
historical :mod:`loraphymac.engine` module is retained as a compatibility
facade.
"""

# =========================================================
# LoRaPhyMAC event runner
# Physical DATA_DOWN / DATA_DOWN_FWD +
# Relay APC RX cleanup, APC TX index, and timing validation
#
# ---------------------------------------------------------
# Based on latest main_simulation.py
#
# Features:
# - Function-based for batch experiment
# - Node PTX mode: FIXED / ADAPTIVE
# - Relay PTX mode: FIXED / ADAPTIVE
# - Gateway PTX mode: FIXED
# - Relay chain link quality for colored relay links
# - visual_links_df = selected_next_hop_links_df + relay_chain_links_df
# - Compatible with run_experiments
# =========================================================

from __future__ import annotations

import os
import pandas as pd
import random

from typing import Dict, Tuple

from loraphymac._version import SIMULATOR_VERSION
from loraphymac.config.defaults import DEFAULT_CONFIG, build_config
from loraphymac.utils.visualizer import plot_routing
from loraphymac.hardware.hardware import (
    validate_ptx,
    validate_initial_ptx,
)

from loraphymac.runtime.node_tx import (
    INDEPENDENT_APPLICATION_SCHEDULING_MODE,
    NodeTX,
    stable_application_rng_seed,
)
from loraphymac.runtime.node_rx import NodeRX
from loraphymac.runtime.relay_rx import RelayRX
from loraphymac.runtime.relay_tx import RelayTX
from loraphymac.runtime.demodulator_pool import DemodulatorPool
from loraphymac.runtime.apc_power import dbm_to_tx_power_index
from loraphymac.runtime.tx_power_policy import apply_tx_power_policy
from loraphymac.runtime.wang_sink_decision import (
    install_wang_gateway_scheduler,
    wang_runtime_enabled,
)
from loraphymac.runtime.wang_power_commands import WangGatewayCommandCoordinator
from loraphymac.mac.frame_defs import frame_size_bytes
from loraphymac.metrics.statistics import Statistics
from loraphymac.metrics.energy_metrics import energy_accounting_model_version
from loraphymac.simulation.builders import (
    build_infrastructure,
    build_routing,
    build_topology,
)
from loraphymac.simulation.exporting import export_simulation_outputs
from loraphymac.simulation.output_collection import collect_simulation_outputs
from loraphymac.simulation.reporting import report_simulation_outputs
from loraphymac.simulation.state import (
    initialize_gateway_ptx_state,
    initialize_node_ptx_state,
    initialize_relay_ptx_state,
)
from loraphymac.experiments.common.contract_registry import experiment_contract_version
from loraphymac.config.validation import (
    CONFIGURATION_VALIDATION_VERSION,
    validate_configuration,
)

# =========================================================
# CORE FUNCTION
# =========================================================
def run_single_simulation(
    config: Dict[str, object] | None = None
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run one simulation scenario.

    Returns
    -------
    core_summary_row : dict
        One core-summary row for run_experiments.

    ptx_history_df : pd.DataFrame
        Adaptive PTx change history.

    margin_history_df : pd.DataFrame
        Link-margin history.

    all_outputs : dict[str, pd.DataFrame]
        All important output matrices.
    """

    cfg = build_config(config)
    validate_configuration(cfg, known_keys=DEFAULT_CONFIG)

    infrastructure = build_infrastructure(cfg)
    env = infrastructure.env
    channel_plan = infrastructure.channel_plan
    air = infrastructure.air
    model = infrastructure.model
    collision_engine = infrastructure.collision_engine
    router = infrastructure.router

    # ============================================================
    # 1. GENERATE TOPOLOGY AND APPLY REGULATORY EIRP LIMITS
    # ============================================================
    topology = build_topology(cfg, infrastructure)
    nodes = topology.nodes
    relays = topology.relays
    gateway = topology.gateway



    # ============================================================
    # 2. COMPUTE AND VALIDATE ROUTING
    # ============================================================
    routing = build_routing(cfg, infrastructure, topology)
    routing_map = routing.routing_map
    all_nodes = routing.all_nodes
    node_map = routing.node_map
    visual_links_df = routing.visual_links_df

    # ------------------------------------------------------------
    # RUN METADATA: DC mode
    # ------------------------------------------------------------
    relay_duty_cycle = cfg["relay_duty_cycle"]

    if relay_duty_cycle is None:
        relay_dc_mode = "unlimited"
    elif abs(float(relay_duty_cycle) - 0.01) < 1e-9:
        relay_dc_mode = "regulatory"
    elif abs(float(relay_duty_cycle) - 0.10) < 1e-9:
        relay_dc_mode = "relaxed"
    else:
        relay_dc_mode = "custom"

    if cfg["run_id"] is None:
        run_id = (
            f"dh{cfg['terrain_delta_h']}_"
            f"{str(cfg['node_ptx_mode'])[0].upper()}"
            f"{str(cfg['relay_ptx_mode'])[0].upper()}_"
            f"dc{relay_dc_mode}_"
            f"s{cfg['seed']}"
        )
    else:
        run_id = str(cfg["run_id"])

    # ============================================================
    # PLOT MULTI-HOP ROUTES
    # ============================================================
    if bool(cfg["plot_routing"]):
        routing_fig_path = None

        if cfg.get("output_dir") is not None:
            os.makedirs(
                str(cfg["output_dir"]),
                exist_ok=True,
            )

            routing_fig_path = os.path.join(
                str(cfg["output_dir"]),
                f"{run_id}_routing_topology.jpg",
            )

        plot_routing(
            nodes=nodes,
            relays=relays,
            gateway=gateway,
            routing_map=routing_map,
            router=router,
            node_map=node_map,
            selected_links_df=visual_links_df,
            feasible_margin_db=cfg["feasible_margin_db"],
            target_margin_db=cfg["node_target_margin_db"],
            show_margin_label=False,
            title=None,
            save_path=routing_fig_path,
            show=True,
        )

    # ============================================================
    # INITIALIZE SIMULATION MODULES
    # ============================================================
    for n in all_nodes:
        n.network_nodes = all_nodes
        n.network_nodes_map = node_map

    stats = Statistics(output_profile=str(cfg["output_profile"]))



    # ============================================================
    # 3. NODE PTX MODE STATE
    # ============================================================
    for i, node in enumerate(nodes):
        initialize_node_ptx_state(node, cfg, i)
        node.rx_window_enabled = bool(cfg.get("node_rx_window_enabled", True))
        node.rx_collision_enabled = bool(
            cfg.get("node_rx_collision_enabled", False)
        )
        node.rx1_delay_s = float(cfg.get("node_rx1_delay_s", 1.0))
        node.rx_window_duration_s = float(cfg.get("node_rx_window_duration_s", 1.5))
        node.rx_windows = []
        node.tx_busy_until = 0.0
        node.is_transmitting = False


    # ============================================================
    # 4. RELAY PTX MODE STATE
    # ============================================================
    for relay in relays:
        initialize_relay_ptx_state(relay, cfg)

    # ============================================================
    # 5. GATEWAY PTX MODE STATE
    # ============================================================
    initialize_gateway_ptx_state(gateway, cfg)

    # ============================================================
    # 5B. OPTIONAL PRE-TRAFFIC PTX POLICY
    # ============================================================
    # Legacy is a strict no-op. Islam-DBPC applies one frozen, distance-based
    # choice after routing and before any SimPy traffic process is registered.
    tx_power_policy_application = apply_tx_power_policy(
        config=cfg,
        nodes=nodes,
        relays=relays,
        gateway=gateway,
        routing_map=routing_map,
        node_map=node_map,
    )

    # ============================================================
    # 5C. WANG A2-ONLY GATEWAY OBSERVATION/SCHEDULER
    # ============================================================
    # Exact model identity is the only enablement path. Active transmitters
    # come from configured routes; expected PTx uses the resolved hardware/EIRP
    # bound at initialization and is never read back during a decision.
    wang_sink_engine = None
    for participant in [*nodes, *relays, gateway]:
        participant.wang_snr_reporting_enabled = False
    wang_runtime_metadata = {}
    if wang_runtime_enabled(cfg):
        wang_sink_engine = install_wang_gateway_scheduler(
            env=env,
            gateway=gateway,
            nodes=nodes,
            relays=relays,
            routing_map=routing_map,
            simulation_horizon_s=float(cfg["sim_time"]),
        )
        wang_runtime_metadata = {
            "wang_sink_decision_enabled": True,
            "wang_observation_window_s": wang_sink_engine.window_s,
            "wang_delivery_grace_s": wang_sink_engine.delivery_grace_s,
            "wang_required_coverage_fraction": 1.0,
            "wang_active_transmitter_count": (
                wang_sink_engine.active_transmitter_count
            ),
            "wang_expected_ptx_provenance": (
                "resolved_hardware_eirp_max_at_initialization"
            ),
        }


    # ------------------------------------------------------------
    # APC physical DATA_DOWN configuration
    # ------------------------------------------------------------
    for entity in list(nodes) + list(relays) + [gateway]:

        entity.apc_tx_power_index = dbm_to_tx_power_index(
            entity,
            int(
                getattr(
                    entity,
                    "default_ptx",
                    getattr(entity, "fixed_ptx", entity.ptx_max),
                )
            ),
        )

        entity.apc_pending_tx_index = None
        entity.apc_pending_ptx = None
        entity.apc_pending_from = None
        entity.apc_pending_time = None
        entity.apc_pending_ack = False

        # --------------------------------------------------------
        # Generic APC downlink override
        # None = no generic override
        # --------------------------------------------------------
        generic_downlink_ptx = cfg.get(
            "apc_downlink_ptx_dbm",
            None,
        )

        if generic_downlink_ptx is not None:
            entity.apc_downlink_ptx_dbm = validate_ptx(
                entity,
                generic_downlink_ptx,
                f"{entity.id}:APC_DOWNLINK",
            )

        # --------------------------------------------------------
        # DATA_DOWN -> End Node
        #
        # None means:
        # use transmitter's maximum regulatory-compliant PTx.
        #
        # Example:
        #   SX1302 + Gant 5.8 dBi -> ptx_max = 20 dBm
        #   SX1302 + Gant 10  dBi -> ptx_max = 16 dBm
        # --------------------------------------------------------
        node_downlink_ptx = cfg.get(
            "apc_downlink_node_ptx_dbm",
            None,
        )

        if node_downlink_ptx is None:
            entity.apc_downlink_node_ptx_dbm = int(
                entity.ptx_max
            )
        else:
            entity.apc_downlink_node_ptx_dbm = validate_ptx(
                entity,
                node_downlink_ptx,
                f"{entity.id}:DATA_DOWN",
            )

        # --------------------------------------------------------
        # DATA_DOWN_FWD -> preceding relay
        #
        # None means:
        # do not force a fixed value here.
        # RelayTX keeps its relay-side forwarding-power policy.
        # --------------------------------------------------------
        relay_downlink_ptx = cfg.get(
            "apc_downlink_relay_ptx_dbm",
            None,
        )

        if relay_downlink_ptx is not None:
            entity.apc_downlink_relay_ptx_dbm = validate_ptx(
                entity,
                relay_downlink_ptx,
                f"{entity.id}:DATA_DOWN_FWD",
            )

        entity.apc_downlink_mode = str(
            cfg.get(
                "apc_downlink_mode",
                "physical_all",
            )
        ).lower().strip()

        entity.apc_fopts_size = int(
            cfg.get("apc_fopts_size", 4)
        )

        entity.apc_downlink_delay_s = float(
            cfg.get("apc_downlink_delay_s", 1.0)
        )

        entity.apc_retry_timeout_s = float(
            cfg.get("apc_retry_timeout_s", 5.0)
        )

    # ------------------------------------------------------------
    # Validate resolved initial PTx
    # ------------------------------------------------------------
    node_ref = nodes[0] if len(nodes) > 0 else None
    relay_ref = relays[0] if len(relays) > 0 else None

    if node_ref is not None:
        validate_initial_ptx(
            node_ref,
            node_ref.default_ptx,
            "NODE",
        )

    if relay_ref is not None:
        validate_initial_ptx(
            relay_ref,
            relay_ref.default_ptx,
            "RELAY",
        )

    validate_initial_ptx(
        gateway,
        gateway.default_ptx,
        "GATEWAY",
    )

    # ------------------------------------------------------------
    # RUN METADATA
    # ------------------------------------------------------------
    stats.set_run_metadata(
        run_id=run_id,
        simulator_version=SIMULATOR_VERSION,
        seed=cfg["seed"],
        experiment_profile=cfg["experiment_profile"],
        experiment_contract_version=experiment_contract_version(
            cfg["experiment_profile"]
        ),
        configuration_validation_version=CONFIGURATION_VALIDATION_VERSION,
        terrain_delta_h=cfg["terrain_delta_h"],
        routing_strategy=routing.routing_strategy,
        route_construction_delta_h=routing.route_construction_delta_h,
        physical_terrain_delta_h=routing.physical_terrain_delta_h,
        routing_snapshot_id=routing.routing_snapshot_id,
        routing_snapshot_sha256=routing.routing_snapshot_sha256,
        routing_snapshot_manifest_sha256=(
            routing.routing_snapshot_manifest_sha256
        ),
        topology_sha256=routing.topology_sha256,
        selected_parent_routing_sha256=(
            routing.selected_parent_routing_sha256
        ),

        # PTX adaptive mode
        node_ptx_mode=cfg["node_ptx_mode"],
        relay_ptx_mode=cfg["relay_ptx_mode"],

        # Energy accounting provenance. Legacy remains the default so the
        # historical 300-run outputs cannot be mistaken for Eq. (10) results.
        energy_accounting_mode=cfg["energy_accounting_mode"],
        energy_accounting_model_version=energy_accounting_model_version(
            cfg["energy_accounting_mode"]
        ),
        feedback_additional_loss_probability=float(
            cfg.get("feedback_additional_loss_probability", 0.0)
        ),

        # Topology
        node_num=cfg["node_num"],
        relay_num=cfg["relay_num"],
        sim_time=cfg["sim_time"],

        # PHY configuration
        routing_ch_id=cfg["routing_ch_id"],
        routing_freq_mhz=channel_plan.get_freq(cfg["routing_ch_id"]),
        node_bw=cfg["node_bw"],
        relay_bw_hz=cfg["relay_bw_hz"],
        routing_hop_penalty_alpha=cfg["routing_hop_penalty_alpha"],

        # Receiver collision/capture reproducibility contract
        collision_mode=cfg["collision_mode"],
        collision_model_version=collision_engine.model_version,
        collision_spectral_model_version=collision_engine.SPECTRAL_MODEL_VERSION,
        capture_threshold_db=cfg["capture_threshold_db"],
        collision_enforce_sf_match=collision_engine.enforce_sf_match,
        collision_sf_interference_mode=collision_engine.sf_interference_mode,
        collision_sir_matrix_version=(
            collision_engine.sir_matrix_version
            if collision_engine.sf_interference_mode == "sir_matrix"
            else None
        ),
        collision_noise_floor_dbm=cfg["collision_noise_floor_dbm"],
        collision_destructive=cfg["collision_destructive"],
        collision_receiver_domain="all_registered_receivers_v1",
        demodulator_timing_model="rx_start_to_rx_end_v2",
        packet_fate_metrics_version="1.0",
        packet_observation_model="hard_horizon_right_censor_v1",

        # Initial PTx
        node_ptx_init=(
            node_ref.default_ptx
            if node_ref is not None else None
        ),

        relay_ptx_init=(
            relay_ref.default_ptx
            if relay_ref is not None else None
        ),

        gateway_ptx_init=gateway.default_ptx,

        # Hardware type
        node_hw_type=node_ref.hw.type if node_ref is not None else None,
        relay_hw_type=relay_ref.hw.type if relay_ref is not None else None,
        gateway_hw_type=gateway.hw.type,

        # Hardware-based PTx range
        node_ptx_min=node_ref.ptx_min if node_ref is not None else None,
        node_ptx_max=node_ref.ptx_max if node_ref is not None else None,

        relay_ptx_min=relay_ref.ptx_min if relay_ref is not None else None,
        relay_ptx_max=relay_ref.ptx_max if relay_ref is not None else None,

        gateway_ptx_min=gateway.ptx_min,
        gateway_ptx_max=gateway.ptx_max,

        # Adaptive PTx control parameters
        node_target_margin_db=cfg["node_target_margin_db"],
        relay_target_margin_db=cfg["relay_target_margin_db"],


        # Relay duty cycle
        relay_duty_cycle=relay_duty_cycle,
        relay_dc_mode=relay_dc_mode,

        # Full-duplex / half-duplex configuration
        rx_idle_poll_interval_ms=cfg["rx_idle_poll_interval_ms"],


        max_rx_idle_wait_ms=cfg[
            "max_rx_idle_wait_ms"
        ],


        relay_tx_full_duplex=getattr(relay_ref.hw, "tx_full_duplex", None)
        if relay_ref is not None else None,
        gateway_tx_full_duplex=getattr(gateway.hw, "tx_full_duplex", None),

        # Multi-PHY / multi-channel TX configuration
        relay_tx_capacity_c=getattr(relay_ref.hw, "tx_phy_radio_count", None)
        if relay_ref is not None else None,
        gateway_tx_capacity_c=getattr(gateway.hw, "tx_phy_radio_count", None),
        relay_tx_ch_ids=str(cfg["relay_tx_ch_ids"]),
        queue_metrics_version="2.0",
        tx_queue_discipline="fifo",
        relay_tx_waiting_queue_capacity_packets=cfg[
            "relay_tx_waiting_queue_capacity_packets"
        ],
        gateway_tx_waiting_queue_capacity_packets=cfg[
            "gateway_tx_waiting_queue_capacity_packets"
        ],
        tx_queue_overflow_policy=cfg["tx_queue_overflow_policy"],

        # LoRaPhyMAC frame annotation
        frame_enabled=cfg["frame_enabled"],
        frame_validation_enabled=cfg["frame_validation_enabled"],
        frame_version=cfg["frame_version"],
        frame_default_ack_req=cfg["frame_default_ack_req"],
        application_scheduling_mode=cfg["application_scheduling_mode"],
        application_rng_policy=cfg["application_rng_policy"],
        application_queue_policy=cfg["application_queue_policy"],
        application_horizon_policy=cfg["application_horizon_policy"],
        application_initial_delay_max_s=cfg[
            "application_initial_delay_max_s"
        ],

        # APC physical DATA_DOWN
        apc_downlink_mode=cfg["apc_downlink_mode"],

        # Application payload carried by DATA_UP.
        node_app_payload_size_bytes=int(
            cfg["node_payload_size"]
        ),
        node_period_s=cfg["node_period"],
        node_jitter_s=cfg["node_jitter"],
        node_sf_list=str(list(cfg["node_sf_list"])),
        relay_sf=cfg["relay_sf"],
        routing_assumed_sf=cfg["routing_assumed_sf"],
        area_x_m=cfg["area_x"],
        area_y_m=cfg["area_y"],
        feasible_margin_db=cfg["feasible_margin_db"],

        # Total LoRaPhyMAC DATA_UP frame.
        node_frame_size_bytes=frame_size_bytes(
            app_payload_size=int(
                cfg["node_payload_size"]
            ),
            fopts_len=0,
        ),

        # APC_REQ length inside FOpts.
        apc_fopts_size=int(
            cfg["apc_fopts_size"]
        ),

        # Total DATA_DOWN / DATA_DOWN_FWD frame.
        apc_downlink_frame_size_bytes=frame_size_bytes(
            app_payload_size=0,
            fopts_len=int(
                cfg["apc_fopts_size"]
            ),
        ),

        apc_downlink_delay_s=cfg["apc_downlink_delay_s"],
        apc_retry_timeout_s=cfg["apc_retry_timeout_s"],

        # End device RX window for physical DATA_DOWN
        node_rx_window_enabled=cfg["node_rx_window_enabled"],
        node_rx1_delay_s=cfg["node_rx1_delay_s"],
        node_rx_window_duration_s=cfg["node_rx_window_duration_s"],
        **dict(tx_power_policy_application.run_metadata or {}),
        **wang_runtime_metadata,

    )

    # Register the operational baseline TPI only after Statistics has its run
    # metadata, so the opt-in profile gate is effective. This is bookkeeping
    # before traffic processes are registered: no SimPy event, RNG draw, radio
    # state change, or scheduling action is introduced.
    if cfg.get("experiment_profile") == "apc_feedback_robustness_sweep_v1":
        for node in nodes:
            stats.register_feedback_tpi_transmitter(
                node.id, "node", getattr(node, "apc_tx_power_index", None)
            )
        for relay in relays:
            stats.register_feedback_tpi_transmitter(
                relay.id, "relay", getattr(relay, "apc_tx_power_index", None)
            )

    for n in all_nodes:
        n.stats = stats

    # ============================================================
    # 6. TX GENERATORS: END DEVICES
    # ============================================================
    node_tx_generators = []
    for node in nodes:
        independent_application_schedule = (
            cfg["application_scheduling_mode"]
            == INDEPENDENT_APPLICATION_SCHEDULING_MODE
        )
        initial_delay = (
            None
            if independent_application_schedule
            else random.uniform(0, cfg["application_initial_delay_max_s"])
        )
        application_rng_seed = (
            stable_application_rng_seed(cfg["seed"], node.id)
            if independent_application_schedule
            else None
        )
        txg = NodeTX(
            env=env,
            node=node,
            routing_map=routing_map,
            air_interface=air,
            model=model,
            # Application payload only.
            # NodeTX adds LoRaPhyMAC overhead internally.
            payload_size=cfg["node_payload_size"],
            period=cfg["node_period"],
            initial_delay=initial_delay,
            jitter=cfg["node_jitter"],
            terrain_delta_h=cfg["terrain_delta_h"],

            # LoRaPhyMAC frame annotation control
            frame_enabled=cfg["frame_enabled"],
            frame_version=cfg["frame_version"],
            frame_default_ack_req=cfg["frame_default_ack_req"],
            application_scheduling_mode=cfg["application_scheduling_mode"],
            application_rng_seed=application_rng_seed,
            application_initial_delay_max_s=cfg[
                "application_initial_delay_max_s"
            ],
        )

        node_tx_generators.append(txg)
        env.process(txg.run())

    # ============================================================
    # DEBUG PRINT MODE
    # ============================================================
    if bool(cfg["print_debug"]):
        print("\n=== NODE PTX MODE ===")
        for node in nodes:
            print(
                f"{node.id}: "
                f"mode={node.ptx_mode}, "
                f"ptx_init={node.default_ptx} dBm, "
                f"target_margin={node.target_margin_db} dB, "
                #f"hyst={node.margin_hyst_db} dB"
            )

        print("\n=== FRAME SIZE CONFIGURATION ===")

        data_up_frame_size = frame_size_bytes(
            app_payload_size=cfg["node_payload_size"],
            fopts_len=0,
        )
        print(
            f"DATA_UP: "
            f"app_payload={cfg['node_payload_size']} B, "
            f"fopts=0 B, "
            f"frame_size={data_up_frame_size} B"
        )

        data_down_frame_size = frame_size_bytes(
            app_payload_size=0,
            fopts_len=cfg["apc_fopts_size"],
        )
        print(
            f"DATA_DOWN/APC: "
            f"app_payload=0 B, "
            f"fopts={cfg['apc_fopts_size']} B, "
            f"frame_size={data_down_frame_size} B"
        )



        print("\n=== RELAY PTX MODE ===")

        for relay in relays:
            print(
                f"{relay.id}: "
                f"mode={relay.ptx_mode}, "
                f"ptx_init={relay.default_ptx} dBm, "
                f"target_margin={relay.target_margin_db} dB, "
                # f"hyst={relay.margin_hyst_db} dB, "
                f"tx_full_duplex="
                f"{getattr(relay.hw, 'tx_full_duplex', False)}, "
                f"tx_phy_radio_count="
                f"{getattr(relay.hw, 'tx_phy_radio_count', 1)}, "
                f"rx_idle_poll_interval_ms="
                f"{cfg['rx_idle_poll_interval_ms']}"
                f", max_rx_idle_wait_ms="
                f"{cfg['max_rx_idle_wait_ms']}"
            )

        print("\n=== GATEWAY PTX MODE ===")

        print(
            f"{gateway.id}: "
            f"mode={gateway.ptx_mode}, "
            f"ptx_init={gateway.default_ptx} dBm, "
            f"tx_full_duplex="
            f"{getattr(gateway.hw, 'tx_full_duplex', False)}, "
            f"tx_phy_radio_count="
            f"{getattr(gateway.hw, 'tx_phy_radio_count', 1)}, "
            f"rx_idle_poll_interval_ms="
            f"{cfg['rx_idle_poll_interval_ms']}"
            f", max_rx_idle_wait_ms="
            f"{cfg['max_rx_idle_wait_ms']}"
        )

    # ============================================================
    # 7. RELAY & GATEWAY TX
    # ============================================================
    relay_tx = {}

    for r in relays + [gateway]:
        waiting_queue_capacity = (
            cfg["gateway_tx_waiting_queue_capacity_packets"]
            if r.id == gateway.id
            else cfg["relay_tx_waiting_queue_capacity_packets"]
        )
        tx = RelayTX(
            env=env,
            relay=r,
            air_interface=air,
            statistics=stats,
            relay_sf=cfg["relay_sf"],
            relay_bw_hz=cfg["relay_bw_hz"],
            relay_ch_ids=cfg["relay_tx_ch_ids"],
            channel_plan=channel_plan,
            relay_duty_cycle=relay_duty_cycle,
            rx_idle_poll_interval_ms=cfg["rx_idle_poll_interval_ms"],
            max_rx_idle_wait_ms=cfg["max_rx_idle_wait_ms"],
            # LoRaPhyMAC frame annotation control
            frame_enabled=cfg["frame_enabled"],
            tx_waiting_queue_capacity_packets=waiting_queue_capacity,
            tx_queue_overflow_policy=cfg["tx_queue_overflow_policy"],

        )

        relay_tx[r.id] = tx

        # TX PHY follows hardware capacity
        for worker_id in range(getattr(tx, "tx_capacity_c", 1)):
            env.process(tx.run(worker_id=worker_id))

        r.tx = tx

    # ============================================================
    # 8. RELAY RX
    # ============================================================
    for relay in relays:
        demod = DemodulatorPool(
            env,
            pool_a_size=relay.hw.demod_pool_A,
            pool_b_size=relay.hw.demod_pool_B,
            receiver_id=relay.id,
            statistics=stats,
        )

        relay.rx = RelayRX(
            env=env,
            relay=relay,
            model=model,
            air_interface=air,
            terrain_delta_h=cfg["terrain_delta_h"],
            collision_engine=collision_engine,
            demodulator_pool=demod,
            relay_tx=relay_tx[relay.id],
            statistics=stats,
        )

        air.register_rx(relay)

    # ============================================================
    # 8B. END DEVICE RX for physical DATA_DOWN APC
    # ============================================================
    for node in nodes:
        node_demod = None
        if node.rx_collision_enabled:
            # SX1276 end devices expose one currently tuned LoRa demodulator.
            node_demod = DemodulatorPool(
                env,
                pool_a_size=0,
                pool_b_size=1,
                receiver_id=node.id,
                statistics=stats,
            )
        node.rx = NodeRX(
            env=env,
            node=node,
            air_interface=air,
            model=model,
            terrain_delta_h=cfg["terrain_delta_h"],
            statistics=stats,
            collision_engine=(collision_engine if node.rx_collision_enabled else None),
            demodulator_pool=node_demod,
            collision_enabled=node.rx_collision_enabled,
        )
        air.register_rx(node)


    # ============================================================
    # 9. GATEWAY RX
    # ============================================================
    demod_gw = DemodulatorPool(
        env,
        pool_a_size=gateway.hw.demod_pool_A,
        pool_b_size=gateway.hw.demod_pool_B,
        receiver_id=gateway.id,
        statistics=stats,
    )

    gateway.rx = RelayRX(
        env=env,
        relay=gateway,
        model=model,
        air_interface=air,
        terrain_delta_h=cfg["terrain_delta_h"],
        collision_engine=collision_engine,
        demodulator_pool=demod_gw,
        relay_tx=relay_tx[gateway.id],
        statistics=stats,
    )

    air.register_rx(gateway)

    # Physical Wang commands are bound only after every RelayTX/RelayRX object
    # exists.  The decision scheduler itself remains main-gateway-only.
    if wang_sink_engine is not None:
        wang_command_coordinator = WangGatewayCommandCoordinator(
            env=env,
            gateway=gateway,
            active_transmitters=wang_sink_engine.active_transmitters,
            routing_map=routing_map,
            relay_tx=relay_tx,
            air_interface=air,
            statistics=stats,
        )
        gateway.wang_command_coordinator = wang_command_coordinator
        wang_sink_engine.attach_command_coordinator(wang_command_coordinator)

    # ============================================================
    # 10. RUN SIMULATION
    # ============================================================
    if bool(cfg["print_summary"]) or bool(cfg["verbose"]):
        print(f"\n🚀 Simulation started | RUN_ID={run_id}")

    env.run(until=cfg["sim_time"])

    if wang_sink_engine is not None:
        stats.run_metadata.update(
            {
                "wang_closed_window_count": len(wang_sink_engine.decisions),
                "wang_no_decision_count": wang_sink_engine.no_decision_count,
            }
        )

    # Snapshot unfinished waiting/in-service visits at the hard observation
    # boundary. This records censoring only and schedules no new SimPy event.
    for tx in relay_tx.values():
        tx.finalize_observation(float(cfg["sim_time"]))
    application_queue_records = [
        record
        for txg in node_tx_generators
        if (record := txg.finalize_observation(float(cfg["sim_time"])))
        is not None
    ]
    if application_queue_records:
        stats.run_metadata.update(
            {
                "application_generated_count": sum(
                    int(row["generated_count"])
                    for row in application_queue_records
                ),
                "application_dequeued_count": sum(
                    int(row["dequeued_count"])
                    for row in application_queue_records
                ),
                "application_queue_max_backlog_packets": max(
                    int(row["max_backlog_packets"])
                    for row in application_queue_records
                ),
                "application_queue_waiting_at_horizon_count": sum(
                    int(row["waiting_at_horizon_count"])
                    for row in application_queue_records
                ),
                "application_queue_in_service_at_horizon_count": sum(
                    int(row["in_service_at_horizon_count"])
                    for row in application_queue_records
                ),
                "application_queue_outstanding_at_horizon_count": sum(
                    int(row["outstanding_at_horizon_count"])
                    for row in application_queue_records
                ),
            }
        )
    stats.finalize_packet_observation(float(cfg["sim_time"]))
    stats.finalize_feedback_robustness(float(cfg["sim_time"]))

    if bool(cfg["print_summary"]) or bool(cfg["verbose"]):
        print("✅ Simulation finished")

    # ============================================================
    # 11. COLLECT, REPORT, AND EXPORT OUTPUTS
    # ============================================================
    collected_outputs = collect_simulation_outputs(
        statistics=stats,
        routing=routing,
    )
    if wang_sink_engine is not None:
        wang_decision_df = pd.DataFrame(wang_sink_engine.decision_rows())
        collected_outputs.tables["wang_sink_decision_df"] = wang_decision_df
        collected_outputs.result_bundle[
            "R21_wang_sink_decisions"
        ] = wang_decision_df
    report_simulation_outputs(
        config=cfg,
        statistics=stats,
        outputs=collected_outputs,
    )
    export_simulation_outputs(
        config=cfg,
        run_id=run_id,
        statistics=stats,
        outputs=collected_outputs,
    )
    return collected_outputs.to_legacy_tuple()


def main() -> None:
    """Run the historical verbose manual scenario."""
    run_single_simulation(
        {
            "verbose": True,
            "print_summary": True,
            "print_detail": True,
            "print_debug": True,
            "print_internal": True,
            "plot_routing": True,
            "export_summary": True,
            "export_results": True,
            "export_debug": True,
        }
    )


if __name__ == "__main__":
    main()
