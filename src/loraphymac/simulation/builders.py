"""Construction steps for simulation infrastructure and topology.

These functions preserve the historical call and random-seeding order.  They
only construct objects; event-process registration remains in the runner.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import simpy

from loraphymac.hardware.hardware import (
    ESP32,
    SX1276,
    SX1302,
    apply_eirp_ptx_bounds,
)
from loraphymac.mac.collision_engine import CollisionEngine
from loraphymac.phy.air_interface import AirInterface
from loraphymac.phy.channel_plan import AS923_2
from loraphymac.phy.propagation import ITM
from loraphymac.routing.routing_engine import RoutingEngine
from loraphymac.routing.snapshot import (
    ROUTING_STRATEGY_FIXED_REFERENCE,
    ROUTING_STRATEGY_TERRAIN_SPECIFIC,
    load_routing_snapshot_bundle,
    routing_sha256,
    topology_sha256,
)
from loraphymac.simulation.context import (
    RoutingArtifacts,
    SimulationInfrastructure,
    SimulationTopology,
)
from loraphymac.utils.generator import GatewayGenerator, NodeGenerator, RelayGenerator


def build_infrastructure(config: Mapping[str, Any]) -> SimulationInfrastructure:
    """Initialize RNGs and construct the shared simulation services."""
    env = simpy.Environment()
    env.simulation_end = float(config["sim_time"])

    # Keep this order stable: topology and event timing depend on both streams.
    np.random.seed(int(config["seed"]))
    random.seed(int(config["seed"]))

    channel_plan = AS923_2()
    air = AirInterface(
        env,
        feasible_margin_db=config["feasible_margin_db"],
        delta_snr=config["delta_snr"],
    )
    model = ITM()
    collision_engine = CollisionEngine(
        mode=config["collision_mode"],
        capture_threshold_dB=config["capture_threshold_db"],
        enforce_sf_match=config["collision_enforce_sf_match"],
        sf_interference_mode=config["collision_sf_interference_mode"],
        sir_matrix=config["collision_sir_matrix"],
        sir_matrix_version=config["collision_sir_matrix_version"],
        sir_matrix_preset=config["collision_sir_matrix_preset"],
        noise_floor_dBm=config["collision_noise_floor_dbm"],
        destructive=config["collision_destructive"],
    )
    router = RoutingEngine(
        routing_hop_penalty_alpha=config["routing_hop_penalty_alpha"]
    )
    return SimulationInfrastructure(
        env=env,
        channel_plan=channel_plan,
        air=air,
        model=model,
        collision_engine=collision_engine,
        router=router,
    )


def build_topology(
    config: Mapping[str, Any],
    infrastructure: SimulationInfrastructure,
) -> SimulationTopology:
    """Generate the topology and apply regulatory EIRP limits."""
    nodes = NodeGenerator.uniform_random(
        n=config["node_num"],
        x_range=(0, config["area_x"]),
        y_range=(-config["area_y"] / 2, config["area_y"] / 2),
        height=config["node_ant_height"],
        radio=SX1276(),
        mcu=ESP32(),
        gain=config["node_ant_gain"],
        channel_plan=infrastructure.channel_plan,
        air_interface=infrastructure.air,
        routing_engine=infrastructure.router,
        collision_engine=infrastructure.collision_engine,
        seed=config["seed"],
    )

    relays = RelayGenerator.linear(
        n=config["relay_num"],
        x_range=(0, config["area_x"]),
        y_fixed=0,
        height=config["relay_ant_height"],
        radio=SX1302(),
        mcu=ESP32(),
        gain=config["relay_ant_gain"],
        channel_plan=infrastructure.channel_plan,
        air_interface=infrastructure.air,
        routing_engine=infrastructure.router,
        collision_engine=infrastructure.collision_engine,
    )

    gateway = GatewayGenerator.fixed(
        x=0,
        y=0,
        height=config["gw_ant_height"],
        radio=SX1302(),
        mcu=ESP32(),
        gain=config["gw_ant_gain"],
        channel_plan=infrastructure.channel_plan,
        air_interface=infrastructure.air,
        routing_engine=infrastructure.router,
        collision_engine=infrastructure.collision_engine,
    )

    for node in nodes:
        apply_eirp_ptx_bounds(
            node,
            max_eirp_dbm=config["node_max_eirp_dbm"],
            antenna_gain_dbi=config["node_ant_gain"],
            feeder_loss_db=config["node_feeder_loss_db"],
        )

    for relay in relays:
        apply_eirp_ptx_bounds(
            relay,
            max_eirp_dbm=config["relay_max_eirp_dbm"],
            antenna_gain_dbi=config["relay_ant_gain"],
            feeder_loss_db=config["relay_feeder_loss_db"],
        )

    apply_eirp_ptx_bounds(
        gateway,
        max_eirp_dbm=config["gw_max_eirp_dbm"],
        antenna_gain_dbi=config["gw_ant_gain"],
        feeder_loss_db=config["gw_feeder_loss_db"],
    )
    return SimulationTopology(nodes=nodes, relays=relays, gateway=gateway)


def build_routing(
    config: Mapping[str, Any],
    infrastructure: SimulationInfrastructure,
    topology: SimulationTopology,
) -> RoutingArtifacts:
    """Compute and validate node routes plus relay-chain link evidence."""
    nodes = topology.nodes
    relays = topology.relays
    gateway = topology.gateway
    router = infrastructure.router

    relay_hops = router.compute_relay_hops(relays, gateway)
    link_dfs = []
    for node in nodes:
        link_dfs.append(
            router.evaluate_links(
                node=node,
                relays=relays,
                relay_hops=relay_hops,
                gateway=gateway,
                model=infrastructure.model,
                air=infrastructure.air,
                sf=config["routing_assumed_sf"],
                bw_hz=config["node_bw"],
                ptx=node.ptx_max,
                ch_id=config["routing_ch_id"],
                channel_plan=infrastructure.channel_plan,
                terrain_delta_h=config["terrain_delta_h"],
            )
        )

    link_df = (
        pd.concat(link_dfs, ignore_index=True) if link_dfs else pd.DataFrame()
    )
    routing_strategy = str(
        config.get("routing_strategy", ROUTING_STRATEGY_TERRAIN_SPECIFIC)
    )
    loaded_snapshot = None
    if routing_strategy == ROUTING_STRATEGY_FIXED_REFERENCE:
        loaded_snapshot = load_routing_snapshot_bundle(
            manifest_path=config.get("routing_snapshot_manifest"),
            expected_manifest_sha256=config.get(
                "routing_snapshot_manifest_sha256"
            ),
            config=config,
            topology=topology,
            frequency_mhz=infrastructure.channel_plan.get_freq(
                config["routing_ch_id"]
            ),
        )
        routing_map = dict(loaded_snapshot.routing_map)
        routing_table = {node.id: routing_map[node.id] for node in nodes}
    elif routing_strategy == ROUTING_STRATEGY_TERRAIN_SPECIFIC:
        # Preserve the historical route-selection path exactly.
        routing_table = router.build_routing_table(link_df=link_df)
        routing_map = router.build_routing_map(
            relays=relays,
            relay_hops=relay_hops,
            routing_table=routing_table,
            gateway=gateway,
        )
    else:
        raise ValueError(f"Unsupported routing_strategy={routing_strategy!r}")

    if not link_df.empty:
        selected_next_hop_links_df = link_df[
            link_df.apply(
                lambda row: routing_table.get(row["Node"]) == row["Target"],
                axis=1,
            )
        ].reset_index(drop=True)
    else:
        selected_next_hop_links_df = pd.DataFrame()

    router.validate_linear_relay_order(
        relays=relays,
        relay_hops=relay_hops,
        gateway=gateway,
    )
    router.validate_routing_map(
        routing_map=routing_map,
        node_ids=[node.id for node in nodes],
        relay_ids=[relay.id for relay in relays],
        gateway_id=gateway.id,
    )

    if bool(config["print_internal"]):
        print("\n🔍 RELAY HOP-CHAIN (DETERMINISTIC)")
        print(relay_hops)
        print("\n🔀 ROUTING TABLE")
        print(routing_table)
        print("\n=== ROUTING MAP ===")
        print(routing_map)

    assert routing_map[gateway.id] is None, "Gateway must be the terminal hop"

    all_nodes = topology.all_nodes
    node_map = {node.id: node for node in all_nodes}
    relay_chain_links_df = router.evaluate_relay_chain_links(
        relays=relays,
        routing_map=routing_map,
        node_map=node_map,
        relay_hops=relay_hops,
        gateway=gateway,
        model=infrastructure.model,
        air=infrastructure.air,
        sf=config["relay_sf"],
        bw_hz=config["relay_bw_hz"],
        ch_id=config["routing_ch_id"],
        channel_plan=infrastructure.channel_plan,
        terrain_delta_h=config["terrain_delta_h"],
    )

    visual_link_frames = [
        frame
        for frame in (selected_next_hop_links_df, relay_chain_links_df)
        if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    visual_links_df = (
        pd.concat(visual_link_frames, ignore_index=True)
        if visual_link_frames
        else pd.DataFrame(columns=relay_chain_links_df.columns)
    )

    if bool(config["print_internal"]):
        print("\n=== RELAY CHAIN LINKS QUALITY ===")
        if not relay_chain_links_df.empty:
            print(relay_chain_links_df[["Node", "Target", "Margin(dB)", "Valid"]])
        else:
            print(relay_chain_links_df)
        print("\n=== VISUAL LINKS DF ===")
        if not visual_links_df.empty:
            print(visual_links_df[["Node", "Target", "Margin(dB)", "Valid"]])
        else:
            print(visual_links_df)

    return RoutingArtifacts(
        relay_hops=relay_hops,
        link_df=link_df,
        routing_table=routing_table,
        selected_next_hop_links_df=selected_next_hop_links_df,
        routing_map=routing_map,
        all_nodes=all_nodes,
        node_map=node_map,
        relay_chain_links_df=relay_chain_links_df,
        visual_links_df=visual_links_df,
        routing_strategy=routing_strategy,
        route_construction_delta_h=(
            loaded_snapshot.reference_delta_h
            if loaded_snapshot is not None
            else config["terrain_delta_h"]
        ),
        physical_terrain_delta_h=config["terrain_delta_h"],
        routing_snapshot_id=(
            loaded_snapshot.snapshot_id if loaded_snapshot is not None else None
        ),
        routing_snapshot_sha256=(
            loaded_snapshot.snapshot_sha256
            if loaded_snapshot is not None
            else None
        ),
        routing_snapshot_manifest_sha256=(
            loaded_snapshot.manifest_sha256
            if loaded_snapshot is not None
            else None
        ),
        topology_sha256=topology_sha256(topology),
        selected_parent_routing_sha256=routing_sha256(routing_map),
    )
