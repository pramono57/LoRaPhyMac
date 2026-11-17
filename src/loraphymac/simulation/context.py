"""Explicit groups of runtime objects used by the simulation orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class SimulationInfrastructure:
    """Shared services constructed once before topology generation."""

    env: Any
    channel_plan: Any
    air: Any
    model: Any
    collision_engine: Any
    router: Any


@dataclass(frozen=True, slots=True)
class SimulationTopology:
    """Generated end devices, relays, and gateway for one run."""

    nodes: list[Any]
    relays: list[Any]
    gateway: Any

    @property
    def all_nodes(self) -> list[Any]:
        """Return entities in the historical node/relay/gateway order."""
        return self.nodes + self.relays + [self.gateway]


@dataclass(frozen=True, slots=True)
class RoutingArtifacts:
    """Routing decisions and link-quality evidence for one topology."""

    relay_hops: dict[Any, Any]
    link_df: pd.DataFrame
    routing_table: dict[Any, Any]
    selected_next_hop_links_df: pd.DataFrame
    routing_map: dict[Any, Any]
    all_nodes: list[Any]
    node_map: dict[Any, Any]
    relay_chain_links_df: pd.DataFrame
    visual_links_df: pd.DataFrame
    routing_strategy: str = "terrain_specific"
    route_construction_delta_h: int | float | None = None
    physical_terrain_delta_h: int | float | None = None
    routing_snapshot_id: str | None = None
    routing_snapshot_sha256: str | None = None
    routing_snapshot_manifest_sha256: str | None = None
    topology_sha256: str | None = None
    selected_parent_routing_sha256: str | None = None
