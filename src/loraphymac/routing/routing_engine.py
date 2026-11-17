# ================================================================
# routing/routing_engine.py
# Physical DATA_DOWN / DATA_DOWN_FWD + 
# Relay APC RX cleanup, APC TX index, and timing validation
#
# Hybrid of V4 (deterministic hop-chain) + improvements from V6.3.
# Stable for long-range LoRa Multi-Hop (10–80 km).
#
# Features:
# - Deterministic hop chain based on distance to the gateway
# - Score = RSSI − α·HopChain
# - Nodes select the best next hop (relay/gateway)
# - Relays automatically form a parent chain
# - Supports multi-hop uplink and downlink
# ================================================================

import numpy as np
import pandas as pd


LINK_COLUMNS = [
    "Node",
    "Target",
    "ChID",
    "Freq(MHz)",
    "Distance(m)",
    "Pathloss(dB)",
    "RSSI(dBm)",
    "SNR(dB)",
    "Margin(dB)",
    "Hop",
    "Score",
    "Valid",
]


class RoutingEngine:

    def __init__(self, routing_hop_penalty_alpha=1):
        """
        α-hop penalty:
            Score = RSSI − α·H
        """
        self.routing_hop_penalty_alpha = routing_hop_penalty_alpha

    # ------------------------------------------------------------
    # 1. Deterministic hop chain based on distance to the gateway
    # ------------------------------------------------------------
    def compute_relay_hops(self, relays, gateway):
        """
        The relay closest to the gateway has hop=1.
        The next relay has hop=2, and so on.
        """
        relay_ids = [relay.id for relay in relays]
        if len(relay_ids) != len(set(relay_ids)):
            raise ValueError("Relay IDs must be unique")
        if gateway.id in set(relay_ids):
            raise ValueError(
                f"Gateway ID {gateway.id!r} must not also be used by a relay"
            )

        rel_sorted = sorted(relays, key=lambda r: abs(r.loc.x - gateway.loc.x))

        hop_map = {}
        for i, r in enumerate(rel_sorted, start=1):
            hop_map[r.id] = i

        # Gateway = hop 0
        hop_map[gateway.id] = 0

        return hop_map

    # ------------------------------------------------------------
    # 2. Evaluate all node-to-relay and node-to-gateway links
    # ------------------------------------------------------------
    def evaluate_links(
        self, node, relays, relay_hops, gateway,
        model, air, sf, bw_hz, ptx,
        freq_mhz=None,
        ch_id=None,
        channel_plan=None,
        terrain_delta_h=90
    ):
        if freq_mhz is None:
            if ch_id is None or channel_plan is None:
                raise ValueError("Provide freq_mhz or ch_id + channel_plan")
            freq_mhz = channel_plan.get_freq(ch_id)
        
        rows = []

        # ---------- Node → Relay ----------
        for relay in relays:

            hop_chain = relay_hops[relay.id]

            d = node.distance_to(relay)
            pl = model.pathloss(
                d_m=d,
                freq_mhz=freq_mhz,
                h_tx=node.h,
                h_rx=relay.h,
                terrain_delta_h=terrain_delta_h
            )

            lb = air.compute_link_budget(
                hw_rx=relay.hw,
                sf=sf,
                bw_hz=bw_hz,
                ptx=ptx,
                gtx=node.gain,
                grx=relay.gain,
                pathloss=pl
            )

            score = lb["rssi"] - self.routing_hop_penalty_alpha * hop_chain 

            rows.append({
                "Node": node.id,
                "Target": relay.id,
                "ChID": ch_id,
                "Freq(MHz)": freq_mhz,
                "Distance(m)": d,
                "Pathloss(dB)": pl,
                "RSSI(dBm)": lb["rssi"],
                "SNR(dB)":  lb["snr"],
                "Margin(dB)": lb["margin"],
                "Hop": hop_chain,
                "Score": score,
                "Valid": lb["feasible"]
            })

        # ---------- Node → Gateway ----------
        d_gw = node.distance_to(gateway)

        pl_gw = model.pathloss(
            d_m=d_gw,
            freq_mhz=freq_mhz,
            h_tx=node.h,
            h_rx=gateway.h,
            terrain_delta_h=terrain_delta_h
        )

        lb_gw = air.compute_link_budget(
            hw_rx=gateway.hw,
            sf=sf,
            bw_hz=bw_hz,
            ptx=ptx,
            gtx=node.gain,
            grx=gateway.gain,
            pathloss=pl_gw
        )

        score_gw = lb_gw["rssi"] - self.routing_hop_penalty_alpha * 0 

        rows.append({
            "Node": node.id,
            "Target": gateway.id,
            "ChID": ch_id,
            "Freq(MHz)": freq_mhz,
            "Distance(m)": d_gw,
            "Pathloss(dB)": pl_gw,
            "RSSI(dBm)": lb_gw["rssi"],
            "SNR(dB)": lb_gw["snr"],
            "Margin(dB)": lb_gw["margin"],
            "Hop": 0,
            "Score": score_gw,
            "Valid": lb_gw["feasible"]
        })

        return pd.DataFrame(rows, columns=LINK_COLUMNS)

    # ------------------------------------------------------------
    # 3. Routing table: Node → best next-hop
    # ------------------------------------------------------------
    def build_routing_table(self, link_df):
        """
        Select the highest-Score candidate for every node.

        ``Valid`` is intentionally not a routing filter. It is a
        planning-time feasibility estimate used by diagnostics and
        visualization. The next-hop receiver performs the authoritative
        feasibility check for every transmitted packet.
        """
        if link_df is None:
            raise TypeError("link_df must be a pandas DataFrame")
        if link_df.empty:
            return {}

        required_columns = {"Node", "Target", "Score"}
        missing_columns = required_columns.difference(link_df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"link_df missing required columns: {missing}")

        rt = {}

        for node_id in link_df["Node"].unique():
            # Deliberately retain candidates with Valid=False. Their packets
            # must still be transmitted and included in the PDR denominator.
            sub = link_df[link_df["Node"] == node_id]

            if len(sub) == 0:
                rt[node_id] = None
            else:
                # Stable sort makes equal-score selection follow the explicit
                # candidate order in link_df instead of an unstable sort.
                best = sub.sort_values(
                    "Score",
                    ascending=False,
                    kind="mergesort",
                ).iloc[0]
                rt[node_id] = best["Target"]

        return rt

    # ------------------------------------------------------------
    # 4. Relay parent chain: hop n → hop (n-1)
    # ------------------------------------------------------------
    def build_relay_chain(self, relays, relay_hops, gateway):
        parent = {}

        relay_ids = [relay.id for relay in relays]
        if len(relay_ids) != len(set(relay_ids)):
            raise ValueError("Relay IDs must be unique")

        relay_by_hop = {}
        for relay in relays:
            if relay.id not in relay_hops:
                raise ValueError(f"Missing hop assignment for relay {relay.id}")
            hop = relay_hops[relay.id]
            if not isinstance(hop, (int, np.integer)) or int(hop) < 1:
                raise ValueError(
                    f"Invalid hop assignment for relay {relay.id}: {hop!r}"
                )
            hop = int(hop)
            if hop in relay_by_hop:
                raise ValueError(f"Multiple relays assigned to hop {hop}")
            relay_by_hop[hop] = relay.id

        expected_hops = set(range(1, len(relays) + 1))
        if set(relay_by_hop) != expected_hops:
            raise ValueError(
                "Relay hops must form a contiguous sequence from 1 to "
                f"{len(relays)}"
            )

        for r in relays:
            hop = int(relay_hops[r.id])

            if hop == 1:
                parent[r.id] = gateway.id
            else:
                parent[r.id] = relay_by_hop[hop - 1]

        parent[gateway.id] = None
        return parent

    # ------------------------------------------------------------
    # 5. Build the complete routing map
    # ------------------------------------------------------------
    def build_routing_map(self, relays, relay_hops, routing_table, gateway):
        # Node → relay/gateway
        routing_map = dict(routing_table)

        # Relay → relay/gateway
        parent_rel = self.build_relay_chain(relays, relay_hops, gateway)
        routing_map.update(parent_rel)

        # Gateway → None
        routing_map[gateway.id] = None

        return routing_map

    def validate_linear_relay_order(self, relays, relay_hops, gateway):
        """Validate the paper scenario Rn -> ... -> R2 -> R1 -> GW."""
        expected = {
            f"R{index}": index
            for index in range(1, len(relays) + 1)
        }
        actual = {
            relay.id: relay_hops.get(relay.id)
            for relay in relays
        }

        if actual != expected:
            raise ValueError(
                "Linear relay scenario requires R1 nearest to the gateway, "
                "followed by R2, R3, ...; "
                f"expected hops={expected}, actual hops={actual}"
            )

        relay_by_id = {relay.id: relay for relay in relays}
        ordered_distances = [
            abs(relay_by_id[f"R{index}"].loc.x - gateway.loc.x)
            for index in range(1, len(relays) + 1)
        ]
        if any(
            farther <= nearer
            for nearer, farther in zip(
                ordered_distances,
                ordered_distances[1:],
            )
        ):
            raise ValueError(
                "Linear relay distances must increase strictly from "
                "R1 toward R2, R3, ..."
            )
        if relay_hops.get(gateway.id) != 0:
            raise ValueError(f"Gateway {gateway.id} must be assigned hop 0")

        return True

    def validate_routing_map(
        self,
        routing_map,
        node_ids,
        relay_ids,
        gateway_id,
    ):
        """
        Validate topology integrity without evaluating PHY feasibility.

        Every node and relay must have a cycle-free topological path to the
        gateway. A link may still have ``Valid=False``; runtime receivers are
        responsible for accepting or dropping the transmitted packet.
        """
        node_ids = list(node_ids)
        relay_ids = list(relay_ids)
        entity_ids = node_ids + relay_ids + [gateway_id]

        seen_ids = set()
        duplicate_ids = set()
        for entity_id in entity_ids:
            if entity_id in seen_ids:
                duplicate_ids.add(entity_id)
            else:
                seen_ids.add(entity_id)
        duplicates = sorted(duplicate_ids, key=str)
        if duplicates:
            raise ValueError(f"Duplicate topology entity IDs: {duplicates}")

        known_ids = set(entity_ids)
        unknown_sources = set(routing_map).difference(known_ids)
        if unknown_sources:
            raise ValueError(
                "routing_map contains unknown source IDs: "
                f"{sorted(unknown_sources, key=str)}"
            )

        if gateway_id not in routing_map:
            raise ValueError(f"Gateway {gateway_id} is missing from routing_map")
        if routing_map[gateway_id] is not None:
            raise ValueError(f"Gateway {gateway_id} must be terminal")

        for source_id in node_ids + relay_ids:
            if source_id not in routing_map:
                raise ValueError(
                    f"Entity {source_id} is missing from routing_map"
                )

            current = source_id
            visited = {current}
            while current != gateway_id:
                next_hop = routing_map.get(current)
                if next_hop is None:
                    raise ValueError(
                        f"Route from {source_id} terminates at {current}, "
                        f"not at gateway {gateway_id}"
                    )
                if next_hop not in known_ids:
                    raise ValueError(
                        f"Route from {source_id} references unknown target "
                        f"{next_hop}"
                    )
                if next_hop in visited:
                    raise ValueError(
                        f"Routing cycle detected from {source_id} at "
                        f"{next_hop}"
                    )
                visited.add(next_hop)
                current = next_hop

        return True


    # ------------------------------------------------------------
    # Evaluate Relay → Relay/Gateway link quality
    # ------------------------------------------------------------
    def evaluate_relay_chain_links(
        self,
        relays,
        routing_map,
        node_map,
        relay_hops,
        gateway,
        model,
        air,
        sf,
        bw_hz,
        ptx=None,
        ch_id=None,
        channel_plan=None,
        freq_mhz=None,
        terrain_delta_h=90
    ):
        """
        Evaluate link quality for relay chain:
            Relay -> Relay/Gateway

        Output columns are aligned with evaluate_links():
            Node, Target, ChID, Freq(MHz), Distance(m),
            Pathloss(dB), RSSI(dBm), SNR(dB),
            Margin(dB), Hop, Score, Valid
        """

        if freq_mhz is None:
            if ch_id is None or channel_plan is None:
                raise ValueError("Provide freq_mhz or ch_id + channel_plan")
            freq_mhz = channel_plan.get_freq(ch_id)

        rows = []

        for relay in relays:
            nxt = routing_map.get(relay.id)

            if nxt is None:
                continue

            if nxt not in node_map:
                raise KeyError(f"Target relay link {nxt} is missing from node_map")

            dst_obj = node_map[nxt]

            d = relay.distance_to(dst_obj)

            pl = model.pathloss(
                d_m=d,
                freq_mhz=freq_mhz,
                h_tx=relay.h,
                h_rx=dst_obj.h,
                terrain_delta_h=terrain_delta_h
            )

            relay_ptx = relay.ptx_max if ptx is None else ptx

            lb = air.compute_link_budget(
                hw_rx=dst_obj.hw,
                sf=sf,
                bw_hz=bw_hz,
                ptx=relay_ptx,
                gtx=relay.gain,
                grx=dst_obj.gain,
                pathloss=pl
            )

            hop_value = relay_hops.get(relay.id, None)

            rows.append({
                "Node": relay.id,
                "Target": nxt,
                "ChID": ch_id,
                "Freq(MHz)": freq_mhz,
                "Distance(m)": d,
                "Pathloss(dB)": pl,
                "RSSI(dBm)": lb["rssi"],
                "SNR(dB)": lb["snr"],
                "Margin(dB)": lb["margin"],
                "Hop": hop_value,
                "Score": lb["rssi"],
                "Valid": lb["feasible"]
            })

        return pd.DataFrame(rows, columns=LINK_COLUMNS)
    
    # ------------------------------------------------------------
    # 6. Multi-hop full path
    # ------------------------------------------------------------
    def get_full_path(self, node_id, routing_map):
        path = [node_id]
        cur = node_id
        visited = {node_id}

        while True:
            if cur not in routing_map:
                raise KeyError(f"Entity {cur} is missing from routing_map")

            nxt = routing_map[cur]
            if nxt is None:
                return path
            if nxt not in routing_map:
                raise KeyError(f"Target {nxt} is missing from routing_map")
            if nxt in visited:
                raise ValueError(f"Routing cycle detected at {nxt}")
            visited.add(nxt)
            path.append(nxt)
            cur = nxt

    def dst_sort_key(dst):
        if dst == "GW":
            return 10_000        # The gateway is always last.
        if isinstance(dst, str) and dst.startswith("R"):
            return int(dst[1:]) # R1, R2, R3, ...
        return 9_000             # Safe fallback
