# =========================================================
# utils/visualizer.py
# Physical DATA_DOWN / DATA_DOWN_FWD + 
# Relay APC RX cleanup, APC TX index, and timing validation
#
# Network visualization utilities
# Principles:
# - The visualizer does not calculate path loss
# - The visualizer does not calculate the link budget
# - All link colors reflect link quality from visual_links_df
# - End-device links use dashed lines
# - Relay-backbone links use solid lines
# =========================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# =========================================================
# LINK QUALITY COLOR
# =========================================================



def _link_color(margin, feasible, feasible_margin_db=3, target_margin_db=6):
    """
    Link quality color based on effective margin.

    Effective margin:
        margin = RSSI - P_rx_sens - M_feas

    Rule:
        margin <= 0            : Not Feasible
        0 < margin < M_target  : Feasible / Marginal
        margin >= M_target     : Good
    """
    if margin is None:
        return "gray", "Unknown"

    target_threshold = float(target_margin_db)

    if (not feasible) or margin <= 0:
        return "red", "Not Feasible: margin ≤ 0 dB"

    if margin < target_threshold:
        return (
            "orange",
            f"Marginal: 0 < margin < {target_threshold} dB"
        )

    return (
        "limegreen",
        f"Good: margin ≥ target margin ({target_threshold} dB)"
    )


def _build_link_quality_map(selected_links_df, feasible_margin_db=3, target_margin_db=6):
    """
    Convert selected_links_df / visual_links_df into dictionary:
        (src, dst) -> quality info

    Expected columns:
        Node, Target, Margin(dB), Valid,
        Distance(m), RSSI(dBm), SNR(dB), Score

    Note:
    ``visual_links_df`` must contain:
    1. selected_next_hop_links_df
    2. relay_chain_links_df
    """
    link_quality = {}

    if selected_links_df is None or selected_links_df.empty:
        return link_quality

    for _, row in selected_links_df.iterrows():
        src = row["Node"]
        dst = row["Target"]

        margin = row.get("Margin(dB)", None)
        feasible = bool(row.get("Valid", True))

        color, label = _link_color(
            margin,
            feasible,
            feasible_margin_db=feasible_margin_db,
            target_margin_db=target_margin_db,
        )

        
        link_quality[(src, dst)] = {
            "color": color,
            "label": label,
            "margin": margin,
            "feasible": feasible,
            "distance_m": row.get("Distance(m)", None),
            "rssi": row.get("RSSI(dBm)", None),
            "snr": row.get("SNR(dB)", None),
            "score": row.get("Score", None),
        }

    return link_quality


def _draw_link(
    ax,
    src_obj,
    dst_obj,
    color,
    linestyle="--",
    linewidth=1.8,
    alpha=0.75,
    zorder=2
):
    """
    Draw one link between src and dst.
    """
    ax.plot(
        [src_obj.loc.x, dst_obj.loc.x],
        [src_obj.loc.y, dst_obj.loc.y],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder
    )


def _draw_margin_label(ax, src_obj, dst_obj, info, color):
    """
    Draw margin label at the middle of a link.
    """
    if info is None:
        return

    margin = info.get("margin", None)
    if margin is None:
        return

    mx = (src_obj.loc.x + dst_obj.loc.x) / 2
    my = (src_obj.loc.y + dst_obj.loc.y) / 2

    ax.text(
        mx,
        my,
        f"{margin:.1f} dB",
        fontsize=7,
        color=color,
        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"),
        zorder=6
    )


def plot_routing(
    nodes,
    relays,
    gateway,
    routing_map,
    router,
    node_map,
    selected_links_df=None,
    feasible_margin_db=3,
    target_margin_db=6,
    show_margin_label=False,
    title=None,
    save_path=None,
    show=True
):
    """
    Plot multi-hop routing topology with link quality color coding.

    Colors:
        limegreen = Good
        orange    = Marginal
        red       = Not Feasible
        gray      = no link-quality data

    Line styles:
        End Device -> Relay/Gateway = dashed
        Relay -> Relay/Gateway      = solid
    """

    fig, ax = plt.subplots(figsize=(16, 8))

    link_quality = _build_link_quality_map(
        selected_links_df,
        feasible_margin_db=feasible_margin_db,
        target_margin_db=target_margin_db,
    )

    # =====================================================
    # Plot End Devices
    # =====================================================
    for n in nodes:
        ax.scatter(
            n.loc.x,
            n.loc.y,
            c="dodgerblue",
            s=60,
            zorder=5
        )
        ax.text(
            n.loc.x + 200,
            n.loc.y + 200,
            n.id,
            fontsize=7
        )

    # =====================================================
    # Plot Relays
    # =====================================================
    for r in relays:
        ax.scatter(
            r.loc.x,
            r.loc.y,
            c="orange",
            s=200,
            zorder=5,
            edgecolors="black",
            linewidths=0.8
        )
        ax.text(
            r.loc.x + 200,
            r.loc.y + 200,
            r.id,
            fontsize=9,
            fontweight="bold"
        )

    # =====================================================
    # Plot Gateway
    # =====================================================
    ax.scatter(
        gateway.loc.x,
        gateway.loc.y,
        c="red",
        s=300,
        zorder=5,
        edgecolors="black",
        linewidths=1.0
    )
    ax.text(
        gateway.loc.x + 200,
        gateway.loc.y + 200,
        gateway.id,
        fontsize=10,
        fontweight="bold"
    )

    # =====================================================
    # Draw End Device Selected Links Only
    # =====================================================
    # Draw only the first link from each node:
    # N -> Relay/Gateway
    # The relay chain is drawn in the dedicated relay block.
    drawn_links = set()

    for node in nodes:
        path = router.get_full_path(node.id, routing_map)

        if len(path) < 2:
            continue

        src = path[0]
        dst = path[1]
        link_key = (src, dst)

        if link_key in drawn_links:
            continue

        drawn_links.add(link_key)

        src_obj = node_map[src]
        dst_obj = node_map[dst]

        info = link_quality.get(link_key, {})
        color = info.get("color", "gray")

        _draw_link(
            ax=ax,
            src_obj=src_obj,
            dst_obj=dst_obj,
            color=color,
            linestyle="--",
            linewidth=1.8,
            alpha=0.75,
            zorder=2
        )

        if show_margin_label:
            _draw_margin_label(ax, src_obj, dst_obj, info, color)

    # =====================================================
    # Draw Relay Links
    # =====================================================
    # Relay links still use their link-quality colors.
    # A solid line, rather than black coloring, distinguishes relay links.
    for relay in relays:
        nxt = routing_map.get(relay.id)

        if nxt is None:
            continue

        link_key = (relay.id, nxt)

        if link_key in drawn_links:
            continue

        drawn_links.add(link_key)

        src_obj = node_map[relay.id]
        dst_obj = node_map[nxt]

        info = link_quality.get(link_key, {})
        color = info.get("color", "gray")

        _draw_link(
            ax=ax,
            src_obj=src_obj,
            dst_obj=dst_obj,
            color=color,
            linestyle="-",
            linewidth=2.8,
            alpha=0.95,
            zorder=3
        )

        if show_margin_label:
            _draw_margin_label(ax, src_obj, dst_obj, info, color)

    # =====================================================
    # Legend
    # =====================================================
    legend_handles = [
        mpatches.Patch(
            color="limegreen",
            label=f"Good: margin ≥ M_target ({target_margin_db} dB)"
        ),
        mpatches.Patch(
            color="orange",
            label=f"Marginal: 0 < margin < M_target ({target_margin_db} dB)"
        ),
        mpatches.Patch(
            color="red",
            label="Not Feasible: margin ≤ 0 dB"
        ),
    ]

    
    line_node = plt.Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="End Device Link"
    )

    line_relay = plt.Line2D(
        [0],
        [0],
        color="black",
        linestyle="-",
        linewidth=2.8,
        label="Relay Link"
    )

    legend_handles.extend([line_node, line_relay])

    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

    ax.set_title(title or "LoRa Multi-Hop Routing — Link Quality")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# MULTI-SCENARIO COMPARISON CHARTS
# =========================================================

def plot_scenario_comparison(results_list, metric="total_energy_mJ"):
    """
    Bar chart comparing one metric across scenarios.
    """
    if metric == "total_energy_mJ":
        _plot_energy_comparison(results_list)
    elif metric == "pdr_percent":
        _plot_pdr_comparison(results_list)
    elif metric == "avg_e2e_delay_ms":
        _plot_delay_comparison(results_list)
    else:
        _plot_energy_comparison(results_list)


def _plot_energy_comparison(results_list):
    """
    Stacked bar chart: energy per role per scenario.
    """
    labels = []
    node_e = []
    relay_e = []
    gw_e = []

    for r in results_list:
        labels.append(r["label"])
        e = r["energy"]
        node_e.append(e.get("nodes_total_mJ", 0))
        relay_e.append(e.get("relays_total_mJ", 0))
        gw_e.append(e.get("gateway_total_mJ", 0))

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x, node_e, width, label="Nodes")
    ax.bar(x, relay_e, width, bottom=node_e, label="Relays")

    bottom_gw = [n + r for n, r in zip(node_e, relay_e)]
    ax.bar(x, gw_e, width, bottom=bottom_gw, label="Gateway")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Energy (mJ)")
    ax.set_title("Energy Consumption Comparison")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    plt.show()


def _plot_pdr_comparison(results_list):
    """
    Bar chart: PDR per scenario.
    """
    labels = []
    values = []

    for r in results_list:
        labels.append(r["label"])
        values.append(r.get("pdr_percent", 0))

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("PDR (%)")
    ax.set_title("Packet Delivery Ratio Comparison")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    plt.show()


def _plot_delay_comparison(results_list):
    """
    Bar chart: average end-to-end delay per scenario.
    """
    labels = []
    values = []

    for r in results_list:
        labels.append(r["label"])
        values.append(r.get("avg_e2e_delay_ms", 0))

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Average E2E Delay (ms)")
    ax.set_title("End-to-End Delay Comparison")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    plt.show()
