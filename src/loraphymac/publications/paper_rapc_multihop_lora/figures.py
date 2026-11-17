"""R-APC multi-hop LoRa publication figure builders."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from loraphymac.publications.paper_rapc_multihop_lora.analysis import (
    APC_MODE_ORDER,
    DEFAULT_CONFIG,
    DEFAULT_COMPARISON_TERRAIN_DELTA_H,
    add_apc_mode,
    assert_publication_ready,
    experiment_paths,
    load_core_aggregate,
    publication_paths,
    publication_spec,
)


FIGURE_FONT_SIZE = 18
FIGURE_STYLE = {
    "font.family": "Times New Roman",
    "font.size": FIGURE_FONT_SIZE,
    "axes.labelsize": FIGURE_FONT_SIZE,
    "axes.titlesize": FIGURE_FONT_SIZE,
    "xtick.labelsize": FIGURE_FONT_SIZE,
    "ytick.labelsize": FIGURE_FONT_SIZE,
    "legend.fontsize": FIGURE_FONT_SIZE,
    "legend.title_fontsize": FIGURE_FONT_SIZE,
    "figure.titlesize": FIGURE_FONT_SIZE,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
}

FIGURE_METRICS = (
    ("delivery_by_horizon_percent", "Packet delivery ratio (%)", "figure_04_pdr"),
    (
        "energy_per_delivered_packet_J",
        "Energy per delivered packet (J)",
        "figure_05_energy_per_delivered_packet",
    ),
    ("avg_e2e_delay_ms", "Average end-to-end delay (ms)", "figure_06_delay"),
)

ENERGY_BREAKDOWN_METRICS = (
    ("total_node_energy_J", "Node", "tab:blue"),
    ("total_relay_energy_J", "Relay", "tab:orange"),
    ("total_gateway_energy_J", "Gateway", "tab:green"),
)

PLOT_MODE_ORDER = ("AA", "AF", "FA", "FF")

MODE_COLORS = {
    "AA": "tab:blue",
    "AF": "tab:orange",
    "FA": "tab:green",
    "FF": "tab:red",
}

MODE_STYLES = {
    "FF": {"marker": "D", "linestyle": ":"},
    "AF": {"marker": "s", "linestyle": "--"},
    "FA": {"marker": "^", "linestyle": "-."},
    "AA": {"marker": "o", "linestyle": "-"},
}

FINAL_NODE_SERIES = (
    ("AF", r"$P_{tx}$ Node Adaptive", "tab:blue", "o", "-"),
    ("FF", r"$P_{tx}$ Node Fixed", "tab:orange", "D", ":"),
)

FINAL_RELAY_SERIES = (
    ("FA", r"$P_{tx}$ Relay Adaptive", "tab:blue", "^", "-."),
    ("FF", r"$P_{tx}$ Relay Fixed", "tab:orange", "D", ":"),
)


def _save_figure(figure, output_base: Path) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]
    figure.tight_layout()
    figure.savefig(paths[0], dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return paths


def _terrain_slug(value: float) -> str:
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else str(numeric).replace(".", "p")


def _validated_metric_frame(
    aggregate: pd.DataFrame,
    *,
    metric: str,
    required_modes: tuple[str, ...],
) -> pd.DataFrame:
    frame = add_apc_mode(aggregate)
    required = {"terrain_delta_h", "metric", "mean"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Paper 1 aggregate is missing columns: {sorted(missing)}")
    expected_terrains = sorted(pd.to_numeric(frame["terrain_delta_h"], errors="raise").unique())
    frame = frame.loc[(frame["metric"] == metric) & frame["Mode"].isin(required_modes)].copy()
    if frame.empty:
        raise ValueError(f"Paper 1 aggregate does not contain metric {metric!r}")

    frame["terrain_delta_h"] = pd.to_numeric(frame["terrain_delta_h"], errors="raise")
    frame["mean"] = pd.to_numeric(frame["mean"], errors="raise")
    if frame["mean"].isna().any():
        raise ValueError(f"Paper 1 metric {metric!r} contains missing mean values")
    duplicates = frame.duplicated(subset=["terrain_delta_h", "Mode"], keep=False)
    if duplicates.any():
        cells = frame.loc[duplicates, ["terrain_delta_h", "Mode"]].drop_duplicates()
        raise ValueError(
            f"Paper 1 metric {metric!r} contains duplicate terrain/mode rows: "
            f"{cells.to_dict(orient='records')}"
        )

    observed = set(zip(frame["terrain_delta_h"], frame["Mode"]))
    missing_cells = [
        f"terrain_delta_h={terrain:g}/{mode}"
        for terrain in expected_terrains
        for mode in required_modes
        if (terrain, mode) not in observed
    ]
    if missing_cells:
        raise ValueError(f"Paper 1 metric {metric!r} has an incomplete APC grid: {missing_cells}")
    return frame


def _plot_mode_series(
    axis,
    frame: pd.DataFrame,
    *,
    mode: str,
    label: str,
    color: str,
    show_confidence_interval: bool = False,
    marker: str | None = None,
    linestyle: str | None = None,
) -> None:
    mode_frame = frame.loc[frame["Mode"] == mode].sort_values("terrain_delta_h")
    x = mode_frame["terrain_delta_h"].to_numpy(dtype=float)
    mean = mode_frame["mean"].to_numpy(dtype=float)
    axis.plot(
        x,
        mean,
        marker=marker or MODE_STYLES[mode]["marker"],
        linestyle=linestyle or MODE_STYLES[mode]["linestyle"],
        linewidth=2,
        markersize=7,
        label=label,
        color=color,
    )
    if show_confidence_interval and {"ci_lower", "ci_upper"}.issubset(mode_frame.columns):
        lower = pd.to_numeric(mode_frame["ci_lower"], errors="coerce").to_numpy(dtype=float)
        upper = pd.to_numeric(mode_frame["ci_upper"], errors="coerce").to_numpy(dtype=float)
        axis.fill_between(x, lower, upper, color=color, alpha=0.05)


def generate_metric_figure(
    aggregate: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    output_base: str | Path,
) -> list[Path]:
    """Plot one APC metric across terrain with 95% confidence intervals."""
    frame = _validated_metric_frame(
        aggregate,
        metric=metric,
        required_modes=APC_MODE_ORDER,
    )

    with plt.rc_context(FIGURE_STYLE):
        figure, axis = plt.subplots(figsize=(8, 5))
        for mode in PLOT_MODE_ORDER:
            _plot_mode_series(
                axis,
                frame,
                mode=mode,
                label=mode,
                color=MODE_COLORS[mode],
            )

        axis.set_xlabel(r"Terrain irregularity, $\Delta h$ (m)")
        axis.set_ylabel(ylabel)
        axis.grid(True)
        axis.legend(ncols=2)
        return _save_figure(figure, Path(output_base))


def generate_energy_breakdown_figure(
    aggregate: pd.DataFrame,
    *,
    terrain_delta_h: float = DEFAULT_COMPARISON_TERRAIN_DELTA_H,
    output_base: str | Path,
) -> list[Path]:
    """Plot mean node, relay, and gateway energy as a stacked APC-mode bar chart."""
    metric_frames = []
    for metric, _, _ in ENERGY_BREAKDOWN_METRICS:
        frame = _validated_metric_frame(
            aggregate,
            metric=metric,
            required_modes=APC_MODE_ORDER,
        )
        terrain = frame["terrain_delta_h"]
        selected = frame.loc[(terrain - float(terrain_delta_h)).abs() < 1e-9].copy()
        if selected.empty:
            raise ValueError(
                f"No Paper 1 {metric!r} rows exist for terrain_delta_h={terrain_delta_h}"
            )
        metric_frames.append(selected)

    with plt.rc_context(FIGURE_STYLE):
        figure, axis = plt.subplots(figsize=(8, 5))
        bottoms = pd.Series(0.0, index=list(APC_MODE_ORDER))
        for (metric, label, color), frame in zip(ENERGY_BREAKDOWN_METRICS, metric_frames):
            values = frame.set_index("Mode")["mean"].reindex(APC_MODE_ORDER).astype(float)
            axis.bar(
                APC_MODE_ORDER,
                values,
                bottom=bottoms,
                label=label,
                color=color,
            )
            bottoms = bottoms.add(values)

        axis.set_xlabel("APC mode")
        axis.set_ylabel("Energy consumption (J)")
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend()
        return _save_figure(figure, Path(output_base))


def generate_final_ptx_figure(
    aggregate: pd.DataFrame,
    *,
    metric: str,
    series: tuple[tuple[str, str, str, str, str], ...],
    ylabel: str,
    output_base: str | Path,
) -> list[Path]:
    """Plot one controlled fixed-versus-adaptive final-power comparison."""
    required_modes = tuple(mode for mode, _, _, _, _ in series)
    frame = _validated_metric_frame(
        aggregate,
        metric=metric,
        required_modes=required_modes,
    )

    with plt.rc_context(FIGURE_STYLE):
        figure, axis = plt.subplots(figsize=(8, 5))
        for mode, label, color, marker, linestyle in series:
            _plot_mode_series(
                axis,
                frame,
                mode=mode,
                label=label,
                color=color,
                show_confidence_interval=True,
                marker=marker,
                linestyle=linestyle,
            )
        axis.set_xlabel(r"Terrain irregularity, $\Delta h$ (m)")
        axis.set_ylabel(ylabel)
        axis.grid(True)
        axis.legend()
        return _save_figure(figure, Path(output_base))


def generate_all_figures(
    experiment_dir: str | Path | None = None,
    *,
    config: str | Path = DEFAULT_CONFIG,
    publication_dir: str | Path | None = None,
    terrain_delta_h: float | None = None,
    manuscript_names: bool = True,
) -> list[Path]:
    """Generate the six ordered legacy-style Paper 1 figures as PNG and PDF."""
    spec = publication_spec(config)
    source_paths = experiment_paths(
        config,
        experiment_dir=experiment_dir,
    )
    output_paths = publication_paths(
        config,
        publication_dir=publication_dir,
    ).ensure()
    if terrain_delta_h is None:
        terrain_delta_h = float(
            spec.options.get(
                "comparison_terrain_delta_h",
                DEFAULT_COMPARISON_TERRAIN_DELTA_H,
            )
        )
    assert_publication_ready(source_paths)
    aggregate = load_core_aggregate(source_paths.experiment_dir)
    slug = _terrain_slug(terrain_delta_h)
    if manuscript_names:
        names = {
            "energy_breakdown": f"figure_01_energy_breakdown_delta_h_{slug}",
            "node_ptx": "figure_02_final_node_ptx",
            "relay_ptx": "figure_03_final_relay_ptx",
            "delivery_by_horizon_percent": "figure_04_pdr",
            "energy_per_delivered_packet_J": "figure_05_energy_per_delivered_packet",
            "avg_e2e_delay_ms": "figure_06_delay",
        }
    else:
        names = {
            "energy_breakdown": f"terrain_energy_breakdown_delta_h_{slug}",
            "node_ptx": "terrain_final_node_ptx",
            "relay_ptx": "terrain_final_relay_ptx",
            "delivery_by_horizon_percent": "terrain_pdr",
            "energy_per_delivered_packet_J": "terrain_energy_per_delivered_packet",
            "avg_e2e_delay_ms": "terrain_delay",
        }
    generated: list[Path] = []
    generated.extend(
        generate_energy_breakdown_figure(
            aggregate,
            terrain_delta_h=terrain_delta_h,
            output_base=output_paths.figures / names["energy_breakdown"],
        )
    )
    generated.extend(
        generate_final_ptx_figure(
            aggregate,
            metric="avg_final_node_ptx",
            series=FINAL_NODE_SERIES,
            ylabel="Average final node transmit power (dBm)",
            output_base=output_paths.figures / names["node_ptx"],
        )
    )
    generated.extend(
        generate_final_ptx_figure(
            aggregate,
            metric="avg_final_relay_ptx",
            series=FINAL_RELAY_SERIES,
            ylabel="Average final relay transmit power (dBm)",
            output_base=output_paths.figures / names["relay_ptx"],
        )
    )
    for metric, ylabel, _ in FIGURE_METRICS:
        generated.extend(
            generate_metric_figure(
                aggregate,
                metric=metric,
                ylabel=ylabel,
                output_base=output_paths.figures / names[metric],
            )
        )
    return generated


__all__ = [
    "ENERGY_BREAKDOWN_METRICS",
    "FINAL_NODE_SERIES",
    "FINAL_RELAY_SERIES",
    "FIGURE_FONT_SIZE",
    "FIGURE_METRICS",
    "FIGURE_STYLE",
    "MODE_COLORS",
    "MODE_STYLES",
    "PLOT_MODE_ORDER",
    "generate_all_figures",
    "generate_energy_breakdown_figure",
    "generate_final_ptx_figure",
    "generate_metric_figure",
]
