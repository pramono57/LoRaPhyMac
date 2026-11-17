"""Paper 2 scalability and relay-bottleneck figure builders."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from loraphymac.publications.paper2_scalability.analysis import (
    DEFAULT_CONFIG,
    experiment_paths,
    load_core_aggregate,
    publication_paths,
)


FIGURE_METRICS = (
    ("delivery_by_horizon_percent", "Delivery by horizon / PDR (%)", "figure_01_delivery"),
    ("relay_fifo_wait_p95_ms", "Relay FIFO wait p95 (ms)", "figure_02_relay_wait"),
    ("total_network_energy_J", "Total network energy (J)", "figure_03_total_energy"),
)


def _scenario_label(duty_cycle, capacity) -> str:
    duty = "∞" if pd.isna(duty_cycle) else f"{float(duty_cycle) * 100:g}%"
    queue = "∞" if pd.isna(capacity) else f"{float(capacity):g}"
    return f"DC={duty}, Q={queue}"


def _save_figure(figure, output_base: Path) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]
    figure.savefig(paths[0], dpi=300, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    plt.close(figure)
    return paths


def generate_scalability_figure(
    aggregate: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    output_base: str | Path,
) -> list[Path]:
    """Plot a metric against node count, faceted by traffic period."""
    required = {
        "node_num",
        "node_period",
        "relay_duty_cycle",
        "relay_tx_waiting_queue_capacity_packets",
        "metric",
        "mean",
    }
    missing = required.difference(aggregate.columns)
    if missing:
        raise ValueError(f"Paper 2 aggregate is missing columns: {sorted(missing)}")
    frame = aggregate.loc[aggregate["metric"] == metric].copy()
    if frame.empty:
        raise ValueError(f"Paper 2 aggregate does not contain metric {metric!r}")

    periods = sorted(pd.to_numeric(frame["node_period"], errors="raise").unique())
    figure, axes = plt.subplots(
        1,
        len(periods),
        figsize=(5.2 * len(periods), 4.5),
        sharey=True,
        squeeze=False,
    )
    for axis, period in zip(axes[0], periods):
        period_frame = frame.loc[pd.to_numeric(frame["node_period"], errors="raise") == period]
        grouped = period_frame.groupby(
            ["relay_duty_cycle", "relay_tx_waiting_queue_capacity_packets"],
            dropna=False,
            sort=True,
        )
        for (duty_cycle, capacity), scenario in grouped:
            scenario = scenario.copy()
            scenario["node_num"] = pd.to_numeric(scenario["node_num"], errors="raise")
            scenario = scenario.sort_values("node_num")
            x = scenario["node_num"].to_numpy(dtype=float)
            mean = pd.to_numeric(scenario["mean"], errors="raise").to_numpy(dtype=float)
            axis.plot(
                x,
                mean,
                marker="o",
                linewidth=1.5,
                label=_scenario_label(duty_cycle, capacity),
            )
            if {"ci_lower", "ci_upper"}.issubset(scenario.columns):
                lower = pd.to_numeric(scenario["ci_lower"], errors="coerce").to_numpy(dtype=float)
                upper = pd.to_numeric(scenario["ci_upper"], errors="coerce").to_numpy(dtype=float)
                axis.fill_between(x, lower, upper, alpha=0.1)
        axis.set_title(f"Traffic period = {period:g} s")
        axis.set_xlabel("Number of end nodes")
        axis.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncols=min(3, math.ceil(len(labels) / 2)))
        figure.subplots_adjust(bottom=0.24)
    return _save_figure(figure, Path(output_base))


def generate_all_figures(
    experiment_dir: str | Path | None = None,
    *,
    config: str | Path = DEFAULT_CONFIG,
    publication_dir: str | Path | None = None,
) -> list[Path]:
    """Generate the standard Paper 2 figure set in PNG and PDF formats."""
    source_paths = experiment_paths(
        config,
        experiment_dir=experiment_dir,
    )
    output_paths = publication_paths(
        config,
        publication_dir=publication_dir,
    ).ensure()
    aggregate = load_core_aggregate(source_paths.experiment_dir)
    generated: list[Path] = []
    for metric, ylabel, filename in FIGURE_METRICS:
        generated.extend(
            generate_scalability_figure(
                aggregate,
                metric=metric,
                ylabel=ylabel,
                output_base=output_paths.figures / filename,
            )
        )
    return generated


__all__ = ["FIGURE_METRICS", "generate_all_figures", "generate_scalability_figure"]
