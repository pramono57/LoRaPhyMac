"""Diagnostic figures for one adaptive-power simulation run.

These builders are shared by exploratory and contracted one-run workflows.
They consume only the stable
:class:`loraphymac.SimulationResult` contract and therefore remain independent
of the internal event runner.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from loraphymac import SimulationResult


DEFAULT_TOLERANCE_DB = 2.0
DEFAULT_NODE_MAX_TIME_S = 300.0
DEFAULT_RELAY_MAX_TIME_S = 20.0

MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p", "*")
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
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
}


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def _filter_entity(frame: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    _require_columns(frame, ("entity_type",), "history table")
    return frame.loc[frame["entity_type"].astype(str).str.lower().eq(entity_type.lower())].copy()


def select_representative_nodes(
    ptx_history: pd.DataFrame,
    margin_history: pd.DataFrame,
    *,
    target_margin_db: float,
    maximum: int = 4,
) -> list[str]:
    """Select deterministic representative nodes using the legacy criteria."""
    if maximum < 1:
        return []

    node_ptx = _filter_entity(ptx_history, "node")
    if node_ptx.empty:
        return []
    _require_columns(
        node_ptx,
        ("entity_id", "time", "old_ptx", "new_ptx"),
        "PTx history",
    )

    numeric = node_ptx.copy()
    for column in ("time", "old_ptx", "new_ptx"):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    numeric = numeric.dropna(subset=["entity_id", "time", "old_ptx", "new_ptx"])
    if numeric.empty:
        return []

    ptx_summary = (
        numeric.sort_values("time")
        .groupby("entity_id", as_index=False)
        .agg(
            first_ptx=("old_ptx", "first"),
            final_ptx=("new_ptx", "last"),
            min_ptx=("new_ptx", "min"),
            max_ptx=("new_ptx", "max"),
            update_count=("new_ptx", "count"),
        )
    )
    ptx_summary["entity_key"] = ptx_summary["entity_id"].astype(str)
    ptx_summary["ptx_reduction_db"] = ptx_summary["first_ptx"] - ptx_summary["final_ptx"]
    ptx_summary["ptx_range_db"] = ptx_summary["max_ptx"] - ptx_summary["min_ptx"]

    node_margin = _filter_entity(margin_history, "node")
    if not node_margin.empty:
        _require_columns(node_margin, ("entity_id", "time", "margin"), "margin history")
        margin_numeric = node_margin.copy()
        margin_numeric["time"] = pd.to_numeric(margin_numeric["time"], errors="coerce")
        margin_numeric["margin"] = pd.to_numeric(margin_numeric["margin"], errors="coerce")
        margin_summary = (
            margin_numeric.dropna(subset=["entity_id", "time", "margin"])
            .sort_values("time")
            .groupby("entity_id", as_index=False)
            .agg(final_margin=("margin", "last"))
        )
        ptx_summary = ptx_summary.merge(margin_summary, on="entity_id", how="left")
    else:
        ptx_summary["final_margin"] = float("nan")
    ptx_summary["target_distance"] = (ptx_summary["final_margin"] - float(target_margin_db)).abs()

    selected: list[str] = []

    def pick(frame: pd.DataFrame, sort_columns: list[str], ascending: list[bool]) -> None:
        candidates = frame.loc[~frame["entity_key"].isin(selected)].copy()
        if candidates.empty or len(selected) >= maximum:
            return
        ordered = candidates.sort_values(
            [*sort_columns, "entity_key"],
            ascending=[*ascending, True],
            na_position="last",
        )
        selected.append(str(ordered.iloc[0]["entity_key"]))

    pick(ptx_summary, ["ptx_reduction_db", "update_count"], [False, False])
    pick(ptx_summary, ["ptx_range_db"], [False])
    pick(ptx_summary.dropna(subset=["target_distance"]), ["target_distance"], [True])
    pick(
        ptx_summary.loc[ptx_summary["update_count"] >= 2],
        ["final_ptx"],
        [False],
    )
    return selected[:maximum]


def select_representative_relays(
    ptx_history: pd.DataFrame,
    margin_history: pd.DataFrame,
) -> list[str]:
    """Return every observed relay in deterministic order."""
    observed: set[str] = set()
    for frame in (ptx_history, margin_history):
        entity_frame = _filter_entity(frame, "relay")
        if not entity_frame.empty:
            _require_columns(entity_frame, ("entity_id",), "relay history")
            observed.update(entity_frame["entity_id"].dropna().astype(str))
    return sorted(observed)


def _mtype_to_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        name = str(value).rsplit(".", maxsplit=1)[-1].upper()
        return {"DATA_UP": 1, "DATA_UP_FWD": 2}.get(name)


def map_power_updates_to_next_transmission(
    ptx_history: pd.DataFrame,
    event_log: pd.DataFrame,
    *,
    entity_ids: Sequence[str],
    entity_type: str,
    max_time_s: float,
    include_initial: bool = True,
) -> pd.DataFrame:
    """Map APC decisions to the next uplink where their PTx is used physically."""
    entity_type = entity_type.lower().strip()
    if entity_type not in {"node", "relay"}:
        raise ValueError(f"Unsupported entity_type={entity_type!r}")
    if not entity_ids or ptx_history.empty:
        return pd.DataFrame(columns=["entity_id", "entity_type", "time", "ptx_dbm"])

    _require_columns(
        ptx_history,
        ("entity_id", "entity_type", "time", "old_ptx", "new_ptx"),
        "PTx history",
    )
    _require_columns(event_log, ("event", "node", "mtype", "time"), "event log")

    entity_keys = {str(entity_id) for entity_id in entity_ids}
    ptx = _filter_entity(ptx_history, entity_type)
    ptx["entity_id"] = ptx["entity_id"].astype(str)
    ptx["time"] = pd.to_numeric(ptx["time"], errors="coerce")
    ptx["old_ptx"] = pd.to_numeric(ptx["old_ptx"], errors="coerce")
    ptx["new_ptx"] = pd.to_numeric(ptx["new_ptx"], errors="coerce")
    ptx = ptx.loc[ptx["entity_id"].isin(entity_keys) & ptx["time"].le(float(max_time_s))].dropna(
        subset=["time", "old_ptx", "new_ptx"]
    )
    if ptx.empty:
        return pd.DataFrame(columns=["entity_id", "entity_type", "time", "ptx_dbm"])

    events = event_log.copy()
    events["node"] = events["node"].astype(str)
    events["time"] = pd.to_numeric(events["time"], errors="coerce")
    events["mtype_int"] = events["mtype"].map(_mtype_to_int)
    uplink_mtype = 1 if entity_type == "node" else 2
    transmissions = events.loc[
        events["event"].astype(str).str.upper().eq("TX")
        & events["node"].isin(entity_keys)
        & events["mtype_int"].eq(uplink_mtype)
        & events["time"].le(float(max_time_s)),
        ["node", "time"],
    ].dropna(subset=["time"])

    rows: list[dict[str, Any]] = []
    for update in ptx.itertuples(index=False):
        next_times = transmissions.loc[
            transmissions["node"].eq(update.entity_id)
            & transmissions["time"].gt(float(update.time)),
            "time",
        ]
        if next_times.empty:
            continue
        rows.append(
            {
                "entity_id": update.entity_id,
                "entity_type": entity_type,
                "time": float(next_times.min()),
                "ptx_dbm": float(update.new_ptx),
            }
        )

    if include_initial:
        initial = ptx.sort_values("time").groupby("entity_id", as_index=False)["old_ptx"].first()
        rows.extend(
            {
                "entity_id": row.entity_id,
                "entity_type": entity_type,
                "time": 0.0,
                "ptx_dbm": float(row.old_ptx),
            }
            for row in initial.itertuples(index=False)
        )

    if not rows:
        return pd.DataFrame(columns=["entity_id", "entity_type", "time", "ptx_dbm"])
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["entity_id", "time"], keep="last")
        .sort_values(["entity_id", "time"])
        .reset_index(drop=True)
    )


def _plot_margin(
    margin_history: pd.DataFrame,
    *,
    entity_ids: Sequence[str],
    target_margin_db: float,
    tolerance_db: float,
    max_time_s: float,
    legend_columns: int,
):
    _require_columns(margin_history, ("entity_id", "time", "margin"), "margin history")
    entity_keys = [str(entity_id) for entity_id in entity_ids]
    frame = margin_history.copy()
    frame["entity_id"] = frame["entity_id"].astype(str)
    frame["time"] = pd.to_numeric(frame["time"], errors="coerce")
    frame["margin"] = pd.to_numeric(frame["margin"], errors="coerce")
    frame = frame.loc[
        frame["entity_id"].isin(entity_keys) & frame["time"].le(float(max_time_s))
    ].dropna(subset=["time", "margin"])
    if frame.empty:
        return None

    figure, axis = plt.subplots(figsize=(8, 5))
    marker_map = {
        entity_id: MARKERS[index % len(MARKERS)]
        for index, entity_id in enumerate(entity_keys)
    }
    for entity_id, group in frame.groupby("entity_id", sort=True):
        group = group.sort_values("time")
        axis.plot(
            group["time"],
            group["margin"],
            marker=marker_map.get(entity_id, "o"),
            markersize=6,
            linewidth=1.8,
            label=entity_id,
        )
    axis.axhline(target_margin_db, linestyle="--", linewidth=1.5, label="Target margin")
    axis.axhline(
        target_margin_db + tolerance_db,
        linestyle=":",
        linewidth=1.3,
        label=r"$M_{\mathrm{target}}+\delta$",
    )
    axis.axhline(
        target_margin_db - tolerance_db,
        linestyle=":",
        linewidth=1.3,
        label=r"$M_{\mathrm{target}}-\delta$",
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Link margin (dB)")
    axis.set_xlim(0, max_time_s)
    axis.grid(True, alpha=0.3)
    legend = axis.legend(ncols=legend_columns, loc="upper right")
    figure.tight_layout()
    legend.set_bbox_to_anchor((1.0, 1.04), transform=axis.transAxes)
    return figure


def _plot_ptx(
    used_power: pd.DataFrame,
    *,
    entity_ids: Sequence[str],
    max_time_s: float,
    legend_columns: int,
):
    if used_power.empty:
        return None
    figure, axis = plt.subplots(figsize=(8, 5))
    entity_keys = [str(entity_id) for entity_id in entity_ids]
    marker_map = {
        entity_id: MARKERS[index % len(MARKERS)]
        for index, entity_id in enumerate(entity_keys)
    }
    for entity_id, group in used_power.groupby("entity_id", sort=True):
        group = group.sort_values("time")
        line = axis.step(
            group["time"],
            group["ptx_dbm"],
            where="post",
            linewidth=1.8,
            label=entity_id,
        )[0]
        axis.scatter(
            group["time"],
            group["ptx_dbm"],
            marker=marker_map.get(entity_id, "o"),
            s=45,
            color=line.get_color(),
        )

    initial_values = used_power.loc[used_power["time"].eq(0.0), "ptx_dbm"].unique()
    if len(initial_values) == 1:
        axis.axhline(
            float(initial_values[0]),
            linestyle=":",
            linewidth=1.2,
            label=r"Initial EIRP-limited $P_{tx}$",
        )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel(r"Transmit power, $P_{tx}$ (dBm)")
    axis.set_xlim(0, max_time_s)
    axis.grid(True, alpha=0.3)
    legend = axis.legend(ncols=legend_columns, loc="lower left")
    figure.tight_layout()
    legend.set_bbox_to_anchor((0.02, 0.05), transform=axis.transAxes)
    return figure


def _save_figure(figure, output_base: Path, formats: Sequence[str]) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for raw_format in formats:
        suffix = str(raw_format).lower().lstrip(".")
        if suffix not in {"png", "pdf", "jpg", "jpeg"}:
            raise ValueError(f"Unsupported figure format: {raw_format!r}")
        path = output_base.with_suffix(f".{suffix}")
        options = {"dpi": 300, "facecolor": "white"} if suffix != "pdf" else {}
        figure.savefig(path, bbox_inches="tight", **options)
        paths.append(path)
    plt.close(figure)
    return paths


def generate_single_run_figures(
    result: SimulationResult,
    *,
    output_dir: str | Path | None = None,
    tolerance_db: float = DEFAULT_TOLERANCE_DB,
    node_max_time_s: float = DEFAULT_NODE_MAX_TIME_S,
    relay_max_time_s: float = DEFAULT_RELAY_MAX_TIME_S,
    formats: Sequence[str] = ("png", "pdf"),
) -> dict[str, list[Path]]:
    """Generate the legacy single-run diagnostics through the stable API.

    Empty node or relay histories are reported as skipped figure groups instead
    of failing the whole notebook. The topology file is produced by the
    simulator itself when ``plot_routing=True``.
    """
    run_directory = Path(result.resolved_config["output_dir"])
    destination = Path(output_dir) if output_dir is not None else run_directory / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    generated: dict[str, list[Path]] = {}

    topology_path = run_directory / f"{result.run_id}_routing_topology.jpg"
    if topology_path.is_file():
        generated["routing_topology"] = [topology_path]

    ptx_history = result.ptx_history
    margin_history = result.margin_history
    event_log = result.table("event_log_df")
    node_target = float(result.resolved_config["node_target_margin_db"])
    relay_target = float(result.resolved_config["relay_target_margin_db"])

    nodes = select_representative_nodes(
        ptx_history,
        margin_history,
        target_margin_db=node_target,
    )
    relays = select_representative_relays(ptx_history, margin_history)

    specifications = (
        (
            "node_margin_convergence",
            "fig_single_node_margin_convergence",
            _filter_entity(margin_history, "node"),
            nodes,
            node_target,
            node_max_time_s,
            2,
        ),
        (
            "relay_margin_convergence",
            "fig_single_relay_margin_convergence",
            _filter_entity(margin_history, "relay"),
            relays,
            relay_target,
            relay_max_time_s,
            1,
        ),
    )
    with plt.rc_context(FIGURE_STYLE):
        for key, filename, history, entities, target, max_time, legend_columns in specifications:
            if not entities or history.empty:
                warnings.warn(f"Skipped {key}: no adaptive history is available", stacklevel=2)
                continue
            figure = _plot_margin(
                history,
                entity_ids=entities,
                target_margin_db=target,
                tolerance_db=tolerance_db,
                max_time_s=max_time,
                legend_columns=legend_columns,
            )
            if figure is not None:
                generated[key] = _save_figure(figure, destination / filename, formats)

        for entity_type, entities, max_time, legend_columns in (
            ("node", nodes, node_max_time_s, 2),
            ("relay", relays, relay_max_time_s, 1),
        ):
            key = f"{entity_type}_ptx_next_tx"
            if not entities:
                warnings.warn(f"Skipped {key}: no adaptive history is available", stacklevel=2)
                continue
            used_power = map_power_updates_to_next_transmission(
                ptx_history,
                event_log,
                entity_ids=entities,
                entity_type=entity_type,
                max_time_s=max_time,
            )
            figure = _plot_ptx(
                used_power,
                entity_ids=entities,
                max_time_s=max_time,
                legend_columns=legend_columns,
            )
            if figure is None:
                warnings.warn(
                    f"Skipped {key}: no subsequent uplink transmission is available",
                    stacklevel=2,
                )
                continue
            generated[key] = _save_figure(
                figure,
                destination / f"fig_single_{entity_type}_ptx_next_tx",
                formats,
            )
    return generated


__all__ = [
    "DEFAULT_NODE_MAX_TIME_S",
    "DEFAULT_RELAY_MAX_TIME_S",
    "DEFAULT_TOLERANCE_DB",
    "FIGURE_FONT_SIZE",
    "FIGURE_STYLE",
    "generate_single_run_figures",
    "map_power_updates_to_next_transmission",
    "select_representative_nodes",
    "select_representative_relays",
]
