"""Paper 1 publication-table builders.

The shared pipeline owns raw and aggregate evidence. This module owns only the
APC-specific transformation from that evidence into manuscript-ready tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from loraphymac.experiments.common.multiseed_pipeline import atomic_write_csv
from loraphymac.experiments.common.outputs import ExperimentArtifactPaths
from loraphymac.publications.common import (
    PublicationArtifactPaths,
    PublicationSpec,
    load_publication_spec,
)


PUBLICATION_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PUBLICATION_DIR / "configs" / "publication_v1.json"
DEFAULT_COMPARISON_TERRAIN_DELTA_H = 90

APC_MODE_ORDER = ("FF", "AF", "FA", "AA")
APC_MODE_LABELS = {
    ("fixed", "fixed"): "FF",
    ("adaptive", "fixed"): "AF",
    ("fixed", "adaptive"): "FA",
    ("adaptive", "adaptive"): "AA",
}

TABLE_09_METRICS = (
    ("delivery_by_horizon_percent", "PDR (%)"),
    ("avg_e2e_delay_ms", "Average End-to-End Delay (ms)"),
    ("total_network_energy_J", "Total Energy (J)"),
    ("total_constrained_energy_J", "Constrained Energy (J)"),
    ("energy_per_delivered_packet_J", "Energy per Delivered Packet (J)"),
    (
        "energy_per_delivered_packet_constrained_J",
        "Constrained Energy per Delivered Packet (J)",
    ),
    ("avg_final_node_ptx", "Average Final Node PTx (dBm)"),
    ("avg_final_relay_ptx", "Average Final Relay PTx (dBm)"),
)

TABLE_10_METRICS = (
    "total_node_energy_J",
    "total_relay_energy_J",
    "total_constrained_energy_J",
    "energy_per_delivered_packet_constrained_J",
)


def publication_spec(config: str | Path = DEFAULT_CONFIG) -> PublicationSpec:
    """Load the mapping from the publication to its source experiment."""
    return load_publication_spec(config)


def publication_sources(config: str | Path = DEFAULT_CONFIG) -> dict[str, Path]:
    """Return role-labelled source experiments, including legacy single-source configs."""
    spec = publication_spec(config)
    if spec.sources:
        return dict(spec.sources)
    if spec.source_experiment_dir is None:
        raise ValueError("Publication configuration has no experiment source")
    return {"terrain_sweep": spec.source_experiment_dir}


def experiment_paths(
    config: str | Path = DEFAULT_CONFIG,
    *,
    experiment_dir: str | Path | None = None,
) -> ExperimentArtifactPaths:
    """Resolve read-only source experiment paths for this publication."""
    source = (
        publication_sources(config).get("terrain_sweep")
        if experiment_dir is None
        else Path(experiment_dir)
    )
    if source is None:
        raise ValueError("Publication sources must include the terrain_sweep role")
    return ExperimentArtifactPaths(Path(source))


def publication_paths(
    config: str | Path = DEFAULT_CONFIG,
    *,
    publication_dir: str | Path | None = None,
) -> PublicationArtifactPaths:
    """Resolve the publication-only destination for tables and figures."""
    destination = (
        publication_spec(config).output_dir if publication_dir is None else Path(publication_dir)
    )
    return PublicationArtifactPaths(Path(destination))


def assert_publication_ready(paths: ExperimentArtifactPaths) -> None:
    """Reject publication output when an on-disk experiment is incomplete."""
    if not paths.status.exists():
        return

    try:
        status = json.loads(paths.status.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read Paper 1 experiment status: {paths.status}") from exc

    if status.get("status") != "PASS" or not status.get("seed_grid_complete", False):
        raise ValueError(
            "Paper 1 seed grid is incomplete or the experiment did not pass; publication "
            f"tables and figures were not generated. Inspect {paths.status}."
        )


def load_core_aggregate(
    experiment_dir: str | Path | None = None,
    *,
    config: str | Path = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Load the authoritative long-form aggregate produced by the pipeline."""
    paths = (
        experiment_paths(config)
        if experiment_dir is None
        else ExperimentArtifactPaths(Path(experiment_dir))
    )
    source = paths.aggregate / "core_metrics_long.csv"
    if not source.exists():
        raise FileNotFoundError(
            f"Paper 1 aggregate is missing: {source}. Run the multi-seed experiment first."
        )
    return pd.read_csv(source)


def add_apc_mode(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the frozen two-letter APC mode label to an aggregate frame."""
    required = {"node_ptx_mode", "relay_ptx_mode"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Paper 1 aggregate is missing factor columns: {sorted(missing)}")
    result = frame.copy()
    pairs = list(zip(result["node_ptx_mode"], result["relay_ptx_mode"]))
    result["Mode"] = [APC_MODE_LABELS.get(pair) for pair in pairs]
    if result["Mode"].isna().any():
        invalid = sorted(
            set(pair for pair, mode in zip(pairs, result["Mode"]) if mode is None),
            key=repr,
        )
        raise ValueError(f"Unknown Paper 1 APC mode combinations: {invalid}")
    return result


def _terrain_slug(value: float) -> str:
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else str(numeric).replace(".", "p")


def _terrain_metric_matrix(
    aggregate: pd.DataFrame,
    *,
    terrain_delta_h: float,
    metrics: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = add_apc_mode(aggregate)
    required = {"terrain_delta_h", "metric", "mean"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Paper 1 aggregate is missing columns: {sorted(missing)}")

    terrain = pd.to_numeric(frame["terrain_delta_h"], errors="coerce")
    selected = frame.loc[(terrain - float(terrain_delta_h)).abs() < 1e-9].copy()
    if selected.empty:
        raise ValueError(f"No Paper 1 aggregate rows exist for terrain_delta_h={terrain_delta_h}")
    selected = selected.loc[selected["metric"].isin(metrics)].copy()

    duplicates = selected.duplicated(subset=["metric", "Mode"], keep=False)
    if duplicates.any():
        pairs = selected.loc[duplicates, ["metric", "Mode"]].drop_duplicates()
        raise ValueError(
            "Paper 1 aggregate contains duplicate metric/mode rows at "
            f"terrain_delta_h={terrain_delta_h}: {pairs.to_dict(orient='records')}"
        )

    matrix = selected.pivot(index="metric", columns="Mode", values="mean")
    matrix = matrix.reindex(index=list(metrics), columns=list(APC_MODE_ORDER))
    if matrix.isna().any().any():
        missing_cells = [
            f"{metric}/{mode}"
            for metric in metrics
            for mode in APC_MODE_ORDER
            if pd.isna(matrix.loc[metric, mode])
        ]
        raise ValueError(
            "Paper 1 aggregate is incomplete for publication at "
            f"terrain_delta_h={terrain_delta_h}; missing {missing_cells}"
        )
    return selected, matrix.astype(float)


def build_table_09(
    aggregate: pd.DataFrame,
    *,
    terrain_delta_h: float = DEFAULT_COMPARISON_TERRAIN_DELTA_H,
) -> pd.DataFrame:
    """Build the legacy-parity APC performance table for the four modes."""
    metrics = tuple(metric for metric, _ in TABLE_09_METRICS)
    _, matrix = _terrain_metric_matrix(
        aggregate,
        terrain_delta_h=terrain_delta_h,
        metrics=metrics,
    )
    labels = dict(TABLE_09_METRICS)

    rows: list[dict[str, float | str]] = []
    for metric in metrics[:6]:
        rows.append(
            {
                "Parameter": labels[metric],
                **{mode: matrix.loc[metric, mode] for mode in APC_MODE_ORDER},
            }
        )

    for metric, label in (
        ("total_network_energy_J", "Energy Saving vs FF (%)"),
        ("total_constrained_energy_J", "Constrained Energy Saving vs FF (%)"),
    ):
        baseline = float(matrix.loc[metric, "FF"])
        if baseline <= 0:
            raise ValueError(f"Cannot calculate {label}: FF baseline must be positive")
        rows.append(
            {
                "Parameter": label,
                **{
                    mode: (baseline - float(matrix.loc[metric, mode])) / baseline * 100.0
                    for mode in APC_MODE_ORDER
                },
            }
        )

    for metric in metrics[6:]:
        rows.append(
            {
                "Parameter": labels[metric],
                **{mode: matrix.loc[metric, mode] for mode in APC_MODE_ORDER},
            }
        )
    return pd.DataFrame(rows, columns=["Parameter", *APC_MODE_ORDER]).round(3)


def build_table_10(
    aggregate: pd.DataFrame,
    *,
    terrain_delta_h: float = DEFAULT_COMPARISON_TERRAIN_DELTA_H,
) -> pd.DataFrame:
    """Build the constrained-energy decomposition and savings table."""
    _, matrix = _terrain_metric_matrix(
        aggregate,
        terrain_delta_h=terrain_delta_h,
        metrics=TABLE_10_METRICS,
    )
    baseline = float(matrix.loc["total_constrained_energy_J", "FF"])
    if baseline <= 0:
        raise ValueError("Cannot calculate constrained-energy saving: FF baseline must be positive")

    rows = []
    for mode in APC_MODE_ORDER:
        constrained = float(matrix.loc["total_constrained_energy_J", mode])
        rows.append(
            {
                "Mode": mode,
                "Node Energy (J)": matrix.loc["total_node_energy_J", mode],
                "Relay Energy (J)": matrix.loc["total_relay_energy_J", mode],
                "Constrained Energy (J)": constrained,
                "Saving vs FF (%)": (baseline - constrained) / baseline * 100.0,
                "Constrained Energy / Delivered Packet (J)": matrix.loc[
                    "energy_per_delivered_packet_constrained_J", mode
                ],
            }
        )
    return pd.DataFrame(rows).round(3)


def build_publication_tables(
    experiment_dir: str | Path | None = None,
    *,
    config: str | Path = DEFAULT_CONFIG,
    publication_dir: str | Path | None = None,
    terrain_delta_h: float | None = None,
    manuscript_names: bool = True,
) -> dict[str, Path]:
    """Create the complete legacy-parity Paper 1 table set."""
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
    table_09 = build_table_09(aggregate, terrain_delta_h=terrain_delta_h)
    table_10 = build_table_10(aggregate, terrain_delta_h=terrain_delta_h)
    table_09_long = table_09.melt(
        id_vars="Parameter",
        value_vars=list(APC_MODE_ORDER),
        var_name="Mode",
        value_name="Value",
    )

    slug = _terrain_slug(terrain_delta_h)
    if manuscript_names:
        table_09_name = f"table_09_apc_performance_delta_h_{slug}.csv"
        table_09_long_name = f"table_09_apc_performance_delta_h_{slug}_long.csv"
        table_10_name = f"table_10_constrained_energy_delta_h_{slug}.csv"
    else:
        table_09_name = f"terrain_apc_performance_delta_h_{slug}.csv"
        table_09_long_name = f"terrain_apc_performance_delta_h_{slug}_long.csv"
        table_10_name = f"terrain_constrained_energy_delta_h_{slug}.csv"
    table_09_path = output_paths.tables / table_09_name
    table_09_long_path = output_paths.tables / table_09_long_name
    table_10_path = output_paths.tables / table_10_name
    atomic_write_csv(table_09_path, table_09)
    atomic_write_csv(table_09_long_path, table_09_long)
    atomic_write_csv(table_10_path, table_10)
    return {
        "comparison": table_09_path,
        "comparison_long": table_09_long_path,
        "table_09": table_09_path,
        "table_09_long": table_09_long_path,
        "table_10": table_10_path,
    }


__all__ = [
    "APC_MODE_LABELS",
    "APC_MODE_ORDER",
    "DEFAULT_COMPARISON_TERRAIN_DELTA_H",
    "TABLE_09_METRICS",
    "TABLE_10_METRICS",
    "add_apc_mode",
    "assert_publication_ready",
    "build_publication_tables",
    "build_table_09",
    "build_table_10",
    "experiment_paths",
    "load_core_aggregate",
    "publication_paths",
    "publication_sources",
    "publication_spec",
]
