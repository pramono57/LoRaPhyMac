"""Paper 2 scalability and relay-bottleneck table builders."""

from __future__ import annotations

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

CORE_METRICS = (
    "delivery_by_horizon_percent",
    "terminal_loss_percent",
    "censored_fraction_percent",
    "avg_e2e_delay_ms",
    "relay_fifo_wait_mean_ms_packet_weighted",
    "relay_fifo_wait_p95_ms",
    "total_network_energy_J",
    "energy_per_delivered_packet_J",
)


def publication_spec(config: str | Path = DEFAULT_CONFIG) -> PublicationSpec:
    return load_publication_spec(config)


def experiment_paths(
    config: str | Path = DEFAULT_CONFIG,
    *,
    experiment_dir: str | Path | None = None,
) -> ExperimentArtifactPaths:
    source = (
        publication_spec(config).source_experiment_dir
        if experiment_dir is None
        else Path(experiment_dir)
    )
    return ExperimentArtifactPaths(Path(source))


def publication_paths(
    config: str | Path = DEFAULT_CONFIG,
    *,
    publication_dir: str | Path | None = None,
) -> PublicationArtifactPaths:
    destination = (
        publication_spec(config).output_dir if publication_dir is None else Path(publication_dir)
    )
    return PublicationArtifactPaths(Path(destination))


def load_core_aggregate(
    experiment_dir: str | Path | None = None,
    *,
    config: str | Path = DEFAULT_CONFIG,
) -> pd.DataFrame:
    paths = (
        experiment_paths(config)
        if experiment_dir is None
        else ExperimentArtifactPaths(Path(experiment_dir))
    )
    source = paths.aggregate / "core_metrics_long.csv"
    if not source.exists():
        raise FileNotFoundError(
            f"Paper 2 aggregate is missing: {source}. Run the multi-seed experiment first."
        )
    return pd.read_csv(source)


def _display_factors(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "relay_duty_cycle" in result:
        result["relay_duty_cycle"] = result["relay_duty_cycle"].map(
            lambda value: "unlimited" if pd.isna(value) else value
        )
    if "relay_tx_waiting_queue_capacity_packets" in result:
        column = "relay_tx_waiting_queue_capacity_packets"
        result[column] = result[column].map(lambda value: "unbounded" if pd.isna(value) else value)
    return result


def build_publication_tables(
    experiment_dir: str | Path | None = None,
    *,
    config: str | Path = DEFAULT_CONFIG,
    publication_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Create Paper 2 core and receiver tables in the standard output folder."""
    source_paths = experiment_paths(
        config,
        experiment_dir=experiment_dir,
    )
    output_paths = publication_paths(
        config,
        publication_dir=publication_dir,
    ).ensure()
    core = _display_factors(load_core_aggregate(source_paths.experiment_dir))
    factors = [
        column
        for column in (
            "node_num",
            "node_period",
            "relay_duty_cycle",
            "relay_tx_waiting_queue_capacity_packets",
        )
        if column in core.columns
    ]
    selected = core.loc[core["metric"].isin(CORE_METRICS)].copy()
    if selected.empty:
        raise ValueError("Paper 2 aggregate contains none of the required core metrics")
    wide = selected.set_index([*factors, "metric"])["mean"].unstack("metric")
    wide = wide.reindex(columns=[metric for metric in CORE_METRICS if metric in wide.columns])
    wide = wide.reset_index()

    core_path = output_paths.tables / "table_01_scalability_core_means.csv"
    core_long_path = output_paths.tables / "table_01_scalability_core_long.csv"
    atomic_write_csv(core_path, wide)
    atomic_write_csv(core_long_path, selected)

    outputs = {"core_means": core_path, "core_long": core_long_path}
    relay_source = source_paths.aggregate / "relay_bottleneck_metrics_long.csv"
    if relay_source.exists():
        relay = _display_factors(pd.read_csv(relay_source))
        relay_path = output_paths.tables / "table_02_relay_bottleneck_long.csv"
        atomic_write_csv(relay_path, relay)
        outputs["relay_bottleneck"] = relay_path
    return outputs


__all__ = [
    "CORE_METRICS",
    "build_publication_tables",
    "experiment_paths",
    "load_core_aggregate",
    "publication_paths",
    "publication_spec",
]
