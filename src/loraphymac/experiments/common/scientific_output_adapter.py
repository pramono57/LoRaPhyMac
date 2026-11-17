"""Compatibility adapter between multiseed runs and managed persistence.

Omitted ``persistence_profile`` deliberately remains on the legacy writer path.
The adapter is only for explicitly selected managed profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from loraphymac.experiments.common.contract_registry import experiment_run_tables
from loraphymac.experiments.common.scientific_output_manager import (
    OutputWriteResult,
    ScientificOutputManager,
)


@dataclass(frozen=True, slots=True)
class PersistenceRoute:
    manager_enabled: bool
    profile: str | None


def persistence_route(config: Mapping[str, Any]) -> PersistenceRoute:
    """Select legacy unless a persistence profile was explicitly configured."""
    value = config.get("persistence_profile")
    return PersistenceRoute(value is not None, None if value is None else str(value))


def persist_multiseed_run(
    *,
    experiment_dir: str | Path,
    experiment_id: str,
    run_id: str,
    config: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
    summary: Mapping[str, Any],
    outputs: Mapping[str, Any],
    annotated_core_summary: pd.DataFrame,
    annotated_tables: Mapping[str, pd.DataFrame],
    annotated_relay_bottleneck: pd.DataFrame,
    status_payload: Mapping[str, Any],
    resolved_config_payload: Mapping[str, Any],
    pipeline_table_filenames: Mapping[str, str],
    validation_results: Mapping[str, bool],
) -> OutputWriteResult:
    """Persist a completed simulation through one atomic managed writer.

    Pipeline-standard filenames are retained so existing master aggregation
    and postprocessors remain readers, not secondary run-output writers.
    """
    route = persistence_route(config)
    if not route.manager_enabled:
        raise ValueError("managed adapter called without explicit persistence_profile")

    tables: dict[str, pd.DataFrame] = {
        key: value.copy() for key, value in annotated_tables.items()
        if isinstance(value, pd.DataFrame)
    }
    registered = experiment_run_tables(config)
    compatibility_keys = set(pipeline_table_filenames) | set(registered)
    compatibility_keys.add("relay_bottleneck_metrics_df")
    for key in compatibility_keys:
        if key not in tables:
            original = outputs.get(key)
            if isinstance(original, pd.DataFrame):
                tables[key] = original.copy()
            else:
                tables[key] = pd.DataFrame()
    tables["core_summary_df"] = annotated_core_summary.copy()
    tables["relay_bottleneck_metrics_df"] = annotated_relay_bottleneck.copy()

    filename_overrides = dict(pipeline_table_filenames)
    filename_overrides.update(registered)
    filename_overrides["relay_bottleneck_metrics_df"] = "relay_bottleneck_metrics.csv"

    manager = ScientificOutputManager(experiment_dir)
    return manager.write_run(
        experiment_id=experiment_id,
        run_id=run_id,
        config=config,
        summary=summary,
        tables=tables,
        validation_tables=outputs,
        source_provenance=source_provenance,
        validation_results=validation_results,
        additional_table_keys=tuple(sorted(compatibility_keys)),
        table_filename_overrides=filename_overrides,
        pipeline_compatibility={
            "core_summary.csv": annotated_core_summary,
            "status.json": status_payload,
        },
        resolved_config_payload=resolved_config_payload,
    )


__all__ = ["PersistenceRoute", "persist_multiseed_run", "persistence_route"]
