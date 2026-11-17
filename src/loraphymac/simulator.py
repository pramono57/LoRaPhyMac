"""Typed public boundary around the single-simulation event runner.

New callers should use this module so internal runner changes do not affect
experiment or notebook code. :mod:`loraphymac.engine` remains only as a
backward-compatible facade.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pandas as pd

from loraphymac._version import SIMULATOR_VERSION
from loraphymac.config.defaults import DEFAULT_CONFIG as _DEFAULT_CONFIG
from loraphymac.config.defaults import build_config as _build_config
from loraphymac.config.validation import ConfigurationError, validate_configuration
from loraphymac.simulation.runner import (
    run_single_simulation as _run_single_simulation,
)


SIMULATOR_API_VERSION = "simulator_api_v1"


@dataclass(frozen=True)
class SimulationResult:
    """Structured result for one deterministic simulation run.

    The mappings are read-only snapshots. DataFrames are not copied by default
    because detailed event tables can be large; use :meth:`table` with
    ``copy=True`` before mutating one.
    """

    resolved_config: Mapping[str, Any]
    summary: Mapping[str, Any]
    tables: Mapping[str, pd.DataFrame]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resolved_config",
            MappingProxyType(deepcopy(dict(self.resolved_config))),
        )
        object.__setattr__(self, "summary", MappingProxyType(deepcopy(dict(self.summary))))
        object.__setattr__(self, "tables", MappingProxyType(dict(self.tables)))

    @property
    def run_id(self) -> Any:
        """Return the resolved run identifier, which may be ``None``."""
        return self.resolved_config.get("run_id")

    @property
    def ptx_history(self) -> pd.DataFrame:
        """Return the adaptive-power history table."""
        return self.table("ptx_history_df")

    @property
    def margin_history(self) -> pd.DataFrame:
        """Return the link-margin history table."""
        return self.table("margin_history_df")

    def table(self, name: str, *, copy: bool = False) -> pd.DataFrame:
        """Return one named output table with an optional defensive copy."""
        try:
            frame = self.tables[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.tables))
            raise KeyError(f"Unknown simulation table {name!r}; available: {available}") from exc
        return frame.copy(deep=True) if copy else frame

    def to_legacy_tuple(
        self,
    ) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
        """Return the historical four-item tuple for temporary compatibility."""
        return (
            dict(self.summary),
            self.ptx_history,
            self.margin_history,
            dict(self.tables),
        )


class Simulator:
    """Stateless, validated entry point for one LoRaPhyMAC simulation."""

    version = SIMULATOR_VERSION
    api_version = SIMULATOR_API_VERSION

    @staticmethod
    def default_config() -> dict[str, Any]:
        """Return an independent copy of the simulator defaults."""
        return deepcopy(_DEFAULT_CONFIG)

    @staticmethod
    def resolve_configuration(
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge overrides with defaults and run centralized validation."""
        if config is not None and not isinstance(config, Mapping):
            raise ConfigurationError("configuration must be a mapping")
        overrides = None if config is None else dict(config)
        resolved = _build_config(overrides)
        validate_configuration(resolved, known_keys=_DEFAULT_CONFIG)
        return resolved

    def run(self, config: Mapping[str, Any] | None = None) -> SimulationResult:
        """Validate and execute one simulation, returning a structured result."""
        resolved = self.resolve_configuration(config)
        summary, ptx_history, margin_history, tables = _run_single_simulation(resolved)
        normalized_tables = dict(tables)
        normalized_tables.setdefault("ptx_history_df", ptx_history)
        normalized_tables.setdefault("margin_history_df", margin_history)
        return SimulationResult(
            resolved_config=resolved,
            summary=summary,
            tables=normalized_tables,
        )


def default_config() -> dict[str, Any]:
    """Return an independent copy of the simulator defaults."""
    return Simulator.default_config()


def resolve_configuration(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve and validate a partial public-API configuration."""
    return Simulator.resolve_configuration(config)


def simulate(config: Mapping[str, Any] | None = None) -> SimulationResult:
    """Run one validated simulation through the stable public API."""
    return Simulator().run(config)


__all__ = [
    "SIMULATOR_API_VERSION",
    "SIMULATOR_VERSION",
    "SimulationResult",
    "Simulator",
    "default_config",
    "resolve_configuration",
    "simulate",
]
