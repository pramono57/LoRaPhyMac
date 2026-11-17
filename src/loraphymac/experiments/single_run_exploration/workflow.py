"""Flexible exploratory single runs built on the stable simulator API.

This workflow deliberately has no publication contract. Every configuration
still passes the central simulator validator, but researchers may change any
scientific parameter while developing a hypothesis or checking model behavior.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from loraphymac import SimulationResult, simulate
from loraphymac.paths import repository_root


WORKFLOW_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = repository_root(WORKFLOW_DIR)
DEFAULT_CONFIG = WORKFLOW_DIR / "configs" / "default_v1.json"
EXPLORATION_ARTIFACT_ROOT = (
    PROJECT_ROOT / "artifacts" / "experiments" / "single_run_exploration"
)


def exploration_output_dir(run_id: str = "exploration_001") -> Path:
    """Return the canonical artifact directory for one exploratory run."""
    normalized = str(run_id).strip()
    if not normalized:
        raise ValueError("run_id must be a non-empty string")
    return EXPLORATION_ARTIFACT_ROOT / normalized


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Exploration configuration must be a JSON object: {config_path}")
    return value


def default_exploration_config(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load the maintained, freely editable exploratory starting point."""
    config = _load_config(config_path)
    run_id = str(config.get("run_id", "exploration_001"))
    destination = exploration_output_dir(run_id) if output_dir is None else Path(output_dir)
    config["output_dir"] = str(destination)
    return config


def run_single_exploration(
    config: Mapping[str, Any] | None = None,
) -> SimulationResult:
    """Run one exploratory scenario after general centralized validation.

    The config mapping contains overrides. Scientific parameters may change,
    but selecting a registered profile activates that profile's contract.
    """
    resolved = default_exploration_config()
    if config is not None:
        overrides = dict(config)
        resolved.update(overrides)
        if "run_id" in overrides and "output_dir" not in overrides:
            resolved["output_dir"] = str(exploration_output_dir(str(resolved["run_id"])))
    return simulate(resolved)


__all__ = [
    "DEFAULT_CONFIG",
    "EXPLORATION_ARTIFACT_ROOT",
    "PROJECT_ROOT",
    "default_exploration_config",
    "exploration_output_dir",
    "run_single_exploration",
]
