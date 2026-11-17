"""One-run, contracted verification of closed-loop APC behavior."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from loraphymac import (
    SIMULATOR_API_VERSION,
    SIMULATOR_VERSION,
    SimulationResult,
    simulate,
)
from loraphymac.experiments.apc_closed_loop_verification.contract import (
    APC_CLOSED_LOOP_CONTRACT_VERSION,
    APC_CLOSED_LOOP_PROFILE,
    validate_apc_closed_loop_outputs,
)
from loraphymac.experiments.common.multiseed_pipeline import (
    atomic_write_json,
    content_hash,
)
from loraphymac.paths import repository_root


WORKFLOW_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = repository_root(WORKFLOW_DIR)
DEFAULT_CONFIG = WORKFLOW_DIR / "configs" / "verification_v1.json"
APC_CLOSED_LOOP_ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
    / "apc_closed_loop_verification"
)


def apc_closed_loop_output_dir(
    run_id: str = "apc_closed_loop_dh90_s42",
) -> Path:
    """Return the canonical directory for one contracted verification."""
    normalized = str(run_id).strip()
    if not normalized:
        raise ValueError("run_id must be a non-empty string")
    return APC_CLOSED_LOOP_ARTIFACT_ROOT / normalized


def verification_report_path(result: SimulationResult) -> Path:
    return Path(result.resolved_config["output_dir"]) / "verification_report.json"


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(
            f"APC closed-loop configuration must be a JSON object: {config_path}"
        )
    return value


def default_apc_closed_loop_config(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load the frozen one-run APC verification configuration."""
    config = _load_config(config_path)
    run_id = str(config.get("run_id", "apc_closed_loop_dh90_s42"))
    destination = (
        apc_closed_loop_output_dir(run_id)
        if output_dir is None
        else Path(output_dir)
    )
    config["output_dir"] = str(destination)
    return config


def _verification_report(result: SimulationResult) -> dict[str, Any]:
    history = result.ptx_history
    entity_types = history["entity_type"].dropna().astype(str).str.lower()
    return {
        "status": "PASS",
        "experiment_profile": APC_CLOSED_LOOP_PROFILE,
        "experiment_contract_version": APC_CLOSED_LOOP_CONTRACT_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "simulator_api_version": SIMULATOR_API_VERSION,
        "run_id": result.run_id,
        "seed": int(result.resolved_config["seed"]),
        "terrain_delta_h": float(result.resolved_config["terrain_delta_h"]),
        "resolved_config_hash": content_hash(dict(result.resolved_config)),
        "checks": {
            "central_configuration_validation": True,
            "closed_loop_contract": True,
            "stage1_status": result.summary["stage1_status"],
            "packet_fate_conservation_delta": int(
                result.summary["packet_fate_conservation_delta"]
            ),
            "collision_outcome_conservation_delta": int(
                result.summary["collision_outcome_conservation_delta"]
            ),
            "apc_total_tx": int(result.summary["apc_total_tx"]),
            "apc_total_rx": int(result.summary["apc_total_rx"]),
            "apc_total_drop": int(result.summary["apc_total_drop"]),
            "ptx_update_count": int(result.summary["ptx_update_count"]),
            "node_ptx_update_count": int(entity_types.eq("node").sum()),
            "relay_ptx_update_count": int(entity_types.eq("relay").sum()),
        },
    }


def run_apc_closed_loop_verification(
    config: Mapping[str, Any] | None = None,
) -> SimulationResult:
    """Run, validate, and record the frozen APC closed-loop verification."""
    resolved = default_apc_closed_loop_config()
    if config is not None:
        overrides = dict(config)
        resolved.update(overrides)
        if "run_id" in overrides and "output_dir" not in overrides:
            resolved["output_dir"] = str(
                apc_closed_loop_output_dir(str(resolved["run_id"]))
            )

    result = simulate(resolved)
    validate_apc_closed_loop_outputs(
        result.resolved_config,
        result.summary,
        result.tables,
    )
    report_path = verification_report_path(result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_path, _verification_report(result))
    return result


__all__ = [
    "APC_CLOSED_LOOP_ARTIFACT_ROOT",
    "DEFAULT_CONFIG",
    "PROJECT_ROOT",
    "apc_closed_loop_output_dir",
    "default_apc_closed_loop_config",
    "run_apc_closed_loop_verification",
    "verification_report_path",
]
