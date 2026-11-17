"""Future runner for paired APC-model comparisons on the shared pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from loraphymac.experiments.common.multiseed_pipeline import (
    ExperimentSpec,
    build_experiment_grid,
    run_experiment,
)
from loraphymac.paths import repository_root
from loraphymac.runtime.islam_dbpc import (
    ISLAM_DBPC_POLICY_ID,
    IslamDBPCController,
    load_frozen_calibration,
)
from loraphymac.runtime.islam_dbpc_802 import ISLAM_DBPC_802_POLICY_ID
from loraphymac.runtime.tx_power_policy import LEGACY_POLICY_ID
from loraphymac.simulator import resolve_configuration


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = repository_root(EXPERIMENT_DIR)
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "comparison.json"


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    display_name: str
    policy_id: str
    simulator_config: Mapping[str, Any]
    calibration: Mapping[str, Any] | None


@dataclass(frozen=True)
class ComparisonDefinition:
    experiment_id: str
    output_dir: Path
    shared_base_config: Mapping[str, Any]
    factors: Mapping[str, Sequence[Any]]
    seeds: Sequence[int]
    models: Mapping[str, Path]
    planned_execution_model_ids: Sequence[str]
    confidence_level: float


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_comparison_definition(path: str | Path = DEFAULT_CONFIG) -> ComparisonDefinition:
    config_path = Path(path).expanduser().resolve()
    raw = _load_json(config_path)
    required = {
        "experiment_id",
        "output_dir",
        "shared_base_config",
        "factors",
        "seeds",
        "models",
        "planned_execution_model_ids",
    }
    unknown = set(raw).difference(required | {"confidence_level"})
    missing = required.difference(raw)
    if missing or unknown:
        raise ValueError(f"Comparison config missing={sorted(missing)} unknown={sorted(unknown)}")
    output_dir = Path(str(raw["output_dir"]))
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    models = {
        str(model_id): (config_path.parent / str(model_path)).resolve()
        for model_id, model_path in raw["models"].items()
    }
    planned = tuple(str(value) for value in raw["planned_execution_model_ids"])
    unknown_planned = set(planned).difference(models)
    if unknown_planned:
        raise ValueError(f"Unknown planned model IDs: {sorted(unknown_planned)}")
    return ComparisonDefinition(
        experiment_id=str(raw["experiment_id"]),
        output_dir=output_dir,
        shared_base_config=dict(raw["shared_base_config"]),
        factors={key: list(values) for key, values in raw["factors"].items()},
        seeds=list(raw["seeds"]),
        models=models,
        planned_execution_model_ids=planned,
        confidence_level=float(raw.get("confidence_level", 0.95)),
    )


def load_model_definition(path: str | Path) -> ModelDefinition:
    model_path = Path(path).expanduser().resolve()
    raw = _load_json(model_path)
    required = {"model_id", "display_name", "policy_id", "simulator_config", "calibration"}
    missing = required.difference(raw)
    unknown = set(raw).difference(required)
    if missing or unknown:
        raise ValueError(f"Model config missing={sorted(missing)} unknown={sorted(unknown)}")
    policy_id = str(raw["policy_id"])
    if policy_id not in {
        LEGACY_POLICY_ID,
        ISLAM_DBPC_POLICY_ID,
        ISLAM_DBPC_802_POLICY_ID,
    }:
        raise ValueError(f"Unknown transmit-power policy {policy_id!r}")
    calibration = raw["calibration"]
    if calibration is not None and not isinstance(calibration, Mapping):
        raise ValueError("calibration must be an object or null")
    return ModelDefinition(
        model_id=str(raw["model_id"]),
        display_name=str(raw["display_name"]),
        policy_id=policy_id,
        simulator_config=dict(raw["simulator_config"]),
        calibration=None if calibration is None else dict(calibration),
    )


def build_model_experiment_spec(
    comparison_path: str | Path = DEFAULT_CONFIG,
    *,
    model_id: str,
) -> ExperimentSpec:
    comparison = load_comparison_definition(comparison_path)
    try:
        model_path = comparison.models[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown model ID {model_id!r}") from exc
    model = load_model_definition(model_path)
    if model.model_id != model_id:
        raise ValueError("Model ID does not match its comparison registry key")

    base_config = dict(comparison.shared_base_config)
    base_config.update(model.simulator_config)
    base_config.update(
        {
            "tx_power_model_id": model.model_id,
            "tx_power_policy_id": model.policy_id,
        }
    )
    if model.policy_id == ISLAM_DBPC_POLICY_ID:
        if model.calibration is None:
            raise ValueError("Islam-DBPC model requires calibration metadata")
        manifest = Path(str(model.calibration["manifest"]))
        if not manifest.is_absolute():
            manifest = (PROJECT_ROOT / manifest).resolve()
        base_config.update(
            {
                "tx_power_calibration_id": model.calibration["calibration_id"],
                "tx_power_calibration_manifest": str(manifest),
                "tx_power_calibration_manifest_sha256": model.calibration["manifest_sha256"],
                "tx_power_calibration_margin_db": model.calibration["margin_db"],
            }
        )
    elif model.calibration is not None:
        raise ValueError("Non-calibrated policies must not declare a DBPC calibration")

    resolved = resolve_configuration(base_config)
    if resolved["energy_accounting_mode"] != "corrected":
        raise ValueError("apc_compare_model requires energy_accounting_mode='corrected'")
    if model.policy_id == ISLAM_DBPC_POLICY_ID:
        calibration = load_frozen_calibration(
            resolved["tx_power_calibration_manifest"],
            expected_manifest_sha256=resolved["tx_power_calibration_manifest_sha256"],
            expected_calibration_id=resolved["tx_power_calibration_id"],
            margin_db=int(resolved["tx_power_calibration_margin_db"]),
        )
        IslamDBPCController(calibration, model.model_id).verify_runtime_configuration(resolved)

    return ExperimentSpec(
        experiment_id=f"{comparison.experiment_id}__{model.model_id}",
        output_dir=comparison.output_dir,
        base_config=base_config,
        factors=dict(comparison.factors),
        seeds=list(comparison.seeds),
        retain_debug_tables=False,
        confidence_level=comparison.confidence_level,
    )


def planned_grid(
    comparison_path: str | Path = DEFAULT_CONFIG,
    *,
    model_id: str = "islam_dbpc_itm0",
) -> list[dict[str, Any]]:
    """Build and validate a future grid without executing any simulation."""
    return build_experiment_grid(build_model_experiment_spec(comparison_path, model_id=model_id))


def run_experiments(
    comparison_path: str | Path = DEFAULT_CONFIG,
    *,
    model_id: str,
    **pipeline_options: Any,
) -> dict[str, Any]:
    """Execute a separately authorized model grid through the shared pipeline."""
    spec = build_model_experiment_spec(comparison_path, model_id=model_id)
    return run_experiment(spec, project_root=PROJECT_ROOT, **pipeline_options)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args(argv)
    result = run_experiments(
        args.config,
        model_id=args.model_id,
        dry_run=args.dry_run,
        max_runs=args.max_runs,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
