"""Unified matched terrain sweep and protected comparison pilots."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from loraphymac.experiments.apc_compare_model.run import (
    build_model_experiment_spec,
    load_comparison_definition,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.contract import (
    APC_COMPARE_MODEL_TERRAIN_SWEEP_MODELS,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_TERRAINS,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_MODELS,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_MODELS,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_SEEDS,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_TERRAINS,
    DBPC_CALIBRATION_MANIFEST_SHA256,
    MAIN_W4_EXPERIMENT_TRACK,
    MAIN_W4_MODELS,
    MAIN_W4_PROFILE,
    MAIN_W4_REPORTING_NAMESPACE,
    PAIRED_ROUTING_PILOT_EXPERIMENT_TRACK,
    PAIRED_ROUTING_PILOT_MODELS,
    PAIRED_ROUTING_PILOT_PROFILE,
    PAIRED_ROUTING_PILOT_REPORTING_NAMESPACE,
    PAIRED_ROUTING_PILOT_SEEDS,
    PAIRED_ROUTING_PILOT_STRATEGIES,
    PAIRED_ROUTING_PILOT_TERRAINS,
    REVIEWER_R1_5_COMPARISON_TERRAINS,
    REVIEWER_R1_5_EXPERIMENT_TRACK,
    REVIEWER_R1_5_MODELS,
    REVIEWER_R1_5_PROFILE,
    REVIEWER_R1_5_REPORTING_NAMESPACE,
    REVIEWER_R1_5_TERRAINS,
    routing_reporting_contract,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.routing_snapshots import (
    RoutingSnapshotBinding,
    load_routing_snapshot_bindings,
)
from loraphymac.experiments.common.multiseed_pipeline import (
    ExperimentSpec,
    _source_fingerprint,
    atomic_write_json,
    build_experiment_grid,
    content_hash,
    file_sha256,
    run_experiment,
    utc_now,
)
from loraphymac.experiments.common.outputs import ExperimentArtifactPaths
from loraphymac.paths import repository_root
from loraphymac.runtime.tx_power_policy import LEGACY_POLICY_ID


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = repository_root(EXPERIMENT_DIR)
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "production_v2_wang.json"
FIVE_MODEL_PILOT_CONFIG = (
    EXPERIMENT_DIR
    / "configs"
    / "pilot_w3c_five_model_n100_dh90_s30_r01.json"
)
MAIN_W4_CONFIG = EXPERIMENT_DIR / "configs" / "main_w4_fixed_reference_v1.json"
REVIEWER_R1_5_CONFIG = (
    EXPERIMENT_DIR
    / "configs"
    / "reviewer_r1_5_routing_sensitivity_v1.json"
)
PAIRED_ROUTING_PILOT_CONFIG = (
    EXPERIMENT_DIR / "configs" / "pilot_w3c_paired_routing_n100_s30_r01.json"
)
LEGACY_CONFIG = EXPERIMENT_DIR / "configs" / "production_v1.json"
E0_CONFIG = (
    EXPERIMENT_DIR.parent / "apc_terrain_sweep" / "configs" / "production_v1.json"
)

MODEL_LABELS = {
    "ff": "FF",
    "aa": "AA (R-APC)",
    "islam_dbpc_802": "DBPC-802",
    "islam_dbpc_itm6": "Islam-DBPC/6",
    "wang_atp_sf7_a2only": "Wang A2-only",
}
PILOT_MODEL_LABELS = {
    "ff": "FF",
    "aa": "R-APC",
    "islam_dbpc_802": "DBPC-802",
    "islam_dbpc_itm6": "DBPC-ITM",
    "wang_atp_sf7_a2only": "Wang-CSPC",
}
MODEL_SPECIFIC_CONFIG_KEYS = frozenset(
    {
        "node_ptx_mode",
        "relay_ptx_mode",
        "tx_power_model_id",
        "tx_power_policy_id",
        "tx_power_calibration_id",
        "tx_power_calibration_manifest",
        "tx_power_calibration_manifest_sha256",
        "tx_power_calibration_margin_db",
    }
)


@dataclass(frozen=True)
class SweepDefinition:
    config_path: Path
    experiment_id: str
    output_dir: Path
    comparison_config: Path
    experiment_profile: str
    models: tuple[str, ...]
    source_model_ids: Mapping[str, str]
    factors: Mapping[str, Sequence[Any]]
    seeds: tuple[int, ...]
    energy_accounting_mode: str
    dbpc_calibration_margin_db: int
    dbpc_calibration_manifest_sha256: str
    shared_config_overrides: Mapping[str, Any]
    confidence_level: float
    experiment_track: str | None
    reporting_namespace: str | None
    routing_strategy: str | None
    routing_strategies: tuple[str, ...]
    routing_reference_delta_h: int | None
    routing_snapshot_bindings_path: Path | None
    routing_snapshot_bindings_sha256: str | None
    snapshot_bindings: Mapping[int, RoutingSnapshotBinding]
    comparison_terrains: tuple[int, ...]
    fixed_reference_source_experiment_id: str | None

    @property
    def experiment_dir(self) -> Path:
        return self.output_dir / self.experiment_id

    def serializable(self) -> dict[str, Any]:
        value = asdict(self)
        value["config_path"] = str(self.config_path)
        value["output_dir"] = str(self.output_dir)
        value["comparison_config"] = str(self.comparison_config)
        value["models"] = list(self.models)
        value["source_model_ids"] = dict(self.source_model_ids)
        value["factors"] = {key: list(values) for key, values in self.factors.items()}
        value["shared_config_overrides"] = dict(self.shared_config_overrides)
        value["seeds"] = list(self.seeds)
        value["comparison_terrains"] = list(self.comparison_terrains)
        value["routing_strategies"] = list(self.routing_strategies)
        value["routing_snapshot_bindings_path"] = (
            None
            if self.routing_snapshot_bindings_path is None
            else str(self.routing_snapshot_bindings_path)
        )
        value["snapshot_bindings"] = {
            str(seed): {
                "manifest_path": str(binding.manifest_path),
                "manifest_sha256": binding.manifest_sha256,
                "snapshot_id": binding.snapshot_id,
                "topology_sha256": binding.topology_sha256,
                "routing_sha256": binding.routing_sha256,
                "reference_delta_h": binding.reference_delta_h,
                "creation_provenance": dict(binding.creation_provenance),
            }
            for seed, binding in sorted(self.snapshot_bindings.items())
        }
        for key in (
            "experiment_track",
            "reporting_namespace",
            "routing_strategy",
            "routing_reference_delta_h",
            "routing_snapshot_bindings_path",
            "routing_snapshot_bindings_sha256",
            "fixed_reference_source_experiment_id",
        ):
            if value[key] is None:
                value.pop(key)
        if not value["snapshot_bindings"]:
            value.pop("snapshot_bindings")
        if not self.comparison_terrains:
            value.pop("comparison_terrains")
        if not self.routing_strategies:
            value.pop("routing_strategies")
        return value


def model_label(model_id: str, experiment_profile: str | None = None) -> str:
    labels = (
        PILOT_MODEL_LABELS
        if experiment_profile
        in {
            APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE,
            MAIN_W4_PROFILE,
            REVIEWER_R1_5_PROFILE,
            PAIRED_ROUTING_PILOT_PROFILE,
        }
        else MODEL_LABELS
    )
    return labels[model_id]


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _resolve_from(config_path: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_sweep_definition(path: str | Path = DEFAULT_CONFIG) -> SweepDefinition:
    config_path = Path(path).expanduser().resolve()
    raw = _read_json(config_path)
    required = {
        "experiment_id",
        "output_dir",
        "comparison_config",
        "models",
        "source_model_ids",
        "factors",
        "seeds",
        "energy_accounting_mode",
        "dbpc_calibration_margin_db",
        "dbpc_calibration_manifest_sha256",
        "confidence_level",
    }
    missing = required.difference(raw)
    optional = {
        "experiment_profile",
        "shared_config_overrides",
        "experiment_track",
        "reporting_namespace",
        "routing_strategy",
        "routing_strategies",
        "routing_reference_delta_h",
        "routing_snapshot_bindings",
        "routing_snapshot_bindings_sha256",
        "comparison_terrains",
        "fixed_reference_source_experiment_id",
    }
    unknown = set(raw).difference(required | optional)
    if missing or unknown:
        raise ValueError(f"Sweep config missing={sorted(missing)} unknown={sorted(unknown)}")

    profile = str(
        raw.get("experiment_profile", APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE)
    )
    binding_path = (
        _resolve_from(config_path, raw["routing_snapshot_bindings"])
        if raw.get("routing_snapshot_bindings") is not None
        else None
    )
    binding_sha256 = (
        str(raw["routing_snapshot_bindings_sha256"])
        if raw.get("routing_snapshot_bindings_sha256") is not None
        else None
    )
    seeds = tuple(int(value) for value in raw["seeds"])
    snapshot_bindings: Mapping[int, RoutingSnapshotBinding] = {}
    if binding_path is not None or binding_sha256 is not None:
        if binding_path is None or binding_sha256 is None:
            raise ValueError(
                "routing snapshot binding path and SHA-256 must be declared together"
            )
        loaded_bindings = load_routing_snapshot_bindings(
            binding_path,
            expected_sha256=binding_sha256,
            expected_seeds=(
                APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS
                if profile == PAIRED_ROUTING_PILOT_PROFILE
                else seeds
            ),
        )
        snapshot_bindings = {seed: loaded_bindings[seed] for seed in seeds}

    definition = SweepDefinition(
        config_path=config_path,
        experiment_id=str(raw["experiment_id"]),
        output_dir=_resolve_from(config_path, raw["output_dir"]),
        comparison_config=_resolve_from(config_path, raw["comparison_config"]),
        experiment_profile=profile,
        models=tuple(str(value) for value in raw["models"]),
        source_model_ids={
            str(key): str(value) for key, value in raw["source_model_ids"].items()
        },
        factors={str(key): list(values) for key, values in raw["factors"].items()},
        seeds=seeds,
        energy_accounting_mode=str(raw["energy_accounting_mode"]),
        dbpc_calibration_margin_db=int(raw["dbpc_calibration_margin_db"]),
        dbpc_calibration_manifest_sha256=str(
            raw["dbpc_calibration_manifest_sha256"]
        ),
        shared_config_overrides=dict(raw.get("shared_config_overrides", {})),
        confidence_level=float(raw["confidence_level"]),
        experiment_track=(
            None
            if raw.get("experiment_track") is None
            else str(raw["experiment_track"])
        ),
        reporting_namespace=(
            None
            if raw.get("reporting_namespace") is None
            else str(raw["reporting_namespace"])
        ),
        routing_strategy=(
            None
            if raw.get("routing_strategy") is None
            else str(raw["routing_strategy"])
        ),
        routing_strategies=tuple(
            str(value) for value in raw.get("routing_strategies", ())
        ),
        routing_reference_delta_h=(
            None
            if raw.get("routing_reference_delta_h") is None
            else int(raw["routing_reference_delta_h"])
        ),
        routing_snapshot_bindings_path=binding_path,
        routing_snapshot_bindings_sha256=binding_sha256,
        snapshot_bindings=snapshot_bindings,
        comparison_terrains=tuple(
            int(value) for value in raw.get("comparison_terrains", ())
        ),
        fixed_reference_source_experiment_id=(
            None
            if raw.get("fixed_reference_source_experiment_id") is None
            else str(raw["fixed_reference_source_experiment_id"])
        ),
    )
    errors = []
    profile_models = {
        APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE: APC_COMPARE_MODEL_TERRAIN_SWEEP_MODELS,
        APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE: (
            APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_MODELS
        ),
        APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE: (
            APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_MODELS
        ),
        MAIN_W4_PROFILE: MAIN_W4_MODELS,
        REVIEWER_R1_5_PROFILE: REVIEWER_R1_5_MODELS,
        PAIRED_ROUTING_PILOT_PROFILE: PAIRED_ROUTING_PILOT_MODELS,
    }
    expected_models = profile_models.get(definition.experiment_profile)
    if expected_models is None:
        errors.append("experiment_profile must be a registered comparison sweep profile")
    elif definition.models != expected_models:
        errors.append(f"models must be {list(expected_models)}")
    expected_sources = {
        "aa": "rapc",
        "islam_dbpc_itm6": "islam_dbpc_itm6",
    }
    expected_overrides: Mapping[str, Any] = {}
    if definition.experiment_profile in {
        APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
        APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE,
        MAIN_W4_PROFILE,
        REVIEWER_R1_5_PROFILE,
        PAIRED_ROUTING_PILOT_PROFILE,
    }:
        expected_sources["wang_atp_sf7_a2only"] = "wang_atp_sf7_a2only"
        if definition.experiment_profile in {
            APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE,
            MAIN_W4_PROFILE,
            REVIEWER_R1_5_PROFILE,
            PAIRED_ROUTING_PILOT_PROFILE,
        }:
            expected_sources["islam_dbpc_802"] = "islam_dbpc_802"
        expected_overrides = {
            "node_num": (
                100
                if definition.experiment_profile
                in {
                    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE,
                    MAIN_W4_PROFILE,
                    REVIEWER_R1_5_PROFILE,
                    PAIRED_ROUTING_PILOT_PROFILE,
                    PAIRED_ROUTING_PILOT_PROFILE,
                }
                else 60
            ),
            "node_rx_collision_enabled": True,
            "frame_default_ack_req": False,
            "application_scheduling_mode": "independent_fifo_v1",
            "application_rng_policy": "sha256_seed_node_v1",
            "application_queue_policy": "unbounded_fifo_v1",
            "application_horizon_policy": "right_censor_waiting_v1",
        }
    if definition.source_model_ids != expected_sources:
        errors.append("source_model_ids do not match the audited model registry")
    if definition.shared_config_overrides != expected_overrides:
        errors.append(
            f"shared_config_overrides must be {dict(expected_overrides)!r}"
        )
    if (
        definition.experiment_profile
        == APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE
    ):
        expected_terrains = APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_TERRAINS
        expected_seeds = APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_SEEDS
    elif definition.experiment_profile == REVIEWER_R1_5_PROFILE:
        expected_terrains = REVIEWER_R1_5_TERRAINS
        expected_seeds = APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS
    elif definition.experiment_profile == PAIRED_ROUTING_PILOT_PROFILE:
        expected_terrains = PAIRED_ROUTING_PILOT_TERRAINS
        expected_seeds = PAIRED_ROUTING_PILOT_SEEDS
    else:
        expected_terrains = APC_COMPARE_MODEL_TERRAIN_SWEEP_TERRAINS
        expected_seeds = APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS
    if tuple(definition.factors) != ("terrain_delta_h",) or tuple(
        definition.factors.get("terrain_delta_h", ())
    ) != expected_terrains:
        errors.append("terrain_delta_h does not match the registered profile")
    if definition.seeds != expected_seeds:
        errors.append("seeds do not match the registered profile")
    if definition.energy_accounting_mode != "corrected":
        errors.append("energy_accounting_mode must be corrected")
    if definition.dbpc_calibration_margin_db != 6:
        errors.append("DBPC calibration margin must be 6 dB")
    if (
        definition.dbpc_calibration_manifest_sha256
        != DBPC_CALIBRATION_MANIFEST_SHA256
    ):
        errors.append("DBPC calibration manifest hash does not match the frozen bundle")
    if definition.confidence_level != 0.95:
        errors.append("confidence_level must be 0.95")
    routing_profiles = {
        MAIN_W4_PROFILE: {
            "experiment_track": MAIN_W4_EXPERIMENT_TRACK,
            "reporting_namespace": MAIN_W4_REPORTING_NAMESPACE,
            "routing_strategy": "fixed_reference",
            "comparison_terrains": APC_COMPARE_MODEL_TERRAIN_SWEEP_TERRAINS,
            "fixed_reference_source_experiment_id": None,
        },
        REVIEWER_R1_5_PROFILE: {
            "experiment_track": REVIEWER_R1_5_EXPERIMENT_TRACK,
            "reporting_namespace": REVIEWER_R1_5_REPORTING_NAMESPACE,
            "routing_strategy": "terrain_specific",
            "comparison_terrains": REVIEWER_R1_5_COMPARISON_TERRAINS,
            "fixed_reference_source_experiment_id": MAIN_W4_REPORTING_NAMESPACE,
        },
        PAIRED_ROUTING_PILOT_PROFILE: {
            "experiment_track": PAIRED_ROUTING_PILOT_EXPERIMENT_TRACK,
            "reporting_namespace": PAIRED_ROUTING_PILOT_REPORTING_NAMESPACE,
            "routing_strategy": None,
            "routing_strategies": PAIRED_ROUTING_PILOT_STRATEGIES,
            "comparison_terrains": PAIRED_ROUTING_PILOT_TERRAINS,
            "fixed_reference_source_experiment_id": None,
        },
    }
    routing_expected = routing_profiles.get(definition.experiment_profile)
    if routing_expected is not None:
        for key, expected in routing_expected.items():
            if getattr(definition, key) != expected:
                errors.append(f"{key} must be {expected!r}")
        if definition.experiment_id != definition.reporting_namespace:
            errors.append("experiment_id must equal the reporting namespace")
        if definition.routing_reference_delta_h != 90:
            errors.append("routing_reference_delta_h must be 90")
        if set(definition.snapshot_bindings) != set(definition.seeds):
            errors.append("every configured seed must have one snapshot binding")
    if errors:
        raise ValueError("Invalid APC comparison terrain sweep: " + "; ".join(errors))
    return definition


def _child_spec(
    definition: SweepDefinition,
    *,
    model_id: str,
) -> ExperimentSpec:
    comparison = load_comparison_definition(definition.comparison_config)
    if model_id == "ff":
        base = dict(comparison.shared_base_config)
        base.update(
            {
                "node_ptx_mode": "fixed",
                "relay_ptx_mode": "fixed",
                "tx_power_policy_id": LEGACY_POLICY_ID,
                "tx_power_model_id": "ff",
            }
        )
    else:
        source_id = definition.source_model_ids[model_id]
        source = build_model_experiment_spec(
            definition.comparison_config,
            model_id=source_id,
        )
        base = dict(source.base_config)
        base["tx_power_model_id"] = model_id

    base.update(definition.shared_config_overrides)
    base["experiment_profile"] = definition.experiment_profile
    base["energy_accounting_mode"] = definition.energy_accounting_mode
    seed_config_overrides: dict[int, dict[str, Any]] = {}
    if definition.routing_strategy is not None:
        base["routing_strategy"] = definition.routing_strategy
        if definition.routing_strategy == "fixed_reference":
            base["routing_reference_delta_h"] = (
                definition.routing_reference_delta_h
            )
            base["routing_snapshot_manifest"] = None
            base["routing_snapshot_manifest_sha256"] = None
            seed_config_overrides = {
                seed: binding.config_overrides()
                for seed, binding in definition.snapshot_bindings.items()
            }
        else:
            base["routing_reference_delta_h"] = None
            base["routing_snapshot_manifest"] = None
            base["routing_snapshot_manifest_sha256"] = None
    return ExperimentSpec(
        experiment_id=model_id,
        output_dir=definition.experiment_dir / "models",
        base_config=base,
        factors={key: list(values) for key, values in definition.factors.items()},
        seeds=list(definition.seeds),
        seed_config_overrides=seed_config_overrides,
        run_annotations={
            "experiment_track": definition.experiment_track,
            "reporting_namespace": definition.reporting_namespace,
        }
        if definition.experiment_track is not None
        else {},
        retain_debug_tables=False,
        confidence_level=definition.confidence_level,
    )


def build_model_specs(
    path: str | Path = DEFAULT_CONFIG,
) -> dict[str, ExperimentSpec]:
    definition = load_sweep_definition(path)
    base_specs = {
        model_id: _child_spec(definition, model_id=model_id)
        for model_id in definition.models
    }
    if definition.experiment_profile == PAIRED_ROUTING_PILOT_PROFILE:
        specs: dict[str, ExperimentSpec] = {}
        for strategy in definition.routing_strategies:
            for model_id, source in base_specs.items():
                base = dict(source.base_config)
                base["routing_strategy"] = strategy
                seed_overrides: dict[int, dict[str, Any]] = {}
                if strategy == "fixed_reference":
                    base["routing_reference_delta_h"] = 90
                    base["routing_snapshot_manifest"] = None
                    base["routing_snapshot_manifest_sha256"] = None
                    seed_overrides = {
                        seed: binding.config_overrides()
                        for seed, binding in definition.snapshot_bindings.items()
                    }
                else:
                    base["routing_reference_delta_h"] = None
                    base["routing_snapshot_manifest"] = None
                    base["routing_snapshot_manifest_sha256"] = None
                key = f"{strategy}__{model_id}"
                specs[key] = ExperimentSpec(
                    experiment_id=model_id,
                    output_dir=(
                        definition.experiment_dir / "strategies" / strategy / "models"
                    ),
                    base_config=base,
                    factors={
                        factor: list(values)
                        for factor, values in definition.factors.items()
                    },
                    seeds=list(definition.seeds),
                    seed_config_overrides=seed_overrides,
                    run_annotations={
                        "experiment_track": definition.experiment_track,
                        "reporting_namespace": definition.reporting_namespace,
                        "routing_strategy": strategy,
                    },
                    retain_debug_tables=False,
                    confidence_level=definition.confidence_level,
                )
    else:
        specs = base_specs
    signature_exclusions = set(MODEL_SPECIFIC_CONFIG_KEYS)
    if definition.experiment_profile == PAIRED_ROUTING_PILOT_PROFILE:
        signature_exclusions.update(
            {
                "routing_strategy",
                "routing_reference_delta_h",
                "routing_snapshot_manifest",
                "routing_snapshot_manifest_sha256",
            }
        )
    signatures = {
        child_id: {
            key: value
            for key, value in spec.base_config.items()
            if key not in signature_exclusions
        }
        for child_id, spec in specs.items()
    }
    if len({content_hash(value, length=64) for value in signatures.values()}) != 1:
        raise ValueError("Shared PHY/MAC, topology, traffic, and duration inputs differ by model")
    comparison = load_comparison_definition(definition.comparison_config)
    e0_base = dict(_read_json(E0_CONFIG)["base_config"])
    shared_base = dict(comparison.shared_base_config)
    for value in (e0_base, shared_base):
        value.pop("experiment_profile", None)
        value.pop("energy_accounting_mode", None)
    if shared_base != e0_base:
        raise ValueError("E1 shared inputs differ from the validated E0 configuration")
    dbpc = next(
        spec.base_config
        for spec in specs.values()
        if spec.base_config.get("tx_power_model_id") == "islam_dbpc_itm6"
    )
    if dbpc.get("tx_power_calibration_manifest_sha256") != (
        definition.dbpc_calibration_manifest_sha256
    ) or int(dbpc.get("tx_power_calibration_margin_db", -1)) != 6:
        raise ValueError("DBPC child does not use the approved frozen margin-6 calibration")
    return specs


def planned_grid(path: str | Path = DEFAULT_CONFIG) -> list[dict[str, Any]]:
    """Return the validated model/terrain/seed plan without simulation."""
    definition = load_sweep_definition(path)
    rows: list[dict[str, Any]] = []
    for child_id, spec in build_model_specs(path).items():
        model_id = str(spec.base_config["tx_power_model_id"])
        for run in build_experiment_grid(spec):
            rows.append(
                {
                    "model_id": model_id,
                    "model_label": model_label(model_id, definition.experiment_profile),
                    "terrain_delta_h": run["factors"]["terrain_delta_h"],
                    "seed": run["seed"],
                    "child_experiment_id": spec.experiment_id,
                    "child_id": child_id,
                    "run_id": run["run_id"],
                    "experiment_track": definition.experiment_track,
                    "reporting_namespace": definition.reporting_namespace,
                    "routing_strategy": run["scientific_config"].get(
                        "routing_strategy", "terrain_specific"
                    ),
                    "routing_reference_delta_h": run["scientific_config"].get(
                        "routing_reference_delta_h"
                    ),
                    "routing_snapshot_manifest_sha256": run[
                        "scientific_config"
                    ].get("routing_snapshot_manifest_sha256"),
                }
            )
    cells = {
        (
            row["model_id"],
            row["terrain_delta_h"],
            row["seed"],
            row["routing_strategy"],
        )
        for row in rows
    }
    expected_count = (
        len(definition.models)
        * len(definition.factors["terrain_delta_h"])
        * len(definition.seeds)
        * max(1, len(definition.routing_strategies))
    )
    if len(rows) != expected_count or len(cells) != len(rows):
        raise ValueError(
            f"Unified comparison grid must contain {expected_count} unique cells"
        )
    return rows


def _source_records(definition: SweepDefinition, specs: Mapping[str, ExperimentSpec]) -> dict:
    comparison = load_comparison_definition(definition.comparison_config)
    dbpc_spec = next(
        spec
        for spec in specs.values()
        if spec.base_config.get("tx_power_model_id") == "islam_dbpc_itm6"
    )
    paths = {
        "sweep_config": definition.config_path,
        "e0_config": E0_CONFIG,
        "comparison_config": definition.comparison_config,
        "aa_model_config": comparison.models[definition.source_model_ids["aa"]],
        "dbpc_model_config": comparison.models[
            definition.source_model_ids["islam_dbpc_itm6"]
        ],
        "dbpc_calibration_manifest": Path(
            dbpc_spec.base_config["tx_power_calibration_manifest"]
        ),
    }
    wang_source = definition.source_model_ids.get("wang_atp_sf7_a2only")
    if wang_source is not None:
        paths["wang_model_config"] = comparison.models[wang_source]
    dbpc_802_source = definition.source_model_ids.get("islam_dbpc_802")
    if dbpc_802_source is not None:
        paths["dbpc_802_model_config"] = comparison.models[dbpc_802_source]
    if definition.routing_snapshot_bindings_path is not None:
        paths["routing_snapshot_bindings"] = (
            definition.routing_snapshot_bindings_path
        )
        for seed, binding in sorted(definition.snapshot_bindings.items()):
            paths[f"routing_snapshot_manifest_seed_{seed}"] = (
                binding.manifest_path
            )
    return {
        label: {"path": str(path), "sha256": file_sha256(path)}
        for label, path in paths.items()
    }


def _git_provenance() -> dict[str, Any]:
    """Return local commit identity without weakening the source fingerprint."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        complete_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(tracked_status)
    except (OSError, subprocess.CalledProcessError):
        return {
            "head": None,
            "tracked_worktree_dirty": None,
            "complete_worktree_dirty": None,
            "status_porcelain_sha256": None,
        }
    return {
        "head": head,
        "tracked_worktree_dirty": dirty,
        "complete_worktree_dirty": bool(complete_status),
        "status_porcelain_sha256": content_hash(
            complete_status.splitlines(), length=64
        ),
    }


def _initialize_parent(
    definition: SweepDefinition,
    specs: Mapping[str, ExperimentSpec],
) -> ExperimentArtifactPaths:
    paths = ExperimentArtifactPaths(definition.experiment_dir).ensure()
    (definition.experiment_dir / "models").mkdir(parents=True, exist_ok=True)
    definition_payload = definition.serializable()
    definition_hash = content_hash(definition_payload, length=64)
    source_fingerprint = _source_fingerprint(PROJECT_ROOT)
    if paths.manifest.exists():
        existing = _read_json(paths.manifest)
        if existing.get("definition_hash") != definition_hash:
            raise ValueError("Existing sweep manifest has a different experiment definition")
        if existing.get("source_fingerprint") != source_fingerprint:
            raise ValueError(
                "Source fingerprint differs from the existing sweep manifest; "
                "use a new experiment_id"
            )
        return paths
    atomic_write_json(
        paths.manifest,
        {
            "manifest_version": (
                "apc_compare_model_terrain_sweep_manifest_v2"
                if definition.experiment_profile
                in {
                    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
                    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE,
                    MAIN_W4_PROFILE,
                    REVIEWER_R1_5_PROFILE,
                }
                else "apc_compare_model_terrain_sweep_manifest_v1"
            ),
            "created_at": utc_now(),
            "experiment_id": definition.experiment_id,
            "definition_hash": definition_hash,
            "definition": definition_payload,
            "source_files": _source_records(definition, specs),
            "source_fingerprint": source_fingerprint,
            "source_git": _git_provenance(),
            "experiment_track": definition.experiment_track,
            "reporting_namespace": definition.reporting_namespace,
            "routing_reporting_contract": routing_reporting_contract(
                definition.experiment_profile
            ),
            "snapshot_creation_provenance": {
                str(seed): dict(binding.creation_provenance)
                for seed, binding in sorted(definition.snapshot_bindings.items())
            },
            "algorithm_variants": {
                "ff": "fixed_operational_maximum",
                "aa": "relay_aware_apc_distributed_feedback",
                "islam_dbpc_itm6": "islam_algorithm2_fixed_sf7_itm90_margin6",
                **(
                    {
                        "islam_dbpc_802": (
                            "islam_algorithm2_fixed_sf7_ieee80211ah_margin0"
                        )
                    }
                    if "islam_dbpc_802" in definition.models
                    else {}
                ),
                **(
                    {"wang_atp_sf7_a2only": "wang_algorithm2_only_fixed_sf7"}
                    if "wang_atp_sf7_a2only" in definition.models
                    else {}
                ),
            },
            "child_manifest_protection": True,
            "planned_run_count": (
                len(specs)
                * len(definition.factors["terrain_delta_h"])
                * len(definition.seeds)
            ),
        },
    )
    return paths


def run_experiments(
    config: str | Path = DEFAULT_CONFIG,
    *,
    resume: bool = True,
    allow_incomplete: bool = False,
    fail_fast: bool = False,
    max_runs: int | None = None,
    dry_run: bool = False,
    show_progress: bool = True,
    build_artifacts: bool = True,
) -> dict[str, Any]:
    """Run protected 75-run child grids and assemble one comparison."""
    definition = load_sweep_definition(config)
    specs = build_model_specs(config)
    grid = planned_grid(config)
    if dry_run:
        return {
            "status": "DRY_RUN",
            "experiment_id": definition.experiment_id,
            "experiment_dir": str(definition.experiment_dir),
            "run_count": len(grid),
            "unique_cell_count": len(
                {
                    (
                        row["model_id"],
                        row["terrain_delta_h"],
                        row["seed"],
                        row["routing_strategy"],
                    )
                    for row in grid
                }
            ),
            "models": list(definition.models),
            "experiment_track": definition.experiment_track,
            "reporting_namespace": definition.reporting_namespace,
            "routing_strategy": definition.routing_strategy,
            "routing_strategies": list(definition.routing_strategies),
            "runs": grid,
        }

    preflight_result = None
    if definition.experiment_profile == PAIRED_ROUTING_PILOT_PROFILE:
        from loraphymac.experiments.apc_compare_model_terrain_sweep.paired_routing_pilot import (
            preflight_paired_routing_pilot,
        )

        preflight_result = preflight_paired_routing_pilot(
            config,
            require_new_output=not definition.experiment_dir.exists(),
        )
    parent_paths = _initialize_parent(definition, specs)
    if preflight_result is not None:
        atomic_write_json(definition.experiment_dir / "preflight.json", preflight_result)
    requested_run_count = len(grid)
    child_summaries: dict[str, Any] = {}
    for child_id, spec in specs.items():
        summary = run_experiment(
            spec,
            project_root=PROJECT_ROOT,
            resume=resume,
            allow_incomplete=allow_incomplete,
            fail_fast=fail_fast,
            max_runs=max_runs,
            dry_run=False,
            show_progress=show_progress,
        )
        child_summaries[child_id] = summary
        complete = summary.get("status") == "PASS" and summary.get(
            "seed_grid_complete"
        ) is True
        if not complete:
            atomic_write_json(
                parent_paths.status,
                {
                    "status": "INCOMPLETE",
                    "experiment_id": definition.experiment_id,
                    "requested_run_count": requested_run_count,
                    "finished_at": utc_now(),
                    "child_summaries": child_summaries,
                },
            )
            return _read_json(parent_paths.status)

    summary: dict[str, Any] = {
        "status": "PASS",
        "experiment_id": definition.experiment_id,
        "requested_run_count": requested_run_count,
        "passed_run_count": requested_run_count,
        "seed_grid_complete": True,
        "finished_at": utc_now(),
        "child_summaries": child_summaries,
        "preflight": preflight_result,
    }
    atomic_write_json(parent_paths.status, summary)
    if build_artifacts:
        if definition.experiment_profile == PAIRED_ROUTING_PILOT_PROFILE:
            from loraphymac.experiments.apc_compare_model_terrain_sweep.paired_routing_pilot import (
                build_paired_pilot_artifacts,
            )

            summary["analysis_artifacts"] = build_paired_pilot_artifacts(config)
        else:
            from loraphymac.experiments.apc_compare_model_terrain_sweep.postprocessing import (
                build_comparison_artifacts,
            )

            summary["analysis_artifacts"] = build_comparison_artifacts(config)
        atomic_write_json(parent_paths.status, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-postprocess", action="store_true")
    args = parser.parse_args(argv)
    result = run_experiments(
        args.config,
        resume=not args.no_resume,
        allow_incomplete=args.allow_incomplete,
        fail_fast=args.fail_fast,
        max_runs=args.max_runs,
        dry_run=args.dry_run,
        show_progress=not args.quiet,
        build_artifacts=not args.no_postprocess,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status") in {"PASS", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONFIG",
    "E0_CONFIG",
    "FIVE_MODEL_PILOT_CONFIG",
    "LEGACY_CONFIG",
    "MAIN_W4_CONFIG",
    "MODEL_LABELS",
    "PILOT_MODEL_LABELS",
    "PROJECT_ROOT",
    "REVIEWER_R1_5_CONFIG",
    "PAIRED_ROUTING_PILOT_CONFIG",
    "SweepDefinition",
    "build_model_specs",
    "load_sweep_definition",
    "model_label",
    "planned_grid",
    "run_experiments",
]
