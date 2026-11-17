"""Scientific contracts for the matched APC-model terrain sweeps."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE = "apc_compare_model_terrain_sweep_v1"
APC_COMPARE_MODEL_TERRAIN_SWEEP_CONTRACT_VERSION = (
    "apc_compare_model_terrain_sweep_contract_v1.0"
)
APC_COMPARE_MODEL_TERRAIN_SWEEP_MODELS = ("ff", "aa", "islam_dbpc_itm6")
APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE = "apc_compare_model_terrain_sweep_v2"
APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_CONTRACT_VERSION = (
    "apc_compare_model_terrain_sweep_contract_v2.0"
)
APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_MODELS = (
    "ff",
    "aa",
    "islam_dbpc_itm6",
    "wang_atp_sf7_a2only",
)
APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE = (
    "apc_compare_model_terrain_sweep_v2_five_model_pilot"
)
APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_CONTRACT_VERSION = (
    "apc_compare_model_terrain_sweep_contract_v2.1_pilot"
)
APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_MODELS = (
    "ff",
    "aa",
    "islam_dbpc_802",
    "islam_dbpc_itm6",
    "wang_atp_sf7_a2only",
)
APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_TERRAINS = (90,)
APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_SEEDS = (30,)
APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_CHILD_RUN_COUNT = 1
APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_RUN_COUNT = 5
MAIN_W4_PROFILE = "apc_compare_model_terrain_sweep_main_w4_fixed_reference_v1"
MAIN_W4_CONTRACT_VERSION = "main_w4_fixed_reference_contract_v1.0"
MAIN_W4_EXPERIMENT_TRACK = "MAIN_W4"
MAIN_W4_REPORTING_NAMESPACE = "main_w4_fixed_reference"
MAIN_W4_MODELS = APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_MODELS
MAIN_W4_CHILD_RUN_COUNT = 75
MAIN_W4_RUN_COUNT = 375
REVIEWER_R1_5_PROFILE = (
    "apc_compare_model_terrain_sweep_reviewer_r1_5_terrain_specific_v1"
)
REVIEWER_R1_5_CONTRACT_VERSION = "reviewer_r1_5_routing_sensitivity_contract_v1.0"
REVIEWER_R1_5_EXPERIMENT_TRACK = "REVIEWER_R1_5"
REVIEWER_R1_5_REPORTING_NAMESPACE = "reviewer_r1_5_routing_sensitivity"
REVIEWER_R1_5_MODELS = MAIN_W4_MODELS
REVIEWER_R1_5_TERRAINS = (20, 150, 160)
REVIEWER_R1_5_COMPARISON_TERRAINS = (20, 90, 150, 160)
REVIEWER_R1_5_CHILD_RUN_COUNT = 15
REVIEWER_R1_5_RUN_COUNT = 75
PAIRED_ROUTING_PILOT_PROFILE = (
    "apc_compare_model_terrain_sweep_paired_routing_pilot_v1"
)
PAIRED_ROUTING_PILOT_CONTRACT_VERSION = "paired_routing_pilot_contract_v1.0"
PAIRED_ROUTING_PILOT_EXPERIMENT_TRACK = "DEVELOPMENT_PAIRED_ROUTING"
PAIRED_ROUTING_PILOT_REPORTING_NAMESPACE = "w3c_paired_routing_n100_s30_r01"
PAIRED_ROUTING_PILOT_MODELS = APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_MODELS
PAIRED_ROUTING_PILOT_TERRAINS = (20, 90, 160)
PAIRED_ROUTING_PILOT_SEEDS = (30,)
PAIRED_ROUTING_PILOT_STRATEGIES = ("fixed_reference", "terrain_specific")
PAIRED_ROUTING_PILOT_CHILD_RUN_COUNT = 3
PAIRED_ROUTING_PILOT_RUN_COUNT = 30
APC_COMPARE_MODEL_TERRAIN_SWEEP_TERRAINS = tuple(range(20, 161, 10))
APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS = (30, 36, 42, 48, 56)
APC_COMPARE_MODEL_TERRAIN_SWEEP_CHILD_RUN_COUNT = 75
APC_COMPARE_MODEL_TERRAIN_SWEEP_RUN_COUNT = 225
APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_RUN_COUNT = 300

DBPC_POLICY_ID = "islam_algorithm2_fixed_sf_ptx_v1"
DBPC_802_POLICY_ID = "islam_algorithm2_ieee80211ah_fixed_sf_ptx_v1"
DBPC_802_ESTIMATOR_ID = "ieee80211ah_outdoor_eq15_margin0_v1"
DBPC_CALIBRATION_ID = "islam_dbpc_itm90_margin0_fixed_sf7_v1"
DBPC_CALIBRATION_MANIFEST_SHA256 = (
    "a61f15f546c9480e9439dc87740f537a9db2f1c845e3654b5becec464bffe6bf"
)
CORRECTED_ENERGY_MODEL_VERSION = "eq10_exclusive_transition_v1"

MODEL_CONFIG = {
    "ff": ("legacy", "fixed", "fixed"),
    "aa": ("legacy", "adaptive", "adaptive"),
    "islam_dbpc_802": (DBPC_802_POLICY_ID, "fixed", "fixed"),
    "islam_dbpc_itm6": (DBPC_POLICY_ID, "fixed", "fixed"),
    "wang_atp_sf7_a2only": ("legacy", "fixed", "fixed"),
}

ROUTING_REPORTING_CONTRACTS = {
    MAIN_W4_PROFILE: {
        "experiment_track": MAIN_W4_EXPERIMENT_TRACK,
        "reporting_namespace": MAIN_W4_REPORTING_NAMESPACE,
        "routing_strategies": ("fixed_reference",),
        "figure_stems": (
            "main_w4_pdr_vs_delta_h",
            "main_w4_energy_vs_delta_h",
            "main_w4_delay_vs_delta_h",
            "main_w4_energy_per_delivered_packet_vs_delta_h",
        ),
        "table_stems": (
            "main_w4_per_run_metrics",
            "main_w4_paired_seed_statistics",
        ),
    },
    REVIEWER_R1_5_PROFILE: {
        "experiment_track": REVIEWER_R1_5_EXPERIMENT_TRACK,
        "reporting_namespace": REVIEWER_R1_5_REPORTING_NAMESPACE,
        "routing_strategies": ("fixed_reference", "terrain_specific"),
        "figure_stems": (
            "r1_5_pdr_fixed_vs_terrain_specific",
            "r1_5_energy_fixed_vs_terrain_specific",
            "r1_5_delay_fixed_vs_terrain_specific",
        ),
        "table_stems": (
            "r1_5_parent_and_full_route_changes",
            "r1_5_paired_routing_statistics",
            "r1_5_model_by_routing_interaction",
        ),
    },
    PAIRED_ROUTING_PILOT_PROFILE: {
        "experiment_track": PAIRED_ROUTING_PILOT_EXPERIMENT_TRACK,
        "reporting_namespace": PAIRED_ROUTING_PILOT_REPORTING_NAMESPACE,
        "routing_strategies": PAIRED_ROUTING_PILOT_STRATEGIES,
        "figure_stems": (),
        "table_stems": (
            "development_fixed_reference_diagnostics",
            "development_r1_5_paired_routing_metrics",
        ),
        "publication_status": "development_only_single_seed",
    },
}

APC_COMPARE_MODEL_TERRAIN_SWEEP_RUN_TABLES = {
    "energy_breakdown_df": "energy_breakdown.csv",
    "final_ptx_distribution_df": "final_ptx_distribution.csv",
    "ptx_history_df": "ptx_history.csv",
    "run_metadata_df": "run_metadata.csv",
}

APC_COMPARE_MODEL_TERRAIN_SWEEP_WANG_RUN_TABLES = {
    "wang_snr_report_lifecycle_df": "wang_snr_report_lifecycle.csv",
    "wang_sink_decision_df": "wang_sink_decisions.csv",
    "wang_power_command_lifecycle_df": "wang_power_command_lifecycle.csv",
}

APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_APPLICATION_RUN_TABLES = {
    "application_queue_metrics_df": "application_queue_metrics.csv",
}

APC_COMPARE_MODEL_TERRAIN_SWEEP_AGGREGATION_METRICS = (
    "pdr_percent",
    "delivery_by_horizon_percent",
    "avg_e2e_delay_ms",
    "total_node_energy_J",
    "total_relay_energy_J",
    "total_gateway_energy_J",
    "total_network_energy_J",
    "energy_per_delivered_packet_J",
    "packet_fate_offered_count",
    "packet_fate_delivered_count",
    "packet_fate_terminal_drop_count",
    "packet_fate_censored_count",
    "packet_fate_conservation_delta",
    "collision_outcome_conservation_delta",
    "apc_total_tx",
    "apc_total_rx",
    "apc_total_drop",
    "ptx_update_count",
)


class APCCompareModelTerrainSweepContractError(ValueError):
    """Raised when a planned or completed comparison violates its contract."""


def _profile_models(profile: Any) -> tuple[str, ...]:
    if profile == APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE:
        return APC_COMPARE_MODEL_TERRAIN_SWEEP_MODELS
    if profile == APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE:
        return APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_MODELS
    if profile == APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE:
        return APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_MODELS
    if profile == MAIN_W4_PROFILE:
        return MAIN_W4_MODELS
    if profile == REVIEWER_R1_5_PROFILE:
        return REVIEWER_R1_5_MODELS
    if profile == PAIRED_ROUTING_PILOT_PROFILE:
        return PAIRED_ROUTING_PILOT_MODELS
    return ()


def _is_v2_profile(profile: Any) -> bool:
    return profile in {
        APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
        APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE,
        MAIN_W4_PROFILE,
        REVIEWER_R1_5_PROFILE,
        PAIRED_ROUTING_PILOT_PROFILE,
    }


def _profile_terrains(profile: Any) -> tuple[int, ...]:
    if profile == APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE:
        return APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_TERRAINS
    if profile == REVIEWER_R1_5_PROFILE:
        return REVIEWER_R1_5_TERRAINS
    if profile == PAIRED_ROUTING_PILOT_PROFILE:
        return PAIRED_ROUTING_PILOT_TERRAINS
    return APC_COMPARE_MODEL_TERRAIN_SWEEP_TERRAINS


def _profile_seeds(profile: Any) -> tuple[int, ...]:
    if profile == APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE:
        return APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_SEEDS
    if profile == PAIRED_ROUTING_PILOT_PROFILE:
        return PAIRED_ROUTING_PILOT_SEEDS
    return APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS


def _close(actual: Any, expected: Any, tolerance: float = 1e-5) -> bool:
    return math.isclose(
        float(actual),
        float(expected),
        rel_tol=1e-9,
        abs_tol=tolerance,
    )


def _model_violations(config: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    model_id = config.get("tx_power_model_id")
    allowed_models = _profile_models(config.get("experiment_profile"))
    if model_id not in allowed_models:
        return [
            f"tx_power_model_id must be one of {list(allowed_models)} "
            f"(got {model_id!r})"
        ]

    policy, node_mode, relay_mode = MODEL_CONFIG[str(model_id)]
    for key, expected in (
        ("tx_power_policy_id", policy),
        ("node_ptx_mode", node_mode),
        ("relay_ptx_mode", relay_mode),
    ):
        if config.get(key) != expected:
            violations.append(f"{key} must be {expected!r} for {model_id}")

    calibration_keys = (
        "tx_power_calibration_id",
        "tx_power_calibration_manifest",
        "tx_power_calibration_manifest_sha256",
    )
    if model_id == "islam_dbpc_itm6":
        expected = {
            "tx_power_calibration_id": DBPC_CALIBRATION_ID,
            "tx_power_calibration_manifest_sha256": DBPC_CALIBRATION_MANIFEST_SHA256,
            "tx_power_calibration_margin_db": 6,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                violations.append(f"{key} must be {value!r} for islam_dbpc_itm6")
        if not config.get("tx_power_calibration_manifest"):
            violations.append("tx_power_calibration_manifest is required for islam_dbpc_itm6")
    elif model_id == "islam_dbpc_802":
        if float(config.get("tx_power_calibration_margin_db", math.nan)) != 0.0:
            violations.append("tx_power_calibration_margin_db must be 0 for islam_dbpc_802")
        if any(config.get(key) is not None for key in calibration_keys):
            violations.append("islam_dbpc_802 must not declare calibration artifacts")
    elif any(config.get(key) is not None for key in calibration_keys):
        violations.append(f"{model_id} must not declare DBPC calibration artifacts")
    return violations


def apc_compare_model_terrain_sweep_config_violations(
    config: Mapping[str, Any],
) -> list[str]:
    """Return per-run configuration violations for the comparison profile."""
    profile = config.get("experiment_profile")
    if not _profile_models(profile):
        return []
    violations = _model_violations(config)
    expected = {
        "energy_accounting_mode": "corrected",
        "sim_time": 3600,
        "routing_assumed_sf": 7,
        "node_sf_list": [7],
        "relay_sf": 7,
    }
    for key, value in expected.items():
        actual = config.get(key)
        if key == "node_sf_list" and actual is not None:
            actual = list(actual)
        if actual != value:
            violations.append(f"{key} must be {value!r} (got {actual!r})")
    if _is_v2_profile(profile):
        v2_expected = {
            "node_num": (
                100
                if profile
                in {
                    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE,
                    MAIN_W4_PROFILE,
                    REVIEWER_R1_5_PROFILE,
                    PAIRED_ROUTING_PILOT_PROFILE,
                }
                else 60
            ),
            "relay_num": 2,
            "node_bw": 125000,
            "relay_bw_hz": 125000,
            "node_payload_size": 50,
            "node_rx_collision_enabled": True,
            "frame_default_ack_req": False,
            "node_ptx_init": None,
            "relay_ptx_init": None,
            "application_scheduling_mode": "independent_fifo_v1",
            "application_rng_policy": "sha256_seed_node_v1",
            "application_queue_policy": "unbounded_fifo_v1",
            "application_horizon_policy": "right_censor_waiting_v1",
        }
        for key, value in v2_expected.items():
            if config.get(key) != value:
                violations.append(f"{key} must be {value!r} (got {config.get(key)!r})")
    if profile == MAIN_W4_PROFILE:
        expected_routing = {
            "routing_strategy": "fixed_reference",
            "routing_reference_delta_h": 90,
        }
        for key, value in expected_routing.items():
            if config.get(key) != value:
                violations.append(f"{key} must be {value!r} (got {config.get(key)!r})")
        for key in (
            "routing_snapshot_manifest",
            "routing_snapshot_manifest_sha256",
        ):
            if not config.get(key):
                violations.append(f"{key} is required for MAIN_W4")
    elif profile == REVIEWER_R1_5_PROFILE:
        expected_routing = {
            "routing_strategy": "terrain_specific",
            "routing_reference_delta_h": None,
            "routing_snapshot_manifest": None,
            "routing_snapshot_manifest_sha256": None,
        }
        for key, value in expected_routing.items():
            if config.get(key) != value:
                violations.append(f"{key} must be {value!r} (got {config.get(key)!r})")
    elif profile == PAIRED_ROUTING_PILOT_PROFILE:
        strategy = config.get("routing_strategy")
        if strategy not in PAIRED_ROUTING_PILOT_STRATEGIES:
            violations.append(
                "routing_strategy must be fixed_reference or terrain_specific"
            )
        elif strategy == "fixed_reference":
            if config.get("routing_reference_delta_h") != 90:
                violations.append(
                    "fixed_reference routing_reference_delta_h must be 90"
                )
            for key in (
                "routing_snapshot_manifest",
                "routing_snapshot_manifest_sha256",
            ):
                if not config.get(key):
                    violations.append(f"{key} is required for fixed_reference")
        else:
            expected_routing = {
                "routing_reference_delta_h": None,
                "routing_snapshot_manifest": None,
                "routing_snapshot_manifest_sha256": None,
            }
            for key, value in expected_routing.items():
                if config.get(key) != value:
                    violations.append(
                        f"{key} must be {value!r} for terrain_specific"
                    )
    if config.get("terrain_delta_h") not in _profile_terrains(profile):
        violations.append(
            f"terrain_delta_h must be one of {list(_profile_terrains(profile))}"
        )
    if config.get("seed") not in _profile_seeds(profile):
        violations.append(f"seed must be one of {list(_profile_seeds(profile))}")
    return violations


def validate_apc_compare_model_terrain_sweep_spec(spec: Any) -> None:
    """Validate one 75-run child grid in either matched sweep version."""
    base = getattr(spec, "base_config", {})
    if not _profile_models(base.get("experiment_profile")):
        return
    errors = _model_violations(base)
    factors = getattr(spec, "factors", {})
    if tuple(factors) != ("terrain_delta_h",):
        errors.append("child factor must be exactly terrain_delta_h")
    elif tuple(factors["terrain_delta_h"]) != _profile_terrains(
        base.get("experiment_profile")
    ):
        errors.append(
            "terrain_delta_h factor does not match the registered profile"
        )
    if tuple(getattr(spec, "seeds", ())) != _profile_seeds(
        base.get("experiment_profile")
    ):
        errors.append("seeds do not match the registered profile")
    if base.get("energy_accounting_mode") != "corrected":
        errors.append("energy_accounting_mode must be corrected")
    if base.get("sim_time") != 3600:
        errors.append("sim_time must be 3600 s")
    if list(base.get("node_sf_list", ())) != [7] or base.get("relay_sf") != 7:
        errors.append("node and relay spreading factor must remain SF7")
    if float(getattr(spec, "confidence_level", 0.0)) != 0.95:
        errors.append("confidence_level must be 0.95")
    if bool(getattr(spec, "retain_debug_tables", False)):
        errors.append("retain_debug_tables must be false")
    if _is_v2_profile(base.get("experiment_profile")):
        probe_terrain = _profile_terrains(base.get("experiment_profile"))[0]
        probe_seed = _profile_seeds(base.get("experiment_profile"))[0]
        probe = {**base, "terrain_delta_h": probe_terrain, "seed": probe_seed}
        seed_overrides = getattr(spec, "seed_config_overrides", {})
        if seed_overrides:
            if set(seed_overrides) != set(getattr(spec, "seeds", ())):
                errors.append("seed snapshot bindings must cover every seed exactly")
            probe.update(seed_overrides.get(probe_seed, {}))
        for violation in apc_compare_model_terrain_sweep_config_violations(
            probe
        ):
            if violation not in errors:
                errors.append(violation)
    if errors:
        raise APCCompareModelTerrainSweepContractError("; ".join(errors))


def apc_compare_model_terrain_sweep_run_tables(
    config: Mapping[str, Any],
) -> dict[str, str]:
    profile = config.get("experiment_profile")
    if not _profile_models(profile):
        return {}
    tables = dict(APC_COMPARE_MODEL_TERRAIN_SWEEP_RUN_TABLES)
    if _is_v2_profile(profile):
        tables.update(APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_APPLICATION_RUN_TABLES)
        tables.update(
            {
                "wang_snr_report_lifecycle_df": (
                    APC_COMPARE_MODEL_TERRAIN_SWEEP_WANG_RUN_TABLES[
                        "wang_snr_report_lifecycle_df"
                    ]
                ),
                "wang_power_command_lifecycle_df": (
                    APC_COMPARE_MODEL_TERRAIN_SWEEP_WANG_RUN_TABLES[
                        "wang_power_command_lifecycle_df"
                    ]
                ),
            }
        )
        if config.get("tx_power_model_id") == "wang_atp_sf7_a2only":
            tables["wang_sink_decision_df"] = (
                APC_COMPARE_MODEL_TERRAIN_SWEEP_WANG_RUN_TABLES[
                    "wang_sink_decision_df"
                ]
            )
    return tables


def routing_reporting_contract(profile: Any) -> dict[str, Any]:
    """Return a copy of the isolated W4/R1.5 reporting namespace contract."""
    return dict(ROUTING_REPORTING_CONTRACTS.get(str(profile), {}))


def validate_apc_compare_model_terrain_sweep_outputs(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> None:
    """Fail closed on model identity, feedback, fate, and energy invariants."""
    if not _profile_models(config.get("experiment_profile")):
        return
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise APCCompareModelTerrainSweepContractError("pandas is required") from exc

    violations = apc_compare_model_terrain_sweep_config_violations(config)
    if violations:
        raise APCCompareModelTerrainSweepContractError("; ".join(violations))

    required_summary = {
        "energy_accounting_mode",
        "energy_accounting_model_version",
        "packet_fate_offered_count",
        "packet_fate_delivered_count",
        "packet_fate_terminal_drop_count",
        "packet_fate_censored_count",
        "packet_fate_conservation_delta",
        "collision_outcome_conservation_delta",
        "total_node_energy_J",
        "total_relay_energy_J",
        "total_gateway_energy_J",
        "total_network_energy_J",
        "energy_per_delivered_packet_J",
        "apc_total_tx",
        "apc_total_rx",
        "apc_total_drop",
        "ptx_update_count",
        "stage1_status",
    }
    missing = sorted(required_summary.difference(summary))
    if missing:
        raise APCCompareModelTerrainSweepContractError(
            f"summary is missing required fields: {missing}"
        )
    required_tables = {
        "packet_fate_df",
        "topology_parent_by_run_df",
        "apc_feedback_lifecycle_df",
        *apc_compare_model_terrain_sweep_run_tables(config),
    }
    missing_tables = sorted(required_tables.difference(outputs))
    if missing_tables:
        raise APCCompareModelTerrainSweepContractError(
            f"outputs are missing required tables: {missing_tables}"
        )
    if any(not isinstance(outputs[name], pd.DataFrame) for name in required_tables):
        raise APCCompareModelTerrainSweepContractError("required outputs must be DataFrames")

    if summary["energy_accounting_mode"] != "corrected" or summary[
        "energy_accounting_model_version"
    ] != CORRECTED_ENERGY_MODEL_VERSION:
        raise APCCompareModelTerrainSweepContractError(
            "corrected Eq. (10) energy accounting must be applied exactly once"
        )
    for key in ("packet_fate_conservation_delta", "collision_outcome_conservation_delta"):
        if int(summary[key]) != 0:
            raise APCCompareModelTerrainSweepContractError(f"{key} must be zero")
    if summary["stage1_status"] != "PASS":
        raise APCCompareModelTerrainSweepContractError("stage1_status must be PASS")

    offered = int(summary["packet_fate_offered_count"])
    delivered = int(summary["packet_fate_delivered_count"])
    terminal = int(summary["packet_fate_terminal_drop_count"])
    censored = int(summary["packet_fate_censored_count"])
    if offered != delivered + terminal + censored:
        raise APCCompareModelTerrainSweepContractError("packet fate counts do not conserve")

    if _is_v2_profile(config.get("experiment_profile")):
        application_summary_keys = {
            "application_generated_count",
            "application_dequeued_count",
            "application_queue_waiting_at_horizon_count",
            "application_queue_in_service_at_horizon_count",
            "application_queue_outstanding_at_horizon_count",
        }
        missing_application = sorted(application_summary_keys.difference(summary))
        if missing_application:
            raise APCCompareModelTerrainSweepContractError(
                "summary is missing application-queue fields: "
                f"{missing_application}"
            )
        queue = outputs["application_queue_metrics_df"]
        required_queue_columns = {
            "node_id",
            "scheduling_mode",
            "rng_policy",
            "queue_policy",
            "horizon_policy",
            "generated_count",
            "dequeued_count",
            "waiting_at_horizon_count",
            "in_service_at_horizon_count",
            "outstanding_at_horizon_count",
        }
        if queue.empty or not required_queue_columns.issubset(queue.columns):
            raise APCCompareModelTerrainSweepContractError(
                "V2 application queue evidence is empty or incomplete"
            )
        if len(queue) != int(config["node_num"]):
            raise APCCompareModelTerrainSweepContractError(
                "V2 application queue evidence must contain one row per node"
            )
        expected_queue_identity = {
            "scheduling_mode": "independent_fifo_v1",
            "rng_policy": "sha256_seed_node_v1",
            "queue_policy": "unbounded_fifo_v1",
            "horizon_policy": "right_censor_waiting_v1",
        }
        for key, expected in expected_queue_identity.items():
            if set(queue[key].astype(str)) != {expected}:
                raise APCCompareModelTerrainSweepContractError(
                    f"V2 application queue {key} does not match {expected}"
                )
        generated = int(queue["generated_count"].sum())
        dequeued = int(queue["dequeued_count"].sum())
        waiting = int(queue["waiting_at_horizon_count"].sum())
        in_service = int(queue["in_service_at_horizon_count"].sum())
        outstanding = int(queue["outstanding_at_horizon_count"].sum())
        if generated != offered or generated != dequeued + waiting:
            raise APCCompareModelTerrainSweepContractError(
                "application offered/admission/dequeue counts do not conserve"
            )
        expected_summary = {
            "application_generated_count": generated,
            "application_dequeued_count": dequeued,
            "application_queue_waiting_at_horizon_count": waiting,
            "application_queue_in_service_at_horizon_count": in_service,
            "application_queue_outstanding_at_horizon_count": outstanding,
        }
        if any(int(summary[key]) != value for key, value in expected_summary.items()):
            raise APCCompareModelTerrainSweepContractError(
                "application queue table does not match core summary"
            )

    component_total = sum(
        float(summary[key])
        for key in (
            "total_node_energy_J",
            "total_relay_energy_J",
            "total_gateway_energy_J",
        )
    )
    if not _close(summary["total_network_energy_J"], component_total):
        raise APCCompareModelTerrainSweepContractError("energy components do not conserve")
    expected_epp = component_total / delivered if delivered else 0.0
    if not _close(summary["energy_per_delivered_packet_J"], expected_epp):
        raise APCCompareModelTerrainSweepContractError(
            "energy_per_delivered_packet_J is inconsistent with delivered packets"
        )

    energy = outputs["energy_breakdown_df"]
    if energy.empty or set(energy["energy_accounting_mode"].astype(str)) != {"corrected"}:
        raise APCCompareModelTerrainSweepContractError(
            "energy breakdown must use corrected accounting"
        )
    if set(energy["energy_accounting_model_version"].astype(str)) != {
        CORRECTED_ENERGY_MODEL_VERSION
    }:
        raise APCCompareModelTerrainSweepContractError(
            "energy breakdown has an unexpected accounting model"
        )

    model_id = str(config["tx_power_model_id"])
    feedback = outputs["apc_feedback_lifecycle_df"]
    history = outputs["ptx_history_df"]
    if model_id in {"ff", "islam_dbpc_802", "islam_dbpc_itm6"}:
        counters = (
            int(summary["apc_total_tx"]),
            int(summary["apc_total_rx"]),
            int(summary["apc_total_drop"]),
            int(summary["ptx_update_count"]),
        )
        if any(counters) or not feedback.empty or not history.empty:
            raise APCCompareModelTerrainSweepContractError(
                f"{model_id} must not produce runtime APC feedback or PTx updates"
            )

    if model_id == "wang_atp_sf7_a2only":
        if any(int(summary[key]) for key in ("apc_total_tx", "apc_total_rx", "apc_total_drop")):
            raise APCCompareModelTerrainSweepContractError(
                "Wang A2-only must not produce R-APC feedback events"
            )
        if not feedback.empty:
            raise APCCompareModelTerrainSweepContractError(
                "Wang A2-only must not populate the R-APC feedback lifecycle"
            )
        decisions = outputs["wang_sink_decision_df"]
        required_decision_columns = {
            "decision_id",
            "window_index",
            "status",
            "reason",
            "coverage_fraction",
            "network_reduction_steps",
        }
        if decisions.empty or not required_decision_columns.issubset(decisions.columns):
            raise APCCompareModelTerrainSweepContractError(
                "Wang R21 decision evidence is empty or incomplete"
            )
        if decisions["window_index"].nunique() != decisions["decision_id"].nunique():
            raise APCCompareModelTerrainSweepContractError(
                "Wang must create exactly one decision record per observation window"
            )
        if int(summary.get("wang_closed_window_count", -1)) != int(
            decisions["decision_id"].nunique()
        ):
            raise APCCompareModelTerrainSweepContractError(
                "Wang closed-window count does not match R21 decisions"
            )

    if model_id == "islam_dbpc_itm6":
        final_ptx = outputs["final_ptx_distribution_df"]
        adaptive = final_ptx.loc[final_ptx["entity_type"].isin(["node", "relay"])]
        if adaptive.empty:
            raise APCCompareModelTerrainSweepContractError(
                "DBPC final PTx evidence is empty"
            )
        required_columns = {
            "final_ptx",
            "initial_ptx",
            "ptx_mode",
            "calibration_manifest_sha256",
            "calibration_margin_db",
            "power_control_feedback_enabled",
        }
        if not required_columns.issubset(adaptive.columns):
            raise APCCompareModelTerrainSweepContractError(
                "DBPC final PTx evidence is missing calibration fields"
            )
        fixed = (
            adaptive["final_ptx"].astype(float)
            == adaptive["initial_ptx"].astype(float)
        ).all()
        if not fixed or set(adaptive["ptx_mode"].astype(str)) != {"fixed"}:
            raise APCCompareModelTerrainSweepContractError(
                "DBPC calibrated PTx must remain fixed for the complete run"
            )
        if set(adaptive["calibration_manifest_sha256"].astype(str)) != {
            DBPC_CALIBRATION_MANIFEST_SHA256
        } or set(adaptive["calibration_margin_db"].astype(int)) != {6}:
            raise APCCompareModelTerrainSweepContractError(
                "DBPC final PTx evidence does not match the frozen margin-6 calibration"
            )
        feedback_enabled = adaptive["power_control_feedback_enabled"].astype("boolean")
        if feedback_enabled.fillna(True).any():
            raise APCCompareModelTerrainSweepContractError(
                "DBPC power-control feedback must remain disabled"
            )

    if model_id == "islam_dbpc_802":
        final_ptx = outputs["final_ptx_distribution_df"]
        active = final_ptx.loc[final_ptx["entity_type"].isin(["node", "relay"])]
        required_columns = {
            "final_ptx",
            "initial_ptx",
            "ptx_mode",
            "estimator_id",
            "policy_margin_db",
            "estimated_path_loss_db",
            "outside_estimated_coverage",
            "power_control_feedback_enabled",
        }
        if active.empty or not required_columns.issubset(active.columns):
            raise APCCompareModelTerrainSweepContractError(
                "DBPC-802 final PTx evidence is empty or incomplete"
            )
        fixed = (
            pd.to_numeric(active["final_ptx"], errors="raise")
            == pd.to_numeric(active["initial_ptx"], errors="raise")
        ).all()
        if not fixed or set(active["ptx_mode"].astype(str)) != {"fixed"}:
            raise APCCompareModelTerrainSweepContractError(
                "DBPC-802 PTx must remain fixed for the complete run"
            )
        if set(active["estimator_id"].astype(str)) != {DBPC_802_ESTIMATOR_ID}:
            raise APCCompareModelTerrainSweepContractError(
                "DBPC-802 estimator provenance is incorrect"
            )
        if set(pd.to_numeric(active["policy_margin_db"], errors="raise")) != {0.0}:
            raise APCCompareModelTerrainSweepContractError(
                "DBPC-802 policy margin must remain 0 dB"
            )
        feedback_enabled = active["power_control_feedback_enabled"].astype("boolean")
        if feedback_enabled.fillna(True).any():
            raise APCCompareModelTerrainSweepContractError(
                "DBPC-802 power-control feedback must remain disabled"
            )


__all__ = [
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_AGGREGATION_METRICS",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_CHILD_RUN_COUNT",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_CONTRACT_VERSION",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_MODELS",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_RUN_COUNT",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_RUN_TABLES",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_TERRAINS",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_CONTRACT_VERSION",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_MODELS",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_CHILD_RUN_COUNT",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_CONTRACT_VERSION",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_MODELS",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_RUN_COUNT",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_SEEDS",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_TERRAINS",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_RUN_COUNT",
    "APC_COMPARE_MODEL_TERRAIN_SWEEP_WANG_RUN_TABLES",
    "APCCompareModelTerrainSweepContractError",
    "DBPC_CALIBRATION_ID",
    "DBPC_CALIBRATION_MANIFEST_SHA256",
    "MAIN_W4_CHILD_RUN_COUNT",
    "MAIN_W4_CONTRACT_VERSION",
    "MAIN_W4_EXPERIMENT_TRACK",
    "MAIN_W4_MODELS",
    "MAIN_W4_PROFILE",
    "MAIN_W4_REPORTING_NAMESPACE",
    "MAIN_W4_RUN_COUNT",
    "PAIRED_ROUTING_PILOT_CHILD_RUN_COUNT",
    "PAIRED_ROUTING_PILOT_CONTRACT_VERSION",
    "PAIRED_ROUTING_PILOT_EXPERIMENT_TRACK",
    "PAIRED_ROUTING_PILOT_MODELS",
    "PAIRED_ROUTING_PILOT_PROFILE",
    "PAIRED_ROUTING_PILOT_REPORTING_NAMESPACE",
    "PAIRED_ROUTING_PILOT_RUN_COUNT",
    "PAIRED_ROUTING_PILOT_SEEDS",
    "PAIRED_ROUTING_PILOT_STRATEGIES",
    "PAIRED_ROUTING_PILOT_TERRAINS",
    "REVIEWER_R1_5_CHILD_RUN_COUNT",
    "REVIEWER_R1_5_COMPARISON_TERRAINS",
    "REVIEWER_R1_5_CONTRACT_VERSION",
    "REVIEWER_R1_5_EXPERIMENT_TRACK",
    "REVIEWER_R1_5_MODELS",
    "REVIEWER_R1_5_PROFILE",
    "REVIEWER_R1_5_REPORTING_NAMESPACE",
    "REVIEWER_R1_5_RUN_COUNT",
    "REVIEWER_R1_5_TERRAINS",
    "ROUTING_REPORTING_CONTRACTS",
    "apc_compare_model_terrain_sweep_config_violations",
    "apc_compare_model_terrain_sweep_run_tables",
    "routing_reporting_contract",
    "validate_apc_compare_model_terrain_sweep_outputs",
    "validate_apc_compare_model_terrain_sweep_spec",
]
