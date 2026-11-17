"""Central, fail-fast validation for scientific simulation configuration.

The validator operates on a fully resolved configuration.  It deliberately
rejects implicit coercions such as ``True`` as seed ``1`` or ``1.5`` as an
integer packet count: silent coercion makes experiment manifests ambiguous and
can compromise reproducibility.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

from loraphymac.experiments.common.contract_registry import (
    experiment_config_violations,
    registered_contract_profiles,
)
from loraphymac.experiments.common.scientific_output_contract import (
    preflight_output_violations,
)
from loraphymac.hardware.hardware import SX1276, SX1302
from loraphymac.mac.collision_engine import CollisionEngine
from loraphymac.runtime.tx_power_policy import (
    ISLAM_DBPC_802_POLICY_ID,
    ISLAM_DBPC_POLICY_ID,
    LEGACY_POLICY_ID,
    SUPPORTED_TX_POWER_POLICIES,
)
from loraphymac.runtime.wang_atp_a2only import WANG_ATP_SF7_A2ONLY_MODEL_ID
from loraphymac.routing.snapshot import (
    ROUTING_REFERENCE_DELTA_H,
    ROUTING_STRATEGY_FIXED_REFERENCE,
    ROUTING_STRATEGY_TERRAIN_SPECIFIC,
)


CONFIGURATION_VALIDATION_VERSION = "central_config_v2"

SUPPORTED_PROFILES = {
    "general_multisf_multibw",
    "paper1_sf7_bw125",
    "apc_compare_model_v1",
    *registered_contract_profiles(),
}
SUPPORTED_SFS = frozenset(range(7, 13))
SUPPORTED_BW_HZ = frozenset({125_000, 250_000, 500_000})
SUPPORTED_CHANNEL_IDS = frozenset(range(8))
MAX_NUMPY_LEGACY_SEED = 2**32 - 1
NODE_Y_EXCLUSION_MIN_M = -1_000.0
NODE_Y_EXCLUSION_MAX_M = 1_000.0
MIN_NODE_PLACEMENT_ACCEPTANCE_FRACTION = 0.01
ITM_MAX_DISTANCE_M = 2_000_000.0
OPTIONAL_CONFIGURATION_KEYS = frozenset(
    {
        "persistence_profile",
        "gzip_scientific_tables",
        "feedback_additional_loss_probability",
    }
)


class ConfigurationError(ValueError):
    """Raised when a simulation configuration is unsafe or inconsistent."""


def _error(key: str, value: Any, requirement: str) -> ConfigurationError:
    return ConfigurationError(f"Invalid configuration {key!r}={value!r}: {requirement}")


def _value(config: Mapping[str, Any], key: str) -> Any:
    try:
        return config[key]
    except KeyError as exc:
        raise ConfigurationError(f"Configuration is missing required key {key!r}") from exc


def _finite_number(
    config: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    value = _value(config, key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _error(key, value, "must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise _error(key, value, "must be a finite number")
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            operator = ">=" if minimum_inclusive else ">"
            raise _error(key, value, f"must be {operator} {minimum:g}")
    if maximum is not None:
        invalid = result > maximum if maximum_inclusive else result >= maximum
        if invalid:
            operator = "<=" if maximum_inclusive else "<"
            raise _error(key, value, f"must be {operator} {maximum:g}")
    return result


def _integer(
    config: Mapping[str, Any],
    key: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = _value(config, key)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise _error(key, value, "must be an integer (booleans are not accepted)")
    result = int(value)
    if minimum is not None and result < minimum:
        raise _error(key, value, f"must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise _error(key, value, f"must be <= {maximum}")
    return result


def _boolean(config: Mapping[str, Any], key: str) -> bool:
    value = _value(config, key)
    if not isinstance(value, bool):
        raise _error(key, value, "must be a boolean")
    return value


def _choice(config: Mapping[str, Any], key: str, choices: set[str] | frozenset[str]) -> str:
    value = _value(config, key)
    if not isinstance(value, str):
        raise _error(key, value, f"must be one of {sorted(choices)}")
    normalized = value.lower().strip()
    if normalized not in choices:
        raise _error(key, value, f"must be one of {sorted(choices)}")
    return normalized


def validate_config_keys(
    config: Mapping[str, Any],
    known_keys: Mapping[str, Any] | Sequence[str],
    *,
    label: str = "configuration",
) -> None:
    """Reject non-mappings and misspelled/unknown simulator keys."""
    if not isinstance(config, Mapping):
        raise ConfigurationError(f"{label} must be a mapping")
    # Output persistence selectors are optional. Keeping them out of
    # DEFAULT_CONFIG prevents them from changing historical resolved-config
    # hashes when omitted.
    allowed = set(known_keys) | OPTIONAL_CONFIGURATION_KEYS
    unknown = set(config).difference(allowed)
    if unknown:
        raise ConfigurationError(f"{label} contains unknown simulator keys: {sorted(unknown)}")


def validate_seed(value: Any, *, key: str = "seed") -> int:
    """Validate a seed accepted by NumPy's legacy global RNG API."""
    temporary = {key: value}
    return _integer(
        temporary,
        key,
        minimum=0,
        maximum=MAX_NUMPY_LEGACY_SEED,
    )


def _validate_optional_integer(
    config: Mapping[str, Any],
    key: str,
    *,
    minimum: int,
) -> int | None:
    if _value(config, key) is None:
        return None
    return _integer(config, key, minimum=minimum)


def _validate_optional_ptx(
    config: Mapping[str, Any],
    key: str,
    supported_values: set[int],
) -> None:
    value = _value(config, key)
    if value is None:
        return
    ptx = _integer(config, key)
    if ptx not in supported_values:
        raise _error(
            key,
            value,
            f"must be one of the hardware- and EIRP-compliant values {sorted(supported_values)}",
        )


def _operational_ptx_values(
    hardware: type,
    max_eirp_dbm: float,
    antenna_gain_dbi: float,
    feeder_loss_db: float,
    label: str,
) -> set[int]:
    ceiling = max_eirp_dbm - antenna_gain_dbi + feeder_loss_db
    supported = {
        int(value)
        for value in hardware.tx_current
        if int(hardware.tx_power_min) <= int(value) <= int(hardware.tx_power_max)
        and float(value) <= ceiling
    }
    if not supported:
        raise ConfigurationError(
            f"Invalid {label} transmit-power constraints: no hardware power "
            f"satisfies EIRP ceiling {max_eirp_dbm:g} dBm with antenna gain "
            f"{antenna_gain_dbi:g} dBi and feeder loss {feeder_loss_db:g} dB"
        )
    return supported


def _matrix_threshold(matrix: Mapping, desired_sf: int, interferer_sf: int) -> float:
    try:
        row = matrix[desired_sf]
    except (KeyError, TypeError):
        row = matrix[str(desired_sf)]
    try:
        value = row[interferer_sf]
    except (KeyError, TypeError):
        value = row[str(interferer_sf)]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("SIR threshold must be finite")
    return result


def validate_experiment_profile(config: Mapping[str, Any]) -> None:
    """Validate experiment-specific constraints without restricting the core model."""
    profile = _choice(config, "experiment_profile", SUPPORTED_PROFILES)
    if profile in registered_contract_profiles():
        violations = experiment_config_violations(profile, config)
        if violations:
            raise ConfigurationError(
                f"{profile} contract violation: " + "; ".join(violations)
            )
        return
    if profile != "paper1_sf7_bw125":
        return

    violations = []
    if list(config["node_sf_list"]) != [7]:
        violations.append("node_sf_list must be [7]")
    if config["node_bw"] != 125_000:
        violations.append("node_bw must be 125000 Hz")
    if config["relay_sf"] != 7:
        violations.append("relay_sf must be 7")
    if config["relay_bw_hz"] != 125_000:
        violations.append("relay_bw_hz must be 125000 Hz")
    if str(config["collision_mode"]).lower().strip() != "capture":
        violations.append("collision_mode must be 'capture'")

    sf_mode = str(config["collision_sf_interference_mode"]).lower().strip()
    if sf_mode == "sir_matrix":
        matrix = config["collision_sir_matrix"]
        if matrix is None:
            preset = str(config["collision_sir_matrix_preset"]).lower().strip()
            matrix = CollisionEngine.SIR_MATRIX_PRESETS.get(preset)
        try:
            co_sf_threshold = _matrix_threshold(matrix, 7, 7)
        except (KeyError, TypeError, ValueError):
            violations.append("SIR matrix must define desired SF7 vs interferer SF7")
        else:
            if abs(co_sf_threshold - 6.0) > 1e-12:
                violations.append("SF7 co-SF SIR threshold must be 6 dB")
    elif abs(float(config["capture_threshold_db"]) - 6.0) > 1e-12:
        violations.append("capture_threshold_db must be 6 dB")

    if violations:
        raise ConfigurationError("paper1_sf7_bw125 profile violation: " + "; ".join(violations))


def _validate_collision_configuration(config: Mapping[str, Any], sfs: set[int]) -> None:
    mode = _choice(config, "collision_mode", {"capture", "sinr"})
    sf_mode = _choice(
        config,
        "collision_sf_interference_mode",
        {"ideal_orthogonal", "sir_matrix", "common_threshold"},
    )
    capture_threshold = _finite_number(config, "capture_threshold_db")
    noise_floor = _finite_number(config, "collision_noise_floor_dbm")
    enforce_sf_match = _boolean(config, "collision_enforce_sf_match")
    destructive = _boolean(config, "collision_destructive")

    matrix = _value(config, "collision_sir_matrix")
    version = _value(config, "collision_sir_matrix_version")
    preset = _value(config, "collision_sir_matrix_preset")
    if matrix is not None and not isinstance(matrix, Mapping):
        raise _error("collision_sir_matrix", matrix, "must be a mapping or None")
    if version is not None and (not isinstance(version, str) or not version.strip()):
        raise _error(
            "collision_sir_matrix_version",
            version,
            "must be a non-empty string or None",
        )
    if matrix is None:
        if not isinstance(preset, str) or not preset.strip():
            raise _error("collision_sir_matrix_preset", preset, "must be a non-empty string")
        if version is not None:
            raise _error(
                "collision_sir_matrix_version",
                version,
                "must be None when collision_sir_matrix is None",
            )
    elif preset is not None and (not isinstance(preset, str) or not preset.strip()):
        raise _error(
            "collision_sir_matrix_preset",
            preset,
            "must be a non-empty string or None",
        )

    try:
        engine = CollisionEngine(
            mode=mode,
            capture_threshold_dB=capture_threshold,
            enforce_sf_match=enforce_sf_match,
            sf_interference_mode=sf_mode,
            sir_matrix=matrix,
            sir_matrix_version=version,
            sir_matrix_preset=preset,
            noise_floor_dBm=noise_floor,
            destructive=destructive,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid collision configuration: {exc}") from exc

    if sf_mode == "sir_matrix":
        missing_pairs = []
        for desired_sf in sorted(sfs):
            for interferer_sf in sorted(sfs):
                try:
                    threshold = engine.required_sir_dB(desired_sf, interferer_sf)
                except ValueError:
                    missing_pairs.append((desired_sf, interferer_sf))
                    continue
                if not math.isfinite(float(threshold)):
                    raise _error(
                        "collision_sir_matrix",
                        matrix,
                        f"threshold SF{desired_sf}/SF{interferer_sf} must be finite",
                    )
        if missing_pairs:
            rendered = ", ".join(
                f"SF{desired}/SF{interferer}" for desired, interferer in missing_pairs
            )
            raise _error(
                "collision_sir_matrix",
                matrix,
                f"must define every configured desired/interferer pair; missing {rendered}",
            )


def validate_configuration(
    config: Mapping[str, Any],
    *,
    known_keys: Mapping[str, Any] | Sequence[str] | None = None,
) -> None:
    """Validate one fully resolved simulator configuration.

    Validation is side-effect free.  It should run before topology generation,
    output-directory creation, or any SimPy process is started.
    """
    if not isinstance(config, Mapping):
        raise ConfigurationError("configuration must be a mapping")
    if known_keys is not None:
        validate_config_keys(config, known_keys)

    gzip_scientific_tables = config.get("gzip_scientific_tables", False)
    if not isinstance(gzip_scientific_tables, bool):
        raise _error(
            "gzip_scientific_tables",
            gzip_scientific_tables,
            "must be a boolean",
        )
    if gzip_scientific_tables and config.get("persistence_profile") not in {
        "scientific",
        "diagnostic",
    }:
        raise _error(
            "gzip_scientific_tables",
            gzip_scientific_tables,
            "requires persistence_profile='scientific' or 'diagnostic'",
        )

    _choice(config, "output_profile", {"summary", "detailed"})

    sim_time = _finite_number(config, "sim_time", minimum=0, minimum_inclusive=False)
    del sim_time
    validate_seed(_value(config, "seed"))

    routing_strategy = _choice(
        config,
        "routing_strategy",
        {
            ROUTING_STRATEGY_TERRAIN_SPECIFIC,
            ROUTING_STRATEGY_FIXED_REFERENCE,
        },
    )
    routing_reference_delta_h = _value(config, "routing_reference_delta_h")
    routing_manifest = _value(config, "routing_snapshot_manifest")
    routing_manifest_sha256 = _value(
        config, "routing_snapshot_manifest_sha256"
    )
    if routing_strategy == ROUTING_STRATEGY_TERRAIN_SPECIFIC:
        if any(
            value is not None
            for value in (
                routing_reference_delta_h,
                routing_manifest,
                routing_manifest_sha256,
            )
        ):
            raise _error(
                "routing_strategy",
                routing_strategy,
                "terrain_specific must not declare a frozen-routing snapshot",
            )
    else:
        if (
            isinstance(routing_reference_delta_h, bool)
            or not isinstance(routing_reference_delta_h, Integral)
            or int(routing_reference_delta_h) != ROUTING_REFERENCE_DELTA_H
        ):
            raise _error(
                "routing_reference_delta_h",
                routing_reference_delta_h,
                f"must be integer {ROUTING_REFERENCE_DELTA_H} for fixed_reference",
            )
        if not isinstance(routing_manifest, (str, os.PathLike)) or not str(
            routing_manifest
        ).strip():
            raise _error(
                "routing_snapshot_manifest",
                routing_manifest,
                "must identify a prebuilt immutable manifest for fixed_reference",
            )
        if not isinstance(routing_manifest_sha256, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", routing_manifest_sha256
        ):
            raise _error(
                "routing_snapshot_manifest_sha256",
                routing_manifest_sha256,
                "must be a 64-character SHA-256 for fixed_reference",
            )

    area_x = _finite_number(config, "area_x", minimum=0, minimum_inclusive=False)
    area_y = _finite_number(config, "area_y", minimum=0, minimum_inclusive=False)
    exclusion_width = NODE_Y_EXCLUSION_MAX_M - NODE_Y_EXCLUSION_MIN_M
    if area_y <= exclusion_width:
        raise _error(
            "area_y",
            config["area_y"],
            f"must be > {exclusion_width:g} m so node placement has space "
            "outside the central exclusion zone",
        )
    half_height = area_y / 2.0
    available_y_span = max(
        0.0,
        min(half_height, NODE_Y_EXCLUSION_MIN_M) - (-half_height),
    ) + max(
        0.0,
        half_height - max(-half_height, NODE_Y_EXCLUSION_MAX_M),
    )
    acceptance_fraction = available_y_span / area_y
    if acceptance_fraction < MIN_NODE_PLACEMENT_ACCEPTANCE_FRACTION:
        raise _error(
            "area_y",
            config["area_y"],
            "leaves less than 1% of the y range available outside the central "
            "exclusion zone; enlarge area_y",
        )
    if math.hypot(area_x, area_y) > ITM_MAX_DISTANCE_M:
        raise ConfigurationError(
            "Invalid configuration area_x/area_y: topology diagonal exceeds "
            "the ITM maximum distance of 2000 km"
        )

    _integer(config, "node_num", minimum=0)
    _integer(config, "relay_num", minimum=0)
    _integer(config, "node_payload_size", minimum=0)
    period = _finite_number(config, "node_period", minimum=0, minimum_inclusive=False)
    jitter = _finite_number(config, "node_jitter", minimum=0)
    if jitter >= period:
        raise _error(
            "node_jitter",
            config["node_jitter"],
            "must be smaller than node_period so every generated interval is positive",
        )
    scheduling_mode = _choice(
        config,
        "application_scheduling_mode",
        {"legacy_post_tx_done_v1", "independent_fifo_v1"},
    )
    rng_policy = _choice(
        config,
        "application_rng_policy",
        {"legacy_global_random_v1", "sha256_seed_node_v1"},
    )
    queue_policy = _choice(
        config,
        "application_queue_policy",
        {"none", "unbounded_fifo_v1"},
    )
    horizon_policy = _choice(
        config,
        "application_horizon_policy",
        {"physical_tx_offer_v1", "right_censor_waiting_v1"},
    )
    _finite_number(config, "application_initial_delay_max_s", minimum=0)
    scheduling_contract = {
        "legacy_post_tx_done_v1": (
            "legacy_global_random_v1",
            "none",
            "physical_tx_offer_v1",
        ),
        "independent_fifo_v1": (
            "sha256_seed_node_v1",
            "unbounded_fifo_v1",
            "right_censor_waiting_v1",
        ),
    }
    expected_rng, expected_queue, expected_horizon = scheduling_contract[
        scheduling_mode
    ]
    for key, actual, expected in (
        ("application_rng_policy", rng_policy, expected_rng),
        ("application_queue_policy", queue_policy, expected_queue),
        ("application_horizon_policy", horizon_policy, expected_horizon),
    ):
        if actual != expected:
            raise _error(
                key,
                config[key],
                f"must be {expected!r} for {scheduling_mode}",
            )

    sf_values = _value(config, "node_sf_list")
    if isinstance(sf_values, (str, bytes)) or not isinstance(sf_values, Sequence) or not sf_values:
        raise _error("node_sf_list", sf_values, "must be a non-empty sequence")
    node_sfs = []
    for index, sf in enumerate(sf_values):
        if isinstance(sf, bool) or not isinstance(sf, Integral):
            raise _error("node_sf_list", sf_values, f"item {index} must be an integer SF")
        node_sfs.append(int(sf))
    unsupported_sfs = sorted(set(node_sfs).difference(SUPPORTED_SFS))
    if unsupported_sfs:
        raise _error(
            "node_sf_list",
            sf_values,
            f"supported spreading factors are {sorted(SUPPORTED_SFS)}",
        )
    if len(node_sfs) != len(set(node_sfs)):
        raise _error("node_sf_list", sf_values, "must not contain duplicates")

    relay_sf = _integer(config, "relay_sf")
    routing_sf = _integer(config, "routing_assumed_sf")
    for key, sf in (("relay_sf", relay_sf), ("routing_assumed_sf", routing_sf)):
        if sf not in SUPPORTED_SFS:
            raise _error(key, sf, f"must be one of {sorted(SUPPORTED_SFS)}")
    for key in ("node_bw", "relay_bw_hz"):
        bandwidth = _integer(config, key)
        if bandwidth not in SUPPORTED_BW_HZ:
            raise _error(key, bandwidth, f"must be one of {sorted(SUPPORTED_BW_HZ)} Hz")

    routing_channel = _integer(config, "routing_ch_id")
    if routing_channel not in SUPPORTED_CHANNEL_IDS:
        raise _error(
            "routing_ch_id",
            routing_channel,
            f"must be one of {sorted(SUPPORTED_CHANNEL_IDS)}",
        )
    relay_channels = _value(config, "relay_tx_ch_ids")
    if (
        isinstance(relay_channels, (str, bytes))
        or not isinstance(relay_channels, Sequence)
        or not relay_channels
    ):
        raise _error("relay_tx_ch_ids", relay_channels, "must be a non-empty sequence")
    normalized_channels = []
    for index, channel in enumerate(relay_channels):
        if isinstance(channel, bool) or not isinstance(channel, Integral):
            raise _error(
                "relay_tx_ch_ids",
                relay_channels,
                f"item {index} must be an integer channel ID",
            )
        normalized_channels.append(int(channel))
    unsupported_channels = sorted(set(normalized_channels).difference(SUPPORTED_CHANNEL_IDS))
    if unsupported_channels:
        raise _error(
            "relay_tx_ch_ids",
            relay_channels,
            f"channel IDs must be in {sorted(SUPPORTED_CHANNEL_IDS)}",
        )
    if len(normalized_channels) != len(set(normalized_channels)):
        raise _error("relay_tx_ch_ids", relay_channels, "must not contain duplicates")

    for key in ("node_ant_height", "relay_ant_height", "gw_ant_height"):
        _finite_number(config, key, minimum=0.5, maximum=3_000.0)
    _finite_number(config, "terrain_delta_h", minimum=0)
    _finite_number(config, "routing_hop_penalty_alpha", minimum=0)
    for key in (
        "feasible_margin_db",
        "delta_snr",
        "node_target_margin_db",
        "relay_target_margin_db",
        "node_ant_gain",
        "relay_ant_gain",
        "gw_ant_gain",
    ):
        _finite_number(config, key)
    for key in ("node_feeder_loss_db", "relay_feeder_loss_db", "gw_feeder_loss_db"):
        _finite_number(config, key, minimum=0)
    for key in ("node_max_eirp_dbm", "relay_max_eirp_dbm", "gw_max_eirp_dbm"):
        _finite_number(config, key)

    node_ptx_values = _operational_ptx_values(
        SX1276,
        float(config["node_max_eirp_dbm"]),
        float(config["node_ant_gain"]),
        float(config["node_feeder_loss_db"]),
        "node",
    )
    relay_ptx_values = _operational_ptx_values(
        SX1302,
        float(config["relay_max_eirp_dbm"]),
        float(config["relay_ant_gain"]),
        float(config["relay_feeder_loss_db"]),
        "relay",
    )
    gateway_ptx_values = _operational_ptx_values(
        SX1302,
        float(config["gw_max_eirp_dbm"]),
        float(config["gw_ant_gain"]),
        float(config["gw_feeder_loss_db"]),
        "gateway",
    )
    _validate_optional_ptx(config, "node_ptx_init", node_ptx_values)
    _validate_optional_ptx(config, "relay_ptx_init", relay_ptx_values & gateway_ptx_values)
    common_ptx_values = node_ptx_values & relay_ptx_values & gateway_ptx_values
    for key in (
        "apc_downlink_ptx_dbm",
        "apc_downlink_node_ptx_dbm",
        "apc_downlink_relay_ptx_dbm",
    ):
        _validate_optional_ptx(config, key, common_ptx_values)

    node_mode = _choice(config, "node_ptx_mode", {"fixed", "adaptive"})
    relay_mode = _choice(config, "relay_ptx_mode", {"fixed", "adaptive"})
    energy_accounting_mode = _choice(
        config, "energy_accounting_mode", {"legacy", "corrected"}
    )
    comparison_profile = config["experiment_profile"] in {
        "apc_compare_model_v1",
        "apc_compare_model_terrain_sweep_v1",
        "apc_compare_model_terrain_sweep_v2",
        "apc_compare_model_terrain_sweep_v2_five_model_pilot",
        "apc_compare_model_terrain_sweep_main_w4_fixed_reference_v1",
        "apc_compare_model_terrain_sweep_reviewer_r1_5_terrain_specific_v1",
            "apc_compare_model_terrain_sweep_paired_routing_pilot_v1",
    }
    if comparison_profile and energy_accounting_mode != "corrected":
        raise _error(
            "energy_accounting_mode",
            energy_accounting_mode,
            "must be 'corrected' for APC model comparison profiles",
        )
    policy_id = _choice(config, "tx_power_policy_id", SUPPORTED_TX_POWER_POLICIES)
    model_id = _value(config, "tx_power_model_id")
    if model_id is not None and (
        not isinstance(model_id, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", model_id)
    ):
        raise _error("tx_power_model_id", model_id, "must be a stable identifier or None")
    _finite_number(config, "tx_power_calibration_margin_db")
    calibration_keys = (
        "tx_power_calibration_id",
        "tx_power_calibration_manifest",
        "tx_power_calibration_manifest_sha256",
    )
    if policy_id == ISLAM_DBPC_POLICY_ID:
        if model_id is None:
            raise _error("tx_power_model_id", model_id, "is required for Islam-DBPC")
        if not comparison_profile:
            raise _error(
                "experiment_profile",
                config["experiment_profile"],
                "must be 'apc_compare_model_v1' or "
                "'apc_compare_model_terrain_sweep_v1' for Islam-DBPC",
            )
        if node_mode != "fixed" or relay_mode != "fixed":
            raise _error(
                "tx_power_policy_id",
                policy_id,
                "requires node_ptx_mode='fixed' and relay_ptx_mode='fixed'",
            )
        for key in calibration_keys:
            value = _value(config, key)
            if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
                raise _error(key, value, "is required for Islam-DBPC")
        digest = str(config["tx_power_calibration_manifest_sha256"])
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise _error(
                "tx_power_calibration_manifest_sha256",
                digest,
                "must be a 64-character SHA-256 digest",
            )
        if float(config["tx_power_calibration_margin_db"]) not in {0.0, 6.0}:
            raise _error(
                "tx_power_calibration_margin_db",
                config["tx_power_calibration_margin_db"],
                "must be 0 or 6 dB for the frozen calibration",
            )
    elif policy_id == ISLAM_DBPC_802_POLICY_ID:
        if model_id != "islam_dbpc_802":
            raise _error(
                "tx_power_model_id",
                model_id,
                "must be 'islam_dbpc_802' for the DBPC-802 policy",
            )
        if not comparison_profile:
            raise _error(
                "experiment_profile",
                config["experiment_profile"],
                "must be an APC model-comparison profile for DBPC-802",
            )
        if node_mode != "fixed" or relay_mode != "fixed":
            raise _error(
                "tx_power_policy_id",
                policy_id,
                "requires node_ptx_mode='fixed' and relay_ptx_mode='fixed'",
            )
        if list(config["node_sf_list"]) != [7] or int(config["relay_sf"]) != 7:
            raise _error(
                "tx_power_policy_id",
                policy_id,
                "requires fixed SF7 for nodes and relays",
            )
        if int(config["routing_assumed_sf"]) != 7:
            raise _error(
                "routing_assumed_sf",
                config["routing_assumed_sf"],
                "must be SF7 for DBPC-802",
            )
        if int(config["node_bw"]) != 125_000 or int(config["relay_bw_hz"]) != 125_000:
            raise _error(
                "tx_power_policy_id",
                policy_id,
                "requires BW125 for nodes and relays",
            )
        if float(config["tx_power_calibration_margin_db"]) != 0.0:
            raise _error(
                "tx_power_calibration_margin_db",
                config["tx_power_calibration_margin_db"],
                "must be 0 dB for DBPC-802",
            )
        if any(config[key] is not None for key in calibration_keys):
            raise _error(
                "tx_power_policy_id",
                policy_id,
                "DBPC-802 is analytic and must not declare frozen calibration artifacts",
            )
    elif policy_id == LEGACY_POLICY_ID and any(
        config[key] is not None for key in calibration_keys
    ):
        raise _error(
            "tx_power_policy_id",
            policy_id,
            "legacy policy must not declare a DBPC calibration",
        )
    if model_id == WANG_ATP_SF7_A2ONLY_MODEL_ID:
        if policy_id != LEGACY_POLICY_ID:
            raise _error(
                "tx_power_policy_id",
                policy_id,
                "Wang A2-only uses the reporting/sink runtime, not a pre-traffic policy",
            )
        if node_mode != "fixed" or relay_mode != "fixed":
            raise _error(
                "tx_power_model_id",
                model_id,
                "requires fixed node and relay PTx modes",
            )
        if list(config["node_sf_list"]) != [7] or int(config["relay_sf"]) != 7:
            raise _error(
                "tx_power_model_id",
                model_id,
                "requires fixed SF7 for nodes and relays",
            )
        if config["node_ptx_init"] is not None or config["relay_ptx_init"] is not None:
            raise _error(
                "tx_power_model_id",
                model_id,
                "requires operational-maximum initial PTx (init values None)",
            )
        if energy_accounting_mode != "corrected":
            raise _error(
                "energy_accounting_mode",
                energy_accounting_mode,
                "must be 'corrected' for Wang A2-only",
            )
        if not _boolean(config, "frame_enabled"):
            raise _error(
                "frame_enabled",
                config["frame_enabled"],
                "must be True for physical Wang SNR reporting",
            )
        if _boolean(config, "frame_default_ack_req"):
            raise _error(
                "frame_default_ack_req",
                config["frame_default_ack_req"],
                "must be False for Wang A2-only (no data ACK/ARQ)",
            )
    _choice(config, "apc_downlink_mode", {"physical_all"})
    frame_enabled = _boolean(config, "frame_enabled")
    node_rx_enabled = _boolean(config, "node_rx_window_enabled")
    node_rx_collision_enabled = _boolean(config, "node_rx_collision_enabled")
    if node_rx_collision_enabled and not node_rx_enabled:
        raise _error(
            "node_rx_collision_enabled",
            node_rx_collision_enabled,
            "requires node_rx_window_enabled=True",
        )
    if (node_mode == "adaptive" or relay_mode == "adaptive") and not frame_enabled:
        raise _error(
            "frame_enabled",
            frame_enabled,
            "must be True when node or relay adaptive power is enabled",
        )
    if node_mode == "adaptive" and not node_rx_enabled:
        raise _error(
            "node_rx_window_enabled",
            node_rx_enabled,
            "must be True when node_ptx_mode='adaptive'",
        )
    _integer(config, "frame_version", minimum=0, maximum=3)
    _integer(config, "apc_fopts_size", minimum=0, maximum=15)
    _finite_number(config, "apc_downlink_delay_s", minimum=0)
    _finite_number(config, "apc_retry_timeout_s", minimum=0, minimum_inclusive=False)
    _finite_number(config, "node_rx1_delay_s", minimum=0)
    _finite_number(config, "node_rx_window_duration_s", minimum=0, minimum_inclusive=False)

    _validate_optional_integer(config, "relay_tx_waiting_queue_capacity_packets", minimum=0)
    _validate_optional_integer(config, "gateway_tx_waiting_queue_capacity_packets", minimum=0)
    _choice(config, "tx_queue_overflow_policy", {"drop_tail"})
    if _value(config, "relay_duty_cycle") is not None:
        _finite_number(
            config,
            "relay_duty_cycle",
            minimum=0,
            maximum=1,
            minimum_inclusive=False,
        )
    _finite_number(config, "rx_idle_poll_interval_ms", minimum=0, minimum_inclusive=False)
    _finite_number(config, "max_rx_idle_wait_ms", minimum=0)

    boolean_keys = (
        "collision_enforce_sf_match",
        "collision_destructive",
        "frame_enabled",
        "frame_validation_enabled",
        "frame_default_ack_req",
        "node_rx_window_enabled",
        "node_rx_collision_enabled",
        "export_summary",
        "export_results",
        "export_debug",
        "export_internal",
        "print_summary",
        "print_detail",
        "print_debug",
        "print_internal",
        "plot_routing",
        "include_mcu_energy",
        "verbose",
    )
    for key in boolean_keys:
        _boolean(config, key)

    _integer(config, "debug_preview_rows", minimum=2, maximum=100)

    output_dir = _value(config, "output_dir")
    output_requested = any(
        bool(config[key])
        for key in (
            "export_summary",
            "export_results",
            "export_debug",
            "export_internal",
            "plot_routing",
        )
    )
    if output_dir is None:
        if output_requested:
            raise _error(
                "output_dir",
                output_dir,
                "must be a non-empty path when export or routing plot is enabled",
            )
    elif not isinstance(output_dir, (str, os.PathLike)) or not str(output_dir).strip():
        raise _error("output_dir", output_dir, "must be a non-empty path or None")
    run_id = _value(config, "run_id")
    if run_id is not None:
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
            raise _error(
                "run_id",
                run_id,
                "must contain only letters, numbers, '_', '-', and '.'",
            )

    _validate_collision_configuration(config, set(node_sfs) | {relay_sf})
    validate_experiment_profile(config)
    output_violations = preflight_output_violations(config)
    if output_violations:
        raise ConfigurationError(
            "Scientific output contract violation: " + "; ".join(output_violations)
        )
