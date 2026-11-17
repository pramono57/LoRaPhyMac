"""Validated tables and IJIES figures for the unified APC-model terrain sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from loraphymac.experiments.apc_compare_model_terrain_sweep.contract import (
    APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_TERRAINS,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE,
    CORRECTED_ENERGY_MODEL_VERSION,
    DBPC_CALIBRATION_MANIFEST_SHA256,
    MAIN_W4_PROFILE,
    REVIEWER_R1_5_PROFILE,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.run import (
    DEFAULT_CONFIG,
    MODEL_LABELS,
    MODEL_SPECIFIC_CONFIG_KEYS,
    build_model_specs,
    load_sweep_definition,
    model_label,
)
from loraphymac.experiments.common.multiseed_pipeline import (
    aggregate_paper_grade,
    atomic_write_csv,
    atomic_write_json,
    content_hash,
    file_sha256,
    utc_now,
)
from loraphymac.experiments.common.outputs import ExperimentArtifactPaths
from loraphymac.hardware.hardware import SX1276, SX1302


METRICS = (
    "pdr_percent",
    "energy_per_delivered_packet_J",
    "avg_e2e_delay_ms",
    "total_network_energy_J",
)
PAIRWISE_CONTRASTS = (
    ("aa_minus_ff", "aa", "ff"),
    ("aa_minus_dbpc_802", "aa", "islam_dbpc_802"),
    ("aa_minus_dbpc", "aa", "islam_dbpc_itm6"),
    ("aa_minus_wang", "aa", "wang_atp_sf7_a2only"),
    ("ff_minus_dbpc", "ff", "islam_dbpc_itm6"),
    ("ff_minus_dbpc_802", "ff", "islam_dbpc_802"),
    ("ff_minus_wang", "ff", "wang_atp_sf7_a2only"),
    ("dbpc_minus_wang", "islam_dbpc_itm6", "wang_atp_sf7_a2only"),
    (
        "dbpc_802_minus_dbpc_itm",
        "islam_dbpc_802",
        "islam_dbpc_itm6",
    ),
)
FIGURES = (
    (
        "pdr_percent",
        "Packet delivery ratio (%)",
        "figure_08_pdr_vs_terrain",
    ),
    (
        "energy_per_delivered_packet_J",
        "Energy per delivered packet (J/packet)",
        "figure_09_energy_per_delivered_packet_vs_terrain",
    ),
    (
        "avg_e2e_delay_ms",
        "Average end-to-end delay (ms)",
        "figure_10_avg_e2e_delay_vs_terrain",
    ),
)

FIGURE_FONT_SIZE = 18
FIGURE_SIZE = (8, 5)
PLOT_MODEL_ORDER = (
    "aa",
    "islam_dbpc_802",
    "islam_dbpc_itm6",
    "wang_atp_sf7_a2only",
    "ff",
)
MODEL_STYLES = {
    "aa": {"color": "tab:blue", "marker": "o", "linestyle": "-"},
    "islam_dbpc_itm6": {
        "color": "tab:orange",
        "marker": "s",
        "linestyle": "--",
    },
    "islam_dbpc_802": {
        "color": "tab:purple",
        "marker": "X",
        "linestyle": (0, (3, 1, 1, 1)),
    },
    "ff": {"color": "tab:red", "marker": "D", "linestyle": ":"},
    "wang_atp_sf7_a2only": {
        "color": "tab:green",
        "marker": "^",
        "linestyle": "-.",
    },
}
FIGURE_STYLE = {
    "font.family": "Times New Roman",
    "font.size": FIGURE_FONT_SIZE,
    "axes.labelsize": FIGURE_FONT_SIZE,
    "axes.titlesize": FIGURE_FONT_SIZE,
    "xtick.labelsize": FIGURE_FONT_SIZE,
    "ytick.labelsize": FIGURE_FONT_SIZE,
    "legend.fontsize": FIGURE_FONT_SIZE,
    "legend.title_fontsize": FIGURE_FONT_SIZE,
    "figure.titlesize": FIGURE_FONT_SIZE,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
}

OFFERED_TRAFFIC_HASH_VERSION = "original_uplink_offer_uid_fcnt_time_ns_v2"
OFFERED_TRAFFIC_TIME_SCALE = 1_000_000_000


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _read_csv(path: Path, *, allow_empty: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        if allow_empty:
            return pd.DataFrame()
        raise


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return _read_csv(path, allow_empty=True)


def _single_row(frame: pd.DataFrame, path: Path) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f"{path}: expected exactly one row, got {len(frame)}")
    return frame.iloc[0]


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _close(actual: Any, expected: Any, tolerance: float = 1e-5) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=tolerance)


def _scientific_signature(resolved: Mapping[str, Any]) -> str:
    config = dict(resolved["scientific_config"])
    for key in MODEL_SPECIFIC_CONFIG_KEYS:
        config.pop(key, None)
    return content_hash(config, length=64)


def _offered_traffic_hash(packet_fates: pd.DataFrame) -> str:
    columns = ["packet_uid", "src_id", "fcnt", "birth_time", "offered_time"]
    missing = set(columns).difference(packet_fates.columns)
    if missing:
        raise ValueError(f"packet_fates.csv missing offered-traffic columns: {sorted(missing)}")
    offered = packet_fates.loc[:, columns].sort_values(columns[:3], kind="stable").copy()
    for column in ("birth_time", "offered_time"):
        values = pd.to_numeric(offered.pop(column), errors="raise")
        if values.isna().any():
            raise ValueError(f"packet_fates.csv contains missing {column} values")
        offered[f"{column}_ns"] = (values * OFFERED_TRAFFIC_TIME_SCALE).round().astype(
            "int64"
        )
    return content_hash(offered.to_dict(orient="records"), length=64)


def _unique_event_count(frame: pd.DataFrame, event: str, identity: str) -> int:
    if frame.empty or not {"event", identity}.issubset(frame.columns):
        return 0
    selected = frame.loc[frame["event"].astype(str) == event, identity].dropna()
    return int(selected.astype(str).nunique())


def _event_count(frame: pd.DataFrame, event: str) -> int:
    if frame.empty or "event" not in frame.columns:
        return 0
    return int((frame["event"].astype(str) == event).sum())


def reconcile_wang_command_lifecycle(
    commands: pd.DataFrame,
    *,
    observation_end_s: float | None = None,
) -> pd.DataFrame:
    """Classify each generated Wang command into one disjoint observation state.

    ``command_unconfirmed`` is an expiry-time observation, not a terminal
    command outcome: the same command can later be proven NOT_APPLIED or
    causally confirmed by an uplink.  Completed runs therefore reconcile final
    status from the complete event history for each physical command ID.  A
    command whose expiry is after ``observation_end_s`` is explicitly
    right-censored rather than being misclassified as NOT_APPLIED.
    """
    columns = [
        "command_id",
        "terminal_status",
        "applied_observed",
        "confirmed_observed",
        "not_applied_observed",
        "unconfirmed_at_expiry_observed",
        "censored_at_horizon",
    ]
    if commands.empty:
        return pd.DataFrame(columns=columns)
    required = {"command_id", "event"}
    missing = required.difference(commands.columns)
    if missing:
        raise ValueError(
            f"Wang command lifecycle missing columns: {sorted(missing)}"
        )

    identified = commands.dropna(subset=["command_id"]).copy()
    identified["command_id"] = identified["command_id"].astype(str)
    generated = set(
        identified.loc[
            identified["event"].astype(str) == "command_generated", "command_id"
        ]
    )
    observed = set(identified["command_id"])
    if observed - generated:
        raise ValueError(
            "Wang lifecycle contains events without command_generated: "
            f"{sorted(observed - generated)}"
        )

    rows: list[dict[str, Any]] = []
    for command_id in sorted(generated):
        events = set(
            identified.loc[
                identified["command_id"] == command_id, "event"
            ].astype(str)
        )
        applied = "command_applied" in events
        confirmed = "command_confirmed_by_uplink" in events
        not_applied = "command_not_applied" in events
        unconfirmed_at_expiry = "command_unconfirmed" in events
        if confirmed and not applied:
            raise ValueError(
                f"Wang command {command_id} is confirmed without application"
            )
        if not_applied and (applied or confirmed):
            raise ValueError(
                f"Wang command {command_id} has contradictory final evidence"
            )
        if confirmed:
            terminal_status = "APPLIED_CONFIRMED"
        elif not_applied:
            terminal_status = "NOT_APPLIED"
        elif applied:
            terminal_status = "APPLIED_UNCONFIRMED"
        elif observation_end_s is not None:
            command_rows = identified.loc[identified["command_id"] == command_id]
            expiry_values = pd.to_numeric(
                command_rows.get("expiry_s", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna().unique()
            if len(expiry_values) != 1 or float(expiry_values[0]) <= observation_end_s:
                raise ValueError(
                    f"Wang command {command_id} has no observable final status"
                )
            terminal_status = "CENSORED_AT_HORIZON"
        else:
            raise ValueError(
                f"Wang command {command_id} has no observable final status"
            )
        rows.append(
            {
                "command_id": command_id,
                "terminal_status": terminal_status,
                "applied_observed": applied,
                "confirmed_observed": confirmed,
                "not_applied_observed": not_applied,
                "unconfirmed_at_expiry_observed": unconfirmed_at_expiry,
                "censored_at_horizon": terminal_status == "CENSORED_AT_HORIZON",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _wang_control_tx_energy_j(airtime: pd.DataFrame) -> tuple[float, float]:
    """Return physical Wang command TX airtime/energy from recorded TX events."""
    if airtime.empty or "traffic_class" not in airtime.columns:
        return 0.0, 0.0
    control = airtime.loc[airtime["traffic_class"].astype(str) == "control_feedback"]
    total_airtime = 0.0
    total_energy = 0.0
    for row in control.itertuples(index=False):
        duration = float(row.total_airtime_s)
        ptx = int(row.ptx_dbm)
        hardware = SX1276 if str(row.entity_type) == "node" else SX1302
        total_airtime += duration
        total_energy += hardware.voltage * (hardware.tx_current[ptx] / 1000.0) * duration
    return total_airtime, total_energy


def _wang_metrics(
    model_id: str,
    reports: pd.DataFrame,
    decisions: pd.DataFrame,
    commands: pd.DataFrame,
    airtime: pd.DataFrame,
    observation_end_s: float | None = None,
) -> dict[str, Any]:
    if model_id != "wang_atp_sf7_a2only":
        return {
            "wang_metrics_applicability": "not_applicable",
            "wang_window_count": math.nan,
            "wang_eligible_window_count": math.nan,
            "wang_mean_snr_coverage_fraction": math.nan,
            "wang_sink_decision_count": math.nan,
            "wang_sink_decision_rate": math.nan,
            "wang_no_decision_count": math.nan,
            "wang_no_decision_reasons": None,
            "wang_snr_reports_generated": math.nan,
            "wang_snr_reports_received": math.nan,
            "wang_snr_reports_lost": math.nan,
            "wang_commands_generated": math.nan,
            "wang_commands_sent": math.nan,
            "wang_commands_applied": math.nan,
            "wang_commands_causally_confirmed": math.nan,
            "wang_commands_expired": math.nan,
            "wang_commands_not_applied": math.nan,
            "wang_commands_unconfirmed": math.nan,
            "wang_command_generated_events": math.nan,
            "wang_commands_generated_unique": math.nan,
            "wang_command_physical_hop_sent_events": math.nan,
            "wang_command_physical_hop_received_events": math.nan,
            "wang_command_applied_events": math.nan,
            "wang_command_confirmed_events": math.nan,
            "wang_commands_unconfirmed_at_expiry_unique": math.nan,
            "wang_commands_applied_confirmed_unique": math.nan,
            "wang_commands_applied_unconfirmed_unique": math.nan,
            "wang_commands_not_applied_unique": math.nan,
            "wang_commands_censored_at_horizon_unique": math.nan,
            "wang_commands_terminal_eligible_unique": math.nan,
            "wang_command_terminal_conservation_delta": math.nan,
            "wang_command_terminal_eligible_conservation_delta": math.nan,
            "wang_control_tx_airtime_s": math.nan,
            "wang_control_tx_energy_J": math.nan,
            "wang_reporting_energy_overhead_J": math.nan,
            "wang_reporting_energy_scope": "not_applicable",
        }

    windows = decisions.sort_values("window_index", kind="stable").drop_duplicates(
        "decision_id"
    )
    window_count = len(windows)
    generated = int((windows["status"].astype(str) == "decision_generated").sum())
    no_decision = windows.loc[windows["status"].astype(str) == "no_decision"]
    reasons = {
        str(reason): int(count)
        for reason, count in no_decision["reason"].astype(str).value_counts().items()
    }
    control_airtime, control_energy = _wang_control_tx_energy_j(airtime)
    terminal = reconcile_wang_command_lifecycle(
        commands, observation_end_s=observation_end_s
    )
    terminal_counts = terminal["terminal_status"].value_counts()
    generated_unique = _unique_event_count(
        commands, "command_generated", "command_id"
    )
    applied_confirmed = int(terminal_counts.get("APPLIED_CONFIRMED", 0))
    applied_unconfirmed = int(terminal_counts.get("APPLIED_UNCONFIRMED", 0))
    not_applied = int(terminal_counts.get("NOT_APPLIED", 0))
    censored = int(terminal_counts.get("CENSORED_AT_HORIZON", 0))
    terminal_eligible = generated_unique - censored
    return {
        "wang_metrics_applicability": "applicable",
        "wang_window_count": window_count,
        "wang_eligible_window_count": generated,
        "wang_mean_snr_coverage_fraction": float(
            pd.to_numeric(windows["coverage_fraction"], errors="raise").mean()
        ),
        "wang_sink_decision_count": generated,
        "wang_sink_decision_rate": generated / window_count if window_count else math.nan,
        "wang_no_decision_count": len(no_decision),
        "wang_no_decision_reasons": json.dumps(reasons, sort_keys=True),
        "wang_snr_reports_generated": _unique_event_count(
            reports, "generated", "report_id"
        ),
        "wang_snr_reports_received": _unique_event_count(
            reports, "received", "report_id"
        ),
        "wang_snr_reports_lost": _unique_event_count(reports, "lost", "report_id"),
        "wang_commands_generated": generated_unique,
        "wang_commands_sent": _unique_event_count(
            commands, "command_sent", "command_id"
        ),
        "wang_commands_applied": _unique_event_count(
            commands, "command_applied", "command_id"
        ),
        "wang_commands_causally_confirmed": _unique_event_count(
            commands, "command_confirmed_by_uplink", "command_id"
        ),
        "wang_commands_expired": _unique_event_count(
            commands, "command_expired", "command_id"
        ),
        "wang_commands_not_applied": _unique_event_count(
            commands, "command_not_applied", "command_id"
        ),
        "wang_commands_unconfirmed": _unique_event_count(
            commands, "command_unconfirmed", "command_id"
        ),
        "wang_command_generated_events": _event_count(
            commands, "command_generated"
        ),
        "wang_commands_generated_unique": generated_unique,
        "wang_command_physical_hop_sent_events": _event_count(
            commands, "command_sent"
        ),
        "wang_command_physical_hop_received_events": _event_count(
            commands, "command_received"
        ),
        "wang_command_applied_events": _event_count(commands, "command_applied"),
        "wang_command_confirmed_events": _event_count(
            commands, "command_confirmed_by_uplink"
        ),
        "wang_commands_unconfirmed_at_expiry_unique": _unique_event_count(
            commands, "command_unconfirmed", "command_id"
        ),
        "wang_commands_applied_confirmed_unique": applied_confirmed,
        "wang_commands_applied_unconfirmed_unique": applied_unconfirmed,
        "wang_commands_not_applied_unique": not_applied,
        "wang_commands_censored_at_horizon_unique": censored,
        "wang_commands_terminal_eligible_unique": terminal_eligible,
        "wang_command_terminal_conservation_delta": generated_unique
        - applied_confirmed
        - applied_unconfirmed
        - not_applied
        - censored,
        "wang_command_terminal_eligible_conservation_delta": terminal_eligible
        - applied_confirmed
        - applied_unconfirmed
        - not_applied,
        "wang_control_tx_airtime_s": control_airtime,
        "wang_control_tx_energy_J": control_energy,
        # SNR reports are piggybacked on DATA_UP/DATA_UP_FWD. Their marginal
        # counterfactual energy is not separately observable in R19.
        "wang_reporting_energy_overhead_J": math.nan,
        "wang_reporting_energy_scope": (
            "command TX energy exact from R19; piggyback report marginal energy "
            "not separately observable"
        ),
    }


def _run_metrics(
    *,
    model_id: str,
    run_dir: Path,
    expected_seed: int,
    expected_terrain: int,
    expected_routing_strategy: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    status = _read_json(run_dir / "status.json")
    if status.get("status") != "PASS":
        raise ValueError(f"{run_dir}: run status is not PASS")

    core_path = run_dir / "core_summary.csv"
    core = _single_row(_read_csv(core_path), core_path)
    resolved = _read_json(run_dir / "resolved_config.json")
    tables = run_dir / "tables"
    fates = _read_csv(tables / "packet_fates.csv")
    topology = _read_csv(tables / "topology_parent_by_run.csv")
    energy = _read_csv(tables / "energy_breakdown.csv")
    final_ptx = _read_csv(tables / "final_ptx_distribution.csv")
    feedback = _read_csv(tables / "apc_feedback_lifecycle.csv", allow_empty=True)
    history = _read_csv(tables / "ptx_history.csv", allow_empty=True)
    airtime = _read_csv(
        tables / "tx_airtime_by_entity_mtype_ptx_channel.csv", allow_empty=True
    )
    wang_reports = _read_optional_csv(tables / "wang_snr_report_lifecycle.csv")
    wang_decisions = _read_optional_csv(tables / "wang_sink_decisions.csv")
    wang_commands = _read_optional_csv(tables / "wang_power_command_lifecycle.csv")

    checks: dict[str, bool] = {
        "status_pass": True,
        "identity_match": (
            str(core["tx_power_model_id"]) == model_id
            and int(core["seed"]) == expected_seed
            and int(core["terrain_delta_h"]) == expected_terrain
        ),
        "packet_conservation": int(core["packet_fate_conservation_delta"]) == 0,
        "collision_conservation": int(core["collision_outcome_conservation_delta"]) == 0,
        "corrected_energy_once": (
            core["energy_accounting_mode"] == "corrected"
            and core["energy_accounting_model_version"]
            == CORRECTED_ENERGY_MODEL_VERSION
            and set(energy["energy_accounting_mode"].astype(str)) == {"corrected"}
            and set(energy["energy_accounting_model_version"].astype(str))
            == {CORRECTED_ENERGY_MODEL_VERSION}
        ),
        "topology_evidence_available": (
            not topology.empty
            and {"entity_id", "entity_type", "parent_id"}.issubset(topology.columns)
        ),
    }
    routing_strategy = str(
        resolved["scientific_config"].get("routing_strategy", "terrain_specific")
    )
    if expected_routing_strategy is not None:
        checks["routing_strategy_match"] = (
            routing_strategy == expected_routing_strategy
        )

    offered = int(core["packet_fate_offered_count"])
    delivered = int(core["packet_fate_delivered_count"])
    terminal = int(core["packet_fate_terminal_drop_count"])
    censored = int(core["packet_fate_censored_count"])
    checks["packet_count_conservation"] = offered == delivered + terminal + censored

    energy_total = float(energy["total_energy_J"].sum())
    checks["energy_conservation"] = _close(core["total_network_energy_J"], energy_total)
    expected_epp = energy_total / delivered if delivered else 0.0
    checks["core_energy_per_delivered_consistent"] = _close(
        core["energy_per_delivered_packet_J"], expected_epp
    )

    delivered_rows = fates.loc[fates["final_status"].astype(str) == "DELIVERED"]
    delay_ms = pd.to_numeric(delivered_rows["e2e_delay_ms"], errors="raise")
    recomputed_delay = float(delay_ms.mean()) if delivered else math.nan
    checks["delivered_count_matches_fates"] = len(delivered_rows) == delivered
    checks["delivered_delay_matches_core"] = (
        delivered == 0
        or _close(core["avg_e2e_delay_ms"], recomputed_delay, tolerance=1e-3)
    )
    recomputed_pdr = 100.0 * delivered / offered if offered else math.nan
    checks["pdr_matches_counts"] = offered > 0 and _close(
        core["delivery_by_horizon_percent"], recomputed_pdr
    )

    if model_id == "islam_dbpc_itm6":
        active = final_ptx.loc[final_ptx["entity_type"].isin(["node", "relay"])]
        checks.update(
            {
                "dbpc_calibration_hash": (
                    str(core["tx_power_calibration_manifest_sha256"])
                    == DBPC_CALIBRATION_MANIFEST_SHA256
                    and set(active["calibration_manifest_sha256"].astype(str))
                    == {DBPC_CALIBRATION_MANIFEST_SHA256}
                ),
                "dbpc_margin_6_db": (
                    int(core["tx_power_calibration_margin_db"]) == 6
                    and set(active["calibration_margin_db"].astype(int)) == {6}
                ),
                "dbpc_fixed_calibrated_ptx": (
                    not active.empty
                    and (
                        pd.to_numeric(active["initial_ptx"], errors="raise")
                        == pd.to_numeric(active["final_ptx"], errors="raise")
                    ).all()
                ),
                "dbpc_feedback_zero": (
                    int(core["apc_total_tx"]) == 0
                    and int(core["apc_total_rx"]) == 0
                    and int(core["apc_total_drop"]) == 0
                    and int(core["ptx_update_count"]) == 0
                    and feedback.empty
                    and history.empty
                    and not _bool(core["tx_power_feedback_enabled"])
                ),
            }
        )
    elif model_id == "ff":
        checks["ff_feedback_zero"] = (
            int(core["apc_total_tx"]) == 0
            and int(core["apc_total_rx"]) == 0
            and int(core["apc_total_drop"]) == 0
            and int(core["ptx_update_count"]) == 0
            and feedback.empty
            and history.empty
        )
    elif model_id == "aa":
        checks["aa_adaptive_configuration"] = (
            core["node_ptx_mode"] == "adaptive"
            and core["relay_ptx_mode"] == "adaptive"
        )
    elif model_id == "islam_dbpc_802":
        active = final_ptx.loc[final_ptx["entity_type"].isin(["node", "relay"])]
        checks.update(
            {
                "dbpc_802_estimator_identity": (
                    not active.empty
                    and set(active["estimator_id"].astype(str))
                    == {"ieee80211ah_outdoor_eq15_margin0_v1"}
                ),
                "dbpc_802_margin_0_db": (
                    not active.empty
                    and set(
                        pd.to_numeric(active["policy_margin_db"], errors="raise")
                    )
                    == {0.0}
                ),
                "dbpc_802_fixed_ptx": (
                    not active.empty
                    and (
                        pd.to_numeric(active["initial_ptx"], errors="raise")
                        == pd.to_numeric(active["final_ptx"], errors="raise")
                    ).all()
                    and history.empty
                ),
                "dbpc_802_feedback_zero": (
                    int(core["apc_total_tx"]) == 0
                    and int(core["apc_total_rx"]) == 0
                    and int(core["apc_total_drop"]) == 0
                    and int(core["ptx_update_count"]) == 0
                    and feedback.empty
                    and not _bool(core["tx_power_feedback_enabled"])
                ),
            }
        )
    else:
        decision_count = (
            wang_decisions["decision_id"].nunique()
            if "decision_id" in wang_decisions.columns
            else 0
        )
        checks.update(
            {
                "wang_r20_r22_serialized": (
                    not wang_reports.empty
                    and not wang_decisions.empty
                    and {"event", "command_id"}.issubset(wang_commands.columns)
                ),
                "wang_decision_windows_match_summary": (
                    decision_count == int(core["wang_closed_window_count"])
                ),
                "wang_apc_feedback_zero": (
                    int(core["apc_total_tx"]) == 0
                    and int(core["apc_total_rx"]) == 0
                    and int(core["apc_total_drop"]) == 0
                    and feedback.empty
                ),
                "wang_no_data_ack": not _bool(core["frame_default_ack_req"]),
                "wang_collision_gate_enabled": _bool(
                    resolved["resolved_simulator_config"][
                        "node_rx_collision_enabled"
                    ]
                ),
            }
        )

    failed = sorted(key for key, passed in checks.items() if not passed)
    display_label = model_label(
        model_id, str(resolved["scientific_config"]["experiment_profile"])
    )
    validation = {
        "model_id": model_id,
        "model_label": display_label,
        "terrain_delta_h": expected_terrain,
        "seed": expected_seed,
        "routing_strategy": routing_strategy,
        "run_id": str(core["run_id"]),
        "run_dir": str(run_dir),
        **checks,
        "run_validation_pass": not failed,
        "failed_checks": ";".join(failed),
        "scientific_input_signature": _scientific_signature(resolved),
        "topology_hash": str(core["topology_hash"]),
        "routing_hash": str(core["selected_parent_routing_hash"]),
        "offered_traffic_hash": _offered_traffic_hash(fates),
        "offered_traffic_hash_version": OFFERED_TRAFFIC_HASH_VERSION,
    }
    metrics = {
        "model_id": model_id,
        "model_label": display_label,
        "terrain_delta_h": expected_terrain,
        "seed": expected_seed,
        "routing_strategy": routing_strategy,
        "run_id": str(core["run_id"]),
        "offered_packets": offered,
        "delivered_packets": delivered,
        "zero_delivery": delivered == 0,
        "pdr_percent": recomputed_pdr,
        "avg_e2e_delay_ms": recomputed_delay,
        "total_network_energy_J": energy_total,
        "energy_per_delivered_packet_J": (
            energy_total / delivered if delivered else math.nan
        ),
        "throughput_bps": (
            delivered
            * int(core["node_app_payload_size_bytes"])
            * 8.0
            / float(core["sim_time"])
        ),
        "control_tx_airtime_s": _wang_control_tx_energy_j(airtime)[0],
        "control_tx_energy_J": _wang_control_tx_energy_j(airtime)[1],
        "apc_feedback_tx": int(core["apc_total_tx"]),
        "ptx_update_count": int(core["ptx_update_count"]),
        "served_node_ratio_95_percent": float(core["served_node_ratio_95_percent"]),
        "avg_hop_count": float(core.get("avg_hop_count", math.nan)),
        "single_hop_original_tx": float(
            core.get("single_hop_original_tx", math.nan)
        ),
        "multi_hop_original_tx": float(
            core.get("multi_hop_original_tx", math.nan)
        ),
        "configured_relay_count": int(core["relay_num"]),
        "relay_waiting_mean_ms": float(core["avg_relay_waiting_time_wq_ms"]),
        "relay_waiting_p95_ms": float(core["relay_fifo_wait_p95_ms"]),
        "relay_max_worker_occupied_fraction": float(
            core["relay_max_worker_occupied_fraction"]
        ),
        "packet_fate_terminal_drop_count": terminal,
        "packet_fate_censored_count": censored,
        "packet_fate_conservation_delta": int(
            core["packet_fate_conservation_delta"]
        ),
        "collision_decision_count": int(core["collision_decision_count"]),
        "collision_drop_decision_count": int(core["collision_drop_decision_count"]),
        "collision_outcome_conservation_delta": int(
            core["collision_outcome_conservation_delta"]
        ),
        **_wang_metrics(
            model_id,
            wang_reports,
            wang_decisions,
            wang_commands,
            airtime,
            observation_end_s=float(core["sim_time"]),
        ),
        "source_core_summary_sha256": file_sha256(core_path),
    }
    return metrics, validation


def _validate_matched_cells(
    validation: pd.DataFrame,
    expected_models: Sequence[str] | None = None,
) -> pd.DataFrame:
    result = validation.copy()
    models = tuple(expected_models or sorted(result["model_id"].astype(str).unique()))
    match_columns = (
        "scientific_input_signature",
        "topology_hash",
        "routing_hash",
        "offered_traffic_hash",
    )
    result["matched_model_count"] = 0
    for column in match_columns:
        result[f"matched_{column}"] = False

    groups = result.groupby(["terrain_delta_h", "seed"], sort=True, dropna=False)
    for _, indexes in groups.groups.items():
        sub = result.loc[indexes]
        model_count = sub["model_id"].nunique()
        result.loc[indexes, "matched_model_count"] = model_count
        for column in match_columns:
            result.loc[indexes, f"matched_{column}"] = (
                model_count == len(models)
                and sub[column].nunique(dropna=False) == 1
            )

    matched_columns = [f"matched_{column}" for column in match_columns]
    result["matched_inputs_pass"] = (
        result["matched_model_count"] == len(models)
    ) & result[matched_columns].all(axis=1)
    result["validation_pass"] = result["run_validation_pass"] & result[
        "matched_inputs_pass"
    ]
    diagnostic_labels = {
        "matched_scientific_input_signature": "cross_model_scientific_input_mismatch",
        "matched_topology_hash": "cross_model_topology_mismatch",
        "matched_routing_hash": "cross_model_routing_mismatch",
        "matched_offered_traffic_hash": "cross_model_exogenous_workload_mismatch",
    }

    def diagnostics(row: pd.Series) -> str:
        failures = [value for value in str(row["failed_checks"]).split(";") if value]
        if row["matched_model_count"] != len(models):
            failures.append("cross_model_model_count_mismatch")
        failures.extend(
            label for column, label in diagnostic_labels.items() if not bool(row[column])
        )
        return ";".join(failures)

    result["failed_checks"] = result.apply(diagnostics, axis=1)
    return result


def build_paired_differences(per_run: pd.DataFrame) -> pd.DataFrame:
    """Build seed-paired model differences; positive means left minus right."""
    index = ["terrain_delta_h", "seed"]
    rows: list[dict[str, Any]] = []
    available_models = set(per_run["model_id"].astype(str))
    for contrast, left, right in PAIRWISE_CONTRASTS:
        if left not in available_models or right not in available_models:
            continue
        left_frame = per_run.loc[per_run["model_id"] == left].set_index(index)
        right_frame = per_run.loc[per_run["model_id"] == right].set_index(index)
        if not left_frame.index.equals(right_frame.index):
            raise ValueError(f"Unmatched terrain/seed cells for {contrast}")
        for cell in left_frame.index:
            terrain, seed = cell
            row: dict[str, Any] = {
                "contrast": contrast,
                "left_model_id": left,
                "right_model_id": right,
                "terrain_delta_h": terrain,
                "seed": seed,
            }
            for metric in METRICS:
                row[metric] = left_frame.at[cell, metric] - right_frame.at[cell, metric]
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_metric(
    statistics: pd.DataFrame,
    metric: str,
    ylabel: str,
    output: Path,
    models: Sequence[str] | None = None,
) -> list[Path]:
    selected = statistics.loc[statistics["metric"] == metric].copy()
    selected_models = tuple(models or sorted(selected["model_id"].astype(str).unique()))
    expected = len(selected_models) * len(APC_COMPARE_MODEL_TERRAIN_SWEEP_TERRAINS)
    if len(selected) != expected or set(selected["n"].astype(int)) != {
        len(APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS)
    }:
        raise ValueError(f"Cannot plot incomplete statistics for {metric}")

    output.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output.with_suffix(".png"), output.with_suffix(".svg")]
    with plt.rc_context(FIGURE_STYLE):
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        for model_id in (item for item in PLOT_MODEL_ORDER if item in selected_models):
            frame = selected.loc[selected["model_id"] == model_id].sort_values(
                "terrain_delta_h"
            )
            x = frame["terrain_delta_h"].to_numpy(dtype=float)
            mean = frame["mean"].to_numpy(dtype=float)
            lower = frame["ci_lower"].to_numpy(dtype=float)
            upper = frame["ci_upper"].to_numpy(dtype=float)
            style = MODEL_STYLES[model_id]
            axis.plot(
                x,
                mean,
                label=MODEL_LABELS[model_id],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=7,
            )
            axis.fill_between(x, lower, upper, color=style["color"], alpha=0.05)
        axis.set_xlabel(r"Terrain irregularity, $\Delta h$ (m)")
        axis.set_ylabel(ylabel)
        axis.grid(True)
        axis.legend(ncols=2)
        figure.tight_layout()
        figure.savefig(outputs[0], dpi=300, bbox_inches="tight", facecolor="white")
        figure.savefig(outputs[1], bbox_inches="tight", facecolor="white")
        plt.close(figure)
    return outputs


def _assert_complete_grid(
    per_run: pd.DataFrame,
    models: Sequence[str],
    terrains: Sequence[int] = APC_COMPARE_MODEL_TERRAIN_SWEEP_TERRAINS,
    seeds: Sequence[int] = APC_COMPARE_MODEL_TERRAIN_SWEEP_SEEDS,
) -> None:
    cells = set(
        zip(
            per_run["model_id"],
            per_run["terrain_delta_h"].astype(int),
            per_run["seed"].astype(int),
        )
    )
    expected = {
        (model, terrain, seed)
        for model in models
        for terrain in terrains
        for seed in seeds
    }
    if len(per_run) != len(expected) or cells != expected:
        raise ValueError(
            f"Postprocessing requires exactly {len(expected)} unique "
            "model/terrain/seed runs"
        )


def _assert_reporting_strategy(
    per_run: pd.DataFrame, expected_routing_strategy: str | None
) -> None:
    if expected_routing_strategy is None:
        return
    observed = set(per_run["routing_strategy"].astype(str))
    if observed != {expected_routing_strategy}:
        raise ValueError(
            "Postprocessing routing-strategy scope mismatch: "
            f"expected {expected_routing_strategy!r}, got {sorted(observed)}"
        )


def build_comparison_artifacts(config: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Validate the complete configured grid, then build CSVs and figures."""
    definition = load_sweep_definition(config)
    if definition.experiment_profile == REVIEWER_R1_5_PROFILE:
        raise ValueError(
            "Reviewer R1.5 routing-sensitivity reporting is isolated from "
            "MAIN_W4 figures and requires its dedicated paired-routing reporter"
        )
    specs = build_model_specs(config)
    parent = ExperimentArtifactPaths(definition.experiment_dir).ensure()
    parent_status = _read_json(parent.status)
    if parent_status.get("status") != "PASS" or not parent_status.get(
        "seed_grid_complete", False
    ):
        raise ValueError("Parent experiment is incomplete; figures were not generated")

    metric_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for model_id, spec in specs.items():
        child = ExperimentArtifactPaths.from_spec(spec)
        status = _read_json(child.status)
        if status.get("status") != "PASS" or not status.get("seed_grid_complete", False):
            raise ValueError(f"Child experiment {model_id} is incomplete")
        source_records.append(
            {
                "model_id": model_id,
                "experiment_manifest": str(child.manifest),
                "experiment_manifest_sha256": file_sha256(child.manifest),
                "experiment_status": str(child.status),
                "experiment_status_sha256": file_sha256(child.status),
            }
        )
        run_dirs = sorted(path for path in child.runs.iterdir() if path.is_dir())
        for run_dir in run_dirs:
            resolved = _read_json(run_dir / "resolved_config.json")
            config_data = resolved["scientific_config"]
            metrics, validation = _run_metrics(
                model_id=model_id,
                run_dir=run_dir,
                expected_seed=int(config_data["seed"]),
                expected_terrain=int(config_data["terrain_delta_h"]),
                expected_routing_strategy=definition.routing_strategy,
            )
            metric_rows.append(metrics)
            validation_rows.append(validation)

    per_run = pd.DataFrame(metric_rows).sort_values(
        ["model_id", "terrain_delta_h", "seed"], kind="stable"
    )
    _assert_complete_grid(
        per_run,
        definition.models,
        definition.factors["terrain_delta_h"],
        definition.seeds,
    )
    _assert_reporting_strategy(per_run, definition.routing_strategy)
    validation = _validate_matched_cells(
        pd.DataFrame(validation_rows), definition.models
    ).sort_values(
        ["model_id", "terrain_delta_h", "seed"], kind="stable"
    )
    atomic_write_csv(parent.tables / "validation_summary.csv", validation)
    if not validation["validation_pass"].all():
        failed = validation.loc[
            ~validation["validation_pass"],
            ["model_id", "terrain_delta_h", "seed", "failed_checks"],
        ]
        raise ValueError(
            "Comparison validation failed; inspect tables/validation_summary.csv: "
            f"{failed.head(10).to_dict(orient='records')}"
        )

    atomic_write_csv(parent.tables / "per_run_metrics.csv", per_run)
    statistics = aggregate_paper_grade(
        per_run,
        group_columns=["model_id", "model_label", "terrain_delta_h"],
        metric_columns=METRICS,
        confidence_level=definition.confidence_level,
    )
    atomic_write_csv(parent.tables / "model_statistics.csv", statistics)

    paired = build_paired_differences(per_run)
    atomic_write_csv(parent.tables / "paired_differences_per_run.csv", paired)
    paired_statistics = aggregate_paper_grade(
        paired,
        group_columns=["contrast", "left_model_id", "right_model_id", "terrain_delta_h"],
        metric_columns=METRICS,
        confidence_level=definition.confidence_level,
    )
    atomic_write_csv(parent.tables / "paired_statistics.csv", paired_statistics)

    figure_paths: list[Path] = []
    if definition.experiment_profile != APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE:
        for metric, ylabel, filename in FIGURES:
            figure_paths.extend(
                _plot_metric(
                    statistics,
                    metric,
                    ylabel,
                    parent.figures / filename,
                    definition.models,
                )
            )

    output_paths = [
        parent.tables / "validation_summary.csv",
        parent.tables / "per_run_metrics.csv",
        parent.tables / "model_statistics.csv",
        parent.tables / "paired_differences_per_run.csv",
        parent.tables / "paired_statistics.csv",
        *figure_paths,
    ]
    manifest_path = definition.experiment_dir / "analysis_manifest.json"
    manifest = {
        "manifest_version": (
            "apc_compare_model_terrain_sweep_analysis_v2"
            if definition.experiment_profile
            in {
                APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
                APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE,
                MAIN_W4_PROFILE,
            }
            else "apc_compare_model_terrain_sweep_analysis_v1"
        ),
        "created_at": utc_now(),
        "experiment_id": definition.experiment_id,
        "validated_run_count": len(per_run),
        "validated_cell_count": len(per_run),
        "validation_pass": True,
        "confidence_level": definition.confidence_level,
        "ci_method": "paired/unpaired Student-t by output table, df=n-1",
        "metric_definitions": {
            "pdr_percent": "100 * delivered packets / offered packets per run",
            "energy_per_delivered_packet_J": (
                "corrected total network energy / delivered packets per run; "
                "NaN for zero-delivery runs"
            ),
            "avg_e2e_delay_ms": "mean delay over delivered packets only",
            "total_network_energy_J": "sum of corrected entity energy per run",
        },
        "delay_interpretation": (
            "Figure 10 is conditional on delivery and must be interpreted together "
            "with Figure 8 PDR."
        ),
        "dbpc_calibration_manifest_sha256": DBPC_CALIBRATION_MANIFEST_SHA256,
        "model_ids": list(definition.models),
        "analysis_scope": (
            "single_cell_pilot"
            if definition.experiment_profile
            == APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE
            else "complete_terrain_sweep"
        ),
        "wang_metric_applicability": (
            "R20-R22 and derived Wang fields are applicable only to "
            "wang_atp_sf7_a2only; non-Wang rows use explicit NaN/not_applicable"
        ),
        "wang_control_energy_scope": (
            "Exact command-frame TX energy derived from R19 hardware current and "
            "airtime; marginal energy of piggyback SNR bytes is not separable"
        ),
        "wang_command_metric_definitions": {
            "legacy_raw_fields": (
                "wang_commands_generated/sent/applied/causally_confirmed/expired/"
                "not_applied/unconfirmed retain their prior unique-ID event semantics"
            ),
            "physical_hop_fields": (
                "command_sent and command_received event-row counts; one command may "
                "contribute multiple physical hops"
            ),
            "unconfirmed_at_expiry": (
                "raw expiry observation; not a mutually exclusive final status"
            ),
            "terminal_statuses": (
                "one of APPLIED_CONFIRMED, APPLIED_UNCONFIRMED, NOT_APPLIED per "
                "generated command ID after complete-lifecycle reconciliation"
            ),
        },
        "offered_traffic_hash": {
            "version": OFFERED_TRAFFIC_HASH_VERSION,
            "records": "original DATA_UP packet_uid, src_id, fcnt, birth and offer time",
            "time_canonicalization": "nearest integer nanosecond",
        },
        "sources": source_records,
        "outputs": [
            {"path": str(path), "sha256": file_sha256(path)} for path in output_paths
        ],
    }
    atomic_write_json(manifest_path, manifest)
    return {
        "status": "PASS",
        "validated_run_count": len(per_run),
        "tables": [str(path) for path in output_paths if path.suffix == ".csv"],
        "figures": [str(path) for path in figure_paths],
        "manifest": str(manifest_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    print(json.dumps(build_comparison_artifacts(args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FIGURES",
    "METRICS",
    "OFFERED_TRAFFIC_HASH_VERSION",
    "build_comparison_artifacts",
    "build_paired_differences",
    "reconcile_wang_command_lifecycle",
]
