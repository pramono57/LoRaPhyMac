"""Validated aggregation for the APC feedback robustness cohort.

The reporter refuses to publish statistics unless the exact approved 30-run
matrix is present and every scientific run bundle passes manifest checks.
"""

from __future__ import annotations

import hashlib
import argparse
import csv
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from scipy.stats import t as student_t

from loraphymac.experiments.apc_feedback_robustness_sweep.contract import (
    APC_FEEDBACK_ROBUSTNESS_PROFILE,
    APC_FEEDBACK_ROBUSTNESS_RUN_TABLES,
    APC_FEEDBACK_ROBUSTNESS_SCENARIOS,
    APC_FEEDBACK_ROBUSTNESS_SEEDS,
    STALE_TPI_COLUMNS,
    STALE_TPI_METADATA_COLUMNS,
    STALE_TPI_PERSISTED_COLUMNS,
    STALE_TPI_TABLE_KEY,
    TABLE_SCHEMAS,
)
from loraphymac.experiments.common.multiseed_pipeline import (
    _read_verified_scientific_tables,
    atomic_write_csv,
    atomic_write_json,
    content_hash,
)
from loraphymac.experiments.common.outputs import ExperimentArtifactPaths
from loraphymac.experiments.common.scientific_output_manager import (
    ScientificOutputManager,
)


CORE_METRICS = (
    "pdr_percent",
    "packet_fate_offered_count",
    "packet_fate_delivered_count",
    "packet_fate_terminal_drop_count",
    "packet_fate_censored_count",
    "total_network_energy_J",
    "energy_per_delivered_packet_J",
    "avg_e2e_delay_ms",
)
GROUP_KEYS = ("node_period", "feedback_additional_loss_probability")
EXPECTED_KEYS = {
    (period, loss, seed)
    for period, loss in APC_FEEDBACK_ROBUSTNESS_SCENARIOS
    for seed in APC_FEEDBACK_ROBUSTNESS_SEEDS
}
PAIRED_METRICS = (*CORE_METRICS, "stale_tpi_transmission_fraction")


def _validate_stale_tpi_columns(columns: Sequence[str], run_id: str) -> None:
    """Validate the exact persisted schema, including pipeline provenance."""
    names = list(columns)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate {STALE_TPI_TABLE_KEY} columns in {run_id}: {duplicates}")
    missing_scientific = sorted(set(STALE_TPI_COLUMNS) - set(names))
    missing_metadata = sorted(set(STALE_TPI_METADATA_COLUMNS) - set(names))
    unexpected = sorted(set(names) - set(STALE_TPI_PERSISTED_COLUMNS))
    if missing_scientific or missing_metadata or unexpected:
        raise ValueError(
            f"{STALE_TPI_TABLE_KEY} schema mismatch in {run_id}: "
            f"missing_scientific={missing_scientific}; "
            f"missing_metadata={missing_metadata}; unexpected={unexpected}"
        )


def _validate_stale_tpi_table(frame: pd.DataFrame, run_id: str) -> None:
    """Fail closed on persisted schema, provenance, and counter integrity."""
    _validate_stale_tpi_columns(frame.columns, run_id)
    if frame.columns.has_duplicates:
        raise ValueError(f"duplicate {STALE_TPI_TABLE_KEY} columns in {run_id}")
    if frame.empty:
        raise ValueError(f"{STALE_TPI_TABLE_KEY} has no transmitter rows in {run_id}")
    for column in (
        "experiment_id", "model_id", "revision", "energy_accounting_model_version",
        "run_id", "transmitter_id", "transmitter_role", "mtype",
    ):
        values = frame[column]
        if values.isna().any() or not values.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ValueError(f"invalid text column {column} in {run_id}")
    for column in ("seed", "node_period"):
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any() or not values.map(math.isfinite).all() or (values % 1 != 0).any():
            raise ValueError(f"invalid integer metadata {column} in {run_id}")
    loss = pd.to_numeric(frame["feedback_additional_loss_probability"], errors="raise")
    if loss.isna().any() or not loss.map(math.isfinite).all() or not loss.between(0, 1).all():
        raise ValueError(f"invalid probability metadata in {run_id}")
    scope = frame["control_scope_id"]
    if not scope.dropna().map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError(f"invalid control_scope_id in {run_id}")
    if not frame["run_id"].astype(str).eq(str(run_id)).all():
        raise ValueError(f"stale-TPI run_id mismatch in {run_id}")
    for column in ("experiment_id", "model_id", "revision", "energy_accounting_model_version"):
        if frame[column].nunique(dropna=False) != 1:
            raise ValueError(f"inconsistent provenance metadata {column} in {run_id}")
    key_columns = [
        "run_id", "transmitter_id", "transmitter_role", "mtype", "control_scope_id"
    ]
    if frame.duplicated(key_columns, keep=False).any():
        raise ValueError(f"duplicate stale-TPI grouping rows in {run_id}")
    count_columns = (
        "total_data_physical_tx_count", "pre_decision_tx_count",
        "post_decision_data_tx_count", "eligible_data_tx_count",
        "stale_tpi_tx_count", "current_tpi_tx_count", "unclassified_tx_count",
        "version_invariant_violation_count",
    )
    for column in count_columns:
        values = pd.to_numeric(frame[column], errors="raise")
        if (
            values.isna().any()
            or not values.map(math.isfinite).all()
            or (values < 0).any()
            or (values % 1 != 0).any()
        ):
            raise ValueError(f"invalid stale-TPI counter {column} in {run_id}")
    if not (
        frame["total_data_physical_tx_count"]
        == frame["pre_decision_tx_count"] + frame["post_decision_data_tx_count"]
    ).all():
        raise ValueError(f"stale-TPI total/pre/post conservation failed in {run_id}")
    if not (
        frame["post_decision_data_tx_count"]
        == frame["eligible_data_tx_count"] + frame["unclassified_tx_count"]
    ).all():
        raise ValueError(f"stale-TPI post-decision conservation failed in {run_id}")
    if not (
        frame["eligible_data_tx_count"]
        == frame["stale_tpi_tx_count"] + frame["current_tpi_tx_count"]
    ).all():
        raise ValueError(f"stale-TPI eligible conservation failed in {run_id}")
    if (frame["version_invariant_violation_count"] > frame["unclassified_tx_count"]).any():
        raise ValueError(f"stale-TPI invariant violations are not unclassified in {run_id}")
    zero = frame["eligible_data_tx_count"].eq(0)
    if frame.loc[zero, "stale_tpi_transmission_fraction"].notna().any():
        raise ValueError(f"stale-TPI zero-denominator fraction must be null in {run_id}")
    nonzero = ~zero
    expected = (
        frame.loc[nonzero, "stale_tpi_tx_count"]
        / frame.loc[nonzero, "eligible_data_tx_count"]
    )
    actual = pd.to_numeric(
        frame.loc[nonzero, "stale_tpi_transmission_fraction"], errors="raise"
    )
    if not (actual - expected).abs().le(1e-15).all():
        raise ValueError(f"stale-TPI fraction disagrees with counts in {run_id}")
    allowed = {"node": "DATA_UP", "relay": "DATA_UP_FWD"}
    if not frame.apply(
        lambda row: allowed.get(str(row["transmitter_role"])) == str(row["mtype"]),
        axis=1,
    ).all():
        raise ValueError(f"stale-TPI transmitter role/mtype mismatch in {run_id}")
    fraction = pd.to_numeric(frame["stale_tpi_transmission_fraction"], errors="raise")
    if not fraction.dropna().map(math.isfinite).all() or not fraction.dropna().between(0, 1).all():
        raise ValueError(f"invalid stale-TPI fraction in {run_id}")


def _validate_stale_tpi_csv_header(path: Path, run_id: str) -> None:
    """Inspect the raw CSV header so pandas cannot hide duplicate names."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream), [])
    _validate_stale_tpi_columns(header, run_id)


@dataclass(frozen=True)
class RunBundle:
    run_id: str
    config: Mapping[str, Any]
    core: pd.DataFrame
    tables: Mapping[str, pd.DataFrame]
    source_fingerprint: str
    config_fingerprint: str
    manifest_sha256: str
    data_class: str = "SIMULATION"


@dataclass(frozen=True)
class CohortCheck:
    expected_count: int
    actual_count: int
    complete: bool
    missing: tuple[tuple[int, float, int], ...]
    duplicates: tuple[tuple[int, float, int], ...]
    unexpected: tuple[tuple[int, float, int], ...]
    errors: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_run_bundle(run_dir: str | Path) -> RunBundle:
    """Read one immutable scientific run after verifying its full file manifest."""
    path = Path(run_dir).resolve()
    manifest_path = path / "scientific_output_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED":
        raise ValueError(f"run manifest is not COMPLETED: {path.name}")
    verified = _read_verified_scientific_tables(path)
    if not set(APC_FEEDBACK_ROBUSTNESS_RUN_TABLES).issubset(verified):
        raise ValueError(f"required robustness tables missing: {path.name}")
    _, stale_path = verified[STALE_TPI_TABLE_KEY]
    _validate_stale_tpi_csv_header(stale_path, path.name)
    manager = ScientificOutputManager(path.parents[2])
    tables: dict[str, pd.DataFrame] = {}
    for key in APC_FEEDBACK_ROBUSTNESS_RUN_TABLES:
        frame = manager.read_table(path, key)
        required_columns = set(TABLE_SCHEMAS[key])
        if not required_columns.issubset(frame.columns):
            raise ValueError(f"{key} schema incomplete in {path.name}")
        tables[key] = frame
    stale = tables[STALE_TPI_TABLE_KEY]
    _validate_stale_tpi_table(stale, path.name)
    # Multi-seed scientific persistence exposes the one-row core summary as a
    # compatibility artifact, rather than a member of the scientific tables map.
    core_path = path / "core_summary.csv"
    if not core_path.is_file():
        raise ValueError(f"core summary artifact missing: {path.name}")
    core = pd.read_csv(core_path)
    if len(core) != 1 or not set(CORE_METRICS).issubset(core.columns):
        raise ValueError(f"core summary schema/row count invalid in {path.name}")
    core_row = core.iloc[0]
    for column in STALE_TPI_METADATA_COLUMNS:
        if column not in core.columns:
            raise ValueError(f"core summary is missing provenance column {column} in {path.name}")
        if column in {"seed", "node_period", "feedback_additional_loss_probability"}:
            stale_values = pd.to_numeric(stale[column], errors="raise")
            core_value = float(core_row[column])
            if not stale_values.eq(core_value).all():
                raise ValueError(f"stale-TPI provenance mismatch for {column} in {path.name}")
        elif not stale[column].astype(str).eq(str(core_row[column])).all():
            raise ValueError(f"stale-TPI provenance mismatch for {column} in {path.name}")
    resolved_record = manifest.get("artifacts", {}).get("resolved_config.json")
    if not resolved_record:
        raise ValueError(f"resolved configuration artifact missing: {path.name}")
    resolved_payload = json.loads((path / resolved_record["path"]).read_text(encoding="utf-8"))
    config = resolved_payload.get("resolved_simulator_config", {})
    persisted_identity = {
        "experiment_id": manifest.get("experiment_id"),
        "model_id": "aa",
        "seed": config.get("seed"),
        "node_period": config.get("node_period"),
        "feedback_additional_loss_probability": config.get(
            "feedback_additional_loss_probability"
        ),
    }
    for column, expected in persisted_identity.items():
        actual = pd.to_numeric(stale[column], errors="raise") if column in {
            "seed", "node_period", "feedback_additional_loss_probability"
        } else stale[column].astype(str)
        if not actual.eq(expected).all():
            raise ValueError(f"stale-TPI provenance mismatch for {column} in {path.name}")
    if config.get("experiment_profile") != APC_FEEDBACK_ROBUSTNESS_PROFILE:
        raise ValueError(f"wrong experiment profile in {path.name}")
    if (
        config.get("output_profile") != "summary"
        or config.get("persistence_profile") != "scientific"
        or config.get("gzip_scientific_tables") is not False
    ):
        raise ValueError(f"output profile/persistence mismatch in {path.name}")
    if manifest.get("run_id") != path.name:
        raise ValueError(f"manifest run_id mismatch in {path.name}")
    if resolved_payload.get("config_hash") != content_hash(config):
        raise ValueError(f"resolved configuration hash mismatch in {path.name}")
    source = manifest.get("source_provenance", {}).get("source_fingerprint")
    if not source:
        raise ValueError(f"source fingerprint missing in {path.name}")
    config_fp = manifest.get("config_fingerprint_sha256")
    if not config_fp:
        raise ValueError(f"configuration fingerprint missing in {path.name}")
    manager = ScientificOutputManager(path.parents[1])
    completion = manager.validate_completed(
        experiment_id=str(manifest["experiment_id"]),
        run_id=path.name,
        config=config,
        source_provenance=manifest["source_provenance"],
        collection_profile=manifest.get("collection_profile"),
        persistence_profile=manifest.get("persistence_profile"),
        additional_table_keys=manifest.get("additional_table_keys", ()),
        table_filename_overrides=manifest.get("table_filename_overrides"),
    )
    if not completion.safe_to_skip:
        raise ValueError(
            f"scientific completion validation failed in {path.name}: "
            + "; ".join(completion.reasons)
        )
    for key in ("run_id", "seed", "node_period", "feedback_additional_loss_probability"):
        if str(core_row[key]) != str(config[key]):
            raise ValueError(f"core summary/config mismatch for {key} in {path.name}")
    if str(core_row["run_id"]) != path.name or str(core_row["model_id"]).lower() != "aa":
        raise ValueError(f"core identity/model mismatch in {path.name}")
    if int(core_row["packet_fate_conservation_delta"]) != 0:
        raise ValueError(f"packet-fate conservation failed in {path.name}")
    if int(core_row["collision_outcome_conservation_delta"]) != 0:
        raise ValueError(f"collision conservation failed in {path.name}")
    if int(core_row["packet_fate_offered_count"]) != sum(
        int(core_row[column]) for column in (
            "packet_fate_delivered_count", "packet_fate_terminal_drop_count",
            "packet_fate_censored_count",
        )
    ):
        raise ValueError(f"packet-fate equation failed in {path.name}")
    attempts = tables["feedback_attempt_metrics_df"]
    transactions = tables["feedback_transaction_metrics_df"]
    if attempts["physical_attempt_id"].duplicated().any():
        raise ValueError(f"duplicate physical APC attempt ID in {path.name}")
    if (
        attempts["synthetic_erasure_decision"].astype(bool)
        & ~attempts["native_reception_status"].eq("success")
    ).any():
        raise ValueError(f"synthetic loss preceded/overrode native RX result in {path.name}")
    if float(config["feedback_additional_loss_probability"]) == 0.0 and attempts[
        "synthetic_erasure_decision"
    ].astype(bool).any():
        raise ValueError(f"nonzero synthetic loss at p=0 in {path.name}")
    attempts_by_feedback = attempts.groupby("feedback_id", sort=False)
    for feedback_id, group in attempts_by_feedback:
        numbers = sorted(int(value) for value in group["attempt_number"])
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(f"nonconsecutive physical attempts for {feedback_id}")
    delivered_ids = set(
        attempts.loc[attempts.logical_delivery_status.eq("delivered"), "feedback_id"].astype(str)
    )
    applied_ids = set(
        transactions.loc[transactions.application_timestamp_s.notna(), "feedback_id"].astype(str)
    )
    if not applied_ids.issubset(delivered_ids):
        raise ValueError(f"TPI application lacks a delivered physical feedback in {path.name}")
    airtime = tables["apc_control_airtime_summary_df"].iloc[0]
    attempt_airtime = pd.to_numeric(attempts["tx_airtime_s"], errors="raise").sum()
    if not math.isclose(
        float(airtime["apc_control_airtime_s"]), float(attempt_airtime),
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError(f"APC physical airtime disagrees with attempt rows in {path.name}")
    return RunBundle(
        run_id=path.name,
        config=config,
        core=core,
        tables=tables,
        source_fingerprint=str(source),
        config_fingerprint=str(config_fp),
        manifest_sha256=_sha256(manifest_path),
    )


def load_cohort(experiment_dir: str | Path) -> tuple[list[RunBundle], CohortCheck]:
    """Load all run directories and report exact cohort completeness."""
    runs_dir = ExperimentArtifactPaths(Path(experiment_dir)).runs
    bundles: list[RunBundle] = []
    errors: list[str] = []
    if runs_dir.is_dir():
        for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
            try:
                status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                if status.get("status") != "PASS":
                    errors.append(f"{run_dir.name}: status is not PASS")
                    continue
                bundles.append(read_run_bundle(run_dir))
            except Exception as exc:  # report bad bundles; never hide incomplete input
                errors.append(f"{run_dir.name}: {type(exc).__name__}: {exc}")
    keys = [
        (int(bundle.config.get("node_period", -1)),
         float(bundle.config.get("feedback_additional_loss_probability", math.nan)),
         int(bundle.config.get("seed", -1)))
        for bundle in bundles
    ]
    counts = pd.Series(keys, dtype=object).value_counts() if keys else pd.Series(dtype=int)
    actual_set = set(keys)
    missing = tuple(sorted(EXPECTED_KEYS - actual_set))
    duplicates = tuple(sorted(key for key, count in counts.items() if count > 1))
    unexpected = tuple(sorted(actual_set - EXPECTED_KEYS))
    if len({bundle.source_fingerprint for bundle in bundles}) > 1:
        errors.append("source fingerprints differ between runs")
    valid = (
        len(bundles) == len(EXPECTED_KEYS)
        and not missing and not duplicates and not unexpected and not errors
    )
    return bundles, CohortCheck(
        expected_count=len(EXPECTED_KEYS), actual_count=len(bundles),
        complete=valid, missing=missing, duplicates=duplicates,
        unexpected=unexpected, errors=tuple(errors),
    )


def _mean_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float, float, float]:
    numeric = pd.to_numeric(values, errors="raise").dropna().astype(float)
    n = len(numeric)
    if n == 0:
        return math.nan, math.nan, math.nan, math.nan
    mean = float(numeric.mean())
    sd = float(numeric.std(ddof=1)) if n > 1 else math.nan
    sem = sd / math.sqrt(n) if n > 1 else math.nan
    critical = float(student_t.ppf((1 + confidence) / 2, n - 1)) if n > 1 else math.nan
    half = critical * sem if n > 1 else math.nan
    return mean, sd, mean - half if n > 1 else math.nan, mean + half if n > 1 else math.nan


def aggregate_complete_cohort(
    bundles: Sequence[RunBundle], check: CohortCheck,
) -> dict[str, pd.DataFrame]:
    """Return run, seed-level and paired outputs; incomplete input is a hard error."""
    if not check.complete:
        raise ValueError(
            f"incomplete robustness cohort: actual={check.actual_count}, "
            f"expected={check.expected_count}, missing={len(check.missing)}"
        )
    source_fingerprints = {run.source_fingerprint for run in bundles}
    if len(source_fingerprints) != 1:
        raise ValueError("source fingerprint mismatch")
    core = pd.concat([run.core for run in bundles], ignore_index=True)
    for metric in CORE_METRICS:
        values = pd.to_numeric(core[metric], errors="raise")
        if not values.map(math.isfinite).all():
            raise ValueError(f"non-finite core metric: {metric}")
    if not core["packet_fate_conservation_delta"].eq(0).all():
        raise ValueError("packet-fate conservation failed")
    if not core["collision_outcome_conservation_delta"].eq(0).all():
        raise ValueError("collision conservation failed")

    long_rows = []
    for _, row in core.iterrows():
        for metric in CORE_METRICS:
            long_rows.append({
                "node_period": int(row["node_period"]),
                "feedback_additional_loss_probability": float(row["feedback_additional_loss_probability"]),
                "seed": int(row["seed"]), "run_id": str(row["run_id"]),
                "metric": metric, "value": float(row[metric]),
            })
    feedback_rows = []
    for run in bundles:
        stale = run.tables[STALE_TPI_TABLE_KEY]
        count_columns = (
            "total_data_physical_tx_count", "pre_decision_tx_count",
            "post_decision_data_tx_count", "eligible_data_tx_count",
            "stale_tpi_tx_count", "current_tpi_tx_count",
            "unclassified_tx_count", "version_invariant_violation_count",
        )
        totals = {
            column: int(pd.to_numeric(stale[column], errors="raise").sum())
            for column in count_columns
        }
        eligible = totals["eligible_data_tx_count"]
        stale_count = totals["stale_tpi_tx_count"]
        stale_fraction = stale_count / eligible if eligible else math.nan
        long_rows.append({
            "node_period": int(run.config["node_period"]),
            "feedback_additional_loss_probability": float(
                run.config["feedback_additional_loss_probability"]
            ),
            "seed": int(run.config["seed"]), "run_id": run.run_id,
            "metric": "stale_tpi_transmission_fraction",
            "value": stale_fraction,
        })
        feedback_row = {
            "run_id": run.run_id,
            "node_period": int(run.config["node_period"]),
            "feedback_additional_loss_probability": float(
                run.config["feedback_additional_loss_probability"]
            ),
            "seed": int(run.config["seed"]),
            **totals,
            "stale_tpi_transmission_fraction": stale_fraction,
            # Compatibility aliases retained for consumers of the Task 3
            # preview reporter; their definitions now use classified TX only.
            "stale_data_tx_count": stale_count,
            "eligible_post_decision_data_tx_count": eligible,
            "stale_data_tx_fraction": stale_fraction,
        }
        for role in ("node", "relay"):
            role_rows = stale.loc[stale.transmitter_role.eq(role)]
            prefix = f"{role}_"
            for column in count_columns:
                feedback_row[prefix + column] = int(
                    pd.to_numeric(role_rows[column], errors="raise").sum()
                )
            role_eligible = feedback_row[prefix + "eligible_data_tx_count"]
            feedback_row[prefix + "stale_tpi_transmission_fraction"] = (
                feedback_row[prefix + "stale_tpi_tx_count"] / role_eligible
                if role_eligible else math.nan
            )
        feedback_rows.append(feedback_row)
    run_metrics = pd.DataFrame(long_rows)
    summary_rows = []
    for (period, loss, metric), group in run_metrics.groupby(
        ["node_period", "feedback_additional_loss_probability", "metric"], sort=True
    ):
        mean, sd, lower, upper = _mean_ci(group["value"])
        summary_rows.append({
            "node_period": period, "feedback_additional_loss_probability": loss,
            "metric": metric,
            "n_seeds": int(group.loc[group["value"].notna(), "seed"].nunique()),
            "mean": mean, "sample_sd": sd, "ci95_lower": lower, "ci95_upper": upper,
        })
    summary = pd.DataFrame(summary_rows)

    paired_rows = []
    for period in (30, 90):
        for loss in (0.1, 0.3):
            for metric in PAIRED_METRICS:
                sub = run_metrics.loc[
                    run_metrics.node_period.eq(period) & run_metrics.metric.eq(metric)
                ]
                treated = sub.loc[sub.feedback_additional_loss_probability.eq(loss)].set_index("seed")["value"]
                baseline = sub.loc[sub.feedback_additional_loss_probability.eq(0.0)].set_index("seed")["value"]
                if set(treated.index) != set(APC_FEEDBACK_ROBUSTNESS_SEEDS) or set(baseline.index) != set(APC_FEEDBACK_ROBUSTNESS_SEEDS):
                    raise ValueError(f"unmatched seed pairs for {period}s/{loss}/{metric}")
                deltas = treated.sort_index() - baseline.sort_index()
                valid_deltas = deltas.dropna()
                mean, sd, lower, upper = _mean_ci(deltas)
                paired_rows.append({
                    "node_period": period, "feedback_additional_loss_probability": loss,
                    "metric": metric, "n_pairs": len(valid_deltas),
                    "excluded_undefined_pairs": len(deltas) - len(valid_deltas),
                    "mean_paired_difference": mean,
                    "sample_sd_difference": sd, "paired_ci95_lower": lower,
                    "paired_ci95_upper": upper,
                })
    paired = pd.DataFrame(paired_rows)

    table_masters: dict[str, pd.DataFrame] = {}
    for key, filename in APC_FEEDBACK_ROBUSTNESS_RUN_TABLES.items():
        table_masters[filename] = pd.concat([run.tables[key] for run in bundles], ignore_index=True)
    for run in bundles:
        attempt = run.tables["feedback_attempt_metrics_df"]
        transaction = run.tables["feedback_transaction_metrics_df"]
        airtime = run.tables["apc_control_airtime_summary_df"]
        tx_count = int(pd.to_numeric(attempt["synthetic_erasure_decision"], errors="coerce").fillna(False).sum())
        native_success = int(attempt["native_reception_status"].eq("success").sum())
        applied = int(transaction["application_timestamp_s"].notna().sum())
        apc_air = float(airtime.iloc[0]["apc_control_airtime_s"])
        all_air = float(airtime.iloc[0]["total_observed_tx_airtime_s"])
        feedback_row = next(item for item in feedback_rows if item["run_id"] == run.run_id)
        feedback_row.update({
            "physical_feedback_attempts": len(attempt),
            "native_successful_receptions": native_success,
            "synthetic_erasures": tx_count, "logical_transactions": len(transaction),
            "applied_transactions": applied,
            "transaction_apply_success_percent": 100.0 * applied / len(transaction) if len(transaction) else math.nan,
            "apc_control_airtime_s": apc_air, "all_tx_airtime_s": all_air,
            "apc_control_airtime_percent": 100.0 * apc_air / all_air if all_air else 0.0,
        })
    return {
        "run_metrics.csv": run_metrics,
        "scenario_summary.csv": summary,
        "paired_loss_effects.csv": paired,
        "feedback_metrics_by_run.csv": pd.DataFrame(feedback_rows),
        **table_masters,
    }


def write_complete_cohort(
    bundles: Sequence[RunBundle], check: CohortCheck, output_dir: str | Path,
    *, data_class: str = "SIMULATION",
) -> dict[str, Any]:
    """Atomically publish outputs only for a complete validated cohort."""
    destination = Path(output_dir).resolve()
    frames = aggregate_complete_cohort(bundles, check)
    if destination.exists():
        raise FileExistsError(f"report output already exists; refusing overwrite: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    checksums = {}
    try:
        for filename, frame in sorted(frames.items()):
            path = destination / filename
            atomic_write_csv(path, frame)
            checksums[filename] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        manifest = {
            "status": "COMPLETED", "data_class": data_class,
            "experiment_profile": APC_FEEDBACK_ROBUSTNESS_PROFILE,
            "expected_runs": len(EXPECTED_KEYS), "actual_runs": len(bundles),
            "source_fingerprint": bundles[0].source_fingerprint,
            "input_manifest_sha256": {run.run_id: run.manifest_sha256 for run in bundles},
            "outputs": checksums,
        }
        atomic_write_json(destination / "report_manifest.json", manifest)
    except Exception:
        # Do not recursively remove a partially published user directory. A
        # failure leaves it visibly incomplete and the caller must inspect it.
        raise
    return manifest


def check_experiment(experiment_dir: str | Path) -> dict[str, Any]:
    """Return completeness and provenance summary; no output is written."""
    bundles, check = load_cohort(experiment_dir)
    return {
        "status": "COMPLETE" if check.complete else "INCOMPLETE",
        "expected_runs": check.expected_count, "actual_runs": check.actual_count,
        "missing": [list(key) for key in check.missing],
        "duplicates": [list(key) for key in check.duplicates],
        "unexpected": [list(key) for key in check.unexpected],
        "errors": list(check.errors),
        "source_fingerprints": sorted({run.source_fingerprint for run in bundles}),
        "runs": [run.run_id for run in bundles],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate a complete APC feedback robustness cohort."
    )
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    bundles, check = load_cohort(args.experiment_dir)
    result = {
        "status": "COMPLETE" if check.complete else "INCOMPLETE",
        "expected_runs": check.expected_count,
        "actual_runs": check.actual_count,
        "missing": [list(key) for key in check.missing],
        "duplicates": [list(key) for key in check.duplicates],
        "unexpected": [list(key) for key in check.unexpected],
        "errors": list(check.errors),
    }
    if not check.complete:
        print(json.dumps(result, indent=2))
        return 2
    if args.output_dir is not None:
        result["report_manifest"] = write_complete_cohort(
            bundles, check, args.output_dir
        )
    print(json.dumps(result, indent=2, default=str))
    return 0


__all__ = [
    "CORE_METRICS", "CohortCheck", "RunBundle", "aggregate_complete_cohort",
    "check_experiment", "load_cohort", "read_run_bundle", "write_complete_cohort",
]


if __name__ == "__main__":
    raise SystemExit(main())
