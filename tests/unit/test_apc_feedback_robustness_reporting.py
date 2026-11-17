from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from loraphymac.experiments.apc_feedback_robustness_sweep.contract import (
    APC_FEEDBACK_ROBUSTNESS_RUN_TABLES,
    APC_FEEDBACK_ROBUSTNESS_SCENARIOS,
    APC_FEEDBACK_ROBUSTNESS_SEEDS,
    STALE_TPI_COLUMNS,
    STALE_TPI_METADATA_COLUMNS,
    STALE_TPI_PERSISTED_COLUMNS,
    STALE_TPI_TABLE_KEY,
    TABLE_SCHEMAS,
)
import loraphymac.experiments.common.scientific_output_manager as output_manager_module
from loraphymac.experiments.common.multiseed_pipeline import content_hash
from loraphymac.experiments.common.scientific_output_contract import required_table_keys
from loraphymac.experiments.common.scientific_output_manager import (
    ScientificOutputManager,
    collect_complete_source_provenance,
)
from loraphymac.experiments.apc_feedback_robustness_sweep.reporting import (
    CORE_METRICS,
    CohortCheck,
    RunBundle,
    _validate_stale_tpi_columns,
    _validate_stale_tpi_csv_header,
    _validate_stale_tpi_table,
    aggregate_complete_cohort,
    load_cohort,
    read_run_bundle,
    write_complete_cohort,
)


def _fixture_cohort():
    bundles = []
    for period, loss in APC_FEEDBACK_ROBUSTNESS_SCENARIOS:
        for seed in APC_FEEDBACK_ROBUSTNESS_SEEDS:
            run_id = f"TEST-{period}-{loss}-{seed}"
            core_values = {
                "run_id": run_id,
                "seed": seed,
                "node_period": period,
                "feedback_additional_loss_probability": loss,
                "packet_fate_conservation_delta": 0,
                "collision_outcome_conservation_delta": 0,
            }
            core_values.update({metric: 80.0 + 10.0 * loss for metric in CORE_METRICS})
            core_values["packet_fate_offered_count"] = 100
            core_values["packet_fate_delivered_count"] = 80
            core_values["packet_fate_terminal_drop_count"] = 20
            core_values["packet_fate_censored_count"] = 0
            tables = {}
            for key, columns in TABLE_SCHEMAS.items():
                row = {column: 0 for column in columns}
                if key == "feedback_attempt_metrics_df":
                    row.update({
                        "run_id": run_id, "feedback_id": run_id + ":f1",
                        "physical_attempt_id": run_id + ":a1", "attempt_number": 1,
                        "native_reception_status": "success",
                        "synthetic_erasure_decision": bool(loss),
                    })
                elif key == "feedback_transaction_metrics_df":
                    row.update({
                        "run_id": run_id, "feedback_id": run_id + ":f1",
                        "application_timestamp_s": 2.0,
                    })
                elif key == "stale_tpi_transmission_summary_df":
                    row.update({
                        "run_id": run_id, "transmitter_id": "N7",
                        "transmitter_role": "node", "mtype": "DATA_UP",
                        "control_scope_id": "GW|N7|node_apc",
                        "total_data_physical_tx_count": 10,
                        "pre_decision_tx_count": 0,
                        "post_decision_data_tx_count": 10,
                        "eligible_data_tx_count": 10,
                        "stale_tpi_tx_count": 2,
                        "current_tpi_tx_count": 8,
                        "unclassified_tx_count": 0,
                        "version_invariant_violation_count": 0,
                        "stale_tpi_transmission_fraction": 0.2,
                    })
                elif key == "apc_control_airtime_summary_df":
                    row.update({
                        "run_id": run_id, "apc_control_tx_count": 1,
                        "apc_control_airtime_s": 0.1, "total_observed_tx_count": 10,
                        "total_observed_tx_airtime_s": 1.0,
                    })
                tables[key] = pd.DataFrame([row], columns=columns)
            bundles.append(RunBundle(
                run_id=run_id,
                config={"node_period": period, "feedback_additional_loss_probability": loss, "seed": seed},
                core=pd.DataFrame([core_values]),
                tables=tables,
                source_fingerprint="test-source-fingerprint",
                config_fingerprint=f"config-{period}-{loss}-{seed}",
                manifest_sha256=f"manifest-{period}-{loss}-{seed}",
                data_class="TEST DATA",
            ))
    check = CohortCheck(30, 30, True, (), (), (), ())
    return bundles, check


def test_incomplete_cohort_cannot_generate_statistics():
    bundles, _ = _fixture_cohort()
    check = CohortCheck(30, 4, False, ((30, 0.1, 30),), (), (), ())
    with pytest.raises(ValueError, match="incomplete robustness cohort"):
        aggregate_complete_cohort(bundles[:4], check)


def test_complete_test_fixture_aggregates_means_paired_seeds_and_denominators():
    bundles, check = _fixture_cohort()
    outputs = aggregate_complete_cohort(bundles, check)
    summary = outputs["scenario_summary.csv"]
    row = summary.loc[
        summary.node_period.eq(30)
        & summary.feedback_additional_loss_probability.eq(0.1)
        & summary.metric.eq("pdr_percent")
    ].iloc[0]
    assert row.n_seeds == 5
    assert row["mean"] == 81.0
    paired = outputs["paired_loss_effects.csv"]
    delta = paired.loc[
        paired.node_period.eq(30)
        & paired.feedback_additional_loss_probability.eq(0.1)
        & paired.metric.eq("pdr_percent")
    ].iloc[0]
    assert delta.n_pairs == 5
    assert delta.mean_paired_difference == 1.0
    feedback = outputs["feedback_metrics_by_run.csv"]
    assert feedback.iloc[0].stale_data_tx_fraction == 0.2
    assert feedback.iloc[0].apc_control_airtime_percent == 10.0


def test_fixture_output_manifest_labels_test_data_and_checksums(tmp_path):
    bundles, check = _fixture_cohort()
    destination = tmp_path / "TEST DATA reporter"
    manifest = write_complete_cohort(
        bundles, check, destination, data_class="TEST DATA",
    )
    assert manifest["status"] == "COMPLETED"
    assert manifest["data_class"] == "TEST DATA"
    for filename, record in manifest["outputs"].items():
        from hashlib import sha256

        assert sha256((destination / filename).read_bytes()).hexdigest() == record["sha256"]
    read_back = pd.read_csv(destination / "scenario_summary.csv")
    assert len(read_back) == 6 * (len(CORE_METRICS) + 1)
    assert set(read_back["n_seeds"]) == {5}
    assert len(pd.read_csv(destination / "paired_loss_effects.csv")) == 2 * 2 * (len(CORE_METRICS) + 1)
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        write_complete_cohort(bundles, check, destination, data_class="TEST DATA")


def _valid_persisted_stale_frame(run_id="FIXTURE-30-0.0-42"):
    row = {
        "experiment_id": "fixture_robustness",
        "model_id": "aa",
        "revision": "APC_FEEDBACK_ROBUSTNESS_SWEEP_V1",
        "energy_accounting_model_version": "eq10_exclusive_transition_v1",
        "seed": 42,
        "node_period": 30,
        "feedback_additional_loss_probability": 0.0,
        "run_id": run_id,
        "transmitter_id": "N7",
        "transmitter_role": "node",
        "mtype": "DATA_UP",
        "control_scope_id": "GW|N7|node_apc",
        "total_data_physical_tx_count": 10,
        "pre_decision_tx_count": 0,
        "post_decision_data_tx_count": 10,
        "eligible_data_tx_count": 10,
        "stale_tpi_tx_count": 2,
        "current_tpi_tx_count": 8,
        "unclassified_tx_count": 0,
        "version_invariant_violation_count": 0,
        "stale_tpi_transmission_fraction": 0.2,
    }
    return pd.DataFrame([row], columns=STALE_TPI_PERSISTED_COLUMNS)


def test_persisted_stale_tpi_schema_and_metadata_are_accepted():
    frame = _valid_persisted_stale_frame()
    _validate_stale_tpi_table(frame, "FIXTURE-30-0.0-42")
    assert tuple(frame.columns[:len(STALE_TPI_METADATA_COLUMNS)]) == STALE_TPI_METADATA_COLUMNS
    assert tuple(frame.columns[len(STALE_TPI_METADATA_COLUMNS):]) == STALE_TPI_COLUMNS


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_scientific", "missing_scientific"),
        ("missing_metadata", "missing_metadata"),
        ("unexpected", "unexpected"),
        ("duplicate", "duplicate"),
    ],
)
def test_persisted_stale_tpi_schema_rejects_invalid_columns(mutation, message):
    columns = list(STALE_TPI_PERSISTED_COLUMNS)
    if mutation == "missing_scientific":
        columns.remove("stale_tpi_tx_count")
    elif mutation == "missing_metadata":
        columns.remove("revision")
    elif mutation == "unexpected":
        columns.append("unregistered_metadata")
    else:
        columns.append("stale_tpi_tx_count")
    with pytest.raises(ValueError, match=message):
        _validate_stale_tpi_columns(columns, "fixture")


def test_raw_csv_duplicate_header_is_rejected_before_pandas_read(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text(",".join((*STALE_TPI_PERSISTED_COLUMNS, "stale_tpi_tx_count")) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        _validate_stale_tpi_csv_header(path, "fixture")


def test_stale_tpi_rejects_invalid_numeric_and_preserves_null_zero_denominator():
    frame = _valid_persisted_stale_frame()
    frame["stale_tpi_tx_count"] = frame["stale_tpi_tx_count"].astype(object)
    frame.loc[0, "stale_tpi_tx_count"] = "not-a-count"
    with pytest.raises((ValueError, TypeError)):
        _validate_stale_tpi_table(frame, "FIXTURE-30-0.0-42")

    zero = _valid_persisted_stale_frame()
    zero.loc[0, [
        "total_data_physical_tx_count", "pre_decision_tx_count",
        "post_decision_data_tx_count", "eligible_data_tx_count",
        "stale_tpi_tx_count", "current_tpi_tx_count", "unclassified_tx_count",
        "version_invariant_violation_count",
    ]] = 0
    zero.loc[0, "stale_tpi_transmission_fraction"] = None
    _validate_stale_tpi_table(zero, "FIXTURE-30-0.0-42")
    assert pd.isna(zero.loc[0, "stale_tpi_transmission_fraction"])


def _persist_reporter_fixture_run(tmp_path, config, source, monkeypatch):
    """Write one deterministic synthetic bundle through the managed writer."""
    monkeypatch.setattr(output_manager_module, "validate_run_output_contract", lambda *_: None)
    experiment_id = "fixture_robustness"
    run_id = config["run_id"]
    metadata = {
        "experiment_id": experiment_id,
        "model_id": "aa",
        "revision": "APC_FEEDBACK_ROBUSTNESS_SWEEP_V1",
        "energy_accounting_model_version": "eq10_exclusive_transition_v1",
        "run_id": run_id,
        "seed": config["seed"],
        "node_period": config["node_period"],
        "feedback_additional_loss_probability": config[
            "feedback_additional_loss_probability"
        ],
    }

    def annotate(frame):
        result = frame.copy()
        for name, value in reversed(list(metadata.items())):
            if name in result.columns:
                result[name] = value
            else:
                result.insert(0, name, value)
        return result

    tables = {
        key: pd.DataFrame(columns=TABLE_SCHEMAS[key])
        for key in required_table_keys(config)
    }
    feedback_id = f"{run_id}:feedback:1"
    attempt = {column: None for column in TABLE_SCHEMAS["feedback_attempt_metrics_df"]}
    attempt.update({
        "run_id": run_id, "feedback_id": feedback_id,
        "physical_attempt_id": f"{run_id}:attempt:1", "attempt_number": 1,
        "controller_id": "GW", "target_transmitter_id": "N7",
        "transmitter_role": "gateway", "mtype": 4, "tx_timestamp_s": 1.0,
        "channel_id": 1, "tx_airtime_s": 0.1, "ptx_dbm": 14.0,
        "native_reception_status": "success", "native_failure_reason": None,
        "logical_failure_reason": None, "synthetic_loss_probability": 0.0,
        "synthetic_erasure_decision": False, "logical_delivery_status": "delivered",
        "tpi_application_status": "applied", "tpi_version": 1,
    })
    tables["feedback_attempt_metrics_df"] = pd.DataFrame(
        [attempt], columns=TABLE_SCHEMAS["feedback_attempt_metrics_df"]
    )
    transaction = {column: None for column in TABLE_SCHEMAS["feedback_transaction_metrics_df"]}
    transaction.update({
        "run_id": run_id, "decision_id": f"{run_id}:decision:1",
        "feedback_id": feedback_id, "target_transmitter_id": "N7",
        "controller_id": "GW", "decision_timestamp_s": 0.5,
        "desired_tpi": 4, "tpi_version": 1, "tpi_state_changed": True,
        "physical_attempts": 1, "native_successful_receptions": 1,
        "synthetic_erasures": 0, "application_timestamp_s": 1.2,
        "application_latency_s": 0.7, "retry_exhausted": False, "status": "applied",
    })
    tables["feedback_transaction_metrics_df"] = pd.DataFrame(
        [transaction], columns=TABLE_SCHEMAS["feedback_transaction_metrics_df"]
    )
    tables[STALE_TPI_TABLE_KEY] = _valid_persisted_stale_frame(run_id).loc[
        :, list(STALE_TPI_COLUMNS)
    ]
    tables["apc_convergence_metrics_df"] = pd.DataFrame(
        columns=TABLE_SCHEMAS["apc_convergence_metrics_df"]
    )
    tables["apc_control_airtime_summary_df"] = pd.DataFrame([{
        "run_id": run_id, "apc_control_tx_count": 1,
        "apc_control_airtime_s": 0.1, "total_observed_tx_count": 10,
        "total_observed_tx_airtime_s": 1.0, "apc_control_airtime_percent": 10.0,
        "by_transmitter_role_json": "{}", "by_mtype_json": "{}",
    }], columns=TABLE_SCHEMAS["apc_control_airtime_summary_df"])
    persisted_tables = {key: annotate(frame) for key, frame in tables.items()}
    core = {
        "run_id": run_id, "model_id": "aa", "seed": config["seed"],
        "node_period": config["node_period"],
        "feedback_additional_loss_probability": config[
            "feedback_additional_loss_probability"
        ],
        "pdr_percent": 80.0, "packet_fate_offered_count": 10,
        "packet_fate_delivered_count": 8, "packet_fate_terminal_drop_count": 2,
        "packet_fate_censored_count": 0, "packet_fate_conservation_delta": 0,
        "collision_outcome_conservation_delta": 0, "total_network_energy_J": 1.0,
        "energy_per_delivered_packet_J": 0.125, "avg_e2e_delay_ms": 10.0,
    }
    core_frame = annotate(pd.DataFrame([core]))
    resolved = {"resolved_simulator_config": config, "config_hash": content_hash(config)}
    experiment_dir = tmp_path / experiment_id
    writer = ScientificOutputManager(experiment_dir)
    return writer.write_run(
        experiment_id=experiment_id, run_id=run_id, config=config,
        summary=core, tables=persisted_tables, source_provenance=source,
        validation_results={"fixture_contract": True}, validation_tables=tables,
        table_filename_overrides=APC_FEEDBACK_ROBUSTNESS_RUN_TABLES,
        pipeline_compatibility={
            "core_summary.csv": core_frame,
            "status.json": {"run_id": run_id, "status": "PASS"},
        }, resolved_config_payload=resolved,
    )


def _reporter_config(run_id, period, loss, seed):
    config_path = Path(__file__).parents[2] / "src/loraphymac/experiments/apc_feedback_robustness_sweep/configs/aa_feedback_robustness_v1.json"
    config = json.loads(config_path.read_text()) ["base_config"]
    config.update({
        "run_id": run_id, "seed": seed, "node_period": period,
        "feedback_additional_loss_probability": loss,
    })
    return config


def test_real_persisted_schema_round_trips_provenance_counts_and_incomplete_cohort(
    tmp_path, monkeypatch,
):
    source = collect_complete_source_provenance(Path(__file__).resolve().parents[2])
    combinations = ((30, 0.0), (30, 0.3), (90, 0.0), (90, 0.3))
    experiment_dir = None
    loaded = []
    for index, (period, loss) in enumerate(combinations):
        run_id = f"fixture-run-{index:02d}"
        config = _reporter_config(run_id, period, loss, 42)
        result = _persist_reporter_fixture_run(tmp_path, config, source, monkeypatch)
        experiment_dir = result.run_dir.parents[1]
        bundle = read_run_bundle(result.run_dir)
        loaded.append(bundle)
        stale = bundle.tables[STALE_TPI_TABLE_KEY]
        assert tuple(stale.columns) == STALE_TPI_PERSISTED_COLUMNS
        assert stale.loc[0, "experiment_id"] == "fixture_robustness"
        assert stale.loc[0, "revision"] == "APC_FEEDBACK_ROBUSTNESS_SWEEP_V1"
        assert stale.loc[0, "stale_tpi_tx_count"] == 2
        assert stale.loc[0, "current_tpi_tx_count"] == 8
        assert stale.loc[0, "unclassified_tx_count"] == 0
    assert experiment_dir is not None
    bundles, check = load_cohort(experiment_dir)
    assert len(bundles) == 4
    assert check.expected_count == 30
    assert check.actual_count == 4
    assert check.complete is False
    assert not check.errors
    with pytest.raises(ValueError, match="incomplete robustness cohort"):
        aggregate_complete_cohort(bundles, check)
    assert not (tmp_path / "final_manuscript_statistics").exists()
