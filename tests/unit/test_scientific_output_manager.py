from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

import loraphymac.experiments.common.scientific_output_manager as output_manager
from loraphymac.experiments.apc_load_sweep.contract import (
    APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.contract import (
    APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
)
from loraphymac.experiments.common.contract_registry import experiment_run_tables
from loraphymac.experiments.common.multiseed_pipeline import (
    RUN_TABLES,
    rebuild_master_tables,
)
from loraphymac.experiments.common.scientific_output_adapter import (
    persist_multiseed_run,
)
from loraphymac.experiments.common.scientific_output_contract import (
    output_contract_for,
    required_table_keys,
)
from loraphymac.experiments.common.scientific_output_manager import (
    MANIFEST_NAME,
    ScientificOutputError,
    ScientificOutputManager,
    collect_complete_source_provenance,
    collect_source_provenance,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    "src/loraphymac/experiments/common/scientific_output_manager.py",
    "src/loraphymac/experiments/common/scientific_output_contract.py",
    "tests/unit/test_scientific_output_manager.py",
)


def _config(*, persistence: str = "scientific", collection: str = "summary") -> dict:
    return {
        "experiment_profile": APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE,
        "output_profile": collection,
        "persistence_profile": persistence,
        "run_id": "r_synthetic_01",
        "seed": 30,
        "terrain_delta_h": 90,
    }


def _tables(config: dict, *, diagnostic: bool = False) -> dict[str, pd.DataFrame]:
    keys = set(required_table_keys(config))
    if diagnostic:
        keys.update(output_manager.DIAGNOSTIC_TABLE_KEYS)
    tables = {key: pd.DataFrame({"value": [1]}) for key in sorted(keys)}
    contract = output_contract_for(config)
    if contract is not None:
        for key, columns in contract.table_schemas.items():
            if key in tables:
                tables[key] = pd.DataFrame(columns=columns)
    return tables


def _source() -> dict:
    return collect_complete_source_provenance(REPO_ROOT)


def _manager(tmp_path: Path) -> ScientificOutputManager:
    return ScientificOutputManager(tmp_path / "synthetic_experiment")


@pytest.fixture(autouse=True)
def bypass_simulator_specific_validator(monkeypatch):
    # These tests cover the manager boundary; experiment validators have their
    # own suites and synthetic tables intentionally do not model real metrics.
    monkeypatch.setattr(output_manager, "validate_run_output_contract", lambda *_: None)


def test_scientific_selection_contains_every_contract_table():
    config = _config()
    tables = _tables(config)
    selected = _manager(Path("/tmp")).select_tables(config, tables)
    assert selected.manager_writes is True
    assert selected.persistence_profile == "scientific"
    assert set(selected.table_keys) == set(required_table_keys(config))


def test_complete_provenance_covers_runtime_config_metrics_control_and_runner():
    provenance = _source()
    paths = set(provenance["scope_paths"])
    assert provenance["dependency_scope_complete"] is True
    assert {
        "src/loraphymac/config/defaults.py",
        "src/loraphymac/simulation/runner.py",
        "src/loraphymac/metrics/statistics.py",
        "src/loraphymac/runtime/node_tx.py",
        "src/loraphymac/experiments/common/multiseed_pipeline.py",
    } <= paths
    assert provenance["runtime_environment"]["distributions"]


def test_missing_required_table_fails_before_write(tmp_path):
    config = _config()
    tables = _tables(config)
    tables.pop(next(iter(tables)))
    with pytest.raises(ScientificOutputError, match="required output tables missing"):
        _manager(tmp_path).write_run(
            experiment_id="synthetic_experiment",
            run_id=config["run_id"],
            config=config,
            summary={},
            tables=tables,
            source_provenance=_source(),
            validation_results={"synthetic": True},
        )
    assert not (_manager(tmp_path).experiment_dir / "runs").exists()


def test_v2_missing_mandatory_table_fails_selection():
    config = _config()
    config.update(
        experiment_profile=APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
        tx_power_model_id="ff",
    )
    tables = _tables(config)
    tables.pop("channel_occupancy_df")
    with pytest.raises(ScientificOutputError, match="channel_occupancy_df"):
        _manager(Path("/tmp")).select_tables(config, tables)


def test_v2_conditional_non_applicable_events_fail_before_write(tmp_path):
    from loraphymac.metrics.apc_metrics import PTX_HISTORY_COLUMNS
    from loraphymac.metrics.instrumentation_metrics import (
        APC_FEEDBACK_LIFECYCLE_COLUMNS,
        WANG_POWER_COMMAND_LIFECYCLE_COLUMNS,
        WANG_SNR_REPORT_LIFECYCLE_COLUMNS,
    )

    config = _config()
    config.update(
        experiment_profile=APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
        tx_power_model_id="ff",
    )
    tables = _tables(config)
    tables["ptx_history_df"] = pd.DataFrame(columns=PTX_HISTORY_COLUMNS)
    tables["apc_feedback_lifecycle_df"] = pd.DataFrame(
        columns=APC_FEEDBACK_LIFECYCLE_COLUMNS
    )
    tables["wang_snr_report_lifecycle_df"] = pd.DataFrame(
        columns=WANG_SNR_REPORT_LIFECYCLE_COLUMNS
    )
    tables["wang_power_command_lifecycle_df"] = pd.DataFrame(
        columns=WANG_POWER_COMMAND_LIFECYCLE_COLUMNS
    )
    tables["apc_feedback_lifecycle_df"] = pd.DataFrame({"event": ["fake"]})

    with pytest.raises(ScientificOutputError, match="must be empty because it is not applicable"):
        _manager(tmp_path).write_run(
            experiment_id="synthetic_experiment",
            run_id=config["run_id"],
            config=config,
            summary={},
            tables=tables,
            source_provenance={},
            validation_results={"synthetic": True},
        )
    assert not (_manager(tmp_path).experiment_dir / "runs").exists()


def test_v2_schema_complete_empty_lifecycle_tables_round_trip(tmp_path):
    from loraphymac.metrics.apc_metrics import PTX_HISTORY_COLUMNS
    from loraphymac.metrics.instrumentation_metrics import (
        APC_FEEDBACK_LIFECYCLE_COLUMNS,
        WANG_POWER_COMMAND_LIFECYCLE_COLUMNS,
        WANG_SNR_REPORT_LIFECYCLE_COLUMNS,
    )

    config = _config()
    config.update(
        experiment_profile=APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
        tx_power_model_id="ff",
    )
    tables = _tables(config)
    for key, columns in {
        "ptx_history_df": PTX_HISTORY_COLUMNS,
        "apc_feedback_lifecycle_df": APC_FEEDBACK_LIFECYCLE_COLUMNS,
        "wang_snr_report_lifecycle_df": WANG_SNR_REPORT_LIFECYCLE_COLUMNS,
        "wang_power_command_lifecycle_df": WANG_POWER_COMMAND_LIFECYCLE_COLUMNS,
    }.items():
        tables[key] = pd.DataFrame(columns=columns)

    manager = _manager(tmp_path)
    result = manager.write_run(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        summary={},
        tables=tables,
        source_provenance=_source(),
        validation_results={"synthetic_boundary": True},
    )
    assert result.status == "COMPLETED"
    check = manager.validate_completed(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        source_provenance=_source(),
    )
    assert check.safe_to_skip is True
    for key in (
        "ptx_history_df",
        "apc_feedback_lifecycle_df",
        "wang_snr_report_lifecycle_df",
        "wang_power_command_lifecycle_df",
    ):
        actual = manager.read_table(result.run_dir, key)
        assert actual.empty
        assert list(actual.columns) == list(tables[key].columns)


def test_unknown_profile_is_rejected_for_explicit_scientific_persistence(tmp_path):
    config = {
        "experiment_profile": "unregistered_profile_v1",
        "output_profile": "summary",
        "persistence_profile": "scientific",
    }
    with pytest.raises(
        ScientificOutputError,
        match="requires a registered scientific output contract",
    ):
        _manager(tmp_path).select_tables(config, {})


def test_experiment_validator_failure_blocks_all_persistence(tmp_path, monkeypatch):
    config = _config()
    manager = _manager(tmp_path)

    def reject(*_args):
        raise ValueError("packet fate mismatch")

    monkeypatch.setattr(output_manager, "validate_run_output_contract", reject)
    with pytest.raises(ScientificOutputError, match="packet fate mismatch"):
        manager.write_run(
            experiment_id="synthetic_experiment",
            run_id=config["run_id"],
            config=config,
            summary={},
            tables=_tables(config),
            source_provenance=_source(),
            validation_results={"synthetic": True},
        )
    assert not (manager.experiment_dir / "runs").exists()


def test_core_is_rejected_for_current_scientific_contract(tmp_path):
    config = _config(persistence="core")
    with pytest.raises(ScientificOutputError, match="requires persistence_profile='scientific'"):
        _manager(tmp_path).select_tables(config, _tables(config))


def test_diagnostic_requires_raw_event_tables():
    config = _config(persistence="diagnostic", collection="detailed")
    tables = _tables(config, diagnostic=True)
    tables.pop("event_log_df")
    with pytest.raises(ScientificOutputError, match="event_log_df"):
        _manager(Path("/tmp")).select_tables(config, tables)


def test_legacy_profile_is_passthrough_and_does_not_change_old_writer_behavior(tmp_path):
    config = _config()
    config.pop("persistence_profile")
    manager = _manager(tmp_path)
    tables = _tables(config)
    selection = manager.select_tables(config, tables)
    assert selection.manager_writes is False
    assert set(selection.table_keys) == set(tables)
    result = manager.write_run(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        summary={},
        tables=tables,
        source_provenance={},
        validation_results={},
    )
    assert result.legacy_passthrough is True
    assert result.status == "LEGACY_PASSTHROUGH"
    assert not manager.experiment_dir.exists()


def test_atomic_complete_manifest_and_idempotent_safe_validation(tmp_path):
    config = _config()
    manager = _manager(tmp_path)
    result = manager.write_run(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        summary={"pdr_percent": 1.0},
        tables=_tables(config),
        source_provenance=_source(),
        validation_results={"packet_fate": True, "energy": True},
    )
    assert result.status == "COMPLETED"
    assert result.run_dir is not None
    manifest = json.loads((result.run_dir / MANIFEST_NAME).read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["validation_results"]["experiment_output_contract"] is True
    first = manager.validate_completed(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        source_provenance=_source(),
    )
    second = manager.validate_completed(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        source_provenance=_source(),
    )
    assert first.safe_to_skip and second.safe_to_skip
    assert first.reasons == second.reasons == ()


def test_incomplete_source_scope_is_explicitly_unsafe_to_skip(tmp_path):
    config = _config()
    manager = _manager(tmp_path)
    result = manager.write_run(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        summary={},
        tables=_tables(config),
        source_provenance=collect_source_provenance(REPO_ROOT, SOURCE_PATHS),
        validation_results={"synthetic": True},
    )
    check = manager.validate_completed(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        source_provenance=collect_source_provenance(REPO_ROOT, SOURCE_PATHS),
    )
    assert result.status == "COMPLETED"
    assert not check.safe_to_skip
    assert "UNSAFE_TO_SKIP: source dependency scope is incomplete" in check.reasons


def test_failed_write_never_publishes_completed_run(tmp_path, monkeypatch):
    config = _config()
    manager = _manager(tmp_path)
    original = manager._atomic_dataframe
    count = 0

    def fail_second(path, frame, *, gzip):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("synthetic write failure")
        original(path, frame, gzip=gzip)

    monkeypatch.setattr(manager, "_atomic_dataframe", fail_second)
    with pytest.raises(OSError, match="synthetic write failure"):
        manager.write_run(
            experiment_id="synthetic_experiment",
            run_id=config["run_id"],
            config=config,
            summary={},
            tables=_tables(config),
            source_provenance=_source(),
            validation_results={"synthetic": True},
        )
    assert not (manager.experiment_dir / "runs" / config["run_id"]).exists()
    assert not list((manager.experiment_dir / "runs").glob("*.partial-*"))
    assert not list((manager.experiment_dir / "runs").glob(".*.lock"))


def test_existing_run_is_never_overwritten(tmp_path):
    config = _config()
    manager = _manager(tmp_path)
    args = {
        "experiment_id": "synthetic_experiment",
        "run_id": config["run_id"],
        "config": config,
        "summary": {},
        "tables": _tables(config),
        "source_provenance": _source(),
        "validation_results": {"synthetic": True},
    }
    first = manager.write_run(**args)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        manager.write_run(**args)
    assert manager.validate_completed(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        source_provenance=args["source_provenance"],
    ).safe_to_skip
    assert first.run_dir is not None


@pytest.mark.extended_research
def test_existing_real_compare_bundle_round_trips_through_scientific_manager(tmp_path):
    source_run = (
        REPO_ROOT
        / "artifacts/experiments/apc_compare_model_terrain_sweep/"
        "apc_compare_model_terrain_sweep_v1_r01/models/aa/runs/"
        "aa__terrain_delta_h-90__s30__5031ecef4a"
    )
    if not source_run.is_dir():
        pytest.skip("validated local production bundle is not present in this checkout")

    original = json.loads((source_run / "resolved_config.json").read_text(encoding="utf-8"))
    config = dict(original["resolved_simulator_config"])
    config["output_profile"] = config.get("output_profile", "detailed")
    config["persistence_profile"] = "scientific"
    summary_frame = pd.read_csv(source_run / "core_summary.csv")
    summary = summary_frame.iloc[0].to_dict()
    status_payload = json.loads((source_run / "status.json").read_text(encoding="utf-8"))
    registered = experiment_run_tables(config)
    table_files = {**RUN_TABLES, **registered}
    outputs = {
        key: pd.read_csv(source_run / "tables" / filename)
        for key, filename in table_files.items()
    }
    outputs["core_summary_df"] = summary_frame
    outputs["relay_bottleneck_metrics_df"] = pd.read_csv(
        source_run / "tables" / "relay_bottleneck_metrics.csv"
    )
    assert status_payload["status"] == "PASS"
    assert config["experiment_profile"] == APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE
    assert set(outputs) >= set(required_table_keys(config))

    experiment_id = "c2_1c_real_bundle_roundtrip"
    run_id = str(config["run_id"])
    effective_config = dict(config)
    source_provenance = collect_complete_source_provenance(REPO_ROOT)
    compatibility_keys = sorted(
        set(RUN_TABLES) | set(registered) | {"relay_bottleneck_metrics_df"}
    )
    filename_overrides = {
        **RUN_TABLES,
        **registered,
        "relay_bottleneck_metrics_df": "relay_bottleneck_metrics.csv",
    }
    resolved_payload = {
        **original,
        "resolved_simulator_config": effective_config,
        "c2_1c_validation_copy": {
            "source_run_id": run_id,
            "source_profile": config["experiment_profile"],
            "source_config_hash": original.get("config_hash"),
            "validation_experiment_id": experiment_id,
        },
    }
    manager = ScientificOutputManager(tmp_path / experiment_id)
    result = persist_multiseed_run(
        experiment_dir=manager.experiment_dir,
        experiment_id=experiment_id,
        run_id=run_id,
        config=effective_config,
        source_provenance=source_provenance,
        summary=summary,
        outputs=outputs,
        annotated_core_summary=summary_frame,
        annotated_tables=outputs,
        annotated_relay_bottleneck=outputs["relay_bottleneck_metrics_df"],
        status_payload=status_payload,
        resolved_config_payload=resolved_payload,
        pipeline_table_filenames=RUN_TABLES,
        validation_results={
            "source_run_pass": True,
            "experiment_output_contract": True,
            "packet_fate_conservation": int(summary["packet_fate_conservation_delta"]) == 0,
            "corrected_energy": config["energy_accounting_mode"] == "corrected",
        },
    )

    assert result.status == "COMPLETED"
    assert result.run_dir is not None
    completed = manager.validate_completed(
        experiment_id=experiment_id,
        run_id=run_id,
        config=effective_config,
        source_provenance=source_provenance,
        additional_table_keys=compatibility_keys,
        table_filename_overrides=filename_overrides,
    )
    assert completed.safe_to_skip, completed.reasons
    for key, expected in outputs.items():
        pd.testing.assert_frame_equal(
            manager.read_table(result.run_dir, key), expected, check_exact=True
        )

    masters = rebuild_master_tables(
        manager.experiment_dir,
        experiment_profile=APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE,
    )
    assert len(masters["all_runs_core_summary.csv"]) == 1
    assert len(masters["per_run_ptx_history.csv"]) == len(outputs["ptx_history_df"])


def test_checksum_corruption_prevents_safe_skip(tmp_path):
    config = _config()
    manager = _manager(tmp_path)
    result = manager.write_run(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        summary={},
        tables=_tables(config),
        source_provenance=_source(),
        validation_results={"synthetic": True},
    )
    assert result.run_dir is not None
    manifest = result.manifest
    assert manifest is not None
    record = next(iter(manifest["tables"].values()))
    artifact = result.run_dir / record["path"]
    original = artifact.read_bytes()
    artifact.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    check = manager.validate_completed(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        source_provenance=_source(),
    )
    assert not check.safe_to_skip
    assert any("checksum mismatch" in reason for reason in check.reasons)


def test_configuration_source_schema_and_contract_drift_fail_closed(tmp_path):
    config = _config()
    manager = _manager(tmp_path)
    result = manager.write_run(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        summary={},
        tables=_tables(config),
        source_provenance=_source(),
        validation_results={"synthetic": True},
    )
    assert result.run_dir is not None

    config_drift = {**config, "seed": 36}
    assert not manager.validate_completed(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config_drift,
        source_provenance=_source(),
    ).safe_to_skip
    source_drift = collect_source_provenance(
        REPO_ROOT, ("src/loraphymac/experiments/common/contract_registry.py",)
    )
    assert not manager.validate_completed(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        source_provenance=source_drift,
    ).safe_to_skip
    changed_schema = ScientificOutputManager(manager.experiment_dir, schema_version="future_v2")
    assert not changed_schema.validate_completed(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        source_provenance=_source(),
    ).safe_to_skip

    path = result.run_dir / MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["experiment_contract_version"] = "wrong_contract"
    path.write_text(json.dumps(manifest))
    check = manager.validate_completed(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        source_provenance=_source(),
    )
    assert not check.safe_to_skip
    assert "manifest mismatch: experiment_contract_version" in check.reasons


def test_legacy_csv_and_gzip_readers_preserve_values(tmp_path):
    manager = _manager(tmp_path)
    frame = pd.DataFrame({"entity": ["relay-1"], "energy_j": [0.123456789012345]})
    csv_path = tmp_path / "legacy.csv"
    frame.to_csv(csv_path, index=False)
    gzip_path = tmp_path / "legacy.csv.gz"
    with gzip.open(gzip_path, "wt", encoding="utf-8") as stream:
        frame.to_csv(stream, index=False)
    pd.testing.assert_frame_equal(manager.read_table(csv_path, "energy_breakdown_df"), frame)
    pd.testing.assert_frame_equal(manager.read_table(gzip_path, "energy_breakdown_df"), frame)


def test_manager_csv_serialization_matches_legacy_for_null_numeric_and_string_columns(tmp_path):
    manager = _manager(tmp_path)
    frame = pd.DataFrame(
        {
            "label": pd.Series(["relay", None, "", "NA"], dtype="object"),
            "count": pd.Series([1, None, 3, 4], dtype="Int64"),
            "energy_j": [0.125, float("nan"), 2.5, 4.0],
        }
    )
    expected = tmp_path / "legacy-compatible.csv"
    managed = tmp_path / "managed-compatible.csv"
    frame.to_csv(expected, index=False)
    manager._atomic_dataframe(managed, frame, gzip=False)
    assert managed.read_bytes() == expected.read_bytes()
    pd.testing.assert_frame_equal(
        manager.read_table(managed, "typed_test_df"),
        pd.read_csv(expected),
    )


def test_legacy_xlsx_reader_uses_verified_sheet_mapping(tmp_path):
    manager = _manager(tmp_path)
    frame = pd.DataFrame({"value": [1, 2]})
    path = tmp_path / "legacy.xlsx"
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name="01_core_summary", index=False)
    pd.testing.assert_frame_equal(manager.read_table(path, "core_summary_df"), frame)


def test_managed_reader_loads_selected_table_from_manifest(tmp_path):
    config = _config()
    manager = _manager(tmp_path)
    tables = _tables(config)
    result = manager.write_run(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        summary={},
        tables=tables,
        source_provenance=_source(),
        validation_results={"synthetic": True},
        gzip_tables=("packet_fate_df",),
    )
    assert result.run_dir is not None
    actual = manager.read_table(result.run_dir, "packet_fate_df")
    pd.testing.assert_frame_equal(actual, tables["packet_fate_df"])


def test_configured_gzip_persists_all_tables_and_validates_checksums(tmp_path):
    config = _config()
    config["gzip_scientific_tables"] = True
    manager = _manager(tmp_path)
    tables = _tables(config)
    result = manager.write_run(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        summary={},
        tables=tables,
        source_provenance=_source(),
        validation_results={"synthetic": True},
    )
    assert result.run_dir is not None
    manifest = json.loads((result.run_dir / MANIFEST_NAME).read_text())
    assert set(manifest["gzip_table_keys"]) == set(manifest["required_table_keys"])
    assert all(record["path"].endswith(".csv.gz") for record in manifest["tables"].values())
    for key, expected in tables.items():
        if key in manifest["tables"]:
            actual = manager.read_table(result.run_dir, key)
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    check = manager.validate_completed(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        source_provenance=_source(),
    )
    assert check.safe_to_skip

    table_path = result.run_dir / manifest["tables"]["packet_fate_df"]["path"]
    table_path.unlink()
    missing = manager.validate_completed(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        source_provenance=_source(),
    )
    assert not missing.safe_to_skip
    assert any("artifact missing" in reason for reason in missing.reasons)


def test_gzip_checksum_detects_corrupted_table(tmp_path):
    config = _config()
    config["gzip_scientific_tables"] = True
    manager = _manager(tmp_path)
    result = manager.write_run(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        summary={},
        tables=_tables(config),
        source_provenance=_source(),
        validation_results={"synthetic": True},
    )
    assert result.run_dir is not None
    manifest = json.loads((result.run_dir / MANIFEST_NAME).read_text())
    table_path = result.run_dir / manifest["tables"]["packet_fate_df"]["path"]
    table_path.write_bytes(table_path.read_bytes()[:-1] + b"x")
    check = manager.validate_completed(
        experiment_id="synthetic_experiment",
        run_id=config["run_id"],
        config=config,
        source_provenance=_source(),
    )
    assert not check.safe_to_skip
    assert any("checksum mismatch" in reason for reason in check.reasons)


def test_gzip_serialization_is_byte_deterministic(tmp_path):
    frame = pd.DataFrame({"label": ["node", "relay"], "energy_j": [0.25, 1.5]})
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    ScientificOutputManager._atomic_dataframe(first, frame, gzip=True)
    ScientificOutputManager._atomic_dataframe(second, frame, gzip=True)
    assert first.read_bytes() == second.read_bytes()
    assert int.from_bytes(first.read_bytes()[4:8], "little") == 0
    assert first.read_bytes()[3] & 0x08 == 0
