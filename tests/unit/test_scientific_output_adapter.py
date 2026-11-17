from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import loraphymac.experiments.common.scientific_output_manager as output_manager
from loraphymac.experiments.apc_load_sweep.contract import (
    APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE,
)
from loraphymac.experiments.common.scientific_output_adapter import (
    persist_multiseed_run,
    persistence_route,
)
from loraphymac.experiments.common.scientific_output_contract import (
    output_contract_for,
    required_table_keys,
)
from loraphymac.experiments.common.scientific_output_manager import (
    collect_complete_source_provenance,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_omitted_profile_keeps_legacy_route_and_explicit_profile_uses_manager():
    assert persistence_route({}).manager_enabled is False
    assert persistence_route({"persistence_profile": None}).manager_enabled is False
    assert persistence_route({"persistence_profile": "scientific"}).manager_enabled is True


def test_manager_adapter_publishes_legacy_pipeline_reader_layout_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(output_manager, "validate_run_output_contract", lambda *_: None)
    config = {
        "experiment_profile": APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE,
        "output_profile": "summary",
        "persistence_profile": "scientific",
        "gzip_scientific_tables": True,
        "run_id": "run_0001",
        "seed": 30,
        "terrain_delta_h": 90,
    }
    outputs = {
        key: pd.DataFrame({"value": [1.25]})
        for key in required_table_keys(config)
    }
    contract = output_contract_for(config)
    for key, columns in contract.table_schemas.items():
        outputs[key] = pd.DataFrame(columns=columns)
    outputs["core_summary_df"] = pd.DataFrame({"pdr_percent": [82.5]})
    summary = {"pdr_percent": 82.5}
    compatibility_summary = pd.DataFrame(
        {"experiment_id": ["managed_test"], "run_id": ["run_0001"], "pdr_percent": [82.5]}
    )
    compatibility_status = {"run_id": "run_0001", "status": "PASS"}
    result = persist_multiseed_run(
        experiment_dir=tmp_path / "managed_test",
        experiment_id="managed_test",
        run_id="run_0001",
        config=config,
        source_provenance=collect_complete_source_provenance(REPO_ROOT),
        summary=summary,
        outputs=outputs,
        annotated_core_summary=compatibility_summary,
        annotated_tables=outputs,
        annotated_relay_bottleneck=pd.DataFrame({"entity_id": ["R1"]}),
        status_payload=compatibility_status,
        resolved_config_payload={"resolved_simulator_config": config},
        pipeline_table_filenames={"packet_fate_df": "packet_fates.csv"},
        validation_results={"experiment_output_contract": True},
    )
    assert result.status == "COMPLETED"
    run_dir = result.run_dir
    assert run_dir is not None
    assert pd.read_csv(run_dir / "core_summary.csv").equals(compatibility_summary)
    assert json.loads((run_dir / "status.json").read_text()) == compatibility_status
    assert (run_dir / "tables" / "packet_fates.csv.gz").is_file()
    assert not (run_dir / "tables" / "packet_fates.csv").exists()
    assert (run_dir / "tables" / "relay_bottleneck_metrics.csv.gz").is_file()
    manifest = json.loads((run_dir / output_manager.MANIFEST_NAME).read_text())
    assert set(manifest["gzip_table_keys"]) == set(manifest["required_table_keys"])
    assert manifest["tables"]["packet_fate_df"]["path"] == "tables/packet_fates.csv.gz"
    assert manifest["tables"]["packet_fate_df"]["sha256"]
    assert (run_dir / "core_summary.csv").is_file()
    assert (run_dir / "status.json").is_file()
    assert manifest["source_provenance"]["dependency_scope_complete"] is True
    assert manifest["compatibility_artifacts"]["status.json"]["sha256"]
    assert manifest["compatibility_artifacts"]["core_summary.csv"]["sha256"]
    provenance = manifest["source_provenance"]
    manager = output_manager.ScientificOutputManager(tmp_path / "managed_test")
    check = manager.validate_completed(
        experiment_id="managed_test",
        run_id="run_0001",
        config=config,
        source_provenance=provenance,
        additional_table_keys=manifest["additional_table_keys"],
        table_filename_overrides=manifest["table_filename_overrides"],
    )
    assert check.safe_to_skip
    (run_dir / "status.json").write_text('{"status":"ALTERED"}', encoding="utf-8")
    tampered = manager.validate_completed(
        experiment_id="managed_test",
        run_id="run_0001",
        config=config,
        source_provenance=provenance,
        additional_table_keys=manifest["additional_table_keys"],
        table_filename_overrides=manifest["table_filename_overrides"],
    )
    assert not tampered.safe_to_skip
    assert any(
        "status.json" in reason and ("size mismatch" in reason or "checksum mismatch" in reason)
        for reason in tampered.reasons
    )

