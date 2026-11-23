from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from types import MappingProxyType

import pytest

from loraphymac import default_config, resolve_configuration
from loraphymac.config.validation import ConfigurationError
from loraphymac.experiments.apc_closed_loop_verification.contract import (
    APC_CLOSED_LOOP_PROFILE,
)
from loraphymac.experiments.apc_load_sweep.contract import (
    APC_LOAD_SWEEP_FINAL_PROFILE,
    APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE,
    APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_PILOT_PROFILE,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.contract import (
    APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_MODELS,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_MODELS,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
)
from loraphymac.experiments.apc_terrain_sweep.contract import (
    APC_TERRAIN_SWEEP_PROFILE,
    PAPER1_APC_PROFILE,
)
from loraphymac.experiments.common.scientific_output_contract import (
    LEGACY_PERSISTENCE_PROFILE,
    ScientificOutputContract,
    effective_persistence_profile,
    output_contract_for,
    preflight_output_violations,
    required_metrics,
    required_table_keys,
    summary_collection_table_keys,
    validate_conditional_table_semantics,
)
from loraphymac.experiments.common.contract_registry import (
    experiment_run_tables,
    experiment_table_filenames,
)
from loraphymac.experiments.common.multiseed_pipeline import master_table_names_for_profile
from loraphymac.experiments.scalability_relay_bottleneck.r1_7_contract import (
    APC_LOAD_SWEEP_PROFILE,
    R1_7_DUTY_PROFILE,
    R1_7_LOAD_PROFILE,
)


CONTRACT_CONFIGS = (
    {"experiment_profile": APC_CLOSED_LOOP_PROFILE},
    {"experiment_profile": APC_TERRAIN_SWEEP_PROFILE, "terrain_delta_h": 90},
    {"experiment_profile": PAPER1_APC_PROFILE, "terrain_delta_h": 90},
    {"experiment_profile": APC_LOAD_SWEEP_PROFILE},
    {"experiment_profile": R1_7_LOAD_PROFILE},
    {"experiment_profile": R1_7_DUTY_PROFILE},
    {"experiment_profile": APC_LOAD_SWEEP_FINAL_PROFILE},
    {"experiment_profile": APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE},
    {"experiment_profile": APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_PILOT_PROFILE},
    {"experiment_profile": APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE},
    {"experiment_profile": APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE},
    {"experiment_profile": APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE},
)


def test_scientific_output_contract_mapping_defaults_are_readonly_and_isolated():
    required = {
        "contract_id": "fixture_contract",
        "required_metrics": frozenset(),
        "required_additional_tables": frozenset(),
        "minimum_collection_profile": "summary",
        "minimum_persistence_profile": "scientific",
        "producer_modules": (),
        "granularity_categories": (),
        "validator_dependencies": (),
        "postprocessor_dependencies": (),
        "provenance_fields": frozenset(),
    }
    first = ScientificOutputContract(**required)
    second = ScientificOutputContract(**required)

    assert tuple(item.name for item in fields(ScientificOutputContract))[-2:] == (
        "conditional_table_semantics",
        "table_schemas",
    )
    for name in ("conditional_table_semantics", "table_schemas"):
        first_mapping = getattr(first, name)
        second_mapping = getattr(second, name)
        assert isinstance(first_mapping, MappingProxyType)
        assert dict(first_mapping) == {}
        assert first_mapping is not second_mapping
        with pytest.raises(TypeError):
            first_mapping["unexpected"] = ()


@pytest.mark.parametrize("profile", ("summary", "detailed"))
@pytest.mark.parametrize("persistence", ("scientific", "diagnostic"))
@pytest.mark.parametrize("base", CONTRACT_CONFIGS)
def test_supported_collection_and_persistence_profiles_are_compatible(profile, persistence, base):
    config = {**base, "output_profile": profile, "persistence_profile": persistence}
    errors = preflight_output_violations(config)
    if profile == "summary" and persistence == "diagnostic":
        assert len(errors) == 1
        assert "requires output_profile='detailed'" in errors[0]
    else:
        assert errors == []


@pytest.mark.parametrize("base", CONTRACT_CONFIGS)
def test_core_persistence_is_rejected_with_missing_evidence(base):
    config = {**base, "output_profile": "summary", "persistence_profile": "core"}
    errors = preflight_output_violations(config)
    assert len(errors) == 1
    assert "requires persistence_profile='scientific'" in errors[0]
    assert "packet_fate" in errors[0]


@pytest.mark.parametrize("base", CONTRACT_CONFIGS)
def test_omitted_persistence_preserves_legacy_compatibility_without_mutation(base):
    config = {**base, "output_profile": "summary"}
    before = deepcopy(config)
    assert effective_persistence_profile(config) == LEGACY_PERSISTENCE_PROFILE
    assert preflight_output_violations(config) == []
    assert config == before


def test_contract_tables_and_metrics_are_registered_and_producible():
    producer_keys = summary_collection_table_keys()
    for config in CONTRACT_CONFIGS:
        contract = output_contract_for(config)
        assert contract is not None
        assert contract.minimum_collection_profile == "summary"
        assert contract.minimum_persistence_profile == "scientific"
        assert required_table_keys(config)
        assert required_table_keys(config) <= producer_keys
        assert required_metrics(config)
        assert contract.producer_modules
        assert contract.granularity_categories
        assert contract.validator_dependencies
        assert contract.postprocessor_dependencies
        assert contract.provenance_fields


def test_invalid_persistence_profile_is_rejected_for_all_experiments():
    config = {"persistence_profile": "compact"}
    assert "persistence_profile" in preflight_output_violations(config)[0]
    with pytest.raises(ConfigurationError, match="persistence_profile must be"):
        resolve_configuration(config)


def test_optional_persistence_key_is_accepted_without_changing_legacy_default():
    baseline = default_config()
    assert "persistence_profile" not in baseline
    resolved = resolve_configuration({"persistence_profile": "diagnostic"})
    assert resolved["persistence_profile"] == "diagnostic"
    legacy = resolve_configuration()
    assert "persistence_profile" not in legacy


def test_preflight_does_not_mutate_resolved_config():
    config = {
        "experiment_profile": APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE,
        "output_profile": "summary",
        "persistence_profile": "scientific",
    }
    before = deepcopy(config)
    assert preflight_output_violations(config) == []
    assert config == before


def test_historical_compare_model_profile_has_separate_compatibility_contract():
    config = {"experiment_profile": APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE}
    contract = output_contract_for(config)
    keys = required_table_keys(config)

    assert contract is not None
    assert contract.contract_id == "apc_compare_model_terrain_sweep_v1_compat"
    assert {"ptx_history_df", "final_ptx_distribution_df", "packet_fate_df"} <= keys
    # V1 did not persist these newer metrics; do not alias it to terrain-sweep.
    assert "node_tx_duty_cycle_df" not in keys
    assert "channel_occupancy_df" not in keys


def test_v2_profiles_have_explicit_model_registry_and_scientific_contract():
    assert APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_MODELS == (
        "ff", "aa", "islam_dbpc_itm6", "wang_atp_sf7_a2only"
    )
    assert APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_MODELS == (
        "ff", "aa", "islam_dbpc_802", "islam_dbpc_itm6", "wang_atp_sf7_a2only"
    )
    for profile in (
        APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
        APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE,
    ):
        config = {"experiment_profile": profile, "output_profile": "summary"}
        contract = output_contract_for(config)
        assert contract is not None
        assert contract.contract_id == "apc_compare_model_terrain_sweep_v2_scientific"
        assert {
            "node_tx_duty_cycle_df",
            "channel_occupancy_df",
            "application_queue_metrics_df",
            "wang_snr_report_lifecycle_df",
            "wang_power_command_lifecycle_df",
        } <= required_table_keys(config)
        assert required_table_keys(config) <= summary_collection_table_keys()
        assert preflight_output_violations(
            {**config, "persistence_profile": "scientific"}
        ) == []
        diagnostic_errors = preflight_output_violations(
            {**config, "persistence_profile": "diagnostic"}
        )
        assert len(diagnostic_errors) == 1
        assert "requires output_profile='detailed'" in diagnostic_errors[0]
        assert preflight_output_violations(
            {**config, "output_profile": "detailed", "persistence_profile": "diagnostic"}
        ) == []

    wang_config = {
        "experiment_profile": APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
        "tx_power_model_id": "wang_atp_sf7_a2only",
    }
    ff_config = {**wang_config, "tx_power_model_id": "ff"}
    assert "wang_sink_decision_df" in required_table_keys(wang_config)
    assert "wang_sink_decision_df" not in required_table_keys(ff_config)
    v2_filenames = experiment_table_filenames(APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE)
    assert {
        "node_tx_duty_cycle.csv",
        "channel_occupancy.csv",
        "hop_summary.csv",
        "frame_hop_event_summary.csv",
        "frame_drop_reason_by_hop.csv",
        "tx_queue_waiting_summary.csv",
        "selected_next_hop_links.csv",
        "relay_chain_links.csv",
        "wang_sink_decisions.csv",
    } <= v2_filenames
    assert {
        "per_run_node_tx_duty_cycle.csv",
        "per_run_channel_occupancy.csv",
    } <= set(master_table_names_for_profile(APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE).values())
    assert "wang_sink_decision_df" in experiment_run_tables(wang_config)
    v2_contract = output_contract_for(ff_config)
    assert v2_contract.conditional_table_semantics["wang_sink_decision_df"]["ff"] == (
        "not_applicable_empty"
    )
    assert v2_contract.conditional_table_semantics["wang_sink_decision_df"][
        "wang_atp_sf7_a2only"
    ] == "required_non_empty"


def test_v2_empty_table_semantics_distinguish_n_a_and_zero_event_schema():
    import pandas as pd

    from loraphymac.metrics.apc_metrics import PTX_HISTORY_COLUMNS
    from loraphymac.metrics.instrumentation_metrics import (
        APC_FEEDBACK_LIFECYCLE_COLUMNS,
        WANG_POWER_COMMAND_LIFECYCLE_COLUMNS,
        WANG_SNR_REPORT_LIFECYCLE_COLUMNS,
    )

    config = {
        "experiment_profile": APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
        "tx_power_model_id": "ff",
    }
    tables = {
        "application_queue_metrics_df": pd.DataFrame({"node_id": [1]}),
        "ptx_history_df": pd.DataFrame(columns=PTX_HISTORY_COLUMNS),
        "apc_feedback_lifecycle_df": pd.DataFrame(
            columns=APC_FEEDBACK_LIFECYCLE_COLUMNS
        ),
        "wang_snr_report_lifecycle_df": pd.DataFrame(
            columns=WANG_SNR_REPORT_LIFECYCLE_COLUMNS
        ),
        "wang_power_command_lifecycle_df": pd.DataFrame(
            columns=WANG_POWER_COMMAND_LIFECYCLE_COLUMNS
        ),
    }
    assert validate_conditional_table_semantics(config, tables) == []

    tables["apc_feedback_lifecycle_df"] = pd.DataFrame({"event": ["unexpected"]})
    errors = validate_conditional_table_semantics(config, tables)
    assert any("must be empty because it is not applicable" in error for error in errors)


def test_v2_wang_lifecycle_requires_events_but_command_table_allows_schema_empty():
    import pandas as pd

    from loraphymac.metrics.instrumentation_metrics import (
        WANG_POWER_COMMAND_LIFECYCLE_COLUMNS,
        WANG_SNR_REPORT_LIFECYCLE_COLUMNS,
    )

    config = {
        "experiment_profile": APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
        "tx_power_model_id": "wang_atp_sf7_a2only",
    }
    tables = {
        "wang_snr_report_lifecycle_df": pd.DataFrame(
            columns=WANG_SNR_REPORT_LIFECYCLE_COLUMNS
        ),
        "wang_power_command_lifecycle_df": pd.DataFrame(
            columns=WANG_POWER_COMMAND_LIFECYCLE_COLUMNS
        ),
    }
    errors = validate_conditional_table_semantics(config, tables)
    assert errors == ["wang_snr_report_lifecycle_df must contain rows for model wang_atp_sf7_a2only"]
    tables["wang_snr_report_lifecycle_df"] = pd.DataFrame(
        [{column: "received" for column in WANG_SNR_REPORT_LIFECYCLE_COLUMNS}],
        columns=WANG_SNR_REPORT_LIFECYCLE_COLUMNS,
    )
    assert validate_conditional_table_semantics(config, tables) == []


def test_apc_load_sweep_declares_exact_empty_event_schemas_and_na_semantics():
    import pandas as pd

    from loraphymac.metrics.apc_metrics import PTX_HISTORY_COLUMNS
    from loraphymac.metrics.instrumentation_metrics import APC_FEEDBACK_LIFECYCLE_COLUMNS

    config = {
        "experiment_profile": APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE,
        "node_ptx_mode": "fixed",
        "relay_ptx_mode": "fixed",
    }
    tables = {
        "ptx_history_df": pd.DataFrame(columns=PTX_HISTORY_COLUMNS),
        "apc_feedback_lifecycle_df": pd.DataFrame(
            columns=APC_FEEDBACK_LIFECYCLE_COLUMNS
        ),
    }
    assert validate_conditional_table_semantics(config, tables) == []
    assert (
        output_contract_for(config).table_schemas["ptx_history_df"]
        == tuple(PTX_HISTORY_COLUMNS)
    )
    tables["packet_fate_df"] = pd.DataFrame({"packet_uid": ["p1"]})
    assert any(
        "packet_fate_df schema" in error
        for error in validate_conditional_table_semantics(config, tables)
    )
    tables["apc_feedback_lifecycle_df"] = pd.DataFrame({"event": ["unexpected"]})
    assert any(
        "must be empty because it is not applicable" in error
        for error in validate_conditional_table_semantics(config, tables)
    )
