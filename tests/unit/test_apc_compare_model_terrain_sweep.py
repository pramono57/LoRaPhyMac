from __future__ import annotations

import json

import pandas as pd
import pytest

from loraphymac.config.validation import ConfigurationError
from loraphymac.experiments.apc_compare_model_terrain_sweep.contract import (
    APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_RUN_COUNT,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_RUN_COUNT,
    APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_RUN_COUNT,
    APCCompareModelTerrainSweepContractError,
    CORRECTED_ENERGY_MODEL_VERSION,
    DBPC_CALIBRATION_MANIFEST_SHA256,
    MAIN_W4_PROFILE,
    MAIN_W4_REPORTING_NAMESPACE,
    MAIN_W4_RUN_COUNT,
    PAIRED_ROUTING_PILOT_PROFILE,
    PAIRED_ROUTING_PILOT_RUN_COUNT,
    REVIEWER_R1_5_PROFILE,
    REVIEWER_R1_5_REPORTING_NAMESPACE,
    REVIEWER_R1_5_RUN_COUNT,
    apc_compare_model_terrain_sweep_config_violations,
    routing_reporting_contract,
    validate_apc_compare_model_terrain_sweep_outputs,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.postprocessing import (
    FIGURE_FONT_SIZE,
    FIGURE_SIZE,
    METRICS,
    MODEL_STYLES,
    PLOT_MODEL_ORDER,
    _offered_traffic_hash,
    _plot_metric,
    _scientific_signature,
    _validate_matched_cells,
    _wang_metrics,
    build_paired_differences,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.run import (
    DEFAULT_CONFIG,
    E0_CONFIG,
    FIVE_MODEL_PILOT_CONFIG,
    LEGACY_CONFIG,
    MAIN_W4_CONFIG,
    PAIRED_ROUTING_PILOT_CONFIG,
    REVIEWER_R1_5_CONFIG,
    build_model_specs,
    load_sweep_definition,
    planned_grid,
    run_experiments,
)
from loraphymac.experiments.common.contract_registry import (
    experiment_aggregation_metrics,
    experiment_run_tables,
)
from loraphymac.experiments.common.multiseed_pipeline import (
    aggregate_paper_grade,
    build_experiment_grid,
)
from loraphymac.hardware.hardware import SX1276, SX1302
from loraphymac.runtime.wang_atp_a2only import WangA2PowerReductionController
from loraphymac.simulator import resolve_configuration


pytestmark = pytest.mark.unit


def test_figure_style_matches_apc_terrain_sweep():
    assert FIGURE_FONT_SIZE == 18
    assert FIGURE_SIZE == (8, 5)
    assert PLOT_MODEL_ORDER == (
        "aa",
        "islam_dbpc_802",
        "islam_dbpc_itm6",
        "wang_atp_sf7_a2only",
        "ff",
    )
    assert MODEL_STYLES == {
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
        "wang_atp_sf7_a2only": {
            "color": "tab:green",
            "marker": "^",
            "linestyle": "-.",
        },
        "ff": {"color": "tab:red", "marker": "D", "linestyle": ":"},
    }


@pytest.mark.extended_research
def test_unified_grid_has_exactly_300_unique_matched_cells():
    definition = load_sweep_definition()
    specs = build_model_specs()
    grid = planned_grid()

    assert definition.experiment_profile == APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE
    assert len(grid) == APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_RUN_COUNT
    assert len(
        {(row["model_id"], row["terrain_delta_h"], row["seed"]) for row in grid}
    ) == 300
    assert set(specs) == {
        "ff",
        "aa",
        "islam_dbpc_itm6",
        "wang_atp_sf7_a2only",
    }
    assert all(spec.base_config["energy_accounting_mode"] == "corrected" for spec in specs.values())
    assert all(spec.base_config["node_num"] == 60 for spec in specs.values())
    assert all(spec.base_config["relay_num"] == 2 for spec in specs.values())
    assert all(spec.base_config["node_rx_collision_enabled"] for spec in specs.values())
    assert all(not spec.base_config["frame_default_ack_req"] for spec in specs.values())
    assert all(
        spec.base_config["application_scheduling_mode"]
        == "independent_fifo_v1"
        for spec in specs.values()
    )
    assert definition.dbpc_calibration_manifest_sha256 == DBPC_CALIBRATION_MANIFEST_SHA256
    assert specs["ff"].base_config["node_ptx_mode"] == "fixed"
    assert specs["aa"].base_config["node_ptx_mode"] == "adaptive"
    assert specs["islam_dbpc_itm6"].base_config[
        "tx_power_calibration_margin_db"
    ] == 6
    wang = specs["wang_atp_sf7_a2only"].base_config
    assert wang["node_ptx_mode"] == wang["relay_ptx_mode"] == "fixed"
    assert wang["node_ptx_init"] is None and wang["relay_ptx_init"] is None


@pytest.mark.extended_research
def test_legacy_three_model_grid_and_shared_configuration_are_preserved():
    definition = load_sweep_definition(LEGACY_CONFIG)
    specs = build_model_specs(LEGACY_CONFIG)
    grid = planned_grid(LEGACY_CONFIG)

    assert definition.experiment_profile == APC_COMPARE_MODEL_TERRAIN_SWEEP_PROFILE
    assert len(grid) == APC_COMPARE_MODEL_TERRAIN_SWEEP_RUN_COUNT
    assert set(specs) == {"ff", "aa", "islam_dbpc_itm6"}

    e0_base = json.loads(E0_CONFIG.read_text(encoding="utf-8"))["base_config"]
    shared = dict(specs["ff"].base_config)
    for value in (e0_base, shared):
        for key in (
            "experiment_profile",
            "energy_accounting_mode",
            "node_ptx_mode",
            "relay_ptx_mode",
            "tx_power_policy_id",
            "tx_power_model_id",
        ):
            value.pop(key, None)
    assert shared == e0_base


@pytest.mark.extended_research
def test_five_model_pilot_has_one_matched_cell_per_model_without_execution(
    monkeypatch,
):
    definition = load_sweep_definition(FIVE_MODEL_PILOT_CONFIG)
    specs = build_model_specs(FIVE_MODEL_PILOT_CONFIG)
    grid = planned_grid(FIVE_MODEL_PILOT_CONFIG)

    assert (
        definition.experiment_profile
        == APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_PROFILE
    )
    assert len(grid) == APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PILOT_RUN_COUNT
    assert {(row["terrain_delta_h"], row["seed"]) for row in grid} == {(90, 30)}
    assert set(specs) == {
        "ff",
        "aa",
        "islam_dbpc_802",
        "islam_dbpc_itm6",
        "wang_atp_sf7_a2only",
    }
    assert all(spec.base_config["node_num"] == 100 for spec in specs.values())
    assert all(
        spec.base_config["application_scheduling_mode"] == "independent_fifo_v1"
        for spec in specs.values()
    )
    dbpc_802 = specs["islam_dbpc_802"].base_config
    assert dbpc_802["tx_power_policy_id"] == (
        "islam_algorithm2_ieee80211ah_fixed_sf_ptx_v1"
    )
    assert dbpc_802["tx_power_calibration_margin_db"] == 0
    assert dbpc_802.get("tx_power_calibration_manifest") is None

    monkeypatch.setattr(
        "loraphymac.experiments.apc_compare_model_terrain_sweep.run.run_experiment",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run attempted simulation execution")
        ),
    )
    result = run_experiments(FIVE_MODEL_PILOT_CONFIG, dry_run=True)
    assert result["status"] == "DRY_RUN"
    assert result["run_count"] == 5


@pytest.mark.extended_research
def test_main_w4_has_375_hash_bound_fixed_reference_cells_without_execution(
    monkeypatch,
):
    definition = load_sweep_definition(MAIN_W4_CONFIG)
    specs = build_model_specs(MAIN_W4_CONFIG)
    grid = planned_grid(MAIN_W4_CONFIG)

    assert definition.experiment_profile == MAIN_W4_PROFILE
    assert definition.experiment_dir.name == MAIN_W4_REPORTING_NAMESPACE
    assert len(grid) == MAIN_W4_RUN_COUNT
    assert set(definition.snapshot_bindings) == {30, 36, 42, 48, 56}
    assert all(row["routing_strategy"] == "fixed_reference" for row in grid)
    assert all(row["routing_reference_delta_h"] == 90 for row in grid)
    for seed in definition.seeds:
        hashes = {
            row["routing_snapshot_manifest_sha256"]
            for row in grid
            if row["seed"] == seed
        }
        assert hashes == {definition.snapshot_bindings[seed].manifest_sha256}
    assert all(spec.seed_config_overrides for spec in specs.values())

    monkeypatch.setattr(
        "loraphymac.experiments.apc_compare_model_terrain_sweep.run.run_experiment",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run attempted simulation execution")
        ),
    )
    result = run_experiments(MAIN_W4_CONFIG, dry_run=True)
    assert result["run_count"] == MAIN_W4_RUN_COUNT
    assert result["reporting_namespace"] == MAIN_W4_REPORTING_NAMESPACE


@pytest.mark.extended_research
def test_r1_5_track_is_isolated_and_plans_only_75_additional_runs():
    main = load_sweep_definition(MAIN_W4_CONFIG)
    sensitivity = load_sweep_definition(REVIEWER_R1_5_CONFIG)
    grid = planned_grid(REVIEWER_R1_5_CONFIG)

    assert sensitivity.experiment_profile == REVIEWER_R1_5_PROFILE
    assert sensitivity.experiment_dir.name == REVIEWER_R1_5_REPORTING_NAMESPACE
    assert sensitivity.experiment_dir != main.experiment_dir
    assert sensitivity.fixed_reference_source_experiment_id == (
        MAIN_W4_REPORTING_NAMESPACE
    )
    assert sensitivity.comparison_terrains == (20, 90, 150, 160)
    assert len(grid) == REVIEWER_R1_5_RUN_COUNT
    assert {row["terrain_delta_h"] for row in grid} == {20, 150, 160}
    assert all(row["routing_strategy"] == "terrain_specific" for row in grid)
    assert all(row["routing_snapshot_manifest_sha256"] is None for row in grid)

    main_reporting = routing_reporting_contract(MAIN_W4_PROFILE)
    sensitivity_reporting = routing_reporting_contract(REVIEWER_R1_5_PROFILE)
    assert main_reporting["reporting_namespace"] != sensitivity_reporting[
        "reporting_namespace"
    ]
    assert set(main_reporting["figure_stems"]).isdisjoint(
        sensitivity_reporting["figure_stems"]
    )
    assert set(main_reporting["table_stems"]).isdisjoint(
        sensitivity_reporting["table_stems"]
    )


@pytest.mark.extended_research
def test_paired_routing_development_pilot_has_exact_30_cell_matrix():
    definition = load_sweep_definition(PAIRED_ROUTING_PILOT_CONFIG)
    grid = planned_grid(PAIRED_ROUTING_PILOT_CONFIG)

    assert definition.experiment_profile == PAIRED_ROUTING_PILOT_PROFILE
    assert definition.experiment_dir.name == "w3c_paired_routing_n100_s30_r01"
    assert len(grid) == PAIRED_ROUTING_PILOT_RUN_COUNT
    assert len(
        {
            (
                row["model_id"],
                row["terrain_delta_h"],
                row["seed"],
                row["routing_strategy"],
            )
            for row in grid
        }
    ) == PAIRED_ROUTING_PILOT_RUN_COUNT
    assert {row["terrain_delta_h"] for row in grid} == {20, 90, 160}
    assert {row["seed"] for row in grid} == {30}
    assert {row["routing_strategy"] for row in grid} == {
        "fixed_reference",
        "terrain_specific",
    }


@pytest.mark.extended_research
def test_paired_routing_tracks_keep_workload_inputs_but_distinct_signatures():
    main_spec = build_model_specs(MAIN_W4_CONFIG)["aa"]
    sensitivity_spec = build_model_specs(REVIEWER_R1_5_CONFIG)["aa"]
    main_run = next(
        row
        for row in build_experiment_grid(main_spec)
        if row["seed"] == 30 and row["factors"]["terrain_delta_h"] == 20
    )
    sensitivity_run = next(
        row
        for row in build_experiment_grid(sensitivity_spec)
        if row["seed"] == 30 and row["factors"]["terrain_delta_h"] == 20
    )
    workload_keys = (
        "seed",
        "node_num",
        "node_period",
        "node_jitter",
        "node_payload_size",
        "application_scheduling_mode",
        "application_rng_policy",
        "application_queue_policy",
        "application_horizon_policy",
    )
    assert {
        key: main_run["scientific_config"][key] for key in workload_keys
    } == {
        key: sensitivity_run["scientific_config"][key]
        for key in workload_keys
    }
    assert _scientific_signature(
        {"scientific_config": main_run["scientific_config"]}
    ) != _scientific_signature(
        {"scientific_config": sensitivity_run["scientific_config"]}
    )


@pytest.mark.extended_research
def test_dry_run_is_read_only_and_reports_300(tmp_path, monkeypatch):
    raw = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["output_dir"] = str(tmp_path / "outputs")
    raw["comparison_config"] = str(
        DEFAULT_CONFIG.parent.parent.parent / "apc_compare_model/configs/comparison.json"
    )
    config = tmp_path / "sweep.json"
    config.write_text(json.dumps(raw), encoding="utf-8")

    def simulation_must_not_run(*args, **kwargs):
        raise AssertionError("dry-run attempted simulation execution")

    monkeypatch.setattr(
        "loraphymac.experiments.apc_compare_model_terrain_sweep.run.run_experiment",
        simulation_must_not_run,
    )

    result = run_experiments(config, dry_run=True)

    assert result["status"] == "DRY_RUN"
    assert result["run_count"] == 300
    assert result["unique_cell_count"] == 300
    assert result["experiment_id"] != load_sweep_definition(LEGACY_CONFIG).experiment_id
    assert not (tmp_path / "outputs").exists()


@pytest.mark.extended_research
def test_profile_contract_registers_wang_r20_r22_tables_and_metrics():
    config = dict(build_model_specs()["wang_atp_sf7_a2only"].base_config)
    tables = experiment_run_tables(config)
    metrics = experiment_aggregation_metrics(APC_COMPARE_MODEL_TERRAIN_SWEEP_V2_PROFILE, ())

    assert tables["energy_breakdown_df"] == "energy_breakdown.csv"
    assert tables["final_ptx_distribution_df"] == "final_ptx_distribution.csv"
    assert tables["wang_snr_report_lifecycle_df"] == "wang_snr_report_lifecycle.csv"
    assert tables["wang_sink_decision_df"] == "wang_sink_decisions.csv"
    assert tables["wang_power_command_lifecycle_df"] == (
        "wang_power_command_lifecycle.csv"
    )
    assert tables["application_queue_metrics_df"] == (
        "application_queue_metrics.csv"
    )
    assert set(METRICS).issubset(metrics)


@pytest.mark.extended_research
def test_dbpc_profile_rejects_wrong_margin_and_legacy_energy():
    config = dict(build_model_specs()["islam_dbpc_itm6"].base_config)
    config.update({"terrain_delta_h": 90, "seed": 30})
    assert apc_compare_model_terrain_sweep_config_violations(config) == []

    config["tx_power_calibration_margin_db"] = 0
    config["energy_accounting_mode"] = "legacy"
    violations = apc_compare_model_terrain_sweep_config_violations(config)
    assert any("margin" in violation for violation in violations)
    assert any("energy_accounting_mode" in violation for violation in violations)


@pytest.mark.extended_research
def test_dbpc_802_pilot_contract_is_analytic_fixed_and_margin_zero():
    config = dict(
        build_model_specs(FIVE_MODEL_PILOT_CONFIG)["islam_dbpc_802"].base_config
    )
    config.update({"terrain_delta_h": 90, "seed": 30})

    assert apc_compare_model_terrain_sweep_config_violations(config) == []
    assert config["node_ptx_mode"] == config["relay_ptx_mode"] == "fixed"
    assert config["tx_power_calibration_margin_db"] == 0
    assert config.get("tx_power_calibration_manifest") is None


@pytest.mark.extended_research
def test_wang_configuration_rejects_data_ack_fail_fast():
    config = dict(build_model_specs()["wang_atp_sf7_a2only"].base_config)
    config["frame_default_ack_req"] = True

    with pytest.raises(ConfigurationError, match="no data ACK/ARQ"):
        resolve_configuration(config)


def _valid_wang_contract_fixture():
    config = dict(build_model_specs()["wang_atp_sf7_a2only"].base_config)
    config.update({"terrain_delta_h": 90, "seed": 30})
    summary = {
        "energy_accounting_mode": "corrected",
        "energy_accounting_model_version": CORRECTED_ENERGY_MODEL_VERSION,
        "packet_fate_offered_count": 1,
        "packet_fate_delivered_count": 1,
        "packet_fate_terminal_drop_count": 0,
        "packet_fate_censored_count": 0,
        "packet_fate_conservation_delta": 0,
        "collision_outcome_conservation_delta": 0,
        "total_node_energy_J": 1.0,
        "total_relay_energy_J": 2.0,
        "total_gateway_energy_J": 3.0,
        "total_network_energy_J": 6.0,
        "energy_per_delivered_packet_J": 6.0,
        "apc_total_tx": 0,
        "apc_total_rx": 0,
        "apc_total_drop": 0,
        "ptx_update_count": 1,
        "stage1_status": "PASS",
        "wang_closed_window_count": 1,
        "application_generated_count": 1,
        "application_dequeued_count": 1,
        "application_queue_waiting_at_horizon_count": 0,
        "application_queue_in_service_at_horizon_count": 0,
        "application_queue_outstanding_at_horizon_count": 0,
    }
    empty = pd.DataFrame()
    outputs = {
        "packet_fate_df": empty,
        "topology_parent_by_run_df": empty,
        "apc_feedback_lifecycle_df": empty,
        "energy_breakdown_df": pd.DataFrame(
            [
                {
                    "energy_accounting_mode": "corrected",
                    "energy_accounting_model_version": CORRECTED_ENERGY_MODEL_VERSION,
                }
            ]
        ),
        "final_ptx_distribution_df": empty,
        "ptx_history_df": empty,
        "run_metadata_df": empty,
        "wang_snr_report_lifecycle_df": empty,
        "wang_sink_decision_df": pd.DataFrame(
            [
                {
                    "decision_id": "window-0",
                    "window_index": 0,
                    "status": "decision_generated",
                    "reason": "complete_active_transmitter_coverage",
                    "coverage_fraction": 1.0,
                    "network_reduction_steps": 1,
                }
            ]
        ),
        "wang_power_command_lifecycle_df": empty,
        "application_queue_metrics_df": pd.DataFrame(
            [
                {
                    "node_id": f"N{index}",
                    "scheduling_mode": "independent_fifo_v1",
                    "rng_policy": "sha256_seed_node_v1",
                    "queue_policy": "unbounded_fifo_v1",
                    "horizon_policy": "right_censor_waiting_v1",
                    "generated_count": int(index == 0),
                    "dequeued_count": int(index == 0),
                    "waiting_at_horizon_count": 0,
                    "in_service_at_horizon_count": 0,
                    "outstanding_at_horizon_count": 0,
                }
                for index in range(60)
            ]
        ),
    }
    return config, summary, outputs


@pytest.mark.extended_research
def test_wang_output_contract_accepts_wang_only_physical_control():
    config, summary, outputs = _valid_wang_contract_fixture()

    validate_apc_compare_model_terrain_sweep_outputs(config, summary, outputs)


@pytest.mark.extended_research
def test_wang_output_contract_rejects_genuine_rapc_contamination():
    config, summary, outputs = _valid_wang_contract_fixture()
    summary["apc_total_tx"] = 1

    with pytest.raises(
        APCCompareModelTerrainSweepContractError,
        match="must not produce R-APC feedback",
    ):
        validate_apc_compare_model_terrain_sweep_outputs(config, summary, outputs)


@pytest.mark.extended_research
def test_common_hardware_bounds_match_wang_contract():
    config = build_model_specs()["wang_atp_sf7_a2only"].base_config
    node_ceiling = (
        config["node_max_eirp_dbm"]
        - config["node_ant_gain"]
        + config["node_feeder_loss_db"]
    )
    relay_ceiling = (
        config["relay_max_eirp_dbm"]
        - config["relay_ant_gain"]
        + config["relay_feeder_loss_db"]
    )
    node_values = sorted(value for value in SX1276.tx_current if value <= node_ceiling)
    relay_values = sorted(value for value in SX1302.tx_current if value <= relay_ceiling)

    assert (min(node_values), max(node_values)) == (5, 17)
    assert (min(relay_values), max(relay_values)) == (12, 20)
    node_bounds = WangA2PowerReductionController.bounds_for("node")
    relay_bounds = WangA2PowerReductionController.bounds_for("relay")
    assert (node_bounds.minimum_dbm, node_bounds.maximum_dbm) == (5, 17)
    assert (relay_bounds.minimum_dbm, relay_bounds.maximum_dbm) == (12, 20)


def _synthetic_per_run() -> pd.DataFrame:
    rows = []
    for terrain in range(20, 161, 10):
        for seed_index, seed in enumerate((30, 36, 42, 48, 56)):
            for model_index, model in enumerate(
                ("ff", "aa", "islam_dbpc_itm6", "wang_atp_sf7_a2only")
            ):
                rows.append(
                    {
                        "model_id": model,
                        "model_label": model,
                        "terrain_delta_h": terrain,
                        "seed": seed,
                        "pdr_percent": 90.0 - model_index - seed_index / 10,
                        "energy_per_delivered_packet_J": 0.5 + model_index / 10,
                        "avg_e2e_delay_ms": 200.0 + terrain + model_index,
                        "total_network_energy_J": 2500.0 + model_index,
                    }
                )
    return pd.DataFrame(rows)


def test_paired_differences_and_figure_outputs_are_data_driven(tmp_path):
    per_run = _synthetic_per_run()
    paired = build_paired_differences(per_run)
    statistics = aggregate_paper_grade(
        per_run,
        group_columns=["model_id", "model_label", "terrain_delta_h"],
        metric_columns=METRICS,
        confidence_level=0.95,
    )

    assert len(paired) == 15 * 5 * 6
    legacy_paired = build_paired_differences(
        per_run.loc[per_run["model_id"] != "wang_atp_sf7_a2only"]
    )
    assert len(legacy_paired) == 15 * 5 * 3
    aa_ff = paired.loc[paired["contrast"] == "aa_minus_ff"]
    assert set(aa_ff["pdr_percent"]) == {-1.0}
    outputs = _plot_metric(
        statistics,
        "pdr_percent",
        "Packet delivery ratio (%)",
        tmp_path / "figure_08_pdr_vs_terrain",
    )
    assert {path.suffix for path in outputs} == {".png", ".svg"}
    assert all(path.stat().st_size > 0 for path in outputs)


def _offered_packets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "packet_uid": "UP:N1:0",
                "src_id": "N1",
                "fcnt": 0,
                "birth_time": 61.641251658050784,
                "offered_time": 61.64156674947935,
            },
            {
                "packet_uid": "UP:N2:0",
                "src_id": "N2",
                "fcnt": 0,
                "birth_time": 62.0,
                "offered_time": 62.000315,
            },
        ]
    )


def test_offered_traffic_hash_ignores_only_sub_nanosecond_float_noise():
    baseline = _offered_packets()
    float_noise = baseline.copy()
    float_noise.loc[0, "birth_time"] += 2.842170943040401e-14
    float_noise.loc[0, "offered_time"] += 2.842170943040401e-14

    genuine_time_change = baseline.copy()
    genuine_time_change.loc[0, "birth_time"] += 1e-6
    missing_packet = baseline.iloc[:1].copy()

    assert _offered_traffic_hash(float_noise) == _offered_traffic_hash(baseline)
    assert _offered_traffic_hash(genuine_time_change) != _offered_traffic_hash(baseline)
    assert _offered_traffic_hash(missing_packet) != _offered_traffic_hash(baseline)


def test_cross_model_workload_mismatch_has_explicit_diagnostic():
    rows = []
    for model_id, traffic_hash in (
        ("ff", "same"),
        ("aa", "different"),
        ("islam_dbpc_itm6", "same"),
        ("wang_atp_sf7_a2only", "same"),
    ):
        rows.append(
            {
                "model_id": model_id,
                "terrain_delta_h": 20,
                "seed": 42,
                "run_validation_pass": True,
                "failed_checks": "",
                "scientific_input_signature": "same",
                "topology_hash": "same",
                "routing_hash": "same",
                "offered_traffic_hash": traffic_hash,
            }
        )

    result = _validate_matched_cells(pd.DataFrame(rows))

    assert not result["validation_pass"].any()
    assert set(result["failed_checks"]) == {"cross_model_exogenous_workload_mismatch"}


def test_wang_r20_r22_derived_metrics_and_non_wang_na():
    reports = pd.DataFrame(
        [
            {"event": "generated", "report_id": "R1"},
            {"event": "received", "report_id": "R1"},
            {"event": "lost", "report_id": "R2"},
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                "decision_id": "D0",
                "window_index": 0,
                "status": "decision_generated",
                "reason": "complete_active_transmitter_coverage",
                "coverage_fraction": 1.0,
            },
            {
                "decision_id": "D1",
                "window_index": 1,
                "status": "no_decision",
                "reason": "insufficient_active_transmitter_coverage",
                "coverage_fraction": 0.9,
            },
        ]
    )
    commands = pd.DataFrame(
        [
            {"event": "command_generated", "command_id": "C1"},
            {"event": "command_sent", "command_id": "C1"},
            {"event": "command_applied", "command_id": "C1"},
            {"event": "command_confirmed_by_uplink", "command_id": "C1"},
        ]
    )
    airtime = pd.DataFrame(
        [
            {
                "traffic_class": "control_feedback",
                "entity_type": "gateway",
                "ptx_dbm": 20,
                "total_airtime_s": 1.0,
            }
        ]
    )

    wang = _wang_metrics(
        "wang_atp_sf7_a2only", reports, decisions, commands, airtime
    )
    legacy = _wang_metrics("ff", reports, decisions, commands, airtime)

    assert wang["wang_window_count"] == 2
    assert wang["wang_sink_decision_rate"] == 0.5
    assert wang["wang_commands_causally_confirmed"] == 1
    assert wang["wang_control_tx_airtime_s"] == 1.0
    assert wang["wang_control_tx_energy_J"] > 0
    assert pd.isna(wang["wang_reporting_energy_overhead_J"])
    assert legacy["wang_metrics_applicability"] == "not_applicable"
    assert pd.isna(legacy["wang_commands_generated"])

