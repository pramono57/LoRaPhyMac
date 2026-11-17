from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import loraphymac.experiments.apc_load_sweep.postprocessing as postprocessing
from loraphymac.experiments.apc_load_sweep.postprocessing import (
    COMBINED_SCENARIOS,
    IncompleteDatasetError,
    MODEL_ORDER,
    SEEDS,
    _paired_comparison_summary,
    _summary_statistics,
    _validate_cross_cohort_configs,
    _validate_source_compatibility,
    validate_pairs,
    validate_combined_dataset,
)
from loraphymac.experiments.apc_load_sweep.contract import (
    APC_LOAD_SWEEP_SCENARIOS,
    APC_LOAD_SWEEP_FULL_FACTORIAL_SCENARIOS,
    APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_PILOT_PROFILE,
    APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE,
    validate_apc_load_sweep_spec,
    validate_apc_load_sweep_full_factorial_spec,
)
from loraphymac.experiments.apc_load_sweep.full_factorial_postprocessing import (
    EXPERIMENT_IDS as FULL_FACTORIAL_EXPERIMENT_IDS,
    _paired as full_factorial_paired_statistics,
    _energy_per_delivered_packet,
    _stats as full_factorial_statistics,
    expected_grid as expected_full_factorial_grid,
    validate_full_factorial_rows,
    IncompleteFullFactorialError,
)
from loraphymac.experiments.apc_load_sweep.postprocessing import _plot_metric as plot_load_metric
from loraphymac.experiments.apc_load_sweep.profile_acceptance_pilot import (
    EQUIVALENCE_TABLES,
    PILOT_DIR,
    SCENARIOS as PROFILE_PILOT_SCENARIOS,
    _pilot_run_id,
    _scientific_config_hash,
    _table_hash,
    validate_pilot_results,
)
from loraphymac.experiments.apc_load_sweep.full_factorial_postprocessing import (
    ARTIFACT_ROOT as FULL_FACTORIAL_ARTIFACT_ROOT,
    EXPERIMENT_IDS as FULL_FACTORIAL_POSTPROCESSING_IDS,
)
from loraphymac.experiments.common.multiseed_pipeline import (
    build_experiment_grid,
    load_experiment_spec,
)
from loraphymac.experiments.common.scientific_output_contract import (
    preflight_output_violations,
)
from loraphymac.config.defaults import build_config
from loraphymac.config.validation import ConfigurationError
from loraphymac.simulator import resolve_configuration


CONFIG_DIR = (
    Path(__file__).resolve().parents[2]
    / "src/loraphymac/experiments/apc_load_sweep/configs"
)


def test_apc_load_sweep_configs_plan_exactly_45_runs_per_model():
    ff = load_experiment_spec(CONFIG_DIR / "ff_v1.json")
    aa = load_experiment_spec(CONFIG_DIR / "aa_v1.json")
    ff_runs, aa_runs = build_experiment_grid(ff), build_experiment_grid(aa)

    assert len(ff_runs) == len(aa_runs) == 45
    assert ff.experiment_id == "apc_load_sweep_ff_r01"
    assert aa.experiment_id == "apc_load_sweep_aa_r01"
    assert ff.output_dir == aa.output_dir
    assert ff.run_annotations["model_id"] == "ff"
    assert aa.run_annotations["model_id"] == "aa"
    assert ff.base_config["node_ptx_mode"] == ff.base_config["relay_ptx_mode"] == "fixed"
    assert aa.base_config["node_ptx_mode"] == aa.base_config["relay_ptx_mode"] == "adaptive"
    assert ff.base_config["energy_accounting_mode"] == aa.base_config["energy_accounting_mode"] == "corrected"
    assert ff.base_config["application_rng_policy"] == aa.base_config["application_rng_policy"]


def test_apc_load_sweep_grid_has_exact_cells_and_common_schedule_config():
    ff = load_experiment_spec(CONFIG_DIR / "ff_v1.json")
    aa = load_experiment_spec(CONFIG_DIR / "aa_v1.json")
    ff_runs, aa_runs = build_experiment_grid(ff), build_experiment_grid(aa)
    expected = {
        (nodes, period, seed)
        for nodes in (50, 100, 200)
        for period in (60, 30, 15)
        for seed in (30, 36, 42, 48, 56)
    }
    actual = {
        (r["factors"]["node_num"], r["factors"]["node_period"], r["seed"])
        for r in ff_runs
    }
    assert actual == expected
    for ff_run, aa_run in zip(ff_runs, aa_runs):
        assert ff_run["seed"] == aa_run["seed"]
        assert ff_run["factors"] == aa_run["factors"]
        for key in ff_run["scientific_config"]:
            if key not in {"node_ptx_mode", "relay_ptx_mode"}:
                assert ff_run["scientific_config"][key] == aa_run["scientific_config"][key]


def test_load_statistics_use_seed_sample_sd_and_student_t_ci():
    frame = pd.DataFrame(
        {"model_id": ["aa"] * 5, "node_period": [30] * 5,
         "node_num": [100] * 5, "pdr_percent": [70, 71, 72, 73, 74]}
    )
    result = _summary_statistics(frame, ["model_id", "node_period", "node_num"], "pdr_percent")
    row = result.iloc[0]
    assert row["n"] == 5
    assert row["mean"] == pytest.approx(72.0)
    assert row["sample_sd"] == pytest.approx(frame["pdr_percent"].std(ddof=1))
    assert row["ci95_lower"] < 72 < row["ci95_upper"]


def test_full_factorial_plot_filters_metric_and_draws_six_four_point_series(
    tmp_path, monkeypatch
):
    rows = []
    expected = {}
    for model_index, model in enumerate(("aa", "ff")):
        for period in (30, 60, 90):
            for node_num in (50, 100, 150, 200):
                pdr = 70.0 + model_index + period / 100 + node_num / 1000
                energy = 0.4 + model_index / 10 + period / 1000 + node_num / 10000
                expected[(model, period, node_num, "pdr_percent")] = pdr
                expected[(model, period, node_num, "energy_per_delivered_packet_J")] = energy
                for metric, mean in (
                    ("pdr_percent", pdr),
                    ("energy_per_delivered_packet_J", energy),
                    ("total_network_energy_J", 3000.0 + node_num),
                ):
                    rows.append({
                        "model_id": model, "node_period": period, "node_num": node_num,
                        "metric": metric, "n": 5, "mean": mean,
                        "ci95_lower": mean - 0.2, "ci95_upper": mean + 0.2,
                    })
    stats = pd.DataFrame(rows)
    saved_figures = []
    real_save = postprocessing._save_figure

    def capture_and_save(figure, base):
        saved_figures.append(figure)
        return real_save(figure, base)

    monkeypatch.setattr(postprocessing, "_save_figure", capture_and_save)
    for metric in ("pdr_percent", "energy_per_delivered_packet_J"):
        paths = plot_load_metric(stats, metric, metric, tmp_path / metric)
        assert len(paths) == 2
        assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
        assert paths[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert b"<svg" in paths[1].read_bytes()
        axis = saved_figures[-1].axes[0]
        assert list(saved_figures[-1].get_size_inches()) == [8.0, 5.0]
        assert axis.xaxis.label.get_size() == 18
        assert axis.yaxis.label.get_size() == 18
        assert axis.xaxis.label.get_family() == ["Times New Roman"]
        assert all(label.get_size() == 18 for label in axis.get_xticklabels())
        legend = axis.get_legend()
        assert all(text.get_fontsize() == 18 for text in legend.get_texts())
        expected_location = 3 if metric == "pdr_percent" else 1
        assert legend._loc == expected_location
        assert legend._ncols == 2
        assert legend.get_frame_on()
        assert [text.get_text() for text in legend.get_texts()] == [
            "30 s (AA)", "60 s (AA)", "90 s (AA)",
            "30 s (FF)", "60 s (FF)", "90 s (FF)",
        ]
        containers = list(axis.containers)
        assert len(containers) == 6
        assert sorted(container.get_label() for container in containers) == sorted(
            f"{period} s ({model.upper()})"
            for period in (30, 60, 90) for model in ("aa", "ff")
        )
        for container in containers:
            line = container.lines[0]
            model = "aa" if "AA" in container.get_label() else "ff"
            period = int(container.get_label().split(" ")[0])
            assert list(line.get_xdata()) == [50, 100, 150, 200]
            assert line.get_marker() == ("o" if model == "aa" else "s")
            assert line.get_linewidth() == 2.0
            assert line.get_markersize() == 7
            assert list(line.get_ydata()) == [
                expected[(model, period, node, metric)] for node in (50, 100, 150, 200)
            ]
        if metric == "pdr_percent":
            assert axis.get_ylim() == (55.0, 90.0)


def test_full_factorial_plot_rejects_invalid_pdr_or_missing_seed_replicates(tmp_path):
    rows = [
        {"model_id": model, "node_period": period, "node_num": node,
         "metric": "pdr_percent", "n": 5, "mean": 75.0,
         "ci95_lower": 74.0, "ci95_upper": 76.0}
        for model in ("aa", "ff") for period in (30, 60, 90)
        for node in (50, 100, 150, 200)
    ]
    invalid = pd.DataFrame(rows)
    invalid.loc[0, "mean"] = 100.1
    with pytest.raises(ValueError, match=r"within \[0, 100\]"):
        plot_load_metric(invalid, "pdr_percent", "PDR", tmp_path / "invalid_pdr")

    incomplete = pd.DataFrame(rows)
    incomplete.loc[0, "n"] = 4
    with pytest.raises(ValueError, match="five seed replicates"):
        plot_load_metric(incomplete, "pdr_percent", "PDR", tmp_path / "invalid_n")


def test_final_configs_plan_60_unique_runs_in_six_cells_without_changing_v1():
    names = (
        "ff_wide_nodes_90s.json", "ff_node150_periods.json",
        "aa_wide_nodes_90s.json", "aa_node150_periods.json",
    )
    planned = []
    for name in names:
        spec = load_experiment_spec(CONFIG_DIR / name)
        validate_apc_load_sweep_spec(spec)
        runs = build_experiment_grid(spec)
        assert len(runs) == 15
        planned.extend((spec, run) for run in runs)

    cells = {
        (run["factors"]["node_num"], run["factors"]["node_period"])
        for _, run in planned
    }
    keys = [
        (spec.run_annotations["model_id"], *run["factors"].values(), run["seed"])
        for spec, run in planned
    ]
    ids = [run["run_id"] for _, run in planned]
    assert len(planned) == len(keys) == len(ids) == 60
    assert len(set(keys)) == len(set(ids)) == 60
    assert cells == set(APC_LOAD_SWEEP_SCENARIOS)
    assert {spec.run_annotations["model_id"] for spec, _ in planned} == {"ff", "aa"}
    assert all(spec.output_dir == load_experiment_spec(CONFIG_DIR / "ff_v1.json").output_dir for spec, _ in planned)

    # Historical configurations remain the original 45-run grids.
    for name, model in (("ff_v1.json", "ff"), ("aa_v1.json", "aa")):
        old = load_experiment_spec(CONFIG_DIR / name)
        assert old.experiment_id == f"apc_load_sweep_{model}_r01"
        assert len(build_experiment_grid(old)) == 45


def test_full_factorial_production_configs_use_csv_scientific_persistence():
    specs = [
        load_experiment_spec(CONFIG_DIR / f"{model}_full_factorial_v1.json")
        for model in ("ff", "aa")
    ]
    planned = []
    for spec in specs:
        validate_apc_load_sweep_full_factorial_spec(spec)
        assert spec.base_config["output_profile"] == "summary"
        assert spec.base_config["persistence_profile"] == "scientific"
        assert spec.base_config["gzip_scientific_tables"] is False
        assert spec.master_table_mode == "aggregate_inputs"
        assert spec.scientific_master_policy is None
        assert len(build_experiment_grid(spec)) == 60
        planned.extend((spec.run_annotations["model_id"], run) for run in build_experiment_grid(spec))

    assert len(planned) == 120
    assert len(
        {
            (model, run["factors"]["node_num"], run["factors"]["node_period"], run["seed"])
            for model, run in planned
        }
    ) == 120
    assert {
        (nodes, period)
        for _, run in planned
        for nodes, period in [
            (run["factors"]["node_num"], run["factors"]["node_period"])
        ]
    } == {
        (nodes, period)
        for nodes in (50, 100, 150, 200)
        for period in (30, 60, 90)
    }
    for _, run in planned:
        assert not preflight_output_violations(run["scientific_config"])


def test_paired_summary_uses_aa_minus_ff_by_seed():
    frame = pd.DataFrame(
        {
            "node_num": [50] * 10,
            "node_period": [90] * 10,
            "seed": [30, 30, 36, 36, 42, 42, 48, 48, 56, 56],
            "model_id": ["ff", "aa"] * 5,
            "pdr_percent": [70, 71, 72, 74, 74, 77, 76, 80, 78, 83],
            "total_network_energy_J": [100, 98, 110, 107, 120, 116, 130, 125, 140, 134],
            "energy_per_delivered_packet_J": [1.0, .9, 1.1, 1.0, 1.2, 1.1, 1.3, 1.2, 1.4, 1.3],
            "avg_e2e_delay_ms": [10, 9, 11, 10, 12, 11, 13, 12, 14, 13],
        }
    )
    summary = _paired_comparison_summary(frame)
    pdr = summary.loc[summary["metric"].eq("PDR (percentage points)")].iloc[0]
    assert pdr["difference_direction"] == "AA - FF"
    assert pdr["n_paired_seeds"] == 5
    assert pdr["mean_difference"] == pytest.approx(3.0)


def test_final_pair_validator_accepts_exactly_30_matched_cells(monkeypatch):
    rows = []
    for model, experiment_id in (("ff", "ff_batch"), ("aa", "aa_batch")):
        for nodes, period in COMBINED_SCENARIOS:
            for seed in SEEDS:
                row = {
                    "model_id": model, "experiment_id": experiment_id,
                    "run_id": f"{model}-{nodes}-{period}-{seed}",
                    "node_num": nodes, "node_period": period, "seed": seed,
                    "experiment_profile": "profile",
                    "terrain_delta_h": 90,
                    "topology_sha256": "same-topology",
                    "selected_parent_routing_sha256": "same-routing",
                }
                row.update({column: "same" for column in postprocessing.COMMON_CONFIG_COLUMNS})
                rows.append(row)
    schedule = pd.DataFrame(
        [{"packet_uid": "p1", "src_id": "N1", "fcnt": 1,
          "birth_time": 0.0, "offered_time": 1.0}]
    )
    monkeypatch.setattr(postprocessing, "_schedule_frame", lambda *_args: schedule.copy())
    monkeypatch.setattr(
        postprocessing,
        "_read_resolved_simulator_config",
        lambda experiment_id, run_id: {
            "node_ptx_mode": "fixed" if experiment_id == "ff_batch" else "adaptive",
            "relay_ptx_mode": "fixed" if experiment_id == "ff_batch" else "adaptive",
            "run_id": run_id,
            "frame_default_ack_req": False,
        },
    )

    result = validate_pairs(pd.DataFrame(rows))
    assert len(result) == 60
    assert result["pair_status"].eq("PASS").all()


def _combined_fixture():
    rows = []
    for model in MODEL_ORDER:
        for nodes, period in COMBINED_SCENARIOS:
            cohort = "historical" if (nodes, period) in postprocessing.HISTORICAL_SCENARIOS else "new"
            for seed in SEEDS:
                rows.append(
                    {
                        "run_id": f"{cohort}-{model}-{nodes}-{period}-{seed}",
                        "experiment_id": f"{cohort}-{model}",
                        "experiment_profile": f"{cohort}_profile",
                        "model_id": model,
                        "node_num": nodes,
                        "node_period": period,
                        "seed": seed,
                        "terrain_delta_h": 90,
                        "source_commit": None,
                        "source_fingerprint": f"fp-{cohort}",
                        "config_hash": f"cfg-{model}-{nodes}-{period}-{seed}",
                        "source_cohort": cohort,
                        "stage1_status": "PASS",
                    }
                )
    return pd.DataFrame(rows)


def test_combined_dataset_is_120_runs_12_cells_60_pairs_and_keeps_provenance():
    frame = _combined_fixture()
    validate_combined_dataset(frame)
    assert len(frame) == 120
    assert frame["source_cohort"].value_counts().to_dict() == {"historical": 60, "new": 60}
    assert frame.groupby(["node_num", "node_period"]).ngroups == 12
    assert frame.groupby(["node_num", "node_period", "seed"]).ngroups == 60
    assert set(frame["model_id"]) == {"ff", "aa"}
    assert {"run_id", "experiment_id", "config_hash", "source_fingerprint"} <= set(frame)


def test_combined_validator_rejects_missing_run_seed_or_model():
    frame = _combined_fixture()
    with pytest.raises(IncompleteDatasetError, match="missing=1"):
        validate_combined_dataset(frame.iloc[1:].copy())
    no_seed = frame.loc[~((frame.node_num == 50) & (frame.node_period == 30) & (frame.seed == 56))]
    with pytest.raises(IncompleteDatasetError):
        validate_combined_dataset(no_seed)
    no_model = frame.loc[frame.model_id.eq("ff")]
    with pytest.raises(IncompleteDatasetError):
        validate_combined_dataset(no_model)


def test_combined_validator_rejects_duplicate_keys_and_15_second_rows():
    frame = _combined_fixture()
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_combined_dataset(duplicate)
    frame.loc[frame.index[0], "node_period"] = 15
    with pytest.raises(ValueError, match="15-second"):
        validate_combined_dataset(frame)


def test_full_factorial_configs_plan_exactly_120_unique_summary_runs():
    names = ("ff_full_factorial_v1.json", "aa_full_factorial_v1.json")
    planned = []
    for name, model in zip(names, ("ff", "aa"), strict=True):
        spec = load_experiment_spec(CONFIG_DIR / name)
        validate_apc_load_sweep_full_factorial_spec(spec)
        runs = build_experiment_grid(spec)
        assert len(runs) == 60
        assert spec.run_annotations["model_id"] == model
        assert spec.base_config["output_profile"] == "summary"
        assert spec.base_config["energy_accounting_mode"] == "corrected"
        assert spec.base_config["experiment_profile"] == APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE
        assert spec.master_table_mode == "aggregate_inputs"
        planned.extend((model, run) for run in runs)
    keys = [(model, run["factors"]["node_num"], run["factors"]["node_period"], run["seed"])
            for model, run in planned]
    run_ids = [run["run_id"] for _, run in planned]
    assert len(keys) == len(set(keys)) == 120
    assert set(keys) == expected_full_factorial_grid()
    assert len(run_ids) == len(set(run_ids)) == 120
    assert {(n, p) for _, n, p, _ in keys} == set(APC_LOAD_SWEEP_FULL_FACTORIAL_SCENARIOS)
    assert not any(period == 15 for _, _, period, _ in keys)
    assert {spec.output_dir for spec in [load_experiment_spec(CONFIG_DIR / n) for n in names]} == {
        load_experiment_spec(CONFIG_DIR / names[0]).output_dir
    }
    assert all("full_factorial_v1" in eid for eid in FULL_FACTORIAL_EXPERIMENT_IDS.values())
    assert load_experiment_spec(CONFIG_DIR / names[0]).output_dir.name == "full_factorial_v1"


def _profile_pilot_config(model: str, nodes: int, period: int, profile: str) -> dict:
    spec = load_experiment_spec(CONFIG_DIR / f"{model}_full_factorial_v1.json")
    run = next(
        item for item in build_experiment_grid(spec)
        if item["seed"] == 30
        and item["factors"] == {"node_num": nodes, "node_period": period}
    )
    config = dict(run["scientific_config"])
    config.update({
        "experiment_profile": APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_PILOT_PROFILE,
        "output_profile": profile,
        "run_id": _pilot_run_id(model, nodes, period, 30, profile),
        "output_dir": str(PILOT_DIR / _pilot_run_id(model, nodes, period, 30, profile)),
    })
    return config


def test_full_factorial_contract_still_requires_summary():
    config = _profile_pilot_config("ff", 50, 90, "summary")
    config["experiment_profile"] = APC_LOAD_SWEEP_FULL_FACTORIAL_PROFILE
    resolve_configuration(build_config(config))

    config["output_profile"] = "detailed"
    with pytest.raises(ConfigurationError, match="output_profile must be summary"):
        resolve_configuration(build_config(config))


@pytest.mark.parametrize("output_profile", ["summary", "detailed"])
def test_acceptance_pilot_contract_accepts_both_output_profiles(output_profile):
    for model in ("ff", "aa"):
        for nodes, period in PROFILE_PILOT_SCENARIOS:
            config = _profile_pilot_config(model, nodes, period, output_profile)
            resolved = resolve_configuration(build_config(config))
            assert resolved["experiment_profile"] == APC_LOAD_SWEEP_PROFILE_ACCEPTANCE_PILOT_PROFILE
            assert resolved["output_profile"] == output_profile


def test_profile_pilot_pairs_have_same_scientific_config_and_isolated_ids():
    run_ids = set()
    for model in ("ff", "aa"):
        for nodes, period in PROFILE_PILOT_SCENARIOS:
            summary = _profile_pilot_config(model, nodes, period, "summary")
            detailed = _profile_pilot_config(model, nodes, period, "detailed")
            assert _scientific_config_hash(summary) == _scientific_config_hash(detailed)
            assert summary["seed"] == detailed["seed"] == 30
            assert summary["output_dir"] != detailed["output_dir"]
            run_ids.update((summary["run_id"], detailed["run_id"]))
    assert len(run_ids) == 12
    assert PILOT_DIR.name == "output_profile_acceptance_v3"
    assert all(run_id.startswith("profile_acceptance_v3_") for run_id in run_ids)
    assert PILOT_DIR != FULL_FACTORIAL_ARTIFACT_ROOT
    assert not any(
        run_id.startswith(tuple(FULL_FACTORIAL_POSTPROCESSING_IDS.values()))
        for run_id in run_ids
    )
    changed_physics = _profile_pilot_config("ff", 50, 90, "summary")
    changed_physics["area_x"] += 1
    assert _scientific_config_hash(changed_physics) != _scientific_config_hash(
        _profile_pilot_config("ff", 50, 90, "summary")
    )


def test_profile_pilot_table_hash_ignores_only_unique_run_id():
    summary = pd.DataFrame({
        "run_id": ["summary-id"], "feedback_id": ["summary-id:apc:000001"],
        "pdr_percent": [95.0],
    })
    detailed = pd.DataFrame({
        "run_id": ["detailed-id"], "feedback_id": ["detailed-id:apc:000001"],
        "pdr_percent": [95.0],
    })
    assert _table_hash(summary, "summary-id") == _table_hash(detailed, "detailed-id")
    assert _table_hash(summary.assign(pdr_percent=94.0), "summary-id") != _table_hash(
        summary, "summary-id"
    )


def test_full_factorial_reporter_fails_fast_on_incomplete_or_duplicate_rows():
    rows = [
        {"model_id": model, "node_num": nodes, "node_period": period,
         "seed": seed, "run_id": f"{model}-{nodes}-{period}-{seed}",
         "stage1_status": "PASS"}
        for model, nodes, period, seed in expected_full_factorial_grid()
    ]
    frame = pd.DataFrame(rows)
    validate_full_factorial_rows(frame)
    with pytest.raises(IncompleteFullFactorialError, match="Expected 120"):
        validate_full_factorial_rows(frame.iloc[:-1])
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_full_factorial_rows(duplicate)
    with pytest.raises(ValueError, match="15-second"):
        validate_full_factorial_rows(frame.assign(node_period=15))


def test_profile_pilot_validator_requires_12_matched_runs_and_exact_table_hashes():
    rows = []
    for model in ("ff", "aa"):
        for nodes, period in PROFILE_PILOT_SCENARIOS:
            common = {name: f"{name}-{model}-{nodes}-{period}"
                      for name in ("offered_schedule_sha256", "topology_sha256",
                                   "routing_sha256", "physical_config_sha256")}
            table_hashes = {table: f"{table}-{nodes}-{period}" for table in EQUIVALENCE_TABLES}
            for profile in ("summary", "detailed"):
                rows.append({"model_id": model, "node_num": nodes,
                             "node_period": period, "output_profile": profile,
                             **common, "table_sha256": table_hashes,
                             "runtime_s": 1.0, "output_bytes": 100,
                             "file_count": 2, "peak_rss_bytes": 1000})
    frame = pd.DataFrame(rows)
    frame["topology_sha256"] = "shared-topology"
    frame["routing_sha256"] = "shared-routing"
    frame["offered_schedule_sha256"] = "shared-offered"
    frame["physical_config_sha256"] = "shared-config"
    assert len(validate_pilot_results(frame)) == 6
    altered = frame.copy(deep=True)
    index = altered.index[(altered.model_id == "ff") & (altered.output_profile == "detailed")][0]
    altered.at[index, "table_sha256"] = dict(altered.at[index, "table_sha256"])
    altered.at[index, "table_sha256"]["core_summary_df"] = "mismatch"
    with pytest.raises(ValueError, match="Scientific outputs differ"):
        validate_pilot_results(altered)


def test_full_factorial_statistics_use_five_seeds_and_handle_zero_delivery():
    rows = []
    for seed, ff_pdr, aa_pdr in zip((30, 36, 42, 48, 56), (70, 72, 74, 76, 78),
                                    (71, 74, 77, 80, 83), strict=True):
        for model, pdr, energy, delivered, delay in (
            ("ff", ff_pdr, 100.0, 100, 10.0),
            ("aa", aa_pdr, 90.0, 0 if seed == 30 else 100, 9.0),
        ):
            rows.append({"model_id": model, "node_num": 50, "node_period": 90,
                         "seed": seed, "pdr_percent": pdr,
                         "total_network_energy_J": energy,
                         "energy_per_delivered_packet_J": energy / delivered if delivered else np.nan,
                         "avg_e2e_delay_ms": delay, "throughput_bps": delivered * 400 / 3600})
    frame = pd.DataFrame(rows)
    pdr_stats = full_factorial_statistics(frame, ["model_id", "node_num", "node_period"], "pdr_percent")
    assert set(pdr_stats["n"]) == {5}
    paired = full_factorial_paired_statistics(frame)
    pdr_pair = paired.loc[paired.metric.eq("pdr_percent")].iloc[0]
    assert pdr_pair.n == 5
    assert pdr_pair.mean_difference == pytest.approx(3.0)
    epp_pair = paired.loc[paired.metric.eq("energy_per_delivered_packet_J")].iloc[0]
    assert epp_pair.n == 4


def test_full_factorial_energy_per_packet_uses_energy_divided_by_delivered_packets():
    result = _energy_per_delivered_packet(
        pd.Series([250.0, 900.0, 1000.0]), pd.Series([100, 300, 0])
    )
    assert result.iloc[0] == pytest.approx(2.5)
    assert result.iloc[1] == pytest.approx(3.0)
    assert np.isnan(result.iloc[2])
    with pytest.raises(ValueError, match="non-negative"):
        _energy_per_delivered_packet(pd.Series([1.0]), pd.Series([-1]))


def _manifest_bundle(old_fingerprint, new_fingerprint):
    ids = [*postprocessing.LEGACY_EXPERIMENT_IDS.values(),
           *(item for group in postprocessing.RUN_GROUP_IDS.values() for item in group)]
    new_ids = set(postprocessing.RUN_GROUP_IDS["ff"] + postprocessing.RUN_GROUP_IDS["aa"])
    return {
        experiment_id: {
            "source_fingerprint": new_fingerprint if experiment_id in new_ids else old_fingerprint
        }
        for experiment_id in ids
    }


def test_source_compatibility_rejects_unreviewed_simulator_file_change(monkeypatch):
    old = {"algorithm": "sha256", "files": {"same.py": "old"}}
    new = {
        "algorithm": "sha256",
        "files": {"same.py": "old", "src/loraphymac/runtime/node_tx.py": "changed"},
    }
    manifests = _manifest_bundle(old, new)
    monkeypatch.setattr(postprocessing, "_source_fingerprint", lambda _root: new)
    with pytest.raises(ValueError, match="unreviewed files"):
        _validate_source_compatibility(manifests)


def test_source_compatibility_accepts_only_audited_nonbehavior_source_change(monkeypatch):
    path = "src/loraphymac/experiments/apc_load_sweep/postprocessing.py"
    old = {"algorithm": "sha256", "files": {path: "old-report-hash"}}
    new = {"algorithm": "sha256", "files": {path: "new-report-hash"}}
    monkeypatch.setattr(postprocessing, "_source_fingerprint", lambda _root: new)
    monkeypatch.setattr(postprocessing, "file_sha256", lambda _path: "new-report-hash")
    result = _validate_source_compatibility(_manifest_bundle(old, new))
    assert result["status"] == "COMPATIBLE_NON_SIMULATION_SOURCE_CHANGES"
    assert result["changed_files"] == [{
        "path": path,
        "classification": "reporting/postprocessing only",
    }]


def test_cross_cohort_configuration_gate_allows_only_declared_model_and_cell_fields():
    baseline = {
        "node_num": 50, "node_period": 30, "seed": 30,
        "run_id": "historical-run", "experiment_profile": "old-profile",
        "node_ptx_mode": "fixed", "relay_ptx_mode": "fixed",
        "energy_accounting_mode": "corrected", "collision_mode": "capture",
    }
    changed = {
        **baseline, "node_num": 150, "node_period": 90, "seed": 56,
        "run_id": "new-run", "experiment_profile": "new-profile",
        "node_ptx_mode": "adaptive", "relay_ptx_mode": "adaptive",
    }
    _validate_cross_cohort_configs({"old": baseline, "new": changed})
    changed["collision_mode"] = "disabled"
    with pytest.raises(ValueError, match="Incompatible resolved"):
        _validate_cross_cohort_configs({"old": baseline, "new": changed})
