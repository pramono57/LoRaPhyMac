from pathlib import Path

from loraphymac.experiments.common.contract_registry import (
    R1_7_DUTY_PROFILE,
    R1_7_LOAD_PROFILE,
)
from loraphymac.experiments.common.multiseed_pipeline import (
    build_experiment_grid,
    load_experiment_spec,
)


CONFIG_DIR = (
    Path(__file__).resolve().parents[2]
    / "src/loraphymac/experiments/scalability_relay_bottleneck/configs"
)


def test_r1_7_load_profiles_plan_45_paired_runs_per_model():
    ff = load_experiment_spec(CONFIG_DIR / "r1_7_load_ff_v1.json")
    aa = load_experiment_spec(CONFIG_DIR / "r1_7_load_aa_v1.json")
    ff_runs = build_experiment_grid(ff)
    aa_runs = build_experiment_grid(aa)

    assert len(ff_runs) == len(aa_runs) == 45
    assert ff.base_config["experiment_profile"] == R1_7_LOAD_PROFILE
    assert ff.output_dir == aa.output_dir
    assert ff.experiment_id != aa.experiment_id
    assert ff.run_annotations["model_id"] == "ff"
    assert aa.run_annotations["model_id"] == "aa"
    assert ff_runs[0]["scientific_config"]["node_ptx_mode"] == "fixed"
    assert aa_runs[0]["scientific_config"]["node_ptx_mode"] == "adaptive"


def test_r1_7_relay_duty_profiles_plan_10_runs_per_model():
    ff = load_experiment_spec(CONFIG_DIR / "r1_7_relay_duty_ff_v1.json")
    aa = load_experiment_spec(CONFIG_DIR / "r1_7_relay_duty_aa_v1.json")

    assert len(build_experiment_grid(ff)) == len(build_experiment_grid(aa)) == 10
    assert ff.base_config["experiment_profile"] == R1_7_DUTY_PROFILE
    assert ff.factors["relay_duty_cycle"] == [0.1, 0.01]
    assert ff.base_config["node_num"] == aa.base_config["node_num"] == 100
    assert ff.base_config["node_period"] == aa.base_config["node_period"] == 30


def test_r1_7_profiles_use_fresh_output_namespace():
    for name in (
        "r1_7_load_ff_v1.json",
        "r1_7_load_aa_v1.json",
        "r1_7_relay_duty_ff_v1.json",
        "r1_7_relay_duty_aa_v1.json",
    ):
        spec = load_experiment_spec(CONFIG_DIR / name)
        assert spec.output_dir.name == "r1_7"
        assert spec.experiment_id.startswith("r1_7_")
