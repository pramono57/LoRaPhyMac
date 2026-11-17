import json
import inspect
from pathlib import Path

import pytest

import loraphymac.engine as legacy_engine
from loraphymac.config import defaults as simulator_defaults
from loraphymac.contracts.paper1_apc import PAPER1_APC_PROFILE as LEGACY_PAPER1_PROFILE
from loraphymac.experiments import multiseed_pipeline as legacy_pipeline
from loraphymac.experiments.apc_terrain_sweep.contract import (
    APC_TERRAIN_SWEEP_PROFILE,
    PAPER1_APC_PROFILE as HISTORICAL_PAPER1_PROFILE,
)
from loraphymac.experiments.apc_terrain_sweep.run import (
    DEFAULT_CONFIG as APC_TERRAIN_SWEEP_CONFIG,
    PROJECT_ROOT as APC_TERRAIN_SWEEP_ROOT,
)
from loraphymac.experiments.common import multiseed_pipeline
from loraphymac.experiments.common.outputs import (
    OUTPUT_LAYOUT_VERSION,
    ExperimentArtifactPaths,
    resolve_master_table_path,
)
from loraphymac.experiments.scalability_relay_bottleneck.run import (
    DEFAULT_CONFIG as SCALABILITY_CONFIG,
    PROJECT_ROOT as SCALABILITY_ROOT,
)
from loraphymac.experiments.apc_closed_loop_verification.workflow import (
    APC_CLOSED_LOOP_ARTIFACT_ROOT,
)
from loraphymac.experiments.single_run_exploration.workflow import (
    EXPLORATION_ARTIFACT_ROOT,
)
from loraphymac.publications.paper_rapc_multihop_lora.analysis import (
    DEFAULT_CONFIG as RAPC_MULTIHOP_LORA_PUBLICATION_CONFIG,
)
from loraphymac.publications.common import load_publication_spec
from loraphymac import SIMULATOR_API_VERSION, SIMULATOR_VERSION
from loraphymac.simulation.reporting import print_one_run_summary
from loraphymac.simulation import runner as simulation_runner


def test_remaining_legacy_facades_point_to_content_based_modules():
    assert LEGACY_PAPER1_PROFILE == HISTORICAL_PAPER1_PROFILE
    assert legacy_pipeline.load_experiment_spec is multiseed_pipeline.load_experiment_spec
    assert legacy_pipeline.run_experiment is multiseed_pipeline.run_experiment


def test_repository_uses_one_src_package_namespace():
    project_root = Path(__file__).resolve().parents[2]
    source_root = project_root / "src" / "loraphymac"

    assert (source_root / "simulator.py").is_file()
    assert (source_root / "engine.py").is_file()
    assert (source_root / "config/defaults.py").is_file()
    for internal_module in (
        "simulation/runner.py",
        "simulation/output_collection.py",
        "simulation/reporting.py",
        "simulation/exporting.py",
    ):
        assert (source_root / internal_module).is_file()
    for legacy_path in (
        "config",
        "contracts",
        "domain",
        "experiments",
        "hardware",
        "mac",
        "metrics",
        "phy",
        "pyitm",
        "routing",
        "runtime",
        "utils",
        "main_simulation_function_based.py",
        "run_multiseed_experiments.py",
    ):
        assert not (project_root / legacy_path).exists()


def test_generated_output_defaults_use_one_artifacts_root():
    project_root = Path(__file__).resolve().parents[2]
    assert simulator_defaults.DEFAULT_CONFIG["output_dir"] == (
        "artifacts/experiments/single_run_exploration/default"
    )
    assert EXPLORATION_ARTIFACT_ROOT == (
        project_root / "artifacts/experiments/single_run_exploration"
    )
    assert APC_CLOSED_LOOP_ARTIFACT_ROOT == (
        project_root / "artifacts/experiments/apc_closed_loop_verification"
    )


def test_scientific_master_reader_prefers_gzip_then_legacy(tmp_path):
    paths = ExperimentArtifactPaths(tmp_path / "cohort")
    paths.scientific_master.mkdir(parents=True)
    compressed = paths.scientific_master / "all_runs_core_summary.csv.gz"
    compressed.write_bytes(b"gzip")
    assert resolve_master_table_path(paths.experiment_dir, "all_runs_core_summary.csv") == compressed

    compressed.unlink()
    legacy = paths.master / "all_runs_core_summary.csv"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("run_id\nrun-1\n", encoding="utf-8")
    assert resolve_master_table_path(paths.experiment_dir, "all_runs_core_summary.csv") == legacy


def test_legacy_engine_is_a_thin_compatibility_facade():
    assert legacy_engine.DEFAULT_CONFIG is simulator_defaults.DEFAULT_CONFIG
    assert legacy_engine.build_config is simulator_defaults.build_config
    assert legacy_engine.run_single_simulation is simulation_runner.run_single_simulation
    assert legacy_engine.print_one_run_summary is print_one_run_summary

    facade_source = inspect.getsource(legacy_engine)
    assert "NodeTX" not in facade_source
    assert "env.run" not in facade_source


def test_runner_delegates_post_horizon_output_work():
    runner_source = inspect.getsource(simulation_runner)
    assert "collect_simulation_outputs(" in runner_source
    assert "report_simulation_outputs(" in runner_source
    assert "export_simulation_outputs(" in runner_source
    assert "get_summary_bundle(" not in runner_source
    assert "export_summary_workbook(" not in runner_source


def test_apc_terrain_sweep_folder_owns_complete_production_grid():
    project_root = Path(__file__).resolve().parents[2]
    assert APC_TERRAIN_SWEEP_ROOT == project_root
    assert APC_TERRAIN_SWEEP_CONFIG == (
        project_root / "src/loraphymac/experiments/apc_terrain_sweep/configs/production_v1.json"
    )

    spec = multiseed_pipeline.load_experiment_spec(APC_TERRAIN_SWEEP_CONFIG)
    grid = multiseed_pipeline.build_experiment_grid(spec)
    aggregate_config = project_root / (
        "src/loraphymac/experiments/apc_terrain_sweep/configs/aggregate_inputs_v1.json"
    )
    aggregate_spec = multiseed_pipeline.load_experiment_spec(aggregate_config)
    aggregate_grid = multiseed_pipeline.build_experiment_grid(aggregate_spec)

    assert spec.base_config["experiment_profile"] == APC_TERRAIN_SWEEP_PROFILE
    assert spec.output_dir == project_root / "artifacts/experiments/apc_terrain_sweep"
    assert spec.experiment_id == "apc_terrain_sweep_v1_r01"
    assert spec.master_table_mode == "full"
    assert aggregate_spec.experiment_id != spec.experiment_id
    assert aggregate_spec.output_dir != spec.output_dir
    assert aggregate_spec.master_table_mode == "aggregate_inputs"
    assert aggregate_spec.base_config == spec.base_config
    assert aggregate_spec.factors == spec.factors
    assert aggregate_spec.seeds == spec.seeds
    assert len(grid) == 300
    assert len(aggregate_grid) == 300


def test_scalability_folder_owns_separate_pilot_grid():
    project_root = Path(__file__).resolve().parents[2]
    assert SCALABILITY_ROOT == project_root
    assert SCALABILITY_CONFIG == (
        project_root
        / "src/loraphymac/experiments/scalability_relay_bottleneck/configs/pilot_v1.json"
    )

    spec = multiseed_pipeline.load_experiment_spec(SCALABILITY_CONFIG)
    grid = multiseed_pipeline.build_experiment_grid(spec)

    assert spec.experiment_id == "scalability_relay_bottleneck_v1_pilot_r01"
    assert spec.output_dir == (project_root / "artifacts/experiments/scalability_relay_bottleneck")
    assert len(grid) == 270


def test_rapc_multihop_lora_publication_maps_source_to_separate_destination():
    project_root = Path(__file__).resolve().parents[2]
    spec = load_publication_spec(RAPC_MULTIHOP_LORA_PUBLICATION_CONFIG)

    assert spec.source_experiment_dir is None
    assert spec.sources["terrain_sweep"] == (
        project_root
        / "artifacts/experiments/apc_terrain_sweep/apc_terrain_sweep_v1_r01"
    )
    assert spec.sources["closed_loop_verification"] == (
        project_root
        / "artifacts/experiments/apc_closed_loop_verification/apc_closed_loop_dh90_s42"
    )
    assert spec.output_dir == (
        project_root
        / "artifacts/publications/paper_rapc_multihop_lora/paper_rapc_multihop_lora_v1"
    )


def test_publication_config_rejects_output_inside_source_experiment(tmp_path):
    config = tmp_path / "publication.json"
    config.write_text(
        json.dumps(
            {
                "publication_id": "unsafe",
                "source_experiment_dir": "experiment",
                "output_dir": "experiment/figures",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be separate"):
        load_publication_spec(config)


def test_standard_output_layout_creates_every_artifact_layer(tmp_path):
    paths = ExperimentArtifactPaths(tmp_path / "paper" / "experiment-v1").ensure()

    assert paths.directory_map() == {
        "experiment": ".",
        "runs": "runs",
        "master": "master",
        "aggregate": "aggregate",
        "tables": "tables",
        "figures": "figures",
        "logs": "logs",
    }
    assert all(
        path.is_dir()
        for path in (
            paths.runs,
            paths.master,
            paths.aggregate,
            paths.tables,
            paths.figures,
            paths.logs,
        )
    )


def test_workflow_notebooks_are_thin_and_output_free():
    project_root = Path(__file__).resolve().parents[2]
    notebooks = (
        project_root
        / "src/loraphymac/experiments/single_run_exploration/notebooks/"
        "single_run_exploration.ipynb",
        project_root
        / "src/loraphymac/experiments/apc_closed_loop_verification/notebooks/"
        "apc_closed_loop_verification.ipynb",
        project_root / "src/loraphymac/experiments/apc_terrain_sweep/notebooks/"
        "apc_terrain_sweep_workflow.ipynb",
        project_root
        / "src/loraphymac/experiments/apc_compare_model_terrain_sweep/notebooks/"
        "apc_compare_model_terrain_sweep_workflow.ipynb",
        project_root
        / "src/loraphymac/publications/paper_rapc_multihop_lora/notebooks/"
        "paper_rapc_multihop_lora_workflow.ipynb",
        project_root / "src/loraphymac/experiments/scalability_relay_bottleneck/notebooks/"
        "scalability_relay_bottleneck_workflow.ipynb",
        project_root / "src/loraphymac/publications/paper2_scalability/notebooks/"
        "paper2_scalability_workflow.ipynb",
    )

    for notebook in notebooks:
        document = json.loads(notebook.read_text(encoding="utf-8"))
        code_cells = [cell for cell in document["cells"] if cell["cell_type"] == "code"]
        code = "\n".join("".join(cell["source"]) for cell in code_cells)
        assert len(code_cells) <= 4
        assert all(cell["execution_count"] is None for cell in code_cells)
        assert all(cell["outputs"] == [] for cell in code_cells)
        assert "run_single_simulation" not in code
        assert "from experiments." not in code
        assert "from loraphymac." in code
        assert "BASE_CONFIG" not in code
        assert "itertools.product" not in code


def test_experiment_manifest_records_output_layout_contract(tmp_path):
    spec = multiseed_pipeline.ExperimentSpec(
        experiment_id="layout-contract",
        output_dir=tmp_path,
        base_config={"sim_time": 10, "node_period": 60},
        factors={"node_num": [10]},
        seeds=[7],
    )
    project_root = Path(__file__).resolve().parents[2]

    experiment_dir = multiseed_pipeline.initialize_experiment(spec, project_root)
    manifest = json.loads((experiment_dir / "experiment_manifest.json").read_text(encoding="utf-8"))

    assert manifest["output_layout_version"] == OUTPUT_LAYOUT_VERSION
    assert manifest["simulator_version"] == SIMULATOR_VERSION
    assert manifest["simulator_api_version"] == SIMULATOR_API_VERSION
    assert manifest["artifact_directories"]["figures"] == "figures"
    assert (experiment_dir / "logs").is_dir()
