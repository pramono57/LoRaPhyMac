from __future__ import annotations

import pandas as pd
import pytest

from loraphymac.experiments.common.outputs import ExperimentArtifactPaths
from loraphymac.experiments.apc_terrain_sweep.postprocessing import (
    build_terrain_sweep_artifacts,
)
from loraphymac.publications.paper_rapc_multihop_lora.analysis import (
    build_publication_tables as build_paper1_tables,
)
from loraphymac.publications.paper_rapc_multihop_lora.figures import (
    FINAL_NODE_SERIES,
    FINAL_RELAY_SERIES,
    FIGURE_STYLE,
    MODE_COLORS,
    PLOT_MODE_ORDER,
    generate_all_figures as generate_all_paper1_figures,
    generate_metric_figure,
)
from loraphymac.publications.paper2_scalability.analysis import (
    build_publication_tables as build_paper2_tables,
)
from loraphymac.publications.paper2_scalability.figures import (
    generate_scalability_figure,
)


def _stats_row(**values):
    return {
        "n": 5,
        "std": 1.0,
        "sem": 0.447,
        "confidence_level": 0.95,
        "ci_method": "student_t",
        "ci_lower": values["mean"] - 1.0,
        "ci_upper": values["mean"] + 1.0,
        "minimum": values["mean"] - 2.0,
        "median": values["mean"],
        "maximum": values["mean"] + 2.0,
        **values,
    }


def _paper1_aggregate() -> pd.DataFrame:
    rows = []
    modes = (
        ("fixed", "fixed", 80.0, 3000.0, 2000.0, 500.0, 500.0, 17.0, 20.0),
        ("adaptive", "fixed", 81.0, 2900.0, 1900.0, 500.0, 500.0, 10.3, 20.0),
        ("fixed", "adaptive", 82.0, 2800.0, 2000.0, 300.0, 500.0, 17.0, 12.0),
        ("adaptive", "adaptive", 83.0, 2700.0, 1900.0, 350.0, 450.0, 10.3, 12.0),
    )
    for terrain_delta_h in (90, 100):
        for (
            node_mode,
            relay_mode,
            delivery,
            energy,
            node_energy,
            relay_energy,
            gateway_energy,
            node_ptx,
            relay_ptx,
        ) in modes:
            constrained_energy = node_energy + relay_energy
            factors = {
                "node_ptx_mode": node_mode,
                "relay_ptx_mode": relay_mode,
                "terrain_delta_h": terrain_delta_h,
            }
            metric_values = {
                "delivery_by_horizon_percent": delivery,
                "avg_e2e_delay_ms": 260.0 + (delivery - 80.0),
                "total_node_energy_J": node_energy,
                "total_relay_energy_J": relay_energy,
                "total_gateway_energy_J": gateway_energy,
                "total_network_energy_J": energy,
                "total_constrained_energy_J": constrained_energy,
                "energy_per_delivered_packet_J": energy / 5000.0,
                "energy_per_delivered_packet_constrained_J": constrained_energy / 5000.0,
                "avg_final_node_ptx": node_ptx,
                "avg_final_relay_ptx": relay_ptx,
            }
            rows.extend(
                _stats_row(**factors, metric=metric, mean=mean)
                for metric, mean in metric_values.items()
            )
    return pd.DataFrame(rows)


def test_paper1_builds_legacy_parity_tables_and_six_ordered_figures(tmp_path):
    paths = ExperimentArtifactPaths(tmp_path / "paper1_apc_v1").ensure()
    publication_dir = tmp_path / "publications" / "paper1_apc_v1"
    aggregate = _paper1_aggregate()
    aggregate.to_csv(paths.aggregate / "core_metrics_long.csv", index=False)

    outputs = build_paper1_tables(
        paths.experiment_dir,
        publication_dir=publication_dir,
    )
    comparison = pd.read_csv(outputs["comparison"])
    constrained = pd.read_csv(outputs["table_10"])

    assert comparison.columns.tolist() == ["Parameter", "FF", "AF", "FA", "AA"]
    assert comparison["Parameter"].tolist() == [
        "PDR (%)",
        "Average End-to-End Delay (ms)",
        "Total Energy (J)",
        "Constrained Energy (J)",
        "Energy per Delivered Packet (J)",
        "Constrained Energy per Delivered Packet (J)",
        "Energy Saving vs FF (%)",
        "Constrained Energy Saving vs FF (%)",
        "Average Final Node PTx (dBm)",
        "Average Final Relay PTx (dBm)",
    ]
    savings = comparison.set_index("Parameter").loc["Energy Saving vs FF (%)"]
    assert savings["AA"] == 10.0
    assert constrained.columns.tolist() == [
        "Mode",
        "Node Energy (J)",
        "Relay Energy (J)",
        "Constrained Energy (J)",
        "Saving vs FF (%)",
        "Constrained Energy / Delivered Packet (J)",
    ]
    assert constrained.set_index("Mode").loc["AA", "Saving vs FF (%)"] == 10.0
    assert len({path for path in outputs.values()}) == 3

    figure_paths = generate_all_paper1_figures(
        paths.experiment_dir,
        publication_dir=publication_dir,
    )
    assert len(figure_paths) == 12
    assert {path.suffix for path in figure_paths} == {".png", ".pdf"}
    assert [path.stem for path in figure_paths[::2]] == [
        "figure_01_energy_breakdown_delta_h_90",
        "figure_02_final_node_ptx",
        "figure_03_final_relay_ptx",
        "figure_04_pdr",
        "figure_05_energy_per_delivered_packet",
        "figure_06_delay",
    ]
    assert all(path.exists() for path in figure_paths)
    assert all(path.is_relative_to(publication_dir) for path in outputs.values())
    assert all(path.is_relative_to(publication_dir) for path in figure_paths)
    assert not any(paths.tables.iterdir())
    assert not any(paths.figures.iterdir())


def test_paper1_uses_legacy_times_new_roman_mode_style():
    assert FIGURE_STYLE["font.family"] == "Times New Roman"
    assert {
        FIGURE_STYLE["font.size"],
        FIGURE_STYLE["axes.labelsize"],
        FIGURE_STYLE["xtick.labelsize"],
        FIGURE_STYLE["ytick.labelsize"],
        FIGURE_STYLE["legend.fontsize"],
    } == {18}
    assert PLOT_MODE_ORDER == ("AA", "AF", "FA", "FF")
    assert MODE_COLORS == {
        "AA": "tab:blue",
        "AF": "tab:orange",
        "FA": "tab:green",
        "FF": "tab:red",
    }
    assert [series[3:] for series in FINAL_NODE_SERIES] == [("o", "-"), ("D", ":")]
    assert [series[3:] for series in FINAL_RELAY_SERIES] == [("^", "-."), ("D", ":")]


def test_paper1_metric_figure_writes_png_and_pdf(tmp_path):
    paths = ExperimentArtifactPaths(tmp_path / "paper1_apc_v1").ensure()
    aggregate = _paper1_aggregate()

    figure_paths = generate_metric_figure(
        aggregate,
        metric="delivery_by_horizon_percent",
        ylabel="Delivery (%)",
        output_base=paths.figures / "test_delivery",
    )
    assert {path.suffix for path in figure_paths} == {".png", ".pdf"}
    assert all(path.exists() for path in figure_paths)


def test_terrain_sweep_postprocessing_owns_semantic_tables_and_figures(tmp_path):
    paths = ExperimentArtifactPaths(tmp_path / "apc_terrain_sweep_v1_r01").ensure()
    _paper1_aggregate().to_csv(paths.aggregate / "core_metrics_long.csv", index=False)
    paths.status.write_text(
        '{"status":"PASS","experiment_id":"apc_terrain_sweep_v1_r01",'
        '"seed_grid_complete":true,"requested_run_count":300,'
        '"passed_run_count_on_disk":300}',
        encoding="utf-8",
    )

    summary = build_terrain_sweep_artifacts(experiment_dir=paths.experiment_dir)

    assert summary["status"] == "PASS"
    assert summary["table_count"] == 3
    assert summary["figure_count"] == 12
    assert (paths.tables / "terrain_apc_performance_delta_h_90.csv").is_file()
    assert (paths.figures / "terrain_pdr.png").is_file()
    assert (paths.experiment_dir / "analysis_manifest.json").is_file()


def test_paper1_rejects_publication_artifacts_for_incomplete_seed_grid(tmp_path):
    paths = ExperimentArtifactPaths(tmp_path / "paper1_apc_v1").ensure()
    _paper1_aggregate().to_csv(paths.aggregate / "core_metrics_long.csv", index=False)
    paths.status.write_text('{"status": "PASS", "seed_grid_complete": false}', encoding="utf-8")

    with pytest.raises(ValueError, match="seed grid is incomplete"):
        build_paper1_tables(
            paths.experiment_dir,
            publication_dir=tmp_path / "publication",
        )
    with pytest.raises(ValueError, match="seed grid is incomplete"):
        generate_all_paper1_figures(
            paths.experiment_dir,
            publication_dir=tmp_path / "publication",
        )


def test_paper2_preserves_unlimited_scenarios_in_tables_and_figures(tmp_path):
    paths = ExperimentArtifactPaths(tmp_path / "paper2_v1").ensure()
    publication_dir = tmp_path / "publications" / "paper2_v1"
    rows = []
    for node_num, delivery in ((50, 95.0), (100, 85.0)):
        factors = {
            "node_num": node_num,
            "node_period": 60,
            "relay_duty_cycle": None,
            "relay_tx_waiting_queue_capacity_packets": None,
        }
        rows.extend(
            [
                _stats_row(
                    **factors,
                    metric="delivery_by_horizon_percent",
                    mean=delivery,
                ),
                _stats_row(
                    **factors,
                    metric="total_network_energy_J",
                    mean=float(node_num * 10),
                ),
            ]
        )
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(paths.aggregate / "core_metrics_long.csv", index=False)

    outputs = build_paper2_tables(
        paths.experiment_dir,
        publication_dir=publication_dir,
    )
    core = pd.read_csv(outputs["core_means"])

    assert len(core) == 2
    assert set(core["relay_duty_cycle"]) == {"unlimited"}
    assert set(core["relay_tx_waiting_queue_capacity_packets"]) == {"unbounded"}
    assert all(path.is_relative_to(publication_dir) for path in outputs.values())
    assert not any(paths.tables.iterdir())

    figure_paths = generate_scalability_figure(
        aggregate,
        metric="delivery_by_horizon_percent",
        ylabel="Delivery (%)",
        output_base=paths.figures / "test_delivery",
    )
    assert all(path.exists() for path in figure_paths)
