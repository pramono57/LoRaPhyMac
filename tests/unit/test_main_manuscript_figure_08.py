from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from loraphymac.experiments.apc_compare_model_terrain_sweep.manuscript_figures import (
    EXPECTED_TERRAINS,
    FIGURE_SPECS,
    FIGURE_08_LABELS,
    FIGURE_08_MODELS,
    build_clean_metric_figure,
    load_clean_metric_data,
    load_figure_08_data,
    validate_main_w4_fixed_reference_source,
)


pytestmark = pytest.mark.unit


def _statistics() -> pd.DataFrame:
    rows = []
    for model_index, model_id in enumerate((*FIGURE_08_MODELS, "islam_dbpc_itm6")):
        for terrain in EXPECTED_TERRAINS:
            for metric_index, metric in enumerate(
                (
                    "pdr_percent",
                    "energy_per_delivered_packet_J",
                    "avg_e2e_delay_ms",
                )
            ):
                rows.append(
                    {
                        "model_id": model_id,
                        "model_label": model_id,
                        "terrain_delta_h": terrain,
                        "metric": metric,
                        "n": 5,
                        "mean": 80.0 - model_index - terrain / 1000.0 + metric_index,
                    }
                )
    return pd.DataFrame(rows)


def _write_main_w4_source(tmp_path):
    root = tmp_path / "main_w4_fixed_reference"
    tables = root / "tables"
    tables.mkdir(parents=True)
    source = tables / "model_statistics.csv"
    _statistics().to_csv(source, index=False)
    manifest = {
        "experiment_track": "MAIN_W4",
        "reporting_namespace": "main_w4_fixed_reference",
        "definition": {
            "routing_strategy": "fixed_reference",
            "comparison_terrains": list(EXPECTED_TERRAINS),
            "seeds": [30, 36, 42, 48, 56],
        },
    }
    (root / "experiment_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return source


def test_loads_exactly_60_mean_pdr_points_and_excludes_dbpc_itm(tmp_path):
    source = _write_main_w4_source(tmp_path)
    frame = load_figure_08_data(source)

    assert len(frame) == 60
    assert set(frame["model_id"]) == set(FIGURE_08_MODELS)
    assert set(frame["terrain_delta_h"]) == set(EXPECTED_TERRAINS)
    assert set(frame["n"]) == {5}
    assert "islam_dbpc_itm6" not in set(frame["model_id"])


def test_incomplete_or_wrong_seed_grid_is_rejected(tmp_path):
    source = _write_main_w4_source(tmp_path)
    frame = pd.read_csv(source).iloc[1:].copy()
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="point_count"):
        load_figure_08_data(source)

    source = _write_main_w4_source(tmp_path / "second")
    frame = pd.read_csv(source)
    frame.loc[frame.index[0], "n"] = 4
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="seed_count"):
        load_figure_08_data(source)


def test_rejects_non_main_or_terrain_specific_source(tmp_path):
    source = _write_main_w4_source(tmp_path)
    manifest_path = source.parent.parent / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["definition"]["routing_strategy"] = "terrain_specific"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="routing_strategy"):
        validate_main_w4_fixed_reference_source(source)


def test_clean_figure_has_four_lines_no_shading_or_error_bars(tmp_path):
    source = _write_main_w4_source(tmp_path)
    frame = load_figure_08_data(source)
    figure, axis = build_clean_metric_figure(
        frame,
        ylabel="Packet delivery ratio (%)",
        legend_location=FIGURE_SPECS["08"]["legend_location"],
        legend_columns=FIGURE_SPECS["08"]["legend_columns"],
        legend_font_size_pt=FIGURE_SPECS["08"]["legend_font_size_pt"],
        y_limits=FIGURE_SPECS["08"]["y_limits"],
    )
    try:
        assert len(axis.lines) == 4
        assert all(len(line.get_xdata()) == 15 for line in axis.lines)
        assert len(axis.collections) == 0
        assert len(axis.containers) == 0
        assert axis.get_title() == ""
        assert axis.get_ylim() == (65.0, 85.0)
        legend = axis.get_legend()
        assert legend is not None
        assert legend.get_frame_on()
        assert legend._ncols == 1
        assert all(text.get_fontsize() == 14 for text in legend.get_texts())
        assert [text.get_text() for text in legend.get_texts()] == [
            FIGURE_08_LABELS[item] for item in FIGURE_08_MODELS
        ]
        figure.canvas.draw()
        legend_box = legend.get_window_extent(figure.canvas.get_renderer())
        axes_box = axis.get_window_extent(figure.canvas.get_renderer())
        assert axes_box.y0 <= legend_box.y0
        assert axes_box.x0 <= legend_box.x0
        assert legend_box.x1 <= axes_box.x1
        assert legend_box.y1 <= axes_box.y1
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("figure_number", "metric"),
    (
        ("09", "energy_per_delivered_packet_J"),
        ("10", "avg_e2e_delay_ms"),
    ),
)
def test_figure_09_and_10_load_exactly_60_source_points(tmp_path, figure_number, metric):
    source = _write_main_w4_source(tmp_path)
    frame = load_clean_metric_data(source, figure_number=figure_number)

    assert len(frame) == 60
    assert set(frame["metric"]) == {metric}
    assert set(frame["model_id"]) == set(FIGURE_08_MODELS)
    assert set(frame["terrain_delta_h"]) == set(EXPECTED_TERRAINS)
    assert set(frame["n"]) == {5}


def test_all_clean_figures_reuse_identical_series_styles_and_artist_structure(tmp_path):
    source = _write_main_w4_source(tmp_path)
    signatures = []
    for figure_number in ("08", "09", "10"):
        frame = load_clean_metric_data(source, figure_number=figure_number)
        figure, axis = build_clean_metric_figure(
            frame,
            ylabel=FIGURE_SPECS[figure_number]["ylabel"],
            legend_location=FIGURE_SPECS[figure_number]["legend_location"],
            legend_columns=FIGURE_SPECS[figure_number]["legend_columns"],
            legend_font_size_pt=FIGURE_SPECS[figure_number]["legend_font_size_pt"],
            y_limits=FIGURE_SPECS[figure_number]["y_limits"],
        )
        try:
            assert len(axis.lines) == 4
            assert all(len(line.get_xdata()) == 15 for line in axis.lines)
            assert len(axis.collections) == 0
            assert len(axis.containers) == 0
            assert axis.get_title() == ""
            assert axis.get_ylabel() == FIGURE_SPECS[figure_number]["ylabel"]
            assert axis.get_legend() is not None
            assert axis.get_legend().get_frame_on()
            if FIGURE_SPECS[figure_number]["y_limits"] is not None:
                actual_limits = axis.get_ylim()
                for actual, expected in zip(
                    actual_limits, FIGURE_SPECS[figure_number]["y_limits"]
                ):
                    if expected is not None:
                        assert actual == expected
            signatures.append(
                [
                    (
                        line.get_label(),
                        line.get_color(),
                        line.get_marker(),
                        line.get_linestyle(),
                        line.get_linewidth(),
                        line.get_markersize(),
                    )
                    for line in axis.lines
                ]
            )
        finally:
            plt.close(figure)
    assert signatures[0] == signatures[1] == signatures[2]


def test_delay_metric_definition_is_conditional_on_successful_delivery():
    definition = FIGURE_SPECS["10"]["metric_definition"]
    assert "successfully delivered packets only" in definition
    assert "conditional on successful delivery" in definition
