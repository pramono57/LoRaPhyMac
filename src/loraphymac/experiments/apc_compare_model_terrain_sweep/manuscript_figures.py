"""Clean, reusable IJIES manuscript figures from frozen aggregate tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from PIL import Image

from loraphymac.experiments.apc_compare_model_terrain_sweep.postprocessing import (
    FIGURE_SIZE,
    FIGURE_STYLE,
    MODEL_STYLES,
)
from loraphymac.experiments.common.multiseed_pipeline import (
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    utc_now,
)


MANUSCRIPT_FONT_FAMILY = "Times New Roman"
MANUSCRIPT_FONT_SIZE = 18
MANUSCRIPT_LINE_WIDTH = 2.0
MANUSCRIPT_MARKER_SIZE = 7.0
MANUSCRIPT_EXPORT_DPI = 300
MANUSCRIPT_LEGEND_COLUMNS = 2
MANUSCRIPT_LEGEND_FRAME_ALPHA = 0.9
MANUSCRIPT_GRID_STYLE = {
    "color": "#b0b0b0",
    "linestyle": "-",
    "linewidth": 0.8,
    "alpha": 1.0,
}
MANUSCRIPT_FIGURE_STYLE = {
    **FIGURE_STYLE,
    "grid.color": MANUSCRIPT_GRID_STYLE["color"],
    "grid.linestyle": MANUSCRIPT_GRID_STYLE["linestyle"],
    "grid.linewidth": MANUSCRIPT_GRID_STYLE["linewidth"],
    "grid.alpha": MANUSCRIPT_GRID_STYLE["alpha"],
}

CLEAN_FIGURE_MODELS = (
    "aa",
    "islam_dbpc_802",
    "wang_atp_sf7_a2only",
    "ff",
)
CLEAN_FIGURE_LABELS = {
    "aa": "R-APC (AA)",
    "islam_dbpc_802": "DBPC-802",
    "wang_atp_sf7_a2only": "CSPC",
    "ff": "Base (FF)",
}
# Backward-compatible names used by the Figure 08 tests and callers.
FIGURE_08_MODELS = CLEAN_FIGURE_MODELS
FIGURE_08_LABELS = CLEAN_FIGURE_LABELS
EXPECTED_TERRAINS = tuple(range(20, 161, 10))
EXPECTED_SEED_COUNT = 5
FIGURE_08_METRIC = "pdr_percent"
FIGURE_08_STEM = "figure_08_pdr_vs_terrain_clean"
FIGURE_08_MANIFEST_VERSION = "main_manuscript_figure_08_v1"
FIGURE_SPECS: dict[str, dict[str, Any]] = {
    "08": {
        "metric": FIGURE_08_METRIC,
        "stem": FIGURE_08_STEM,
        "ylabel": "Packet delivery ratio (%)",
        "legend_location": "lower left",
        "legend_columns": 1,
        "legend_font_size_pt": 14,
        "y_limits": (65.0, 85.0),
        "audit_mean_column": "mean_pdr_percent",
        "manifest_version": FIGURE_08_MANIFEST_VERSION,
        "metric_definition": "Mean packet delivery ratio across five paired seeds.",
    },
    "09": {
        "metric": "energy_per_delivered_packet_J",
        "stem": "figure_09_energy_per_delivered_packet_clean",
        "ylabel": "Energy per delivered packet (J/packet)",
        "legend_location": "upper right",
        "legend_columns": MANUSCRIPT_LEGEND_COLUMNS,
        "legend_font_size_pt": MANUSCRIPT_FONT_SIZE,
        "y_limits": (None, 0.80),
        "audit_mean_column": "mean_energy_per_delivered_packet_J",
        "manifest_version": "main_manuscript_figure_09_v1",
        "metric_definition": (
            "Per-run corrected total network energy divided by delivered packets, "
            "then averaged across five paired seeds."
        ),
    },
    "10": {
        "metric": "avg_e2e_delay_ms",
        "stem": "figure_10_avg_e2e_delay_clean",
        "ylabel": "Average end-to-end delay (ms)",
        "legend_location": "center right",
        "legend_columns": MANUSCRIPT_LEGEND_COLUMNS,
        "legend_font_size_pt": MANUSCRIPT_FONT_SIZE,
        "y_limits": None,
        "audit_mean_column": "mean_avg_e2e_delay_ms",
        "manifest_version": "main_manuscript_figure_10_v1",
        "metric_definition": (
            "Mean end-to-end delay of successfully delivered packets only; this metric "
            "is conditional on successful delivery and must be interpreted together with PDR."
        ),
    },
}


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def validate_main_w4_fixed_reference_source(statistics_csv: str | Path) -> Path:
    """Validate that the statistics table belongs to MAIN_W4 FIXED_REFERENCE."""
    source = Path(statistics_csv).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    manifest_path = source.parent.parent / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _read_json(manifest_path)
    definition = manifest.get("definition")
    if not isinstance(definition, Mapping):
        raise ValueError(f"{manifest_path}: missing definition")
    checks = {
        "experiment_track": manifest.get("experiment_track") == "MAIN_W4",
        "reporting_namespace": manifest.get("reporting_namespace") == "main_w4_fixed_reference",
        "routing_strategy": definition.get("routing_strategy") == "fixed_reference",
        "comparison_terrains": tuple(definition.get("comparison_terrains", ()))
        == EXPECTED_TERRAINS,
        "seeds": tuple(definition.get("seeds", ())) == (30, 36, 42, 48, 56),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"MAIN_W4 FIXED_REFERENCE source validation failed: {failed}")
    return manifest_path


def load_clean_metric_data(
    statistics_csv: str | Path,
    *,
    figure_number: str,
) -> pd.DataFrame:
    """Load and strictly validate one 60-point clean manuscript metric."""
    if figure_number not in FIGURE_SPECS:
        raise ValueError(f"Unsupported clean manuscript figure: {figure_number}")
    spec = FIGURE_SPECS[figure_number]
    source = Path(statistics_csv).expanduser().resolve()
    frame = pd.read_csv(source)
    required = {"model_id", "terrain_delta_h", "metric", "n", "mean"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"model_statistics.csv is missing columns: {sorted(missing)}")

    selected = frame.loc[
        (frame["metric"].astype(str) == spec["metric"])
        & frame["model_id"].astype(str).isin(CLEAN_FIGURE_MODELS)
    ].copy()
    selected["terrain_delta_h"] = pd.to_numeric(selected["terrain_delta_h"], errors="raise").astype(
        int
    )
    selected["n"] = pd.to_numeric(selected["n"], errors="raise").astype(int)
    selected["mean"] = pd.to_numeric(selected["mean"], errors="raise")

    expected_cells = {
        (model_id, terrain) for model_id in CLEAN_FIGURE_MODELS for terrain in EXPECTED_TERRAINS
    }
    observed_cells = set(zip(selected["model_id"], selected["terrain_delta_h"]))
    duplicates = selected.duplicated(subset=["model_id", "terrain_delta_h"], keep=False)
    failures: list[str] = []
    if len(selected) != len(expected_cells):
        failures.append(f"point_count={len(selected)} expected={len(expected_cells)}")
    if observed_cells != expected_cells:
        failures.append("model_terrain_grid")
    if duplicates.any():
        failures.append("duplicate_model_terrain_cells")
    if set(selected["n"]) != {EXPECTED_SEED_COUNT}:
        failures.append("seed_count")
    if not selected["mean"].map(math.isfinite).all():
        failures.append("non_finite_mean")
    if failures:
        raise ValueError(f"Figure {figure_number} data validation failed: {failures}")

    order = {model_id: index for index, model_id in enumerate(CLEAN_FIGURE_MODELS)}
    selected["legend_label"] = selected["model_id"].map(CLEAN_FIGURE_LABELS)
    selected["model_order"] = selected["model_id"].map(order)
    return selected.sort_values(["model_order", "terrain_delta_h"]).reset_index(drop=True)


def load_figure_08_data(statistics_csv: str | Path) -> pd.DataFrame:
    """Load and strictly validate the 60 mean PDR points used by Figure 08."""
    return load_clean_metric_data(statistics_csv, figure_number="08")


def resolve_exact_font(font_family: str = MANUSCRIPT_FONT_FAMILY) -> dict[str, str]:
    """Resolve the requested font without allowing Matplotlib fallback."""
    properties = font_manager.FontProperties(family=font_family)
    font_path = Path(font_manager.findfont(properties, fallback_to_default=False)).resolve()
    resolved_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    if resolved_name != font_family:
        raise ValueError(
            f"Font fallback is not allowed: requested={font_family!r}, resolved={resolved_name!r}"
        )
    return {
        "requested_family": font_family,
        "resolved_family": resolved_name,
        "resolved_path": str(font_path),
        "resolved_font_sha256": file_sha256(font_path),
    }


def build_clean_metric_figure(
    selected: pd.DataFrame,
    *,
    ylabel: str,
    legend_location: str = "lower left",
    legend_columns: int = MANUSCRIPT_LEGEND_COLUMNS,
    legend_font_size_pt: int = MANUSCRIPT_FONT_SIZE,
    y_limits: tuple[float | None, float | None] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Build a mean-only four-model figure using the reusable manuscript style."""
    with plt.rc_context(MANUSCRIPT_FIGURE_STYLE):
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        for model_id in CLEAN_FIGURE_MODELS:
            model = selected.loc[selected["model_id"] == model_id].sort_values("terrain_delta_h")
            style = MODEL_STYLES[model_id]
            axis.plot(
                model["terrain_delta_h"].to_numpy(dtype=float),
                model["mean"].to_numpy(dtype=float),
                label=CLEAN_FIGURE_LABELS[model_id],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=MANUSCRIPT_LINE_WIDTH,
                markersize=MANUSCRIPT_MARKER_SIZE,
            )
        axis.set_xlabel(r"Terrain irregularity, $\Delta h$ (m)")
        axis.set_ylabel(ylabel)
        if y_limits is not None:
            axis.set_ylim(*y_limits)
        axis.grid(True)
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(
            handles,
            labels,
            ncols=legend_columns,
            loc=legend_location,
            fontsize=legend_font_size_pt,
            frameon=True,
            framealpha=MANUSCRIPT_LEGEND_FRAME_ALPHA,
            fancybox=True,
            edgecolor="#b0b0b0",
        )
        figure.tight_layout()
        return figure, axis


def generate_clean_metric_figure(
    statistics_csv: str | Path,
    output_dir: str | Path,
    *,
    figure_number: str,
    reference_figure: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate one clean four-model figure and its audit artifacts."""
    if figure_number not in FIGURE_SPECS:
        raise ValueError(f"Unsupported clean manuscript figure: {figure_number}")
    spec = FIGURE_SPECS[figure_number]
    source = Path(statistics_csv).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    experiment_manifest = validate_main_w4_fixed_reference_source(source)
    selected = load_clean_metric_data(source, figure_number=figure_number)
    font_record = resolve_exact_font()

    png_path = output / f"{spec['stem']}.png"
    svg_path = output / f"{spec['stem']}.svg"
    data_path = output / f"{spec['stem']}_data.csv"
    manifest_path = output / f"{spec['stem']}_manifest.json"
    targets = (png_path, svg_path, data_path, manifest_path)
    existing = [str(path) for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing Figure {figure_number} outputs: {existing}"
        )
    output.mkdir(parents=True, exist_ok=True)

    figure, axis = build_clean_metric_figure(
        selected,
        ylabel=spec["ylabel"],
        legend_location=spec["legend_location"],
        legend_columns=spec["legend_columns"],
        legend_font_size_pt=spec["legend_font_size_pt"],
        y_limits=spec["y_limits"],
    )
    diagnostics = {
        "line_count": len(axis.lines),
        "collection_count": len(axis.collections),
        "container_count": len(axis.containers),
        "title": axis.get_title(),
        "legend_labels": [text.get_text() for text in axis.get_legend().get_texts()],
        "legend_frame_on": axis.get_legend().get_frame_on(),
    }
    if diagnostics != {
        "line_count": 4,
        "collection_count": 0,
        "container_count": 0,
        "title": "",
        "legend_labels": [CLEAN_FIGURE_LABELS[item] for item in CLEAN_FIGURE_MODELS],
        "legend_frame_on": True,
    }:
        plt.close(figure)
        raise ValueError(f"Unexpected Figure {figure_number} artist structure: {diagnostics}")

    figure.savefig(png_path, dpi=MANUSCRIPT_EXPORT_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    audit_data = selected[["model_id", "legend_label", "terrain_delta_h", "n", "mean"]].rename(
        columns={"n": "seed_count", "mean": spec["audit_mean_column"]}
    )
    atomic_write_csv(data_path, audit_data)

    with Image.open(png_path) as image:
        png_size = [int(image.width), int(image.height)]
        png_dpi = [float(value) for value in image.info.get("dpi", ())]

    reference_record: dict[str, Any] | None = None
    if reference_figure is not None:
        reference = Path(reference_figure).expanduser().resolve()
        if not reference.is_file():
            raise FileNotFoundError(reference)
        reference_record = {
            "path": str(reference),
            "sha256": file_sha256(reference),
        }

    manifest = {
        "manifest_version": spec["manifest_version"],
        "status": "PASS",
        "created_at": utc_now(),
        "scope": f"MAIN_W4_FIXED_REFERENCE_FIGURE_{figure_number}_ONLY",
        "source": {
            "model_statistics_csv": str(source),
            "model_statistics_sha256": file_sha256(source),
            "experiment_manifest": str(experiment_manifest),
            "experiment_manifest_sha256": file_sha256(experiment_manifest),
            "metric": spec["metric"],
            "metric_definition": spec["metric_definition"],
            "models": list(CLEAN_FIGURE_MODELS),
            "terrains": list(EXPECTED_TERRAINS),
            "seed_count_per_point": EXPECTED_SEED_COUNT,
            "plotted_point_count": len(selected),
            "selected_data_sha256": file_sha256(data_path),
        },
        "reference_figure": reference_record,
        "font": font_record,
        "visual_standard": {
            "figure_size_inches": list(FIGURE_SIZE),
            "aspect_ratio": float(FIGURE_SIZE[0] / FIGURE_SIZE[1]),
            "font_size_pt": MANUSCRIPT_FONT_SIZE,
            "axis_label_size_pt": MANUSCRIPT_FONT_SIZE,
            "tick_label_size_pt": MANUSCRIPT_FONT_SIZE,
            "legend_size_pt": spec["legend_font_size_pt"],
            "line_width_pt": MANUSCRIPT_LINE_WIDTH,
            "marker_size_pt": MANUSCRIPT_MARKER_SIZE,
            "y_limits": list(spec["y_limits"]) if spec["y_limits"] else None,
            "legend_columns": spec["legend_columns"],
            "legend_location": spec["legend_location"],
            "legend_frame": {
                "enabled": True,
                "framealpha": MANUSCRIPT_LEGEND_FRAME_ALPHA,
                "fancybox": True,
                "outside_axes": False,
                "inside_axes": True,
            },
            "grid": MANUSCRIPT_GRID_STYLE,
            "export_dpi": MANUSCRIPT_EXPORT_DPI,
            "confidence_interval": "not_rendered",
            "error_bars": "not_rendered",
            "model_styles": {
                model_id: {
                    "label": CLEAN_FIGURE_LABELS[model_id],
                    **MODEL_STYLES[model_id],
                }
                for model_id in CLEAN_FIGURE_MODELS
            },
        },
        "render_validation": {
            **diagnostics,
            "png_pixel_size": png_size,
            "png_dpi": png_dpi,
        },
        "outputs": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in (png_path, svg_path, data_path)
        ],
    }
    atomic_write_json(manifest_path, manifest)
    return {
        "status": "PASS",
        "png": str(png_path),
        "svg": str(svg_path),
        "data": str(data_path),
        "manifest": str(manifest_path),
        "point_count": len(selected),
        "font": font_record,
    }


def generate_figure_08(
    statistics_csv: str | Path,
    output_dir: str | Path,
    *,
    reference_figure: str | Path | None = None,
) -> dict[str, Any]:
    """Generate the clean four-model Figure 08 and its audit artifacts."""
    return generate_clean_metric_figure(
        statistics_csv,
        output_dir,
        figure_number="08",
        reference_figure=reference_figure,
    )


def generate_figure_09(
    statistics_csv: str | Path,
    output_dir: str | Path,
    *,
    reference_figure: str | Path | None = None,
) -> dict[str, Any]:
    """Generate clean energy-per-delivered-packet Figure 09."""
    return generate_clean_metric_figure(
        statistics_csv,
        output_dir,
        figure_number="09",
        reference_figure=reference_figure,
    )


def generate_figure_10(
    statistics_csv: str | Path,
    output_dir: str | Path,
    *,
    reference_figure: str | Path | None = None,
) -> dict[str, Any]:
    """Generate clean delivered-packet-conditional delay Figure 10."""
    return generate_clean_metric_figure(
        statistics_csv,
        output_dir,
        figure_number="10",
        reference_figure=reference_figure,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate clean four-model MAIN_W4 manuscript figures."
    )
    parser.add_argument("--statistics-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-figure")
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=tuple(FIGURE_SPECS),
        default=["08"],
        help="Figure numbers to generate (default: 08).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs for the selected figures.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    results = {
        figure_number: generate_clean_metric_figure(
            args.statistics_csv,
            args.output_dir,
            figure_number=figure_number,
            reference_figure=args.reference_figure,
            overwrite=args.overwrite,
        )
        for figure_number in args.figures
    }
    result: dict[str, Any]
    if len(results) == 1:
        result = next(iter(results.values()))
    else:
        result = {"status": "PASS", "figures": results}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
