"""Filesystem export policy for a completed simulation run."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from loraphymac.simulation.output_collection import CollectedSimulationOutputs


def export_simulation_outputs(
    *,
    config: Mapping[str, Any],
    run_id: str,
    statistics: Any,
    outputs: CollectedSimulationOutputs,
) -> dict[str, str]:
    """Write enabled summary/result/debug workbooks and return their paths."""
    export_summary = bool(
        config.get("export_summary", config.get("export_results", False))
    )
    if not (
        export_summary
        or bool(config["export_results"])
        or bool(config["export_debug"])
    ):
        return {}

    output_dir = str(config["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    tag = (
        f"run_{run_id}_"
        f"n{config['node_num']}_r{config['relay_num']}_dh{config['terrain_delta_h']}_"
        f"nodeptx_{config['node_ptx_mode']}_relayptx_{config['relay_ptx_mode']}_"
        f"bw{config['node_bw']}_ptx{config['node_ptx_init']}_t{config['sim_time']}"
    )
    exported: dict[str, str] = {}

    if export_summary:
        summary_file = os.path.join(output_dir, f"summary_{tag}.xlsx")
        statistics.export_bundle_to_excel(outputs.summary_bundle, summary_file)
        exported["summary"] = summary_file
        if bool(config["print_summary"]) or bool(config["verbose"]):
            print(f"\n📁 Summary export finished: {summary_file}")

    if bool(config["export_results"]):
        result_file = os.path.join(output_dir, f"results_{tag}.xlsx")
        statistics.export_bundle_to_excel(outputs.result_bundle, result_file)
        exported["results"] = result_file
        if bool(config["print_summary"]) or bool(config["verbose"]):
            print(f"\n📊 Result export finished: {result_file}")

    if bool(config["export_debug"]) and config.get("output_profile", "detailed") == "detailed":
        debug_file = os.path.join(output_dir, f"debug_{tag}.xlsx")
        statistics.export_bundle_to_excel(outputs.debug_bundle, debug_file)
        exported["debug"] = debug_file
        if bool(config["print_summary"]) or bool(config["verbose"]):
            print(f"\n🐞 Debug export finished: {debug_file}")

    return exported
