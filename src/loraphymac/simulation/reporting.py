"""Console reporting for completed simulation outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from loraphymac.simulation.output_collection import CollectedSimulationOutputs


def _display_table(value: Any, *, max_rows: int = 10) -> None:
    """Render a bounded DataFrame preview without flooding Jupyter output."""
    if isinstance(value, pd.DataFrame):
        print(value.to_string(index=False, max_rows=max_rows))
        if len(value) > max_rows:
            print(
                f"[preview {max_rows} of {len(value)} rows; "
                "the complete table is retained in the exported workbook]"
            )
    else:
        print(value)


def report_simulation_outputs(
    *,
    config: Mapping[str, Any],
    statistics: Any,
    outputs: CollectedSimulationOutputs,
) -> None:
    """Print the configured summary, detail, and debug views."""
    tables = outputs.tables
    preview_rows = int(config.get("debug_preview_rows", 10))

    def display_table(value: Any) -> None:
        _display_table(value, max_rows=preview_rows)

    if bool(config["print_summary"]):
        print("\n=== CORE SUMMARY ===")
        print(
            statistics.view_as_vertical(
                statistics.view_core_summary(tables["core_summary_df"])
            ).to_markdown(index=False, tablefmt="github")
        )

    if bool(config["print_detail"]):
        print("\n=== ENERGY BREAKDOWN ===")
        display_table(
            statistics.view_energy_breakdown(tables["energy_breakdown_df"])
        )

        print("\n=== RELAY DUTY CYCLE SUMMARY ===")
        display_table(
            statistics.view_relay_duty_cycle(tables["relay_duty_cycle_df"])
        )

        print("\n=== RELAY TX CHANNEL AIRTIME / DUTY CYCLE MATRIX ===")
        display_table(
            statistics.view_per_channel_duty_cycle(
                tables["per_channel_duty_cycle_df"]
            )
        )

        print("\n=== RELAY TX QUEUE WAITING TIME SUMMARY ===")
        display_table(tables["tx_queue_waiting_time_summary_df"])

        print("\n=== RELAY TX M/G/c QUEUE METRICS ===")
        display_table(tables["relay_tx_mgc_queue_metrics_df"])

        print("\n=== FINAL PTX DISTRIBUTION ===")
        display_table(
            statistics.view_final_ptx_distribution(
                tables["final_ptx_distribution_df"]
            )
        )

        print("\n=== FRAME SUMMARY ===")
        frame_summary_df = tables["frame_summary_df"]
        if not frame_summary_df.empty and hasattr(
            statistics,
            "view_frame_summary_vertical",
        ):
            print(
                statistics.view_frame_summary_vertical(frame_summary_df).to_string(
                    index=False
                )
            )
        else:
            display_table(frame_summary_df)

        print("\n=== FRAME HOP EVENT SUMMARY ===")
        display_table(tables["frame_hop_event_summary_df"])

    if bool(config["print_debug"]):
        print("\n=== RUN METADATA ===")
        run_metadata_df = tables["run_metadata_df"]
        if not run_metadata_df.empty:
            print(
                statistics.view_as_vertical(run_metadata_df).to_markdown(
                    index=False,
                    tablefmt="github",
                )
            )
        else:
            print(run_metadata_df)

        print("\n=== SELECTED NEXT-HOP LINKS ===")
        display_table(tables["selected_next_hop_links_df"])

        print("\n=== RELAY CHAIN LINKS ===")
        display_table(tables["relay_chain_links_df"])

        print("\n=== VISUAL LINKS ===")
        display_table(tables["visual_links_df"])

        print("\n=== NODE RX MATRIX ===")
        display_table(tables["node_rx_matrix_df"])

        print("\n=== RELAY RX STATISTICS ===")
        display_table(tables["relay_rx_stats_df"])

        print("\n=== FRAME MTYPE SUMMARY ===")
        display_table(tables["frame_mtype_summary_df"])

        print("\n=== FRAME DROP REASON BY HOP ===")
        display_table(tables["frame_drop_reason_by_hop_df"])

        print("\n=== RELAY/GATEWAY RX CHANNEL USAGE ===")
        display_table(tables["rx_channel_usage_df"])

        print("\n=== DEMODULATOR POOL SUMMARY ===")
        display_table(tables["demodulator_pool_summary_df"])

        print("\n=== DEMODULATOR EVENT LOG ===")
        display_table(tables["demodulator_event_log_df"])


def print_one_run_summary(outputs: Mapping[str, pd.DataFrame]) -> None:
    if "selected_next_hop_links_df" in outputs:
        print("\n=== SELECTED NEXT-HOP LINKS ===")
        print(outputs["selected_next_hop_links_df"])

    if "relay_chain_links_df" in outputs:
        print("\n=== RELAY CHAIN LINKS ===")
        print(outputs["relay_chain_links_df"])

    if "visual_links_df" in outputs:
        print("\n=== VISUAL LINKS ===")
        print(outputs["visual_links_df"])

    if "run_metadata_df" in outputs:
        print("\n=== RUN METADATA ===")
        print(outputs["run_metadata_df"])

    if "node_rx_matrix_df" in outputs:
        print("\n=== NODE RX MATRIX ===")
        print(outputs["node_rx_matrix_df"])

    if "relay_rx_stats_df" in outputs:
        print("\n=== RX STATISTICS ===")
        print(outputs["relay_rx_stats_df"])

    if "rx_channel_usage_df" in outputs:
        print("\n=== RX CHANNEL USAGE ===")
        print(outputs["rx_channel_usage_df"])

    if "relay_duty_cycle_df" in outputs:
        print("\n=== RELAY DUTY CYCLE MATRIX ===")
        relay_duty_cycle_df = outputs["relay_duty_cycle_df"]
        cols = [
            c for c in [
                "entity_id",
                "total_tx_time_s",
                "relay_duty_cycle_percent",
            ]
            if c in relay_duty_cycle_df.columns
        ]
        print(relay_duty_cycle_df[cols])

    if "per_channel_duty_cycle_df" in outputs:
        print("\n=== PER-CHANNEL DUTY CYCLE MATRIX ===")
        per_channel_duty_cycle_df = outputs["per_channel_duty_cycle_df"]
        cols = [
            c for c in [
                "entity_id",
                "channel",
                "tx_packet_count",
                "total_airtime_per_channel_s",
                "per_channel_duty_cycle_percent",
            ]
            if c in per_channel_duty_cycle_df.columns
        ]
        print(per_channel_duty_cycle_df[cols])

    if "energy_breakdown_df" in outputs:
        print("\n=== ENERGY BREAKDOWN ===")
        energy_breakdown_df = outputs["energy_breakdown_df"]
        cols = [
            c for c in [
                "entity_id",
                "entity_type",
                "duplex_mode",
                "tx_full_duplex",
                "tx_time_s",
                "rx_time_s",
                "energy_tx_J",
                "energy_rx_J",
                "energy_sleep_J",
                "energy_transition_J",
                "total_energy_J",
            ]
            if c in energy_breakdown_df.columns
        ]
        print(energy_breakdown_df[cols])

    if "tx_queue_waiting_time_summary_df" in outputs:
        print("\n=== TX QUEUE WAITING TIME SUMMARY ===")
        print(outputs["tx_queue_waiting_time_summary_df"])

    if "relay_tx_mgc_queue_metrics_df" in outputs:
        print("\n=== RELAY TX M/G/c QUEUE METRICS ===")
        print(outputs["relay_tx_mgc_queue_metrics_df"])

    if "hop_summary_df" in outputs:
        print("\n=== HOP SUMMARY ===")
        print(outputs["hop_summary_df"])

    if "frame_summary_df" in outputs:
        print("\n=== FRAME SUMMARY ===")
        print(outputs["frame_summary_df"])

    if "frame_mtype_summary_df" in outputs:
        print("\n=== FRAME MTYPE SUMMARY ===")
        print(outputs["frame_mtype_summary_df"])

    if "frame_hop_event_summary_df" in outputs:
        print("\n=== FRAME HOP EVENT SUMMARY ===")
        print(outputs["frame_hop_event_summary_df"])

    if "frame_drop_reason_by_hop_df" in outputs:
        print("\n=== FRAME DROP REASON BY HOP ===")
        print(outputs["frame_drop_reason_by_hop_df"])

    if "core_summary_df" in outputs:
        print("\n=== CORE SUMMARY ===")
        print(outputs["core_summary_df"])

    if "final_ptx_distribution_df" in outputs:
        print("\n=== FINAL PTX DISTRIBUTION ===")
        final_ptx_distribution_df = outputs["final_ptx_distribution_df"]
        cols = [
            c for c in [
                "entity_id",
                "entity_type",
                "ptx_mode",
                "final_ptx",
                "last_margin",
                "last_feedback_from",
            ]
            if c in final_ptx_distribution_df.columns
        ]
        print(final_ptx_distribution_df[cols])

    if "apc_control_history_df" in outputs:
        print("\n=== APC CONTROL HISTORY ===")
        print(outputs["apc_control_history_df"])
