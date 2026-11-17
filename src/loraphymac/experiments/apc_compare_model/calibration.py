"""Freeze the already-audited Stage 2B.3a DBPC-ITM calibration bundle.

No path-loss fitting or performance-result calibration occurs here.  The
coverage values are the approved Stage 2B.3a ITM/90, SF7 values; this module
only materializes them, hashes the bundle, and replays the canonical audit CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from loraphymac.runtime.islam_dbpc import (
    CALIBRATION_VERSION,
    ISLAM_DBPC_BASELINE_NAME,
    ISLAM_DBPC_POLICY_ID,
    sha256_file,
)


CALIBRATION_ID = "islam_dbpc_itm90_margin0_fixed_sf7_v1"
APPROVED_AUDIT_SHA256 = "c369d591069546d94d7e3b087235012eb2a5596b9f2195f88e13203c21449c65"
AUDITED_SOURCE_COMMIT = "9d8fdeb2d78f42c0d55251db4f8b874bd8549117"
NODE_LADDER = (5, 7, 9, 11, 13, 15, 17)
RELAY_LADDER = (12, 14, 16, 18, 20)

# Approved Stage 2B.3a ITM/90 coverage boundaries in metres.  These are
# materialized rather than recomputed so the calibration cannot drift with a
# later propagation implementation.
AUDITED_COVERAGE_M = {
    0: {
        "node": (
            (5, 7491.705445),
            (7, 8739.365775),
            (9, 10113.361034),
            (11, 11610.983264),
            (13, 13227.926889),
            (15, 14958.770486),
            (17, 16797.459356),
        ),
        "relay": (
            (12, 25543.091332),
            (14, 27570.908226),
            (16, 29648.079954),
            (18, 31771.436233),
            (20, 33938.094411),
        ),
    },
    6: {
        "node": (
            (5, 4502.270002),
            (7, 5375.910911),
            (9, 6371.079721),
            (11, 7491.705445),
            (13, 8739.365775),
            (15, 10113.361034),
            (17, 11610.983264),
        ),
        "relay": (
            (12, 19792.919048),
            (14, 21649.980818),
            (16, 23568.146158),
            (18, 25543.091332),
            (20, 27570.908226),
        ),
    },
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _coverage_rows(margin_db: int) -> list[dict[str, Any]]:
    rows = []
    for entity_type in ("node", "relay"):
        for ptx_dbm, max_distance_m in AUDITED_COVERAGE_M[margin_db][entity_type]:
            rows.append(
                {
                    "calibration_id": CALIBRATION_ID,
                    "method_id": ISLAM_DBPC_POLICY_ID,
                    "propagation_model": "ITM_delta_h_90",
                    "margin_db": margin_db,
                    "entity_type": entity_type,
                    "ptx_dbm": ptx_dbm,
                    "max_distance_m": f"{max_distance_m:.6f}",
                    "distance_floor_m": "0.001",
                    "comparison_rule": "distance_m < max_distance_m",
                    "use_snr": "false",
                }
            )
    return rows


def _select(entity_type: str, margin_db: int, distance_m: float) -> tuple[int, bool, bool]:
    coverage = AUDITED_COVERAGE_M[margin_db][entity_type]
    for ptx_dbm, max_distance_m in coverage:
        if float(distance_m) < max_distance_m:
            return ptx_dbm, ptx_dbm == coverage[-1][0], False
    return coverage[-1][0], True, True


def audit_replay(audit_csv: str | Path) -> dict[str, Any]:
    """Replay approved DBPC-ITM decisions without recalculating coverage."""
    path = Path(audit_csv)
    if sha256_file(path) != APPROVED_AUDIT_SHA256:
        raise ValueError("Stage 2B.3a audit CSV SHA-256 mismatch")
    row_count = 0
    link_keys = set()
    itm_rows = []
    mismatches = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            link_keys.add(
                (
                    row["seed"],
                    row["terrain_delta_h_route_m"],
                    row["entity_id"],
                    row["parent_id"],
                )
            )
            if row["dbpc_variant"] != "DBPC-ITM":
                continue
            margin = int(row["calibration_margin_db"])
            selected, at_max, outside = _select(
                row["entity_type"], margin, float(row["distance_m"])
            )
            itm_rows.append(row)
            expected = (
                int(row["selected_ptx_dbm"]),
                row["selected_at_max_ptx"].lower() == "true",
                row["outside_estimated_coverage"].lower() == "true",
            )
            actual = (selected, at_max, outside)
            if actual != expected:
                mismatches.append(
                    {
                        "seed": row["seed"],
                        "terrain_delta_h_route_m": row["terrain_delta_h_route_m"],
                        "entity_id": row["entity_id"],
                        "margin_db": margin,
                        "expected": expected,
                        "actual": actual,
                    }
                )
    return {
        "audit_row_count": row_count,
        "unique_link_case_count": len(link_keys),
        "itm_replay_row_count": len(itm_rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _distribution_rows(audit_csv: Path) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    with audit_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dbpc_variant"] == "DBPC-ITM":
                groups[(int(row["calibration_margin_db"]), row["entity_type"])].append(row)

    result = []
    for margin_db, entity_type in sorted(groups):
        rows = groups[(margin_db, entity_type)]
        counts = Counter(int(row["selected_ptx_dbm"]) for row in rows)
        mean_ptx = sum(int(row["selected_ptx_dbm"]) for row in rows) / len(rows)
        saturation = sum(row["selected_at_max_ptx"].lower() == "true" for row in rows)
        outside = sum(row["outside_estimated_coverage"].lower() == "true" for row in rows)
        ladder = NODE_LADDER if entity_type == "node" else RELAY_LADDER
        for ptx_dbm in ladder:
            result.append(
                {
                    "calibration_id": CALIBRATION_ID,
                    "method_id": ISLAM_DBPC_POLICY_ID,
                    "margin_db": margin_db,
                    "entity_type": entity_type,
                    "link_case_count": len(rows),
                    "selected_ptx_dbm": ptx_dbm,
                    "selection_count": counts[ptx_dbm],
                    "selection_percent": f"{100.0 * counts[ptx_dbm] / len(rows):.6f}",
                    "mean_selected_ptx_dbm": f"{mean_ptx:.6f}",
                    "saturation_percent": f"{100.0 * saturation / len(rows):.6f}",
                    "outside_estimated_coverage_percent": f"{100.0 * outside / len(rows):.6f}",
                }
            )
    return result


def create_calibration_bundle(output_dir: str | Path, audit_csv: str | Path) -> dict[str, Any]:
    """Create one immutable, hash-protected bundle from the approved audit."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen calibration directory: {output}")
    audit_path = Path(audit_csv).resolve()
    replay = audit_replay(audit_path)
    if replay["audit_row_count"] != 30_600 or replay["unique_link_case_count"] != 7_650:
        raise ValueError("Approved audit CSV row/link conservation failed")
    if replay["mismatch_count"]:
        raise ValueError(f"Approved ITM replay has {replay['mismatch_count']} mismatches")
    output.mkdir(parents=True)

    calibration_config = {
        "propagation_model": "ITM",
        "calibration_terrain_delta_h_m": 90,
        "frequency_mhz": 921.4,
        "sf": 7,
        "bandwidth_hz": 125000,
        "coding_rate": "4/5",
        "use_snr": False,
        "distance_floor_m": 0.001,
        "comparison_rule": "distance_m < max_distance_m",
        "receiver_sensitivity_dbm": -126,
        "primary_margin_db": 0,
        "analytical_sensitivity_margin_db": 6,
        "node_ptx_ladder_dbm": list(NODE_LADDER),
        "relay_ptx_ladder_dbm": list(RELAY_LADDER),
    }
    inputs = {
        "calibration_id": CALIBRATION_ID,
        "calibration_version": CALIBRATION_VERSION,
        "method_id": ISLAM_DBPC_POLICY_ID,
        "baseline_identity": ISLAM_DBPC_BASELINE_NAME,
        "source_git_commit": AUDITED_SOURCE_COMMIT,
        "source_audit": {
            "filename": audit_path.name,
            "sha256": APPROVED_AUDIT_SHA256,
            "row_count": 30_600,
            "unique_link_case_count": 7_650,
        },
        "topology_parent_provenance": {
            "artifact": "topology_parent_by_run.csv",
            "sha256": "74e91b13c1a20f44c5a11ec4a5c8a65c199f2dcb04be9c40e267caefd6b4ba84",
            "routing_rule": "existing selected parent; no DBPC reselection",
        },
        "calibration_configuration": calibration_config,
        "configuration_sha256": hashlib.sha256(_canonical_json(calibration_config)).hexdigest(),
        "coverage_origin": "frozen Stage 2B.3a audit; not recalculated in Stage 2B.3b",
        "evaluation_outcomes_used": False,
    }
    _write_json(output / "calibration_inputs.json", inputs)

    coverage_columns = (
        "calibration_id",
        "method_id",
        "propagation_model",
        "margin_db",
        "entity_type",
        "ptx_dbm",
        "max_distance_m",
        "distance_floor_m",
        "comparison_rule",
        "use_snr",
    )
    for margin in (0, 6):
        _write_csv(
            output / f"coverage_table_margin_{margin}.csv",
            _coverage_rows(margin),
            coverage_columns,
        )
    distribution_columns = (
        "calibration_id",
        "method_id",
        "margin_db",
        "entity_type",
        "link_case_count",
        "selected_ptx_dbm",
        "selection_count",
        "selection_percent",
        "mean_selected_ptx_dbm",
        "saturation_percent",
        "outside_estimated_coverage_percent",
    )
    _write_csv(
        output / "calibration_distribution_audit.csv",
        _distribution_rows(audit_path),
        distribution_columns,
    )

    protected_files = (
        "calibration_inputs.json",
        "coverage_table_margin_0.csv",
        "coverage_table_margin_6.csv",
        "calibration_distribution_audit.csv",
    )
    manifest = {
        "calibration_id": CALIBRATION_ID,
        "calibration_version": CALIBRATION_VERSION,
        "method_id": ISLAM_DBPC_POLICY_ID,
        "baseline_identity": ISLAM_DBPC_BASELINE_NAME,
        "source_git_commit": AUDITED_SOURCE_COMMIT,
        "input_data_sha256": {"stage2b3a_audit_csv": APPROVED_AUDIT_SHA256},
        "configuration_sha256": inputs["configuration_sha256"],
        "topology_parent_provenance": inputs["topology_parent_provenance"],
        "propagation_model": "ITM_delta_h_90",
        "phy": {
            "frequency_mhz": 921.4,
            "sf": 7,
            "bandwidth_hz": 125000,
            "coding_rate": "4/5",
            "receiver_sensitivity_dbm": -126,
            "use_snr": False,
        },
        "ptx_ladders_dbm": {"node": list(NODE_LADDER), "relay": list(RELAY_LADDER)},
        "distance_floor_m": 0.001,
        "strict_comparison_rule": "distance_m < max_distance_m",
        "primary_margin_db": 0,
        "analytical_sensitivity_margin_db": 6,
        "coverage_tables": {
            "0": "coverage_table_margin_0.csv",
            "6": "coverage_table_margin_6.csv",
        },
        "runtime_constraints": {
            "routing_ch_id": 0,
            "node_sf_list": [7],
            "node_bw": 125000,
            "relay_sf": 7,
            "relay_bw_hz": 125000,
            "node_ant_height": 2.0,
            "node_ant_gain": 2.15,
            "relay_ant_height": 25.0,
            "relay_ant_gain": 5.8,
            "gw_ant_height": 25.0,
            "gw_ant_gain": 5.8,
            "node_feeder_loss_db": 0.0,
            "relay_feeder_loss_db": 0.0,
            "gw_feeder_loss_db": 0.0,
            "node_max_eirp_dbm": 20.0,
            "relay_max_eirp_dbm": 26.0,
            "gw_max_eirp_dbm": 26.0,
        },
        "file_sha256": {name: sha256_file(output / name) for name in protected_files},
        "manifest_self_hash_scheme": (
            "sha256(canonical_json(manifest excluding manifest_payload_sha256))"
        ),
    }
    manifest["manifest_payload_sha256"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    _write_json(output / "calibration_manifest.json", manifest)
    return {
        **replay,
        "manifest_sha256": sha256_file(output / "calibration_manifest.json"),
        "file_sha256": manifest["file_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = create_calibration_bundle(args.output_dir, args.audit_csv)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
