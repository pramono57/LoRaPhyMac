"""Immutable routing snapshots for fixed-reference terrain experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


ROUTING_STRATEGY_TERRAIN_SPECIFIC = "terrain_specific"
ROUTING_STRATEGY_FIXED_REFERENCE = "fixed_reference"
ROUTING_REFERENCE_DELTA_H = 90
ROUTING_SNAPSHOT_SCHEMA_VERSION = "loraphymac_routing_snapshot_v1"
ROUTING_SNAPSHOT_MANIFEST_SCHEMA_VERSION = (
    "loraphymac_routing_snapshot_manifest_v1"
)
ROUTING_SNAPSHOT_FILENAME = "routing_snapshot.json"
ROUTING_EVIDENCE_FILENAME = "reference_link_evidence.csv"
ROUTING_MANIFEST_FILENAME = "routing_snapshot_manifest.json"


class RoutingSnapshotError(ValueError):
    """Raised when a fixed-reference routing snapshot is unsafe or invalid."""


@dataclass(frozen=True, slots=True)
class LoadedRoutingSnapshot:
    """Verified immutable routing state loaded before runtime processes exist."""

    snapshot_id: str
    seed: int
    reference_delta_h: int
    routing_map: Mapping[str, str | None]
    full_routes: Mapping[str, tuple[str, ...]]
    topology_sha256: str
    routing_sha256: str
    snapshot_sha256: str
    manifest_sha256: str
    manifest_path: Path


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used by snapshot hashes."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def content_sha256(value: Any) -> str:
    """Hash one JSON-compatible value with canonical ordering."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entity_type(entity_id: str) -> str:
    if entity_id == "GW" or entity_id.startswith("GW"):
        return "gateway"
    if entity_id.startswith("R"):
        return "relay"
    return "node"


def topology_records(topology: Any) -> list[dict[str, Any]]:
    """Return canonical geometry and hardware identity for one topology."""
    records = []
    for entity in sorted(topology.all_nodes, key=lambda item: str(item.id)):
        entity_id = str(entity.id)
        records.append(
            {
                "entity_id": entity_id,
                "entity_type": _entity_type(entity_id),
                "x_m": float(entity.loc.x),
                "y_m": float(entity.loc.y),
                "height_m": float(entity.h),
                "antenna_gain_dbi": float(entity.gain),
                "hardware_type": type(entity.hw).__name__,
                "ptx_min_dbm": int(entity.ptx_min),
                "ptx_max_dbm": int(entity.ptx_max),
            }
        )
    return records


def topology_sha256(topology: Any) -> str:
    """Hash deterministic geometry plus routing-relevant hardware identity."""
    return content_sha256(topology_records(topology))


def canonical_routing_records(
    routing_map: Mapping[Any, Any],
) -> list[dict[str, str | None]]:
    """Normalize a routing map without relying on insertion order."""
    normalized: dict[str, str | None] = {}
    for raw_source, raw_parent in routing_map.items():
        source = str(raw_source)
        parent = None if raw_parent is None else str(raw_parent)
        if not source:
            raise RoutingSnapshotError("routing map contains an empty source ID")
        if parent == "":
            raise RoutingSnapshotError(
                f"routing map contains an empty parent for {source!r}"
            )
        if source in normalized and normalized[source] != parent:
            raise RoutingSnapshotError(
                f"routing map has ambiguous normalized source {source!r}"
            )
        normalized[source] = parent
    return [
        {"source_id": source, "parent_id": normalized[source]}
        for source in sorted(normalized)
    ]


def routing_sha256(routing_map: Mapping[Any, Any]) -> str:
    """Hash normalized parent assignments."""
    return content_sha256(canonical_routing_records(routing_map))


def full_routes(
    routing_map: Mapping[Any, Any],
    *,
    gateway_id: str = "GW",
) -> dict[str, tuple[str, ...]]:
    """Validate and materialize every complete route to the gateway."""
    normalized = {
        row["source_id"]: row["parent_id"]
        for row in canonical_routing_records(routing_map)
    }
    if gateway_id not in normalized or normalized[gateway_id] is not None:
        raise RoutingSnapshotError(f"gateway {gateway_id!r} must be terminal")

    result: dict[str, tuple[str, ...]] = {}
    for source in sorted(normalized):
        route = [source]
        visited = {source}
        current = source
        while current != gateway_id:
            parent = normalized.get(current)
            if parent is None:
                raise RoutingSnapshotError(
                    f"route from {source!r} terminates at {current!r}, not gateway"
                )
            if parent not in normalized:
                raise RoutingSnapshotError(
                    f"route from {source!r} references unknown parent {parent!r}"
                )
            if parent in visited:
                raise RoutingSnapshotError(
                    f"routing cycle from {source!r} encountered at {parent!r}"
                )
            route.append(parent)
            visited.add(parent)
            current = parent
        result[source] = tuple(route)
    return result


def routing_configuration_contract(
    config: Mapping[str, Any],
    *,
    frequency_mhz: float,
) -> dict[str, Any]:
    """Capture every declared input used to reconstruct reference routing."""
    keys = (
        "node_num",
        "relay_num",
        "area_x",
        "area_y",
        "node_ant_height",
        "relay_ant_height",
        "gw_ant_height",
        "node_ant_gain",
        "relay_ant_gain",
        "gw_ant_gain",
        "node_max_eirp_dbm",
        "relay_max_eirp_dbm",
        "gw_max_eirp_dbm",
        "node_feeder_loss_db",
        "relay_feeder_loss_db",
        "gw_feeder_loss_db",
        "routing_assumed_sf",
        "routing_ch_id",
        "node_bw",
        "relay_sf",
        "relay_bw_hz",
        "routing_hop_penalty_alpha",
        "feasible_margin_db",
    )
    contract = {key: config[key] for key in keys}
    contract.update(
        {
            "routing_frequency_mhz": float(frequency_mhz),
            "routing_propagation_model": "ITM",
            "node_parent_selection": "highest_score",
            "feasibility_is_selection_filter": False,
            "relay_parent_selection": "deterministic_linear_relay_chain",
        }
    )
    return contract


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_routing_snapshot_bundle(
    *,
    output_dir: str | Path,
    config: Mapping[str, Any],
    topology: Any,
    routing: Any,
    reference_link_evidence: pd.DataFrame,
    frequency_mhz: float,
    source_provenance: Mapping[str, Any],
    source_config: Mapping[str, Any],
) -> Path:
    """Create a three-file immutable snapshot bundle without overwriting."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = output / ROUTING_SNAPSHOT_FILENAME
    evidence_path = output / ROUTING_EVIDENCE_FILENAME
    manifest_path = output / ROUTING_MANIFEST_FILENAME
    existing = [
        str(path)
        for path in (snapshot_path, evidence_path, manifest_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "routing snapshot bundle is immutable; existing files: "
            + ", ".join(existing)
        )

    seed = int(config["seed"])
    reference_delta_h = int(config["terrain_delta_h"])
    if reference_delta_h != ROUTING_REFERENCE_DELTA_H:
        raise RoutingSnapshotError(
            f"reference snapshot requires delta_h={ROUTING_REFERENCE_DELTA_H}"
        )
    topology_rows = topology_records(topology)
    route_rows = canonical_routing_records(routing.routing_map)
    routes = full_routes(routing.routing_map)
    topology_hash = content_sha256(topology_rows)
    routing_hash = content_sha256(route_rows)
    snapshot_id = (
        f"itm_dh{reference_delta_h}_s{seed}_{routing_hash[:12]}"
    )
    routing_contract = routing_configuration_contract(
        config,
        frequency_mhz=frequency_mhz,
    )
    snapshot = {
        "schema_version": ROUTING_SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "routing_strategy": ROUTING_STRATEGY_FIXED_REFERENCE,
        "seed": seed,
        "reference_delta_h": reference_delta_h,
        "gateway_id": "GW",
        "topology": topology_rows,
        "routing_map": route_rows,
        "full_routes": {
            source: list(route) for source, route in sorted(routes.items())
        },
        "routing_configuration": routing_contract,
        "topology_sha256": topology_hash,
        "selected_parent_routing_sha256": routing_hash,
        "source_provenance": dict(source_provenance),
        "source_config": dict(source_config),
        "source_config_sha256": content_sha256(dict(source_config)),
    }

    _write_json_atomic(snapshot_path, snapshot)
    try:
        reference_link_evidence.to_csv(evidence_path, index=False)
        snapshot_hash = file_sha256(snapshot_path)
        evidence_hash = file_sha256(evidence_path)
        manifest = {
            "schema_version": ROUTING_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "seed": seed,
            "reference_delta_h": reference_delta_h,
            "routing_strategy": ROUTING_STRATEGY_FIXED_REFERENCE,
            "topology_sha256": topology_hash,
            "selected_parent_routing_sha256": routing_hash,
            "routing_snapshot": {
                "path": ROUTING_SNAPSHOT_FILENAME,
                "sha256": snapshot_hash,
            },
            "reference_link_evidence": {
                "path": ROUTING_EVIDENCE_FILENAME,
                "sha256": evidence_hash,
                "row_count": int(len(reference_link_evidence)),
            },
            "source_provenance": dict(source_provenance),
            "source_config_sha256": snapshot["source_config_sha256"],
        }
        _write_json_atomic(manifest_path, manifest)
    except Exception:
        for path in (snapshot_path, evidence_path):
            if path.exists():
                path.unlink()
        raise
    return manifest_path


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RoutingSnapshotError(f"cannot read routing JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RoutingSnapshotError(f"routing JSON {path} must contain an object")
    return value


def _bundle_file(manifest_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise RoutingSnapshotError(f"{label} path must be a relative filename")
    candidate = (manifest_path.parent / value).resolve()
    if candidate.parent != manifest_path.parent.resolve():
        raise RoutingSnapshotError(f"{label} path escapes the snapshot bundle")
    if not candidate.is_file():
        raise RoutingSnapshotError(f"{label} file is missing: {candidate}")
    return candidate


def _require_sha256(value: Any, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise RoutingSnapshotError(f"{label} must be a 64-character SHA-256")
    return text


def load_routing_snapshot_bundle(
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    config: Mapping[str, Any],
    topology: Any,
    frequency_mhz: float,
) -> LoadedRoutingSnapshot:
    """Load and validate a fixed route; never regenerate or fall back."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise RoutingSnapshotError(
            f"fixed-reference routing manifest is missing: {manifest_file}"
        )
    expected_hash = _require_sha256(
        expected_manifest_sha256,
        "routing_snapshot_manifest_sha256",
    )
    actual_manifest_hash = file_sha256(manifest_file)
    if actual_manifest_hash != expected_hash:
        raise RoutingSnapshotError(
            "routing snapshot manifest hash mismatch: "
            f"expected {expected_hash}, got {actual_manifest_hash}"
        )
    manifest = _read_json_object(manifest_file)
    if manifest.get("schema_version") != ROUTING_SNAPSHOT_MANIFEST_SCHEMA_VERSION:
        raise RoutingSnapshotError("unsupported routing snapshot manifest schema")

    snapshot_info = manifest.get("routing_snapshot")
    evidence_info = manifest.get("reference_link_evidence")
    if not isinstance(snapshot_info, Mapping) or not isinstance(
        evidence_info, Mapping
    ):
        raise RoutingSnapshotError("routing manifest file records are incomplete")
    snapshot_file = _bundle_file(
        manifest_file,
        snapshot_info.get("path"),
        "routing snapshot",
    )
    evidence_file = _bundle_file(
        manifest_file,
        evidence_info.get("path"),
        "reference evidence",
    )
    snapshot_hash = _require_sha256(
        snapshot_info.get("sha256"), "routing snapshot sha256"
    )
    evidence_hash = _require_sha256(
        evidence_info.get("sha256"), "reference evidence sha256"
    )
    if file_sha256(snapshot_file) != snapshot_hash:
        raise RoutingSnapshotError("routing snapshot file hash mismatch")
    if file_sha256(evidence_file) != evidence_hash:
        raise RoutingSnapshotError("reference link evidence file hash mismatch")

    snapshot = _read_json_object(snapshot_file)
    if snapshot.get("schema_version") != ROUTING_SNAPSHOT_SCHEMA_VERSION:
        raise RoutingSnapshotError("unsupported routing snapshot schema")
    if snapshot.get("snapshot_id") != manifest.get("snapshot_id"):
        raise RoutingSnapshotError("snapshot ID differs from manifest")
    if snapshot.get("routing_strategy") != ROUTING_STRATEGY_FIXED_REFERENCE:
        raise RoutingSnapshotError("snapshot is not fixed_reference")
    if int(snapshot.get("seed", -1)) != int(config["seed"]):
        raise RoutingSnapshotError("routing snapshot seed mismatch")
    reference_delta_h = int(snapshot.get("reference_delta_h", -1))
    if reference_delta_h != ROUTING_REFERENCE_DELTA_H or reference_delta_h != int(
        config["routing_reference_delta_h"]
    ):
        raise RoutingSnapshotError("routing snapshot reference delta_h mismatch")

    expected_contract = routing_configuration_contract(
        config,
        frequency_mhz=frequency_mhz,
    )
    if snapshot.get("routing_configuration") != expected_contract:
        raise RoutingSnapshotError(
            "current routing configuration differs from snapshot assumptions"
        )

    current_topology = topology_records(topology)
    current_topology_hash = content_sha256(current_topology)
    if snapshot.get("topology") != current_topology:
        raise RoutingSnapshotError(
            "current topology identities, coordinates, heights, or hardware differ "
            "from the routing snapshot"
        )
    if snapshot.get("topology_sha256") != current_topology_hash:
        raise RoutingSnapshotError("routing snapshot topology hash mismatch")
    if manifest.get("topology_sha256") != current_topology_hash:
        raise RoutingSnapshotError("routing manifest topology hash mismatch")

    raw_routes = snapshot.get("routing_map")
    if not isinstance(raw_routes, Sequence) or isinstance(raw_routes, (str, bytes)):
        raise RoutingSnapshotError("routing snapshot parent map is invalid")
    route_map: dict[str, str | None] = {}
    for row in raw_routes:
        if not isinstance(row, Mapping) or "source_id" not in row:
            raise RoutingSnapshotError("routing snapshot contains an invalid route row")
        source = str(row["source_id"])
        parent = row.get("parent_id")
        parent = None if parent is None else str(parent)
        if source in route_map:
            raise RoutingSnapshotError(f"duplicate routing source {source!r}")
        route_map[source] = parent

    expected_ids = {row["entity_id"] for row in current_topology}
    if set(route_map) != expected_ids:
        raise RoutingSnapshotError(
            "routing snapshot entity identities do not match current topology"
        )
    verified_routes = full_routes(route_map, gateway_id=str(snapshot.get("gateway_id")))
    serialized_routes = snapshot.get("full_routes")
    if not isinstance(serialized_routes, Mapping) or {
        str(source): tuple(route) for source, route in serialized_routes.items()
    } != verified_routes:
        raise RoutingSnapshotError("snapshot full routes disagree with parent map")
    route_hash = routing_sha256(route_map)
    if snapshot.get("selected_parent_routing_sha256") != route_hash:
        raise RoutingSnapshotError("routing snapshot parent hash mismatch")
    if manifest.get("selected_parent_routing_sha256") != route_hash:
        raise RoutingSnapshotError("routing manifest parent hash mismatch")

    if not math.isfinite(float(config["terrain_delta_h"])):
        raise RoutingSnapshotError("physical terrain delta_h must be finite")
    return LoadedRoutingSnapshot(
        snapshot_id=str(snapshot["snapshot_id"]),
        seed=int(snapshot["seed"]),
        reference_delta_h=reference_delta_h,
        routing_map=dict(route_map),
        full_routes=dict(verified_routes),
        topology_sha256=current_topology_hash,
        routing_sha256=route_hash,
        snapshot_sha256=snapshot_hash,
        manifest_sha256=actual_manifest_hash,
        manifest_path=manifest_file,
    )


__all__ = [
    "LoadedRoutingSnapshot",
    "ROUTING_EVIDENCE_FILENAME",
    "ROUTING_MANIFEST_FILENAME",
    "ROUTING_REFERENCE_DELTA_H",
    "ROUTING_SNAPSHOT_FILENAME",
    "ROUTING_SNAPSHOT_MANIFEST_SCHEMA_VERSION",
    "ROUTING_SNAPSHOT_SCHEMA_VERSION",
    "ROUTING_STRATEGY_FIXED_REFERENCE",
    "ROUTING_STRATEGY_TERRAIN_SPECIFIC",
    "RoutingSnapshotError",
    "canonical_json",
    "canonical_routing_records",
    "content_sha256",
    "file_sha256",
    "full_routes",
    "load_routing_snapshot_bundle",
    "routing_configuration_contract",
    "routing_sha256",
    "topology_records",
    "topology_sha256",
    "write_routing_snapshot_bundle",
]
