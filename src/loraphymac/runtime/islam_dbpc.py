"""Frozen distance-to-parent power selection for the Islam DBPC adaptation.

This module implements only the audited fixed-SF/variable-PTx adaptation of
Islam et al. Algorithm 2.  It does not implement feedback, ACK handling,
retries, learning, or dynamic SF selection.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from loraphymac.runtime.apc_power import _get_supported_ptx_values


ISLAM_DBPC_POLICY_ID = "islam_algorithm2_fixed_sf_ptx_v1"
ISLAM_DBPC_BASELINE_NAME = "Islam-DBPC-PTx fixed-SF adaptation"
CALIBRATION_VERSION = "islam_dbpc_itm90_fixed_sf7_v1"


class CalibrationIntegrityError(ValueError):
    """Raised when a frozen DBPC calibration bundle is not trustworthy."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise CalibrationIntegrityError(f"Invalid boolean value in coverage table: {value!r}")


@dataclass(frozen=True)
class DBPCSelection:
    """One deterministic Algorithm-2 PTx choice."""

    entity_id: str
    entity_type: str
    parent_id: str
    distance_m: float
    selected_ptx_dbm: int
    selected_at_max_ptx: bool
    outside_estimated_coverage: bool
    max_distance_m: float


@dataclass(frozen=True)
class FrozenDBPCCalibration:
    """Validated immutable coverage lookup loaded from one manifest."""

    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    margin_db: int
    coverage_by_entity: Mapping[str, tuple[tuple[int, float], ...]]

    @property
    def calibration_id(self) -> str:
        return str(self.manifest["calibration_id"])

    @property
    def method_id(self) -> str:
        return str(self.manifest["method_id"])

    @property
    def ptx_ladders(self) -> Mapping[str, tuple[int, ...]]:
        return {
            key: tuple(int(value) for value in values)
            for key, values in self.manifest["ptx_ladders_dbm"].items()
        }

    def select(
        self,
        *,
        entity_id: str,
        entity_type: str,
        parent_id: str,
        distance_m: float,
    ) -> DBPCSelection:
        """Select the lowest PTx whose frozen coverage strictly exceeds distance."""
        try:
            coverage = self.coverage_by_entity[entity_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported DBPC entity type {entity_type!r}") from exc
        distance = float(distance_m)
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError(f"Invalid distance_m={distance_m!r}")

        feasible = [(ptx, limit) for ptx, limit in coverage if distance < limit]
        if feasible:
            selected_ptx, max_distance = feasible[0]
            outside = False
        else:
            selected_ptx, max_distance = coverage[-1]
            outside = True
        return DBPCSelection(
            entity_id=str(entity_id),
            entity_type=entity_type,
            parent_id=str(parent_id),
            distance_m=distance,
            selected_ptx_dbm=int(selected_ptx),
            selected_at_max_ptx=int(selected_ptx) == int(coverage[-1][0]),
            outside_estimated_coverage=outside,
            max_distance_m=float(max_distance),
        )


def load_frozen_calibration(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_calibration_id: str,
    margin_db: int,
) -> FrozenDBPCCalibration:
    """Load and fully verify a frozen DBPC calibration bundle."""
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise CalibrationIntegrityError(f"Calibration manifest not found: {path}")
    actual_manifest_hash = sha256_file(path)
    if actual_manifest_hash != str(expected_manifest_sha256).lower():
        raise CalibrationIntegrityError(
            "Calibration manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, got {actual_manifest_hash}"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationIntegrityError(f"Invalid calibration manifest: {exc}") from exc

    if manifest.get("method_id") != ISLAM_DBPC_POLICY_ID:
        raise CalibrationIntegrityError("Calibration method identity does not match DBPC policy")
    if manifest.get("calibration_version") != CALIBRATION_VERSION:
        raise CalibrationIntegrityError("Unsupported calibration version")
    if manifest.get("calibration_id") != expected_calibration_id:
        raise CalibrationIntegrityError("Calibration ID does not match approved configuration")

    declared_payload_hash = manifest.get("manifest_payload_sha256")
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    actual_payload_hash = hashlib.sha256(_canonical_json(payload)).hexdigest()
    if declared_payload_hash != actual_payload_hash:
        raise CalibrationIntegrityError("Calibration manifest payload hash mismatch")

    bundle_dir = path.parent
    file_hashes = manifest.get("file_sha256", {})
    if not isinstance(file_hashes, Mapping) or not file_hashes:
        raise CalibrationIntegrityError("Calibration manifest has no file hashes")
    for filename, expected_hash in file_hashes.items():
        if Path(filename).name != filename:
            raise CalibrationIntegrityError("Calibration manifest contains an unsafe filename")
        artifact = bundle_dir / filename
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise CalibrationIntegrityError(f"Calibration file hash mismatch: {filename}")

    margin_key = str(int(margin_db))
    try:
        coverage_filename = manifest["coverage_tables"][margin_key]
    except (KeyError, TypeError) as exc:
        raise CalibrationIntegrityError(
            f"Calibration does not provide margin {margin_db:g} dB"
        ) from exc
    if coverage_filename not in file_hashes:
        raise CalibrationIntegrityError("Selected coverage table is not hash-protected")

    coverage_rows: dict[str, list[tuple[int, float]]] = {"node": [], "relay": []}
    with (bundle_dir / coverage_filename).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("calibration_id") != expected_calibration_id:
                raise CalibrationIntegrityError("Coverage row calibration ID mismatch")
            if row.get("method_id") != ISLAM_DBPC_POLICY_ID:
                raise CalibrationIntegrityError("Coverage row method ID mismatch")
            if int(row["margin_db"]) != int(margin_db):
                raise CalibrationIntegrityError("Coverage row margin mismatch")
            if row.get("comparison_rule") != "distance_m < max_distance_m":
                raise CalibrationIntegrityError("Coverage row comparison rule mismatch")
            if not _parse_bool(row["use_snr"]):
                entity_type = row["entity_type"]
                coverage_rows.setdefault(entity_type, []).append(
                    (int(row["ptx_dbm"]), float(row["max_distance_m"]))
                )
            else:
                raise CalibrationIntegrityError("Approved calibration requires use_snr=false")

    frozen_coverage: dict[str, tuple[tuple[int, float], ...]] = {}
    for entity_type in ("node", "relay"):
        rows = tuple(sorted(coverage_rows.get(entity_type, [])))
        expected_ladder = tuple(int(v) for v in manifest["ptx_ladders_dbm"][entity_type])
        if tuple(ptx for ptx, _ in rows) != expected_ladder:
            raise CalibrationIntegrityError(f"{entity_type} coverage ladder mismatch")
        if any(b <= a for (_, a), (_, b) in zip(rows, rows[1:])):
            raise CalibrationIntegrityError(f"{entity_type} coverage must increase with PTx")
        frozen_coverage[entity_type] = rows

    return FrozenDBPCCalibration(
        manifest_path=path,
        manifest_sha256=actual_manifest_hash,
        manifest=manifest,
        margin_db=int(margin_db),
        coverage_by_entity=frozen_coverage,
    )


class IslamDBPCController:
    """Apply one frozen DBPC choice per routed node/relay before traffic."""

    def __init__(self, calibration: FrozenDBPCCalibration, model_id: str):
        self.calibration = calibration
        self.model_id = str(model_id)

    def verify_runtime_configuration(self, config: Mapping[str, Any]) -> None:
        constraints = self.calibration.manifest.get("runtime_constraints", {})
        for key, expected in constraints.items():
            actual = config.get(key)
            if isinstance(expected, list):
                actual = (
                    list(actual)
                    if isinstance(actual, Sequence) and not isinstance(actual, str)
                    else actual
                )
            if actual != expected:
                raise CalibrationIntegrityError(
                    f"Runtime configuration {key!r}={actual!r} does not match "
                    f"calibration value {expected!r}"
                )

    def apply(
        self,
        *,
        config: Mapping[str, Any],
        nodes: Sequence[Any],
        relays: Sequence[Any],
        routing_map: Mapping[str, str | None],
        node_map: Mapping[str, Any],
    ) -> tuple[DBPCSelection, ...]:
        self.verify_runtime_configuration(config)
        selections = []
        for entity_type, entities in (("node", nodes), ("relay", relays)):
            expected_ladder = tuple(self.calibration.ptx_ladders[entity_type])
            for entity in entities:
                actual_ladder = tuple(sorted(_get_supported_ptx_values(entity)))
                if actual_ladder != expected_ladder:
                    raise CalibrationIntegrityError(
                        f"{entity.id}: operational PTx ladder {actual_ladder} does not "
                        f"match frozen calibration {expected_ladder}"
                    )
                parent_id = routing_map.get(entity.id)
                if parent_id is None or parent_id not in node_map:
                    raise ValueError(f"{entity.id}: DBPC requires a selected parent")
                distance_m = float(entity.distance_to(node_map[parent_id]))
                selection = self.calibration.select(
                    entity_id=entity.id,
                    entity_type=entity_type,
                    parent_id=parent_id,
                    distance_m=distance_m,
                )

                entity.ptx_mode = "fixed"
                entity.fixed_ptx = selection.selected_ptx_dbm
                entity.default_ptx = selection.selected_ptx_dbm
                entity.tx_power_policy_id = ISLAM_DBPC_POLICY_ID
                entity.tx_power_model_id = self.model_id
                entity.tx_power_calibration_id = self.calibration.calibration_id
                entity.tx_power_calibration_manifest_sha256 = self.calibration.manifest_sha256
                entity.tx_power_calibration_margin_db = self.calibration.margin_db
                entity.tx_power_parent_id = selection.parent_id
                entity.tx_power_parent_distance_m = selection.distance_m
                entity.tx_power_initial_ptx_dbm = selection.selected_ptx_dbm
                entity.tx_power_selected_at_max_ptx = selection.selected_at_max_ptx
                entity.tx_power_outside_estimated_coverage = selection.outside_estimated_coverage
                entity.tx_power_feedback_enabled = False
                selections.append(selection)
        return tuple(selections)


__all__ = [
    "CALIBRATION_VERSION",
    "CalibrationIntegrityError",
    "DBPCSelection",
    "FrozenDBPCCalibration",
    "ISLAM_DBPC_BASELINE_NAME",
    "ISLAM_DBPC_POLICY_ID",
    "IslamDBPCController",
    "load_frozen_calibration",
    "sha256_file",
]
