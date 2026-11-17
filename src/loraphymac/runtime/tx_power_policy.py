"""Thin dispatcher for pre-traffic transmit-power policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from loraphymac.runtime.islam_dbpc import (
    ISLAM_DBPC_POLICY_ID,
    IslamDBPCController,
    load_frozen_calibration,
)
from loraphymac.runtime.islam_dbpc_802 import (
    ISLAM_DBPC_802_ESTIMATOR_ID,
    ISLAM_DBPC_802_POLICY_ID,
    IslamDBPC802Controller,
)


LEGACY_POLICY_ID = "legacy"
SUPPORTED_TX_POWER_POLICIES = frozenset(
    {LEGACY_POLICY_ID, ISLAM_DBPC_POLICY_ID, ISLAM_DBPC_802_POLICY_ID}
)


@dataclass(frozen=True)
class TxPowerPolicyApplication:
    """Metadata and deterministic per-entity outcomes from policy application."""

    policy_id: str
    model_id: str | None
    selections: tuple[Any, ...] = ()
    run_metadata: Mapping[str, Any] | None = None


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _topology_hash(entities: Sequence[Any]) -> str:
    rows = []
    for entity in sorted(entities, key=lambda item: str(item.id)):
        rows.append(
            {
                "id": str(entity.id),
                "x_m": float(entity.loc.x),
                "y_m": float(entity.loc.y),
                "height_m": float(entity.h),
            }
        )
    return _stable_hash(rows)


def _routing_hash(routing_map: Mapping[str, str | None]) -> str:
    return _stable_hash(sorted((str(source), target) for source, target in routing_map.items()))


def apply_tx_power_policy(
    *,
    config: Mapping[str, Any],
    nodes: Sequence[Any],
    relays: Sequence[Any],
    gateway: Any,
    routing_map: Mapping[str, str | None],
    node_map: Mapping[str, Any],
) -> TxPowerPolicyApplication:
    """Dispatch one selected policy without altering the legacy algorithm."""
    policy_id = str(config.get("tx_power_policy_id", LEGACY_POLICY_ID)).strip().lower()
    if policy_id not in SUPPORTED_TX_POWER_POLICIES:
        raise ValueError(f"Unknown transmit-power policy {policy_id!r}")
    model_id = config.get("tx_power_model_id")

    # Default simulations have no explicit model identity.  Preserve their
    # state and output schema exactly.
    if policy_id == LEGACY_POLICY_ID:
        metadata = {}
        if model_id is not None:
            metadata = {
                "tx_power_model_id": str(model_id),
                "tx_power_policy_id": LEGACY_POLICY_ID,
                "tx_power_calibration_id": None,
                "tx_power_calibration_manifest_sha256": None,
                "topology_hash": _topology_hash([*nodes, *relays, gateway]),
                "selected_parent_routing_hash": _routing_hash(routing_map),
                "traffic_schedule_provenance": (
                    "seeded_python_random_node_initial_delay_and_jitter_v1"
                ),
            }
        return TxPowerPolicyApplication(
            policy_id=policy_id,
            model_id=None if model_id is None else str(model_id),
            run_metadata=metadata,
        )

    if policy_id == ISLAM_DBPC_802_POLICY_ID:
        controller = IslamDBPC802Controller(str(model_id))
        selections = controller.apply(
            config=config,
            nodes=nodes,
            relays=relays,
            routing_map=routing_map,
            node_map=node_map,
        )
        metadata = {
            "tx_power_model_id": str(model_id),
            "tx_power_policy_id": policy_id,
            "tx_power_calibration_id": None,
            "tx_power_calibration_manifest_sha256": None,
            "tx_power_calibration_margin_db": None,
            "tx_power_estimator_id": ISLAM_DBPC_802_ESTIMATOR_ID,
            "tx_power_estimator_frequency_mhz": (
                selections[0].frequency_mhz if selections else None
            ),
            "tx_power_policy_margin_db": 0.0,
            "tx_power_feedback_enabled": False,
            "tx_power_selection_timing": "once_after_routing_before_traffic",
            "tx_power_saturated_entity_count": sum(
                int(item.selected_at_max_ptx) for item in selections
            ),
            "tx_power_outside_coverage_entity_count": sum(
                int(item.outside_estimated_coverage) for item in selections
            ),
            "topology_hash": _topology_hash([*nodes, *relays, gateway]),
            "selected_parent_routing_hash": _routing_hash(routing_map),
            "traffic_schedule_provenance": (
                "seeded_python_random_node_initial_delay_and_jitter_v1"
            ),
        }
        return TxPowerPolicyApplication(
            policy_id=policy_id,
            model_id=str(model_id),
            selections=selections,
            run_metadata=metadata,
        )

    calibration = load_frozen_calibration(
        config["tx_power_calibration_manifest"],
        expected_manifest_sha256=config["tx_power_calibration_manifest_sha256"],
        expected_calibration_id=config["tx_power_calibration_id"],
        margin_db=int(config["tx_power_calibration_margin_db"]),
    )
    controller = IslamDBPCController(calibration, str(model_id))
    selections = controller.apply(
        config=config,
        nodes=nodes,
        relays=relays,
        routing_map=routing_map,
        node_map=node_map,
    )
    metadata = {
        "tx_power_model_id": str(model_id),
        "tx_power_policy_id": policy_id,
        "tx_power_calibration_id": calibration.calibration_id,
        "tx_power_calibration_manifest_sha256": calibration.manifest_sha256,
        "tx_power_calibration_margin_db": calibration.margin_db,
        "tx_power_feedback_enabled": False,
        "tx_power_selection_timing": "once_after_routing_before_traffic",
        "tx_power_saturated_entity_count": sum(
            int(item.selected_at_max_ptx) for item in selections
        ),
        "tx_power_outside_coverage_entity_count": sum(
            int(item.outside_estimated_coverage) for item in selections
        ),
        "topology_hash": _topology_hash([*nodes, *relays, gateway]),
        "selected_parent_routing_hash": _routing_hash(routing_map),
        "traffic_schedule_provenance": "seeded_python_random_node_initial_delay_and_jitter_v1",
    }
    return TxPowerPolicyApplication(
        policy_id=policy_id,
        model_id=str(model_id),
        selections=selections,
        run_metadata=metadata,
    )


__all__ = [
    "ISLAM_DBPC_802_POLICY_ID",
    "LEGACY_POLICY_ID",
    "SUPPORTED_TX_POWER_POLICIES",
    "TxPowerPolicyApplication",
    "apply_tx_power_policy",
]
