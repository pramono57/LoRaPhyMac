"""Pre-traffic DBPC-802 fixed-SF7 power selection.

This is the LoRaPhyMAC power-only adaptation of Islam et al. Algorithm 2.
IEEE 802.11ah Equation (15) is used only as the policy-side path-loss
estimate. The simulator's physical propagation remains ITM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from loraphymac.phy.propagation import IEEE80211ah
from loraphymac.runtime.apc_power import _get_supported_ptx_values


ISLAM_DBPC_802_MODEL_ID = "islam_dbpc_802"
ISLAM_DBPC_802_POLICY_ID = "islam_algorithm2_ieee80211ah_fixed_sf_ptx_v1"
ISLAM_DBPC_802_ESTIMATOR_ID = "ieee80211ah_outdoor_eq15_margin0_v1"
ISLAM_DBPC_802_POLICY_MARGIN_DB = 0.0


@dataclass(frozen=True)
class DBPC802Selection:
    """One deterministic Algorithm-2 power choice and its policy evidence."""

    entity_id: str
    entity_type: str
    parent_id: str
    distance_m: float
    frequency_mhz: float
    sf: int
    bandwidth_hz: int
    estimated_path_loss_db: float
    transmitter_gain_dbi: float
    receiver_gain_dbi: float
    receiver_sensitivity_dbm: float
    policy_margin_db: float
    selected_ptx_dbm: int
    estimated_rssi_dbm: float
    estimated_margin_db: float
    estimated_feasible: bool
    selected_at_max_ptx: bool
    outside_estimated_coverage: bool
    max_distance_m: float


class IslamDBPC802Controller:
    """Apply DBPC-802 once after routing and before traffic starts."""

    def __init__(self, model_id: str, estimator: Any | None = None):
        if str(model_id) != ISLAM_DBPC_802_MODEL_ID:
            raise ValueError(
                f"DBPC-802 requires model_id={ISLAM_DBPC_802_MODEL_ID!r}"
            )
        self.model_id = str(model_id)
        self.estimator = IEEE80211ah() if estimator is None else estimator

    @staticmethod
    def verify_runtime_configuration(config: Mapping[str, Any]) -> None:
        if str(config.get("tx_power_model_id")) != ISLAM_DBPC_802_MODEL_ID:
            raise ValueError("DBPC-802 model identity is missing or incorrect")
        if str(config.get("tx_power_policy_id")) != ISLAM_DBPC_802_POLICY_ID:
            raise ValueError("DBPC-802 policy identity is missing or incorrect")
        if config.get("node_ptx_mode") != "fixed" or config.get(
            "relay_ptx_mode"
        ) != "fixed":
            raise ValueError("DBPC-802 requires fixed node and relay PTx modes")
        if list(config.get("node_sf_list", ())) != [7] or int(
            config.get("relay_sf", -1)
        ) != 7:
            raise ValueError("DBPC-802 requires fixed SF7 for nodes and relays")
        if int(config.get("routing_assumed_sf", -1)) != 7:
            raise ValueError("DBPC-802 requires SF7 routing assumptions")
        if int(config.get("node_bw", -1)) != 125_000 or int(
            config.get("relay_bw_hz", -1)
        ) != 125_000:
            raise ValueError("DBPC-802 requires BW125 for nodes and relays")
        if float(config.get("tx_power_calibration_margin_db", math.nan)) != (
            ISLAM_DBPC_802_POLICY_MARGIN_DB
        ):
            raise ValueError("DBPC-802 policy margin must be 0 dB")
        for key in (
            "tx_power_calibration_id",
            "tx_power_calibration_manifest",
            "tx_power_calibration_manifest_sha256",
        ):
            if config.get(key) is not None:
                raise ValueError(f"DBPC-802 must not declare {key}")

    @staticmethod
    def _receiver_sensitivity_dbm(parent: Any, *, sf: int, bw_hz: int) -> float:
        try:
            sensitivity = float(parent.hw.sensitivity[int(bw_hz) // 1000][int(sf)])
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"{parent.id}: receiver sensitivity is unavailable for SF{sf}/BW{bw_hz}"
            ) from exc
        if not math.isfinite(sensitivity):
            raise ValueError(f"{parent.id}: receiver sensitivity must be finite")
        return sensitivity

    @staticmethod
    def _max_distance_m(
        *,
        max_ptx_dbm: int,
        transmitter_gain_dbi: float,
        receiver_gain_dbi: float,
        receiver_sensitivity_dbm: float,
        frequency_mhz: float,
    ) -> float:
        maximum_loss_db = (
            float(max_ptx_dbm)
            + float(transmitter_gain_dbi)
            + float(receiver_gain_dbi)
            - float(receiver_sensitivity_dbm)
            - ISLAM_DBPC_802_POLICY_MARGIN_DB
        )
        exponent = (
            maximum_loss_db
            - 23.3
            - 21.0 * math.log10(float(frequency_mhz) / 900.0)
        ) / 37.6
        return 10.0**exponent

    def select(
        self,
        *,
        entity: Any,
        entity_type: str,
        parent: Any,
        distance_m: float,
        frequency_mhz: float,
        sf: int,
        bandwidth_hz: int,
    ) -> DBPC802Selection:
        distance = float(distance_m)
        frequency = float(frequency_mhz)
        if not math.isfinite(distance) or distance <= 0.0:
            raise ValueError(f"{entity.id}: distance_m must be finite and > 0")
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("DBPC-802 frequency must be finite and > 0")

        path_loss_db = float(
            self.estimator.pathloss(
                d_m=distance,
                freq_mhz=frequency,
            )
        )
        if not math.isfinite(path_loss_db):
            raise ValueError("DBPC-802 estimated path loss must be finite")
        transmitter_gain = float(entity.gain)
        receiver_gain = float(parent.gain)
        sensitivity = self._receiver_sensitivity_dbm(
            parent,
            sf=int(sf),
            bw_hz=int(bandwidth_hz),
        )
        ladder = tuple(sorted(_get_supported_ptx_values(entity)))

        selected_ptx = ladder[-1]
        estimated_rssi = (
            selected_ptx + transmitter_gain + receiver_gain - path_loss_db
        )
        for candidate in ladder:
            candidate_rssi = (
                candidate + transmitter_gain + receiver_gain - path_loss_db
            )
            # Islam Algorithm 2 uses d <= coverage. At the link-budget
            # boundary this is estimated RSSI >= sensitivity + policy margin.
            if candidate_rssi >= sensitivity + ISLAM_DBPC_802_POLICY_MARGIN_DB:
                selected_ptx = candidate
                estimated_rssi = candidate_rssi
                break

        estimated_margin = (
            estimated_rssi
            - sensitivity
            - ISLAM_DBPC_802_POLICY_MARGIN_DB
        )
        estimated_feasible = estimated_margin >= 0.0
        return DBPC802Selection(
            entity_id=str(entity.id),
            entity_type=str(entity_type),
            parent_id=str(parent.id),
            distance_m=distance,
            frequency_mhz=frequency,
            sf=int(sf),
            bandwidth_hz=int(bandwidth_hz),
            estimated_path_loss_db=path_loss_db,
            transmitter_gain_dbi=transmitter_gain,
            receiver_gain_dbi=receiver_gain,
            receiver_sensitivity_dbm=sensitivity,
            policy_margin_db=ISLAM_DBPC_802_POLICY_MARGIN_DB,
            selected_ptx_dbm=int(selected_ptx),
            estimated_rssi_dbm=float(estimated_rssi),
            estimated_margin_db=float(estimated_margin),
            estimated_feasible=estimated_feasible,
            selected_at_max_ptx=int(selected_ptx) == int(ladder[-1]),
            outside_estimated_coverage=not estimated_feasible,
            max_distance_m=self._max_distance_m(
                max_ptx_dbm=ladder[-1],
                transmitter_gain_dbi=transmitter_gain,
                receiver_gain_dbi=receiver_gain,
                receiver_sensitivity_dbm=sensitivity,
                frequency_mhz=frequency,
            ),
        )

    def apply(
        self,
        *,
        config: Mapping[str, Any],
        nodes: Sequence[Any],
        relays: Sequence[Any],
        routing_map: Mapping[str, str | None],
        node_map: Mapping[str, Any],
    ) -> tuple[DBPC802Selection, ...]:
        self.verify_runtime_configuration(config)
        selections = []
        for entity_type, entities in (("node", nodes), ("relay", relays)):
            sf = 7 if entity_type == "node" else int(config["relay_sf"])
            bandwidth_hz = int(
                config["node_bw"]
                if entity_type == "node"
                else config["relay_bw_hz"]
            )
            for entity in entities:
                parent_id = routing_map.get(entity.id)
                if parent_id is None or parent_id not in node_map:
                    raise ValueError(f"{entity.id}: DBPC-802 requires a selected parent")
                parent = node_map[parent_id]
                distance_m = float(entity.distance_to(parent))
                frequency_mhz = float(
                    entity.ch_plan.get_freq(int(config["routing_ch_id"]))
                )
                selection = self.select(
                    entity=entity,
                    entity_type=entity_type,
                    parent=parent,
                    distance_m=distance_m,
                    frequency_mhz=frequency_mhz,
                    sf=sf,
                    bandwidth_hz=bandwidth_hz,
                )

                entity.ptx_mode = "fixed"
                entity.fixed_ptx = selection.selected_ptx_dbm
                entity.default_ptx = selection.selected_ptx_dbm
                entity.tx_power_policy_id = ISLAM_DBPC_802_POLICY_ID
                entity.tx_power_model_id = self.model_id
                entity.tx_power_calibration_id = None
                entity.tx_power_calibration_manifest_sha256 = None
                entity.tx_power_calibration_margin_db = None
                entity.tx_power_parent_id = selection.parent_id
                entity.tx_power_parent_distance_m = selection.distance_m
                entity.tx_power_initial_ptx_dbm = selection.selected_ptx_dbm
                entity.tx_power_selected_at_max_ptx = selection.selected_at_max_ptx
                entity.tx_power_outside_estimated_coverage = (
                    selection.outside_estimated_coverage
                )
                entity.tx_power_feedback_enabled = False
                entity.tx_power_estimator_id = ISLAM_DBPC_802_ESTIMATOR_ID
                entity.tx_power_estimator_frequency_mhz = selection.frequency_mhz
                entity.tx_power_estimated_path_loss_db = (
                    selection.estimated_path_loss_db
                )
                entity.tx_power_receiver_sensitivity_dbm = (
                    selection.receiver_sensitivity_dbm
                )
                entity.tx_power_estimated_rssi_dbm = selection.estimated_rssi_dbm
                entity.tx_power_estimated_margin_db = selection.estimated_margin_db
                entity.tx_power_policy_margin_db = selection.policy_margin_db
                selections.append(selection)
        return tuple(selections)


__all__ = [
    "DBPC802Selection",
    "ISLAM_DBPC_802_ESTIMATOR_ID",
    "ISLAM_DBPC_802_MODEL_ID",
    "ISLAM_DBPC_802_POLICY_ID",
    "ISLAM_DBPC_802_POLICY_MARGIN_DB",
    "IslamDBPC802Controller",
]
