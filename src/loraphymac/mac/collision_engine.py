# ============================================================================
# mac/collision_engine.py
# Physical DATA_DOWN / DATA_DOWN_FWD + 
# Relay APC RX cleanup, APC TX index, and timing validation
#
# — RF-Collision Engine (receiver-scoped, spectral-overlap aware)
# - RF eligibility uses direct time overlap and physical spectrum overlap.
# - ch_id is retained as trace metadata; freq_mhz + bw_hz are authoritative.
# - Multi-SF interference supports ideal orthogonality or a directional SIR matrix.
# - Two modes:
#   (A) "capture": each desired packet is evaluated only against packets
#       that overlap it directly. This avoids transitive-overlap capture.
#   (B) "sinr":    survivor(s) based on SINR >= required_snr(sf) (more realistic)
# - Optional destructive=True applies only to legacy SINR grouping
# - Output invariant: survivors + failed == input packets
# - NOTE: Does NOT model demodulator pool exhaustion (do it in DemodulatorPool)
# ============================================================================

from __future__ import annotations
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

from loraphymac.runtime.packet import stable_packet_object_id


class CollisionEngine:
    """Receiver-scoped LoRa collision and capture model.

    Capture mode evaluates every direct time-and-spectrum-overlap edge.
    Partial spectral overlap scales interferer power. The desired packet
    survives only when all directional SIR margins pass; the dominant
    interferer is the edge with the smallest margin.
    """

    CAPTURE_MODEL_VERSION = "pairwise_spectral_sir_v3"
    SINR_MODEL_VERSION = "spectral_group_sinr_v2"
    SPECTRAL_MODEL_VERSION = "rectangular_overlap_fraction_v1"
    RAKIBUL_MATRIX_VERSION = "rakibul_islam_2023_table4"
    RAKIBUL_2023_SIR_MATRIX = {
        7: {7: 6.0, 8: -16.0, 9: -18.0, 10: -19.0, 11: -19.0, 12: -20.0},
        8: {7: -24.0, 8: 6.0, 9: -20.0, 10: -22.0, 11: -22.0, 12: -22.0},
        9: {7: -27.0, 8: -27.0, 9: 6.0, 10: -23.0, 11: -25.0, 12: -25.0},
        10: {7: -30.0, 8: -30.0, 9: -30.0, 10: 6.0, 11: -26.0, 12: -28.0},
        11: {7: -33.0, 8: -33.0, 9: -33.0, 10: -33.0, 11: 6.0, 12: -29.0},
        12: {7: -36.0, 8: -36.0, 9: -36.0, 10: -36.0, 11: -36.0, 12: 6.0},
    }
    SIR_MATRIX_PRESETS = {
        RAKIBUL_MATRIX_VERSION: RAKIBUL_2023_SIR_MATRIX,
    }

    def __init__(
        self,
        mode: str = "capture",              # "sinr" (recommended) or "capture"
        capture_threshold_dB: float = 6.0,  # used in "capture" mode ref: Islam et al., 2023
        enforce_sf_match: Optional[bool] = True,
        sf_interference_mode: Optional[str] = None,
        sir_matrix: Optional[Mapping] = None,
        sir_matrix_version: Optional[str] = None,
        sir_matrix_preset: str = RAKIBUL_MATRIX_VERSION,
        noise_floor_dBm: float = -120.0,    # approximate; adjust to BW+NF if you have it
        destructive: bool = True,           # if True: at most ONE survivor per overlap group
    ):
        self.mode = str(mode).lower().strip()
        if self.mode not in ("sinr", "capture"):
            raise ValueError("collision mode must be 'capture' or 'sinr'")
        self.capture_threshold = float(capture_threshold_dB)
        if sf_interference_mode is None:
            # Compatibility for legacy constructor callers. New experiments
            # should select ideal_orthogonal or sir_matrix explicitly.
            sf_interference_mode = (
                "ideal_orthogonal"
                if enforce_sf_match is not False
                else "common_threshold"
            )
        self.sf_interference_mode = str(sf_interference_mode).lower().strip()
        if self.sf_interference_mode not in (
            "ideal_orthogonal",
            "sir_matrix",
            "common_threshold",
        ):
            raise ValueError(
                "sf_interference_mode must be 'ideal_orthogonal' or "
                "'sir_matrix'"
            )
        self.enforce_sf_match = self.sf_interference_mode == "ideal_orthogonal"

        if sir_matrix is not None:
            if not sir_matrix_version:
                raise ValueError(
                    "A custom sir_matrix requires a non-empty sir_matrix_version"
                )
            self.sir_matrix = self._normalize_sir_matrix(sir_matrix)
            self.sir_matrix_version = str(sir_matrix_version)
            self.sir_matrix_preset = None
        else:
            preset = str(sir_matrix_preset).lower().strip()
            if preset not in self.SIR_MATRIX_PRESETS:
                raise ValueError(f"Unknown SIR matrix preset: {sir_matrix_preset!r}")
            self.sir_matrix = self._normalize_sir_matrix(
                self.SIR_MATRIX_PRESETS[preset]
            )
            self.sir_matrix_version = preset
            self.sir_matrix_preset = preset
        self.noise_floor_dBm = float(noise_floor_dBm)
        self.destructive = bool(destructive)
        self._resolution_sequence = 0
        self.last_decision_records: List[dict] = []
        self.last_interference_edge_records: List[dict] = []

    @property
    def model_version(self) -> str:
        return (
            self.CAPTURE_MODEL_VERSION
            if self.mode == "capture"
            else self.SINR_MODEL_VERSION
        )

    @staticmethod
    def _normalize_sir_matrix(matrix: Mapping) -> Dict[int, Dict[int, float]]:
        normalized: Dict[int, Dict[int, float]] = {}
        for desired_sf, row in matrix.items():
            if not isinstance(row, Mapping):
                raise ValueError("Each sir_matrix row must be a mapping")
            desired_key = int(desired_sf)
            normalized[desired_key] = {
                int(interferer_sf): float(threshold)
                for interferer_sf, threshold in row.items()
            }
        if not normalized:
            raise ValueError("sir_matrix must not be empty")
        return normalized

    # -----------------------------
    # Helpers: get/set compatibility
    # -----------------------------
    @staticmethod
    def _get(pkt: Any, name: str, default=None):
        if hasattr(pkt, name):
            return getattr(pkt, name)
        try:
            return pkt[name]
        except Exception:
            return default

    @staticmethod
    def _set(pkt: Any, name: str, value):
        try:
            setattr(pkt, name, value)
            return
        except Exception:
            pass
        try:
            pkt[name] = value
        except Exception:
            pass

    def _set_reason(self, pkt: Any, reason: str) -> None:
        self._set(pkt, "reason", reason)

    # -----------------------------
    # Timing normalization
    # -----------------------------
    def _ensure_timing(self, pkt: Any) -> None:
        rx_start = self._get(pkt, "rx_start", None)
        rx_end = self._get(pkt, "rx_end", None)
        if rx_start is not None and rx_end is not None:
            return

        t_arrive = self._get(pkt, "t_arrive", None)
        t_end = self._get(pkt, "t_end", None)
        if t_arrive is not None and t_end is not None:
            self._set(pkt, "rx_start", t_arrive)
            self._set(pkt, "rx_end", t_end)
            return

        tx_start = float(self._get(pkt, "tx_start", 0.0))
        toa = float(self._get(pkt, "toa", 0.0))
        self._set(pkt, "rx_start", tx_start)
        self._set(pkt, "rx_end", tx_start + toa)

    @staticmethod
    def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
        return (a0 < b1) and (a1 > b0)

    @staticmethod
    def _overlap_duration(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(0.0, min(a1, b1) - max(a0, b0))

    # -----------------------------
    # Physical spectrum and directional SF interference
    # -----------------------------
    @staticmethod
    def _finite_float(value: Any, field: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CollisionEngine: invalid {field}={value!r}") from exc
        if not math.isfinite(result):
            raise ValueError(f"CollisionEngine: non-finite {field}={value!r}")
        return result

    def _packet_sf(self, packet: Any) -> int:
        value = self._get(
            packet,
            "sf",
            self._get(packet, "spreading_factor", None),
        )
        try:
            sf = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CollisionEngine: invalid sf={value!r}") from exc
        return sf

    def _validate_packet(self, packet: Any) -> None:
        self._finite_float(self._get(packet, "rx_start", None), "rx_start")
        self._finite_float(self._get(packet, "rx_end", None), "rx_end")
        self._finite_float(self._get(packet, "rssi", None), "rssi")
        self._finite_float(self._get(packet, "freq_mhz", None), "freq_mhz")
        bw_hz = self._finite_float(self._get(packet, "bw_hz", None), "bw_hz")
        if bw_hz <= 0.0:
            raise ValueError("CollisionEngine: bw_hz must be > 0")
        self._packet_sf(packet)

    def _spectral_overlap_hz(self, p1: Any, p2: Any) -> float:
        center_1_hz = self._finite_float(
            self._get(p1, "freq_mhz", None), "freq_mhz"
        ) * 1_000_000.0
        center_2_hz = self._finite_float(
            self._get(p2, "freq_mhz", None), "freq_mhz"
        ) * 1_000_000.0
        bw_1_hz = self._finite_float(self._get(p1, "bw_hz", None), "bw_hz")
        bw_2_hz = self._finite_float(self._get(p2, "bw_hz", None), "bw_hz")
        low_1, high_1 = center_1_hz - bw_1_hz / 2.0, center_1_hz + bw_1_hz / 2.0
        low_2, high_2 = center_2_hz - bw_2_hz / 2.0, center_2_hz + bw_2_hz / 2.0
        return max(0.0, min(high_1, high_2) - max(low_1, low_2))

    def _sf_pair_is_eligible(self, desired_sf: int, interferer_sf: int) -> bool:
        if self.sf_interference_mode == "ideal_orthogonal":
            return int(desired_sf) == int(interferer_sf)
        return True

    def required_sir_dB(self, desired_sf: int, interferer_sf: int) -> float:
        if self.sf_interference_mode in ("ideal_orthogonal", "common_threshold"):
            return self.capture_threshold
        try:
            return self.sir_matrix[int(desired_sf)][int(interferer_sf)]
        except KeyError as exc:
            raise ValueError(
                "SIR matrix "
                f"{self.sir_matrix_version!r} has no threshold for "
                f"desired SF{desired_sf} versus interferer SF{interferer_sf}"
            ) from exc

    def _pair_evaluation(self, packet: Any, other: Any) -> Optional[dict]:
        p0 = float(self._get(packet, "rx_start", 0.0))
        p1 = float(self._get(packet, "rx_end", 0.0))
        q0 = float(self._get(other, "rx_start", 0.0))
        q1 = float(self._get(other, "rx_end", 0.0))
        if not self._overlap(p0, p1, q0, q1):
            return None

        desired_sf = self._packet_sf(packet)
        interferer_sf = self._packet_sf(other)
        if not self._sf_pair_is_eligible(desired_sf, interferer_sf):
            return None

        spectral_overlap_hz = self._spectral_overlap_hz(packet, other)
        if spectral_overlap_hz <= 0.0:
            return None

        interferer_bw_hz = self._finite_float(
            self._get(other, "bw_hz", None), "bw_hz"
        )
        overlap_fraction = min(1.0, spectral_overlap_hz / interferer_bw_hz)
        interferer_rssi_dbm = self._finite_float(
            self._get(other, "rssi", None), "rssi"
        )
        effective_interferer_rssi_dbm = (
            interferer_rssi_dbm + 10.0 * math.log10(overlap_fraction)
        )
        desired_rssi_dbm = self._finite_float(
            self._get(packet, "rssi", None), "rssi"
        )
        required_sir_db = self.required_sir_dB(desired_sf, interferer_sf)
        pair_sir_db = desired_rssi_dbm - effective_interferer_rssi_dbm
        pair_margin_db = pair_sir_db - required_sir_db
        return {
            "interferer": other,
            "time_overlap_s": self._overlap_duration(p0, p1, q0, q1),
            "spectral_overlap_hz": spectral_overlap_hz,
            "spectral_overlap_fraction": overlap_fraction,
            "effective_interferer_rssi_dbm": effective_interferer_rssi_dbm,
            "required_sir_db": required_sir_db,
            "pair_sir_db": pair_sir_db,
            "pair_margin_db": pair_margin_db,
            "pair_pass": pair_margin_db >= 0.0,
        }

    def _direct_interference_edges(
        self, packet: Any, packets: List[Any]
    ) -> List[dict]:
        edges = []
        for other in packets:
            if other is packet:
                continue
            edge = self._pair_evaluation(packet, other)
            if edge is not None:
                edges.append(edge)
        return edges

    def _direct_interferers(self, packet: Any, packets: List[Any]) -> List[Any]:
        return [
            edge["interferer"]
            for edge in self._direct_interference_edges(packet, packets)
        ]

    # -----------------------------
    # PHY utilities
    # -----------------------------
    @staticmethod
    def _dbm_to_mw(dbm: float) -> float:
        return 10.0 ** (dbm / 10.0)

    @staticmethod
    def _mw_to_db(x_mw: float) -> float:
        return 10.0 * math.log10(max(x_mw, 1e-30))

    @staticmethod
    def required_snr_dB(sf: int) -> float:
        """
        Typical LoRa demod minimum SNR (approx, commonly used in simulators).
        Adjust if you have a calibrated table per BW/CR.
        """
        table = {
            5: -2.5,
            6: -5.0,
            7: -7.5,
            8: -10.0,
            9: -12.5,
            10: -15.0,
            11: -17.5,
            12: -20.0,
        }
        return table.get(int(sf), -7.5)

    def _sinr_dB(self, pkt: Any, interferers: List[Any]) -> float:
        """
        Group-based evaluation:
        - interferers = packet list in the same group
        - BUT only count those that time-overlap with pkt (RF-correct)
        """
        s_dbm = float(self._get(pkt, "rssi", -999.0))
        s_mw = self._dbm_to_mw(s_dbm)
    
        i_mw = 0.0
        for q in interferers:
            if q is pkt:
                continue
            edge = self._pair_evaluation(pkt, q)
            if edge is None:
                continue
            i_mw += self._dbm_to_mw(
                edge["effective_interferer_rssi_dbm"]
            )
    
        n_mw = self._dbm_to_mw(self.noise_floor_dBm)
        sinr = s_mw / (i_mw + n_mw)
        return self._mw_to_db(sinr)

    # -----------------------------
    # Build RF-eligibility groups (legacy SINR batching only)
    # -----------------------------
    def _build_groups(self, packets: List[Any]) -> List[List[Any]]:
        if not packets:
            return []

        for p in packets:
            self._ensure_timing(p)

        n = len(packets)
        visited = [False] * n
        groups: List[List[Any]] = []

        for i in range(n):
            if visited[i]:
                continue

            stack = [i]
            visited[i] = True
            comp_idx = []

            while stack:
                u = stack.pop()
                comp_idx.append(u)

                pu = packets[u]

                for v in range(n):
                    if visited[v] or v == u:
                        continue

                    pv = packets[v]
                    if self._pair_evaluation(pu, pv) is None:
                        continue

                    visited[v] = True
                    stack.append(v)

            groups.append([packets[k] for k in comp_idx])

        return groups

    def build_rf_groups(self, packets: List[Any]) -> List[List[Any]]:
        """Return connected components of direct RF-eligible packet edges."""
        return self._build_groups(packets)

    def packets_share_rf_edge(self, first: Any, second: Any) -> bool:
        """Return whether two packets are direct RF-interference candidates."""
        return self._pair_evaluation(first, second) is not None

    # -----------------------------
    # Paper-grade decision evidence
    # -----------------------------
    def _interference_edge_record(
        self,
        *,
        packet: Any,
        edge: dict,
        resolution_id: int,
        receiver_id: Optional[str],
        decision_time: Optional[float],
        dominant: bool,
    ) -> dict:
        interferer = edge["interferer"]
        return {
            "collision_model_version": self.model_version,
            "spectral_model_version": self.SPECTRAL_MODEL_VERSION,
            "collision_mode": self.mode,
            "sf_interference_mode": self.sf_interference_mode,
            "sir_matrix_version": (
                self.sir_matrix_version
                if self.sf_interference_mode == "sir_matrix"
                else None
            ),
            "resolution_id": int(resolution_id),
            "receiver_id": receiver_id,
            "decision_time": decision_time,
            "desired_packet_object_id": stable_packet_object_id(packet),
            "desired_packet_id": self._get(packet, "packet_id", None),
            "desired_packet_uid": self._get(packet, "packet_uid", None),
            "desired_src_id": self._get(packet, "src_id", None),
            "desired_fcnt": self._get(packet, "fcnt", None),
            "desired_current_tx": self._get(packet, "current_tx", None),
            "desired_prev_hop": self._get(packet, "prev_hop", None),
            "desired_next_hop": self._get(packet, "next_hop", None),
            "desired_hop_index": self._get(packet, "hop_index", None),
            "desired_mtype": self._get(packet, "mtype", None),
            "desired_ch_id": self._get(packet, "ch_id", None),
            "desired_freq_mhz": self._get(packet, "freq_mhz", None),
            "desired_sf": self._packet_sf(packet),
            "desired_bw_hz": self._get(packet, "bw_hz", None),
            "desired_rssi_dbm": self._get(packet, "rssi", None),
            "desired_rx_start": self._get(packet, "rx_start", None),
            "desired_rx_end": self._get(packet, "rx_end", None),
            "interferer_packet_object_id": stable_packet_object_id(interferer),
            "interferer_packet_id": self._get(interferer, "packet_id", None),
            "interferer_packet_uid": self._get(interferer, "packet_uid", None),
            "interferer_src_id": self._get(interferer, "src_id", None),
            "interferer_fcnt": self._get(interferer, "fcnt", None),
            "interferer_current_tx": self._get(
                interferer, "current_tx", None
            ),
            "interferer_prev_hop": self._get(interferer, "prev_hop", None),
            "interferer_next_hop": self._get(interferer, "next_hop", None),
            "interferer_hop_index": self._get(interferer, "hop_index", None),
            "interferer_mtype": self._get(interferer, "mtype", None),
            "interferer_ch_id": self._get(interferer, "ch_id", None),
            "interferer_freq_mhz": self._get(interferer, "freq_mhz", None),
            "interferer_sf": self._packet_sf(interferer),
            "interferer_bw_hz": self._get(interferer, "bw_hz", None),
            "interferer_rssi_dbm": self._get(interferer, "rssi", None),
            "interferer_rx_start": self._get(interferer, "rx_start", None),
            "interferer_rx_end": self._get(interferer, "rx_end", None),
            "time_overlap_s": edge["time_overlap_s"],
            "spectral_overlap_hz": edge["spectral_overlap_hz"],
            "spectral_overlap_fraction": edge["spectral_overlap_fraction"],
            "effective_interferer_rssi_dbm": edge[
                "effective_interferer_rssi_dbm"
            ],
            "required_sir_db": edge["required_sir_db"],
            "pair_sir_db": edge["pair_sir_db"],
            "pair_margin_db": edge["pair_margin_db"],
            "pair_pass": bool(edge["pair_pass"]),
            "dominant_interferer": bool(dominant),
        }

    def _decision_record(
        self,
        *,
        packet: Any,
        interferers: List[Any],
        resolution_id: int,
        receiver_id: Optional[str],
        decision_time: Optional[float],
        outcome: str,
        survived: bool,
        captured: bool,
        strongest_interferer: Any = None,
        dominant_interferer: Any = None,
        capture_delta_db: Optional[float] = None,
        required_sir_db: Optional[float] = None,
        min_pair_margin_db: Optional[float] = None,
        dominant_effective_interferer_rssi_dbm: Optional[float] = None,
    ) -> dict:
        p0 = float(self._get(packet, "rx_start", 0.0))
        p1 = float(self._get(packet, "rx_end", 0.0))
        overlaps = [
            self._overlap_duration(
                p0,
                p1,
                float(self._get(q, "rx_start", 0.0)),
                float(self._get(q, "rx_end", 0.0)),
            )
            for q in interferers
        ]
        return {
            "collision_model_version": self.model_version,
            "spectral_model_version": self.SPECTRAL_MODEL_VERSION,
            "collision_mode": self.mode,
            "sf_interference_mode": self.sf_interference_mode,
            "sir_matrix_version": (
                self.sir_matrix_version
                if self.sf_interference_mode == "sir_matrix"
                else None
            ),
            "resolution_id": int(resolution_id),
            "receiver_id": receiver_id,
            "decision_time": decision_time,
            "packet_object_id": stable_packet_object_id(packet),
            "packet_id": self._get(packet, "packet_id", None),
            "packet_uid": self._get(packet, "packet_uid", None),
            "src_id": self._get(packet, "src_id", None),
            "fcnt": self._get(packet, "fcnt", None),
            "current_tx": self._get(packet, "current_tx", None),
            "prev_hop": self._get(packet, "prev_hop", None),
            "next_hop": self._get(packet, "next_hop", None),
            "hop_index": self._get(packet, "hop_index", None),
            "mtype": self._get(packet, "mtype", None),
            "ch_id": self._get(packet, "ch_id", None),
            "freq_mhz": self._get(packet, "freq_mhz", None),
            "sf": self._get(packet, "sf", self._get(packet, "spreading_factor", None)),
            "bw_hz": self._get(packet, "bw_hz", None),
            "rx_start": p0,
            "rx_end": p1,
            "rssi_dbm": self._get(packet, "rssi", None),
            "outcome": str(outcome),
            "survived": bool(survived),
            "captured": bool(captured),
            "demod_admitted": self._get(packet, "demod_admitted", None),
            "demod_reason": self._get(packet, "demod_reason", None),
            "demod_pool": self._get(packet, "demod_pool", None),
            "demod_acquire_time": self._get(packet, "demod_acquire_time", None),
            "demod_release_time": self._get(packet, "demod_release_time", None),
            "direct_interferer_count": int(len(interferers)),
            "contender_count": int(1 + len(interferers)),
            "strongest_interferer_packet_id": self._get(
                strongest_interferer, "packet_id", None
            ) if strongest_interferer is not None else None,
            "strongest_interferer_src_id": self._get(
                strongest_interferer, "src_id", None
            ) if strongest_interferer is not None else None,
            "strongest_interferer_rssi_dbm": self._get(
                strongest_interferer, "rssi", None
            ) if strongest_interferer is not None else None,
            "dominant_interferer_packet_id": self._get(
                dominant_interferer, "packet_id", None
            ) if dominant_interferer is not None else None,
            "dominant_interferer_src_id": self._get(
                dominant_interferer, "src_id", None
            ) if dominant_interferer is not None else None,
            "dominant_interferer_rssi_dbm": self._get(
                dominant_interferer, "rssi", None
            ) if dominant_interferer is not None else None,
            "dominant_effective_interferer_rssi_dbm": (
                dominant_effective_interferer_rssi_dbm
            ),
            "capture_delta_db": capture_delta_db,
            "capture_threshold_db": (
                required_sir_db
                if required_sir_db is not None
                else self.capture_threshold
            ),
            "required_sir_db": required_sir_db,
            "min_pair_margin_db": min_pair_margin_db,
            "max_pair_overlap_s": max(overlaps) if overlaps else 0.0,
            "total_pair_overlap_s": sum(overlaps),
            "enforce_sf_match": self.enforce_sf_match,
            "destructive": self.destructive,
            "sinr_db": self._get(packet, "sinr_dB", None),
            "reason": self._get(packet, "reason", None),
        }

    def _publish_records(
        self,
        records: List[dict],
        edge_records: List[dict],
        statistics: Any,
    ) -> None:
        self.last_decision_records = [dict(row) for row in records]
        self.last_interference_edge_records = [
            dict(row) for row in edge_records
        ]
        recorder = getattr(statistics, "record_collision_decision", None)
        if callable(recorder):
            for row in records:
                recorder(**row)
        edge_recorder = getattr(
            statistics,
            "record_collision_interference_edge",
            None,
        )
        if callable(edge_recorder):
            for row in edge_records:
                edge_recorder(**row)

    # -----------------------------
    # Public API
    # -----------------------------
    def resolve(
        self,
        packets: List[Any],
        receiver_id: Optional[str] = None,
        event_time: Optional[float] = None,
        statistics: Any = None,
    ) -> Tuple[List[Any], List[Any]]:
        """Resolve receiver-scoped RF collisions and emit one decision per packet."""
        if not packets:
            self.last_decision_records = []
            self.last_interference_edge_records = []
            return [], []

        for packet in packets:
            self._ensure_timing(packet)
            self._validate_packet(packet)

        self._resolution_sequence += 1
        resolution_id = self._resolution_sequence
        if event_time is None:
            event_time = max(float(self._get(p, "rx_end", 0.0)) for p in packets)
        else:
            event_time = float(event_time)

        survivors: List[Any] = []
        failed: List[Any] = []
        records: List[dict] = []
        edge_records: List[dict] = []

        if self.mode == "capture":
            for packet in packets:
                direct_edges = self._direct_interference_edges(packet, packets)
                interferers = [edge["interferer"] for edge in direct_edges]
                if not direct_edges:
                    self._set(packet, "captured", False)
                    survivors.append(packet)
                    records.append(self._decision_record(
                        packet=packet,
                        interferers=interferers,
                        resolution_id=resolution_id,
                        receiver_id=receiver_id,
                        decision_time=event_time,
                        outcome="SINGLETON_SURVIVOR",
                        survived=True,
                        captured=False,
                    ))
                    continue

                dominant_edge = min(
                    direct_edges,
                    key=lambda edge: edge["pair_margin_db"],
                )
                dominant_interferer = dominant_edge["interferer"]
                strongest_interferer = max(
                    interferers,
                    key=lambda q: float(self._get(q, "rssi", -999.0)),
                )
                for edge in direct_edges:
                    edge_records.append(self._interference_edge_record(
                        packet=packet,
                        edge=edge,
                        resolution_id=resolution_id,
                        receiver_id=receiver_id,
                        decision_time=event_time,
                        dominant=edge is dominant_edge,
                    ))

                capture_delta_db = dominant_edge["pair_sir_db"]
                captured = all(edge["pair_pass"] for edge in direct_edges)
                self._set(packet, "captured", captured)

                if captured:
                    survivors.append(packet)
                    outcome = "CAPTURE_SURVIVOR"
                else:
                    self._set_reason(packet, "collision")
                    failed.append(packet)
                    outcome = "COLLISION_DROP"

                records.append(self._decision_record(
                    packet=packet,
                    interferers=interferers,
                    resolution_id=resolution_id,
                    receiver_id=receiver_id,
                    decision_time=event_time,
                    outcome=outcome,
                    survived=captured,
                    captured=captured,
                    strongest_interferer=strongest_interferer,
                    dominant_interferer=dominant_interferer,
                    capture_delta_db=capture_delta_db,
                    required_sir_db=dominant_edge["required_sir_db"],
                    min_pair_margin_db=dominant_edge["pair_margin_db"],
                    dominant_effective_interferer_rssi_dbm=dominant_edge[
                        "effective_interferer_rssi_dbm"
                    ],
                ))

        else:  # Legacy SINR mode retained for backwards-compatible experiments.
            groups = self._build_groups(packets)
            for group in groups:
                if len(group) == 1:
                    survivors.append(group[0])
                    continue

                group_sorted = sorted(
                    group,
                    key=lambda p: float(self._get(p, "rssi", -999.0)),
                    reverse=True,
                )
                passed = []
                for packet in group_sorted:
                    sf = int(self._get(
                        packet,
                        "sf",
                        self._get(packet, "spreading_factor", 0),
                    ))
                    sinr_db = self._sinr_dB(packet, group_sorted)
                    self._set(packet, "sinr_dB", sinr_db)
                    if sinr_db >= self.required_snr_dB(sf):
                        passed.append(packet)

                selected = []
                if passed and self.destructive:
                    selected = [max(
                        passed,
                        key=lambda p: (
                            float(self._get(p, "sinr_dB", -999.0)),
                            float(self._get(p, "rssi", -999.0)),
                        ),
                    )]
                elif passed:
                    selected = list(passed)

                for packet in group_sorted:
                    if packet in selected:
                        survivors.append(packet)
                    else:
                        self._set_reason(packet, "collision")
                        failed.append(packet)

            survivor_ids = {id(packet) for packet in survivors}
            for packet in packets:
                direct_edges = self._direct_interference_edges(packet, packets)
                interferers = [edge["interferer"] for edge in direct_edges]
                strongest = max(
                    interferers,
                    key=lambda q: float(self._get(q, "rssi", -999.0)),
                ) if interferers else None
                dominant_edge = min(
                    direct_edges,
                    key=lambda edge: edge["pair_margin_db"],
                ) if direct_edges else None
                for edge in direct_edges:
                    edge_records.append(self._interference_edge_record(
                        packet=packet,
                        edge=edge,
                        resolution_id=resolution_id,
                        receiver_id=receiver_id,
                        decision_time=event_time,
                        dominant=edge is dominant_edge,
                    ))
                survived = id(packet) in survivor_ids
                records.append(self._decision_record(
                    packet=packet,
                    interferers=interferers,
                    resolution_id=resolution_id,
                    receiver_id=receiver_id,
                    decision_time=event_time,
                    outcome=("SINR_SURVIVOR" if survived else "COLLISION_DROP"),
                    survived=survived,
                    captured=False,
                    strongest_interferer=strongest,
                    dominant_interferer=(
                        dominant_edge["interferer"]
                        if dominant_edge is not None else None
                    ),
                    required_sir_db=(
                        dominant_edge["required_sir_db"]
                        if dominant_edge is not None else None
                    ),
                    min_pair_margin_db=(
                        dominant_edge["pair_margin_db"]
                        if dominant_edge is not None else None
                    ),
                    dominant_effective_interferer_rssi_dbm=(
                        dominant_edge["effective_interferer_rssi_dbm"]
                        if dominant_edge is not None else None
                    ),
                ))

        assert len(survivors) + len(failed) == len(packets), "BUG: resolve() lost packets"
        assert len(records) == len(packets), "BUG: missing collision decision records"
        self._publish_records(records, edge_records, statistics)
        return survivors, failed
