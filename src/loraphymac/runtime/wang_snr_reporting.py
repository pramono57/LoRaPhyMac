"""Physical uplink state and causal SNR reports for Wang A2-only.

Every Wang uplink carries a transmitter-authored state header. A receiver
measures SNR locally, combines it with that decoded header, and appends one
compact report record when the packet is forwarded. No receiver reads the
simulator-only ``packet.ptx_dbm`` field as evidence of transmitter state.

The FOpts envelope is 3, 9, or 15 bytes::

    CID | version+current-PTx | last-applied-command-seq | report[0..2]

Each six-byte report stores hop+declared-PTx, SNR, command sequence/epoch,
and observation time. Thus a two-relay path fits exactly in the 15-byte
FOpts limit while the gateway measures the final-hop SNR locally.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from loraphymac.mac.frame_defs import CID, decode_fctrl, encode_fctrl, frame_size_bytes


WANG_SNR_MAX_FOPTS_BYTES = 15
WANG_TX_STATE_HEADER_SIZE_BYTES = 3
WANG_SNR_REPORT_SIZE_BYTES = 6
WANG_MAX_CARRIED_REPORTS = 2
WANG_NO_COMMAND_APPLIED_SEQ = 0
WANG_MAX_COMMAND_SEQ = 255

_WIRE_VERSION = 0
_MIN_ENCODED_PTX_DBM = 5
_MAX_ENCODED_PTX_DBM = 20
_REPORT_WIRE = struct.Struct(">BhBH")


class WangSNRReportError(ValueError):
    """Base error for an invalid physical Wang uplink envelope."""


class WangSNRReportCapacityError(WangSNRReportError):
    """Raised when a physical report cannot fit in FOpts."""


class FrozenWangRoutingMap(Mapping):
    """Validated immutable routing snapshot used by Wang receivers."""

    def __init__(self, values):
        self._values = MappingProxyType(dict(values))

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


@dataclass(frozen=True, slots=True)
class WangTransmitterState:
    """State explicitly declared by the current physical transmitter."""

    actual_ptx_dbm: int
    last_applied_command_seq: int = WANG_NO_COMMAND_APPLIED_SEQ

    @property
    def ptx_epoch(self) -> int:
        """Power epoch; it advances with an applied Wang command sequence."""
        return self.last_applied_command_seq


@dataclass(frozen=True, slots=True)
class WangSNRReport:
    """One receiver SNR observation plus transmitter-declared power state."""

    packet_uid: str
    hop_index: int
    tx_id: str
    rx_id: str
    snr_db: float
    observed_ptx_dbm: int
    observation_time_s: float
    last_applied_command_seq: int = WANG_NO_COMMAND_APPLIED_SEQ

    @property
    def ptx_epoch(self) -> int:
        return self.last_applied_command_seq

    @property
    def report_id(self) -> str:
        return (
            f"{self.packet_uid}:wang-snr:{self.hop_index}:"
            f"{self.tx_id}->{self.rx_id}"
        )


@dataclass(frozen=True, slots=True)
class WangGatewayReceivedReport:
    """One report accepted from a physically decoded frame at the gateway."""

    report: WangSNRReport
    gateway_arrival_time_s: float


@dataclass(frozen=True, slots=True)
class WangPhysicalUplinkIdentity:
    """Identity reconstructed only from frame headers and provisioned routes."""

    application_packet_id: str
    source_id: str
    frame_counter: int
    hop_index: int
    tx_id: str
    rx_id: str
    route: tuple[str, ...]


def reporting_enabled(entity) -> bool:
    """Return the explicit Wang reporting gate; absent means disabled."""
    return bool(getattr(entity, "wang_snr_reporting_enabled", False))


def freeze_wang_routing_map(routing_map: Mapping) -> FrozenWangRoutingMap:
    """Return a normalized immutable routing snapshot, rejecting ambiguity."""
    if isinstance(routing_map, FrozenWangRoutingMap):
        return routing_map
    if not isinstance(routing_map, Mapping):
        raise WangSNRReportError("immutable routing map is required")
    normalized = {}
    for raw_source, raw_parent in routing_map.items():
        source = str(raw_source)
        if not source:
            raise WangSNRReportError("routing map contains an empty source ID")
        parent = None if raw_parent is None else str(raw_parent)
        if parent == "":
            raise WangSNRReportError(
                f"routing map contains an empty parent for {source!r}"
            )
        if source in normalized and normalized[source] != parent:
            raise WangSNRReportError(
                f"routing map has ambiguous normalized source {source!r}"
            )
        normalized[source] = parent

    for source in normalized:
        current = source
        visited = set()
        while True:
            if current in visited:
                raise WangSNRReportError(
                    f"routing map cycle encountered from {source!r}"
                )
            visited.add(current)
            if current not in normalized:
                raise WangSNRReportError(
                    f"routing map target {current!r} is not provisioned"
                )
            parent = normalized[current]
            if parent is None:
                break
            current = parent
    return FrozenWangRoutingMap(normalized)


def _application_packet_identity(packet) -> tuple[str, str, int]:
    source_id = str(getattr(packet, "src_id", ""))
    if not source_id:
        raise WangSNRReportError("physical SrcID is required")
    raw_fcnt = getattr(packet, "fcnt", None)
    if isinstance(raw_fcnt, bool):
        raise WangSNRReportError("physical FCnt must be an integer")
    try:
        frame_counter = int(raw_fcnt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WangSNRReportError("physical FCnt must be an integer") from exc
    if frame_counter != raw_fcnt or not 0 <= frame_counter <= 255:
        raise WangSNRReportError("physical one-byte FCnt is outside 0..255")
    return f"UP:{source_id}:{frame_counter}", source_id, frame_counter


def _route_from_source(source_id: str, routing_map: Mapping) -> tuple[str, ...]:
    frozen = freeze_wang_routing_map(routing_map)
    if source_id not in frozen:
        raise WangSNRReportError(
            f"physical source {source_id!r} is absent from routing map"
        )
    route = [source_id]
    current = source_id
    while frozen[current] is not None:
        current = frozen[current]
        route.append(current)
    link_count = len(route) - 1
    if not 1 <= link_count <= 15:
        raise WangSNRReportError(
            "physical route must contain between 1 and 15 directed links"
        )
    return tuple(route)


def reconstruct_physical_uplink_identity(
    packet,
    *,
    routing_map: Mapping,
    receiver_id: str | None = None,
) -> WangPhysicalUplinkIdentity:
    """Resolve the active link without packet.path/packet_uid runtime metadata."""
    packet_id, source_id, frame_counter = _application_packet_identity(packet)
    route = _route_from_source(source_id, routing_map)
    raw_hop_index = getattr(packet, "hop_index", None)
    if isinstance(raw_hop_index, bool):
        raise WangSNRReportError("physical HopCtrl hop_index must be an integer")
    try:
        hop_index = int(raw_hop_index)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WangSNRReportError(
            "physical HopCtrl hop_index must be an integer"
        ) from exc
    if hop_index != raw_hop_index or not 0 <= hop_index < len(route) - 1:
        raise WangSNRReportError(
            f"physical hop_index {raw_hop_index!r} is invalid for route {route!r}"
        )

    raw_hop_limit = getattr(packet, "hop_limit", None)
    try:
        hop_limit = int(raw_hop_limit)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WangSNRReportError("physical HopCtrl hop_limit is invalid") from exc
    if isinstance(raw_hop_limit, bool) or hop_limit != raw_hop_limit:
        raise WangSNRReportError("physical HopCtrl hop_limit must be an integer")
    if hop_limit != len(route) - 1:
        raise WangSNRReportError(
            f"physical hop_limit {hop_limit} disagrees with configured route"
        )

    tx_id = route[hop_index]
    rx_id = route[hop_index + 1]
    next_hop = str(getattr(packet, "next_hop", ""))
    if next_hop != rx_id:
        raise WangSNRReportError(
            f"physical NextHop {next_hop!r} disagrees with route receiver {rx_id!r}"
        )
    dst_id = str(getattr(packet, "dst_id", ""))
    if dst_id != route[-1]:
        raise WangSNRReportError(
            f"physical DstID {dst_id!r} disagrees with route terminal {route[-1]!r}"
        )
    if receiver_id is not None and str(receiver_id) != rx_id:
        raise WangSNRReportError(
            f"local receiver {receiver_id!r} disagrees with physical link {rx_id!r}"
        )
    return WangPhysicalUplinkIdentity(
        application_packet_id=packet_id,
        source_id=source_id,
        frame_counter=frame_counter,
        hop_index=hop_index,
        tx_id=tx_id,
        rx_id=rx_id,
        route=route,
    )


def _validate_ptx(ptx_dbm: int) -> int:
    if isinstance(ptx_dbm, bool) or int(ptx_dbm) != ptx_dbm:
        raise WangSNRReportError("transmitter-declared PTx must be integer dBm")
    value = int(ptx_dbm)
    if not _MIN_ENCODED_PTX_DBM <= value <= _MAX_ENCODED_PTX_DBM:
        raise WangSNRReportError(
            "transmitter-declared PTx is outside physical wire range 5..20 dBm"
        )
    return value


def _validate_command_seq(command_seq: int) -> int:
    if isinstance(command_seq, bool) or int(command_seq) != command_seq:
        raise WangSNRReportError("last-applied command sequence must be an integer")
    value = int(command_seq)
    if not WANG_NO_COMMAND_APPLIED_SEQ <= value <= WANG_MAX_COMMAND_SEQ:
        raise WangSNRReportError("last-applied command sequence is outside 0..255")
    return value


def _encode_state_octet(ptx_dbm: int) -> int:
    ptx_code = _validate_ptx(ptx_dbm) - _MIN_ENCODED_PTX_DBM
    return (_WIRE_VERSION << 4) | ptx_code


def _decode_state_octet(value: int) -> int:
    version = (int(value) >> 4) & 0x0F
    if version != _WIRE_VERSION:
        raise WangSNRReportError(f"unsupported Wang uplink wire version {version}")
    return _MIN_ENCODED_PTX_DBM + (int(value) & 0x0F)


def encode_transmitter_state(state: WangTransmitterState) -> bytes:
    """Encode the physical three-byte current-transmitter declaration."""
    return bytes(
        (
            int(CID.WANG_SNR_REPORT),
            _encode_state_octet(state.actual_ptx_dbm),
            _validate_command_seq(state.last_applied_command_seq),
        )
    )


def _route_link(route: tuple[str, ...], hop_index: int) -> tuple[str, str]:
    if hop_index < 0 or hop_index + 1 >= len(route):
        raise WangSNRReportError(
            f"hop_index {hop_index} has no directed link in route {route!r}"
        )
    return route[hop_index], route[hop_index + 1]


def encode_report(report: WangSNRReport) -> bytes:
    """Encode one six-byte carried report at conservative resolutions."""
    hop_index = int(report.hop_index)
    if not 0 <= hop_index <= 15:
        raise WangSNRReportError("report hop_index is outside 0..15")
    ptx_code = _validate_ptx(report.observed_ptx_dbm) - _MIN_ENCODED_PTX_DBM
    hop_ptx = (hop_index << 4) | ptx_code
    snr_centi_db = math.floor(float(report.snr_db) * 100.0)
    time_deciseconds = math.floor(float(report.observation_time_s) * 10.0)
    if not -32768 <= snr_centi_db <= 32767:
        raise WangSNRReportError("SNR is outside signed centi-dB wire range")
    if not 0 <= time_deciseconds <= 65535:
        raise WangSNRReportError("observation time is outside 0..6553.5 s wire range")
    return _REPORT_WIRE.pack(
        hop_ptx,
        snr_centi_db,
        _validate_command_seq(report.last_applied_command_seq),
        time_deciseconds,
    )


def _decode_wire_envelope(packet):
    """Decode physical bytes without assigning route-dependent identities."""
    raw = bytes(getattr(packet, "fopts", b"") or b"")
    if len(raw) < WANG_TX_STATE_HEADER_SIZE_BYTES:
        raise WangSNRReportError("Wang uplink is missing transmitter state bytes")
    if len(raw) > WANG_SNR_MAX_FOPTS_BYTES:
        raise WangSNRReportError("Wang uplink exceeds the 15-byte FOpts limit")
    report_bytes = len(raw) - WANG_TX_STATE_HEADER_SIZE_BYTES
    if report_bytes % WANG_SNR_REPORT_SIZE_BYTES:
        raise WangSNRReportError("Wang uplink length is not 3 + N*6 bytes")
    report_count = report_bytes // WANG_SNR_REPORT_SIZE_BYTES
    if report_count > WANG_MAX_CARRIED_REPORTS:
        raise WangSNRReportCapacityError("Wang uplink carries more than two reports")
    if raw[0] != int(CID.WANG_SNR_REPORT):
        raise WangSNRReportError(f"unexpected Wang FOpts CID 0x{raw[0]:02x}")

    state = WangTransmitterState(
        actual_ptx_dbm=_decode_state_octet(raw[1]),
        last_applied_command_seq=_validate_command_seq(raw[2]),
    )
    wire_reports = []
    offset = WANG_TX_STATE_HEADER_SIZE_BYTES
    while offset < len(raw):
        hop_ptx, snr_centi_db, command_seq, time_deciseconds = (
            _REPORT_WIRE.unpack(raw[offset : offset + WANG_SNR_REPORT_SIZE_BYTES])
        )
        hop_index = (hop_ptx >> 4) & 0x0F
        ptx_dbm = _MIN_ENCODED_PTX_DBM + (hop_ptx & 0x0F)
        wire_reports.append(
            (
                hop_index,
                float(snr_centi_db) / 100.0,
                ptx_dbm,
                float(time_deciseconds) / 10.0,
                int(command_seq),
            )
        )
        offset += WANG_SNR_REPORT_SIZE_BYTES
    return state, tuple(wire_reports)


def decode_uplink_envelope(
    packet,
    *,
    routing_map: Mapping | None = None,
    receiver_id: str | None = None,
) -> tuple[WangTransmitterState, list[WangSNRReport]]:
    """Decode frame bytes and reconstruct identities from physical headers."""
    state, wire_reports = _decode_wire_envelope(packet)
    if routing_map is None:
        if wire_reports:
            raise WangSNRReportError(
                "immutable routing map is required to decode carried reports"
            )
        return state, []

    carrier = reconstruct_physical_uplink_identity(
        packet,
        routing_map=routing_map,
        receiver_id=receiver_id,
    )
    reports = []
    for (
        hop_index,
        snr_db,
        ptx_dbm,
        observation_time_s,
        command_seq,
    ) in wire_reports:
        if hop_index >= carrier.hop_index:
            raise WangSNRReportError(
                f"carried report hop {hop_index} is not before carrier hop "
                f"{carrier.hop_index}"
            )
        tx_id, rx_id = _route_link(carrier.route, hop_index)
        reports.append(
            WangSNRReport(
                packet_uid=carrier.application_packet_id,
                hop_index=hop_index,
                tx_id=tx_id,
                rx_id=rx_id,
                snr_db=snr_db,
                observed_ptx_dbm=ptx_dbm,
                observation_time_s=observation_time_s,
                last_applied_command_seq=command_seq,
            )
        )
    return state, reports


def decode_reports(
    packet,
    *,
    routing_map: Mapping | None = None,
    receiver_id: str | None = None,
) -> list[WangSNRReport]:
    """Return reports carried in the physical Wang uplink envelope."""
    return decode_uplink_envelope(
        packet,
        routing_map=routing_map,
        receiver_id=receiver_id,
    )[1]


def report_from_successful_rx(
    packet,
    receiver_id: str,
    observation_time_s: float,
    *,
    routing_map: Mapping,
) -> WangSNRReport:
    """Combine local SNR with state decoded from the received frame bytes."""
    if not bool(getattr(packet, "decoded", False)):
        raise WangSNRReportError("SNR report requires decoded=True")
    if bool(getattr(packet, "collided", False)):
        raise WangSNRReportError("collided reception cannot generate an SNR report")
    if getattr(packet, "feasible", True) is False:
        raise WangSNRReportError("infeasible reception cannot generate an SNR report")
    if not bool(getattr(packet, "is_intended_receiver", True)):
        raise WangSNRReportError("non-target reception cannot generate an SNR report")

    snr_db = getattr(packet, "snr", None)
    if snr_db is None or not math.isfinite(float(snr_db)):
        raise WangSNRReportError("finite receiver SNR is required")
    if not math.isfinite(float(observation_time_s)):
        raise WangSNRReportError("finite observation time is required")

    identity = reconstruct_physical_uplink_identity(
        packet,
        routing_map=routing_map,
        receiver_id=receiver_id,
    )
    transmitter_state, _ = _decode_wire_envelope(packet)
    return WangSNRReport(
        packet_uid=identity.application_packet_id,
        hop_index=identity.hop_index,
        tx_id=identity.tx_id,
        rx_id=identity.rx_id,
        snr_db=float(snr_db),
        observed_ptx_dbm=transmitter_state.actual_ptx_dbm,
        observation_time_s=float(observation_time_s),
        last_applied_command_seq=transmitter_state.last_applied_command_seq,
    )


def _replace_fopts(packet, raw: bytes) -> None:
    old_fctrl = decode_fctrl(getattr(packet, "fctrl", 0))
    packet.fopts = bytes(raw)
    packet.fctrl = encode_fctrl(
        apc=bool(old_fctrl.get("apc", False)),
        ack_req=bool(old_fctrl.get("ack_req", False)),
        bcast=bool(old_fctrl.get("bcast", False)),
        pending=bool(old_fctrl.get("pending", False)),
        fopts_len=len(raw),
    )
    packet.payload_size = frame_size_bytes(
        app_payload_size=int(getattr(packet, "app_payload_size", 0)),
        fopts_len=len(raw),
    )


def declare_transmitter_state(
    packet,
    *,
    actual_ptx_dbm: int,
    last_applied_command_seq: int = WANG_NO_COMMAND_APPLIED_SEQ,
) -> None:
    """Write current transmitter state while preserving carried reports."""
    raw = bytes(getattr(packet, "fopts", b"") or b"")
    carried = b""
    if raw:
        _decode_wire_envelope(packet)
        carried = raw[WANG_TX_STATE_HEADER_SIZE_BYTES:]
    state = WangTransmitterState(actual_ptx_dbm, last_applied_command_seq)
    updated = encode_transmitter_state(state) + carried
    if len(updated) > WANG_SNR_MAX_FOPTS_BYTES:
        raise WangSNRReportCapacityError("Wang transmitter state exceeds FOpts")
    _replace_fopts(packet, updated)


def append_report(
    packet,
    report: WangSNRReport,
    *,
    routing_map: Mapping,
) -> None:
    """Append a causal report to an existing transmitter-authored envelope."""
    raw = bytes(getattr(packet, "fopts", b"") or b"")
    _decode_wire_envelope(packet)
    identity = reconstruct_physical_uplink_identity(
        packet,
        routing_map=routing_map,
        receiver_id=report.rx_id,
    )
    if (
        report.packet_uid != identity.application_packet_id
        or report.hop_index != identity.hop_index
        or report.tx_id != identity.tx_id
        or report.rx_id != identity.rx_id
    ):
        raise WangSNRReportError(
            "report identity disagrees with physical frame headers and routing"
        )
    encoded = encode_report(report)
    new_len = len(raw) + len(encoded)
    if new_len > WANG_SNR_MAX_FOPTS_BYTES:
        raise WangSNRReportCapacityError(
            f"Wang report needs {new_len} FOpts bytes; maximum is "
            f"{WANG_SNR_MAX_FOPTS_BYTES}"
        )
    _replace_fopts(packet, raw + encoded)


def accept_at_gateway(
    gateway,
    report: WangSNRReport,
    gateway_arrival_time_s: float,
) -> WangGatewayReceivedReport | None:
    """Store one physically received report, suppressing duplicate identity."""
    arrival_time = float(gateway_arrival_time_s)
    if not math.isfinite(arrival_time):
        raise WangSNRReportError("finite gateway arrival time is required")
    if not hasattr(gateway, "wang_snr_report_ids"):
        gateway.wang_snr_report_ids = set()
    if not hasattr(gateway, "wang_received_snr_reports"):
        gateway.wang_received_snr_reports = []
    if not hasattr(gateway, "wang_received_snr_observations"):
        gateway.wang_received_snr_observations = []
    if report.report_id in gateway.wang_snr_report_ids:
        return None
    received = WangGatewayReceivedReport(
        report=report,
        gateway_arrival_time_s=arrival_time,
    )
    gateway.wang_snr_report_ids.add(report.report_id)
    gateway.wang_received_snr_reports.append(report)
    gateway.wang_received_snr_observations.append(received)
    return received


__all__ = [
    "WANG_MAX_CARRIED_REPORTS",
    "WANG_MAX_COMMAND_SEQ",
    "WANG_NO_COMMAND_APPLIED_SEQ",
    "WANG_SNR_MAX_FOPTS_BYTES",
    "WANG_SNR_REPORT_SIZE_BYTES",
    "WANG_TX_STATE_HEADER_SIZE_BYTES",
    "FrozenWangRoutingMap",
    "WangGatewayReceivedReport",
    "WangPhysicalUplinkIdentity",
    "WangSNRReport",
    "WangSNRReportCapacityError",
    "WangSNRReportError",
    "WangTransmitterState",
    "accept_at_gateway",
    "append_report",
    "declare_transmitter_state",
    "decode_reports",
    "decode_uplink_envelope",
    "encode_report",
    "encode_transmitter_state",
    "freeze_wang_routing_map",
    "reconstruct_physical_uplink_identity",
    "report_from_successful_rx",
    "reporting_enabled",
]
