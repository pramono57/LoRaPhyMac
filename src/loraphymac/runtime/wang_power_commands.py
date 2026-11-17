"""Physical Wang A2-only power commands and causal application.

The command is carried entirely in FOpts.  ``target_id`` is the physical
LoRaPhyMAC ``DstID`` field; every other receiver-required value is encoded in
the eleven-byte command below.  Runtime packet attributes are never used as
command evidence.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from loraphymac.mac.frame_defs import (
    CID,
    FPort,
    MType,
    encode_fctrl,
    frame_size_bytes,
)
from loraphymac.runtime.packet import Packet
from loraphymac.runtime.wang_atp_a2only import WangA2PowerReductionController


WANG_POWER_COMMAND_VERSION = 1
WANG_POWER_COMMAND_FOPTS_BYTES = 11
WANG_COMMAND_EXPIRY_S = 300.0
_COMMAND_WIRE = struct.Struct(">7B2H")


class WangPowerCommandError(ValueError):
    """Raised when a Wang command is malformed or unsafe to apply."""


@dataclass(frozen=True, slots=True)
class WangPowerCommand:
    target_id: str
    command_seq: int
    window_index: int
    expected_ptx_dbm: int
    expected_epoch: int
    target_ptx_dbm: int
    issued_at_s: float
    expiry_s: float

    @property
    def command_id(self) -> str:
        return (
            f"WANG-CMD:{self.target_id}:W{self.window_index:03d}:"
            f"S{self.command_seq:03d}"
        )


@dataclass(frozen=True, slots=True)
class WangCommandApplication:
    status: str
    reason: str
    command: WangPowerCommand | None
    old_ptx_dbm: int
    new_ptx_dbm: int
    old_epoch: int
    new_epoch: int

    @property
    def applied(self) -> bool:
        return self.status == "applied"


@dataclass(frozen=True, slots=True)
class WangNodeRX1Context:
    """Immutable physical context for one node uplink's RX1 window."""

    source_node_id: str
    uplink_packet_uid: str
    uplink_tx_end_s: float
    uplink_channel: int
    uplink_freq_mhz: float
    rx1_channel: int
    rx1_freq_mhz: float
    rx1_sf: int
    rx1_bw_hz: int
    window_id: str


@dataclass(slots=True)
class WangGatewayTargetState:
    """Gateway belief; pending never implies physical application."""

    target_id: str
    entity_type: str
    initial_ptx_dbm: int
    last_confirmed_ptx_dbm: int
    last_confirmed_epoch: int = 0
    next_command_seq: int = 1
    pending_command: WangPowerCommand | None = None
    pending_command_seq: int | None = None
    pending_target_ptx_dbm: int | None = None
    pending_command_expiry_s: float | None = None
    pending_status: str = "none"
    status: str = "confirmed_initial_state"

    def begin_pending(self, command: WangPowerCommand) -> None:
        self.pending_command = command
        self.pending_command_seq = int(command.command_seq)
        self.pending_target_ptx_dbm = int(command.target_ptx_dbm)
        self.pending_command_expiry_s = float(command.expiry_s)
        self.pending_status = "pending_unconfirmed"
        self.status = "pending_unconfirmed"

    def clear_pending(self, *, status: str) -> None:
        self.pending_command = None
        self.pending_command_seq = None
        self.pending_target_ptx_dbm = None
        self.pending_command_expiry_s = None
        self.pending_status = "none"
        self.status = str(status)


def _uint8(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise WangPowerCommandError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= 255:
        raise WangPowerCommandError(f"{name} is outside {minimum}..255")
    return result


def _time_deciseconds(value: float, name: str) -> int:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise WangPowerCommandError(f"{name} must be finite and >= 0")
    encoded = math.floor(numeric * 10.0)
    if encoded > 65535:
        raise WangPowerCommandError(f"{name} exceeds 6553.5 s wire range")
    return encoded


def encode_wang_power_command(command: WangPowerCommand) -> bytes:
    """Encode an immutable eleven-byte Wang command."""
    if not str(command.target_id):
        raise WangPowerCommandError("target_id must be non-empty")
    command_seq = _uint8(command.command_seq, "command_seq", minimum=1)
    window_index = _uint8(command.window_index, "window_index")
    expected_ptx = _uint8(command.expected_ptx_dbm, "expected_ptx_dbm")
    expected_epoch = _uint8(command.expected_epoch, "expected_epoch")
    target_ptx = _uint8(command.target_ptx_dbm, "target_ptx_dbm")
    issued = _time_deciseconds(command.issued_at_s, "issued_at_s")
    expiry = _time_deciseconds(command.expiry_s, "expiry_s")
    if expiry <= issued:
        raise WangPowerCommandError("expiry must be after issued_at")
    raw = _COMMAND_WIRE.pack(
        int(CID.WANG_POWER_COMMAND),
        WANG_POWER_COMMAND_VERSION,
        command_seq,
        window_index,
        expected_ptx,
        expected_epoch,
        target_ptx,
        issued,
        expiry,
    )
    if len(raw) != WANG_POWER_COMMAND_FOPTS_BYTES:
        raise AssertionError("unexpected Wang command wire size")
    return raw


def decode_wang_power_command(packet) -> WangPowerCommand:
    """Decode command evidence exclusively from FOpts and physical DstID."""
    raw = bytes(getattr(packet, "fopts", b"") or b"")
    if len(raw) != WANG_POWER_COMMAND_FOPTS_BYTES:
        raise WangPowerCommandError(
            f"Wang command must be {WANG_POWER_COMMAND_FOPTS_BYTES} FOpts bytes"
        )
    (
        cid,
        version,
        command_seq,
        window_index,
        expected_ptx,
        expected_epoch,
        target_ptx,
        issued_ds,
        expiry_ds,
    ) = _COMMAND_WIRE.unpack(raw)
    if cid != int(CID.WANG_POWER_COMMAND):
        raise WangPowerCommandError(f"unexpected Wang command CID 0x{cid:02x}")
    if version != WANG_POWER_COMMAND_VERSION:
        raise WangPowerCommandError(
            f"unsupported Wang command version {version}"
        )
    target_id = getattr(packet, "dst_id", None)
    if target_id in (None, ""):
        raise WangPowerCommandError("Wang command requires physical DstID")
    command = WangPowerCommand(
        target_id=str(target_id),
        command_seq=int(command_seq),
        window_index=int(window_index),
        expected_ptx_dbm=int(expected_ptx),
        expected_epoch=int(expected_epoch),
        target_ptx_dbm=int(target_ptx),
        issued_at_s=float(issued_ds) / 10.0,
        expiry_s=float(expiry_ds) / 10.0,
    )
    # Reuse the encoder's range and temporal validation.
    encode_wang_power_command(command)
    return command


def is_wang_power_command(packet) -> bool:
    raw = bytes(getattr(packet, "fopts", b"") or b"")
    return bool(raw) and raw[0] == int(CID.WANG_POWER_COMMAND)


def apply_wang_power_command(entity, packet, *, now_s: float) -> WangCommandApplication:
    """Validate and apply one absolute target exactly once at physical RX."""
    old_ptx = int(getattr(entity, "default_ptx"))
    old_epoch = int(getattr(entity, "wang_last_applied_command_seq", 0))
    try:
        command = decode_wang_power_command(packet)
    except WangPowerCommandError as exc:
        return WangCommandApplication(
            "rejected", str(exc), None, old_ptx, old_ptx, old_epoch, old_epoch
        )

    entity_id = str(getattr(entity, "id", ""))
    if command.target_id != entity_id:
        reason = "wrong_target"
    elif float(now_s) < command.issued_at_s:
        reason = "future_issued_command"
    elif float(now_s) > command.expiry_s:
        reason = "command_expired"
    elif command.command_seq == old_epoch:
        reason = "duplicate_command"
    elif command.command_seq < old_epoch:
        reason = "stale_command_sequence"
    elif command.expected_epoch != old_epoch:
        reason = "expected_epoch_mismatch"
    elif command.expected_ptx_dbm != old_ptx:
        reason = "expected_ptx_mismatch"
    elif command.target_ptx_dbm > old_ptx:
        reason = "ptx_increase_forbidden"
    else:
        entity_type = "relay" if entity_id.startswith("R") else "node"
        bounds = WangA2PowerReductionController.bounds_for(entity_type)
        supported = getattr(getattr(entity, "hw", None), "tx_current", {})
        if not bounds.minimum_dbm <= command.target_ptx_dbm <= bounds.maximum_dbm:
            reason = "target_ptx_outside_wang_bounds"
        elif supported and command.target_ptx_dbm not in supported:
            reason = "target_ptx_not_supported_by_hardware"
        else:
            entity.default_ptx = int(command.target_ptx_dbm)
            # Wang relays use the existing fixed-forwarding selection path.
            if entity_type == "relay":
                entity.fixed_ptx = int(command.target_ptx_dbm)
            entity.wang_last_applied_command_seq = int(command.command_seq)
            return WangCommandApplication(
                "applied",
                "absolute_target_applied",
                command,
                old_ptx,
                int(command.target_ptx_dbm),
                old_epoch,
                int(command.command_seq),
            )

    status = "duplicate" if reason == "duplicate_command" else "rejected"
    return WangCommandApplication(
        status, reason, command, old_ptx, old_ptx, old_epoch, old_epoch
    )


class WangGatewayCommandCoordinator:
    """Route physical commands and reconcile them only from causal uplinks."""

    def __init__(
        self,
        *,
        env,
        gateway,
        active_transmitters,
        routing_map,
        relay_tx,
        air_interface,
        statistics,
        expiry_s: float = WANG_COMMAND_EXPIRY_S,
    ) -> None:
        self.env = env
        self.gateway = gateway
        self.routing_map = dict(routing_map)
        self.relay_tx = dict(relay_tx)
        self.air = air_interface
        self.stats = statistics
        self.expiry_s = float(expiry_s)
        if str(gateway.id) != "GW":
            raise ValueError("Wang command coordinator requires main gateway GW")
        if not math.isfinite(self.expiry_s) or self.expiry_s <= 0:
            raise ValueError("Wang command expiry must be finite and > 0")
        self.states = {
            item.entity_id: WangGatewayTargetState(
                target_id=item.entity_id,
                entity_type=item.entity_type,
                initial_ptx_dbm=int(item.expected_ptx_dbm),
                last_confirmed_ptx_dbm=int(item.expected_ptx_dbm),
                last_confirmed_epoch=int(item.ptx_epoch),
            )
            for item in active_transmitters
        }
        self._held_node_packets = {}

    def _record(self, event, *, packet=None, command=None, receiver_id=None, reason=None):
        self.stats.record_wang_power_command_event(
            event=event,
            time=float(self.env.now),
            packet=packet,
            command=command,
            receiver_id=receiver_id,
            reason=reason,
        )

    def _reverse_route(self, target_id: str) -> list[str]:
        uplink = [str(target_id)]
        seen = {str(target_id)}
        current = str(target_id)
        while current != str(self.gateway.id):
            parent = self.routing_map.get(current)
            if parent in (None, ""):
                raise WangPowerCommandError(
                    f"target {target_id} has no configured route to GW"
                )
            parent = str(parent)
            if parent in seen:
                raise WangPowerCommandError(
                    f"routing cycle while building command path for {target_id}"
                )
            uplink.append(parent)
            seen.add(parent)
            current = parent
        return list(reversed(uplink))

    def _packet_for(self, command: WangPowerCommand) -> Packet:
        path = self._reverse_route(command.target_id)
        if len(path) < 2:
            raise WangPowerCommandError("command target cannot be the gateway")
        next_hop = path[1]
        mtype = MType.DATA_DOWN_FWD if next_hop.startswith("R") else MType.DATA_DOWN
        fopts = encode_wang_power_command(command)
        payload_size = frame_size_bytes(app_payload_size=0, fopts_len=len(fopts))
        gateway_tx = self.relay_tx[str(self.gateway.id)]
        ch_id = int(gateway_tx.relay_tx_ch_ids[0])
        freq_mhz = float(gateway_tx.channel_plan.get_freq(ch_id))
        sf = int(gateway_tx.relay_sf)
        bw_hz = int(gateway_tx.relay_bw_hz)
        toa = float(
            self.air.compute_toa(sf=sf, bw_hz=bw_hz, payload_bytes=payload_size)
        )
        fcnt = int(getattr(self.gateway, "fcnt", 0))
        self.gateway.fcnt = fcnt + 1
        return Packet(
            src_id=str(self.gateway.id),
            current_tx=str(self.gateway.id),
            prev_hop=str(self.gateway.id),
            next_hop=next_hop,
            path=path,
            hop_index=0,
            ch_id=ch_id,
            freq_mhz=freq_mhz,
            sf=sf,
            bw_hz=bw_hz,
            ptx_dbm=int(self.gateway.default_ptx),
            payload_size=payload_size,
            toa=toa,
            tx_start=float(self.env.now),
            tx_end=float(self.env.now) + toa,
            packet_uid=command.command_id,
            app_payload_size=0,
            mtype=mtype,
            version=0,
            rfu=0,
            dst_id=command.target_id,
            hop_limit=len(path) - 1,
            fctrl=encode_fctrl(
                apc=False,
                ack_req=False,
                bcast=False,
                pending=False,
                fopts_len=len(fopts),
            ),
            fcnt=fcnt,
            fopts=fopts,
            fport=FPort.MAC_CONTROL,
            tx_mode="WANG_A2_DOWN",
            birth_time=float(self.env.now),
        )

    def handle_window_decision(self, record) -> None:
        """Turn a complete sink decision into zero or more physical frames."""
        if getattr(record, "status", None) != "decision_generated":
            return
        for entity_decision in record.entity_decisions:
            target_id = str(entity_decision.entity_id)
            if int(entity_decision.target_ptx_dbm) >= int(
                entity_decision.current_ptx_dbm
            ):
                continue
            state = self.states[target_id]
            if state.pending_command is not None:
                self._record(
                    "command_unconfirmed",
                    command=state.pending_command,
                    reason="outstanding_command_blocks_new_command",
                )
                continue
            if state.next_command_seq > 255:
                self._record(
                    "command_rejected",
                    reason=f"command_sequence_exhausted:{target_id}",
                )
                state.status = "sequence_exhausted"
                continue
            command = WangPowerCommand(
                target_id=target_id,
                command_seq=state.next_command_seq,
                window_index=int(record.window_index),
                expected_ptx_dbm=state.last_confirmed_ptx_dbm,
                expected_epoch=state.last_confirmed_epoch,
                target_ptx_dbm=int(entity_decision.target_ptx_dbm),
                issued_at_s=float(self.env.now),
                expiry_s=float(self.env.now) + self.expiry_s,
            )
            packet = self._packet_for(command)
            state.begin_pending(command)
            state.next_command_seq += 1
            self._record("command_generated", packet=packet, command=command)
            if str(packet.next_hop).startswith("R"):
                accepted = self.relay_tx[str(self.gateway.id)].put(packet)
                if accepted:
                    self._record("command_queued", packet=packet, command=command)
            else:
                self._hold_for_node_rx1(str(self.gateway.id), packet, command)
            self.env.process(self._mark_unconfirmed_at_expiry(command))

    def _hold_for_node_rx1(self, holder_id, packet, command) -> None:
        key = (str(holder_id), str(command.target_id))
        if key in self._held_node_packets:
            self._record(
                "command_duplicate",
                packet=packet,
                command=command,
                receiver_id=str(holder_id),
                reason="duplicate_pending_node_command",
            )
            return
        self._held_node_packets[key] = packet
        self._record(
            "command_queued",
            packet=packet,
            command=command,
            receiver_id=str(holder_id),
            reason="awaiting_target_rx1",
        )

    def _discard_held_command(self, command: WangPowerCommand) -> None:
        """Remove an expired node-bound frame once non-application is proven."""
        for key, packet in tuple(self._held_node_packets.items()):
            try:
                held = decode_wang_power_command(packet)
            except WangPowerCommandError:
                continue
            if held == command:
                self._held_node_packets.pop(key, None)

    def receive_at_relay(self, relay, packet) -> bool:
        """Apply at a relay target or physically forward along reverse route."""
        try:
            command = decode_wang_power_command(packet)
        except WangPowerCommandError as exc:
            self._record(
                "command_rejected",
                packet=packet,
                receiver_id=str(relay.id),
                reason=str(exc),
            )
            return True
        self._record(
            "command_received",
            packet=packet,
            command=command,
            receiver_id=str(relay.id),
        )
        if float(self.env.now) > command.expiry_s:
            self._record(
                "command_expired",
                packet=packet,
                command=command,
                receiver_id=str(relay.id),
                reason="expired_at_relay_receive",
            )
            return True
        if command.target_id == str(relay.id):
            self._apply_at_target(relay, packet, command)
            return True

        path = list(getattr(packet, "path", []) or [])
        current_index = int(getattr(packet, "hop_index", -1)) + 1
        if current_index >= len(path) or str(path[current_index]) != str(relay.id):
            self._record(
                "command_rejected",
                packet=packet,
                command=command,
                receiver_id=str(relay.id),
                reason="reverse_route_position_mismatch",
            )
            return True
        packet.advance_hop()
        packet.reset_rx_runtime_metadata(preserve_t_arrive=False)
        if str(packet.next_hop).startswith("R"):
            packet.mtype = MType.DATA_DOWN_FWD
            accepted = self.relay_tx[str(relay.id)].put(packet)
            if accepted:
                self._record(
                    "command_forwarded",
                    packet=packet,
                    command=command,
                    receiver_id=str(relay.id),
                )
        else:
            packet.mtype = MType.DATA_DOWN
            self._hold_for_node_rx1(str(relay.id), packet, command)
        return True

    def observe_successful_uplink(self, receiver, uplink_packet) -> None:
        """Release a held final hop only for the target's real RX1 window."""
        if getattr(uplink_packet, "mtype", None) not in (
            MType.DATA_UP,
            MType.DATA_UP_FWD,
        ):
            return
        target_id = str(getattr(uplink_packet, "prev_hop", ""))
        if target_id.startswith("R") or not target_id:
            return
        key = (str(receiver.id), target_id)
        packet = self._held_node_packets.get(key)
        if packet is None:
            return
        command = decode_wang_power_command(packet)
        node = receiver.network_nodes_map.get(target_id)
        if node is None:
            self._record(
                "command_rejected",
                packet=packet,
                command=command,
                receiver_id=str(receiver.id),
                reason="target_missing_from_network_map",
            )
            return
        try:
            rx1_context = self._snapshot_node_rx1_context(
                node=node,
                uplink_packet=uplink_packet,
            )
        except WangPowerCommandError as exc:
            self._record(
                "command_rejected",
                packet=packet,
                command=command,
                receiver_id=str(receiver.id),
                reason=f"invalid_node_rx1_context:{exc}",
            )
            return
        # Consume the held command only after a complete physical context has
        # been captured.  Later forwarding may freely mutate uplink_packet.
        self._held_node_packets.pop(key, None)
        release_at = rx1_context.uplink_tx_end_s + float(
            getattr(node, "rx1_delay_s", 1.0)
        )
        self.env.process(
            self._release_for_rx1(
                holder_id=str(receiver.id),
                packet=packet,
                command=command,
                rx1_context=rx1_context,
                release_at_s=release_at,
            )
        )

    @staticmethod
    def _snapshot_node_rx1_context(*, node, uplink_packet) -> WangNodeRX1Context:
        source_node_id = str(getattr(uplink_packet, "prev_hop", ""))
        packet_uid_value = getattr(uplink_packet, "packet_uid", None)
        packet_uid = "" if packet_uid_value is None else str(packet_uid_value)
        if not source_node_id or source_node_id.startswith("R"):
            raise WangPowerCommandError("invalid source node identity")
        if source_node_id != str(getattr(node, "id", "")):
            raise WangPowerCommandError("source node does not match RX1 target")
        if not packet_uid:
            raise WangPowerCommandError("uplink packet UID is required")
        try:
            tx_end = float(uplink_packet.tx_end)
            channel = int(uplink_packet.ch_id)
            freq_mhz = float(uplink_packet.freq_mhz)
            sf = int(uplink_packet.sf)
            bw_hz = int(uplink_packet.bw_hz)
        except (AttributeError, TypeError, ValueError) as exc:
            raise WangPowerCommandError("incomplete uplink PHY context") from exc
        if not math.isfinite(tx_end) or tx_end < 0:
            raise WangPowerCommandError("invalid uplink TX end time")
        if not math.isfinite(freq_mhz) or freq_mhz <= 0:
            raise WangPowerCommandError("invalid uplink frequency")
        if channel < 0 or sf <= 0 or bw_hz <= 0:
            raise WangPowerCommandError("invalid uplink PHY values")
        window_id = (
            f"RX1:{source_node_id}:{packet_uid}:"
            f"{tx_end:.9f}:{channel}:{sf}:{bw_hz}"
        )
        # NodeTX._run_rx1_window derives RX1 tuning from this exact uplink
        # physical context.  No control/backbone channel is substituted.
        return WangNodeRX1Context(
            source_node_id=source_node_id,
            uplink_packet_uid=packet_uid,
            uplink_tx_end_s=tx_end,
            uplink_channel=channel,
            uplink_freq_mhz=freq_mhz,
            rx1_channel=channel,
            rx1_freq_mhz=freq_mhz,
            rx1_sf=sf,
            rx1_bw_hz=bw_hz,
            window_id=window_id,
        )

    def _release_for_rx1(
        self, *, holder_id, packet, command, rx1_context, release_at_s
    ):
        delay = max(0.0, float(release_at_s) - float(self.env.now))
        if delay:
            yield self.env.timeout(delay)
        if float(self.env.now) > command.expiry_s:
            self._record(
                "command_expired",
                packet=packet,
                command=command,
                receiver_id=holder_id,
                reason="expired_before_target_rx1",
            )
            return
        packet.ch_id = int(rx1_context.rx1_channel)
        packet.freq_mhz = float(rx1_context.rx1_freq_mhz)
        packet.sf = int(rx1_context.rx1_sf)
        packet.bw_hz = int(rx1_context.rx1_bw_hz)
        packet.reset_rx_runtime_metadata(preserve_t_arrive=False)
        accepted = self.relay_tx[holder_id].put(packet)
        if accepted:
            self._record(
                "command_forwarded",
                packet=packet,
                command=command,
                receiver_id=holder_id,
                reason="released_for_physical_rx1",
            )

    def apply_at_node(self, node, packet) -> WangCommandApplication:
        try:
            command = decode_wang_power_command(packet)
        except WangPowerCommandError as exc:
            result = apply_wang_power_command(
                node, packet, now_s=float(self.env.now)
            )
            self._record(
                "command_rejected",
                packet=packet,
                receiver_id=str(node.id),
                reason=str(exc),
            )
            return result
        self._record(
            "command_received",
            packet=packet,
            command=command,
            receiver_id=str(node.id),
        )
        return self._apply_at_target(node, packet, command)

    def _apply_at_target(self, entity, packet, command) -> WangCommandApplication:
        result = apply_wang_power_command(entity, packet, now_s=float(self.env.now))
        if result.applied:
            self._record(
                "command_applied",
                packet=packet,
                command=command,
                receiver_id=str(entity.id),
            )
            self.stats.record_ptx_update(
                entity_id=str(entity.id),
                entity_type=("relay" if str(entity.id).startswith("R") else "node"),
                old_ptx=result.old_ptx_dbm,
                new_ptx=result.new_ptx_dbm,
                time=float(self.env.now),
                feedback_from=getattr(packet, "prev_hop", None),
                apply_source="physical_wang_data_down",
                control_mtype=getattr(packet, "mtype", None),
            )
        else:
            event = (
                "command_duplicate"
                if result.status == "duplicate"
                else "command_expired"
                if result.reason == "command_expired"
                else "command_rejected"
            )
            self._record(
                event,
                packet=packet,
                command=command,
                receiver_id=str(entity.id),
                reason=result.reason,
            )
        return result

    def reconcile_report(self, received) -> WangPowerCommand | None:
        """Reconcile only from causal transmitter-declared uplink state."""
        report = received.report
        state = self.states.get(str(report.tx_id))
        if state is None or state.pending_command is None:
            return None
        command = state.pending_command
        try:
            observation_time = float(report.observation_time_s)
            arrival_time = float(received.gateway_arrival_time_s)
        except (TypeError, ValueError):
            return None
        if (
            not math.isfinite(observation_time)
            or not math.isfinite(arrival_time)
            or arrival_time < observation_time
            or observation_time < command.issued_at_s
        ):
            return None
        report_seq = int(report.last_applied_command_seq)
        report_ptx = int(report.observed_ptx_dbm)
        if (
            report_seq == command.command_seq
            and report_ptx == command.target_ptx_dbm
        ):
            state.last_confirmed_ptx_dbm = command.target_ptx_dbm
            state.last_confirmed_epoch = command.command_seq
            state.clear_pending(status="confirmed_by_uplink")
            self._record(
                "command_confirmed_by_uplink",
                command=command,
                receiver_id=str(self.gateway.id),
                reason="post_command_declared_state_matches_target",
            )
            return command

        # Receivers reject a command strictly after expiry.  Therefore, a
        # transmitter observation made strictly after expiry that still
        # declares the last confirmed state proves that this command was not
        # applied.  Arrival time is deliberately irrelevant; the physical
        # observation timestamp controls causality.
        if (
            observation_time > command.expiry_s
            and report_seq == state.last_confirmed_epoch
            and report_ptx == state.last_confirmed_ptx_dbm
        ):
            self._discard_held_command(command)
            state.clear_pending(status="not_applied_confirmed_by_uplink")
            self._record(
                "command_not_applied",
                command=command,
                receiver_id=str(self.gateway.id),
                reason="post_expiry_declared_state_unchanged",
            )
        return None

    def _mark_unconfirmed_at_expiry(self, command):
        delay = max(0.0, command.expiry_s - float(self.env.now))
        if delay:
            yield self.env.timeout(delay)
        state = self.states.get(command.target_id)
        if state is not None and state.pending_command == command:
            state.status = "expired_unconfirmed_application_unknown"
            state.pending_status = "expired_unconfirmed_application_unknown"
            self._record(
                "command_unconfirmed",
                command=command,
                reason="no_causal_uplink_confirmation_by_expiry",
            )


__all__ = [
    "WANG_COMMAND_EXPIRY_S",
    "WANG_POWER_COMMAND_FOPTS_BYTES",
    "WANG_POWER_COMMAND_VERSION",
    "WangCommandApplication",
    "WangGatewayCommandCoordinator",
    "WangGatewayTargetState",
    "WangNodeRX1Context",
    "WangPowerCommand",
    "WangPowerCommandError",
    "apply_wang_power_command",
    "decode_wang_power_command",
    "encode_wang_power_command",
    "is_wang_power_command",
]
