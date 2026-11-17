# ============================================================
# runtime/packet.py
# Physical DATA_DOWN / DATA_DOWN_FWD + 
# Relay APC RX cleanup, APC TX index, and timing validation
#
# Packet runtime representation for LoRaPhyMAC
# Compatible with:
# - DATA_UP
# - DATA_UP_FWD
# - DATA_DOWN
# - DATA_DOWN_FWD
# - ACK
# ============================================================

import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Optional, Any


def stable_packet_object_id(packet: Any) -> int:
    """Return a deterministic packet-instance fingerprint for output records.

    The legacy output column is named ``packet_object_id``.  Its former value
    was Python's ``id(packet)``, which changes between simulation executions.
    This stable fingerprint uses packet identity and physical-hop fields only;
    it is for diagnostics and sorting, never for simulation decisions.
    """
    fields = (
        "packet_uid",
        "packet_id",
        "src_id",
        "current_tx",
        "prev_hop",
        "next_hop",
        "dst_id",
        "hop_index",
        "mtype",
        "fcnt",
        "ch_id",
        "freq_mhz",
        "sf",
        "bw_hz",
        "tx_start",
        "tx_end",
    )
    values = {
        name: (
            packet.get(name)
            if isinstance(packet, dict)
            else getattr(packet, name, None)
        )
        for name in fields
    }
    encoded = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    # Keep a positive signed-64-bit value for CSV/XLSX compatibility.
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


@dataclass
class Packet:
    """
    Runtime representation of a LoRa packet propagating in the air.

    Notes:
    - A packet stores frame, PHY, routing, and RX-result metadata.
    - A packet does not define MType logic. That logic remains in
      NodeTX, RelayTX, RelayRX, and NodeRX.
    """

    # ------------------------------------------------------------
    # Identity and routing (required, no defaults)
    # ------------------------------------------------------------
    src_id: str
    current_tx: str
    next_hop: str
    path: List[str]
    hop_index: int

    # ------------------------------------------------------------
    # PHY parameters (required)
    # ------------------------------------------------------------
    ch_id: int
    freq_mhz: float
    sf: int
    bw_hz: int
    ptx_dbm: float

    # Total LoRaPhyMAC frame size passed to PHY ToA calculation.
    payload_size: int
    
    
    # ------------------------------------------------------------
    # Timing (required)
    # ------------------------------------------------------------
    toa: float
    tx_start: float
    tx_end: float

    # Immutable identity for one application/physical packet within a run.
    # It is preserved by copy() and across every forwarding hop so finite-
    # horizon packet fate can be reconciled without relying on object ids.
    packet_uid: Optional[str] = None

    # Run-local physical transmission identity. A new value is assigned at
    # each actual TX start and carried by AirInterface receiver copies; a
    # forwarding/retry TX receives its own fresh physical identity.
    physical_tx_id: Optional[str] = None

    # ------------------------------------------------------------
    # Payload metadata
    # ------------------------------------------------------------
    # Application payload before LoRaPhyMAC header/FOpts/MIC.
    app_payload_size: int = 0

    
    # ------------------------------------------------------------
    # LoRaPhyMAC Frame Header
    # ------------------------------------------------------------
    # Optional[int] keeps the baseline safe when frame_enabled=False.
    # For example, node_tx.py may set mtype=None for the baseline.
    mtype: Optional[int] = 0x01
    version: Optional[int] = 0
    rfu: Optional[int] = 0

    dst_id: Optional[str] = None
    hop_limit: Optional[int] = 15

    fctrl: int = 0
    fcnt: int = 0
    fopts: bytes = b""
    fport: Optional[int] = 0x01
    mic_crc: Optional[int] = None

    # ------------------------------------------------------------
    # Runtime hop metadata
    # ------------------------------------------------------------
    # prev_hop is an explicit field rather than a dynamic attribute.
    prev_hop: Optional[str] = None
    tx_mode: Optional[str] = None
    # tx_mode examples:
    # "DIRECT", "RELAY", "APC_DOWN", "ACK"

    # ------------------------------------------------------------
    # Runtime RX results
    # ------------------------------------------------------------
    rssi: Optional[float] = None
    snr: Optional[float] = None
    margin: Optional[float] = None
    pathloss: Optional[float] = None
    feasible: Optional[bool] = None
    rx_node: Optional[str] = None
    tx_gain: Optional[float] = None
    tx_hw: Optional[Any] = None
    distance_m: Optional[float] = None

    # ------------------------------------------------------------
    # RX timing metadata
    # ------------------------------------------------------------
    rx_start: Optional[float] = None
    rx_end: Optional[float] = None
    t_arrive: Optional[float] = None
    t_end: Optional[float] = None

    # ------------------------------------------------------------
    # State flags
    # ------------------------------------------------------------
    collided: bool = False
    captured: bool = False
    decoded: bool = False
    reason: Optional[str] = None

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------
    birth_time: float = field(default=0.0)
    rx_time: Optional[float] = None

    # ------------------------------------------------------------
    # Queue metadata
    # ------------------------------------------------------------
    tx_queue_visit_id: Optional[str] = None
    tx_queue_owner_id: Optional[str] = None
    tx_queue_arrival_time: Optional[float] = None
    tx_queue_dequeue_time: Optional[float] = None
    tx_duty_ready_time: Optional[float] = None
    tx_rx_idle_time: Optional[float] = None
    tx_service_end_time: Optional[float] = None
    tx_worker_id: Optional[int] = None
    tx_queue_status: Optional[str] = None

    # ------------------------------------------------------------
    # APC metadata
    # ------------------------------------------------------------
    apc_old_ptx_dbm: Optional[float] = None
    apc_new_ptx_dbm: Optional[float] = None
    apc_info: Optional[dict] = None
    apc_uplink_src_id: Optional[str] = None
    apc_uplink_hop_index: Optional[int] = None

    # ------------------------------------------------------------
    # Post init normalization
    # ------------------------------------------------------------
    def __post_init__(self):
        if self.path is None:
            self.path = []

        if self.fopts is None:
            self.fopts = b""
        else:
            self.fopts = bytes(self.fopts)

        if self.prev_hop is None:
            self.prev_hop = self.current_tx
        
        # >>> PATCH: normalize frame-size metadata
        self.payload_size = int(self.payload_size)
        self.app_payload_size = int(self.app_payload_size)
    
        if self.payload_size <= 0:
            raise ValueError(
                f"payload_size must be > 0, "
                f"got {self.payload_size}"
            )
    
        if self.app_payload_size < 0:
            raise ValueError(
                f"app_payload_size must be >= 0, "
                f"got {self.app_payload_size}"
            )
    
        if self.app_payload_size > self.payload_size:
            raise ValueError(
                "app_payload_size cannot exceed payload_size: "
                f"app={self.app_payload_size}, "
                f"frame={self.payload_size}"
            )
    
        # FOptsLen must be representable by the FCtrl field.
        if len(self.fopts) > 15:
            raise ValueError(
                f"FOpts length must be <= 15 bytes, "
                f"got {len(self.fopts)}"
            )


    
    # ------------------------------------------------------------
    # Helper properties
    # ------------------------------------------------------------
    @property
    def duration(self) -> float:
        return float(self.toa)

    @property
    def hop_count(self) -> int:
        return max(0, len(self.path) - 1)

    @property
    def is_last_hop(self) -> bool:
        return int(self.hop_index) >= len(self.path) - 1

    @property
    def latency(self) -> Optional[float]:
        if self.rx_time is None:
            return None
        return float(self.rx_time) - float(self.birth_time)

    # ------------------------------------------------------------
    # Hop control
    # ------------------------------------------------------------
    def advance_hop(self):
        """
        Advance packet to the next hop in path.

        Notes:
        - ``prev_hop`` is set to the transmitter before the hop changes.
        - ``current_tx`` becomes the node/relay at the new ``hop_index``.
        - ``next_hop`` becomes the next physical destination.
        """
        if self.is_last_hop:
            return

        self.prev_hop = self.current_tx
        self.hop_index += 1
        self.current_tx = self.path[self.hop_index]

        if not self.is_last_hop:
            self.next_hop = self.path[self.hop_index + 1]
        else:
            self.next_hop = self.current_tx

    # ------------------------------------------------------------
    # RX runtime cleanup
    # ------------------------------------------------------------
    def reset_rx_runtime_metadata(self, preserve_t_arrive: bool = True):
        """
        Clear RX-result metadata before transmitting a packet on a new hop.

        Fields such as ``rx_time``, ``rssi``, ``margin``, ``rx_start``, and
        ``rx_end`` are observations from the previous receiver. Carrying them
        to the next hop can make the gateway record relay RX time as gateway
        RX time, producing an incorrect multi-hop end-to-end delay.

        ``preserve_t_arrive=True`` is used by ``AirInterface.transmit()``:
        the transmitter may already have set ``t_arrive`` to schedule
        propagation for the new hop, so it must not always be cleared.
        """
        self.rssi = None
        self.snr = None
        self.margin = None
        self.pathloss = None
        self.feasible = None
        self.rx_node = None
        self.distance_m = None

        self.rx_start = None
        self.rx_end = None
        self.t_end = None
        self.rx_time = None

        if not preserve_t_arrive:
            self.t_arrive = None

        self.collided = False
        self.captured = False
        self.decoded = False
        self.reason = None

        # Some modules add this flag dynamically.
        if hasattr(self, "phy_ready"):
            self.phy_ready = False
        if hasattr(self, "demod_drop"):
            self.demod_drop = False
        for attr in (
            "demod_admitted",
            "demod_reason",
            "demod_pool",
            "demod_acquire_time",
            "demod_release_time",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

        # Half-duplex RX cancellation state. Do not carry cancellation state
        # from the previous hop when a packet is copied or forwarded
        # to the next hop.
        if hasattr(self, "rx_cancelled_by_tx"):
            self.rx_cancelled_by_tx = False
        if hasattr(self, "rx_cancelled_at"):
            self.rx_cancelled_at = None
        if hasattr(self, "rx_cancelled_by"):
            self.rx_cancelled_by = None
        if hasattr(self, "drop_time"):
            self.drop_time = None

        return self

    # ------------------------------------------------------------
    # Safe copy for multi-hop TX/RX
    # ------------------------------------------------------------
    def copy(self):
        """
        Safe copy for air delivery and forwarding.

        Used by ``AirInterface.transmit()`` so the receiver gets a copy
        rather than the original packet object.
        """

        pkt = Packet(
            # ----------------------------------------------------
            # Identity & Routing
            # ----------------------------------------------------
            src_id=self.src_id,
            current_tx=self.current_tx,
            next_hop=self.next_hop,
            path=list(self.path),
            hop_index=self.hop_index,

            # ----------------------------------------------------
            # PHY Parameters
            # ----------------------------------------------------
            ch_id=self.ch_id,
            freq_mhz=self.freq_mhz,
            sf=self.sf,
            bw_hz=self.bw_hz,
            ptx_dbm=self.ptx_dbm,
      
            # Total physical frame size
            payload_size=self.payload_size,

            # ----------------------------------------------------
            # Timing
            # ----------------------------------------------------
            toa=self.toa,
            tx_start=self.tx_start,
            tx_end=self.tx_end,
            packet_uid=self.packet_uid,
            physical_tx_id=self.physical_tx_id,

            # Application payload only
            app_payload_size=self.app_payload_size,
            
            # ----------------------------------------------------
            # LoRaPhyMAC Frame Header
            # ----------------------------------------------------
            mtype=self.mtype,
            version=self.version,
            rfu=self.rfu,
            dst_id=self.dst_id,
            hop_limit=self.hop_limit,
            fctrl=self.fctrl,
            fcnt=self.fcnt,
            fopts=bytes(self.fopts or b""),
            fport=self.fport,
            mic_crc=self.mic_crc,

            # ----------------------------------------------------
            # Runtime hop metadata
            # ----------------------------------------------------
            prev_hop=self.current_tx,
            tx_mode=self.tx_mode,

            # ----------------------------------------------------
            # Runtime RX results
            # ----------------------------------------------------
            rssi=self.rssi,
            snr=self.snr,
            margin=self.margin,
            pathloss=self.pathloss,
            feasible=self.feasible,
            rx_node=self.rx_node,
            tx_gain=self.tx_gain,
            tx_hw=self.tx_hw,
            distance_m=self.distance_m,

            # ----------------------------------------------------
            # RX timing metadata
            # ----------------------------------------------------
            rx_start=self.rx_start,
            rx_end=self.rx_end,
            t_arrive=self.t_arrive,
            t_end=self.t_end,

            # ----------------------------------------------------
            # State flags
            # ----------------------------------------------------
            collided=self.collided,
            captured=self.captured,
            decoded=self.decoded,
            reason=self.reason,

            # ----------------------------------------------------
            # Metadata
            # ----------------------------------------------------
            birth_time=self.birth_time,
            rx_time=self.rx_time,

            # ----------------------------------------------------
            # Queue metadata
            # ----------------------------------------------------
            tx_queue_visit_id=self.tx_queue_visit_id,
            tx_queue_owner_id=self.tx_queue_owner_id,
            tx_queue_arrival_time=self.tx_queue_arrival_time,
            tx_queue_dequeue_time=self.tx_queue_dequeue_time,
            tx_duty_ready_time=self.tx_duty_ready_time,
            tx_rx_idle_time=self.tx_rx_idle_time,
            tx_service_end_time=self.tx_service_end_time,
            tx_worker_id=self.tx_worker_id,
            tx_queue_status=self.tx_queue_status,

            # ----------------------------------------------------
            # APC metadata
            # ----------------------------------------------------
            apc_old_ptx_dbm=self.apc_old_ptx_dbm,
            apc_new_ptx_dbm=self.apc_new_ptx_dbm,
            apc_info=self.apc_info,
            apc_uplink_src_id=self.apc_uplink_src_id,
            apc_uplink_hop_index=self.apc_uplink_hop_index,
        )

        # RX runtime metadata from the previous hop must not reach the next
        # receiver. Keep t_arrive because AirInterface may use it to
        # schedule propagation.
        pkt.reset_rx_runtime_metadata(preserve_t_arrive=True)

        return pkt
