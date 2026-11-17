# ============================================================
# runtime/node_tx.py
# Physical DATA_DOWN / DATA_DOWN_FWD + 
# Relay APC RX cleanup, APC TX index, and timing validation
#
# End Device Uplink TX Generator
# Compatible with MType:
# - DATA_UP       : Node -> Relay/GW
# - DATA_UP_FWD   : Relay -> Relay/GW, handled by RelayTX
# - DATA_DOWN     : GW/Relay -> Node, handled by NodeRX
# - DATA_DOWN_FWD : GW/Relay -> Relay, handled by RelayTX/RelayRX
# - ACK           : handled by RX/TX control modules
# ============================================================

import hashlib
import random
from dataclasses import dataclass

import simpy

from loraphymac.runtime.packet import Packet
from loraphymac.mac.frame_defs import MType, FPort, encode_fctrl,frame_size_bytes
from loraphymac.runtime.wang_snr_reporting import (
    WangTransmitterState,
    encode_transmitter_state,
    reporting_enabled,
)


LEGACY_APPLICATION_SCHEDULING_MODE = "legacy_post_tx_done_v1"
INDEPENDENT_APPLICATION_SCHEDULING_MODE = "independent_fifo_v1"


def stable_application_rng_seed(experiment_seed: int, node_id: str) -> int:
    """Return a stable, process-independent seed for one application source."""
    material = f"sha256_seed_node_v1|{int(experiment_seed)}|{node_id}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class ApplicationMessage:
    """One exogenous application arrival awaiting physical transmission."""

    src_id: str
    fcnt: int
    packet_uid: str
    app_payload_size: int
    birth_time: float
    offered_time: float
    current_tx: str
    next_hop: str | None = None
    hop_index: int = 0
    mtype: int = int(MType.DATA_UP)


class NodeTX:
    """
    SimPy-based uplink packet generator for End Devices.

    Responsibilities:
    - Create the initial packet at the node.
    - Determine the uplink path from ``routing_map``.
    - Create a DATA_UP frame.
    - Send the packet through ``node.tx_event()``.
    - Open the RX1 window for APC DATA_DOWN when enabled.

    MType notes:
    - NodeTX creates only ``MType.DATA_UP``.
    - RelayTX creates ``DATA_UP_FWD`` when forwarding an uplink.
    """

    def __init__(
        self,
        env,
        node,
        routing_map,
        air_interface,
        model,
        payload_size,
        period,
        initial_delay=0.0,
        jitter=0.0,
        terrain_delta_h=90,
        frame_enabled=True,
        frame_version=0,
        frame_default_ack_req=False,
        application_scheduling_mode=LEGACY_APPLICATION_SCHEDULING_MODE,
        application_rng_seed=None,
        application_initial_delay_max_s=30.0,
    ):
        self.env = env
        self.node = node
        self.routing_map = routing_map
        self.air = air_interface
        self.model = model

        # Configuration value represents application payload only.
        self.app_payload_size = int(payload_size)
        
        if self.app_payload_size < 0:
            raise ValueError(
                f"{self.node.id}: payload_size must be >= 0"
            )
                         
        self.period = float(period)
        self.application_scheduling_mode = str(
            application_scheduling_mode
        ).lower().strip()
        if self.application_scheduling_mode not in {
            LEGACY_APPLICATION_SCHEDULING_MODE,
            INDEPENDENT_APPLICATION_SCHEDULING_MODE,
        }:
            raise ValueError(
                f"Unsupported application_scheduling_mode="
                f"{application_scheduling_mode!r}"
            )
        self.application_initial_delay_max_s = float(
            application_initial_delay_max_s
        )
        self._application_rng = None
        if (
            self.application_scheduling_mode
            == INDEPENDENT_APPLICATION_SCHEDULING_MODE
        ):
            if application_rng_seed is None:
                raise ValueError(
                    "independent_fifo_v1 requires application_rng_seed"
                )
            self._application_rng = random.Random(int(application_rng_seed))
            self.initial_delay = (
                self._application_rng.uniform(
                    0.0,
                    self.application_initial_delay_max_s,
                )
                if initial_delay is None
                else float(initial_delay)
            )
            self.application_queue = simpy.Store(self.env)
        else:
            if initial_delay is None:
                raise ValueError("legacy scheduling requires initial_delay")
            self.initial_delay = float(initial_delay)
            self.application_queue = None
        self.jitter = float(jitter)
        self.terrain_delta_h = terrain_delta_h

        # LoRaPhyMAC frame annotation control
        self.frame_enabled = bool(frame_enabled)
        self.frame_version = int(frame_version)
        self.frame_default_ack_req = bool(frame_default_ack_req)

        # Ensure node frame counter exists
        if not hasattr(self.node, "fcnt"):
            self.node.fcnt = 0

        self._next_application_fcnt = int(self.node.fcnt)
        self._next_application_due = self.initial_delay
        self._application_in_service = None
        self._application_generated_count = 0
        self._application_dequeued_count = 0
        self._application_max_backlog = 0
        self._application_waiting_times_s = []
        self._application_finalized = False

    # ---------------------------------------------------------
    # Main SimPy process
    # ---------------------------------------------------------
    def run(self):
        if (
            self.application_scheduling_mode
            == INDEPENDENT_APPLICATION_SCHEDULING_MODE
        ):
            yield from self._run_independent()
            return
        yield from self._run_legacy()

    def _run_legacy(self):
        """Preserve the historical post-TX_DONE application clock exactly."""
        # Initial offset -> counted as sleep
        if self.initial_delay > 0:
            yield from self.node.sleep_event(self.env, self.initial_delay)

        while True:
            fcnt = int(getattr(self.node, "fcnt", 0))
            pkt = yield from self._transmit_application(
                fcnt=fcnt,
                birth_time=float(self.env.now),
            )
            if pkt is None:
                yield from self.node.sleep_event(self.env, self.period)
                continue

            # -------------------------------------------------
            # 8. Increment frame counter after TX attempt
            # -------------------------------------------------
            self.node.fcnt = fcnt + 1

            # -------------------------------------------------
            # 9. Compute next TX time.
            #    Keep old behavior: period is counted after TX_DONE.
            #    RX1 is inserted inside this interval.
            # -------------------------------------------------
            interval = self.period

            if self.jitter > 0:
                interval += random.uniform(-self.jitter, self.jitter)

            interval = max(float(interval), 0.0)
            next_tx_due = float(self.env.now) + interval

            # -------------------------------------------------
            # 10. Run RX1 window with RX energy accounting
            # -------------------------------------------------
            if bool(getattr(self.node, "rx_window_enabled", False)):
                yield from self._run_rx1_window(pkt)

            # -------------------------------------------------
            # 11. Sleep remaining time until next period
            # -------------------------------------------------
            sleep_time = max(next_tx_due - float(self.env.now), 0.0)

            if sleep_time > 0:
                yield from self.node.sleep_event(self.env, sleep_time)

    def _application_producer(self):
        """Generate exogenous arrivals without waiting for the radio path."""
        if self.initial_delay > 0:
            yield self.env.timeout(self.initial_delay)

        while True:
            now = float(self.env.now)
            fcnt = self._next_application_fcnt
            message = ApplicationMessage(
                src_id=self.node.id,
                fcnt=fcnt,
                packet_uid=f"UP:{self.node.id}:{fcnt}",
                app_payload_size=self.app_payload_size,
                birth_time=now,
                offered_time=now,
                current_tx=self.node.id,
            )
            self._next_application_fcnt += 1
            self.node.fcnt = self._next_application_fcnt
            self._application_generated_count += 1
            if getattr(self.node, "stats", None) is not None:
                self.node.stats.record_application_offer(message)
            yield self.application_queue.put(message)
            self._application_max_backlog = max(
                self._application_max_backlog,
                len(self.application_queue.items),
            )

            interval = self.period
            if self.jitter > 0:
                interval += self._application_rng.uniform(
                    -self.jitter,
                    self.jitter,
                )
            interval = max(float(interval), 0.0)
            self._next_application_due = now + interval
            yield self.env.timeout(interval)

    def _run_independent(self):
        """Consume an unbounded FIFO while its producer runs independently."""
        self.env.process(self._application_producer())
        while True:
            if not self.application_queue.items:
                sleep_time = max(
                    float(self._next_application_due) - float(self.env.now),
                    0.0,
                )
                if sleep_time > 0:
                    yield from self.node.sleep_event(self.env, sleep_time)

            message = yield self.application_queue.get()
            dequeue_time = float(self.env.now)
            self._application_dequeued_count += 1
            waiting_time = max(0.0, dequeue_time - message.offered_time)
            self._application_waiting_times_s.append(waiting_time)
            self._application_in_service = message
            if getattr(self.node, "stats", None) is not None:
                self.node.stats.record_packet_fate_state(
                    message,
                    state="QUEUE_IN_SERVICE",
                    entity_id=self.node.id,
                    time=dequeue_time,
                )

            pkt = yield from self._transmit_application(
                fcnt=message.fcnt,
                birth_time=message.birth_time,
                packet_uid=message.packet_uid,
                offered_time=message.offered_time,
                queue_dequeue_time=dequeue_time,
            )
            if pkt is None:
                if getattr(self.node, "stats", None) is not None:
                    self.node.stats.record_packet_fate_state(
                        message,
                        state="TERMINAL_DROP",
                        entity_id=self.node.id,
                        time=float(self.env.now),
                        reason="no_route",
                    )
                self._application_in_service = None
                continue

            self._application_in_service = None
            if bool(getattr(self.node, "rx_window_enabled", False)):
                yield from self._run_rx1_window(pkt)

    def _transmit_application(
        self,
        *,
        fcnt,
        birth_time,
        packet_uid=None,
        offered_time=None,
        queue_dequeue_time=None,
    ):
        """Build a frame from current radio state and perform one physical TX."""
        path = self._build_path()
        if len(path) < 2:
            return None
        next_hop = path[1]
        if not hasattr(self.node, "network_nodes_map"):
            raise AttributeError(f"{self.node.id}: missing network_nodes_map")
        if next_hop not in self.node.network_nodes_map:
            raise KeyError(
                f"{self.node.id}: next_hop={next_hop} not found in network_nodes_map"
            )

        tx_mode = "DIRECT" if len(path) == 2 else "RELAY"
        ch_id, freq_mhz = self.node.select_channel(self.env.now)
        sf = int(self.node.default_sf)
        bw_hz = int(self.node.default_bw_hz)
        ptx = int(self.node.default_ptx)

        fopts = b""
        if self.frame_enabled and reporting_enabled(self.node):
            fopts = encode_transmitter_state(
                WangTransmitterState(
                    actual_ptx_dbm=ptx,
                    last_applied_command_seq=int(
                        getattr(self.node, "wang_last_applied_command_seq", 0)
                    ),
                )
            )

        if self.frame_enabled:
            node_apc_enabled = (
                str(getattr(self.node, "ptx_mode", "fixed")).lower().strip()
                == "adaptive"
            )
            frame_kwargs = {
                "mtype": MType.DATA_UP,
                "version": self.frame_version,
                "rfu": 0,
                "dst_id": path[-1],
                "hop_limit": max(1, len(path) - 1),
                "fctrl": encode_fctrl(
                    apc=node_apc_enabled,
                    ack_req=self.frame_default_ack_req,
                    bcast=False,
                    pending=False,
                    fopts_len=len(fopts),
                ),
                "fcnt": int(fcnt),
                "fopts": fopts,
                "fport": FPort.SENSOR_DATA,
            }
        else:
            frame_kwargs = {
                "mtype": None,
                "version": None,
                "rfu": None,
                "dst_id": None,
                "hop_limit": None,
                "fctrl": 0,
                "fcnt": int(fcnt),
                "fopts": fopts,
                "fport": None,
            }

        physical_payload_size = (
            frame_size_bytes(
                app_payload_size=self.app_payload_size,
                fopts_len=len(fopts),
            )
            if self.frame_enabled
            else self.app_payload_size
        )
        toa = self.air.compute_toa(
            sf=sf,
            bw_hz=bw_hz,
            payload_bytes=physical_payload_size,
        )
        now = float(self.env.now)
        pkt = Packet(
            src_id=self.node.id,
            current_tx=self.node.id,
            prev_hop=self.node.id,
            next_hop=next_hop,
            path=path,
            hop_index=0,
            ch_id=int(ch_id),
            freq_mhz=float(freq_mhz),
            sf=sf,
            bw_hz=bw_hz,
            ptx_dbm=ptx,
            payload_size=physical_payload_size,
            toa=float(toa),
            tx_start=now,
            tx_end=now + float(toa),
            packet_uid=packet_uid,
            app_payload_size=self.app_payload_size,
            birth_time=float(birth_time),
            tx_mode=tx_mode,
            **frame_kwargs,
        )
        if offered_time is not None:
            pkt.tx_queue_visit_id = f"APP:{pkt.packet_uid}"
            pkt.tx_queue_owner_id = self.node.id
            pkt.tx_queue_arrival_time = float(offered_time)
            pkt.tx_queue_dequeue_time = float(queue_dequeue_time)
            pkt.tx_queue_status = "IN_SERVICE"

        yield from self.node.tx_event(
            env=self.env,
            packet=pkt,
            model=self.model,
            nexthop=self.node.network_nodes_map[pkt.next_hop],
            terrain_delta_h=self.terrain_delta_h,
        )
        return pkt

    def finalize_observation(self, observation_end_time):
        """Record queue state at the hard horizon without scheduling events."""
        if (
            self.application_scheduling_mode
            != INDEPENDENT_APPLICATION_SCHEDULING_MODE
            or self._application_finalized
        ):
            return None
        self._application_finalized = True
        waiting = len(self.application_queue.items)
        in_service = 0
        if self._application_in_service is not None:
            row = getattr(self.node.stats, "packet_fates", {}).get(
                self._application_in_service.packet_uid,
                {},
            )
            in_service = int(row.get("latest_state") == "QUEUE_IN_SERVICE")
        waits = self._application_waiting_times_s
        record = {
            "run_id": self.node.stats.run_metadata.get("run_id"),
            "node_id": self.node.id,
            "scheduling_mode": self.application_scheduling_mode,
            "rng_policy": "sha256_seed_node_v1",
            "queue_policy": "unbounded_fifo_v1",
            "horizon_policy": "right_censor_waiting_v1",
            "observation_end_time": float(observation_end_time),
            "generated_count": self._application_generated_count,
            "dequeued_count": self._application_dequeued_count,
            "max_backlog_packets": self._application_max_backlog,
            "mean_queue_waiting_time_s": (
                sum(waits) / len(waits) if waits else 0.0
            ),
            "max_queue_waiting_time_s": max(waits) if waits else 0.0,
            "waiting_at_horizon_count": waiting,
            "in_service_at_horizon_count": in_service,
            "outstanding_at_horizon_count": waiting + in_service,
        }
        self.node.stats.record_application_queue_metrics(record)
        return record
                
    # ---------------------------------------------------------
    # Run RX1 window for DATA_DOWN APC with energy accounting
    # ---------------------------------------------------------
    def _run_rx1_window(self, pkt):
        """
        Open RX1 after uplink TX and account node RX energy.

        LoRaWAN-like behavior:
        - RX1 starts at pkt.tx_end + rx1_delay_s.
        - If no preamble/DATA_DOWN is received, node listens until RX1 timeout.
        - If valid DATA_DOWN is decoded, NodeRX closes the window early.
        """
        if not bool(getattr(self.node, "rx_window_enabled", False)):
            return

        rx1_delay_s = float(getattr(self.node, "rx1_delay_s", 1.0))
        rx_window_duration_s = float(
            getattr(self.node, "rx_window_duration_s", 0.25)
        )

        rx_start = float(pkt.tx_end) + rx1_delay_s
        rx_end = rx_start + rx_window_duration_s

        # Sleep until RX1 starts. This prevents double-counting sleep
        # during the RX window itself.
        sleep_before_rx_s = max(rx_start - float(self.env.now), 0.0)

        if sleep_before_rx_s > 0:
            yield from self.node.sleep_event(self.env, sleep_before_rx_s)

        # Open RX1 and account actual RX time.
        yield from self.node.open_rx_window(
            env=self.env,
            rx_start_s=rx_start,
            rx_end_s=rx_end,
            ch_id=getattr(pkt, "ch_id", None),
            freq_mhz=getattr(pkt, "freq_mhz", None),
            bw_hz=getattr(pkt, "bw_hz", None),
            sf=getattr(pkt, "sf", None),
            label="RX1",
        )
        """
        If APC DATA_DOWN uses a dedicated control channel, such as
        relay_tx_ch_ids=[1], set the RX1 ch_id to that downlink channel
        rather than pkt.ch_id.
        """
    
    # ---------------------------------------------------------
    # Build full uplink path from routing_map
    # ---------------------------------------------------------
    def _build_path(self):
        """
        Build full uplink path:
            Node -> Relay -> ... -> GW

        routing_map example:
            N1 -> R2
            R2 -> R1
            R1 -> GW
            GW -> None
        """
        path = [self.node.id]
        cur = self.node.id
        visited = {cur}

        while True:
            if cur not in self.routing_map:
                raise KeyError(f"{cur}: missing from routing_map")

            nxt = self.routing_map[cur]

            if nxt is None:
                return path

            if nxt not in self.routing_map:
                raise KeyError(f"{cur}: target {nxt} missing from routing_map")

            if nxt in visited:
                raise ValueError(f"Routing cycle detected at {nxt}")

            visited.add(nxt)
            path.append(nxt)
            cur = nxt
