# ============================================================
# runtime/relay_tx.py
# Physical DATA_DOWN / DATA_DOWN_FWD +
# Relay APC RX cleanup, APC TX index, and timing validation
#
# Refactored RelayTX packet preparation
#
# Relay TX bottleneck (M/G/c queue)
# - Relay PHY injected by loraphymac.simulation.runner
# - Relay PTX supports FIXED / ADAPTIVE
# - PTx validation uses validate_ptx() from hardware.py
# - Supports relay_duty_cycle = None / 0.01 / 0.10
# - Records relay TX energy, total airtime, and per-channel airtime

# Supports relay/gateway energy summary for:
# - half-duplex or RX-while-TX relay model
# - single or multi TX radio capacity

# ============================================================

import simpy

from loraphymac.domain.node import observed_duration_s
from loraphymac.hardware.hardware import validate_ptx
from loraphymac.mac.frame_defs import MType, decode_fctrl, encode_fctrl
from loraphymac.runtime.wang_snr_reporting import (
    declare_transmitter_state,
    reporting_enabled,
)

class RelayTX:
    def __init__(
        self,
        env,
        relay,
        air_interface,
        statistics,
        relay_sf,
        relay_bw_hz,
        relay_ch_ids,
        channel_plan,
        relay_duty_cycle=0.01,
        rx_idle_poll_interval_ms=5.0,
        max_rx_idle_wait_ms=200.0,
        frame_enabled=True,
        tx_waiting_queue_capacity_packets=None,
        tx_queue_overflow_policy="drop_tail",
    ):
        self.env = env
        self.relay = relay
        self.air = air_interface
        self.stats = statistics
        # ----------------------------------------------------
        # Relay PHY injected from main_simulation.py
        # ----------------------------------------------------
        self.relay_sf = int(relay_sf)
        self.relay_bw_hz = int(relay_bw_hz)
        self.relay_tx_ch_ids = list(relay_ch_ids)
        self.channel_plan = channel_plan
        self.tx_rr_index = 0
        self.relay_duty_cycle = relay_duty_cycle
        self.frame_enabled = bool(frame_enabled)

        # Maximum polling interval for RX state.
        # This is not a maximum total waiting time.

        # Polling interval, not a total timeout.
        self.rx_idle_poll_interval_ms = max( 0.001,float(rx_idle_poll_interval_ms),)

        # Total RX-priority limit before admission closes.
        self.max_rx_idle_wait_ms = max(0.0,float(max_rx_idle_wait_ms),)

        # ----------------------------------------------------
        # M/G/c forwarding queue for relay TX
        # ----------------------------------------------------
        # tx_phy_radio_count defines the number of physical TX radio paths.
        # tx_capacity_c is interpreted as the number of parallel TX servers.
        # tx_waiting_queue is the shared forwarding queue.
        #
        # In main_simulation, tx_capacity_c workers are started:
        #     for _ in range(tx.tx_capacity_c):
        #         env.process(tx.run())
        #
        # Therefore, no additional simpy.Resource is required here.
        # The service capacity is controlled directly by the number of workers.
        tx_phy_radio_count = int(getattr(self.relay.hw, "tx_phy_radio_count", 1))
        self.tx_capacity_c = max(1, tx_phy_radio_count)

        if (
            not bool(getattr(self.relay.hw, "tx_full_duplex", False))
            and self.tx_capacity_c > 1
        ):
            raise ValueError(
                f"{self.relay.id}: half-duplex hardware must use "
                f"tx_phy_radio_count=1, got {self.tx_capacity_c}"
            )



        # Shared FIFO forwarding queue. Capacity counts waiting packets only;
        # packets already claimed by workers are tracked separately.
        if tx_waiting_queue_capacity_packets is None:
            self.tx_waiting_queue_capacity_packets = None
            self.tx_waiting_queue = simpy.Store(env)
        else:
            capacity = int(tx_waiting_queue_capacity_packets)
            if capacity < 1:
                raise ValueError(
                    f"{self.relay.id}: TX waiting queue capacity must be >= 1 "
                    "or None"
                )
            self.tx_waiting_queue_capacity_packets = capacity
            self.tx_waiting_queue = simpy.Store(env, capacity=capacity)

        self.tx_queue_overflow_policy = str(
            tx_queue_overflow_policy
        ).lower().strip()
        if self.tx_queue_overflow_policy != "drop_tail":
            raise ValueError(
                f"{self.relay.id}: unsupported TX queue overflow policy "
                f"{tx_queue_overflow_policy!r}"
            )

        self.in_service_packets = {}
        self.tx_queue_enqueue_sequence = 0
        self._queue_observation_finalized = False

        if not hasattr(self.relay, "tx_active_count"):
            self.relay.tx_active_count = 0

        # ----------------------------------------------------
        # Runtime counters and defaults
        # ----------------------------------------------------
        if not hasattr(self.relay, "total_tx_time_s"):
            self.relay.total_tx_time_s = 0.0

        if not hasattr(self.relay, "energy_tx_mJ"):
            self.relay.energy_tx_mJ = 0.0

        if not hasattr(self.relay, "energy_transition_mJ"):
            self.relay.energy_transition_mJ = 0.0

        if not hasattr(self.relay, "tx_airtime_per_channel"):
            self.relay.tx_airtime_per_channel = {}

        if not hasattr(self.relay, "tx_packet_count_per_channel"):
            self.relay.tx_packet_count_per_channel = {}

        if not hasattr(self.relay, "total_transition_time_s"):
            self.relay.total_transition_time_s = 0.0

        # ----------------------------------------------------
        # Duty-cycle timer
        # ----------------------------------------------------
        # Retain the global timer for backward compatibility.
        if not hasattr(self.relay, "next_tx_allowed"):
            self.relay.next_tx_allowed = 0.0

        # Per-channel timers for multi-PHY or multi-channel TX.
        if not hasattr(self.relay, "next_tx_allowed_per_channel"):
            self.relay.next_tx_allowed_per_channel = {}

        # NOTE:
        # If main_simulation.py has already set relay.include_mcu_energy,
        # do not overwrite it. Keep the False default for backward-compatible
        # relay behavior.
        if not hasattr(self.relay, "include_mcu_energy"):
            self.relay.include_mcu_energy = False
        # ----------------------------------------------------
        # Full-duplex / half-duplex runtime state
        # ----------------------------------------------------

        if not hasattr(self.relay, "tx_pending"):
            self.relay.tx_pending = False

        if not hasattr(self.relay, "is_transmitting"):
            self.relay.is_transmitting = False

        if not hasattr(self.relay, "tx_busy_until"):
            self.relay.tx_busy_until = 0.0

        if not hasattr(self.relay, "rx_busy_until"):
            self.relay.rx_busy_until = 0.0

        if not hasattr(self.relay, "is_switching_to_tx"):
            self.relay.is_switching_to_tx = False

        if not hasattr(self.relay, "tx_transition_until"):
            self.relay.tx_transition_until = 0.0


        # Active RX registry shared with RelayRX.
        #
        # On half-duplex hardware, RelayTX waits for every
        # active RX session to finish before starting an RX-to-TX transition.
        #
        # Cancellation is used only as a safeguard for detecting
        # unexpected race conditions.
        if not hasattr(self.relay, "active_rx_packets"):
            self.relay.active_rx_packets = []

        #print(f"[INIT] RelayTX {relay.id} Ready")

    def _record_queue_length(self, event=None):
        """Record current waiting-queue length for time-weighted queue metrics."""
        q_len = len(getattr(self.tx_waiting_queue, "items", []) or [])
        self.stats.record_tx_queue_length(
            relay_id=self.relay.id,
            queue_length=q_len,
            time=self.env.now,
            event=event,
        )

    def _record_queue_state(self, event=None):
        """Record waiting, in-service, and total-system occupancy."""
        waiting_count = len(
            getattr(self.tx_waiting_queue, "items", []) or []
        )
        in_service_count = len(self.in_service_packets)
        self.stats.record_tx_queue_state(
            relay_id=self.relay.id,
            time=self.env.now,
            waiting_count=waiting_count,
            in_service_count=in_service_count,
            event=event,
        )

    @staticmethod
    def _safe_time(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _traffic_class(packet):
        mtype = getattr(packet, "mtype", None)
        if mtype in (MType.DATA_UP, MType.DATA_UP_FWD):
            return "uplink_forward"
        if mtype in (MType.DATA_DOWN, MType.DATA_DOWN_FWD):
            return "apc_control"
        return "legacy"

    def _record_queue_lifecycle(
        self,
        packet,
        status,
        observation_end_time=None,
    ):
        """Record one terminal or right-censored queue visit."""
        arrival = self._safe_time(
            getattr(packet, "tx_queue_arrival_time", None)
        )
        dequeue = self._safe_time(
            getattr(packet, "tx_queue_dequeue_time", None)
        )
        duty_ready = self._safe_time(
            getattr(packet, "tx_duty_ready_time", None)
        )
        rx_idle = self._safe_time(
            getattr(packet, "tx_rx_idle_time", None)
        )
        tx_start = self._safe_time(getattr(packet, "tx_start", None))
        tx_end = self._safe_time(getattr(packet, "tx_end", None))
        service_end = self._safe_time(
            getattr(packet, "tx_service_end_time", None)
        )
        observation_end = self._safe_time(observation_end_time)

        def elapsed(end, start):
            if end is None or start is None:
                return None
            return max(0.0, end - start)

        completed = status == "COMPLETED"
        observed_airtime_s = None
        if tx_start is not None and tx_end is not None:
            observed_tx_end = tx_end
            if not completed:
                if observation_end is None:
                    observed_tx_end = tx_start
                else:
                    observed_tx_end = min(tx_end, observation_end)
            observed_airtime_s = elapsed(observed_tx_end, tx_start)
        packet.tx_queue_status = status

        self.stats.record_tx_queue_lifecycle({
            "queue_visit_id": getattr(packet, "tx_queue_visit_id", None),
            "entity_id": self.relay.id,
            "entity_type": self._entity_type(self.relay.id),
            "worker_id": getattr(packet, "tx_worker_id", None),
            "packet_uid": getattr(packet, "packet_uid", None),
            "src_id": getattr(packet, "src_id", None),
            "fcnt": getattr(packet, "fcnt", None),
            "mtype": getattr(packet, "mtype", None),
            "traffic_class": self._traffic_class(packet),
            "hop_index": getattr(packet, "hop_index", None),
            "channel": getattr(packet, "ch_id", None),
            "queue_capacity_packets": self.tx_waiting_queue_capacity_packets,
            "status": status,
            "queue_arrival_time": arrival,
            "queue_dequeue_time": dequeue,
            "duty_ready_time": duty_ready,
            "rx_idle_time": rx_idle,
            "tx_start": tx_start,
            "tx_end": tx_end,
            "service_end_time": service_end,
            "observation_end_time": observation_end,
            "fifo_wait_s": elapsed(dequeue, arrival),
            "duty_wait_s": elapsed(duty_ready, dequeue),
            "half_duplex_wait_s": elapsed(rx_idle, duty_ready),
            "rx_to_tx_transition_s": elapsed(tx_start, rx_idle),
            "airtime_s": elapsed(tx_end, tx_start) if completed else None,
            "observed_airtime_s": observed_airtime_s,
            "tx_to_rx_transition_s": (
                elapsed(service_end, tx_end) if completed else None
            ),
            "worker_service_s": (
                elapsed(service_end, dequeue) if completed else None
            ),
            "time_to_tx_start_s": elapsed(tx_start, arrival),
            "system_sojourn_s": (
                elapsed(service_end, arrival) if completed else None
            ),
            "censored_waiting_age_s": (
                elapsed(observation_end, arrival)
                if status == "CENSORED_WAITING" else None
            ),
            "censored_service_age_s": (
                elapsed(observation_end, dequeue)
                if status == "CENSORED_IN_SERVICE" else None
            ),
        })

    def finalize_observation(self, observation_end_time):
        """Snapshot unfinished queue visits without mutating the simulation."""
        if self._queue_observation_finalized:
            return

        self._record_queue_state(event="observation_end")
        for packet in list(self.tx_waiting_queue.items):
            self._record_queue_lifecycle(
                packet,
                status="CENSORED_WAITING",
                observation_end_time=observation_end_time,
            )
        for packet in list(self.in_service_packets.values()):
            self._record_queue_lifecycle(
                packet,
                status="CENSORED_IN_SERVICE",
                observation_end_time=observation_end_time,
            )

        self._queue_observation_finalized = True

    # --------------------------------------------------------
    # Queue packet from RelayRX
    # --------------------------------------------------------
    def put(self, packet):
        self.tx_queue_enqueue_sequence += 1
        packet.tx_queue_visit_id = (
            f"{self.relay.id}:{self.tx_queue_enqueue_sequence}"
        )
        packet.tx_queue_owner_id = self.relay.id
        packet.tx_queue_arrival_time = self.env.now
        packet.tx_queue_dequeue_time = None
        packet.tx_duty_ready_time = None
        packet.tx_rx_idle_time = None
        packet.tx_service_end_time = None
        packet.tx_worker_id = None
        packet.tx_queue_status = "ENQUEUE_ATTEMPT"
        # The packet may still carry physical TX timestamps from the previous
        # hop. They are not valid for this new queue visit and must not make a
        # waiting/overflow visit look as if RelayTX had already transmitted it.
        packet.tx_start = None
        packet.tx_end = None

        self.stats.record_relay_rx(
            self.relay.id,
            "tx_queue_enqueue_attempt_count",
        )

        capacity = self.tx_waiting_queue_capacity_packets
        if (
            capacity is not None
            and len(self.tx_waiting_queue.items) >= capacity
        ):
            packet.tx_queue_status = "OVERFLOW_DROP"
            packet.drop_time = float(self.env.now)
            packet.reason = "forwarding_queue_overflow"
            self.stats.record_relay_rx(
                self.relay.id,
                "tx_queue_overflow_drop_count",
            )
            self.stats.record_drop(
                packet,
                reason="forwarding_queue_overflow",
                node_id=self.relay.id,
            )
            self._record_queue_lifecycle(
                packet,
                status="OVERFLOW_DROP",
                observation_end_time=self.env.now,
            )
            return False

        self.stats.record_relay_rx(
            self.relay.id,
            "tx_queue_accepted_count",
        )
        # Backward-compatible accepted-arrival counter.
        self.stats.record_relay_rx(
            self.relay.id,
            "tx_queue_arrival_count",
        )

        put_event = self.tx_waiting_queue.put(packet)
        if not put_event.triggered:
            raise RuntimeError(
                f"{self.relay.id}: accepted queue put did not complete "
                "immediately"
            )
        packet.tx_queue_status = "WAITING"
        self.stats.record_packet_fate_state(
            packet,
            state="QUEUE_WAITING",
            entity_id=self.relay.id,
            time=self.env.now,
        )
        self._record_queue_state(event="enqueue")
        return True

    # --------------------------------------------------------
    # Select PTx for forwarding
    # --------------------------------------------------------
    def _select_ptx_for_forward(self, packet):
        mode = str(getattr(self.relay, "ptx_mode", "fixed")).lower().strip()

        if mode == "adaptive":
            return int(getattr(self.relay, "default_ptx", packet.ptx_dbm))

        return int(getattr(self.relay, "fixed_ptx", packet.ptx_dbm))



    def _select_ptx_for_downlink_control(self, packet):
        """
        Select physical PTx for APC/control downlink.

        MType:
        - DATA_DOWN     : GW/Relay -> Node
        - DATA_DOWN_FWD : GW/Relay -> Relay

        Priority:
        1. Specific downlink PTx for node/relay
        2. Generic APC downlink PTx
        3. Relay adaptive/fixed forwarding PTx
        """

        mtype = getattr(packet, "mtype", None)

        if mtype == MType.DATA_DOWN:
            return int(
                getattr(
                    self.relay,
                    "apc_downlink_node_ptx_dbm",
                    getattr(
                        self.relay,
                        "apc_downlink_ptx_dbm",
                        self._select_ptx_for_forward(packet),
                    ),
                )
            )

        if mtype == MType.DATA_DOWN_FWD:
            return int(
                getattr(
                    self.relay,
                    "apc_downlink_relay_ptx_dbm",
                    getattr(
                        self.relay,
                        "apc_downlink_ptx_dbm",
                        self._select_ptx_for_forward(packet),
                    ),
                )
            )

        return self._select_ptx_for_forward(packet)


    def _select_next_tx_channel(self, packet):
        """
        Select a relay/gateway TX channel using round-robin scheduling.

        Used by:
        - DATA_UP_FWD
        - DATA_DOWN_FWD

        Not used by node-bound DATA_DOWN.
        Node-bound DATA_DOWN follows the node DATA_UP channel.
        """
        if len(self.relay_tx_ch_ids) == 0:
            raise ValueError(
                f"{self.relay.id}: relay_tx_ch_ids is empty."
            )

        idx = self.tx_rr_index % len(self.relay_tx_ch_ids)
        self.tx_rr_index += 1

        packet.ch_id = int(self.relay_tx_ch_ids[idx])
        packet.freq_mhz = float(self.channel_plan.get_freq(packet.ch_id))

        return packet


    def _apply_common_relay_phy(
        self,
        packet,
        selected_ptx,
        use_relay_tx_channel=True,
    ):
        packet.sf = self.relay_sf
        packet.bw_hz = self.relay_bw_hz

        if use_relay_tx_channel:
            # DATA_UP_FWD and DATA_DOWN_FWD use a relay-backbone channel.
            packet = self._select_next_tx_channel(packet)
            packet.downlink_channel_policy = "relay_tx_ch_ids"

        else:
            # Node-bound DATA_DOWN retains the node DATA_UP ch_id.
            # relay_tx_ch_ids must not be used for DATA_DOWN.
            uplink_ch_id = getattr(packet, "ch_id", None)

            if uplink_ch_id is None:
                raise ValueError(
                    f"{self.relay.id}: DATA_DOWN packet missing uplink ch_id."
                )

            packet.ch_id = int(uplink_ch_id)
            packet.freq_mhz = float(self.channel_plan.get_freq(packet.ch_id))
            packet.downlink_channel_policy = "follow_uplink_ch"

        packet.ptx_dbm = validate_ptx(
            entity=self.relay,
            ptx=selected_ptx,
            label=self.relay.id,
        )

        # >>> PATCH:
        # RelayTX must preserve the frame size calculated by NodeTX or RelayRX.
        try:
            packet.payload_size = int(packet.payload_size)
        except (TypeError, ValueError):
            raise ValueError(
                f"{self.relay.id}: invalid packet.payload_size="
                f"{getattr(packet, 'payload_size', None)}"
            )

        if packet.payload_size <= 0:
            raise ValueError(
                f"{self.relay.id}: packet.payload_size must be > 0, "
                f"got {packet.payload_size}"
            )

        packet.toa = self.air.compute_toa(
            sf=packet.sf,
            bw_hz=packet.bw_hz,
            payload_bytes=packet.payload_size,
        )

        return packet

    def _prepare_downlink_control_packet(self, packet):
        """
        Prepare DATA_DOWN / DATA_DOWN_FWD.

        DATA_DOWN:
            Relay/GW -> Node
            The channel follows the node DATA_UP channel.

        DATA_DOWN_FWD:
            Relay/GW -> Relay
            The channel uses relay_tx_ch_ids.
        """
        packet.current_tx = self.relay.id
        packet.prev_hop = self.relay.id

        if getattr(packet, "next_hop", None) is None:
            raise ValueError(
                f"{self.relay.id}: APC downlink packet missing next_hop."
            )

        selected_ptx = self._select_ptx_for_downlink_control(packet)
        mtype = getattr(packet, "mtype", None)

        if mtype == MType.DATA_DOWN:
            # Node RX1 follows the node DATA_UP channel.
            old_ch = getattr(packet, "ch_id", None)

            packet = self._apply_common_relay_phy(
                packet=packet,
                selected_ptx=selected_ptx,
                use_relay_tx_channel=False,
            )

            if bool(getattr(self.relay, "debug_downlink_channel", False)):
                print(
                    f"[DATA_DOWN_CH] t={self.env.now:.6f} "
                    f"tx={self.relay.id} "
                    f"target={packet.next_hop} "
                    f"input_ch={old_ch} "
                    f"final_ch={packet.ch_id} "
                    f"apc_uplink_ch={getattr(packet, 'apc_uplink_ch_id', None)} "
                    f"policy={getattr(packet, 'downlink_channel_policy', None)}"
                )

            return packet

        if mtype == MType.DATA_DOWN_FWD:
            # The relay backbone uses relay_tx_ch_ids.
            return self._apply_common_relay_phy(
                packet=packet,
                selected_ptx=selected_ptx,
                use_relay_tx_channel=True,
            )

        raise ValueError(
            f"{self.relay.id}: invalid downlink control mtype={mtype}"
        )

    def _prepare_uplink_forward_packet(self, packet):
        """
        Convert DATA_UP or DATA_UP_FWD to DATA_UP_FWD.

        This path is for uplink forwarding only:
        - Node -> Relay/GW
        - Relay -> Relay/GW
        """
        packet.current_tx = self.relay.id
        packet.prev_hop = self.relay.id
        selected_ptx = self._select_ptx_for_forward(packet)

        if self.frame_enabled:
            packet.mtype = MType.DATA_UP_FWD

            if getattr(packet, "dst_id", None) is None:
                packet.dst_id = packet.path[-1]

            if getattr(packet, "hop_limit", None) is None:
                packet.hop_limit = max(1, len(packet.path) - 1)

            relay_apc_enabled = (
                str(getattr(self.relay, "ptx_mode", "fixed")).lower().strip()
                == "adaptive"
            )

            old_fctrl_info = decode_fctrl(getattr(packet, "fctrl", 0))

            if reporting_enabled(self.relay):
                declare_transmitter_state(
                    packet,
                    actual_ptx_dbm=selected_ptx,
                    last_applied_command_seq=int(
                        getattr(self.relay, "wang_last_applied_command_seq", 0)
                    ),
                )

            fopts = getattr(packet, "fopts", b"")
            if fopts is None:
                fopts = b""

            packet.fctrl = encode_fctrl(
                apc=relay_apc_enabled,
                ack_req=bool(old_fctrl_info.get("ack_req", False)),
                bcast=bool(old_fctrl_info.get("bcast", False)),
                pending=bool(old_fctrl_info.get("pending", False)),
                fopts_len=len(fopts),
            )

        return self._apply_common_relay_phy(
            packet=packet,
            selected_ptx=selected_ptx,
        )

    def _prepare_forward_packet(self, packet):
        """
        Dispatcher TX packet preparation.

        Jalur:
        - DATA_DOWN / DATA_DOWN_FWD  -> downlink control/APC
        - DATA_UP / DATA_UP_FWD      -> uplink forwarding
        - mtype=None                 -> legacy/frame-off uplink
        """
        mtype = getattr(packet, "mtype", None)

        # ----------------------------------------------------
        # Downlink control path
        # ----------------------------------------------------
        if mtype in (MType.DATA_DOWN, MType.DATA_DOWN_FWD):
            return self._prepare_downlink_control_packet(packet)

        # ----------------------------------------------------
        # Guard unsupported MType
        # ----------------------------------------------------
        # Allow mtype=None for the frame_enabled=False baseline.
        if mtype is not None and mtype not in (MType.DATA_UP, MType.DATA_UP_FWD):
            raise ValueError(
                f"{self.relay.id}: unsupported packet mtype for relay TX: {mtype}"
            )

        # ----------------------------------------------------
        # Uplink forwarding path
        # ----------------------------------------------------
        return self._prepare_uplink_forward_packet(packet)

    # --------------------------------------------------------
    # Record relay airtime
    # --------------------------------------------------------
    def _record_airtime(self, packet):
        toa = observed_duration_s(self.env, float(packet.toa))

        # Total relay TX airtime
        self.relay.total_tx_time_s += toa

        # Per-channel relay TX airtime
        ch_id = getattr(packet, "ch_id", None)

        if ch_id is not None:
            ch_id = int(ch_id)

            self.relay.tx_airtime_per_channel[ch_id] = (
                self.relay.tx_airtime_per_channel.get(ch_id, 0.0)
                + toa
            )

            self.relay.tx_packet_count_per_channel[ch_id] = (
                self.relay.tx_packet_count_per_channel.get(ch_id, 0)
                + 1
            )

    # --------------------------------------------------------
    # Record relay TX energy
    # --------------------------------------------------------
    def _record_energy(self, packet):
        # Validate again before energy calculation.
        # This protects energy model from invalid PTx.
        ptx = validate_ptx(
            entity=self.relay,
            ptx=packet.ptx_dbm,
            label=self.relay.id,
        )

        i_tx_mA = float(self.relay.hw.tx_current[ptx])
        voltage = float(self.relay.hw.voltage)
        toa = observed_duration_s(self.env, float(packet.toa))

        # Radio TX energy in mJ
        energy_tx_mJ = voltage * (i_tx_mA / 1000.0) * toa * 1000.0
        self.relay.energy_tx_mJ += energy_tx_mJ

        # Optional MCU active energy during TX
        if (
            getattr(self.relay, "include_mcu_energy", False)
            and getattr(self.relay, "mcu", None) is not None
        ):
            self.relay.energy_tx_mJ += self.relay.mcu.energy_active(toa)

    def _record_transition_energy(self, t_ms, current_mA):
        """
        Record relay radio transition energy.

        Parameters
        ----------
        t_ms : float
            Transition duration in millisecond.
        current_mA : float
            Current during transition in mA.
        """
        if t_ms is None or t_ms <= 0:
            return

        t_s = observed_duration_s(self.env, float(t_ms) / 1000.0)

        # Accumulate radio transition duration.
        self.relay.total_transition_time_s += t_s

        # Radio transition energy in mJ:
        # E = V * I * t
        # V = volt
        # I = ampere
        # t = second
        # result converted to mJ
        e_radio_mJ = (
            self.relay.hw.voltage
            * (float(current_mA) / 1000.0)
            * t_s
            * 1000.0
        )

        self.relay.energy_transition_mJ += e_radio_mJ

        # Optional MCU active energy during transition
        if (
            getattr(self.relay, "include_mcu_energy", False)
            and getattr(self.relay, "mcu", None) is not None
        ):
            self.relay.energy_transition_mJ += self.relay.mcu.energy_active(t_s)



    # --------------------------------------------------------
    # Update relay duty cycle
    # --------------------------------------------------------
    def _update_duty_cycle(self, packet):
        dc = self.relay_duty_cycle
        ch_id = int(getattr(packet, "ch_id", -1))

        if not hasattr(self.relay, "next_tx_allowed_per_channel"):
            self.relay.next_tx_allowed_per_channel = {}

        if dc is None:
            # Unlimited mode: no duty-cycle restriction
            self.relay.next_tx_allowed_per_channel[ch_id] = self.env.now

            # Populate the global timer for backward compatibility and debugging.
            self.relay.next_tx_allowed = self.env.now
            return

        dc = float(dc)

        if dc <= 0 or dc > 1:
            raise ValueError(
                f"{self.relay.id}: invalid relay_duty_cycle={dc}. "
                "Use 0 < dc <= 1 or None."
            )

        off_time = (float(packet.toa) / dc) - float(packet.toa)
        next_allowed = self.env.now + off_time

        # ----------------------------------------------------
        # Per-channel duty-cycle timer
        # ----------------------------------------------------
        self.relay.next_tx_allowed_per_channel[ch_id] = next_allowed

        # ----------------------------------------------------
        # Retain the global timer for compatibility and debugging.
        # For multi-channel operation, conservatively use the maximum
        # of all channel timers.
        # ----------------------------------------------------
        self.relay.next_tx_allowed = max(
            self.relay.next_tx_allowed_per_channel.values()
        )

    # --------------------------------------------------------
    # Half-duplex RX state and safety guard
    # --------------------------------------------------------
    def _entity_type(self, entity_id):
        s = str(entity_id)
        if s == "GW" or s.startswith("GW"):
            return "gateway"
        if s.startswith("R"):
            return "relay"
        return "node"

    def _recompute_rx_busy_until(self):
        """
        Refresh rx_busy_until from RX sessions that are still active.
        """
        now = float(self.env.now)

        active = []
        rx_ends = []

        for pkt in list(
            getattr(self.relay, "active_rx_packets", []) or []
        ):
            if bool(getattr(pkt, "rx_cancelled_by_tx", False)):
                continue

            try:
                rx_start = float(
                    getattr(pkt, "rx_start", 0.0) or 0.0
                )
                rx_end = float(
                    getattr(pkt, "rx_end", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                continue

            if rx_start <= now < rx_end:
                active.append(pkt)
                rx_ends.append(rx_end)

        self.relay.active_rx_packets = active
        self.relay.rx_busy_until = (
            max(rx_ends)
            if rx_ends
            else now
        )

    def _cancel_overlapping_rx_windows(self, tx_start):
        """
        Safeguard for detecting active RX sessions that unexpectedly
        overlap the start of TX.

        In the normal RX-priority flow, this function must always produce
        cancelled_count == 0.
        """
        if getattr(self.relay.hw, "tx_full_duplex", False):
            return 0

        active = list(getattr(self.relay, "active_rx_packets", []) or [])
        if not active:
            return 0

        remaining = []
        cancelled_count = 0
        reason = "half_duplex_rx_interrupted_by_tx"

        for pkt in active:
            if bool(getattr(pkt, "rx_cancelled_by_tx", False)):
                continue

            rx_start = getattr(pkt, "rx_start", None)
            rx_end = getattr(pkt, "rx_end", None)

            try:
                rx_start_f = float(rx_start)
                rx_end_f = float(rx_end)
            except Exception:
                remaining.append(pkt)
                continue

            overlaps = rx_start_f <= float(tx_start) < rx_end_f

            if not overlaps:
                # Remove already-finished RX windows; keep future/current ones.
                if rx_end_f > float(tx_start):
                    remaining.append(pkt)
                continue

            cancelled_count += 1

            pkt.decoded = False
            pkt.collided = False
            pkt.rx_cancelled_by_tx = True
            pkt.rx_cancelled_at = float(tx_start)
            pkt.rx_cancelled_by = self.relay.id
            pkt.drop_time = float(tx_start)
            pkt.reason = reason

            self.stats.record_relay_rx(self.relay.id, "half_duplex_drop_count")
            self.stats.record_relay_rx(
                self.relay.id,
                "rx_interrupted_by_tx_count",
            )
            self.stats.record_node_rx(pkt.src_id, "half_duplex_drop_count")
            self.stats.record_node_rx(
                pkt.src_id,
                "rx_interrupted_by_tx_count",
            )
            self.stats.record_drop(
                packet=pkt,
                reason=reason,
                node_id=self.relay.id,
            )

            tx_entity = self.relay.network_nodes_map.get(
                getattr(pkt, "prev_hop", getattr(pkt, "current_tx", None))
            )
            if tx_entity is not None:
                self.stats.record_margin_sample(
                    entity_id=tx_entity.id,
                    entity_type=self._entity_type(tx_entity.id),
                    margin=getattr(pkt, "margin", None),
                    rssi=getattr(pkt, "rssi", None),
                    snr=getattr(pkt, "snr", None),
                    decoded=False,
                    reason=reason,
                    prev_hop=getattr(pkt, "prev_hop", None),
                    next_hop=getattr(pkt, "next_hop", None),
                    time=float(tx_start),
                )

        self.relay.active_rx_packets = remaining
        self._recompute_rx_busy_until()
        return cancelled_count

    def _refresh_active_rx_state(self):
        """
        Remove completed or cancelled active RX sessions and recalculate
        rx_busy_until.

        Return
        ------
        bool
            True when an active RX session remains.
        """
        now = float(self.env.now)

        active = []
        rx_ends = []

        for pkt in list(
            getattr(self.relay, "active_rx_packets", []) or []
        ):
            if bool(getattr(pkt, "rx_cancelled_by_tx", False)):
                continue

            try:
                rx_start = float(getattr(pkt, "rx_start", 0.0) or 0.0)
                rx_end = float(getattr(pkt, "rx_end", 0.0) or 0.0)
            except Exception:
                continue

            # Retain only RX sessions that are still active.
            if rx_start <= now < rx_end:
                active.append(pkt)
                rx_ends.append(rx_end)

        self.relay.active_rx_packets = active

        if rx_ends:
            self.relay.rx_busy_until = max(rx_ends)
            return True

        self.relay.rx_busy_until = now
        return False


    def _wait_until_rx_idle(self):
        """
        Bounded RX priority with an admission gate.

        Stages:
        1. RelayTX waits for active RX sessions to finish.
        2. Before max_rx_idle_wait_ms is reached,
           new RX packets remain admissible.
        3. Once the limit is reached, tx_pending=True.
        4. RelayRX rejects new arrivals.
        5. Already-active RX sessions are allowed to finish.
        6. TX starts after active_rx_packets is empty.
        """
        if bool(
            getattr(
                self.relay.hw,
                "tx_full_duplex",
                False,
            )
        ):
            return

        poll_s = max(
            float(self.rx_idle_poll_interval_ms) / 1000.0,
            1e-9,
        )

        max_wait_s = max(
            float(self.max_rx_idle_wait_ms) / 1000.0,
            0.0,
        )

        wait_start = float(self.env.now)

        while True:
            has_active_rx = self._refresh_active_rx_state()

            if not has_active_rx:
                return

            waited_s = (
                float(self.env.now) - wait_start
            )

            # Once the waiting limit is reached,
            # close admission to new RX sessions.
            if waited_s >= max_wait_s:
                self.relay.tx_pending = True

            now = float(self.env.now)

            rx_busy_until = float(
                getattr(
                    self.relay,
                    "rx_busy_until",
                    now,
                )
            )

            remaining_s = max(
                0.0,
                rx_busy_until - now,
            )

            if remaining_s <= 1e-9:
                yield self.env.timeout(1e-9)
                continue

            yield self.env.timeout(
                min(
                    remaining_s,
                    poll_s,
                )
            )

    # --------------------------------------------------------
    # Direct TX process
    # --------------------------------------------------------
    def _direct_tx(self, packet):
        # ----------------------------------------------------
        # STEP 1: DUTY-CYCLE WAITING
        # ----------------------------------------------------
        ch_id = int(
            getattr(
                packet,
                "ch_id",
                -1,
            )
        )

        if not hasattr(
            self.relay,
            "next_tx_allowed_per_channel",
        ):
            self.relay.next_tx_allowed_per_channel = {}

        next_tx_allowed = (
            self.relay.next_tx_allowed_per_channel.get(
                ch_id,
                0.0,
            )
        )

        if float(self.env.now) < float(next_tx_allowed):
            yield self.env.timeout(
                float(next_tx_allowed)
                - float(self.env.now)
            )

        packet.tx_duty_ready_time = float(self.env.now)

        # ----------------------------------------------------
        # STEP 2: BOUNDED RX-PRIORITY
        # ----------------------------------------------------
        yield from self._wait_until_rx_idle()
        packet.tx_rx_idle_time = float(self.env.now)

        # >>> PATCH:
        # Close RX admission immediately after RX becomes idle.
        # There is no yield between returning from the wait and this assignment.
        if not bool(
            getattr(
                self.relay.hw,
                "tx_full_duplex",
                False,
            )
        ):
            self.relay.tx_pending = True

        # ----------------------------------------------------
        # STEP 3: RX -> TX TRANSITION
        # ----------------------------------------------------
        t_ms, i_trans, i_tx_final = (
            self.relay.hw.rx_to_tx(
                packet.ptx_dbm
            )
        )

        transition_s = max(
            float(t_ms) / 1000.0,
            0.0,
        )

        # The radio cannot receive from this point onward.
        self.relay.is_switching_to_tx = True

        self.relay.tx_transition_until = (
            float(self.env.now)
            + transition_s
        )

        self._record_transition_energy(
            t_ms,
            i_trans,
        )

        if transition_s > 0:
            yield self.env.timeout(
                transition_s
            )

        # ----------------------------------------------------
        # STEP 4: ENTER TX STATE WITHOUT A GAP
        # ----------------------------------------------------
        self.relay.tx_power = packet.ptx_dbm
        self.relay.tx_current_mA = float(
            i_tx_final
        )
        self.relay.freq = packet.freq_mhz

        self.relay.tx_active_count = (
            int(
                getattr(
                    self.relay,
                    "tx_active_count",
                    0,
                )
            )
            + 1
        )

        self.relay.is_transmitting = True

        packet.tx_start = float(self.env.now)
        packet.tx_end = (
            packet.tx_start
            + float(packet.toa)
        )

        self.relay.tx_busy_until = max(
            float(
                getattr(
                    self.relay,
                    "tx_busy_until",
                    0.0,
                )
            ),
            float(packet.tx_end),
        )

        # is_transmitting is already True,
        # The transition flag can now be cleared safely.
        self.relay.is_switching_to_tx = False
        self.relay.tx_transition_until = float(
            self.env.now
        )

        # ----------------------------------------------------
        # STEP 5: HALF-DUPLEX SAFETY GUARD
        # ----------------------------------------------------
        # In the RX-priority flow, active RX sessions should never be
        # cancelled. This function remains only to
        # detect race conditions.
        cancelled_count = self._cancel_overlapping_rx_windows(
            packet.tx_start
        )

        if cancelled_count > 0:
            print(
                f"[HALF_DUPLEX_WARNING] "
                f"relay={self.relay.id} "
                f"tx_start={packet.tx_start:.9f} "
                f"cancelled_rx={cancelled_count}"
            )

        # ----------------------------------------------------
        # STEP 6: AIRTIME AND ENERGY ACCOUNTING
        # ----------------------------------------------------
        self._record_airtime(packet)
        self._record_energy(packet)

        # ----------------------------------------------------
        # STEP 7: RECORD TX EVENT
        # ----------------------------------------------------
        self.stats.record_tx(
            packet,
            node_id=self.relay.id,
        )

        # ----------------------------------------------------
        # STEP 8: CLEAR PREVIOUS RX METADATA
        # ----------------------------------------------------
        try:
            if hasattr(packet, "reset_rx_runtime_metadata"):
                packet.reset_rx_runtime_metadata(
                    preserve_t_arrive=False
                )
        except Exception:
            pass

        # ----------------------------------------------------
        # STEP 9: PROPAGATION TIMING
        # ----------------------------------------------------
        rx_entity = None

        try:
            rx_entity = self.relay.network_nodes_map.get(
                getattr(packet, "next_hop", None)
            )
        except Exception:
            rx_entity = None

        if rx_entity is not None:
            try:
                d_m = float(
                    self.relay.distance_to(rx_entity)
                )
                prop_s = d_m / 3e8
            except Exception:
                d_m = None
                prop_s = 0.0
        else:
            d_m = None
            prop_s = 0.0

        packet.distance_m = d_m
        packet.t_arrive = (
            float(packet.tx_start) + float(prop_s)
        )
        packet.t_end = (
            float(packet.t_arrive) + float(packet.toa)
        )

        self.air.transmit(packet)

        # ----------------------------------------------------
        # STEP 10: TX AIRTIME
        # ----------------------------------------------------
        yield self.env.timeout(float(packet.toa))



        # ----------------------------------------------------
        # STEP 11: TX -> RX TRANSITION
        # ----------------------------------------------------
        t_ms, i_trans, i_rx_final = (
            self.relay.hw.tx_to_rx()
        )

        tx_to_rx_transition_s = max(
            float(t_ms) / 1000.0,
            0.0,
        )

        # Keep the radio TX-busy during
        # TX -> RX transition.
        self.relay.tx_busy_until = max(
            float(
                getattr(
                    self.relay,
                    "tx_busy_until",
                    0.0,
                )
            ),
            float(self.env.now)
            + tx_to_rx_transition_s,
        )

        self._record_transition_energy(
            t_ms,
            i_trans,
        )

        if tx_to_rx_transition_s > 0:
            yield self.env.timeout(
                tx_to_rx_transition_s
            )

        # ----------------------------------------------------
        # STEP 12: RADIO IS NOW RX-READY
        # ----------------------------------------------------
        self.relay.tx_power = None
        self.relay.tx_current_mA = None
        self.relay.rx_current_mA = float(
            i_rx_final
        )

        self.relay.tx_active_count = max(
            0,
            int(
                getattr(
                    self.relay,
                    "tx_active_count",
                    0,
                )
            )
            - 1,
        )

        self.relay.is_transmitting = (
            self.relay.tx_active_count > 0
        )

        if self.relay.tx_active_count == 0:
            self.relay.tx_busy_until = float(
                self.env.now
            )

        # Defensive cleanup transition.
        self.relay.is_switching_to_tx = False
        self.relay.tx_transition_until = float(
            self.env.now
        )

        # >>> PATCH:
        # Reopen RX admission only after the radio
        # has fully returned to the RX-ready state.
        self.relay.tx_pending = False

        if (
            self.tx_capacity_c == 1
            and self.relay.tx_active_count != 0
        ):
            raise RuntimeError(
                f"{self.relay.id}: "
                f"invalid tx_active_count="
                f"{self.relay.tx_active_count} "
                f"after TX completion"
            )

        # ----------------------------------------------------
        # STEP 13: UPDATE DUTY CYCLE
        # ----------------------------------------------------
        self._update_duty_cycle(packet)



    # --------------------------------------------------------
    # Main TX loop
    # --------------------------------------------------------
    def run(self, worker_id=0):
        """
        Relay TX worker.

        M/G/c interpretation:
        - One RelayTX object owns one shared queue.
        - main_simulation starts c workers, where c = tx_capacity_c.
        - Each worker repeatedly takes one packet from the queue and transmits it.
        """

        while True:
            # ----------------------------------------------------
            # 1. Take packet from shared queue
            # ----------------------------------------------------
            packet = yield self.tx_waiting_queue.get()
            packet.tx_queue_dequeue_time = float(self.env.now)
            packet.tx_worker_id = int(worker_id)
            packet.tx_queue_status = "IN_SERVICE"
            self.in_service_packets[int(worker_id)] = packet
            self.stats.record_packet_fate_state(
                packet,
                state="QUEUE_IN_SERVICE",
                entity_id=self.relay.id,
                time=self.env.now,
            )
            self._record_queue_state(event="dequeue")

            # ----------------------------------------------------
            # 2. Record waiting time in TX queue
            # ----------------------------------------------------
            waiting_time_wq_s = (
                float(self.env.now)
                - float(packet.tx_queue_arrival_time)
            )
            self.stats.record_tx_queue_waiting_time(
                self.relay.id,
                waiting_time_wq_s,
            )

            # ----------------------------------------------------
            # 3. Safety check
            # ----------------------------------------------------
            if getattr(packet, "next_hop", None) is None:
                packet.tx_service_end_time = float(self.env.now)
                self.stats.record_relay_rx(
                    self.relay.id,
                    "tx_queue_internal_drop_count",
                )
                packet.drop_time = float(self.env.now)
                packet.reason = "forwarding_invalid_next_hop"
                self.stats.record_drop(
                    packet,
                    reason=packet.reason,
                    node_id=self.relay.id,
                )
                self._record_queue_lifecycle(
                    packet,
                    status="INVALID_NEXT_HOP",
                    observation_end_time=self.env.now,
                )
                self.in_service_packets.pop(int(worker_id), None)
                self._record_queue_state(event="invalid_next_hop")
                continue

            # ----------------------------------------------------
            # 4. Prepare forwarding packet
            # ----------------------------------------------------
            packet = self._prepare_forward_packet(packet)

            # ----------------------------------------------------
            # 5. Transmit packet
            # ----------------------------------------------------
            yield from self._direct_tx(packet)

            packet.tx_service_end_time = float(self.env.now)
            self._record_queue_lifecycle(packet, status="COMPLETED")
            self.in_service_packets.pop(int(worker_id), None)
            self._record_queue_state(event="service_end")

            # ----------------------------------------------------
            # 6. Record TX done
            # ----------------------------------------------------
            self.stats.record_relay_rx(self.relay.id, "tx_done")
