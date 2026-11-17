# ============================================================
# runtime/relay_rx.py
# Physical DATA_DOWN / DATA_DOWN_FWD + 
# Relay APC RX cleanup, APC TX index, and timing validation
#_build_apc_info
# - Demodulator admission at physical RX start; release at physical RX end
# - Adaptive Ptx feedback is applied HERE (RelayRX side)
# - Added hooks for Q1 matrices:
#   * PTX history
#   * margin history
#   * delivered packet matrix
# - Supports HALF-DUPLEX and FULL-DUPLEX behavior
# ============================================================

from collections import defaultdict
from loraphymac.runtime.packet import Packet
from loraphymac.mac.frame_defs import MType, FPort, encode_fctrl, decode_fctrl,frame_size_bytes
from loraphymac.runtime.apc_power import (
    dbm_to_tx_power_index,
    tx_power_index_to_dbm,
    get_max_tx_power_index,
)

from loraphymac.runtime.apc_commands import (
    encode_apc_req,
    decode_apc_req,
    APC_STATUS_DR_OK,
    APC_STATUS_PTX_OK,
    APC_STATUS_CH_OK,
    APC_STATUS_NBTRANS_OK,
    APC_STATUS_APPLIED,
)
from loraphymac.runtime.apc_controller_state import (
    APCControllerLinkState,
    DEFAULT_APC_MAX_RETRIES,
)
from loraphymac.runtime.wang_snr_reporting import (
    WangSNRReportCapacityError,
    WangSNRReportError,
    accept_at_gateway,
    append_report,
    decode_reports,
    report_from_successful_rx,
    reporting_enabled,
)
from loraphymac.runtime.wang_power_commands import is_wang_power_command
from loraphymac.runtime.receiver_collision_window import (
    register_rf_window_packet,
    wait_until_window_quiet,
)


class RelayRX:
    # Relay/GW receivers observe all feasible LoRa bursts in their RF domain,
    # including packets whose logical next_hop is another receiver.
    accepts_non_target_rf = True

    def __init__(
        self,
        env,
        relay,
        air_interface,
        model,
        terrain_delta_h,
        collision_engine,
        demodulator_pool,
        relay_tx,
        statistics,
    ):
        self.env = env
        self.relay = relay
        self.air = air_interface
        self.model = model
        self.terrain_delta_h = terrain_delta_h
        self.collision = collision_engine
        self.demod = demodulator_pool
        self.stats = statistics
        self.relay_tx = relay_tx

        # Controller-owned causal belief. Keys identify one directed
        # transmitter -> next-hop-controller link. No entry points back to a
        # mutable remote transmitter state object.
        self.apc_controller_link_states = {}

        # ----------------------------------------------------
        # RX DETECTOR GUARD (8 CHANNEL, ALWAYS-ON)
        # ----------------------------------------------------
        self.rx_detector_guard = set(range(8))

        # ----------------------------------------------------
        # Dynamic receiver RF windows. Logical channels with overlapping
        # spectra can merge, while disjoint spectra keep independent timing.
        # ----------------------------------------------------
        self.demod_windows = defaultdict(list)
        self.demod_leader_busy = set()
        self.demod_window_sequence = 0

        # ----------------------------------------------------
        # Full-duplex / half-duplex runtime state
        # ----------------------------------------------------
        # RelayTX sets is_transmitting and tx_busy_until.
        # RelayRX uses these fields to drop packets during half-duplex TX.
        #
        # RelayRX updates rx_busy_until after calculating rx_end.
        # RelayTX uses rx_busy_until and active_rx_packets
        # to wait until every active RX session finishes before
        # starting an RX-to-TX transition in half-duplex mode.
        # ----------------------------------------------------
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
            
        if not hasattr(self.relay, "tx_pending"):
            self.relay.tx_pending = False
        
        # Active RX registry shared with RelayTX.
        #
        # On half-duplex hardware, RelayTX must wait until this registry
        # is empty before starting an RX-to-TX transition.
        #
        # The cancellation flag remains only as a safeguard
        # against unexpected race conditions.
        if not hasattr(self.relay, "active_rx_packets"):
            self.relay.active_rx_packets = []

        #print(f"[INIT] RelayRX {relay.id} Ready")

    # ====================================================
    # PUSH-BASED ARRIVAL
    # ====================================================
    def on_arrival(self, packet):
        self.env.process(self._rx_process(packet))

    # ====================================================
    # Helper
    # ====================================================
    def _entity_type(self, entity_id):
        s = str(entity_id)
        if s == "GW" or s.startswith("GW"):
            return "gateway"
        if s.startswith("R"):
            return "relay"
        return "node"

    def _apc_controller_key(self, transmitter_id):
        return (str(transmitter_id), str(self.relay.id))

    def _get_apc_controller_link_state(self, tx_entity):
        """Return local belief initialized only from provisioned constants."""
        key = self._apc_controller_key(tx_entity.id)
        state = self.apc_controller_link_states.get(key)
        if state is not None:
            return state

        # ``fixed_ptx`` is the immutable startup selection established before
        # traffic. Unlike ``default_ptx``/``apc_tx_power_index``, R-APC never
        # mutates it, so it is legitimate pre-provisioned controller knowledge.
        provisioned_ptx = getattr(tx_entity, "fixed_ptx", None)
        if provisioned_ptx is None:
            provisioned_ptx = getattr(tx_entity, "ptx_max")
        provisioned_ptx = int(provisioned_ptx)
        initial_index = dbm_to_tx_power_index(tx_entity, provisioned_ptx)
        state = APCControllerLinkState.from_preprovisioned_state(
            controller_id=self.relay.id,
            transmitter_id=tx_entity.id,
            initial_tx_power_index=initial_index,
            max_tx_power_index=get_max_tx_power_index(tx_entity),
            target_margin_db=float(getattr(tx_entity, "target_margin_db", 6.0)),
        )
        self.apc_controller_link_states[key] = state
        return state

    def _handle_wang_snr_success(self, packet):
        """Generate/forward only causal SNR evidence; never make a decision."""
        if not reporting_enabled(self.relay):
            return
        if getattr(packet, "mtype", None) not in (
            None,
            MType.DATA_UP,
            MType.DATA_UP_FWD,
        ):
            return

        routing_map = getattr(self.relay, "wang_routing_map", None)
        if routing_map is None:
            self.stats.record_wang_snr_report_event(
                event="invalid",
                time=self.env.now,
                packet=packet,
                receiver_id=self.relay.id,
                reason="immutable Wang routing map is unavailable",
            )
            return
        # Metrics may later decode carried report bytes during physical TX/drop
        # logging. Give that observer the same immutable, pre-provisioned map.
        self.stats.wang_routing_map = routing_map

        try:
            local_report = report_from_successful_rx(
                packet,
                receiver_id=self.relay.id,
                observation_time_s=float(self.env.now),
                routing_map=routing_map,
            )
        except WangSNRReportError as exc:
            # A failed/collided reception is not a report loss: no report was
            # generated.  Invalid successful-reception metadata is observable.
            if bool(getattr(packet, "decoded", False)):
                self.stats.record_wang_snr_report_event(
                    event="invalid",
                    time=self.env.now,
                    packet=packet,
                    receiver_id=self.relay.id,
                    reason=str(exc),
                )
            return

        self.stats.record_wang_snr_report_event(
            event="generated",
            time=self.env.now,
            packet=packet,
            report=local_report,
            receiver_id=self.relay.id,
        )

        if self.relay.id != "GW":
            try:
                append_report(
                    packet,
                    local_report,
                    routing_map=routing_map,
                )
            except WangSNRReportCapacityError as exc:
                self.stats.record_wang_snr_report_event(
                    event="overflow",
                    time=self.env.now,
                    packet=packet,
                    report=local_report,
                    receiver_id=self.relay.id,
                    reason=str(exc),
                )
            except WangSNRReportError as exc:
                self.stats.record_wang_snr_report_event(
                    event="invalid",
                    time=self.env.now,
                    packet=packet,
                    report=local_report,
                    receiver_id=self.relay.id,
                    reason=str(exc),
                )
            return

        try:
            carried_reports = decode_reports(
                packet,
                routing_map=routing_map,
                receiver_id=self.relay.id,
            )
        except WangSNRReportError as exc:
            carried_reports = []
            self.stats.record_wang_snr_report_event(
                event="invalid",
                time=self.env.now,
                packet=packet,
                receiver_id=self.relay.id,
                reason=str(exc),
            )

        for report in [*carried_reports, local_report]:
            received = accept_at_gateway(
                self.relay,
                report,
                gateway_arrival_time_s=float(self.env.now),
            )
            accepted = received is not None
            decision_engine = getattr(
                self.relay,
                "wang_sink_decision_engine",
                None,
            )
            if accepted and decision_engine is not None:
                decision_engine.accept_observation(received)
            self.stats.record_wang_snr_report_event(
                event="received" if accepted else "duplicate",
                time=self.env.now,
                packet=packet,
                report=report,
                receiver_id=self.relay.id,
            )

    # ====================================================
    # Half-duplex active RX registry
    # ====================================================
    def _recompute_rx_busy_until(self):
        now = float(self.env.now)
        active = []
        rx_ends = []
    
        for pkt in list(
            getattr(self.relay, "active_rx_packets", []) or []
        ):
            try:
                rx_start = float(
                    getattr(pkt, "rx_start", 0.0) or 0.0
                )
                rx_end = float(
                    getattr(pkt, "rx_end", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                continue
    
            if bool(getattr(pkt, "rx_cancelled_by_tx", False)):
                continue
    
            if not (rx_start <= now < rx_end):
                continue
    
            active.append(pkt)
            rx_ends.append(rx_end)
    
        self.relay.active_rx_packets = active
        self.relay.rx_busy_until = (
            max(rx_ends)
            if rx_ends
            else now
        )

    def _register_active_rx(self, packet):
        """
        Register a feasible packet that is inside its RX interval.
    
        RelayTX uses active_rx_packets and rx_busy_until to
        wait for every active RX session to finish before starting
        an RX-to-TX transition on half-duplex hardware.
        """
        if getattr(self.relay.hw, "tx_full_duplex", False):
            return

        if not hasattr(self.relay, "active_rx_packets"):
            self.relay.active_rx_packets = []

        # Reset cancellation markers for this RX attempt.
        packet.rx_cancelled_by_tx = False
        packet.rx_cancelled_at = None
        packet.rx_cancelled_by = None

        # Avoid duplicate registration of the same packet object.
        if not any(id(p) == id(packet) for p in self.relay.active_rx_packets):
            self.relay.active_rx_packets.append(packet)

        self._recompute_rx_busy_until()

    def _unregister_active_rx(self, packet):
        """
        Remove a packet from the active RX registry after its RF window ends
        or after it has been cancelled.
        """
        active = getattr(self.relay, "active_rx_packets", [])
        self.relay.active_rx_packets = [
            p for p in active if id(p) != id(packet)
        ]
        self._recompute_rx_busy_until()

    def _is_rx_cancelled(self, packet):
        return bool(getattr(packet, "rx_cancelled_by_tx", False))


    def _build_apc_info(
        self,
        tx_entity,
        packet,
        old_tx_index,
        new_tx_index,
    ):
        """
        Build the APC command context for physical DATA_DOWN.

        Clean APC contract:
        - Does not carry old_ptx_dbm/new_ptx_dbm.
        - Does not derive tx_power_index from dBm.
        - The APC_REQ frame carries only TxPowerIndex.
        """
        if tx_entity is None:
            return None

        controller_id = self.relay.id
        controller_type = self._entity_type(controller_id)

        target_id = tx_entity.id
        target_type = self._entity_type(target_id)

        if target_type == "node":
            control_scope = "node_apc"
        elif target_type == "relay":
            control_scope = "relay_apc"
        else:
            control_scope = "unknown"

        try:
            old_tx_index = int(old_tx_index)
            new_tx_index = int(new_tx_index)
        except Exception:
            return None

        dr = None
        ch_mask = 0xFF
        nb_trans = 0x01

        status = (
            APC_STATUS_DR_OK
            | APC_STATUS_PTX_OK
            | APC_STATUS_CH_OK
            | APC_STATUS_NBTRANS_OK
            | APC_STATUS_APPLIED
        )

        context = {
            "controller_id": controller_id,
            "controller_type": controller_type,
            "target_id": target_id,
            "target_type": target_type,
            "control_scope": control_scope,

            # APC index-domain state.
            "old_tx_power_index": old_tx_index,
            "tx_power_index": new_tx_index,

            "dr": dr,
            "ch_mask": ch_mask,
            "nb_trans": nb_trans,
        }

        return {
            "context": context,
            "status": status,

            # Original uplink context.
            "src_id": getattr(packet, "src_id", None),
            "hop_index": getattr(packet, "hop_index", None),

            # Snapshot the original uplink PHY state when making the APC decision.
            "uplink_ch_id": getattr(packet, "ch_id", None),
            "uplink_freq_mhz": getattr(packet, "freq_mhz", None),
            "uplink_sf": getattr(packet, "sf", None),
            "uplink_bw_hz": getattr(packet, "bw_hz", None),

            # Decision / uplink metric.
            "decision_margin": getattr(packet, "margin", None),
            "decision_rssi": getattr(packet, "rssi", None),
            "decision_snr": getattr(packet, "snr", None),
            "decision_prev_hop": getattr(packet, "prev_hop", None),
            "decision_next_hop": getattr(packet, "next_hop", None),
            "decision_mtype": getattr(packet, "mtype", None),
            "decision_time": self.env.now,
        }

    def _make_apc_downlink_packet(
        self,
        tx_entity,
        packet,
        apc_info,
    ):
        """
        Create a physical APC packet:
        - DATA_DOWN for a target node
        - DATA_DOWN_FWD for a target relay

        Clean APC contract:
        - APC_REQ carries only TxPowerIndex.
        - Does not carry old_ptx_dbm/new_ptx_dbm.
        - packet.ptx_dbm is the physical relay/gateway PTx used for downlink.
        """
        controller_id = self.relay.id
        target_id = str(tx_entity.id)

        # ----------------------------------------------------
        # Select MType based on target entity
        # ----------------------------------------------------
        if target_id.startswith("R"):
            apc_mtype = MType.DATA_DOWN_FWD
        else:
            apc_mtype = MType.DATA_DOWN

        # ----------------------------------------------------
        # Build APC_REQ FOpts from apc_info.
        # ----------------------------------------------------
        try:
            tx_power_index = int(apc_info["context"]["tx_power_index"])
            dr = apc_info["context"].get("dr", None)
            ch_mask = int(apc_info["context"].get("ch_mask", 0xFF))
            nb_trans = int(apc_info["context"].get("nb_trans", 0x01))

            fopts_req = encode_apc_req(
                dr=0 if dr is None else int(dr),
                tx_power_index=tx_power_index,
                ch_mask=ch_mask,
                nb_trans=nb_trans,
            )
        except Exception as e:
            raise ValueError(
                f"{self.relay.id}: failed to build APC_REQ FOpts "
                f"for target={target_id}: {e}"
            )

        # ----------------------------------------------------
        # Compute DATA_DOWN / DATA_DOWN_FWD frame size
        # ----------------------------------------------------
        
        # APC command is carried entirely in FOpts.
        # DATA_DOWN carries no application payload.
        app_payload_size = 0
        
        actual_fopts_len = len(fopts_req)
        
        configured_fopts_len = int(
            getattr(
                self.relay,
                "apc_fopts_size",
                actual_fopts_len,
            )
        )
        
        if configured_fopts_len != actual_fopts_len:
            raise ValueError(
                f"{self.relay.id}: APC FOpts size mismatch: "
                f"configured={configured_fopts_len}, "
                f"encoded={actual_fopts_len}"
            )
        
        physical_payload_size = frame_size_bytes(
            app_payload_size=app_payload_size,
            fopts_len=actual_fopts_len,
        )

        # Path satu hop: controller -> target
        path = [controller_id, target_id]

        # ----------------------------------------------------
        # Snapshot the original uplink channel when making the APC decision.
        # Node-bound DATA_DOWN follows the node uplink channel.
        # ----------------------------------------------------
        try:
            ch_id = int(apc_info.get("uplink_ch_id"))
        except Exception:
            try:
                ch_id = int(packet.ch_id)
            except Exception:
                ch_id = 0

        freq_mhz = apc_info.get("uplink_freq_mhz", None)
        if freq_mhz is None:
            freq_mhz = getattr(packet, "freq_mhz", None)

        if freq_mhz is None:
            try:
                freq_mhz = float(self.relay.ch_plan.get_freq(ch_id))
            except Exception:
                freq_mhz = 0.0

        sf = int(apc_info.get("uplink_sf") or getattr(packet, "sf", 7))
        bw_hz = int(
            apc_info.get("uplink_bw_hz")
            or getattr(packet, "bw_hz", 125000)
        )

        # ----------------------------------------------------
        # This is the physical relay/gateway PTx used to send DATA_DOWN.
        # This is not the target node/relay PTx.
        # RelayTX can still override or validate through
        # _select_ptx_for_downlink_control().
        # ----------------------------------------------------
        ptx_dbm = int(
            getattr(
                self.relay,
                "default_ptx",
                getattr(self.relay, "fixed_ptx", 20),
            )
        )

        toa = self.air.compute_toa(
            sf=sf,
            bw_hz=bw_hz,
            payload_bytes=physical_payload_size,
        )

        down_pkt = Packet(
            # ------------------------------------------------
            # Identity & routing
            # ------------------------------------------------
            src_id=controller_id,
            current_tx=controller_id,
            prev_hop=controller_id,
            next_hop=target_id,
            path=path,
            hop_index=0,

            # ------------------------------------------------
            # PHY
            # ------------------------------------------------
            ch_id=ch_id,
            freq_mhz=freq_mhz,
            sf=sf,
            bw_hz=bw_hz,
            ptx_dbm=ptx_dbm,
            # Total DATA_DOWN / DATA_DOWN_FWD frame size.
            payload_size=physical_payload_size,

            

            # ------------------------------------------------
            # Timing
            # ------------------------------------------------
            toa=toa,
            tx_start=self.env.now,
            tx_end=self.env.now + toa,

            # DATA_DOWN APC does not carry application data.
            app_payload_size=app_payload_size,

            # ------------------------------------------------
            # Frame
            # ------------------------------------------------
            mtype=apc_mtype,
            version=getattr(packet, "version", 0),
            rfu=0,
            dst_id=target_id,
            hop_limit=1,
            fctrl=encode_fctrl(
                apc=True,
                ack_req=False,
                bcast=False,
                pending=False,
                fopts_len=len(fopts_req),
            ),
            fcnt=int(getattr(self.relay, "fcnt", 0)),
            fopts=fopts_req,
            fport=FPort.MAC_CONTROL,
            mic_crc=None,

            # ------------------------------------------------
            # Runtime metadata
            # ------------------------------------------------
            tx_mode="APC_DOWN",
            birth_time=self.env.now,
        )

        self.relay.fcnt = int(getattr(self.relay, "fcnt", 0)) + 1

        # ----------------------------------------------------
        # APC metadata used when the target receives the packet.
        # Does not carry old_ptx_dbm/new_ptx_dbm.
        # ----------------------------------------------------
        down_pkt.apc_info = apc_info

        down_pkt.apc_uplink_src_id = getattr(packet, "src_id", None)
        down_pkt.apc_uplink_hop_index = getattr(packet, "hop_index", None)
        down_pkt.apc_uplink_ch_id = apc_info.get("uplink_ch_id", None)
        down_pkt.apc_uplink_freq_mhz = apc_info.get("uplink_freq_mhz", None)
        down_pkt.apc_uplink_sf = apc_info.get("uplink_sf", None)
        down_pkt.apc_uplink_bw_hz = apc_info.get("uplink_bw_hz", None)

        down_pkt.downlink_channel_policy = (
            "follow_original_uplink_ch"
            if apc_mtype == MType.DATA_DOWN
            else "relay_tx_ch_ids"
        )

        # Decision/uplink metric carried as metadata.
        down_pkt.apc_decision_margin = apc_info.get("decision_margin", None)
        down_pkt.apc_decision_rssi = apc_info.get("decision_rssi", None)
        down_pkt.apc_decision_snr = apc_info.get("decision_snr", None)
        down_pkt.apc_decision_prev_hop = apc_info.get("decision_prev_hop", None)
        down_pkt.apc_decision_next_hop = apc_info.get("decision_next_hop", None)
        down_pkt.apc_decision_mtype = apc_info.get("decision_mtype", None)
        down_pkt.apc_decision_time = apc_info.get("decision_time", None)

        return down_pkt

    
    def _send_physical_apc_downlink(
        self,
        tx_entity,
        packet,
        apc_info,
    ):
        """
        Send physical APC DATA_DOWN through the RelayTX queue.
        Applies to both nodes and relays.

        The pending-state guard uses TxPowerIndex, not PTx in dBm.
        """
        downlink_delay_s = float(
            getattr(self.relay, "apc_downlink_delay_s", 1.0)
        )

        yield self.env.timeout(downlink_delay_s)

        # ------------------------------------------------------------
        # Guard using controller-owned pending state only. Target-side apply
        # state is intentionally invisible without a physical answer.
        # ------------------------------------------------------------
        try:
            target_tx_index = int(apc_info["context"]["tx_power_index"])
        except Exception:
            return

        feedback_mtype = (
            MType.DATA_DOWN_FWD
            if self._entity_type(tx_entity.id) == "relay"
            else MType.DATA_DOWN
        )

        state = self.apc_controller_link_states.get(
            self._apc_controller_key(tx_entity.id)
        )
        pending = None if state is None else state.pending_command
        feedback_id = apc_info.get("feedback_id")

        if (
            pending is None
            or pending.target_tx_power_index != target_tx_index
            or pending.feedback_id != feedback_id
        ):
            self.stats.record_apc_feedback_event(
                event="dropped",
                time=self.env.now,
                apc_info=apc_info,
                feedback_id=feedback_id,
                mtype=feedback_mtype,
                stale_tpi=True,
                reason="controller_pending_command_changed_before_send",
            )
            return

        down_pkt = self._make_apc_downlink_packet(
            tx_entity=tx_entity,
            packet=packet,
            apc_info=apc_info,
        )

        self.relay_tx.put(down_pkt)
    

    # ====================================================
    # Adaptive Ptx feedback
    # ====================================================
    def _apply_adaptive_ptx_feedback(self, tx_entity, packet):
        """
        Build adaptive PTx feedback from the RSSI margin.

        Clean APC contract:
        - The controller uses TxPowerIndex.
        - The APC_REQ frame carries TxPowerIndex.
        - The APC command does not carry ptx_dbm.
        - PHY, AirInterface, energy, and command application still use ptx_dbm.
        """
        if tx_entity is None:
            return

        # ============================================================
        # Frame-level APC capability/request guard
        # ============================================================
        if getattr(packet, "mtype", None) in (MType.DATA_UP, MType.DATA_UP_FWD):
            fctrl_info = decode_fctrl(getattr(packet, "fctrl", 0))
            packet_apc_enabled = bool(fctrl_info.get("apc", False))

            if not packet_apc_enabled:
                return

        # ============================================================
        # Internal entity policy guard
        # ============================================================
        if not hasattr(tx_entity, "ptx_mode"):
            return

        mode = str(getattr(tx_entity, "ptx_mode", "fixed")).lower().strip()

        if mode == "fixed":
            return

        if mode != "adaptive":
            return

        # ------------------------------------------------------------
        # Margin guard
        # ------------------------------------------------------------
        margin = getattr(packet, "margin", None)
        if margin is None:
            return

        # ------------------------------------------------------------
        # Physical APC only
        # ------------------------------------------------------------
        apc_downlink_mode = str(
            getattr(self.relay, "apc_downlink_mode", "physical_all")
        ).lower().strip()

        if apc_downlink_mode != "physical_all":
            return

        # ============================================================
        # Controller-owned causal state
        # ============================================================
        try:
            state = self._get_apc_controller_link_state(tx_entity)
        except Exception:
            return

        apc_retry_timeout_s = float(
            getattr(self.relay, "apc_retry_timeout_s", 5.0)
        )
        max_retries = int(
            getattr(self.relay, "apc_max_retries", DEFAULT_APC_MAX_RETRIES)
        )
        feedback_mtype = (
            MType.DATA_DOWN_FWD
            if self._entity_type(tx_entity.id) == "relay"
            else MType.DATA_DOWN
        )

        # With no physical APC_ANS, an outstanding command is never confirmed.
        # Before its timeout, do nothing. After timeout, retry only the same
        # absolute target and stop after a bounded number of retries.
        if state.pending_command is not None:
            retry_status = state.retry_status(
                now_s=self.env.now,
                retry_timeout_s=apc_retry_timeout_s,
                max_retries=max_retries,
            )
            if retry_status == "waiting":
                return

            pending = state.pending_command
            apc_info = self._build_apc_info(
                tx_entity=tx_entity,
                packet=packet,
                old_tx_index=pending.source_estimated_tx_power_index,
                new_tx_index=pending.target_tx_power_index,
            )
            if apc_info is None:
                return
            apc_info["feedback_id"] = pending.feedback_id
            apc_info["retry_timeout_s"] = apc_retry_timeout_s
            apc_info["context"].update(
                {
                    "confirmed_tx_power_index": state.confirmed_tx_power_index,
                    "estimated_tx_power_index": state.estimated_tx_power_index,
                    "estimate_provenance": state.estimate_provenance,
                    "retry_count": pending.retry_count,
                    "max_retries": max_retries,
                }
            )

            if retry_status == "exhausted":
                state.mark_retry_exhausted()
                self.stats.record_apc_feedback_event(
                    event="retry_exhausted",
                    time=self.env.now,
                    apc_info=apc_info,
                    feedback_id=pending.feedback_id,
                    mtype=feedback_mtype,
                    reason="no_physical_apc_answer_after_bounded_retries",
                )
                return

            pending = state.mark_retry(now_s=self.env.now)
            apc_info["context"]["retry_count"] = pending.retry_count
            self.stats.record_apc_feedback_event(
                event="retry_generated",
                time=self.env.now,
                apc_info=apc_info,
                feedback_id=pending.feedback_id,
                mtype=feedback_mtype,
                reason="controller_local_timeout_without_confirmation",
            )
            self.env.process(
                self._send_physical_apc_downlink(
                    tx_entity=tx_entity,
                    packet=packet,
                    apc_info=apc_info,
                )
            )
            return

        # ============================================================
        # APC decision in TxPowerIndex domain using explicit estimate
        # ============================================================
        current_tx_index = int(state.estimated_tx_power_index)
        apc_margin = float(margin) - float(state.target_margin_db)

        # Positive n_step  -> margin berlebih -> index naik -> PTx turun
        # Negative n_step  -> margin kurang   -> index turun -> PTx naik
        n_step = int(apc_margin / 2.0)

        new_tx_index = current_tx_index + n_step
        new_tx_index = max(0, min(state.max_tx_power_index, int(new_tx_index)))

        if new_tx_index == current_tx_index:
            return

        # ============================================================
        # Build the command context for physical DATA_DOWN.
        # ============================================================
        apc_info = self._build_apc_info(
            tx_entity=tx_entity,
            packet=packet,
            old_tx_index=current_tx_index,
            new_tx_index=new_tx_index,
        )

        if apc_info is None:
            return

        feedback_id = self.stats.allocate_apc_feedback_id()
        apc_info["feedback_id"] = feedback_id
        apc_info["retry_timeout_s"] = apc_retry_timeout_s
        state.begin_unconfirmed_command(
            feedback_id=feedback_id,
            target_tx_power_index=new_tx_index,
            now_s=self.env.now,
        )
        apc_info["context"].update(
            {
                "confirmed_tx_power_index": state.confirmed_tx_power_index,
                "estimated_tx_power_index": state.estimated_tx_power_index,
                "estimate_provenance": state.estimate_provenance,
                "retry_count": 0,
                "max_retries": max_retries,
            }
        )
        self.stats.record_apc_feedback_event(
            event="generated",
            time=self.env.now,
            apc_info=apc_info,
            feedback_id=feedback_id,
            mtype=feedback_mtype,
        )

        # ============================================================
        # Send physical DATA_DOWN.
        # ============================================================
        self.env.process(
            self._send_physical_apc_downlink(
                tx_entity=tx_entity,
                packet=packet,
                apc_info=apc_info,
            )
        )
        return
        
    # ====================================================
    # Utility
    # ====================================================
    def _wait_until_window_quiet(self, w, eps=1e-9):
        """
        Wait until the burst finishes:
        now >= max(rx_end)
        """
        yield from wait_until_window_quiet(self.env, w, eps=eps)

    def _group_overlapping(self, packets):
        """
        Group packets by physical RF edges.

        The receiver buffer spans logical channels. Components are formed
        from time overlap, physical spectrum, and the SF interference mode in
        CollisionEngine. Overlap chains are used only for batching.
        """
        return self.collision.build_rf_groups(packets)

    def _register_rf_window_packet(self, packet):
        """Attach a packet to, or merge, its active spectral RF component."""
        key, window, self.demod_window_sequence = register_rf_window_packet(
            collision=self.collision,
            windows=self.demod_windows,
            packet=packet,
            next_sequence=self.demod_window_sequence,
        )
        return key, window

    # ====================================================
    # RX PIPELINE
    # ====================================================
    def _rx_process(self, packet):
        # ----------------------------------------------------
        # STEP 1: CLASSIFY LOGICAL TARGET
        # ----------------------------------------------------
        # Every feasible RF burst is evaluated at every registered relay/GW.
        # Only the logical next-hop contributes packet-fate and delivery
        # statistics; other copies remain receiver-local interferers.
        is_intended_receiver = (
            getattr(packet, "next_hop", None) == self.relay.id
        )
        packet.is_intended_receiver = is_intended_receiver
    
        # ----------------------------------------------------
        # STEP 1A: NORMALIZE ARRIVAL TIME
        # ----------------------------------------------------
        # AirInterface invokes RelayRX when a packet arrives.
        # If t_arrive is unavailable,
        # use the current simulator time.
        # ----------------------------------------------------
        t_arrive = getattr(packet, "t_arrive", None)
    
        if t_arrive is None:
            t_arrive = float(self.env.now)
            packet.t_arrive = t_arrive
        else:
            try:
                t_arrive = float(t_arrive)
                packet.t_arrive = t_arrive
            except (TypeError, ValueError):
                t_arrive = float(self.env.now)
                packet.t_arrive = t_arrive

        # ----------------------------------------------------
        # ARRIVAL ACCOUNTING
        # ----------------------------------------------------
        if is_intended_receiver:
            self.stats.record_relay_rx(self.relay.id, "rx_arrival_count")
            self.stats.record_node_rx(packet.src_id, "rx_arrival_count")

        
        # ----------------------------------------------------
        # STEP 1B: HALF-DUPLEX ADMISSION GUARD
        # ----------------------------------------------------
        # State radio half-duplex:
        #
        # 1. is_switching_to_tx=True or now < tx_transition_until
        #    The radio is performing an RX-to-TX transition.
        #
        # 2. is_transmitting=True or now < tx_busy_until
        #    The radio is transmitting.
        #
        # A packet arriving in either state does not start RX,
        # so rx_start and rx_end remain None.
        #
        # Skip this guard when tx_full_duplex=True.
        # ----------------------------------------------------

        if not bool(
            getattr(self.relay.hw,"tx_full_duplex",False,)
        ):
            now = float(self.env.now)
        
            tx_pending = bool(getattr(
                self.relay,"tx_pending", False,)
            )
        
            is_switching_to_tx = bool(
                getattr(self.relay,"is_switching_to_tx",False,)
            )
        
            tx_transition_until = float(
                getattr(self.relay,"tx_transition_until",0.0,)
            )
        
            is_transmitting = bool(
                getattr(self.relay,"is_transmitting",False,)
            )
        
            tx_busy_until = float(
                getattr(
                    self.relay,
                    "tx_busy_until",
                    0.0,
                )
            )
        
            transition_busy = (
                is_switching_to_tx
                or now < tx_transition_until
            )
        
            tx_busy = (
                is_transmitting
                or now < tx_busy_until
            )
        
            # >>> PATCH: admission gate
            # Drop the packet only when the transition or TX is active.
            if tx_pending or transition_busy or tx_busy:
                if not is_intended_receiver:
                    return
                if transition_busy:
                    drop_reason = (
                        "half_duplex_tx_transition_busy"
                    )
                elif tx_busy:
                    drop_reason = (
                        "half_duplex_tx_busy"
                    )
                else:
                    drop_reason = (
                        "half_duplex_tx_pending"
                    )
        
                packet.decoded = False
                packet.collided = False
                packet.captured = False
                packet.feasible = None
        
                packet.reason = drop_reason
        
                packet.rx_start = None
                packet.rx_end = None
                packet.rx_time = now
                packet.drop_time = now

                # Packet-source statistics
                self.stats.record_relay_rx(
                    self.relay.id,
                    "half_duplex_drop_count",
                )
                self.stats.record_node_rx(
                    packet.src_id,
                    "half_duplex_drop_count",
                )

                # Admission-gate-specific counter.
                if drop_reason == "half_duplex_tx_pending":
                    self.stats.record_relay_rx(
                        self.relay.id,
                        "tx_pending_drop_count",
                    )
                    self.stats.record_node_rx(
                        packet.src_id,
                        "tx_pending_drop_count",
                    )
                # Event log
                self.stats.record_drop(
                    packet=packet,
                    reason=drop_reason,
                    node_id=self.relay.id,
                )
        
                return





        # ----------------------------------------------------
        # STEP 2: RX DETECTOR GUARD
        # ----------------------------------------------------
        ch = packet.ch_id
        if ch is None:
            raise ValueError("Packet missing ch_id in RelayRX")

        if ch not in self.rx_detector_guard:
            return

        # ----------------------------------------------------
        # STEP 3: TX entity lookup
        # ----------------------------------------------------
        tx_entity = self.relay.network_nodes_map.get(packet.prev_hop)
        if tx_entity is None:
            return



        # ----------------------------------------------------
        # STEP 4: PATHLOSS + LINK BUDGET
        # ----------------------------------------------------
        d = tx_entity.distance_to(self.relay)
        packet.distance_m = d

        if not hasattr(self.relay, "pathloss_cache"):
            self.relay.pathloss_cache = {}

        link = (
            packet.prev_hop,
            self.relay.id,
            round(float(packet.freq_mhz), 6),
            float(self.terrain_delta_h),
        )
        
        if link not in self.relay.pathloss_cache:
            pl = self.model.pathloss(
                d_m=d,
                freq_mhz=packet.freq_mhz,
                h_tx=tx_entity.h,
                h_rx=self.relay.h,
                terrain_delta_h=self.terrain_delta_h,
            )
            self.relay.pathloss_cache[link] = pl
        
        packet.pathloss = self.relay.pathloss_cache[link]

        lb = self.air.compute_link_budget(
            hw_rx=self.relay.hw,
            sf=packet.sf,
            bw_hz=packet.bw_hz,
            ptx=packet.ptx_dbm,
            gtx=tx_entity.gain,
            grx=self.relay.gain,
            pathloss=packet.pathloss,
        )

        feasible = lb.get("feasible", False)
        rssi = lb.get("rssi", None)
        snr = lb.get("snr", None)
        margin = lb.get("margin", None)

        # -----------------------------------------------
        # Optional margin sample for non-feasible packet
        # -----------------------------------------------

        if (not feasible) or (not isinstance(rssi, (int, float))):
            packet.rssi = rssi
            packet.snr = snr
            packet.margin = margin
            packet.feasible = False
        
            packet.decoded = False
            packet.collided = False
            packet.reason = "not_feasible"
            packet.drop_time = float(self.env.now)
            packet.rx_time = float(self.env.now)
        
            if not is_intended_receiver:
                return

            self.stats.record_margin_sample(
                entity_id=tx_entity.id,
                entity_type=self._entity_type(tx_entity.id),
                margin=margin,
                rssi=rssi,
                snr=snr,
                decoded=False,
                reason="not_feasible",
                prev_hop=getattr(packet, "prev_hop", None),
                next_hop=getattr(packet, "next_hop", None),
                time=self.env.now,
            )
        
            self.stats.record_relay_rx(
                self.relay.id,
                "rx_not_feasible",
            )
            self.stats.record_node_rx(
                packet.src_id,
                "rx_not_feasible",
            )
            self.stats.record_drop(
                packet=packet,
                reason=packet.reason,
                node_id=self.relay.id,
            )
            return
        

        # ----------------------------------------------------
        # STEP 5: Bind PHY state + RX timing
        # ----------------------------------------------------
        packet.rssi = float(rssi)
        packet.snr = float(snr)
        packet.margin = float(margin)
        packet.feasible = True
        packet.phy_ready = True
        
        # AirInterface sets t_arrive and _rx_process() normalizes it
        # at the start of RX processing. Do not recalculate it in RelayRX.
        # packet.t_arrive = packet.tx_start + (packet.distance_m / 3e8)
        packet.rx_start = float(packet.t_arrive)
        packet.rx_end = packet.rx_start + float(packet.toa)

        # ----------------------------------------------------
        # RX BUSY WINDOW UPDATE
        # ----------------------------------------------------
        # RelayTX uses this value to wait for the RX buffer/window
        # before transmitting in half-duplex mode.
        #
        # In full-duplex mode, this value may still be updated,
        # but RelayTX ignores it because tx_full_duplex=True.
        # ----------------------------------------------------
        self.relay.rx_busy_until = max(
            float(getattr(self.relay, "rx_busy_until", 0.0)),
            float(packet.rx_end),
        )

        # Register the packet as an active RX session.
        # RelayTX must wait for every active RX session to finish
        # before starting an RX-to-TX transition.
        #
        # Cancellation remains only as a safeguard
        # against unexpected race conditions.

        self._register_active_rx(packet)
        if is_intended_receiver:
            self.stats.record_packet_fate_state(
                packet,
                state="RX_ACTIVE",
                entity_id=self.relay.id,
                time=packet.rx_start,
            )

            self.stats.record_relay_rx(self.relay.id, "rx_feasible")
            self.stats.record_node_rx(packet.src_id, "rx_feasible")

        # ----------------------------------------------------
        # STEP 5B: DEMODULATOR ADMISSION AT PHYSICAL RX START
        # ----------------------------------------------------
        # Every feasible RF burst remains in the collision window, including
        # bursts rejected by the finite modem pool: they still radiate and can
        # interfere. The final drop reason is assigned only after collision
        # resolution so one packet has exactly one terminal loss cause.
        packet.demod_admitted = bool(self.demod.try_acquire(packet))
        if packet.demod_admitted:
            if is_intended_receiver:
                self.stats.record_relay_rx(self.relay.id, "rx_demod_acquired")
                self.stats.record_node_rx(packet.src_id, "rx_demod_acquired")
            self.demod.schedule_release(packet, release_at=packet.rx_end)

        # ----------------------------------------------------
        # STEP 6: Register into RF window
        # ----------------------------------------------------
        # Dynamic spectral windows preserve independent timing for disjoint
        # bands and merge logical channels when their physical bands overlap.
        window_key, w = self._register_rf_window_packet(packet)

        # follower exits
        if window_key in self.demod_leader_busy:
            return

        # leader for this receiver-wide spectral window
        self.demod_leader_busy.add(window_key)

        # allow same-time arrivals
        yield self.env.timeout(1e-9)

        # wait until RF window quiet
        yield from self._wait_until_window_quiet(w)

        # ----------------------------------------------------
        # STEP 7: RF COLLISION RESOLVE
        # ----------------------------------------------------
        ready_packets = list(w)

        # RX windows are no longer active after _wait_until_window_quiet().
        # Remove them from active_rx_packets now. Packets cancelled earlier by
        # RelayTX are kept marked and filtered out below, so they cannot be
        # decoded, forwarded, or delivered.
        for _pkt in ready_packets:
            self._unregister_active_rx(_pkt)

        ready_packets = [
            pkt for pkt in ready_packets
            if not self._is_rx_cancelled(pkt)
        ]

        if not ready_packets:
            self.demod_windows[window_key] = []
            self.demod_leader_busy.discard(window_key)
            return

        groups = self._group_overlapping(ready_packets)

        survivors = []
        failed = []

        for g in groups:
            s, f = self.collision.resolve(
                g,
                receiver_id=self.relay.id,
                event_time=self.env.now,
                statistics=self.stats,
            )

            for p in s:
                p.collided = False
            for p in f:
                p.collided = True
                if not hasattr(p, "reason") or p.reason is None:
                    p.reason = "collision"

            survivors.extend(s)
            failed.extend(f)

        # ----------------------------------------------------
        # STEP 8: Collision accounting
        # ----------------------------------------------------
        for pkt in failed:
            pkt.decoded = False
            pkt.collided = True
            if not bool(getattr(pkt, "is_intended_receiver", False)):
                continue
            """
            if not hasattr(pkt, "reason") or pkt.reason is None:
                pkt.reason = "collision"

            self.stats.record_relay_rx(self.relay.id, "rx_collision")
            self.stats.record_node_rx(pkt.src_id, "rx_collision")
            self.stats.record_drop(packet=pkt, reason=pkt.reason, node_id=self.relay.id)
            """
            if not hasattr(pkt, "reason") or pkt.reason is None:
                pkt.reason = "collision"
        
            # Resolve collisions only after every RF burst has finished.
            pkt.drop_time = float(self.env.now)
            pkt.rx_time = float(self.env.now)
        
            self.stats.record_relay_rx(
                self.relay.id,
                "rx_collision",
            )
            self.stats.record_node_rx(
                pkt.src_id,
                "rx_collision",
            )
            self.stats.record_drop(
                packet=pkt,
                reason=pkt.reason,
                node_id=self.relay.id,
            )

            # -----------------------------------------------
            # NEW: record failed margin sample
            # -----------------------------------------------
            pkt_tx_entity = self.relay.network_nodes_map.get(pkt.prev_hop)
            if pkt_tx_entity is not None:
                self.stats.record_margin_sample(
                    entity_id=pkt_tx_entity.id,
                    entity_type=self._entity_type(pkt_tx_entity.id),
                    margin=getattr(pkt, "margin", None),
                    rssi=getattr(pkt, "rssi", None),
                    snr=getattr(pkt, "snr", None),
                    decoded=False,
                    reason="collided",
                    prev_hop=getattr(pkt, "prev_hop", None),
                    next_hop=getattr(pkt, "next_hop", None),
                    time=self.env.now,
                )

        # ----------------------------------------------------
        # STEP 9: HARD GUARD
        # ----------------------------------------------------
        classified = set(id(p) for p in survivors) | set(id(p) for p in failed)
        missing = [p for p in ready_packets if id(p) not in classified]

        if missing:
            print(f"[GUARD] RelayRX: unclassified packets on ch={ch}")
            for m in missing:
                print(f"  fallback collision: src={m.src_id} rssi={m.rssi:.2f}")

            for m in missing:
                m.decoded = False
                m.reason = "collision"
                failed.append(m)

        # =====================================================
        # STEP 10: APPLY RX-START DEMODULATOR DECISION
        # =====================================================
        survivors_after_demod = []
        failed_after_demod = list(failed)

        for pkt in survivors:
            if not bool(getattr(pkt, "is_intended_receiver", False)):
                continue
            self.stats.record_relay_rx(self.relay.id, "rx_survivor")
            self.stats.record_node_rx(pkt.src_id, "rx_survivor")

            if not bool(getattr(pkt, "demod_admitted", False)):
                pkt.decoded = False
                pkt.reason = getattr(pkt, "demod_reason", "demod_busy")
                pkt.collided = False
                pkt.drop_time = float(self.env.now)
                pkt.rx_time = float(self.env.now)
                self.stats.record_relay_rx(self.relay.id, "demodulator_drop_count")
                self.stats.record_node_rx(pkt.src_id, "demodulator_drop_count")
                self.stats.record_drop(packet=pkt, reason=pkt.reason, node_id=self.relay.id)

                # -----------------------------------------------
                # NEW: record failed margin sample after demod drop
                # -----------------------------------------------
                pkt_tx_entity = self.relay.network_nodes_map.get(pkt.prev_hop)
                if pkt_tx_entity is not None:
                    self.stats.record_margin_sample(
                        entity_id=pkt_tx_entity.id,
                        entity_type=self._entity_type(pkt_tx_entity.id),
                        margin=getattr(pkt, "margin", None),
                        rssi=getattr(pkt, "rssi", None),
                        snr=getattr(pkt, "snr", None),
                        decoded=False,
                        reason="demod_drop",
                        prev_hop=getattr(pkt, "prev_hop", None),
                        next_hop=getattr(pkt, "next_hop", None),
                        time=self.env.now,
                    )

                failed_after_demod.append(pkt)
            else:
                survivors_after_demod.append(pkt)

        survivors = survivors_after_demod
        failed = failed_after_demod

 
        # =====================================================
        # STEP 11: FORWARD / GATEWAY
        # =====================================================
        for pkt in survivors:
            pkt.decoded = True
            pkt.demod_drop = False

            # ----------------------------------------------------
            # Physical DATA_DOWN_FWD APC received by relay target
            # ----------------------------------------------------
            if getattr(pkt, "mtype", None) == MType.DATA_DOWN_FWD:
                target_entity = self.relay

                # Always use the RX time at the current receiver.
                # A multi-hop packet may carry rx_time from its previous hop if
                # metadata was not cleared; do not use that stale value.
                pkt.rx_time = self.env.now

                if is_wang_power_command(pkt) and reporting_enabled(self.relay):
                    gateway = self.relay.network_nodes_map.get("GW")
                    coordinator = getattr(
                        gateway, "wang_command_coordinator", None
                    )
                    if coordinator is None:
                        pkt.decoded = False
                        pkt.reason = "relay_wang_command_coordinator_missing"
                        self.stats.record_drop(
                            packet=pkt,
                            reason=pkt.reason,
                            node_id=self.relay.id,
                        )
                        continue
                    self.stats.record_relay_rx(self.relay.id, "rx_decoded")
                    self.stats.record_rx_channel_use(
                        relay_id=self.relay.id,
                        ch_id=pkt.ch_id,
                        toa=pkt.toa,
                    )
                    self.stats.record_rx(pkt, node_id=self.relay.id)
                    coordinator.receive_at_relay(self.relay, pkt)
                    continue

                # ------------------------------------------------
                # Decode APC_REQ from FCtrl + FOpts
                # ------------------------------------------------
                fctrl_info = decode_fctrl(getattr(pkt, "fctrl", 0))
                is_apc = bool(fctrl_info.get("apc", False))

                if not is_apc:
                    pkt.decoded = False
                    pkt.reason = "relay_downlink_not_apc"
                    self.stats.record_drop(
                        packet=pkt,
                        reason=pkt.reason,
                        node_id=self.relay.id,
                    )
                    continue

                # Controlled synthetic erasure applies after native receiver
                # checks but before parsing/application of the APC command.
                # It never enters record_drop or packet-fate accounting.
                if self.stats.evaluate_additional_feedback_loss(
                    pkt, self.relay.id, float(self.env.now)
                ):
                    pkt.decoded = False
                    pkt.reason = "synthetic_additional_feedback_erasure"
                    continue

                try:
                    req = decode_apc_req(getattr(pkt, "fopts", b""))
                except Exception:
                    pkt.decoded = False
                    pkt.reason = "relay_apc_req_decode_failed"
                    self.stats.record_drop(
                        packet=pkt,
                        reason=pkt.reason,
                        node_id=self.relay.id,
                    )
                    continue

                # ------------------------------------------------
                # Extract TxPowerIndex from APC_REQ
                # ------------------------------------------------
                try:
                    new_tx_index = int(req["tx_power_index"])
                except Exception:
                    pkt.decoded = False
                    pkt.reason = "relay_apc_tx_index_missing"
                    self.stats.record_drop(
                        packet=pkt,
                        reason=pkt.reason,
                        node_id=self.relay.id,
                    )
                    continue

                old_tx_index = getattr(
                    target_entity,
                    "apc_tx_power_index",
                    None,
                )

                old_ptx_dbm = int(
                    getattr(target_entity, "default_ptx", pkt.ptx_dbm)
                )

                # ------------------------------------------------
                # Convert TxPowerIndex -> physical PTx dBm
                # ------------------------------------------------
                try:
                    new_ptx_dbm = tx_power_index_to_dbm(
                        target_entity,
                        new_tx_index,
                    )
                except Exception:
                    pkt.decoded = False
                    pkt.reason = "relay_apc_ptx_index_invalid"

                    # Optional debug output; enable temporarily when needed.
                    if bool(getattr(self.relay, "debug_apc_relay", False)):
                        try:
                            print(
                                "[RELAY_APC_INDEX_INVALID]",
                                f"t={self.env.now:.6f}",
                                f"target={target_entity.id}",
                                f"req_index={new_tx_index}",
                                f"current_index={old_tx_index}",
                                f"supported_ptx={sorted(target_entity.hw.tx_current.keys(), reverse=True)}",
                                f"fopts={getattr(pkt, 'fopts', b'').hex()}",
                            )
                        except Exception:
                            pass

                    self.stats.record_drop(
                        packet=pkt,
                        reason=pkt.reason,
                        node_id=self.relay.id,
                    )
                    continue

                # ------------------------------------------------
                # Apply PTx after DATA_DOWN_FWD is really received
                # ------------------------------------------------
                apc_info = getattr(pkt, "apc_info", None) or {}

                decision_margin = getattr(
                    pkt,
                    "apc_decision_margin",
                    apc_info.get("decision_margin", None),
                )
                decision_rssi = getattr(
                    pkt,
                    "apc_decision_rssi",
                    apc_info.get("decision_rssi", None),
                )
                decision_snr = getattr(
                    pkt,
                    "apc_decision_snr",
                    apc_info.get("decision_snr", None),
                )

                apc_rx_margin = getattr(pkt, "margin", None)
                apc_rx_rssi = getattr(pkt, "rssi", None)
                apc_rx_snr = getattr(pkt, "snr", None)

                # Backward-compatible last_* now refers to decision/uplink metric.
                target_entity.last_margin = decision_margin
                target_entity.last_rssi = decision_rssi
                target_entity.last_snr = decision_snr

                # Explicit decision/uplink metric.
                target_entity.last_decision_margin = decision_margin
                target_entity.last_decision_rssi = decision_rssi
                target_entity.last_decision_snr = decision_snr

                # Explicit APC downlink RX metric.
                target_entity.last_apc_rx_margin = apc_rx_margin
                target_entity.last_apc_rx_rssi = apc_rx_rssi
                target_entity.last_apc_rx_snr = apc_rx_snr

                target_entity.last_feedback_from = getattr(pkt, "prev_hop", None)
                target_entity.last_feedback_time = self.env.now

                # ------------------------------------------------
                # Final apply:
                # APC state = TxPowerIndex
                # PHY state = dBm
                # ------------------------------------------------
                target_entity.apc_tx_power_index = new_tx_index
                target_entity.default_ptx = int(new_ptx_dbm)

                self.stats.record_ptx_update(
                    entity_id=target_entity.id,
                    entity_type=self._entity_type(target_entity.id),
                    old_ptx=old_ptx_dbm,
                    new_ptx=int(new_ptx_dbm),

                    old_tx_power_index=old_tx_index,
                    new_tx_power_index=new_tx_index,
                    tx_power_index=new_tx_index,

                    # Legacy aliases = decision/uplink metric.
                    margin=decision_margin,
                    rssi=decision_rssi,
                    snr=decision_snr,

                    # Explicit decision/uplink metric.
                    decision_margin=decision_margin,
                    decision_rssi=decision_rssi,
                    decision_snr=decision_snr,

                    # Explicit physical APC downlink RX metric.
                    apc_rx_margin=apc_rx_margin,
                    apc_rx_rssi=apc_rx_rssi,
                    apc_rx_snr=apc_rx_snr,

                    feedback_from=getattr(pkt, "prev_hop", None),
                    src_id=getattr(
                        pkt,
                        "apc_uplink_src_id",
                        getattr(pkt, "src_id", None),
                    ),
                    hop_index=getattr(
                        pkt,
                        "apc_uplink_hop_index",
                        getattr(pkt, "hop_index", None),
                    ),
                    time=self.env.now,

                    apply_source="physical_data_down_fwd",
                    control_mtype=getattr(pkt, "mtype", None),
                )

                self.stats.record_relay_rx(self.relay.id, "rx_decoded")
                self.stats.record_rx_channel_use(
                    relay_id=self.relay.id,
                    ch_id=pkt.ch_id,
                    toa=pkt.toa,
                )

                self.stats.record_rx(pkt, node_id=self.relay.id)

                expected_old_index = apc_info.get("context", {}).get(
                    "old_tx_power_index"
                )
                stale_tpi = None
                if expected_old_index is not None and old_tx_index is not None:
                    stale_tpi = int(old_tx_index) != int(expected_old_index)
                self.stats.record_apc_feedback_event(
                    event="applied",
                    time=self.env.now,
                    packet=pkt,
                    applied_tx_power_index=new_tx_index,
                    old_ptx_dbm=old_ptx_dbm,
                    applied_ptx_dbm=int(new_ptx_dbm),
                    stale_tpi=stale_tpi,
                    apc_rx_margin=apc_rx_margin,
                    apc_rx_rssi=apc_rx_rssi,
                    apc_rx_snr=apc_rx_snr,
                )

                continue

            
            # ----------------------------------------------------
            # Only DATA_UP / DATA_UP_FWD are forwarded upstream.
            # DATA_DOWN_FWD is handled in its dedicated block.
            # Node-bound DATA_DOWN is handled by NodeRX, not RelayRX.
            # ACK and other MTypes must not be forwarded automatically
            # as uplink.
            # ----------------------------------------------------
            mtype = getattr(pkt, "mtype", None)
            
            # Treat mtype=None as a legacy uplink when frame_enabled=False.
            # Drop only explicit MTypes other than DATA_UP or DATA_UP_FWD.
            if mtype is not None and mtype not in (MType.DATA_UP, MType.DATA_UP_FWD):
                pkt.decoded = False
                pkt.reason = "relay_unsupported_mtype"
            
                self.stats.record_drop(
                    packet=pkt,
                    reason=pkt.reason,
                    node_id=self.relay.id,
                )
            
                continue

            # -----------------------------------------------
            # Adaptive Ptx feedback based on THIS uplink result
            # -----------------------------------------------
            pkt_tx_entity = self.relay.network_nodes_map.get(pkt.prev_hop)
            self._apply_adaptive_ptx_feedback(pkt_tx_entity, pkt)

            self.stats.record_relay_rx(self.relay.id, "rx_decoded")
            self.stats.record_node_rx(pkt.src_id, "rx_decoded")

            self.stats.record_rx_channel_use(
                relay_id=self.relay.id,
                ch_id=pkt.ch_id,
                toa=pkt.toa,
            )
            # -----------------------------------------------
            # Set rx_time BEFORE record_rx().
            # Overwrite at every receiver so RX events and
            # delivered_packets use the current hop RX time rather than
            # an rx_time that may come from a previous relay.
            # -----------------------------------------------
            pkt.rx_time = self.env.now

            self.stats.record_rx(pkt, node_id=self.relay.id)

            # -----------------------------------------------
            # NEW: record successful margin sample
            # -----------------------------------------------
            if pkt_tx_entity is not None:
                self.stats.record_margin_sample(
                    entity_id=pkt_tx_entity.id,
                    entity_type=self._entity_type(pkt_tx_entity.id),
                    margin=getattr(pkt, "margin", None),
                    rssi=getattr(pkt, "rssi", None),
                    snr=getattr(pkt, "snr", None),
                    decoded=True,
                    prev_hop=getattr(pkt, "prev_hop", None),
                    next_hop=getattr(pkt, "next_hop", None),
                    time=self.env.now,
                )

            # Wang A2-only W2B: reporting only. This call neither imports nor
            # invokes the W1 power controller and cannot change PTx.
            self._handle_wang_snr_success(pkt)
            if reporting_enabled(self.relay):
                gateway = self.relay.network_nodes_map.get("GW")
                coordinator = getattr(
                    gateway, "wang_command_coordinator", None
                )
                if coordinator is not None:
                    coordinator.observe_successful_uplink(self.relay, pkt)

            if self.relay.id == "GW":
                self.stats.record_relay_rx(self.relay.id, "rx_gateway")
                self.stats.record_node_rx(pkt.src_id, "rx_gateway")
                self.stats.record_node_hop_index(pkt.src_id, getattr(pkt, "hop_index", None))

                birth_time = getattr(pkt, "birth_time", None)
                rx_time = getattr(pkt, "rx_time", None)

                if birth_time is not None and rx_time is not None and rx_time >= birth_time:
                    e2e_delay_s = rx_time - birth_time
                    self.stats.record_node_e2e_delay(pkt.src_id, e2e_delay_s)

                # -----------------------------------------------
                # NEW: delivered packet matrix
                # -----------------------------------------------
                self.stats.record_packet_fate_state(
                    pkt,
                    state="DELIVERED",
                    entity_id=self.relay.id,
                    time=getattr(pkt, "rx_time", None),
                )
                self.stats.record_delivered_packet(
                    packet_uid=getattr(pkt, "packet_uid", None),
                    src_id=pkt.src_id,
                    fcnt=getattr(pkt, "fcnt", None),
                    birth_time=getattr(pkt, "birth_time", None),
                    rx_time=getattr(pkt, "rx_time", None),
                    hop_count=getattr(pkt, "hop_index", None),
                    current_tx=getattr(pkt, "current_tx", None),
                    node_id=self.relay.id,
                )

                continue

            pkt.advance_hop()

            # The packet will be retransmitted on a new hop. Clear the
            # current-hop RX results before the next TX/RX.
            # Preserve routing, frame, and APC metadata.
            if hasattr(pkt, "reset_rx_runtime_metadata"):
                pkt.reset_rx_runtime_metadata(preserve_t_arrive=False)

            queue_accepted = self.relay_tx.put(pkt)

            if queue_accepted:
                self.stats.record_relay_rx(self.relay.id, "rx_forwarded")
                self.stats.record_node_rx(pkt.src_id, "rx_forwarded")

        # ----------------------------------------------------
        # STEP 12: Cleanup
        # ----------------------------------------------------
        self.demod_windows[window_key] = []
        self.demod_leader_busy.discard(window_key)
