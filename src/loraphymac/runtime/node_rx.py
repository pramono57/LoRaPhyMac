# ============================================================
# runtime/node_rx.py
# Physical DATA_DOWN / DATA_DOWN_FWD + 
# Relay APC RX cleanup, APC TX index, and timing validation
#
# Physical DATA_DOWN APC receiver for End Device
# ============================================================

from collections import defaultdict

from loraphymac.runtime.apc_commands import decode_apc_req, encode_apc_ans
from loraphymac.runtime.apc_power import tx_power_index_to_dbm
from loraphymac.mac.frame_defs import MType, decode_fctrl
from loraphymac.runtime.wang_power_commands import is_wang_power_command
from loraphymac.runtime.wang_snr_reporting import reporting_enabled
from loraphymac.runtime.receiver_collision_window import (
    register_rf_window_packet,
    wait_until_window_quiet,
)


class NodeRX:
    # Historical default: only intended DATA_DOWN is scheduled. The opt-in
    # instance flag enables receiver-wide RF observation for shared collision
    # evaluation without changing old experiment behavior.
    accepts_non_target_rf = False

    def __init__(
        self,
        env,
        node,
        air_interface,
        model,
        terrain_delta_h,
        statistics,
        collision_engine=None,
        demodulator_pool=None,
        collision_enabled=False,
    ):
        self.env = env
        self.node = node
        self.air = air_interface
        self.model = model
        self.terrain_delta_h = terrain_delta_h
        self.stats = statistics
        self.collision = collision_engine
        self.demod = demodulator_pool
        self.collision_enabled = bool(collision_enabled)
        if self.collision_enabled and (self.collision is None or self.demod is None):
            raise ValueError(
                "NodeRX collision mode requires collision_engine and demodulator_pool"
            )
        # AirInterface consults the receiver instance, so the historical class
        # default remains False while the opt-in path observes unrelated RF.
        self.accepts_non_target_rf = self.collision_enabled
        self.demod_windows = defaultdict(list)
        self.demod_leader_busy = set()
        self.demod_window_sequence = 0
        self._ambient_packets = []

        if not hasattr(self.node, "rx_windows"):
            self.node.rx_windows = []

        if not hasattr(self.node, "is_transmitting"):
            self.node.is_transmitting = False

        if not hasattr(self.node, "tx_busy_until"):
            self.node.tx_busy_until = 0.0

    def on_arrival(self, packet):
        self.env.process(self._rx_process(packet))

    def _entity_type(self, entity_id):
        s = str(entity_id)
        if s == "GW" or s.startswith("GW"):
            return "gateway"
        if s.startswith("R"):
            return "relay"
        return "node"

    def _is_rx_window_open_for_packet(self, packet):
        """
        Check the RX window for DATA_DOWN.
    
        Validated fields:
        - time
        - ch_id
        - bw_hz
    
        ``freq_mhz`` is not used as a channel guard, but remains part of
        the path-loss and link-budget calculations.
        """
        windows = getattr(self.node, "rx_windows", [])
        t = float(self.env.now)
    
        pkt_ch_id = getattr(packet, "ch_id", None)
        pkt_bw_hz = getattr(packet, "bw_hz", None)
    
        for w in windows:
            if isinstance(w, dict):
                if bool(w.get("closed", False)):
                    continue
    
                w_start = float(w.get("start", 0.0))
                w_end = float(w.get("end", 0.0))
    
                if not (w_start <= t <= w_end):
                    continue
    
                win_ch_id = w.get("ch_id", None)
                win_bw_hz = w.get("bw_hz", None)
    
                if win_ch_id is not None:
                    if pkt_ch_id is None:
                        continue
    
                    if int(pkt_ch_id) != int(win_ch_id):
                        continue
    
                if win_bw_hz is not None:
                    if pkt_bw_hz is None:
                        continue
    
                    if int(pkt_bw_hz) != int(win_bw_hz):
                        continue
    
                return True
    
            else:
                try:
                    w_start, w_end = w
                    w_start = float(w_start)
                    w_end = float(w_end)
                except Exception:
                    continue
    
                if w_start <= t <= w_end:
                    return True
    
        return False
        
    def _drop(self, packet, reason, *, collided=False):
        packet.decoded = False
        packet.collided = bool(collided)
        packet.reason = reason
        packet.rx_time = self.env.now

        self.stats.record_node_rx(self.node.id, f"{reason}_count")
        self.stats.record_drop(
            packet=packet,
            reason=reason,
            node_id=self.node.id,
        )

    def _active_rx_window(self, packet, *, strict_tuning):
        """Return the active RX1 window matching this physical burst."""
        now = float(self.env.now)
        for window in getattr(self.node, "rx_windows", []):
            if not isinstance(window, dict) or bool(window.get("closed", False)):
                continue
            if not float(window.get("start", 0.0)) <= now <= float(
                window.get("end", 0.0)
            ):
                continue
            if not strict_tuning:
                return window
            checks = (
                ("ch_id", int),
                ("bw_hz", int),
                ("sf", int),
            )
            mismatch = False
            for field, converter in checks:
                expected = window.get(field)
                actual = getattr(packet, field, None)
                if expected is None:
                    continue
                if actual is None or converter(actual) != converter(expected):
                    mismatch = True
                    break
            if mismatch:
                continue
            expected_freq = window.get("freq_mhz")
            actual_freq = getattr(packet, "freq_mhz", None)
            if expected_freq is not None and (
                actual_freq is None
                or abs(float(actual_freq) - float(expected_freq)) > 1e-9
            ):
                continue
            return window
        return None

    def _prepare_receiver_phy(self, packet):
        """Calculate this burst's terrain-aware link budget at this NodeRX."""
        tx_id = getattr(packet, "prev_hop", None)
        if tx_id is None:
            tx_id = getattr(packet, "current_tx", None)
        tx_entity = self.node.network_nodes_map.get(tx_id)
        if tx_entity is None:
            return None, False

        distance_m = tx_entity.distance_to(self.node)
        packet.distance_m = distance_m
        if not hasattr(self.node, "node_rx_pathloss_cache"):
            self.node.node_rx_pathloss_cache = {}
        cache_key = (
            tx_entity.id,
            self.node.id,
            round(float(packet.freq_mhz), 6),
            float(self.terrain_delta_h),
        )
        if cache_key not in self.node.node_rx_pathloss_cache:
            self.node.node_rx_pathloss_cache[cache_key] = self.model.pathloss(
                d_m=distance_m,
                freq_mhz=packet.freq_mhz,
                h_tx=tx_entity.h,
                h_rx=self.node.h,
                terrain_delta_h=self.terrain_delta_h,
            )
        pathloss = self.node.node_rx_pathloss_cache[cache_key]
        link_budget = self.air.compute_link_budget(
            hw_rx=self.node.hw,
            sf=packet.sf,
            bw_hz=packet.bw_hz,
            ptx=packet.ptx_dbm,
            gtx=tx_entity.gain,
            grx=self.node.gain,
            pathloss=pathloss,
        )
        packet.pathloss = pathloss
        packet.rssi = link_budget["rssi"]
        packet.snr = link_budget["snr"]
        packet.margin = link_budget["margin"]
        packet.feasible = bool(link_budget["feasible"])

        arrival = getattr(packet, "t_arrive", None)
        if arrival is None:
            arrival = getattr(packet, "tx_start", self.env.now)
        packet.rx_start = float(arrival)
        end = getattr(packet, "t_end", None)
        if end is None:
            end = packet.rx_start + float(getattr(packet, "toa", 0.0) or 0.0)
        packet.rx_end = float(end)
        return tx_entity, packet.feasible

    def _register_collision_packet(self, packet):
        key, window, self.demod_window_sequence = register_rf_window_packet(
            collision=self.collision,
            windows=self.demod_windows,
            packet=packet,
            next_sequence=self.demod_window_sequence,
        )
        return key, window

    def _admit_demodulator(self, packet, rx_window):
        if rx_window is None:
            packet.demod_admitted = False
            packet.demod_reason = "preamble_outside_rx1"
            return
        packet.demod_admitted = bool(self.demod.try_acquire(packet))
        if packet.demod_admitted:
            self.demod.schedule_release(packet, release_at=packet.rx_end)

    def _rx_process_collision(self, packet):
        """Opt-in single-channel RX1 path using RelayRX CollisionEngine rules."""
        intended = (
            getattr(packet, "next_hop", None) == self.node.id
            and getattr(packet, "mtype", None) == MType.DATA_DOWN
        )
        packet.is_intended_receiver = intended
        tx_entity, feasible = self._prepare_receiver_phy(packet)
        if tx_entity is None:
            if intended:
                self._drop(packet, "node_downlink_unknown_tx")
            return

        # Retain only real, still-active physical bursts. This receiver-local
        # RF history lets a frame that began just before RX1 interfere during
        # the overlapping portion without inventing a second transmission.
        now = float(self.env.now)
        self._ambient_packets = [
            current
            for current in self._ambient_packets
            if float(getattr(current, "rx_end", now)) > now
        ]
        if feasible:
            self._ambient_packets.append(packet)

        if not feasible:
            if intended:
                self._drop(packet, "node_downlink_not_feasible")
            return

        if not intended:
            # A destination-independent burst joins an already active desired
            # component only when CollisionEngine finds a physical RF edge.
            for _key, active in list(self.demod_windows.items()):
                if not active or not any(
                    bool(getattr(current, "is_intended_receiver", False))
                    and self.collision.packets_share_rf_edge(packet, current)
                    for current in active
                ):
                    continue
                tuned_window = self._active_rx_window(packet, strict_tuning=True)
                self._admit_demodulator(packet, tuned_window)
                self._register_collision_packet(packet)
                self.node.extend_rx_window_until(
                    env=self.env,
                    hold_until_s=float(packet.rx_end) + 1e-7,
                    reason="interfering_burst_detected",
                )
                break
            return

        rx_window_enabled = bool(getattr(self.node, "rx_window_enabled", True))
        rx_window = self._active_rx_window(packet, strict_tuning=True)
        if rx_window_enabled and rx_window is None:
            self._drop(packet, "node_rx_window_closed_or_tuning_mismatch")
            return

        tx_busy_until = float(getattr(self.node, "tx_busy_until", 0.0))
        if bool(getattr(self.node, "is_transmitting", False)) or now < tx_busy_until:
            self._drop(packet, "node_half_duplex_tx_busy")
            return

        self._admit_demodulator(packet, rx_window)
        window_key, window_packets = self._register_collision_packet(packet)
        for interferer in list(self._ambient_packets):
            if interferer is packet:
                continue
            if self.collision.packets_share_rf_edge(packet, interferer):
                _, window_packets = self._register_collision_packet(interferer)

        self.node.extend_rx_window_until(
            env=self.env,
            hold_until_s=max(float(p.rx_end) for p in window_packets) + 1e-7,
            reason="data_down_collision_window",
        )
        if window_key in self.demod_leader_busy:
            return
        self.demod_leader_busy.add(window_key)
        yield self.env.timeout(1e-9)
        yield from wait_until_window_quiet(self.env, window_packets)

        ready_packets = list(window_packets)
        survivors = []
        failed = []
        for group in self.collision.build_rf_groups(ready_packets):
            group_survivors, group_failed = self.collision.resolve(
                group,
                receiver_id=self.node.id,
                event_time=self.env.now,
                statistics=self.stats,
            )
            survivors.extend(group_survivors)
            failed.extend(group_failed)

        for failed_packet in failed:
            if bool(getattr(failed_packet, "is_intended_receiver", False)):
                self._drop(failed_packet, "collision", collided=True)

        for survivor in survivors:
            if not bool(getattr(survivor, "is_intended_receiver", False)):
                continue
            if not bool(getattr(survivor, "demod_admitted", False)):
                self._drop(
                    survivor,
                    getattr(survivor, "demod_reason", "demod_busy"),
                )
                continue
            # The RX1 window was held a tiny amount beyond the RF component,
            # so the unchanged legacy payload/APC/Wang handler can run after
            # the shared physical decision without a second airtime wait.
            yield from self._rx_process_legacy(survivor)

        self.demod_windows[window_key] = []
        self.demod_leader_busy.discard(window_key)

    def _rx_process(self, packet):
        if self.collision_enabled:
            yield from self._rx_process_collision(packet)
            return
        yield from self._rx_process_legacy(packet)

    def _rx_process_legacy(self, packet):
        
        # ----------------------------------------------------
        # SimPy generator guard
        # ----------------------------------------------------
        # NodeRX is invoked by env.process(), so this function must remain a
        # generator even though RX processing is instantaneous.
        yield self.env.timeout(0)        
        
        # ----------------------------------------------------
        # STEP 1: Filter target
        # ----------------------------------------------------
        if getattr(packet, "next_hop", None) != self.node.id:
            return

        # ----------------------------------------------------
        # STEP 2: Only DATA_DOWN is handled by NodeRX.
        # ----------------------------------------------------
        if getattr(packet, "mtype", None) != MType.DATA_DOWN:
            return

        # ----------------------------------------------------
        # STEP 3: RX window guard
        # ----------------------------------------------------
        rx_window_enabled = bool(
            getattr(self.node, "rx_window_enabled", True)
        )

        if rx_window_enabled:
            if not self._is_rx_window_open_for_packet(packet):
                """
                print(
                    f"[NODE-RX-WINDOW-CLOSED] "
                    f"t={self.env.now:.6f} "
                    f"node={self.node.id} "
                    f"pkt={getattr(packet, 'current_tx', None)}->{getattr(packet, 'next_hop', None)} "
                    f"src={getattr(packet, 'src_id', None)} "
                    f"dst={getattr(packet, 'dst_id', None)} "
                    f"mtype={getattr(packet, 'mtype', None)} "
                    f"tx_start={getattr(packet, 'tx_start', None)} "
                    f"tx_end={getattr(packet, 'tx_end', None)} "
                    f"toa={getattr(packet, 'toa', None)} "
                    f"rx_windows={getattr(self.node, 'rx_windows', [])}"
                )                
                """
                
                self._drop(packet, "node_rx_window_closed")
                return

        # ----------------------------------------------------
        # STEP 4: Half-duplex guard
        # ----------------------------------------------------
        tx_busy_until = float(getattr(self.node, "tx_busy_until", 0.0))
        is_tx_busy = bool(getattr(self.node, "is_transmitting", False))

        if is_tx_busy or self.env.now < tx_busy_until:
            self._drop(packet, "node_half_duplex_tx_busy")
            return

        # ----------------------------------------------------
        # STEP 5: TX entity lookup
        # ----------------------------------------------------
        tx_id = getattr(packet, "prev_hop", None)
        if tx_id is None:
            tx_id = getattr(packet, "current_tx", None)

        tx_entity = self.node.network_nodes_map.get(tx_id)
        if tx_entity is None:
            self._drop(packet, "node_downlink_unknown_tx")
            return

        # ----------------------------------------------------
        # STEP 6: Link budget
        # ----------------------------------------------------
        d = tx_entity.distance_to(self.node)
        packet.distance_m = d

        pl = self.model.pathloss(
            d_m=d,
            freq_mhz=packet.freq_mhz,
            h_tx=tx_entity.h,
            h_rx=self.node.h,
            terrain_delta_h=self.terrain_delta_h,
        )

        lb = self.air.compute_link_budget(
            hw_rx=self.node.hw,
            sf=packet.sf,
            bw_hz=packet.bw_hz,
            ptx=packet.ptx_dbm,
            gtx=tx_entity.gain,
            grx=self.node.gain,
            pathloss=pl,
        )

        packet.rssi = lb["rssi"]
        packet.snr = lb["snr"]
        packet.margin = lb["margin"]
        packet.pathloss = pl
        packet.feasible = lb["feasible"]

        if not lb["feasible"]:
            self._drop(packet, "node_downlink_not_feasible")
            return

        # ----------------------------------------------------
        # STEP 6B: Preamble/frame detected inside RX1.
        # LoRaWAN-like behavior:
        # if preamble is detected, radio stays active until
        # the downlink frame is fully demodulated.
        # ----------------------------------------------------
        # Use propagation-aware arrival timing when available.
        # IMPORTANT: getattr(..., default) does not protect against an attribute
        # that exists but is None, so handle None explicitly.
        t_arrive = getattr(packet, "t_arrive", None)
        if t_arrive is None:
            t_arrive = getattr(packet, "tx_start", None)
        if t_arrive is None:
            t_arrive = self.env.now
        
        try:
            packet.rx_start = float(t_arrive)
        except Exception:
            packet.rx_start = float(self.env.now)
        
        t_end = getattr(packet, "t_end", None)
        if t_end is None:
            toa = getattr(packet, "toa", 0.0)
            try:
                toa = float(toa)
            except Exception:
                toa = 0.0
            t_end = packet.rx_start + toa
        
        try:
            packet.rx_end = float(t_end)
        except Exception:
            packet.rx_end = packet.rx_start + float(getattr(packet, "toa", 0.0) or 0.0)

        try:
            self.node.extend_rx_window_until(
                env=self.env,
                hold_until_s=packet.rx_end,
                reason="data_down_detected",
            )
        except Exception:
            pass

        wait_s = max(0.0, packet.rx_end - float(self.env.now))

        if wait_s > 0:
            yield self.env.timeout(wait_s)

        packet.rx_time = self.env.now

        # Wang A2-only uses a distinct physical CID and absolute PTx command.
        # This path is reached only after RX1, link-budget, and PHY timing
        # checks above have succeeded.
        if is_wang_power_command(packet) and reporting_enabled(self.node):
            gateway = self.node.network_nodes_map.get("GW")
            coordinator = getattr(gateway, "wang_command_coordinator", None)
            if coordinator is None:
                self._drop(packet, "node_wang_command_coordinator_missing")
                return
            result = coordinator.apply_at_node(self.node, packet)
            if not result.applied:
                self._drop(packet, f"node_wang_{result.reason}")
                return
            packet.decoded = True
            self.stats.record_rx(packet, node_id=self.node.id)
            try:
                self.node.close_rx_window(
                    env=self.env,
                    reason="valid_wang_power_command_decoded",
                )
            except Exception:
                pass
            return

        
        # ----------------------------------------------------
        # STEP 7: Decode APC_REQ from FOpts
        # ----------------------------------------------------
        fctrl_info = decode_fctrl(getattr(packet, "fctrl", 0))
        is_apc = bool(fctrl_info.get("apc", False))

        if not is_apc:
            self._drop(packet, "node_downlink_not_apc")
            return

        # Additional loss is evaluated only after RX-window, half-duplex,
        # feasibility, and physical frame reception have succeeded. It is a
        # receiver-side erasure, not a PHY collision or packet terminal drop.
        if self.stats.evaluate_additional_feedback_loss(
            packet, self.node.id, float(self.env.now)
        ):
            packet.decoded = False
            packet.reason = "synthetic_additional_feedback_erasure"
            return

        try:
            req = decode_apc_req(getattr(packet, "fopts", b""))
        except Exception:
            self._drop(packet, "node_apc_req_decode_failed")
            return


        # ----------------------------------------------------
        # STEP 8: Apply TxPowerIndex to node physical PTx
        # ----------------------------------------------------
        
        old_ptx = int(getattr(self.node, "default_ptx", packet.ptx_dbm))

        old_tx_index = getattr(self.node, "apc_tx_power_index", None)
        new_tx_index = int(req["tx_power_index"])

        try:
            new_ptx = tx_power_index_to_dbm(
                self.node,
                new_tx_index,
            )
        except Exception:
            self._drop(packet, "node_apc_ptx_index_invalid")
            return

        # Clean APC state.
        self.node.apc_tx_power_index = new_tx_index

        # Keep PHY state in dBm for NodeTX, AirInterface, and energy calculations.
        self.node.default_ptx = int(new_ptx)

        apc_info = getattr(packet, "apc_info", None) or {}

        decision_margin = getattr(
            packet,
            "apc_decision_margin",
            apc_info.get("decision_margin", None),
        )
        decision_rssi = getattr(
            packet,
            "apc_decision_rssi",
            apc_info.get("decision_rssi", None),
        )
        decision_snr = getattr(
            packet,
            "apc_decision_snr",
            apc_info.get("decision_snr", None),
        )

        apc_rx_margin = getattr(packet, "margin", None)
        apc_rx_rssi = getattr(packet, "rssi", None)
        apc_rx_snr = getattr(packet, "snr", None)

        # Backward-compatible:
        # last_* now refers to decision/uplink metrics.
        self.node.last_margin = decision_margin
        self.node.last_rssi = decision_rssi
        self.node.last_snr = decision_snr

        # Explicit decision/uplink metric.
        self.node.last_decision_margin = decision_margin
        self.node.last_decision_rssi = decision_rssi
        self.node.last_decision_snr = decision_snr

        # Explicit APC downlink RX metric.
        self.node.last_apc_rx_margin = apc_rx_margin
        self.node.last_apc_rx_rssi = apc_rx_rssi
        self.node.last_apc_rx_snr = apc_rx_snr

        self.node.last_feedback_from = tx_entity.id
        self.node.last_feedback_time = self.env.now

        # Controller pending/belief state is deliberately not reachable from
        # this target-side handler. Without physical APC_ANS, local application
        # cannot confirm delivery to the remote controller.

        packet.decoded = True
        packet.rx_time = self.env.now
       
        # Close RX1 early after valid DATA_DOWN APC is decoded.
        try:
            self.node.close_rx_window(
                env=self.env,
                reason="valid_data_down_apc_decoded",
            )
        except Exception:
            pass


        
        # ----------------------------------------------------
        # STEP 10: Record RX and PTx update
        # ----------------------------------------------------
        self.stats.record_rx(packet, node_id=self.node.id)

        self.stats.record_ptx_update(
            entity_id=self.node.id,
            entity_type="node",
            old_ptx=old_ptx,
            new_ptx=int(new_ptx),

            old_tx_power_index=old_tx_index,
            new_tx_power_index=new_tx_index,
            tx_power_index=new_tx_index,

            # Legacy aliases = decision/uplink metric.
            margin=decision_margin,
            rssi=decision_rssi,
            snr=decision_snr,

            # Explicit fields.
            decision_margin=decision_margin,
            decision_rssi=decision_rssi,
            decision_snr=decision_snr,

            apc_rx_margin=apc_rx_margin,
            apc_rx_rssi=apc_rx_rssi,
            apc_rx_snr=apc_rx_snr,

            feedback_from=tx_entity.id,
            src_id=getattr(
                packet,
                "apc_uplink_src_id",
                getattr(packet, "src_id", None),
            ),
            hop_index=getattr(
                packet,
                "apc_uplink_hop_index",
                getattr(packet, "hop_index", None),
            ),
            time=self.env.now,

            apply_source="physical_data_down",
            control_mtype=getattr(packet, "mtype", None),
        )

        # ----------------------------------------------------
        # STEP 11: Record APC_ANS_SHADOW as logical answer
        # ----------------------------------------------------
        apc_info = getattr(packet, "apc_info", None)

        if apc_info is not None:
            context = apc_info.get("context", {})
            status = apc_info.get("status", None)
            fopts_ans_hex = encode_apc_ans(status).hex()

            self.stats.record_apc_shadow(
                time=self.env.now,
                event="APC_ANS_SHADOW",

                sender_id=self.node.id,
                sender_type="node",
                receiver_id=tx_entity.id,
                receiver_type=self._entity_type(tx_entity.id),

                controller_id=context.get("controller_id", tx_entity.id),
                controller_type=context.get(
                    "controller_type",
                    self._entity_type(tx_entity.id),
                ),
                target_id=self.node.id,
                target_type="node",

                control_direction="upstream_response",
                control_scope="node_apc",

                # Physical dBm-domain.
                old_ptx=old_ptx,
                new_ptx=int(new_ptx),

                # APC TxPowerIndex-domain.
                old_tx_power_index=old_tx_index,
                new_tx_power_index=new_tx_index,
                tx_power_index=new_tx_index,

                dr=context.get("dr", None),
                ch_mask=context.get("ch_mask", None),
                nb_trans=context.get("nb_trans", None),
                status=status,

                # Legacy aliases point to decision/uplink metrics to remain
                # consistent with ptx_history.
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

                src_id=getattr(
                    packet,
                    "apc_uplink_src_id",
                    getattr(packet, "src_id", None),
                ),
                hop_index=getattr(
                    packet,
                    "apc_uplink_hop_index",
                    getattr(packet, "hop_index", None),
                ),

                # These are the original link metrics used by the controller
                # for its decision, not DATA_DOWN prev_hop/next_hop.
                uplink_prev_hop=apc_info.get("decision_prev_hop", None),
                uplink_next_hop=apc_info.get("decision_next_hop", None),

                fopts_hex=fopts_ans_hex,
                note="physical_node_apc_answer",
            )

            expected_old_index = context.get("old_tx_power_index")
            stale_tpi = None
            if expected_old_index is not None and old_tx_index is not None:
                stale_tpi = int(old_tx_index) != int(expected_old_index)
            self.stats.record_apc_feedback_event(
                event="applied",
                time=self.env.now,
                packet=packet,
                applied_tx_power_index=new_tx_index,
                old_ptx_dbm=old_ptx,
                applied_ptx_dbm=int(new_ptx),
                stale_tpi=stale_tpi,
                apc_rx_margin=apc_rx_margin,
                apc_rx_rssi=apc_rx_rssi,
                apc_rx_snr=apc_rx_snr,
            )
