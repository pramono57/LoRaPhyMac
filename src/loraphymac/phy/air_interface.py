# ================================================================
# phy/air_interface.py
# Physical DATA_DOWN / DATA_DOWN_FWD + 
# Relay APC RX cleanup, APC TX index, and timing validation
#
# LoRa PHY Engine (RSSI, SNR, Sensitivity, ToA, MaxDistance)
# Event-driven delivery with propagation delay (t_arrive)
# ================================================================

import numpy as np

class AirInterface:
    
    def __init__(self, env, feasible_margin_db=3, delta_snr=3):
        self.env = env
        self.feasible_margin_db = feasible_margin_db
        self.delta_snr = delta_snr
        self.rx_nodes_map = {}
        self.rx_endpoints_map = {}
    # -----------------------------------------------------------
    # REGISTER RX NODE (relay/gateway)
    # -----------------------------------------------------------
    def register_rx(self, relay):
        """
        Register receiver endpoint into air interface.
        Expect relay has attribute: relay.rx (instance of RelayRX)
        """
        # Store the RX instance, not the class.
        self.rx_nodes_map[relay.id] = relay.rx
        self.rx_endpoints_map[relay.id] = relay

    # -----------------------------------------------------------
    # TRANSMIT (broadcast into air, then deliver to next hop)
    # -----------------------------------------------------------
    def transmit(self, packet):
        """
        Broadcast independent packet copies to every registered RF receiver.

        ``next_hop`` remains the intended logical receiver. Other receivers
        observe the burst as potential interference. A transmitter never
        receives its own burst.
        """
        current_tx = getattr(packet, "current_tx", None)

        for receiver_id, rx in tuple(self.rx_nodes_map.items()):
            if receiver_id == current_tx:
                continue

            pkt = packet.copy()
            pkt.rx_node = receiver_id
            pkt.is_intended_receiver = (
                getattr(packet, "next_hop", None) == receiver_id
            )
            if (
                not pkt.is_intended_receiver
                and not bool(getattr(rx, "accepts_non_target_rf", True))
            ):
                continue
            self._set_receiver_arrival(pkt, receiver_id)
            self.env.process(self._deliver(rx, pkt))

    def _set_receiver_arrival(self, packet, receiver_id):
        """Set propagation timing for one transmitter/receiver pair."""
        receiver = self.rx_endpoints_map.get(receiver_id)
        transmitter = self.rx_endpoints_map.get(
            getattr(packet, "current_tx", None)
        )

        distance_m = None
        if transmitter is not None and receiver is not None:
            try:
                distance_m = float(transmitter.distance_to(receiver))
            except (AttributeError, TypeError, ValueError):
                distance_m = None

        if distance_m is not None:
            packet.distance_m = distance_m
            packet.t_arrive = float(packet.tx_start) + distance_m / 3e8
        elif packet.t_arrive is None:
            packet.t_arrive = float(packet.tx_start)

        packet.t_end = float(packet.t_arrive) + float(packet.toa)


    def _deliver(self, rx, pkt):
        """
        Internal delivery coroutine:
        - wait until pkt.t_arrive (if set and > now)
        - then call rx.on_arrival(pkt)
        """
        t_arrive = getattr(pkt, "t_arrive", None)
        if isinstance(t_arrive, (int, float)) and t_arrive > self.env.now:
            yield self.env.timeout(t_arrive - self.env.now)

        # ARRIVAL AT RX (event-driven, no queue)
        rx.on_arrival(pkt)

    # -----------------------------------------------------------
    # 1. RSSI
    # -----------------------------------------------------------
    def compute_rssi(self, ptx, gtx, grx, pathloss):
        """
        RSSI = Ptx + Gtx + Grx – Pathloss
        """
        return ptx + gtx + grx - pathloss

    # -----------------------------------------------------------
    # 2. Noise Floor (BW Hz)
    # -----------------------------------------------------------
    def compute_noise_floor(self, bw_hz, noise_figure):
        """
        Noise floor = Thermal + 10*log10(BW) + NF
        """
        thermal = -174  # dBm/Hz
        return thermal + 10 * np.log10(bw_hz) + noise_figure

    # -----------------------------------------------------------
    # 3. Sensitivity Lookup
    # -----------------------------------------------------------
    def compute_sensitivity(self, hw_rx, sf, bw_hz):
        bw_khz = bw_hz // 1000
        if bw_khz not in hw_rx.sensitivity:
            raise ValueError(
                f"Unsupported bandwidth: {bw_khz} kHz for {type(hw_rx).__name__}"
            )
    
        if sf not in hw_rx.sensitivity[bw_khz]:
            raise ValueError(
                f"Unsupported SF: SF{sf} for bandwidth {bw_khz} kHz in {type(hw_rx).__name__}"
            )
            
        return hw_rx.sensitivity[bw_khz][sf]

    # -----------------------------------------------------------
    # 4. SNR from RSSI & Noise
    # -----------------------------------------------------------
    def compute_snr(self, rssi, bw_hz, noise_figure):
        noise_floor = self.compute_noise_floor(bw_hz, noise_figure)
        return rssi - noise_floor

   
    # ===================================================================
    # 7. PHY LINK BUDGET
    # ===================================================================
    def compute_link_budget(
        self,
        hw_rx,
        sf,
        bw_hz,
        ptx,
        gtx,
        grx,
        pathloss,
        feasible_margin_db=None,
    ):
        """
        Evaluate the LoRaPhyMAC PHY.
    
        Margin definitions:
            raw_margin = RSSI - P_rx_sens
            margin     = RSSI - P_rx_sens - M_feas
    
        Feasibility:
            feasible = margin > 0
                     = RSSI > P_rx_sens + M_feas
    
        Note:
            SNR and minimum SNR are still calculated and returned as
            observational metrics for possible SNR-based ADR development,
            but they do not determine feasibility.
        """
        if feasible_margin_db is None:
            feasible_margin_db = self.feasible_margin_db
    
        feasible_margin_db = float(feasible_margin_db)
    
        # --------------------------------------------------------
        # RSSI and sensitivity
        # --------------------------------------------------------
        rssi = self.compute_rssi(ptx, gtx, grx, pathloss)
        sens = self.compute_sensitivity(hw_rx, sf, bw_hz)
    
        # --------------------------------------------------------
        # SNR diagnostic only
        # --------------------------------------------------------
        noise_floor = self.compute_noise_floor(bw_hz, hw_rx.noise_figure)
        snr = self.compute_snr(rssi, bw_hz, hw_rx.noise_figure)
        snr_min_req = hw_rx.snr_min[sf]
    
        # --------------------------------------------------------
        # RSSI margin
        # --------------------------------------------------------
        raw_margin = rssi - sens
    
        # Effective margin after feasible compensation
        margin = raw_margin - feasible_margin_db
    
        # --------------------------------------------------------
        # Feasibility RSSI-based only
        # --------------------------------------------------------
        feasible_rssi = margin > 0
        feasible = feasible_rssi
    
        return {
            "rssi": rssi,
            "sensitivity": sens,
    
            # Margin before M_feas compensation
            "raw_margin": raw_margin,
    
            # Effective margin used by routing, RX, APC, and the visualizer
            "margin": margin,
    
            # Feasibility compensation value
            "mfeas": feasible_margin_db,
    
            # SNR is for observation and potential LoRaWAN-style ADR only
            "noise_floor": noise_floor,
            "snr": snr,
            "snr_min": snr_min_req,
    
            # Final feasibility result
            "feasible": feasible,
            "feasible_rssi": feasible_rssi,
        }    


    
    # ===================================================================
    # 8. LoRa ToA (Time-on-Air)
    # ===================================================================
    def compute_payload_sym(
        self, sf, bw_hz, payload_bytes,
        cr=1, crc_enabled=True, header_enabled=True, low_data_rate_opt=None
    ):
        """
        Return the number of payload symbols as an integer.
        ``cr`` is the LoRa coding-rate index: 1..4 (4/5..4/8).
        """
        if sf < 5 or sf > 12:
            raise ValueError(f"Unsupported SF: {sf}")   
            
        # Low Data Rate Optimization (DE)
        if low_data_rate_opt is None:
            low_data_rate_opt = 1 if (bw_hz == 125000 and sf >= 11) else 0

        DE = low_data_rate_opt
        IH = 0 if header_enabled else 1
        CRC = 1 if crc_enabled else 0

        # --- PATCH CHANGE ---
        # Ensure an integer result rather than a NumPy float.
        payload_sym = 8 + max(
            np.ceil(
                (8 * payload_bytes
                 - 4 * sf
                 + 28
                 + 16 * CRC
                 - 20 * IH) /
                (4 * (sf - 2 * DE))
            ) * (cr + 4),
            0
        )
        return int(payload_sym)

    def compute_toa(
        self, sf, bw_hz, payload_bytes,
        cr=1,
        preamble_len=8,
        crc_enabled=True,
        header_enabled=True,
        low_data_rate_opt=None
    ):
        """
        Return the total LoRa time-on-air in seconds.
        """
        T_sym = (2 ** sf) / bw_hz

        payload_sym = self.compute_payload_sym(
            sf, bw_hz, payload_bytes,
            cr, crc_enabled, header_enabled, low_data_rate_opt
        )

        T_preamble = (preamble_len + 4.25) * T_sym
        T_packet = T_preamble + payload_sym * T_sym

        return T_packet

    # ===================================================================
    # 9. Data Rate (bps)
    # ===================================================================
    def compute_data_rate(self, sf, bw_hz, cr=1):
        """
        Approximate theoretical LoRa data rate:
          R = SF * BW / 2^SF * (4/(4+CR))
        ``cr`` is the LoRa coding-rate index: 1..4 (4/5..4/8).
        """
    
        cr_factor = 4.0 / (4.0 + float(cr))  # --- PATCH CHANGE ---
        return (sf * bw_hz / (2 ** sf)) * cr_factor

    # ===================================================================
    # 10. Maximum Distance (binary search)
    # ===================================================================
    def max_distance(
        self, model, hw_tx, hw_rx,
        sf, bw_hz, ptx,
        gtx, grx,
        h_tx, h_rx,
        freq_mhz, terrain_delta_h=0,
        feasible_margin_db=None, delta_snr=None,
        d_min=10, d_max=50000,
        iterations=40,
        use_snr=False,              # --- PATCH ADD ---
    ):
        """
        Find the maximum distance using binary search.
        If ``use_snr=True``, use both RSSI and SNR.
        Otherwise, use RSSI only (baseline).
        """
        if feasible_margin_db is None:
            feasible_margin_db = self.feasible_margin_db
        if delta_snr is None:
            delta_snr = self.delta_snr

        low = float(d_min)
        high = float(d_max)
        if low < 0 or high <= low:
            raise ValueError("max_distance requires 0 <= d_min < d_max")
        if int(iterations) <= 0:
            raise ValueError("max_distance iterations must be > 0")
        
        
        for _ in range(int(iterations)):
            mid = (low + high) / 2

            pl = model.pathloss(
                d_m=mid,
                freq_mhz=freq_mhz,
                h_tx=h_tx,
                h_rx=h_rx,
                terrain_delta_h=terrain_delta_h
            )

            lb = self.compute_link_budget(
                hw_rx=hw_rx,
                sf=sf,
                bw_hz=bw_hz,
                ptx=ptx,
                gtx=gtx,
                grx=grx,
                pathloss=pl,
                feasible_margin_db=feasible_margin_db,
            )

            feasible = bool(lb["feasible"])
            if use_snr:
                feasible = feasible and (
                    float(lb["snr"])
                    > float(lb["snr_min"]) + float(delta_snr)
                )

            if feasible:
                low = mid
            else:
                high = mid

        return low
