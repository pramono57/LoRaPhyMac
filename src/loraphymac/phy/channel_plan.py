# ================================================================
# phy/channel_plan.py
# Physical DATA_DOWN / DATA_DOWN_FWD + 
# Relay APC RX cleanup, APC TX index, and timing validation
#
# Compatible for LoRaPHY & LoRaWAN
# ================================================================

class AS923_2:

    def __init__(self, mode="loraphy", downlink_dwell_time=0):
        """
        mode ∈ ["loraphy", "lorawan"]
        Default: LoRaPHY (unrestricted SF, BW, and PTx)
        """
        self.mode = mode.lower()
        if downlink_dwell_time not in (0, 1):
            raise ValueError("downlink_dwell_time must be 0 or 1")
        self.downlink_dwell_time = downlink_dwell_time

        # ============================================================
        # 1. CHANNEL LIST (8 channels, 200 kHz spacing)
        # ============================================================
        self.channels = [
            {"id": 0, "freq": 921.4},
            {"id": 1, "freq": 921.6},
            {"id": 2, "freq": 921.8},
            {"id": 3, "freq": 922.0},
            {"id": 4, "freq": 922.2},
            {"id": 5, "freq": 922.4},
            {"id": 6, "freq": 922.6},
            {"id": 7, "freq": 922.8},

        ]

        # channel_busy_until[ch_id] = timestamp
        self.channel_busy_until = {c["id"]: 0.0 for c in self.channels}

        # ============================================================
        # Build mode-specific rules
        # ============================================================
        if self.mode == "lorawan":
            self._build_lorawan_rules()
        else:
            self._build_loraphy_rules()

        # LBT configuration (applies to both LoRaPHY & LoRaWAN)
        self.lbt_enabled = True
        self.lbt_threshold = -85  # dBm (Indonesia baseline)


    # ================================================================
    # 2. LoRaPHY rules (simple PHY simulation)
    # ================================================================
    def _build_loraphy_rules(self):
        self.dr_table = None               # DR not used
        self.rx1_offset_table = None
        self.rx2_channel_id = None
        self.rx2_dr = None
        self.join_channels = None
        self.max_payload = None
        self.max_eirp = None
        self.duty_cycle = 1.0              # unlimited


    def get_loraphymac_dr_params(self, dr):
        table = {
            0: ("SF12", 125),
            1: ("SF11", 125),
            2: ("SF10", 125),
            3: ("SF9", 125),
            4: ("SF8", 125),
            5: ("SF7", 125),
            6: ("SF7", 250),
        }
        sf_str, bw_khz = table[int(dr)]
        return int(sf_str.replace("SF", "")), int(bw_khz) * 1000
    

    
    # ================================================================
    # 3. LoRaWAN rules (full AS923-2 compliance)
    # ================================================================
    def _build_lorawan_rules(self):

        # Data Rate table
        self.dr_table = {
            0: ("SF12", 125),
            1: ("SF11", 125),
            2: ("SF10", 125),   # RX2 default
            3: ("SF9", 125),
            4: ("SF8", 125),
            5: ("SF7", 125),
            6: ("SF7", 250),
        }

        # RX1 offset table
        self.rx1_offset_table = {i: i for i in range(8)}

        # RX2 default channel = channel 0 (AS923 spec)
        self.rx2_channel_id = 0
        self.rx2_dr = 2

        # Join channels
        self.join_channels = [0, 1]  # channel ID, not frequency

        # Max payload per DR
        self.max_payload = {
            0: 59, 1: 59, 2: 123,
            3: 123, 4: 230, 5: 230,
            6: 230
        }

        self.max_eirp = 16      # node limit
        self.duty_cycle = 0.01  # 1%


    # ================================================================
    # 4. Basic utilities
    # ================================================================
    def num_channels(self):
        return len(self.channels)

    def get_id(self, freq_mhz):
        """
        Convert frequency (MHz) → channel ID.
        """
        for ch in self.channels:
            if abs(ch["freq"] - freq_mhz) < 1e-6:
                return ch["id"]
        raise ValueError(f"Frequency {freq_mhz} MHz was not found in the channel list")


    def get_freq(self, ch_id):
        """
        Convert channel ID → frequency (MHz).
        """
        for ch in self.channels:
            if int(ch["id"]) == int(ch_id):
                return ch["freq"]
    
        raise ValueError(f"Channel ID {ch_id} was not found in the channel list")


    # ================================================================
    # 5. LoRaWAN-specific helpers
    # ================================================================
    def get_dr_params(self, dr):
        if self.dr_table is None:
            raise ValueError("Data rate not available in LoRaPHY mode")
        return self.dr_table[dr]

    def get_rx1_params(self, uplink_ch_id, uplink_dr, offset):
        """Return AS923 RX1 channel/DR using RP002-1.0.5 Tables 74-75."""
        if self.mode == "loraphy":
            return None

        if uplink_dr not in self.dr_table:
            raise ValueError(f"Unsupported AS923 uplink data rate: DR{uplink_dr}")
        if offset not in self.rx1_offset_table:
            raise ValueError(f"AS923 RX1DROffset must be in the 0..7 range, got {offset}")

        rx1_ch = uplink_ch_id
        dr_offset = self.rx1_offset_table[offset]
        minimum_dr = 2 if self.downlink_dwell_time == 1 else 0
        if dr_offset <= 5:
            rx1_dr = max(minimum_dr, uplink_dr - dr_offset)
        else:
            rx1_dr = min(7, max(minimum_dr, uplink_dr + dr_offset - 5))
        if rx1_dr not in self.dr_table:
            raise ValueError(
                f"AS923 RX1 mapping requires unsupported downstream data rate DR{rx1_dr}"
            )
        return rx1_ch, rx1_dr

    def get_rx2_params(self):
        if self.mode == "loraphy":
            return None
        return self.rx2_channel_id, self.rx2_dr
