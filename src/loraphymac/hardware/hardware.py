# ================================================================
# hardware/hardware.py
# Physical DATA_DOWN / DATA_DOWN_FWD + 
# Relay APC RX cleanup, APC TX index, and timing validation
#
# ================================================================

# -------------------------------------------------------
# SX1276 — End Device - Node - Radio
# -------------------------------------------------------
class SX1276:
    type = "SX1276"
    noise_figure = 6
 
    # RX channels and demodulator pool
    rx_channels_max = 8
    demod_pool_A = 1 # SF5–SF12
    demod_pool_B = 0 # only one demodulator in pool A

    # TX channel limit
    tx_phy_radio_count = 1
    tx_full_duplex = False   # Half duplex (TX/RX cannot happen simultaneously)

    # Minimum and maximum TX power
    tx_power_min = 5
    tx_power_max = 20

    # Supported bandwidth constraints (kHz)
    supported_bw = [125, 250, 500]

    # Sensitivity table
    sensitivity = {
        125: {
            7: -123, 8: -126, 9: -129,
            10: -132, 11: -134, 12: -137
        },
        250: {
            7: -120, 8: -123, 9: -126,
            10: -129, 11: -131, 12: -134
        },
        500: {
            7: -117, 8: -120, 9: -123,
            10: -126, 11: -128, 12: -131
        }
    }
    snr_min = {
        7:-7.5, 8:-10, 9:-12.5, 10:-15, 11:-17.5, 12:-20
    }

    # -------------------------------------------------------
    #  VOLTAGE & CURRENT MODEL SX1276 
    # -------------------------------------------------------
    # Operating voltage for SX1276 (datasheet: 1.8–3.9 V)
    # LoRa modules always use 3.3 V.
    voltage = 3.3   # Volt
  
    # -------------------------------------------------------
    # CURRENT (mA)
    # -------------------------------------------------------
    sleep_current = 0.001      # 1 µA
    idle_current = 0.0015      # 1.5 µA
    standby_current = 1.6      # Standby mode
    fs_current = 5.8           # Synthesizer current
    rx_current = 11.5          # Receive
    cad_current = 1.0          # CAD approx

        
    # SX1276 TX current (mA), from the datasheet and interpolation
    tx_current = {
        # RFO (energy-efficient)
        5:  17.0,
        6:  18.5,
        7:  20.0,   # datasheet
        8:  21.5,
        9:  23.0,
        10: 24.5,
        11: 26.0,
        12: 27.5,
        13: 29.0,   # datasheet
        14: 31.0,
    
        # PA_BOOST (high power)
        15: 65.0,   # minimum PA_BOOST characteristic
        16: 76.0,
        17: 87.0,   # datasheet
        18: 98.0,
        19: 109.0,
        20: 120.0   # datasheet
    }

    # -------------------------------------------------------
    #  TIMING MODEL SX1276 (UPDATED, SYNCHRONIZED WITH DATASHEET)
    # -------------------------------------------------------
    """ SYNCRONIZED WITH DATASHEET"""
    
    ts_osc = 0.25              # 250 µs = 0.25 ms
    ts_fs  = 0.06              # 60 µs = 0.06 ms

    ts_re_table = {
        125000: 0.071,            # 71 µs
        250000: 0.063,            # 63 µs
        500000: 0.063             # 63 µs
    }


    def loRa_bitrate(self, sf, bw_hz, cr=4/5):
        return (sf * (bw_hz )) / (2 ** sf) * cr  # bps

    def t_bit(self, sf, bw_hz, cr=4/5):
        Tbit_s = 1 / self.loRa_bitrate(sf, bw_hz, cr)
        return Tbit_s                            # convert to ms

    # TS_TR in ms
    def ts_tr(self, sf, bw_hz, cr=4/5):
        return 0.005 + 0.5 * self.t_bit(sf, bw_hz, cr)    # 5 µs = 0.005 ms

    def ts_hop(self, freq_mhz, freq_new_mhz):
        delta = abs(freq_new_mhz - freq_mhz)
        return 0.02 if delta <= 1 else 0.04  # ms

    # ===============================================================
    # For RSSI ready timing
    # datasheet: TS_RSSI depends on smoothing & BW
    # Use simple proportional model (still realistic):
    # ===============================================================
    def ts_rssi(self, bw_hz, smoothing=1):
        return smoothing * (1 / (bw_hz)) * 1000  # s → ms

    # ===============================================================
    # TRANSITIONS (ms-returning functions)
    # ===============================================================
    def sleep_to_standby(self):
        return self.ts_osc, self.standby_current

    def standby_to_fs(self):
        return self.ts_fs, self.fs_current
    
    def fs_to_tx(self, sf, bw_hz, tx_power):
        """
        Datasheet:
        FSTX → TX = TS_TR
        Current during TS_TR = IDDFS
        After transition → TX current (IDDT)
        """
        t = self.ts_tr(sf, bw_hz)
        i_trans = self.fs_current
        i_tx_final = self.tx_current[tx_power]
        return t, i_trans, i_tx_final
   
    def fs_to_rx(self, bw_hz):
        return self.ts_re_table[bw_hz], self.rx_current
        
    def tx_to_rx(self, freq_mhz, freq_new_mhz, bw_hz):
        t = self.ts_hop(freq_mhz, freq_new_mhz) + self.ts_re_table[bw_hz]
        return t, self.rx_current

    def rx_to_tx(self, freq_mhz, freq_new_mhz, sf, bw_hz, tx_power):
        """
        RX → FSTX → TX
        Timing = TS_HOP + TS_TR
        During transition = IDDFS
        After = TX final current
        """
        t = self.ts_hop(freq_mhz, freq_new_mhz) + self.ts_tr(sf, bw_hz)
        i_trans = self.fs_current
        i_tx_final = self.tx_current[tx_power]
        return t, i_trans, i_tx_final
        
    def tx_to_tx(self, freq_mhz, freq_new_mhz, sf, bw_hz, new_power):
        """
        TX(old) → PaRamp-down → HOP → PaRamp-up + TS_TR → TX(new)
        """
        pa_ramp = 0.04  # 40 µs = 0.04 ms
        ts_hop = self.ts_hop(freq_mhz, freq_new_mhz)
        ts_tr  = self.ts_tr(sf, bw_hz)

        t_transition = pa_ramp + ts_hop + pa_ramp + ts_tr
        i_trans = self.fs_current
        i_tx_final = self.tx_current[new_power]

        return t_transition, i_trans, i_tx_final


    def rx_rssi_time(self, bw_hz):
        return (self.ts_re_table[bw_hz] + self.ts_rssi(bw_hz)), self.rx_current
        
    def rx_continuous(self):
        return "continuous", self.rx_current
    
    def rx_single(self, duration_ms):
        # RX only for duration_ms, then goes to standby
        return duration_ms, self.rx_current, self.standby_current

    def cad_time(self, sf, bw_hz):
        tsym_ms = ((2 ** sf) / (bw_hz)) * 1000
        return 2 * tsym_ms, self.cad_current   




    # -------------------------------------------------------
    # SX1302 — Gateway/Relay Corecell
    # -------------------------------------------------------
class SX1302:
    """
    SX1302 Corecell Gateway Concentrator
    - RX: 8 channels × 8 SF = 64 parallel demod
    - TX: only 1 channel & 1 SF at a time (half-duplex)
    - NF = 3 dB (excellent)
    - Main components:
      - SX1302 – Baseband Processor (Digital DSP)
      - SX1250 – RF Transceiver (TX Driver)
      - SKY66420 – Front-End Module (Power Amplifier / LNA)
      
       SX1303 --> SX1250 --> SKY66420

       SX1302 Datasheet — Semtech DS.SX1302, Rev 3, 2020  
       SKY66420 — Skyworks Datasheet, Rev 1, 2021  
      
    """
    type = "SX1302"
    # Noise Figure typical SX1302
    noise_figure = 3  # dB


    # RX channels and demodulator pool
    rx_channels_max = 8
    demod_pool_A = 8 # SF5–SF12
    demod_pool_B = 8 # SF5–SF10

    # TX channel limit
    # In this simulator, SX1302 represents a relay-gateway platform.
    # tx_phy_radio_count=2 means the relay gateway has two TX-capable LoRa radio paths,
    # not a single bare SX1302 chip with two simultaneous TX chains.
    tx_phy_radio_count = 1   # - single or multi TX radio capacity
    tx_full_duplex = False   # False: Half duplex , True : Full Duplex , RX-while-TX relay
    

    # Minimum and maximum TX power
    tx_power_min = 12
    tx_power_max = 27
 
    # Supported bandwidth constraints
    supported_bw_hz = [125000, 250000, 500000]

    # SX1302 sensitivity (datasheet values)
    sensitivity = {
        125: {
            5: -122, 6: -124, 7: -126,
            8: -128, 9: -130, 10: -133,
            11: -135, 12: -137
        },
        250: {
            5: -119, 6: -121, 7: -123,
            8: -125, 9: -127, 10: -130,
            11: -132, 12: -134
        },
        500: {
            5: -115, 6: -117, 7: -119,
            8: -121, 9: -123, 10: -126,
            11: -128, 12: -130
        }
    }

    # Minimum SNR requirement
    snr_min = {
        5:  -1,
        6:  -3,
        7:  -7.5,
        8:  -10,
        9:  -12.5,
        10: -15,
        11: -17.5,
        12: -20
    }

    # -------------------------------------------------------
    #  VOLTAGE & CURRENT MODEL CORECELL SX1302 
    # ------------------------------------------------------- 
    # RX and TX current based on the Corecell datasheet (mA)
    voltage = 5  # volts for the entire Corecell system
    
    rx_current = 39           # mA 
    idle_current = 10         # mA 

    # TX current (mA) — total current per power level (SX1302 + MCU + SX1250 + SKY66420)
    tx_current = {
        # Efficient path: SX1302 41 mA + internal MCU module 5 mA + SKY66420 12 mA + variable SX1250 current
        12: 145,
        13: 146,
        14: 148,   #datasheet corecell
        15: 150,    
        16: 151,
        17: 153,
        
        # SKY66420 PA_BOOST (SX1302 41 mA + internal MCU module 5 mA + SX1250 95 mA + variable SKY66420 current)
        # Ptx >= 23 dBm uses SKY66420 PA path (high current)       
        18: 190,  
        19: 201,  
        20: 214,
        21: 231,
        22: 252, 
        23: 278,    
        24: 312,     
        25: 354,        
        26: 407,   #datasheet corecell
        27: 421    #datasheet corecell
    }

    # -------------------------------------------------------
    #  TIMING MODEL SX1302 
    # -------------------------------------------------------
    
    # SX1302 SPI timing (ms)
    t_nss_sck  = 0.1    #
    t_nss_miso = 0.15   #

    # SX1250 timing (ms)
    ts_tx  = 0.036     # RX → TX
    ts_rx  = 0.041     # TX → RX
    txrmp  = 0.00034   # PA ramp (optional, not dominant)

    # SKY66420 FEM timing
    tsky_ontx = 0.001
    tsky_onrx = 0.0025
    tsky_off  = 0.001

    def rx_to_tx(self, tx_power):
        if tx_power is None:
            raise ValueError("SX1302.rx_to_tx() requires tx_power")        
        t_trans = self.ts_tx
        i_trans = self.idle_current
        i_tx_final = self.tx_current[tx_power]
        #print(f"[SX1302] RX_TO_TX called | tx_power={tx_power}")
        return t_trans, i_trans, i_tx_final
    
    def tx_to_rx(self):
        t_trans = self.ts_rx
        i_trans = self.idle_current
        #print("[SX1302] TX_TO_RX called")
        return t_trans, i_trans, self.rx_current





class ESP32:
    """
    ESP32 MCU Power Profile
    - Operating voltage: 3.3 V
    - Current-consumption modes based on the ESP32-WROOM / ESP32-WROVER datasheet.
    """

    voltage = 3.3  # Volt

    # Current consumption (mA)
    active_modem_current = 80.0    # CPU active (WiFi/BT off)
    active_current = 10.0          # modem-sleep
    sleep_current = 1.0            # light sleep
    deep_sleep_current = 0.02      # 20 µA

    # Wake-up timing (ms)
    wake_up_time_ms = 2.0

    # ============================================================
    # ENERGY COMPUTATION HELPERS (mJ)
    # ============================================================

    def energy_active_modem(self, time_s):
        """Return energy consumed while the CPU is fully active (Wi-Fi and Bluetooth on)."""
        current_a = self.active_modem_current / 1000.0
        energy_j = self.voltage * current_a * time_s
        return energy_j * 1000.0

    def energy_active(self, time_s):
        """Return energy consumed in modem-sleep mode (Wi-Fi and Bluetooth off)."""
        current_a = self.active_current / 1000.0
        energy_j = self.voltage * current_a * time_s
        return energy_j * 1000.0

    def energy_sleep(self, time_s):
        """Return energy consumed in light-sleep mode (RAM active, clock off)."""
        current_a = self.sleep_current / 1000.0
        energy_j = self.voltage * current_a * time_s
        return energy_j * 1000.0

    def energy_deep_sleep(self, time_s):
        """Return energy consumed in deep-sleep mode."""
        current_a = self.deep_sleep_current / 1000.0
        energy_j = self.voltage * current_a * time_s
        return energy_j * 1000.0

    def energy_wake_up(self):
        """
        Return energy consumed during the wake-up transition from sleep to
        active mode without the modem (Wi-Fi/Bluetooth). The calculation assumes
        ``active_current`` throughout the wake-up interval.
        """
        time_s = self.wake_up_time_ms / 1000.0
        current_a = self.active_current / 1000.0
        energy_j = self.voltage * current_a * time_s
        return energy_j * 1000.0  # mJ

    # Optional helper wrapper
    def compute_energy(self, state: str, time_s: float):
        """
        Flexible interface matching the SX1276/SX1302 energy calculations.
        state ∈ ["active_modem", "active", "sleep", "deep_sleep"]
        """
        if state == "active_modem":
            return self.energy_active_modem(time_s)
        elif state == "active":
            return self.energy_active(time_s)
        elif state == "sleep":
            return self.energy_sleep(time_s)
        elif state == "deep_sleep":
            return self.energy_deep_sleep(time_s)
        else:
            raise ValueError(f"Unknown ESP32 energy state: {state}")


class HardwareFactory:
    @staticmethod
    def get(name):
        name = name.lower()
        if name == "sx1276":
            return SX1276()
        if name == "sx1302":
            return SX1302()
        if name == "esp32":
            return ESP32()            
        raise ValueError(f"Unknown hardware type: {name}")


# ================================================================
# HARDWARE-BASED PTX UTILITIES
# ================================================================

def get_hw_ptx_bounds(hw):
    """
    Get PTx min/max directly from hardware model.

    Priority:
    1. hw.tx_power_min / hw.tx_power_max
    2. min/max keys from hw.tx_current

    Returns:
        (ptx_min, ptx_max)
    """

    tx_current = getattr(hw, "tx_current", None)

    if tx_current is None or len(tx_current) == 0:
        raise ValueError(
            f"{type(hw).__name__} does not define tx_current table."
        )

    supported_ptx = sorted(int(p) for p in tx_current.keys())

    hw_min = getattr(hw, "tx_power_min", None)
    hw_max = getattr(hw, "tx_power_max", None)

    ptx_min = int(hw_min) if hw_min is not None else min(supported_ptx)
    ptx_max = int(hw_max) if hw_max is not None else max(supported_ptx)

    missing = [
        p for p in range(ptx_min, ptx_max + 1)
        if p not in supported_ptx
    ]

    if missing:
        hw_type = getattr(hw, "type", type(hw).__name__)
        raise ValueError(
            f"{hw_type}: missing tx_current entries for PTx={missing} dBm"
        )

    return ptx_min, ptx_max


def apply_hw_ptx_bounds(entity):
    """
    Attach ptx_min and ptx_max to entity based on its hardware model.

    Entity can be:
    - Node
    - Relay
    - Gateway

    Requirement:
    - entity.hw exists
    - entity.hw defines tx_power_min, tx_power_max, tx_current
    """

    if not hasattr(entity, "hw"):
        raise ValueError(
            f"{entity} does not have hardware attribute 'hw'."
        )

    ptx_min, ptx_max = get_hw_ptx_bounds(entity.hw)

    entity.ptx_min = ptx_min
    entity.ptx_max = ptx_max

    return ptx_min, ptx_max

def apply_eirp_ptx_bounds(
    entity,
    max_eirp_dbm,
    antenna_gain_dbi,
    feeder_loss_db=0.0,
):
    """
    Apply regulatory EIRP constraint on top of hardware capability.

    EIRP = PTx + G_ant - L_feeder

    Therefore:
        PTx_reg_max = EIRP_max - G_ant + L_feeder

    Final operational maximum:
        min(hardware capability, regulatory limit)

    The selected PTx must exist in hw.tx_current.
    """

    # -------------------------------------------------
    # 1. Hardware capability
    # -------------------------------------------------
    hw_min, hw_max = get_hw_ptx_bounds(entity.hw)

    # -------------------------------------------------
    # 2. Regulatory conducted-power ceiling
    # -------------------------------------------------
    regulatory_ceiling = (
        float(max_eirp_dbm)
        - float(antenna_gain_dbi)
        + float(feeder_loss_db)
    )

    # -------------------------------------------------
    # 3. Supported hardware PTx values
    # -------------------------------------------------
    supported = sorted(
        int(p)
        for p in entity.hw.tx_current.keys()
        if int(p) >= hw_min
        and int(p) <= hw_max
        and float(p) <= regulatory_ceiling
    )

    if not supported:
        raise ValueError(
            f"{entity.id}: no hardware PTx satisfies "
            f"EIRP_max={max_eirp_dbm} dBm, "
            f"Gant={antenna_gain_dbi} dBi, "
            f"Lfeed={feeder_loss_db} dB."
        )

    entity.ptx_min = hw_min
    entity.ptx_max = max(supported)

    # Metadata/debug
    entity.max_eirp_dbm = float(max_eirp_dbm)
    entity.antenna_gain_dbi = float(antenna_gain_dbi)
    entity.feeder_loss_db = float(feeder_loss_db)
    entity.regulatory_ptx_ceiling_dbm = regulatory_ceiling

    return entity.ptx_min, entity.ptx_max

def validate_ptx(entity, ptx, label=None):
    """
    Validate a PTx value against entity hardware.

    This function is intended for both:
    - initial PTx validation
    - runtime packet PTx validation
    - adaptive PTx validation

    Parameters
    ----------
    entity : object
        Node / Relay / Gateway object with .hw attribute.

    ptx : int or float
        Transmission power in dBm.

    label : str, optional
        Human-readable label for error messages.

    Returns
    -------
    int
        Validated integer PTx.
    """

    if entity is None:
        raise ValueError("validate_ptx(): entity is None.")

    if not hasattr(entity, "hw"):
        raise ValueError(
            f"{entity} does not have hardware attribute 'hw'."
        )

    hw = entity.hw
    hw_type = getattr(hw, "type", type(hw).__name__)
    entity_id = getattr(entity, "id", "ENTITY")
    label = label or entity_id

    tx_current = getattr(hw, "tx_current", None)

    if tx_current is None or len(tx_current) == 0:
        raise ValueError(
            f"{label}: {hw_type} does not define tx_current table."
        )

    ptx = int(ptx)

    # Ensure ptx_min and ptx_max exist
    if not hasattr(entity, "ptx_min") or not hasattr(entity, "ptx_max"):
        apply_hw_ptx_bounds(entity)

    pmin = int(entity.ptx_min)
    pmax = int(entity.ptx_max)

    if ptx < pmin or ptx > pmax:
        raise ValueError(
            f"{label}: PTx={ptx} dBm is outside hardware range "
            f"[{pmin}, {pmax}] dBm for {hw_type}."
        )

    if ptx not in tx_current:
        raise ValueError(
            f"{label}: tx_current for PTx={ptx} dBm "
            f"is not defined in {hw_type}."
        )

    return ptx


def validate_initial_ptx(entity, ptx_init, label=None):
    """
    Backward-compatible wrapper for initial PTx validation.
    """

    return validate_ptx(entity, ptx_init, label=label)


"""
def validate_initial_ptx(entity, ptx_init, label=None):
    
    #Validate initial PTx against entity hardware-based PTx bounds.
    

    label = label or getattr(entity, "id", "ENTITY")

    if not hasattr(entity, "ptx_min") or not hasattr(entity, "ptx_max"):
        apply_hw_ptx_bounds(entity)

    pmin = int(entity.ptx_min)
    pmax = int(entity.ptx_max)

    hw_type = getattr(entity.hw, "type", type(entity.hw).__name__)

    if ptx_init < pmin or ptx_init > pmax:
        raise ValueError(
            f"{label}: initial PTx={ptx_init} dBm is outside "
            f"hardware range [{pmin}, {pmax}] dBm for {hw_type}"
        )

    if ptx_init not in entity.hw.tx_current:
        raise ValueError(
            f"{label}: tx_current for initial PTx={ptx_init} dBm "
            f"is not defined in {hw_type}"
        )

    return True
"""
