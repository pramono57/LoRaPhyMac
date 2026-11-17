# ================================================================
# domain/node.py
# Physical DATA_DOWN / DATA_DOWN_FWD + 
# Relay APC RX cleanup, APC TX index, and timing validation
#
# - BusyMap, Timeline Logger
# - Optional MCU energy accounting
# - Fixed node TX statistics
# - Robust sleep current fallback
# ================================================================


from enum import Enum
from loraphymac.hardware.hardware import validate_ptx


def observed_duration_s(env, duration_s, start_time=None):
    """Clip a state duration to the configured finite observation horizon.

    SimPy stops before resuming processes whose timeout ends exactly at a
    numeric ``env.run(until=...)`` boundary. Energy is therefore accounted
    before the timeout, but only for the part of the state that lies inside
    ``env.simulation_end`` when that boundary is configured.
    """
    duration_s = max(float(duration_s), 0.0)
    if duration_s == 0.0:
        return 0.0

    horizon = getattr(env, "simulation_end", None)
    if horizon is None:
        return duration_s

    start = float(env.now if start_time is None else start_time)
    remaining = max(float(horizon) - start, 0.0)
    return min(duration_s, remaining)


# ============================================================
# ENUM STATE
# ============================================================
class NodeState(Enum):
    SLEEP = 0
    STANDBY = 1
    FS = 2
    TX = 3
    RX = 4


# ============================================================
# MAIN NODE CLASS
# ============================================================
class Node:

    def __init__(
        self,
        node_id,
        loc,
        height,
        radio_hw,
        mcu_hw,
        gain_dbi,
        channel_plan,
        air_interface,
        routing_engine,
        collision_engine,
        channel_mode="MULTI",
        rx_mode="single",
        lbt_mode="disable",
        include_mcu_energy=True,   # NEW
        terrain_delta_h=90,
    ):

        # Identity
        self.id = node_id
        self.loc = loc
        self.h = height
        self.gain = gain_dbi
        self.terrain_delta_h = terrain_delta_h

        # Hardware
        self.hw = radio_hw
        self.mcu = mcu_hw
        self.include_mcu_energy = include_mcu_energy
        

        # Engines
        self.air = air_interface
        self.routing = routing_engine
        self.collision = collision_engine
        self.ch_plan = channel_plan

        # Modes
        self.channel_mode = channel_mode
        self.lbt_mode = lbt_mode
        self.rx_mode = rx_mode

        # State
        self.state = NodeState.SLEEP
        self.freq = None
        self.tx_power = None
        self.tx_current_mA = None
        self.rx_current_mA = None

        # BusyMap (timestamp only)
        self.busy_map = {ch["id"]: 0.0 for ch in self.ch_plan.channels}

        # Energy accounting
        self.energy_tx_mJ = 0.0
        self.energy_rx_mJ = 0.0
        self.energy_sleep_mJ = 0.0
        self.energy_idle_mJ = 0.0
        self.energy_transition_mJ = 0.0
        self.energy_cad_mJ = 0.0
        self.total_transition_time_s = 0.0

        # TX airtime accounting
        self.total_tx_time_s = 0.0
        self.tx_airtime_per_channel = {}
        self.tx_packet_count_per_channel = {}

        # RX airtime accounting for Class-A-like RX windows
        self.total_rx_time_s = 0.0
        self.rx_window_count = 0
        self.rx_window_timeout_count = 0
        self.rx_window_downlink_count = 0
        self.rx_window_early_close_count = 0
        
        self.rx_windows = []
        self.current_rx_window = None
        self.rx_busy_until = 0.0
        
        self.is_transmitting = False
        self.tx_busy_until = 0.0
        
        
        # Delay / Sleep tracking
        self.last_sleep_duration_s = 0.0
        self.total_sleep_time_s = 0.0

        # Statistics
        self.next_tx_allowed = 0.0
        self.last_rssi = None
        self.last_snr = None
        self.last_rx_delay = 0.0

        # Timeline logger
        self.timeline = []
        self.log_state("INIT", time=0.0)

    # ============================================================
    # Timeline Logger
    # ============================================================
    def log_state(self, event, env=None, time=None):
        now = env.now if env is not None else time
        self.timeline.append({
            "node": self.id,
            "time": now,
            "state": self.state.name,
            "event": event
        })

    # ============================================================
    # Utility
    # ============================================================
    def distance_to(self, other):
        dx = self.loc.x - other.loc.x
        dy = self.loc.y - other.loc.y
        return (dx * dx + dy * dy) ** 0.5

    # ============================================================
    # SELECT CHANNEL
    # ============================================================
    def select_channel(self, now):
        if isinstance(self.channel_mode, int):
            return self.channel_mode, self.ch_plan.get_freq(self.channel_mode)

        best_ch = None
        best_score = 1e18

        for ch in self.ch_plan.channels:
            ch_id = ch["id"]
            score = self.busy_map[ch_id]
            if best_ch is None or score < best_score:
                best_ch = ch_id
                best_score = score

        return best_ch, self.ch_plan.get_freq(best_ch)

    # ============================================================
    # RADIO TRANSITIONS
    # ============================================================
    def radio_to_tx(self, env, sf, bw_hz, ptx, new_freq_mhz):
        def apply_energy_and_timeout(t_ms, current_mA):
            if t_ms <= 0:
                return
            t_s = t_ms / 1000.0
            observed_s = observed_duration_s(env, t_s)
            self.total_transition_time_s += observed_s

            # Radio transition energy
            e_radio = (
                self.hw.voltage
                * (current_mA / 1000.0)
                * observed_s
                * 1000.0
            )
            self.energy_transition_mJ += e_radio

            # Optional MCU energy
            if self.include_mcu_energy and self.mcu is not None:
                self.energy_transition_mJ += self.mcu.energy_active(observed_s)

            yield env.timeout(t_s)

        if self.state == NodeState.SLEEP:
            t, i = self.hw.sleep_to_standby()
            yield from apply_energy_and_timeout(t, i)
            self.state = NodeState.STANDBY
            self.log_state("SLEEP_TO_STANDBY", env)

        if self.state == NodeState.STANDBY:
            t, i = self.hw.standby_to_fs()
            yield from apply_energy_and_timeout(t, i)
            self.state = NodeState.FS
            self.log_state("STANDBY_TO_FS", env)

        if self.state == NodeState.FS:
            t, i_trans, i_tx = self.hw.fs_to_tx(sf, bw_hz, ptx)
            yield from apply_energy_and_timeout(t, i_trans)
            self.state = NodeState.TX
            self.tx_power = ptx
            self.tx_current_mA = float(i_tx)
            self.freq = new_freq_mhz
            self.log_state("FS_TO_TX", env)
            return

        if self.state == NodeState.RX:
            t, i_trans, i_tx = self.hw.rx_to_tx(self.freq, new_freq_mhz, sf, bw_hz, ptx)
            yield from apply_energy_and_timeout(t, i_trans)
            self.state = NodeState.TX
            self.tx_power = ptx
            self.tx_current_mA = float(i_tx)
            self.freq = new_freq_mhz
            self.log_state("RX_TO_TX", env)
            return

        if self.state == NodeState.TX:
            t, i_trans, i_tx = self.hw.tx_to_tx(self.freq, new_freq_mhz, sf, bw_hz, ptx)
            yield from apply_energy_and_timeout(t, i_trans)
            self.state = NodeState.TX
            self.tx_power = ptx
            self.tx_current_mA = float(i_tx)
            self.freq = new_freq_mhz
            self.log_state("TX_TO_TX", env)
            return

        raise RuntimeError("Invalid state in radio_to_tx()")

    # ============================================================
    # RX WINDOW ENERGY ACCOUNTING
    # ============================================================
    def _account_rx_energy_s(self, duration_s, reason="RX_WINDOW",time=None):
        """
        Account node RX energy for Class-A-like RX windows.

        Energy:
            E_rx = V * I_rx * t_rx

        Notes:
        - duration_s is in seconds.
        - energy is accumulated in mJ, consistent with existing code.
        - total_rx_time_s is used by Statistics.get_energy_breakdown_df().
        """
        t_s = max(float(duration_s), 0.0)

        if t_s <= 0:
            return

        i_rx_mA = float(getattr(self.hw, "rx_current", 0.0))
        v = float(getattr(self.hw, "voltage", 0.0))

        # Radio RX energy in mJ
        self.energy_rx_mJ += v * (i_rx_mA / 1000.0) * t_s * 1000.0

        # Optional MCU active energy while radio RX is active
        if self.include_mcu_energy and self.mcu is not None:
            self.energy_rx_mJ += self.mcu.energy_active(t_s)

        self.total_rx_time_s += t_s

        self.timeline.append({
            "node": self.id,
            "time": time,
            "state": "RX",
            "event": f"{reason}_ENERGY_ACCOUNTED",
            "duration_s": t_s,
        })


    def _adjust_rx_energy_s(self, duration_delta_s, reason="RX_WINDOW_CORRECTION"):
        """Apply a signed correction to RX time and its linear energy model."""
        delta_s = float(duration_delta_s)
        if delta_s == 0.0:
            return

        i_rx_mA = float(getattr(self.hw, "rx_current", 0.0))
        voltage = float(getattr(self.hw, "voltage", 0.0))
        energy_delta_mJ = voltage * (i_rx_mA / 1000.0) * delta_s * 1000.0
        if self.include_mcu_energy and self.mcu is not None:
            energy_delta_mJ += self.mcu.energy_active(delta_s)

        self.energy_rx_mJ = max(0.0, self.energy_rx_mJ + energy_delta_mJ)
        self.total_rx_time_s = max(0.0, self.total_rx_time_s + delta_s)
        self.timeline.append({
            "node": self.id,
            "time": None,
            "state": "RX",
            "event": reason,
            "duration_delta_s": delta_s,
        })


    def open_rx_window(
        self,
        env,
        rx_start_s,
        rx_end_s,
        ch_id=None,
        freq_mhz=None,
        bw_hz=None,
        sf=None,
        label="RX1",
    ):
        """
        Open a Class-A-like RX window and account actual RX time/energy.

        Behavior:
        - If no downlink is received, RX energy = full window duration.
        - If a valid DATA_DOWN is decoded and NodeRX calls close_rx_window(),
          RX energy = rx_start -> close_time.
        - If a preamble/frame is detected and NodeRX calls extend_rx_window_until(),
          the RX window can be extended until the frame end, LoRaWAN-like.
        """
        rx_start_s = float(rx_start_s)
        rx_end_s = float(rx_end_s)

        if rx_end_s <= rx_start_s:
            return None

        if not hasattr(self, "rx_windows"):
            self.rx_windows = []

        # Remove expired windows
        now = float(env.now)
        cleaned = []
        for w in self.rx_windows:
            if isinstance(w, dict):
                if float(w.get("end", 0.0)) >= now and not bool(w.get("closed", False)):
                    cleaned.append(w)
            else:
                try:
                    s, e = w
                    if float(e) >= now:
                        cleaned.append(w)
                except Exception:
                    pass

        self.rx_windows = cleaned

        # Use dict window to support early close and extension.
        close_event = env.event()
        window = {
            "label": label,
            "start": rx_start_s,
            "end": rx_end_s,
            "ch_id": ch_id,
            "freq_mhz": freq_mhz,
            "bw_hz": bw_hz,
            "sf": sf,
            "closed": False,
            "close_reason": None,
            "close_time": None,
            "event": close_event,
        }

        self.rx_windows.append(window)

        # Wait until RX start
        if env.now < rx_start_s:
            yield env.timeout(rx_start_s - env.now)

        # If expired before opening
        if env.now >= float(window["end"]):
            self.rx_windows = [w for w in self.rx_windows if w is not window]
            return window

        # Open radio RX
        self.state = NodeState.RX
        self.current_rx_window = window
        self.rx_busy_until = float(window["end"])
        self.rx_window_count += 1

        self.timeline.append({
            "node": self.id,
            "time": env.now,
            "state": "RX",
            "event": f"{label}_OPEN",
            "rx_start": rx_start_s,
            "rx_end": float(window["end"]),
        })

        # Account the planned RX interval up front. This is necessary because
        # SimPy does not resume this generator at a hard numeric run boundary.
        account_start = float(env.now)
        planned_rx_s = observed_duration_s(
            env,
            float(window["end"]) - account_start,
            start_time=account_start,
        )
        window["energy_account_start"] = account_start
        window["energy_accounted_until"] = account_start + planned_rx_s
        self._account_rx_energy_s(
            planned_rx_s,
            reason=f"{label}_PLANNED",
            time=env.now,
        )

        # Wait until timeout or early close.
        # This loop supports extension by NodeRX.
        while True:
            if bool(window.get("closed", False)):
                break

            remaining_s = float(window.get("end", rx_end_s)) - float(env.now)

            if remaining_s <= 0:
                break

            result = yield close_event | env.timeout(remaining_s)

            if close_event in result:
                break

            # If timeout fired but NodeRX extended the end time,
            # loop continues and waits the additional duration.
            if float(env.now) >= float(window.get("end", rx_end_s)):
                break

        close_time = float(window.get("close_time") or env.now)
        actual_rx_s = max(0.0, close_time - rx_start_s)
        horizon = getattr(env, "simulation_end", None)
        actual_account_end = close_time
        if horizon is not None:
            actual_account_end = min(actual_account_end, float(horizon))
        actual_account_s = max(0.0, actual_account_end - account_start)
        accounted_until = float(
            window.get("energy_accounted_until", account_start + planned_rx_s)
        )
        accounted_s = max(0.0, accounted_until - account_start)
        correction_s = actual_account_s - accounted_s
        self._adjust_rx_energy_s(
            correction_s,
            reason=f"{label}_ENERGY_CORRECTION",
        )

        if bool(window.get("closed", False)):
            self.rx_window_downlink_count += 1
            if close_time < rx_end_s:
                self.rx_window_early_close_count += 1
        else:
            self.rx_window_timeout_count += 1

        self.state = NodeState.SLEEP
        self.current_rx_window = None
        self.rx_busy_until = 0.0

        self.timeline.append({
            "node": self.id,
            "time": env.now,
            "state": "SLEEP",
            "event": f"{label}_CLOSE",
            "actual_rx_s": actual_rx_s,
            "close_reason": window.get("close_reason", "timeout"),
        })

        # Remove this window
        self.rx_windows = [w for w in self.rx_windows if w is not window]

        return window


    def extend_rx_window_until(self, env, hold_until_s, reason="PREAMBLE_DETECTED"):
        """
        Extend the active RX window until a detected frame is fully received.
        This follows the LoRaWAN idea: if preamble is detected, stay active
        until the downlink frame is demodulated.
        """
        hold_until_s = float(hold_until_s)
        now = float(env.now)

        candidates = []

        if self.current_rx_window is not None:
            candidates.append(self.current_rx_window)

        for w in getattr(self, "rx_windows", []):
            if isinstance(w, dict) and w not in candidates:
                candidates.append(w)

        for w in candidates:
            if not isinstance(w, dict):
                continue

            if bool(w.get("closed", False)):
                continue

            w_start = float(w.get("start", 0.0))
            w_end = float(w.get("end", 0.0))

            if w_start <= now <= w_end:
                if hold_until_s > w_end:
                    w["end"] = hold_until_s
                    self.rx_busy_until = hold_until_s

                    accounted_until = w.get("energy_accounted_until", None)
                    if accounted_until is not None:
                        new_accounted_until = hold_until_s
                        horizon = getattr(env, "simulation_end", None)
                        if horizon is not None:
                            new_accounted_until = min(
                                new_accounted_until,
                                float(horizon),
                            )
                        extension_s = max(
                            0.0,
                            new_accounted_until - float(accounted_until),
                        )
                        self._account_rx_energy_s(
                            extension_s,
                            reason=f"{w.get('label', 'RX')}_EXTENSION",
                            time=env.now,
                        )
                        w["energy_accounted_until"] = (
                            float(accounted_until) + extension_s
                        )

                    self.timeline.append({
                        "node": self.id,
                        "time": env.now,
                        "state": self.state.name,
                        "event": f"{w.get('label', 'RX')}_EXTEND",
                        "reason": reason,
                        "new_rx_end": hold_until_s,
                    })

                return True

        return False


    def close_rx_window(self, env, reason="DOWNLINK_DECODED"):
        """
        Close current RX window early after a valid DATA_DOWN has been decoded.
        """
        w = self.current_rx_window

        if w is None or not isinstance(w, dict):
            return False

        if bool(w.get("closed", False)):
            return False

        w["closed"] = True
        w["close_reason"] = reason
        w["close_time"] = float(env.now)

        ev = w.get("event", None)

        if ev is not None and not ev.triggered:
            ev.succeed(reason)

        return True



    
    # ============================================================
    # TX EVENT
    # ============================================================
    def tx_event(self, env, packet, model, nexthop, terrain_delta_h):

        # Duty-cycle gate.
        # NOTE: Node duty-cycle update is currently disabled because
        # NODE_PERIOD is assumed safe for the current scenario.       
        # Duty-cycle
        if env.now < self.next_tx_allowed:
            yield env.timeout(self.next_tx_allowed - env.now)

        # Channel already selected in TXGenerator
        freq = packet.freq_mhz
        sf = packet.sf
        bw_hz = packet.bw_hz
        ptx = validate_ptx(self, packet.ptx_dbm, label=self.id)
        packet.ptx_dbm = ptx

        # Radio → TX
        yield from self.radio_to_tx(env, sf, bw_hz, ptx, freq)

        # Time on Air
        toa = packet.toa
        packet.tx_start = env.now
        packet.tx_end = packet.tx_start + toa
        observed_toa = observed_duration_s(
            env,
            toa,
            start_time=packet.tx_start,
        )

        self.is_transmitting = True
        self.tx_busy_until = packet.tx_end

        # TX airtime accounting
        self.total_tx_time_s += observed_toa
        
        ch_id = getattr(packet, "ch_id", None)
        if ch_id is not None:
            ch_id = int(ch_id)
        
            self.tx_airtime_per_channel[ch_id] = (
                self.tx_airtime_per_channel.get(ch_id, 0.0)
                + observed_toa
            )
        
            self.tx_packet_count_per_channel[ch_id] = (
                self.tx_packet_count_per_channel.get(ch_id, 0)
                + 1
            )

        # Record TX event log
        if hasattr(self, "stats") and self.stats is not None:
            self.stats.record_tx(packet, node_id=self.id)

        # Record original node TX count only for initial injection
        if packet.hop_index == 0:
            if hasattr(self, "stats") and self.stats is not None:
                self.stats.record_node_tx(packet.src_id)

        # Busy map
        self.busy_map[packet.ch_id] = packet.tx_end

        # TX energy (radio)
        i_tx_a = self.hw.tx_current[ptx] / 1000.0
        self.energy_tx_mJ += self.hw.voltage * i_tx_a * observed_toa * 1000.0

        # TX energy (optional MCU)
        if self.include_mcu_energy and self.mcu is not None:
            self.energy_tx_mJ += self.mcu.energy_active(observed_toa)

        # Send the packet over the air when TX starts.
        self.air.transmit(packet)

        # Keep the node in TX state for the full airtime.
        yield env.timeout(toa)
        self.log_state("TX_DONE", env)
        
        self.is_transmitting = False
        self.tx_busy_until = env.now
        self.tx_power = None
        self.tx_current_mA = None
        self.state = NodeState.SLEEP
        
        return packet


    # ============================================================
    # SLEEP EVENT
    # ============================================================
    def sleep_event(self, env, duration_s):
        if duration_s <= 0:
            return

        requested_s = float(duration_s)
        t_s = observed_duration_s(env, requested_s)

        # Robust fallback for sleep current naming
        if hasattr(self.hw, "sleep_current"):
            sleep_current_mA = self.hw.sleep_current
        elif hasattr(self.hw, "txsleep_current"):
            sleep_current_mA = self.hw.txsleep_current
        else:
            sleep_current_mA = 0.001  # fallback 1 µA

        # Radio sleep energy
        i_a = sleep_current_mA / 1000.0
        e = self.hw.voltage * i_a * t_s * 1000.0
        self.energy_sleep_mJ += e

        # Optional MCU sleep energy
        if self.include_mcu_energy and self.mcu is not None:
            self.energy_sleep_mJ += self.mcu.energy_sleep(t_s)

        # Update state & stats
        self.state = NodeState.SLEEP
        self.last_sleep_duration_s = t_s
        self.total_sleep_time_s += t_s

        # Timeline log
        self.timeline.append({
            "node": self.id,
            "time": env.now,
            "state": "SLEEP",
            "event": "SLEEP_WINDOW"
        })

        yield env.timeout(requested_s)

    # ============================================================
    # REPORT SUMMARY
    # ============================================================
    def report(self):
        return {
            "node": self.id,
            "mcu_energy": self.include_mcu_energy,
            "tx_energy_mJ": self.energy_tx_mJ,
            "rx_energy_mJ": self.energy_rx_mJ,
            "sleep_energy_mJ": self.energy_sleep_mJ,
            "idle_energy_mJ": self.energy_idle_mJ,
            "transition_energy_mJ": self.energy_transition_mJ,
            "cad_energy_mJ": self.energy_cad_mJ,
            "total_energy_mJ": (
                self.energy_tx_mJ
                + self.energy_rx_mJ
                + self.energy_sleep_mJ
                + self.energy_idle_mJ
                + self.energy_transition_mJ
                + self.energy_cad_mJ
            ),
            "last_rssi": self.last_rssi,
            "last_snr": self.last_snr,
        }
