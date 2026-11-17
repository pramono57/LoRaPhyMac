# ============================================================
# runtime/demodulator_pool.py
# SX1302-aware non-blocking demodulator pool
#
# Fixes:
# - Pool A eligibility: SF5-SF12
# - Pool B eligibility: SF5-SF10
# - Packet-level ACQUIRE / REJECT / RELEASE occupancy logging
# - Peak occupancy tracking per pool and total
# ============================================================

from __future__ import annotations

from typing import Any, Optional

from loraphymac.runtime.packet import stable_packet_object_id


class DemodulatorPool:
    """Logical SX1302 multi-SF modem pool.

    Pool A
        ``capacity_a`` modems eligible for SF5-SF12.

    Pool B
        ``capacity_b`` modems eligible for SF5-SF10.

    RelayRX acquires a modem at the physical ``rx_start`` and schedules its
    release at the absolute ``rx_end``. The optional absolute deadline is
    captured when release is scheduled, so forwarding cannot move the event.
    """

    POOL_A_MIN_SF = 5
    POOL_A_MAX_SF = 12
    POOL_B_MIN_SF = 5
    POOL_B_MAX_SF = 10

    def __init__(
        self,
        env,
        pool_a_size: int = 8,
        pool_b_size: int = 8,
        receiver_id: Optional[str] = None,
        statistics: Any = None,
    ):
        self.env = env
        self.receiver_id = receiver_id
        self.stats = statistics

        self.capacity_a = max(0, int(pool_a_size))
        self.capacity_b = max(0, int(pool_b_size))

        self.used_a = 0
        self.used_b = 0

        # packet object id -> assigned pool
        self.packet_pool: dict[int, str] = {}

        # Immutable RX context captured at successful acquisition.
        # This prevents RELEASE events from reading packet fields that may
        # already have been mutated for forwarding to the next hop.
        self.packet_context: dict[int, dict[str, Any]] = {}

        # Local fallback/debug log. The same event is also forwarded to
        # Statistics.record_demodulator_event() when available.
        self.event_log: list[dict[str, Any]] = []

        self.acquire_attempt_count = 0
        self.acquire_success_count = 0
        self.release_count = 0
        self.demod_busy_count = 0
        self.unsupported_sf_count = 0

        self.peak_used_a = 0
        self.peak_used_b = 0
        self.peak_used_total = 0

    # ============================================================
    # Helpers
    # ============================================================
    @staticmethod
    def _packet_sf(packet) -> Optional[int]:
        value = getattr(packet, "sf", None)
        if value is None:
            value = getattr(packet, "spreading_factor", None)
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _eligible_for_a(cls, sf: int) -> bool:
        return cls.POOL_A_MIN_SF <= int(sf) <= cls.POOL_A_MAX_SF

    @classmethod
    def _eligible_for_b(cls, sf: int) -> bool:
        return cls.POOL_B_MIN_SF <= int(sf) <= cls.POOL_B_MAX_SF

    def _update_peaks(self) -> None:
        self.peak_used_a = max(self.peak_used_a, self.used_a)
        self.peak_used_b = max(self.peak_used_b, self.used_b)
        self.peak_used_total = max(
            self.peak_used_total,
            self.used_a + self.used_b,
        )

    def _snapshot_packet_context(self, packet) -> dict[str, Any]:
        """Capture immutable RX metadata at successful modem assignment."""
        return {
            "packet_uid": getattr(packet, "packet_uid", None),
            "src_id": getattr(packet, "src_id", None),
            "current_tx": getattr(packet, "current_tx", None),
            "prev_hop": getattr(packet, "prev_hop", None),
            "next_hop": getattr(packet, "next_hop", None),
            "mtype": getattr(packet, "mtype", None),
            "hop_index": getattr(packet, "hop_index", None),
            "ch_id": getattr(packet, "ch_id", None),
            "sf": self._packet_sf(packet),
            "bw_hz": getattr(packet, "bw_hz", None),
            "toa_s": getattr(packet, "toa", None),
            "rx_start": getattr(packet, "rx_start", None),
            "rx_end": getattr(packet, "rx_end", None),
        }

    def _record_event(
        self,
        *,
        event: str,
        packet,
        pool: Optional[str],
        reason: Optional[str],
        used_a_before: int,
        used_b_before: int,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        ctx = context if context is not None else self._snapshot_packet_context(packet)

        row = {
            "time": float(self.env.now),
            "receiver_id": self.receiver_id,
            "event": str(event),
            "pool": pool,
            "reason": reason,
            "packet_object_id": stable_packet_object_id(packet),
            "packet_uid": ctx.get("packet_uid"),
            "src_id": ctx.get("src_id"),
            "current_tx": ctx.get("current_tx"),
            "prev_hop": ctx.get("prev_hop"),
            "next_hop": ctx.get("next_hop"),
            "mtype": ctx.get("mtype"),
            "hop_index": ctx.get("hop_index"),
            "ch_id": ctx.get("ch_id"),
            "sf": ctx.get("sf"),
            "bw_hz": ctx.get("bw_hz"),
            "toa_s": ctx.get("toa_s"),
            "rx_start": ctx.get("rx_start"),
            "rx_end": ctx.get("rx_end"),
            "used_a_before": int(used_a_before),
            "used_b_before": int(used_b_before),
            "used_total_before": int(used_a_before + used_b_before),
            "used_a_after": int(self.used_a),
            "used_b_after": int(self.used_b),
            "used_total_after": int(self.used_a + self.used_b),
            "capacity_a": int(self.capacity_a),
            "capacity_b": int(self.capacity_b),
            "capacity_total": int(self.capacity_a + self.capacity_b),
            "peak_used_a": int(self.peak_used_a),
            "peak_used_b": int(self.peak_used_b),
            "peak_used_total": int(self.peak_used_total),
        }
        self.event_log.append(row)

        recorder = getattr(self.stats, "record_demodulator_event", None)
        if callable(recorder):
            recorder(**row)

    # ============================================================
    # Try acquire (non-blocking)
    # ============================================================
    def try_acquire(self, packet) -> bool:
        """Attempt an immediate SX1302 modem assignment.

        Assignment policy preserves the SF5-SF12 pool for high-SF traffic:

        - SF5-SF10: Pool B first, then Pool A as fallback
        - SF11-SF12: Pool A only
        - Other/missing SF: rejected as ``demod_unsupported_sf``
        """
        packet_key = id(packet)
        used_a_before = self.used_a
        used_b_before = self.used_b
        self.acquire_attempt_count += 1

        # Defensive guard against a duplicate acquisition of the same object.
        if packet_key in self.packet_pool:
            packet.demod_reason = "demod_duplicate_acquire"
            self._record_event(
                event="REJECT",
                packet=packet,
                pool=self.packet_pool[packet_key],
                reason=packet.demod_reason,
                used_a_before=used_a_before,
                used_b_before=used_b_before,
            )
            return False

        sf = self._packet_sf(packet)
        if sf is None or not self._eligible_for_a(sf):
            self.unsupported_sf_count += 1
            packet.demod_reason = "demod_unsupported_sf"
            self._record_event(
                event="REJECT",
                packet=packet,
                pool=None,
                reason=packet.demod_reason,
                used_a_before=used_a_before,
                used_b_before=used_b_before,
            )
            return False

        selected_pool: Optional[str] = None

        # SF5-SF10: use the specialised low-SF pool first, preserving
        # Pool A capacity for SF11-SF12. Pool A remains a fallback.
        if self._eligible_for_b(sf):
            if self.used_b < self.capacity_b:
                self.used_b += 1
                selected_pool = "B"
            elif self.used_a < self.capacity_a:
                self.used_a += 1
                selected_pool = "A"

        # SF11-SF12: only Pool A is eligible.
        elif self.used_a < self.capacity_a:
            self.used_a += 1
            selected_pool = "A"

        if selected_pool is None:
            self.demod_busy_count += 1
            packet.demod_reason = "demod_busy"
            self._record_event(
                event="REJECT",
                packet=packet,
                pool=None,
                reason=packet.demod_reason,
                used_a_before=used_a_before,
                used_b_before=used_b_before,
            )
            return False

        self.packet_pool[packet_key] = selected_pool
        self.packet_context[packet_key] = self._snapshot_packet_context(packet)

        packet.demod_pool = selected_pool
        packet.demod_acquire_time = float(self.env.now)
        packet.demod_reason = None

        self.acquire_success_count += 1
        self._update_peaks()
        self._record_event(
            event="ACQUIRE",
            packet=packet,
            pool=selected_pool,
            reason=None,
            used_a_before=used_a_before,
            used_b_before=used_b_before,
            context=self.packet_context[packet_key],
        )
        return True

    # ============================================================
    # Release at the physical RX end
    # ============================================================
    def schedule_release(self, packet, release_at: Optional[float] = None) -> None:
        """Schedule release, preferably at an explicit absolute RX deadline.

        ``release_at`` is the paper-grade path used by RelayRX. When omitted,
        the historical ``env.now + packet.toa`` behavior is retained for
        direct callers that do not carry receiver timing metadata.
        """
        if release_at is None:
            try:
                release_at = float(self.env.now) + max(
                    0.0,
                    float(getattr(packet, "toa", 0.0) or 0.0),
                )
            except (TypeError, ValueError):
                release_at = float(self.env.now)
        else:
            release_at = float(release_at)

        self.env.process(self._release_at_deadline(packet, release_at))

    def _release_at_deadline(self, packet, release_at: float):
        hold_s = max(0.0, float(release_at) - float(self.env.now))

        if hold_s > 0.0:
            yield self.env.timeout(hold_s)

        used_a_before = self.used_a
        used_b_before = self.used_b
        packet_key = id(packet)
        pool = self.packet_pool.pop(packet_key, None)
        context = self.packet_context.pop(packet_key, None)

        if pool == "A":
            self.used_a = max(0, self.used_a - 1)
        elif pool == "B":
            self.used_b = max(0, self.used_b - 1)
        else:
            # A missing mapping is logged instead of silently corrupting usage.
            self._record_event(
                event="RELEASE_MISSING",
                packet=packet,
                pool=None,
                reason="demod_release_without_acquire",
                used_a_before=used_a_before,
                used_b_before=used_b_before,
                context=context,
            )
            return

        packet.demod_release_time = float(self.env.now)
        self.release_count += 1
        self._record_event(
            event="RELEASE",
            packet=packet,
            pool=pool,
            reason=None,
            used_a_before=used_a_before,
            used_b_before=used_b_before,
            context=context,
        )

    # ============================================================
    # Direct inspection helpers
    # ============================================================
    def get_event_log(self) -> list[dict[str, Any]]:
        return list(self.event_log)

    def get_summary(self) -> dict[str, Any]:
        return {
            "receiver_id": self.receiver_id,
            "timing_model": "rx_start_to_rx_end_v2",
            "capacity_a": self.capacity_a,
            "capacity_b": self.capacity_b,
            "capacity_total": self.capacity_a + self.capacity_b,
            "pool_a_sf_range": "SF5-SF12",
            "pool_b_sf_range": "SF5-SF10",
            "acquire_attempt_count": self.acquire_attempt_count,
            "acquire_success_count": self.acquire_success_count,
            "release_count": self.release_count,
            "demod_busy_count": self.demod_busy_count,
            "unsupported_sf_count": self.unsupported_sf_count,
            "peak_used_a": self.peak_used_a,
            "peak_used_b": self.peak_used_b,
            "peak_used_total": self.peak_used_total,
            "used_a_at_end": self.used_a,
            "used_b_at_end": self.used_b,
            "active_mapping_count_at_end": len(self.packet_pool),
            "active_context_count_at_end": len(self.packet_context),
        }
