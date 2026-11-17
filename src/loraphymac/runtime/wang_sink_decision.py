"""Main-gateway-only observation windows for Wang A2-only decisions.

This module consumes only reports physically accepted by the gateway and
creates immutable decision records.  An optional W2C2B transport callback may
turn a record into a physical command; this decision engine never mutates a
radio entity's power state itself.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from loraphymac.runtime.wang_atp_a2only import (
    WANG_ATP_SF7_A2ONLY_MODEL_ID,
    WangA2PowerReductionController,
)
from loraphymac.runtime.wang_snr_reporting import (
    WangGatewayReceivedReport,
    freeze_wang_routing_map,
)


WANG_OBSERVATION_WINDOW_S = 300.0
WANG_DELIVERY_GRACE_S = 10.0
WANG_REQUIRED_COVERAGE_FRACTION = 1.0
WANG_MIN_REPORTS_PER_TRANSMITTER = 1


def wang_runtime_enabled(config) -> bool:
    """Use exact model identity as the fail-closed Wang runtime gate."""
    return (
        str(config.get("tx_power_model_id", "")).strip().lower()
        == WANG_ATP_SF7_A2ONLY_MODEL_ID
    )


def build_active_transmitter_population(nodes, relays, routing_map):
    """Build the active set from configured routes, never successful reports."""
    configured_parents = {
        str(parent)
        for parent in routing_map.values()
        if parent not in (None, "")
    }
    active = []
    for node in nodes:
        if routing_map.get(node.id) in (None, ""):
            continue
        expected_ptx = int(node.ptx_max)
        if int(node.default_ptx) != expected_ptx:
            raise ValueError(
                f"{node.id}: Wang initial PTx must equal resolved ptx_max"
            )
        active.append(
            WangActiveTransmitter(
                entity_id=str(node.id),
                entity_type="node",
                expected_ptx_dbm=expected_ptx,
            )
        )
    for relay in relays:
        if str(relay.id) not in configured_parents:
            continue
        if routing_map.get(relay.id) in (None, ""):
            continue
        expected_ptx = int(relay.ptx_max)
        if int(relay.default_ptx) != expected_ptx:
            raise ValueError(
                f"{relay.id}: Wang initial PTx must equal resolved ptx_max"
            )
        active.append(
            WangActiveTransmitter(
                entity_id=str(relay.id),
                entity_type="relay",
                expected_ptx_dbm=expected_ptx,
            )
        )
    return tuple(sorted(active, key=lambda item: item.entity_id))


def last_closable_window_index(simulation_horizon_s: float) -> int:
    """Return the last cutoff strictly before a numeric SimPy horizon."""
    horizon = float(simulation_horizon_s)
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("simulation horizon must be finite and > 0")
    ratio = (horizon - WANG_DELIVERY_GRACE_S) / WANG_OBSERVATION_WINDOW_S
    return max(-1, math.ceil(ratio) - 2)


def install_wang_gateway_scheduler(
    *,
    env,
    gateway,
    nodes,
    relays,
    routing_map,
    simulation_horizon_s: float,
):
    """Install reporting and the decision cadence on the main gateway only."""
    if str(gateway.id) != "GW":
        raise ValueError("Wang scheduler requires main gateway GW")
    active = build_active_transmitter_population(nodes, relays, routing_map)
    engine = WangGatewaySinkDecisionEngine(active, gateway_id=str(gateway.id))
    immutable_routing_map = freeze_wang_routing_map(routing_map)
    for participant in [*nodes, *relays, gateway]:
        participant.wang_snr_reporting_enabled = True
        participant.wang_routing_map = immutable_routing_map
    for transmitter in [*nodes, *relays]:
        if not hasattr(transmitter, "wang_last_applied_command_seq"):
            transmitter.wang_last_applied_command_seq = 0
    gateway.wang_sink_decision_engine = engine
    last_index = last_closable_window_index(simulation_horizon_s)
    if last_index >= 0:
        env.process(engine.run(env, last_window_index=last_index))
    return engine


@dataclass(frozen=True, slots=True)
class WangActiveTransmitter:
    """Predeclared sink-side expectation for one active transmitter."""

    entity_id: str
    entity_type: str
    expected_ptx_dbm: int
    ptx_epoch: int = 0

    def __post_init__(self) -> None:
        if not str(self.entity_id):
            raise ValueError("active transmitter ID must be non-empty")
        bounds = WangA2PowerReductionController.bounds_for(self.entity_type)
        if int(self.expected_ptx_dbm) != self.expected_ptx_dbm:
            raise ValueError("expected PTx must be an integer dBm value")
        if not bounds.minimum_dbm <= int(self.expected_ptx_dbm) <= bounds.maximum_dbm:
            raise ValueError(
                f"expected PTx for {self.entity_id} is outside "
                f"[{bounds.minimum_dbm}, {bounds.maximum_dbm}] dBm"
            )
        if int(self.ptx_epoch) < 0:
            raise ValueError("PTx epoch must be non-negative")


@dataclass(frozen=True, slots=True)
class WangEntityDecisionRecord:
    """One hardware-clamped target derived from the common network Nstep."""

    entity_id: str
    entity_type: str
    ptx_epoch: int
    current_ptx_dbm: int
    expected_ptx_dbm: int
    target_ptx_dbm: int
    report_count: int
    source_report_ids: tuple[str, ...]
    transmitter_mean_snr_db: float
    network_reduction_steps: int
    reason: str


@dataclass(frozen=True, slots=True)
class WangWindowDecisionRecord:
    """Exactly one auditable decision/no-decision record for one window."""

    decision_id: str
    controller_id: str
    window_index: int
    window_start_s: float
    window_end_s: float
    delivery_grace_s: float
    decision_cutoff_s: float
    decision_time_s: float
    status: str
    reason: str
    active_transmitter_count: int
    covered_transmitter_count: int
    coverage_fraction: float
    required_coverage_fraction: float
    minimum_reports_per_transmitter: int
    active_transmitter_ids: tuple[str, ...]
    covered_transmitter_ids: tuple[str, ...]
    missing_transmitter_ids: tuple[str, ...]
    source_report_ids: tuple[str, ...]
    rejection_counts: tuple[tuple[str, int], ...]
    network_snr_avg_db: float | None
    network_snr_mad_db: float | None
    snr_required_db: float | None
    network_margin_db: float | None
    network_reduction_steps: int | None
    entity_decisions: tuple[WangEntityDecisionRecord, ...]


class WangGatewaySinkDecisionEngine:
    """Window and invoke W1 only at main GW; application remains physical."""

    def __init__(
        self,
        active_transmitters,
        *,
        gateway_id: str = "GW",
        window_s: float = WANG_OBSERVATION_WINDOW_S,
        delivery_grace_s: float = WANG_DELIVERY_GRACE_S,
    ) -> None:
        self.gateway_id = str(gateway_id)
        if self.gateway_id != "GW":
            raise ValueError("Wang A2 sink decisions are restricted to main gateway GW")
        self.window_s = float(window_s)
        self.delivery_grace_s = float(delivery_grace_s)
        if not math.isfinite(self.window_s) or self.window_s <= 0:
            raise ValueError("observation window must be finite and > 0")
        if not math.isfinite(self.delivery_grace_s) or self.delivery_grace_s < 0:
            raise ValueError("delivery grace must be finite and >= 0")

        active = tuple(active_transmitters)
        if not active:
            raise ValueError("active transmitter population must be predeclared")
        self._active = {}
        for item in active:
            if not isinstance(item, WangActiveTransmitter):
                raise TypeError("active population must contain WangActiveTransmitter")
            if item.entity_id in self._active:
                raise ValueError(f"duplicate active transmitter {item.entity_id!r}")
            self._active[item.entity_id] = item

        self._pending = []
        self._seen_report_ids = set()
        self._rejections = defaultdict(list)
        self._decisions = {}
        self.command_coordinator = None

    @property
    def active_transmitters(self) -> tuple[WangActiveTransmitter, ...]:
        return tuple(self._active[item] for item in sorted(self._active))

    def attach_command_coordinator(self, coordinator) -> None:
        """Attach physical transport after runtime TX/RX objects exist."""
        if self.command_coordinator is not None:
            raise RuntimeError("Wang command coordinator is already attached")
        self.command_coordinator = coordinator

    @property
    def decisions(self) -> tuple[WangWindowDecisionRecord, ...]:
        return tuple(self._decisions[index] for index in sorted(self._decisions))

    @property
    def no_decision_count(self) -> int:
        return sum(record.status == "no_decision" for record in self._decisions.values())

    @property
    def active_transmitter_count(self) -> int:
        return len(self._active)

    def decision_rows(self) -> list[dict]:
        """Flatten immutable records for audit/export without becoming input."""
        rows = []
        for record in self.decisions:
            common = {
                "decision_id": record.decision_id,
                "controller_id": record.controller_id,
                "window_index": record.window_index,
                "window_start_s": record.window_start_s,
                "window_end_s": record.window_end_s,
                "delivery_grace_s": record.delivery_grace_s,
                "decision_cutoff_s": record.decision_cutoff_s,
                "decision_time_s": record.decision_time_s,
                "status": record.status,
                "reason": record.reason,
                "active_transmitter_count": record.active_transmitter_count,
                "covered_transmitter_count": record.covered_transmitter_count,
                "coverage_fraction": record.coverage_fraction,
                "required_coverage_fraction": record.required_coverage_fraction,
                "missing_transmitter_ids": record.missing_transmitter_ids,
                "source_report_ids": record.source_report_ids,
                "rejection_counts": record.rejection_counts,
                "network_snr_avg_db": record.network_snr_avg_db,
                "network_snr_mad_db": record.network_snr_mad_db,
                "snr_required_db": record.snr_required_db,
                "network_margin_db": record.network_margin_db,
                "network_reduction_steps": record.network_reduction_steps,
            }
            if not record.entity_decisions:
                rows.append(common)
                continue
            for entity in record.entity_decisions:
                rows.append(
                    {
                        **common,
                        "entity_id": entity.entity_id,
                        "entity_type": entity.entity_type,
                        "ptx_epoch": entity.ptx_epoch,
                        "current_ptx_dbm": entity.current_ptx_dbm,
                        "expected_ptx_dbm": entity.expected_ptx_dbm,
                        "target_ptx_dbm": entity.target_ptx_dbm,
                        "report_count": entity.report_count,
                        "entity_source_report_ids": entity.source_report_ids,
                        "transmitter_mean_snr_db": entity.transmitter_mean_snr_db,
                        "entity_reason": entity.reason,
                    }
                )
        return rows

    def _window_index(self, observation_time_s: float) -> int:
        return math.floor(float(observation_time_s) / self.window_s)

    def _reject(self, window_index: int, report_id: str | None, reason: str) -> None:
        self._rejections[int(window_index)].append((report_id, str(reason)))

    def accept_observation(self, received: WangGatewayReceivedReport) -> bool:
        """Accept one gateway-delivered envelope; no decision is made here."""
        if not isinstance(received, WangGatewayReceivedReport):
            raise TypeError("sink accepts only WangGatewayReceivedReport envelopes")
        report = received.report
        if self.command_coordinator is not None:
            confirmed = self.command_coordinator.reconcile_report(received)
            if confirmed is not None and report.tx_id in self._active:
                previous = self._active[report.tx_id]
                self._active[report.tx_id] = WangActiveTransmitter(
                    entity_id=previous.entity_id,
                    entity_type=previous.entity_type,
                    expected_ptx_dbm=int(confirmed.target_ptx_dbm),
                    ptx_epoch=int(confirmed.command_seq),
                )
        report_id = getattr(report, "report_id", None)
        try:
            observation_time = float(
                getattr(report, "observation_time_s", math.nan)
            )
        except (TypeError, ValueError):
            observation_time = math.nan
        window_index = (
            self._window_index(observation_time)
            if math.isfinite(observation_time) and observation_time >= 0
            else -1
        )
        if not report_id:
            self._reject(window_index, None, "malformed_report_id")
            return False
        if report_id in self._seen_report_ids:
            self._reject(window_index, report_id, "duplicate_report")
            return False
        self._seen_report_ids.add(report_id)

        if window_index < 0:
            self._reject(window_index, report_id, "invalid_observation_time")
            return False
        if window_index in self._decisions:
            self._reject(window_index, report_id, "late_closed_window")
            return False
        self._pending.append(received)
        return True

    def close_window(
        self,
        window_index: int,
        *,
        decision_time_s: float,
    ) -> WangWindowDecisionRecord:
        """Close one non-overlapping window exactly once after its cutoff."""
        index = int(window_index)
        if index < 0:
            raise ValueError("window index must be non-negative")
        if index in self._decisions:
            return self._decisions[index]

        start = index * self.window_s
        end = start + self.window_s
        cutoff = end + self.delivery_grace_s
        decision_time = float(decision_time_s)
        if not math.isfinite(decision_time) or decision_time < cutoff:
            raise ValueError(
                f"window {index} cannot close before cutoff {cutoff:.6f} s"
            )

        eligible_by_entity = defaultdict(list)
        retained = []
        for received in self._pending:
            report = received.report
            observation_time = float(report.observation_time_s)
            if observation_time >= end:
                retained.append(received)
                continue
            if observation_time < start:
                self._reject(index, report.report_id, "wrong_window_stale")
                continue

            reason = self._eligibility_rejection(received, cutoff=cutoff)
            if reason is not None:
                self._reject(index, report.report_id, reason)
                continue
            eligible_by_entity[report.tx_id].append(received)
        self._pending = retained

        active_ids = tuple(sorted(self._active))
        covered_ids = tuple(
            entity_id
            for entity_id in active_ids
            if len(eligible_by_entity.get(entity_id, ()))
            >= WANG_MIN_REPORTS_PER_TRANSMITTER
        )
        missing_ids = tuple(sorted(set(active_ids) - set(covered_ids)))
        coverage = len(covered_ids) / len(active_ids)
        rejection_counts = Counter(
            reason for _, reason in self._rejections.get(index, ())
        )
        source_report_ids = tuple(
            sorted(
                received.report.report_id
                for entity_id in covered_ids
                for received in eligible_by_entity[entity_id]
            )
        )
        decision_id = f"WANG-A2-W{index:04d}-C{int(round(cutoff * 1000)):010d}"

        if coverage < WANG_REQUIRED_COVERAGE_FRACTION:
            record = WangWindowDecisionRecord(
                decision_id=decision_id,
                controller_id=self.gateway_id,
                window_index=index,
                window_start_s=start,
                window_end_s=end,
                delivery_grace_s=self.delivery_grace_s,
                decision_cutoff_s=cutoff,
                decision_time_s=decision_time,
                status="no_decision",
                reason="insufficient_active_transmitter_coverage",
                active_transmitter_count=len(active_ids),
                covered_transmitter_count=len(covered_ids),
                coverage_fraction=coverage,
                required_coverage_fraction=WANG_REQUIRED_COVERAGE_FRACTION,
                minimum_reports_per_transmitter=WANG_MIN_REPORTS_PER_TRANSMITTER,
                active_transmitter_ids=active_ids,
                covered_transmitter_ids=covered_ids,
                missing_transmitter_ids=missing_ids,
                source_report_ids=source_report_ids,
                rejection_counts=tuple(sorted(rejection_counts.items())),
                network_snr_avg_db=None,
                network_snr_mad_db=None,
                snr_required_db=None,
                network_margin_db=None,
                network_reduction_steps=None,
                entity_decisions=(),
            )
            self._decisions[index] = record
            return record

        transmitter_means = {
            entity_id: math.fsum(
                item.report.snr_db for item in eligible_by_entity[entity_id]
            )
            / len(eligible_by_entity[entity_id])
            for entity_id in active_ids
        }
        # Equal transmitter weighting: a heavily loaded forwarding relay has
        # one mean in the network vector, exactly like an end device.
        network_result = WangA2PowerReductionController.decide(
            entity_type="node",
            current_ptx_dbm=17,
            snr_samples_db=(transmitter_means[item] for item in active_ids),
        )
        common_steps = network_result.reduction_steps

        entity_decisions = []
        for entity_id in active_ids:
            spec = self._active[entity_id]
            observations = eligible_by_entity[entity_id]
            bounds = WangA2PowerReductionController.bounds_for(spec.entity_type)
            target_ptx = max(
                bounds.minimum_dbm,
                spec.expected_ptx_dbm - 3 * common_steps,
            )
            entity_decisions.append(
                WangEntityDecisionRecord(
                    entity_id=entity_id,
                    entity_type=spec.entity_type,
                    ptx_epoch=spec.ptx_epoch,
                    current_ptx_dbm=spec.expected_ptx_dbm,
                    expected_ptx_dbm=spec.expected_ptx_dbm,
                    target_ptx_dbm=target_ptx,
                    report_count=len(observations),
                    source_report_ids=tuple(
                        sorted(item.report.report_id for item in observations)
                    ),
                    transmitter_mean_snr_db=transmitter_means[entity_id],
                    network_reduction_steps=common_steps,
                    reason=(
                        "power_reduced"
                        if target_ptx < spec.expected_ptx_dbm
                        else "no_reduction"
                    ),
                )
            )

        record = WangWindowDecisionRecord(
            decision_id=decision_id,
            controller_id=self.gateway_id,
            window_index=index,
            window_start_s=start,
            window_end_s=end,
            delivery_grace_s=self.delivery_grace_s,
            decision_cutoff_s=cutoff,
            decision_time_s=decision_time,
            status="decision_generated",
            reason="complete_active_transmitter_coverage",
            active_transmitter_count=len(active_ids),
            covered_transmitter_count=len(covered_ids),
            coverage_fraction=coverage,
            required_coverage_fraction=WANG_REQUIRED_COVERAGE_FRACTION,
            minimum_reports_per_transmitter=WANG_MIN_REPORTS_PER_TRANSMITTER,
            active_transmitter_ids=active_ids,
            covered_transmitter_ids=covered_ids,
            missing_transmitter_ids=missing_ids,
            source_report_ids=source_report_ids,
            rejection_counts=tuple(sorted(rejection_counts.items())),
            network_snr_avg_db=float(network_result.snr_avg_db),
            network_snr_mad_db=float(network_result.snr_mad_db),
            snr_required_db=network_result.snr_required_db,
            network_margin_db=float(network_result.margin_db),
            network_reduction_steps=common_steps,
            entity_decisions=tuple(entity_decisions),
        )
        self._decisions[index] = record
        if self.command_coordinator is not None:
            self.command_coordinator.handle_window_decision(record)
        return record

    def _eligibility_rejection(
        self,
        received: WangGatewayReceivedReport,
        *,
        cutoff: float,
    ) -> str | None:
        report = received.report
        try:
            arrival = float(received.gateway_arrival_time_s)
        except (TypeError, ValueError):
            return "invalid_gateway_arrival_time"
        try:
            observation = float(report.observation_time_s)
        except (TypeError, ValueError):
            return "invalid_observation_time"
        if not math.isfinite(arrival):
            return "invalid_gateway_arrival_time"
        if not math.isfinite(observation) or observation < 0:
            return "invalid_observation_time"
        if observation > arrival:
            return "future_dated_observation"
        if arrival > cutoff:
            return "arrival_after_decision_cutoff"
        if report.tx_id not in self._active:
            return "inactive_transmitter"
        if not str(report.rx_id):
            return "malformed_receiver_id"
        try:
            snr = float(report.snr_db)
        except (TypeError, ValueError):
            return "invalid_snr"
        if not math.isfinite(snr):
            return "invalid_snr"
        spec = self._active[report.tx_id]
        try:
            observed_ptx = int(report.observed_ptx_dbm)
        except (TypeError, ValueError, OverflowError):
            return "invalid_observed_ptx"
        if observed_ptx != report.observed_ptx_dbm:
            return "invalid_observed_ptx"
        try:
            observed_epoch = int(report.ptx_epoch)
        except (TypeError, ValueError, OverflowError):
            return "invalid_ptx_epoch"
        if observed_epoch != report.ptx_epoch or observed_epoch != spec.ptx_epoch:
            return "outdated_ptx_epoch"
        if observed_ptx != int(spec.expected_ptx_dbm):
            return "unexpected_declared_ptx"
        return None

    def run(self, env, *, last_window_index: int):
        """Optional SimPy cadence; caller must attach it only to main GW."""
        for index in range(int(last_window_index) + 1):
            cutoff = (index + 1) * self.window_s + self.delivery_grace_s
            delay = max(0.0, cutoff - float(env.now))
            if delay:
                yield env.timeout(delay)
            # Let already-scheduled same-time gateway decode events complete
            # before applying the inclusive arrival <= cutoff eligibility rule.
            yield env.timeout(0)
            self.close_window(index, decision_time_s=float(env.now))


__all__ = [
    "WANG_DELIVERY_GRACE_S",
    "WANG_MIN_REPORTS_PER_TRANSMITTER",
    "WANG_OBSERVATION_WINDOW_S",
    "WANG_REQUIRED_COVERAGE_FRACTION",
    "WangActiveTransmitter",
    "WangEntityDecisionRecord",
    "WangGatewaySinkDecisionEngine",
    "WangWindowDecisionRecord",
    "build_active_transmitter_population",
    "install_wang_gateway_scheduler",
    "last_closable_window_index",
    "wang_runtime_enabled",
]
