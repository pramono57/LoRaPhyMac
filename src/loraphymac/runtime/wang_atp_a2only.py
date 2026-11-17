"""Pure Wang-inspired Algorithm 2 power-reduction controller.

W0 contract for model ``wang_atp_sf7_a2only``
================================================

Inputs
------
* SNR samples from reports that the sink causally received during one closed
  observation window.  The caller, not this pure controller, owns report
  provenance and must not supply receiver/global simulator statistics.
* The current entity PTx and entity type (``node`` or ``relay``).
* The SF7 demodulation requirement, defaulting to -7.5 dB.

Output and algorithm
--------------------
The controller returns one immutable decision.  For at least one received SNR
sample it computes::

    margin = mean(SNR) - mean_absolute_deviation(SNR) - SNR_required
    steps = max(0, floor(margin / 3))
    target_PTx = max(PTx_min, current_PTx - 3 * steps)

All reduction steps are applied in that one decision.  PTx never increases.
Node bounds are 5..17 dBm and relay bounds are 12..20 dBm.  Initial PTx is the
entity maximum.  SF remains fixed at 7.

State and missing observations
------------------------------
This controller is stateless.  Runtime code will own the observation window,
current PTx, command lifecycle, and physical delivery.  ``None`` denotes a
missing observation and is ignored.  If no samples remain, the result is a
no-op with reason ``missing_snr_samples``.  Non-finite samples fail closed.

Minimum W2 interface
--------------------
The future sink path must provide received reports containing report ID,
transmitting entity/link, SNR, observation time, and sink-receipt time.  A
decision must create a command containing decision ID, target entity, previous
and target PTx, creation time, and source report IDs.  The target may update
PTx only after that command is physically received; command airtime, energy,
queueing, collision, loss, and age must remain observable.

This module intentionally has no registry, PHY/MAC, routing, ACK/retry, energy,
or simulator dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


WANG_ATP_SF7_A2ONLY_MODEL_ID = "wang_atp_sf7_a2only"
SF7_SNR_REQUIRED_DB = -7.5


@dataclass(frozen=True)
class WangPowerBounds:
    """Operational conducted-power bounds for one transmitter class."""

    minimum_dbm: int
    maximum_dbm: int

    def __post_init__(self) -> None:
        if self.minimum_dbm > self.maximum_dbm:
            raise ValueError("minimum PTx must not exceed maximum PTx")


NODE_POWER_BOUNDS = WangPowerBounds(minimum_dbm=5, maximum_dbm=17)
RELAY_POWER_BOUNDS = WangPowerBounds(minimum_dbm=12, maximum_dbm=20)


@dataclass(frozen=True)
class WangA2Decision:
    """One stateless network-SNR power-reduction decision for one entity."""

    entity_type: str
    current_ptx_dbm: int
    target_ptx_dbm: int
    ptx_min_dbm: int
    ptx_max_dbm: int
    snr_sample_count: int
    snr_avg_db: float | None
    snr_mad_db: float | None
    snr_required_db: float
    margin_db: float | None
    reduction_steps: int
    reason: str

    @property
    def changed(self) -> bool:
        return self.target_ptx_dbm != self.current_ptx_dbm


class WangA2PowerReductionController:
    """Isolated fixed-SF7 Algorithm 2 controller with no runtime side effects."""

    _BOUNDS = {
        "node": NODE_POWER_BOUNDS,
        "relay": RELAY_POWER_BOUNDS,
    }

    @classmethod
    def bounds_for(cls, entity_type: str) -> WangPowerBounds:
        normalized = str(entity_type).strip().lower()
        try:
            return cls._BOUNDS[normalized]
        except KeyError as exc:
            raise ValueError(f"unsupported Wang entity type {entity_type!r}") from exc

    @classmethod
    def initial_ptx_dbm(cls, entity_type: str) -> int:
        """Return the operational maximum required by the W0 contract."""
        return cls.bounds_for(entity_type).maximum_dbm

    @classmethod
    def decide(
        cls,
        *,
        entity_type: str,
        current_ptx_dbm: int,
        snr_samples_db: Iterable[float | None],
        snr_required_db: float = SF7_SNR_REQUIRED_DB,
    ) -> WangA2Decision:
        """Return a pure, non-increasing PTx decision for one entity."""
        normalized_type = str(entity_type).strip().lower()
        bounds = cls.bounds_for(normalized_type)
        current = cls._validate_ptx(current_ptx_dbm, bounds)
        required = float(snr_required_db)
        if not math.isfinite(required):
            raise ValueError("SNR requirement must be finite")

        samples = []
        for sample in snr_samples_db:
            if sample is None:
                continue
            value = float(sample)
            if not math.isfinite(value):
                raise ValueError("SNR samples must be finite or None")
            samples.append(value)

        if not samples:
            return WangA2Decision(
                entity_type=normalized_type,
                current_ptx_dbm=current,
                target_ptx_dbm=current,
                ptx_min_dbm=bounds.minimum_dbm,
                ptx_max_dbm=bounds.maximum_dbm,
                snr_sample_count=0,
                snr_avg_db=None,
                snr_mad_db=None,
                snr_required_db=required,
                margin_db=None,
                reduction_steps=0,
                reason="missing_snr_samples",
            )

        average = math.fsum(samples) / len(samples)
        mad = math.fsum(abs(value - average) for value in samples) / len(samples)
        margin = average - mad - required
        reduction_steps = max(0, math.floor(margin / 3.0))
        target = max(bounds.minimum_dbm, current - 3 * reduction_steps)

        return WangA2Decision(
            entity_type=normalized_type,
            current_ptx_dbm=current,
            target_ptx_dbm=target,
            ptx_min_dbm=bounds.minimum_dbm,
            ptx_max_dbm=bounds.maximum_dbm,
            snr_sample_count=len(samples),
            snr_avg_db=average,
            snr_mad_db=mad,
            snr_required_db=required,
            margin_db=margin,
            reduction_steps=reduction_steps,
            reason="power_reduced" if target < current else "no_reduction",
        )

    @staticmethod
    def _validate_ptx(value: int, bounds: WangPowerBounds) -> int:
        if isinstance(value, bool):
            raise ValueError("current PTx must be an integer dBm value")
        current = int(value)
        if current != value:
            raise ValueError("current PTx must be an integer dBm value")
        if not bounds.minimum_dbm <= current <= bounds.maximum_dbm:
            raise ValueError(
                f"current PTx {current} dBm is outside "
                f"[{bounds.minimum_dbm}, {bounds.maximum_dbm}] dBm"
            )
        return current


__all__ = [
    "NODE_POWER_BOUNDS",
    "RELAY_POWER_BOUNDS",
    "SF7_SNR_REQUIRED_DB",
    "WANG_ATP_SF7_A2ONLY_MODEL_ID",
    "WangA2Decision",
    "WangA2PowerReductionController",
    "WangPowerBounds",
]
