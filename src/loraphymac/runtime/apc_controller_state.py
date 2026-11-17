"""Controller-owned causal state for R-APC links.

The next-hop receiver owns this state.  It deliberately contains no reference
to the remote transmitter object: actual transmitter TPI/PTx remains private
to that transmitter until a future physical protocol proves otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_APC_MAX_RETRIES = 2


@dataclass(frozen=True, slots=True)
class APCPendingCommand:
    """One absolute-TPI command awaiting an unimplemented physical answer."""

    feedback_id: str
    source_estimated_tx_power_index: int
    target_tx_power_index: int
    first_attempt_time_s: float
    last_attempt_time_s: float
    retry_count: int = 0


@dataclass(slots=True)
class APCControllerLinkState:
    """Causal controller belief for one directed transmitter->controller link.

    ``confirmed_tx_power_index`` is initialized from deterministic provisioned
    configuration.  With no physical APC_ANS, later commands never advance it.
    ``estimated_tx_power_index`` is an explicit open-loop assumption used only
    after sending an absolute command; it is never labelled as confirmation.
    """

    controller_id: str
    transmitter_id: str
    confirmed_tx_power_index: int
    estimated_tx_power_index: int
    max_tx_power_index: int
    target_margin_db: float
    estimate_provenance: str = "preprovisioned_initial_state"
    pending_command: APCPendingCommand | None = None

    @classmethod
    def from_preprovisioned_state(
        cls,
        *,
        controller_id: str,
        transmitter_id: str,
        initial_tx_power_index: int,
        max_tx_power_index: int,
        target_margin_db: float,
    ) -> "APCControllerLinkState":
        initial = int(initial_tx_power_index)
        maximum = int(max_tx_power_index)
        if not 0 <= initial <= maximum:
            raise ValueError("initial TxPowerIndex is outside provisioned bounds")
        return cls(
            controller_id=str(controller_id),
            transmitter_id=str(transmitter_id),
            confirmed_tx_power_index=initial,
            estimated_tx_power_index=initial,
            max_tx_power_index=maximum,
            target_margin_db=float(target_margin_db),
        )

    @property
    def state_is_confirmed(self) -> bool:
        return (
            self.pending_command is None
            and self.estimate_provenance == "preprovisioned_initial_state"
            and self.estimated_tx_power_index == self.confirmed_tx_power_index
        )

    def begin_unconfirmed_command(
        self,
        *,
        feedback_id: str,
        target_tx_power_index: int,
        now_s: float,
    ) -> APCPendingCommand:
        if self.pending_command is not None:
            raise RuntimeError("cannot replace an outstanding APC command")
        target = int(target_tx_power_index)
        if not 0 <= target <= self.max_tx_power_index:
            raise ValueError("target TxPowerIndex is outside provisioned bounds")
        now = float(now_s)
        pending = APCPendingCommand(
            feedback_id=str(feedback_id),
            source_estimated_tx_power_index=self.estimated_tx_power_index,
            target_tx_power_index=target,
            first_attempt_time_s=now,
            last_attempt_time_s=now,
        )
        self.pending_command = pending
        self.estimated_tx_power_index = target
        self.estimate_provenance = "unconfirmed_command_assumption"
        return pending

    def retry_status(
        self,
        *,
        now_s: float,
        retry_timeout_s: float,
        max_retries: int = DEFAULT_APC_MAX_RETRIES,
    ) -> str:
        """Return ``waiting``, ``retry``, ``exhausted``, or ``idle``.

        A retry repeats the same absolute target.  Exhaustion removes only the
        pending transport record; it does not turn the estimate into confirmed
        state and does not infer whether any prior transmission was delivered.
        """

        pending = self.pending_command
        if pending is None:
            return "idle"
        timeout = float(retry_timeout_s)
        if timeout <= 0:
            raise ValueError("retry timeout must be > 0")
        retry_limit = int(max_retries)
        if retry_limit < 0:
            raise ValueError("max retries must be >= 0")
        now = float(now_s)
        if now - pending.last_attempt_time_s < timeout:
            return "waiting"
        if pending.retry_count < retry_limit:
            return "retry"
        return "exhausted"

    def mark_retry(self, *, now_s: float) -> APCPendingCommand:
        pending = self.pending_command
        if pending is None:
            raise RuntimeError("cannot retry without an outstanding APC command")
        updated = APCPendingCommand(
            feedback_id=pending.feedback_id,
            source_estimated_tx_power_index=pending.source_estimated_tx_power_index,
            target_tx_power_index=pending.target_tx_power_index,
            first_attempt_time_s=pending.first_attempt_time_s,
            last_attempt_time_s=float(now_s),
            retry_count=pending.retry_count + 1,
        )
        self.pending_command = updated
        return updated

    def mark_retry_exhausted(self) -> APCPendingCommand:
        pending = self.pending_command
        if pending is None:
            raise RuntimeError("cannot exhaust an idle APC link")
        self.pending_command = None
        self.estimate_provenance = "unconfirmed_retry_exhausted"
        return pending


__all__ = [
    "APCControllerLinkState",
    "APCPendingCommand",
    "DEFAULT_APC_MAX_RETRIES",
]
