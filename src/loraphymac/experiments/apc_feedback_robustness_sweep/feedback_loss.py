"""Deterministic additional erasure for physical R-APC feedback attempts."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


LOSS_POLICY_VERSION = "sha256_feedback_attempt_v1"


def feedback_loss_uniform(
    *, seed: int, controller_id: str, target_id: str, decision_ordinal: str,
    attempt_number: int,
) -> float:
    """Return a reproducible U[0,1) value; experimental factors are excluded."""
    payload = {
        "policy": LOSS_POLICY_VERSION,
        "seed": int(seed),
        "controller_id": str(controller_id),
        "target_id": str(target_id),
        "decision_ordinal": str(decision_ordinal),
        "attempt_number": int(attempt_number),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    return value / 2**64


def additional_feedback_erasure(
    probability: float, *, seed: int, controller_id: str, target_id: str,
    feedback_id: str, attempt_number: int,
) -> tuple[bool, float | None]:
    """Evaluate a receiver-side extra loss without touching simulator RNG."""
    p = float(probability)
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("feedback loss probability must be finite and in [0, 1]")
    if p == 0.0:
        return False, None
    ordinal = str(feedback_id).rsplit(":apc:", 1)[-1]
    uniform = feedback_loss_uniform(
        seed=seed,
        controller_id=controller_id,
        target_id=target_id,
        decision_ordinal=ordinal,
        attempt_number=attempt_number,
    )
    return uniform < p, uniform


def is_apc_feedback_frame(packet: Any, receiver_id: str) -> bool:
    """Require downlink MType, APC FCtrl, metadata, and matching target."""
    from loraphymac.mac.frame_defs import MType, decode_fctrl

    try:
        mtype = int(getattr(packet, "mtype", -1))
        if mtype not in (int(MType.DATA_DOWN), int(MType.DATA_DOWN_FWD)):
            return False
        info = getattr(packet, "apc_info", None)
        context = (info or {}).get("context", {})
        return (
            bool(decode_fctrl(getattr(packet, "fctrl", 0)).get("apc", False))
            and info is not None
            and str(context.get("target_id")) == str(receiver_id)
            and bool(context.get("controller_id"))
            and bool(info.get("feedback_id"))
        )
    except (TypeError, ValueError, AttributeError):
        return False
