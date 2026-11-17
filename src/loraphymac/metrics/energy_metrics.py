"""Read-only energy result tables for the :class:`Statistics` facade."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd


ENERGY_ACCOUNTING_MODES = frozenset({"legacy", "corrected"})


def normalize_energy_accounting_mode(value: Any) -> str:
    """Return a validated, normalized energy-accounting mode."""
    mode = str(value).lower().strip()
    if mode not in ENERGY_ACCOUNTING_MODES:
        raise ValueError(
            f"unsupported energy_accounting_mode={value!r}; "
            f"expected one of {sorted(ENERGY_ACCOUNTING_MODES)}"
        )
    return mode


def energy_accounting_model_version(mode: Any) -> str:
    """Return the immutable model identity written to run artifacts."""
    normalized = normalize_energy_accounting_mode(mode)
    if normalized == "corrected":
        return "eq10_exclusive_transition_v1"
    return "legacy_continuous_rx_without_transition_exclusion_v1"


def continuous_rx_energy_mj(
    entity: Any,
    sim_time: float,
    accounting_mode: str = "legacy",
) -> tuple[float, float, float]:
    """Calculate continuous-RX energy under a named accounting model.

    ``legacy`` preserves the historical model. ``corrected`` implements
    manuscript Eq. (10) for the single-radio, half-duplex hardware used by
    Paper 1: ``T_rx = T_sim - T_tx - T_transition``. The corrected model
    rejects non-exclusive radio configurations and inconsistent accumulated
    durations rather than silently subtracting overlapping time.
    """
    mode = normalize_energy_accounting_mode(accounting_mode)
    sim_time = max(float(sim_time), 0.0)
    tx_time_s = max(float(getattr(entity, "total_tx_time_s", 0.0)), 0.0)
    transition_time_s = max(
        float(getattr(entity, "total_transition_time_s", 0.0)),
        0.0,
    )
    tx_full_duplex = bool(getattr(entity.hw, "tx_full_duplex", False))

    if mode == "legacy" and tx_full_duplex:
        rx_time_s = sim_time
    elif mode == "legacy":
        rx_time_s = max(sim_time - tx_time_s, 0.0)
    else:
        tx_capacity = int(getattr(entity.hw, "tx_phy_radio_count", 1))
        if tx_full_duplex or tx_capacity != 1:
            raise ValueError(
                "corrected energy accounting requires mutually exclusive "
                "single-radio half-duplex TX, transition, and RX states"
            )
        exclusive_time_s = tx_time_s + transition_time_s
        tolerance_s = max(1e-9, sim_time * 1e-12)
        if exclusive_time_s > sim_time + tolerance_s:
            raise ValueError(
                "corrected energy accounting received non-exclusive durations: "
                f"tx={tx_time_s:.12g}s + transition={transition_time_s:.12g}s "
                f"> sim_time={sim_time:.12g}s"
            )
        rx_time_s = max(sim_time - exclusive_time_s, 0.0)

    voltage_v = float(getattr(entity.hw, "voltage", 0.0))
    rx_current_ma = float(getattr(entity.hw, "rx_current", 0.0))
    rx_energy_mj = voltage_v * (rx_current_ma / 1000.0) * rx_time_s * 1000.0

    if getattr(entity, "include_mcu_energy", True) and getattr(entity, "mcu", None) is not None:
        rx_energy_mj += entity.mcu.energy_active(rx_time_s)

    return rx_energy_mj, rx_time_s, tx_time_s


def build_energy_breakdown_df(
    nodes: Sequence[Any],
    run_metadata: Mapping[str, Any],
    infer_entity_type: Callable[[str], str],
    continuous_rx_energy: Callable[[Any], tuple[float, float, float]],
    millijoule_to_joule: Callable[[float], float],
) -> pd.DataFrame:
    """Build the node/relay/gateway energy matrix in joules."""
    rows = []
    accounting_mode = normalize_energy_accounting_mode(
        run_metadata.get("energy_accounting_mode", "legacy")
    )
    model_version = energy_accounting_model_version(accounting_mode)

    for node in nodes:
        entity_type = infer_entity_type(node.id)

        tx_energy_mj = float(getattr(node, "energy_tx_mJ", 0.0))
        sleep_energy_mj = float(getattr(node, "energy_sleep_mJ", 0.0))
        idle_energy_mj = float(getattr(node, "energy_idle_mJ", 0.0))
        transition_energy_mj = float(
            getattr(
                node,
                "energy_transition_mJ",
                0.0,
            )
        )
        cad_energy_mj = float(getattr(node, "energy_cad_mJ", 0.0))

        if entity_type in ("relay", "gateway") and getattr(
            node,
            "rx_always_on",
            True,
        ):
            rx_energy_mj, rx_time_s, tx_time_s = continuous_rx_energy(node)
        else:
            rx_energy_mj = float(getattr(node, "energy_rx_mJ", 0.0))
            rx_time_s = float(getattr(node, "total_rx_time_s", 0.0))
            tx_time_s = float(getattr(node, "total_tx_time_s", 0.0))

        total_energy_mj = (
            tx_energy_mj
            + rx_energy_mj
            + sleep_energy_mj
            + idle_energy_mj
            + transition_energy_mj
            + cad_energy_mj
        )

        rows.append(
            {
                "run_id": run_metadata.get("run_id"),
                "entity_id": node.id,
                "entity_type": entity_type,
                "energy_accounting_mode": accounting_mode,
                "energy_accounting_model_version": model_version,
                "include_mcu_energy": getattr(node, "include_mcu_energy", True),
                "rx_always_on": getattr(
                    node,
                    "rx_always_on",
                    entity_type in ("relay", "gateway"),
                ),
                "tx_full_duplex": getattr(node.hw, "tx_full_duplex", False),
                "duplex_mode": (
                    "full_duplex" if getattr(node.hw, "tx_full_duplex", False) else "half_duplex"
                ),
                "tx_time_s": tx_time_s,
                "rx_time_s": rx_time_s,
                "total_transition_time_s": float(
                    getattr(node, "total_transition_time_s", 0.0)
                ),
                "rx_window_count": getattr(node, "rx_window_count", None),
                "rx_window_timeout_count": getattr(
                    node,
                    "rx_window_timeout_count",
                    None,
                ),
                "rx_window_downlink_count": getattr(
                    node,
                    "rx_window_downlink_count",
                    None,
                ),
                "energy_tx_J": millijoule_to_joule(tx_energy_mj),
                "energy_rx_J": millijoule_to_joule(rx_energy_mj),
                "energy_sleep_J": millijoule_to_joule(sleep_energy_mj),
                "energy_idle_J": millijoule_to_joule(idle_energy_mj),
                "energy_transition_J": millijoule_to_joule(transition_energy_mj),
                "energy_cad_J": millijoule_to_joule(cad_energy_mj),
                "total_energy_J": millijoule_to_joule(total_energy_mj),
            }
        )

    return pd.DataFrame(rows)
