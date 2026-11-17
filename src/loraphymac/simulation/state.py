"""Initialization of mutable radio and APC state before process startup."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loraphymac.hardware.hardware import validate_ptx


def initialize_node_ptx_state(
    node: Any,
    config: Mapping[str, Any],
    index: int,
) -> None:
    """Initialize one end device while preserving historical defaults."""
    node.channel_mode = index % 8
    node.default_sf = config["node_sf_list"][index % len(config["node_sf_list"])]
    node.default_bw_hz = config["node_bw"]
    node.include_mcu_energy = bool(config["include_mcu_energy"])
    node.ptx_mode = str(config["node_ptx_mode"]).lower().strip()

    if config["node_ptx_init"] is None:
        initial_ptx = int(node.ptx_max)
    else:
        initial_ptx = validate_ptx(node, config["node_ptx_init"], "NODE")

    node.fixed_ptx = initial_ptx
    node.default_ptx = initial_ptx
    node.target_margin_db = config["node_target_margin_db"]
    node.last_rssi = None
    node.last_snr = None
    node.last_margin = None
    node.last_feedback_from = None
    node.last_feedback_time = None
    node.rx_always_on = False
    node.total_tx_time_s = 0.0
    node.fcnt = 0


def initialize_relay_ptx_state(relay: Any, config: Mapping[str, Any]) -> None:
    """Initialize one relay's radio and APC state."""
    relay.ptx_mode = str(config["relay_ptx_mode"]).lower().strip()

    if config["relay_ptx_init"] is None:
        initial_ptx = int(relay.ptx_max)
    else:
        initial_ptx = validate_ptx(relay, config["relay_ptx_init"], "RELAY")

    relay.fixed_ptx = initial_ptx
    relay.default_ptx = initial_ptx
    relay.include_mcu_energy = bool(config["include_mcu_energy"])
    relay.rx_always_on = True
    relay.total_tx_time_s = 0.0
    relay.target_margin_db = config["relay_target_margin_db"]
    relay.last_rssi = None
    relay.last_snr = None
    relay.last_margin = None
    relay.last_feedback_from = None
    relay.last_feedback_time = None
    relay.fcnt = 0


def initialize_gateway_ptx_state(gateway: Any, config: Mapping[str, Any]) -> None:
    """Initialize gateway radio state using the relay downlink power setting."""
    gateway.ptx_mode = "fixed"

    if config["relay_ptx_init"] is None:
        initial_ptx = int(gateway.ptx_max)
    else:
        initial_ptx = validate_ptx(gateway, config["relay_ptx_init"], "GATEWAY")

    gateway.fixed_ptx = initial_ptx
    gateway.default_ptx = initial_ptx
    gateway.include_mcu_energy = bool(config["include_mcu_energy"])
    gateway.rx_always_on = True
    gateway.total_tx_time_s = 0.0
    gateway.target_margin_db = config["relay_target_margin_db"]
    gateway.last_rssi = None
    gateway.last_snr = None
    gateway.last_margin = None
    gateway.last_feedback_from = None
    gateway.last_feedback_time = None
    gateway.fcnt = 0
