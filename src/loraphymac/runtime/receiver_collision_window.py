"""Shared receiver-scoped RF-window bookkeeping.

The physical collision decision remains owned by :class:`CollisionEngine`.
This module only keeps the dynamic packet components used by both RelayRX and
the optional single-channel NodeRX collision path.
"""


def wait_until_window_quiet(env, packets, eps=1e-9):
    """Wait until every burst currently attached to a component has ended."""
    while True:
        if not packets:
            return
        max_end = max(packet.rx_end for packet in packets if hasattr(packet, "rx_end"))
        hold_s = max(0.0, float(max_end) - float(env.now))
        if hold_s <= eps:
            return
        yield env.timeout(hold_s)


def register_rf_window_packet(
    *, collision, windows, packet, next_sequence
):
    """Attach a burst to, or merge, its active physical RF component."""
    matching_keys = []
    for key, window_packets in list(windows.items()):
        if not window_packets:
            continue
        if any(
            collision.packets_share_rf_edge(packet, existing)
            for existing in window_packets
        ):
            matching_keys.append(key)

    sequence = int(next_sequence)
    if not matching_keys:
        sequence += 1
        key = f"rf_window_{sequence}"
        window = windows[key]
    else:
        key = min(
            matching_keys,
            key=lambda value: int(str(value).rsplit("_", 1)[-1]),
        )
        window = windows[key]
        for merged_key in matching_keys:
            if merged_key == key:
                continue
            merged_window = windows[merged_key]
            for existing in merged_window:
                if all(existing is not current for current in window):
                    window.append(existing)
            # The former leader sees an empty list and cannot decide twice.
            merged_window.clear()

    window.append(packet)
    return key, window, sequence
