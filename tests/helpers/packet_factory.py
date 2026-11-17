from loraphymac.runtime.packet import Packet


def make_packet(**overrides):
    """Build a minimal valid Packet without invoking simulator runtime."""
    values = {
        "src_id": "N1",
        "current_tx": "N1",
        "next_hop": "R1",
        "path": ["N1", "R1", "GW"],
        "hop_index": 0,
        "ch_id": 0,
        "freq_mhz": 921.4,
        "sf": 7,
        "bw_hz": 125_000,
        "ptx_dbm": 20,
        "payload_size": 62,
        "toa": 0.1,
        "tx_start": 1.0,
        "tx_end": 1.1,
        "app_payload_size": 50,
    }
    values.update(overrides)
    return Packet(**values)

