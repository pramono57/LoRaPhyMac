from types import SimpleNamespace


class RecordingStatistics:
    def __init__(self):
        self.events = []

    def record_demodulator_event(self, **event):
        self.events.append(dict(event))


def make_demod_packet(packet_id="P1", sf=7, toa=0.1, **overrides):
    values = {
        "packet_id": packet_id,
        "src_id": "N1",
        "current_tx": "N1",
        "prev_hop": "N1",
        "next_hop": "R1",
        "mtype": 1,
        "hop_index": 0,
        "ch_id": 0,
        "sf": sf,
        "bw_hz": 125_000,
        "toa": toa,
        "rx_start": 1.0,
        "rx_end": 1.0 + float(toa or 0),
    }
    values.update(overrides)
    return SimpleNamespace(**values)

