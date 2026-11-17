from dataclasses import dataclass


class ConstantPathlossModel:
    def __init__(self, pathloss_db=120.0):
        self.pathloss_db = float(pathloss_db)
        self.calls = []

    def pathloss(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self.pathloss_db


class LinearPathlossModel:
    """Monotonic fake useful for search tests without invoking ITM."""

    def __init__(self, intercept_db=80.0, db_per_meter=0.001):
        self.intercept_db = float(intercept_db)
        self.db_per_meter = float(db_per_meter)

    def pathloss(self, d_m, **kwargs):
        return self.intercept_db + self.db_per_meter * float(d_m)


class RecordingReceiver:
    def __init__(self, env):
        self.env = env
        self.packets = []
        self.arrival_times = []

    def on_arrival(self, packet):
        self.packets.append(packet)
        self.arrival_times.append(float(self.env.now))


@dataclass
class FakeEndpoint:
    id: str
    rx: object

