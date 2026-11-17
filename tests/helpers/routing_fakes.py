from dataclasses import dataclass
import copy

from loraphymac.hardware.hardware import SX1276, SX1302


@dataclass(frozen=True)
class FakeLocation:
    x: float
    y: float = 0.0


class FakeRoutingEntity:
    def __init__(self, entity_id, x, y=0.0, height=2.0, gain=0.0, hw=None):
        self.id = entity_id
        self.loc = FakeLocation(float(x), float(y))
        self.h = float(height)
        self.gain = float(gain)
        self.hw = hw if hw is not None else (
            SX1276() if str(entity_id).startswith("N") else SX1302()
        )

    def distance_to(self, other):
        dx = self.loc.x - other.loc.x
        dy = self.loc.y - other.loc.y
        return (dx * dx + dy * dy) ** 0.5


class DistancePathlossModel:
    """Controlled model keyed by rounded link distance in metres."""

    def __init__(self, losses_by_distance, default=None):
        self.losses_by_distance = dict(losses_by_distance)
        self.default = default
        self.calls = []

    def pathloss(self, d_m, **kwargs):
        row = {"d_m": float(d_m), **kwargs}
        self.calls.append(row)
        key = round(float(d_m), 9)
        if key in self.losses_by_distance:
            return float(self.losses_by_distance[key])
        if self.default is not None:
            return float(self.default)
        raise KeyError(f"No controlled pathloss for distance {d_m}")


class RecordingLinkBudgetAir:
    """Link-budget fake where RSSI and feasibility derive from known pathloss."""

    def __init__(self, feasible_pathloss_max=150.0):
        self.feasible_pathloss_max = float(feasible_pathloss_max)
        self.calls = []

    def compute_link_budget(self, **kwargs):
        self.calls.append(copy.copy(kwargs))
        rssi = (
            float(kwargs["ptx"])
            + float(kwargs["gtx"])
            + float(kwargs["grx"])
            - float(kwargs["pathloss"])
        )
        feasible = float(kwargs["pathloss"]) <= self.feasible_pathloss_max
        return {
            "rssi": rssi,
            "snr": rssi + 120.0,
            "margin": self.feasible_pathloss_max - float(kwargs["pathloss"]),
            "feasible": feasible,
        }


def make_topology(relay_positions=(10_000, 20_000), node_x=30_000):
    gateway = FakeRoutingEntity("GW", 0, height=25, gain=5.8)
    relays = [
        FakeRoutingEntity(f"R{i}", x, height=25, gain=5.8)
        for i, x in enumerate(relay_positions, start=1)
    ]
    node = FakeRoutingEntity("N1", node_x, height=2, gain=2.15)
    return node, relays, gateway


def assert_terminal_route(path, gateway_id="GW"):
    assert path[-1] == gateway_id
    assert len(path) == len(set(path))

