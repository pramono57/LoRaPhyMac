from dataclasses import dataclass
import math


@dataclass(eq=False)
class CollisionPacket:
    packet_id: str
    rx_start: float = 1.0
    rx_end: float = 2.0
    ch_id: int | None = 0
    freq_mhz: float | None = 923.2
    sf: int | None = 7
    bw_hz: int | None = 125_000
    rssi: float | None = -100.0
    reason: str | None = None
    toa: float = 1.0
    tx_start: float = 1.0
    t_arrive: float | None = None
    t_end: float | None = None


def make_collision_packet(packet_id="P1", **overrides):
    values = {"packet_id": packet_id}
    values.update(overrides)
    return CollisionPacket(**values)


def reference_sinr_db(signal_dbm, interferer_dbms, noise_floor_dbm):
    signal_mw = 10.0 ** (signal_dbm / 10.0)
    interference_mw = sum(10.0 ** (value / 10.0) for value in interferer_dbms)
    noise_mw = 10.0 ** (noise_floor_dbm / 10.0)
    return 10.0 * math.log10(signal_mw / (interference_mw + noise_mw))


def assert_partition(inputs, survivors, failed):
    input_ids = [id(packet) for packet in inputs]
    survivor_ids = [id(packet) for packet in survivors]
    failed_ids = [id(packet) for packet in failed]
    assert len(survivors) + len(failed) == len(inputs)
    assert len(set(survivor_ids)) == len(survivor_ids)
    assert len(set(failed_ids)) == len(failed_ids)
    assert set(survivor_ids).isdisjoint(failed_ids)
    assert set(survivor_ids + failed_ids) == set(input_ids)
