# ================================================================
# utils/generator.py
# Physical DATA_DOWN / DATA_DOWN_FWD + 
# Relay APC RX cleanup, APC TX index, and timing validation
#
# Generators for nodes, relays, and gateways
# by: Sigit Pramono
# ================================================================

import numpy as np
from numbers import Integral

from loraphymac.domain.node import Node


class Location:
    """2D coordinate class. Antenna height is stored in Node.h."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return np.sqrt(dx * dx + dy * dy)


class NodeGenerator:

    @staticmethod
    def uniform_random(
        n,
        x_range,
        y_range,
        height,
        radio,
        mcu,
        gain,
        channel_plan,
        air_interface,
        routing_engine,
        collision_engine,
        seed=None,
        y_exclusion_min=-1000,
        y_exclusion_max=1000,
    ):
        if isinstance(n, bool) or not isinstance(n, Integral) or int(n) < 0:
            raise ValueError("n must be a non-negative integer")
        n = int(n)
        if len(x_range) != 2 or len(y_range) != 2:
            raise ValueError("x_range and y_range must each contain two values")
        x_min, x_max = (float(x_range[0]), float(x_range[1]))
        y_min, y_max = (float(y_range[0]), float(y_range[1]))
        exclusion_min = float(y_exclusion_min)
        exclusion_max = float(y_exclusion_max)
        if not np.isfinite([x_min, x_max, y_min, y_max]).all():
            raise ValueError("placement ranges must contain finite values")
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("placement range minimum must be smaller than maximum")
        if exclusion_min >= exclusion_max:
            raise ValueError("y exclusion minimum must be smaller than maximum")

        available_below = max(0.0, min(y_max, exclusion_min) - y_min)
        available_above = max(0.0, y_max - max(y_min, exclusion_max))
        if n > 0 and available_below + available_above <= 0.0:
            raise ValueError(
                "y_range is fully inside the node-placement exclusion zone"
            )

        if seed is not None:
            np.random.seed(seed)

        xs = np.random.uniform(x_min, x_max, n)

        ys = []
        attempts = 0
        max_attempts = max(10_000, n * 1_000)
        while len(ys) < n:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    "node placement could not sample enough positions outside "
                    "the y exclusion zone; enlarge area_y"
                )
            y = np.random.uniform(y_min, y_max)

            # Avoid the central area near the X-axis/relay backbone.
            if y <= exclusion_min or y >= exclusion_max:
                ys.append(y)

        nodes = []
        for i, (x, y) in enumerate(zip(xs, ys), start=1):
            nodes.append(
                Node(
                    node_id=f"N{i}",
                    loc=Location(x, y),
                    height=height,
                    radio_hw=radio,
                    mcu_hw=mcu,
                    gain_dbi=gain,
                    channel_plan=channel_plan,
                    air_interface=air_interface,
                    routing_engine=routing_engine,
                    collision_engine=collision_engine,
                )
            )

        return nodes

    @staticmethod
    def linear(
        n,
        x_range,
        y_range,
        height,
        radio,
        mcu,
        gain,
        channel_plan,
        air_interface,
        routing_engine,
        collision_engine,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)

        xs = np.linspace(x_range[0], x_range[1], n + 2)[1:-1]

        if isinstance(y_range, (tuple, list)) and len(y_range) == 2:
            y_span = max(abs(y_range[0]), abs(y_range[1]))
        else:
            y_span = abs(y_range)

        y_bottom = -y_span * 1.25
        y_top = y_span * 0.75

        nodes = []
        for i, x in enumerate(xs, start=1):
            y = y_bottom if (i - 1) % 2 == 0 else y_top

            nodes.append(
                Node(
                    node_id=f"N{i}",
                    loc=Location(x, y),
                    height=height,
                    radio_hw=radio,
                    mcu_hw=mcu,
                    gain_dbi=gain,
                    channel_plan=channel_plan,
                    air_interface=air_interface,
                    routing_engine=routing_engine,
                    collision_engine=collision_engine,
                )
            )

        return nodes


class RelayGenerator:

    @staticmethod
    def linear(
        n,
        x_range,
        y_fixed,
        height,
        radio,
        mcu,
        gain,
        channel_plan,
        air_interface,
        routing_engine,
        collision_engine,
    ):
        xs = [
            x_range[0] + ((2 * i) / (n + 3)) * (x_range[1] - x_range[0])
            for i in range(1, n + 1)
        ]

        relays = []
        for i, x in enumerate(xs, start=1):
            relays.append(
                Node(
                    node_id=f"R{i}",
                    loc=Location(x, y_fixed),
                    height=height,
                    radio_hw=radio,
                    mcu_hw=mcu,
                    gain_dbi=gain,
                    channel_plan=channel_plan,
                    air_interface=air_interface,
                    routing_engine=routing_engine,
                    collision_engine=collision_engine,
                )
            )

        return relays


class GatewayGenerator:

    @staticmethod
    def fixed(
        x,
        y,
        height,
        radio,
        mcu,
        gain,
        channel_plan,
        air_interface,
        routing_engine,
        collision_engine,
    ):
        return Node(
            node_id="GW",
            loc=Location(x, y),
            height=height,
            radio_hw=radio,
            mcu_hw=mcu,
            gain_dbi=gain,
            channel_plan=channel_plan,
            air_interface=air_interface,
            routing_engine=routing_engine,
            collision_engine=collision_engine,
        )
