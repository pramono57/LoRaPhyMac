"""Read-only RelayTX queue result tables for the :class:`Statistics` facade.

Runtime queue instrumentation remains owned by ``Statistics`` and ``RelayTX``.
This module only normalizes recorded samples and derives finite-horizon queue
metrics from them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd


TX_QUEUE_LIFECYCLE_COLUMNS = [
    "queue_visit_id",
    "entity_id",
    "entity_type",
    "worker_id",
    "packet_uid",
    "src_id",
    "fcnt",
    "mtype",
    "traffic_class",
    "hop_index",
    "channel",
    "queue_capacity_packets",
    "status",
    "queue_arrival_time",
    "queue_dequeue_time",
    "duty_ready_time",
    "rx_idle_time",
    "tx_start",
    "tx_end",
    "service_end_time",
    "observation_end_time",
    "fifo_wait_s",
    "duty_wait_s",
    "half_duplex_wait_s",
    "rx_to_tx_transition_s",
    "airtime_s",
    "observed_airtime_s",
    "tx_to_rx_transition_s",
    "worker_service_s",
    "time_to_tx_start_s",
    "system_sojourn_s",
    "censored_waiting_age_s",
    "censored_service_age_s",
]

_EMPTY_QUEUE_STATE = {
    "avg_waiting_queue_length_lq": 0.0,
    "max_waiting_queue_length": 0,
    "avg_in_service_count": 0.0,
    "max_in_service_count": 0,
    "avg_system_count_l": 0.0,
    "max_system_count": 0,
}


def build_tx_queue_lifecycle_df(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Normalize queue visits to the stable RelayTX lifecycle schema."""
    if not records:
        return pd.DataFrame(columns=TX_QUEUE_LIFECYCLE_COLUMNS)
    return pd.DataFrame(records, columns=TX_QUEUE_LIFECYCLE_COLUMNS)


def queue_length_metrics(
    samples: Sequence[Mapping[str, Any]],
    sim_time_s: float | None,
) -> dict[str, float | int]:
    """Integrate the waiting-queue length as a piecewise-constant process."""
    try:
        horizon = float(sim_time_s)
    except Exception:
        horizon = 0.0

    if horizon <= 0:
        horizon = 0.0

    if not samples:
        return {"avg_queue_length": 0.0, "max_queue_length": 0}

    cleaned = []
    for sample in samples:
        try:
            time = float(sample.get("time", 0.0))
            queue_length = max(0, int(sample.get("queue_length", 0)))
            cleaned.append((time, queue_length))
        except Exception:
            continue

    if not cleaned:
        return {"avg_queue_length": 0.0, "max_queue_length": 0}

    cleaned.sort(key=lambda row: row[0])
    if horizon <= 0:
        horizon = max(time for time, _ in cleaned)

    if horizon <= 0:
        return {
            "avg_queue_length": 0.0,
            "max_queue_length": max(queue_length for _, queue_length in cleaned),
        }

    area = 0.0
    previous_time = 0.0
    previous_queue_length = 0
    maximum_queue_length = 0

    for time, queue_length in cleaned:
        time = max(0.0, min(float(time), horizon))
        if time >= previous_time:
            area += previous_queue_length * (time - previous_time)
            previous_time = time
            previous_queue_length = queue_length
            maximum_queue_length = max(maximum_queue_length, queue_length)

    if previous_time < horizon:
        area += previous_queue_length * (horizon - previous_time)

    return {
        "avg_queue_length": round(float(area / horizon), 6),
        "max_queue_length": int(maximum_queue_length),
    }


def percentile_ms(delays_s: Sequence[float], quantile: float) -> float:
    """Return a queue-delay percentile in milliseconds."""
    if not delays_s:
        return 0.0
    try:
        return float(pd.Series(list(delays_s), dtype="float64").quantile(quantile) * 1000.0)
    except Exception:
        return 0.0


def build_tx_queue_waiting_time_df(
    waiting_times: Mapping[Any, Sequence[float]],
) -> pd.DataFrame:
    """Build one row per observed RelayTX FIFO waiting time."""
    rows = []
    for relay_id, delays in waiting_times.items():
        for delay_s in delays:
            rows.append(
                {
                    "relay": relay_id,
                    "waiting_time_wq_s": delay_s,
                    "waiting_time_wq_ms": delay_s * 1000.0,
                }
            )
    return pd.DataFrame(rows)


def build_tx_queue_waiting_time_summary_df(
    waiting_times: Mapping[Any, Sequence[float]],
    relay_stats: Mapping[Any, Mapping[str, Any]],
    queue_length_samples: Mapping[Any, Sequence[Mapping[str, Any]]],
    run_metadata: Mapping[str, Any],
    queue_length_for_entity: Callable[[Any], Mapping[str, float | int]] | None = None,
    percentile_in_ms: Callable[[Sequence[float], float], float] = percentile_ms,
) -> pd.DataFrame:
    """Summarize legacy waiting-time and waiting-queue occupancy metrics."""
    rows = []
    relay_ids = set(waiting_times) | set(relay_stats) | set(queue_length_samples)
    sim_time_s = run_metadata.get("sim_time", None)

    for relay_id in sorted(relay_ids):
        delays = list(waiting_times.get(relay_id, []) or [])
        counters = relay_stats.get(relay_id, {})
        arrival_count = int(counters.get("tx_queue_arrival_count", 0))
        completed_count = int(counters.get("tx_done", 0))
        backlog_count = max(0, arrival_count - completed_count)
        completion_ratio = completed_count / arrival_count * 100.0 if arrival_count > 0 else 0.0
        if queue_length_for_entity is None:
            queue_metrics = queue_length_metrics(
                list(queue_length_samples.get(relay_id, []) or []),
                sim_time_s,
            )
        else:
            queue_metrics = queue_length_for_entity(relay_id)

        if delays:
            average_ms = round(sum(delays) / len(delays) * 1000.0, 3)
            minimum_ms = round(min(delays) * 1000.0, 3)
            p95_ms = round(percentile_in_ms(delays, 0.95), 3)
            p99_ms = round(percentile_in_ms(delays, 0.99), 3)
            maximum_ms = round(max(delays) * 1000.0, 3)
        else:
            average_ms = minimum_ms = p95_ms = p99_ms = maximum_ms = 0.0

        rows.append(
            {
                "relay": relay_id,
                "count": len(delays),
                "avg_waiting_time_wq_ms": average_ms,
                "min_waiting_time_wq_ms": minimum_ms,
                "p95_waiting_time_wq_ms": p95_ms,
                "p99_waiting_time_wq_ms": p99_ms,
                "max_waiting_time_wq_ms": maximum_ms,
                "avg_queue_length": queue_metrics["avg_queue_length"],
                "max_queue_length": queue_metrics["max_queue_length"],
                "tx_queue_arrival_count": arrival_count,
                "tx_done": completed_count,
                "tx_queue_backlog_count": backlog_count,
                "tx_queue_completion_ratio_percent": round(float(completion_ratio), 3),
            }
        )

    return pd.DataFrame(rows)


def tx_queue_state_metrics(
    samples: Sequence[Mapping[str, Any]],
    sim_time_s: float,
) -> dict[str, float | int]:
    """Integrate waiting, in-service, and system occupancy over a finite horizon."""
    try:
        horizon = max(0.0, float(sim_time_s))
    except (TypeError, ValueError):
        horizon = 0.0

    if horizon <= 0 or not samples:
        return dict(_EMPTY_QUEUE_STATE)

    cleaned = []
    for sequence, sample in enumerate(samples):
        try:
            cleaned.append(
                (
                    max(0.0, min(float(sample["time"]), horizon)),
                    sequence,
                    max(0, int(sample.get("waiting_count", 0))),
                    max(0, int(sample.get("in_service_count", 0))),
                    max(0, int(sample.get("system_count", 0))),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    if not cleaned:
        return dict(_EMPTY_QUEUE_STATE)

    cleaned.sort(key=lambda row: (row[0], row[1]))
    area_waiting = 0.0
    area_in_service = 0.0
    area_system = 0.0
    previous_time = 0.0
    previous_waiting = 0
    previous_in_service = 0
    previous_system = 0
    maximum_waiting = 0
    maximum_in_service = 0
    maximum_system = 0

    for time, _, waiting, in_service, system in cleaned:
        elapsed = max(0.0, time - previous_time)
        area_waiting += previous_waiting * elapsed
        area_in_service += previous_in_service * elapsed
        area_system += previous_system * elapsed
        previous_time = time
        previous_waiting = waiting
        previous_in_service = in_service
        previous_system = system
        maximum_waiting = max(maximum_waiting, waiting)
        maximum_in_service = max(maximum_in_service, in_service)
        maximum_system = max(maximum_system, system)

    tail = max(0.0, horizon - previous_time)
    area_waiting += previous_waiting * tail
    area_in_service += previous_in_service * tail
    area_system += previous_system * tail

    return {
        "avg_waiting_queue_length_lq": area_waiting / horizon,
        "max_waiting_queue_length": maximum_waiting,
        "avg_in_service_count": area_in_service / horizon,
        "max_in_service_count": maximum_in_service,
        "avg_system_count_l": area_system / horizon,
        "max_system_count": maximum_system,
    }


def queue_mean_ms(frame: pd.DataFrame, column: str) -> float:
    """Return the mean of a seconds-valued queue column in milliseconds."""
    if frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean() * 1000.0) if not values.empty else 0.0


def queue_percentile_ms(
    frame: pd.DataFrame,
    column: str,
    quantile: float,
) -> float:
    """Return a seconds-valued queue column percentile in milliseconds."""
    if frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.quantile(quantile) * 1000.0) if not values.empty else 0.0


def build_relay_tx_queue_metrics_df(
    nodes: Sequence[Any],
    sim_time_s: float | None,
    lifecycle: pd.DataFrame,
    relay_stats: Mapping[Any, Mapping[str, Any]],
    queue_state_samples: Mapping[Any, Sequence[Mapping[str, Any]]],
    infer_entity_type: Callable[[Any], str],
    state_metrics_for_entity: Callable[[Any, float], Mapping[str, float | int]] | None = None,
    mean_in_ms: Callable[[pd.DataFrame, str], float] = queue_mean_ms,
    percentile_in_ms: Callable[[pd.DataFrame, str, float], float] = queue_percentile_ms,
) -> pd.DataFrame:
    """Build measured FIFO/worker queue metrics for RelayTX v2."""
    try:
        horizon = max(0.0, float(sim_time_s))
    except (TypeError, ValueError):
        horizon = 0.0

    rows = []
    for entity in nodes:
        entity_id = getattr(entity, "id", None)
        entity_type = infer_entity_type(entity_id)
        if entity_type not in ("relay", "gateway"):
            continue

        capacity_c = max(
            1,
            int(getattr(getattr(entity, "hw", None), "tx_phy_radio_count", 1)),
        )
        counters = relay_stats.get(entity_id, {})
        attempt_count = int(counters.get("tx_queue_enqueue_attempt_count", 0))
        accepted_count = int(
            counters.get(
                "tx_queue_accepted_count",
                counters.get("tx_queue_arrival_count", 0),
            )
        )
        overflow_count = int(counters.get("tx_queue_overflow_drop_count", 0))
        internal_drop_count = int(counters.get("tx_queue_internal_drop_count", 0))

        sub = lifecycle[lifecycle["entity_id"] == entity_id]
        completed = sub[sub["status"] == "COMPLETED"]
        fifo_observed = sub[sub["queue_dequeue_time"].notna()]
        tx_started = sub[sub["tx_start"].notna()]
        waiting_end_count = int((sub["status"] == "CENSORED_WAITING").sum())
        in_service_end_count = int((sub["status"] == "CENSORED_IN_SERVICE").sum())
        unfinished_end_count = waiting_end_count + in_service_end_count
        completed_count = int(len(completed))

        if state_metrics_for_entity is None:
            state = tx_queue_state_metrics(
                list(queue_state_samples.get(entity_id, []) or []),
                horizon,
            )
        else:
            state = state_metrics_for_entity(entity_id, horizon)

        arrival_rate = accepted_count / horizon if horizon > 0 else 0.0
        completion_rate = completed_count / horizon if horizon > 0 else 0.0
        accumulation_rate = unfinished_end_count / horizon if horizon > 0 else 0.0
        mean_airtime_s = mean_in_ms(completed, "airtime_s") / 1000.0
        mean_worker_service_s = mean_in_ms(completed, "worker_service_s") / 1000.0
        offered_airtime_load = arrival_rate * mean_airtime_s / capacity_c
        effective_load = arrival_rate * mean_worker_service_s / capacity_c
        worker_occupied_fraction = state["avg_in_service_count"] / capacity_c
        total_airtime_s = float(getattr(entity, "total_tx_time_s", 0.0))
        observed_airtime_s = (
            float(
                pd.to_numeric(
                    sub["observed_airtime_s"],
                    errors="coerce",
                )
                .dropna()
                .sum()
            )
            if not sub.empty
            else 0.0
        )
        radio_tx_fraction = observed_airtime_s / (capacity_c * horizon) if horizon > 0 else 0.0
        completion_ratio = completed_count / accepted_count * 100.0 if accepted_count > 0 else 0.0
        conservation_delta = accepted_count - (
            completed_count + internal_drop_count + unfinished_end_count
        )
        attempt_delta = attempt_count - (accepted_count + overflow_count)

        rows.append(
            {
                "queue_metrics_version": "2.0",
                "queue_discipline": "fifo",
                "entity_id": entity_id,
                "entity_type": entity_type,
                "tx_capacity_c": capacity_c,
                "enqueue_attempt_count": attempt_count,
                "accepted_count": accepted_count,
                "overflow_drop_count": overflow_count,
                "completed_count": completed_count,
                "internal_drop_count": internal_drop_count,
                "waiting_end_count": waiting_end_count,
                "in_service_end_count": in_service_end_count,
                "unfinished_end_count": unfinished_end_count,
                "completion_ratio_percent": completion_ratio,
                "arrival_rate_pkt_s": arrival_rate,
                "completion_rate_pkt_s": completion_rate,
                "unfinished_accumulation_rate_pkt_s": accumulation_rate,
                "avg_fifo_wait_ms": mean_in_ms(fifo_observed, "fifo_wait_s"),
                "p95_fifo_wait_ms": percentile_in_ms(fifo_observed, "fifo_wait_s", 0.95),
                "p99_fifo_wait_ms": percentile_in_ms(fifo_observed, "fifo_wait_s", 0.99),
                "avg_duty_wait_ms": mean_in_ms(completed, "duty_wait_s"),
                "avg_half_duplex_wait_ms": mean_in_ms(completed, "half_duplex_wait_s"),
                "avg_time_to_tx_start_ms": mean_in_ms(tx_started, "time_to_tx_start_s"),
                "avg_worker_service_ms": mean_worker_service_s * 1000.0,
                "avg_system_sojourn_ms": mean_in_ms(completed, "system_sojourn_s"),
                **state,
                "radio_tx_fraction": radio_tx_fraction,
                "worker_occupied_fraction": worker_occupied_fraction,
                "offered_airtime_load_ratio": offered_airtime_load,
                "effective_offered_load_estimate": effective_load,
                "observed_airtime_within_window_s": observed_airtime_s,
                "total_tx_time_s": total_airtime_s,
                "attempt_conservation_delta": attempt_delta,
                "accepted_conservation_delta": conservation_delta,
                # Legacy aliases retain the airtime interpretation and are not
                # equivalent to measured worker utilization.
                "arrival_count": accepted_count,
                "tx_queue_backlog_count": unfinished_end_count,
                "tx_queue_completion_ratio_percent": completion_ratio,
                "p95_waiting_time_wq_ms": percentile_in_ms(fifo_observed, "fifo_wait_s", 0.95),
                "p99_waiting_time_wq_ms": percentile_in_ms(fifo_observed, "fifo_wait_s", 0.99),
                "avg_queue_length": state["avg_waiting_queue_length_lq"],
                "max_queue_length": state["max_waiting_queue_length"],
                "lambda_pkt_s": arrival_rate,
                "mean_service_time_s": mean_airtime_s,
                "mean_service_time_ms": mean_airtime_s * 1000.0,
                "mu_pkt_s_per_server": (1.0 / mean_airtime_s if mean_airtime_s > 0 else 0.0),
                "rho": offered_airtime_load,
                "rho_percent": offered_airtime_load * 100.0,
            }
        )

    return pd.DataFrame(rows)


def build_tx_queue_traffic_class_metrics_df(
    lifecycle: pd.DataFrame,
    infer_entity_type: Callable[[Any], str],
    mean_in_ms: Callable[[pd.DataFrame, str], float] = queue_mean_ms,
    percentile_in_ms: Callable[[pd.DataFrame, str, float], float] = queue_percentile_ms,
) -> pd.DataFrame:
    """Summarize completed and censored queue visits by traffic class."""
    if lifecycle.empty:
        return pd.DataFrame()

    rows = []
    for (entity_id, traffic_class), sub in lifecycle.groupby(
        ["entity_id", "traffic_class"],
        dropna=False,
        sort=True,
    ):
        completed = sub[sub["status"] == "COMPLETED"]
        fifo_observed = sub[sub["queue_dequeue_time"].notna()]
        rows.append(
            {
                "entity_id": entity_id,
                "entity_type": infer_entity_type(entity_id),
                "traffic_class": traffic_class,
                "visit_count": int(len(sub)),
                "completed_count": int(len(completed)),
                "overflow_drop_count": int((sub["status"] == "OVERFLOW_DROP").sum()),
                "censored_waiting_count": int((sub["status"] == "CENSORED_WAITING").sum()),
                "censored_in_service_count": int((sub["status"] == "CENSORED_IN_SERVICE").sum()),
                "avg_fifo_wait_ms": mean_in_ms(fifo_observed, "fifo_wait_s"),
                "p95_fifo_wait_ms": percentile_in_ms(fifo_observed, "fifo_wait_s", 0.95),
                "avg_duty_wait_ms": mean_in_ms(completed, "duty_wait_s"),
                "avg_half_duplex_wait_ms": mean_in_ms(completed, "half_duplex_wait_s"),
                "avg_worker_service_ms": mean_in_ms(completed, "worker_service_s"),
                "avg_system_sojourn_ms": mean_in_ms(completed, "system_sojourn_s"),
            }
        )
    return pd.DataFrame(rows)


def build_relay_tx_mgc_queue_metrics_legacy_df(
    nodes: Sequence[Any],
    sim_time_s: float | None,
    relay_stats: Mapping[Any, Mapping[str, Any]],
    waiting_times: Mapping[Any, Sequence[float]],
    queue_length_samples: Mapping[Any, Sequence[Mapping[str, Any]]],
    infer_entity_type: Callable[[Any], str],
    queue_length_for_entity: Callable[[Any, float | None], Mapping[str, float | int]] | None = None,
    percentile_in_ms: Callable[[Sequence[float], float], float] = percentile_ms,
) -> pd.DataFrame:
    """Build the deprecated airtime-based M/G/c compatibility table."""
    rows = []
    horizon = float(sim_time_s) if sim_time_s is not None else None

    for entity in nodes:
        entity_id = getattr(entity, "id", None)
        if entity_id is None:
            continue

        entity_type = infer_entity_type(entity_id)
        if entity_type not in ("relay", "gateway"):
            continue

        capacity_c = max(
            1,
            int(getattr(getattr(entity, "hw", None), "tx_phy_radio_count", 1)),
        )
        counters = relay_stats.get(entity_id, {})
        arrival_count = int(counters.get("tx_queue_arrival_count", 0))
        completed_count = int(counters.get("tx_done", 0))
        total_tx_time_s = float(getattr(entity, "total_tx_time_s", 0.0))
        lambda_pkt_s = arrival_count / horizon if horizon and horizon > 0 else 0.0
        mean_service_time_s = total_tx_time_s / completed_count if completed_count > 0 else 0.0
        mu_pkt_s_per_server = 1.0 / mean_service_time_s if mean_service_time_s > 0 else 0.0
        rho = (
            lambda_pkt_s / (capacity_c * mu_pkt_s_per_server)
            if capacity_c > 0 and mu_pkt_s_per_server > 0
            else 0.0
        )
        backlog_count = max(0, arrival_count - completed_count)
        completion_ratio = completed_count / arrival_count * 100.0 if arrival_count > 0 else 0.0
        delays = list(waiting_times.get(entity_id, []) or [])
        if queue_length_for_entity is None:
            queue_metrics = queue_length_metrics(
                list(queue_length_samples.get(entity_id, []) or []),
                horizon,
            )
        else:
            queue_metrics = queue_length_for_entity(entity_id, horizon)

        rows.append(
            {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "tx_capacity_c": capacity_c,
                "arrival_count": arrival_count,
                "completed_count": completed_count,
                "tx_queue_backlog_count": backlog_count,
                "tx_queue_completion_ratio_percent": round(float(completion_ratio), 3),
                "p95_waiting_time_wq_ms": round(float(percentile_in_ms(delays, 0.95)), 3),
                "p99_waiting_time_wq_ms": round(float(percentile_in_ms(delays, 0.99)), 3),
                "avg_queue_length": queue_metrics["avg_queue_length"],
                "max_queue_length": queue_metrics["max_queue_length"],
                "lambda_pkt_s": round(lambda_pkt_s, 6),
                "mean_service_time_s": round(mean_service_time_s, 6),
                "mean_service_time_ms": round(mean_service_time_s * 1000.0, 3),
                "mu_pkt_s_per_server": round(mu_pkt_s_per_server, 6),
                "rho": round(rho, 6),
                "rho_percent": round(rho * 100.0, 3),
                "total_tx_time_s": round(total_tx_time_s, 6),
            }
        )

    return pd.DataFrame(rows)
