# APC Feedback Robustness Sweep v1

This is an opt-in experiment profile for the approved 30-run AA matrix. It
compares 30 s and 90 s application periods at additional receiver-side R-APC
feedback erasure probabilities 0%, 10%, and 30%, with seeds 30, 36, 42, 48,
56. All other scenario settings are frozen in `configs/aa_feedback_robustness_v1.json`.

The synthetic erasure is evaluated only for a decoded APC-marked `DATA_DOWN`
or `DATA_DOWN_FWD` frame whose metadata identifies the current target. It is
evaluated after native RX/collision/half-duplex/window feasibility and before
the APC command is parsed/applied. Erasure does not call `record_drop`, does
not affect application packet fate, and is not classified as a collision.
Retries retain the logical `feedback_id`; each physical TX attempt gets a new
attempt ID. The erasure sample is SHA-256 keyed by policy, seed, controller,
target, logical-decision ordinal, and attempt number. Loss probability and
application period are not part of that key. No global RNG is used.

The five contract tables are persisted per run:

- `feedback_attempt_metrics.csv`: physical APC attempts, native outcome,
  synthetic erasure, delivery/application status, and observed airtime.
- `feedback_transaction_metrics.csv`: logical decisions, attempts, delivery,
  application latency, retry exhaustion, and censoring status.
- `stale_tpi_by_transmitter.csv`: per-transmitter/per-message-type physical
  DATA-TX counts, including pre-decision, stale, current, and unclassified
  outcomes. Its denominator is eligible post-decision TX only; zero eligible
  TX is represented by a null fraction. The streaming table does not retain a
  raw event history.
- `apc_convergence_metrics.csv`: link-margin first-entry proxy and
  right-censored records; this is not stable convergence.
- `apc_control_airtime_summary.csv`: APC physical TX airtime and count,
  broken down by transmitter role and MType, divided by total observed TX
  airtime from the same recorder.

Core PDR, delay, node/relay/gateway energy, corrected energy accounting,
packet-fate, collision, and provenance remain supplied by the existing
scientific output pipeline. The pinned corrected-energy implementation is
`eq10_exclusive_transition_v1`.

Dry-run the matrix without starting simulations:

```bash
MPLBACKEND=Agg "$HOME/Documents/LoRaPhyMAC/.venv-output-profile/bin/python" \
  -m loraphymac.experiments.apc_feedback_robustness_sweep.run --dry-run
```

The production command is deliberately not included here; this implementation
stage does not authorize the 30 simulations.
