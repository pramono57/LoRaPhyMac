# Metrics architecture

LoRaPhyMAC separates simulation-time instrumentation from read-only scientific
analysis. This boundary prevents presentation or publication changes from
silently changing the simulated physical/MAC process.

## Data flow

```text
runtime TX/RX/collision processes
              │ record_* calls
              ▼
       Statistics raw state
              │ stable get_* facade
              ▼
      metric-family builders
              │ pandas DataFrames
              ▼
 core summary / output bundles
              │
              ├── output collection
              ├── console views
              └── Excel export
```

The runtime may write only through `Statistics` recorder and finalization
methods. Metric-family modules receive recorded state or a facade and must not
mutate the simulator, entities, packets, or configuration.

Experiment and publication code consumes the resulting tables. Paper-specific
filters, labels, aggregation grids, and figures belong under `experiments/` or
`publications/`, not in the generic metrics layer.

## Module ownership

| Module | Responsibility | May mutate simulation state? |
|---|---|---:|
| `statistics.py` | Raw state, `record_*` methods, finite-horizon finalization, stable `get_*` facade | Yes |
| `packet_metrics.py` | Packet-fate, delivery, and terminal-failure tables | No |
| `collision_metrics.py` | Collision decisions, interference edges, and collision summary | No |
| `demodulator_metrics.py` | Demodulator event log and pool summary | No |
| `receiver_metrics.py` | Relay RX, channel use, node RX matrix, and hop summary | No |
| `energy_metrics.py` | Entity energy components and network-ready energy table | No |
| `duty_cycle_metrics.py` | Aggregate and per-channel TX duty cycle | No |
| `queue_metrics.py` | Queue lifecycle, finite-horizon occupancy, waiting time, load, and conservation | No |
| `frame_metrics.py` | Normalized frame events, MType, hop, drop, and frame/APC summary | No |
| `apc_metrics.py` | PTx history, APC control history, margin history, and final PTx distribution | No |
| `core_summary.py` | Cross-metric, single-row simulation summary | No |
| `output_bundles.py` | Stable summary/result/debug workbook composition | No |
| `views.py` | Compact notebook and console views | No |
| `excel_export.py` | DataFrame-bundle serialization to Excel | External file only |

`core_summary.py` and `output_bundles.py` intentionally consume the stable
facade because they coordinate several metric families. Metric-family modules
must not import `Statistics`; this keeps the dependency direction acyclic.

## Facade contract

Existing callers continue to import:

```python
from loraphymac.metrics.statistics import Statistics
```

Public `Statistics.get_*` signatures, returned table names, column names, and
units are compatibility boundaries. An extraction may replace an inline
calculation with delegation, but it must not change those observable results.

The facade retains:

- raw state containers and runtime recorders;
- packet identity and event-timing helpers required while recording;
- finite-horizon packet-fate finalization;
- lightweight raw snapshots such as `to_dataframe()` and run metadata;
- compatibility wrappers for all derived tables, bundles, views, and exports.

The facade must not regain paper-specific plotting or new derived DataFrame
aggregation. New derived metrics belong in a focused sibling module.

## Scientific contracts

### Packet fate and horizon

- PDR is delivery by the configured hard observation horizon.
- Every offered original uplink must be delivered, terminally dropped, or
  right-censored.
- `offered = delivered + terminal_drop + censored`.
- Censored packets are reported separately and are not silently counted as a
  physical/MAC drop.

### Collision and demodulation

- Collision decisions and directed interference edges are separate evidence.
- Collision outcome conservation must remain zero.
- Singleton, capture, SINR-survivor, and collision-drop outcomes retain their
  distinct meanings.
- Demodulator exhaustion is distinct from RF collision and half-duplex loss.

### Relay queue

- Occupancy is integrated as a piecewise-constant process over the finite
  simulation horizon.
- Waiting, in-service, and total-system occupancy remain separate.
- `enqueue_attempt = accepted + overflow_drop`.
- `accepted = completed + internal_drop + unfinished_at_horizon`.
- Waiting and in-service visits at the horizon are censored, not completed.
- `queue_metrics_version=2.0` is the canonical measured queue contract; the
  historical M/G/c fields remain compatibility aliases.

### Energy and duty cycle

- Internal entity energy may be accumulated in mJ; exported energy tables use
  joules.
- Network energy includes node, relay, and gateway energy.
- Constrained energy includes node and relay energy only.
- Aggregate relay duty cycle and per-channel duty cycle are separate metrics.

### APC and transmit power

- `final_ptx`, `old_ptx`, and `new_ptx` are physical power in dBm.
- `final_tx_power_index`, `old_tx_power_index`, and
  `new_tx_power_index` are APC index-domain values.
- Decision/uplink margin and physical APC-downlink RX margin are distinct.
- Legacy `last_margin`, `last_rssi`, and `last_snr` aliases resolve to the
  decision/uplink observations when available.

### Frame metrics

- An original uplink TX is `DATA_UP` with `hop_index == 0`.
- Frame PDR uses delivered packets divided by original uplink TX events.
- Physical APC success uses `DATA_DOWN` and `DATA_DOWN_FWD` TX/RX events.
- Frame drop causes keep collision, half-duplex, infeasible-link, and
  demodulator failures separate.

## Adding a metric safely

1. Record the minimum raw evidence through `Statistics` or an owning runtime
   component. Do not calculate publication aggregates inside an event loop.
2. Put deterministic, read-only derivation in the appropriate metric-family
   module. Create a new focused module only when ownership is genuinely new.
3. Expose the result through a thin `Statistics.get_*` wrapper so existing
   notebook and simulator boundaries remain stable.
4. Add the table to `output_bundles.py` only when it belongs in the maintained
   summary, result, or debug contract.
5. Test empty input, malformed optional values, units, column order, and at
   least one independently calculated scientific invariant.
6. Run the fixed-seed regression tests and the complete test suite before
   accepting the change.
7. Put paper-specific processing in the relevant publication module and record
   its provenance; never specialize a generic simulator metric for one paper.

## Compatibility and deprecation

Compatibility aliases must be documented in code and covered by tests. A
legacy field or method may be removed only in an explicitly versioned API
change after all notebooks, experiment runners, publication builders, and
stored-data readers have migrated.
