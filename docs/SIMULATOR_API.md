# Simulator API

New code should use the stable `loraphymac` package instead of importing the
large legacy module directly. The API resolves defaults, rejects unknown or
invalid configuration, executes one deterministic simulation, and returns a
named result object.

The current simulator release is V1.0 (`1.0.0`) and the public calling
interface remains `simulator_api_v1`. These versions are intentionally
independent; see [`VERSIONING.md`](VERSIONING.md).

```python
from loraphymac import SIMULATOR_API_VERSION, SIMULATOR_VERSION, simulate

print(SIMULATOR_VERSION)      # 1.0.0
print(SIMULATOR_API_VERSION)  # simulator_api_v1

result = simulate(
    {
        "run_id": "example-s42",
        "seed": 42,
        "sim_time": 600,
        "node_num": 20,
    }
)

print(result.summary["delivery_by_horizon_percent"])
packet_fates = result.table("packet_fate_summary_df", copy=True)
```

`SimulationResult` contains:

- `resolved_config`: complete validated configuration used by the engine;
- `summary`: one-run scalar metrics;
- `tables`: all named DataFrame evidence;
- `ptx_history` and `margin_history`: convenience properties;
- `to_legacy_tuple()`: temporary adapter for callers not yet migrated.

The returned mappings are read-only. Detailed DataFrames are shared to avoid a
large automatic memory copy; request `copy=True` before modifying a table.

## Compatibility boundary

`loraphymac.engine.run_single_simulation()` remains available as a thin
compatibility facade. Its implementation now lives in
`loraphymac.simulation.runner`; the shared multi-seed pipeline uses
`simulate()`, so notebooks and paper runners do not depend on either internal
module or on the legacy four-item tuple. This relocation does not change
simulator equations, event order, random-number generation, or output tables.

Internal construction is split under `loraphymac.simulation`: `context.py`
groups objects belonging to one run, `builders.py` constructs infrastructure,
topology, and routing, `state.py` initializes radio/APC state, and `results.py`
assembles scalar summary fields. `runner.py` only registers processes, executes
the SimPy horizon, and coordinates the post-run boundary. Table collection is
owned by `output_collection.py`, configured console views by `reporting.py`,
and workbook writes by `exporting.py`. Canonical defaults and the configuration
merge helper are in `loraphymac.config.defaults`. These modules are not the
supported public API.

Four short fixed-seed FF/AF/FA/AA scenarios form a numerical regression gate.
Any further engine extraction must preserve their packet, PDR, delay, energy,
final-power, and APC-control results unless an intentional scientific model
change is separately documented and versioned.


## Maintained single-run workflows

Two thin workflows use this API and the same diagnostic builders.
`single_run_exploration` is flexible and carries no publication contract;
`apc_closed_loop_verification` freezes and validates the Paper 1 behavior
check. Their notebooks, CLI commands, and output namespaces are documented in
`OUTPUT_LAYOUT.md`.
