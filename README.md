# LoRaPhyMAC

LoRaPhyMAC is an event-driven research simulator for LoRa physical/MAC
behavior, multi-hop relay forwarding, collision and capture, finite relay
resources, energy accounting, and adaptive transmit-power control (APC).

Current simulator release: **V1.0** (Python package version `1.0.0`). Release,
API, experiment-contract, dataset, and metric-schema versions are independent;
see [`docs/VERSIONING.md`](docs/VERSIONING.md).
Release history is recorded in [`CHANGELOG.md`](CHANGELOG.md).

The project is intended for reproducible simulation studies. It is not a full
LoRaWAN protocol implementation and must not be treated as an interoperability
or certification test tool.

## Current research scope

- Paper 1: Adaptive Power Control (APC).
- Paper 2: Scalability and relay bottleneck analysis.
- Future studies reuse content-based experiments; one completed experiment may
  support one or more publication-specific analyses.

The frozen Paper 1 design is documented in
[`docs/PAPER1_APC_CONTRACT.md`](docs/PAPER1_APC_CONTRACT.md) and provided as the
machine-readable
[`src/loraphymac/experiments/apc_terrain_sweep/configs/production_v1.json`](src/loraphymac/experiments/apc_terrain_sweep/configs/production_v1.json)
experiment specification. Execution and publication concerns are separated:

- `src/loraphymac/experiments/apc_terrain_sweep/` for the 300-run APC grid;
- `src/loraphymac/experiments/apc_load_sweep/` for the official IJIES Reviewer R1.7 load experiment;
- `src/loraphymac/experiments/scalability_relay_bottleneck/` for the scalability pilot and Relay Duty sensitivity (the former R1.7 Load configs there are legacy/deprecated);
- `src/loraphymac/experiments/single_run_exploration/` for flexible one-run research;
- `src/loraphymac/experiments/apc_closed_loop_verification/` for the contracted Paper 1 behavior check;
- `src/loraphymac/experiments/common/` for reusable execution infrastructure.
- `src/loraphymac/publications/paper_rapc_multihop_lora/` for R-APC multi-hop
  LoRa tables and figures;
- `src/loraphymac/publications/paper2_scalability/` for Paper 2 outputs.

### APC Load Sweep and IJIES R1.7

The APC Load runner defaults to the Minimal APC Load Standard V1.0 contract.
Its [`standard_v1.json`](src/loraphymac/experiments/apc_load_sweep/configs/standard_v1.json)
specifies the paired AA/FF grid: four node counts × three reporting periods ×
five seeds, for 120 runs. A read-only grid preview is available with:

```bash
PYTHONPATH=src python -m loraphymac.experiments.apc_load_sweep.run --dry-run
```

Selecting `--profile pilot` requires an isolated `--output-root`; pilot output
is two-run, non-production integration evidence. Standard cohorts use the
`sweep_cohort_v1` layout, reserve a new cohort for each execution, and do not
resume an existing cohort. A Standard production sweep was not run as part of
this release preparation.

The IJIES R1.7 full-factorial workflow remains explicitly available through
the positional legacy configurations
[`ff_full_factorial_v1.json`](src/loraphymac/experiments/apc_load_sweep/configs/ff_full_factorial_v1.json)
and
[`aa_full_factorial_v1.json`](src/loraphymac/experiments/apc_load_sweep/configs/aa_full_factorial_v1.json).
Each config plans 60 runs and uses the legacy `paper_artifacts_v1` lifecycle.
The R1.7 production launcher continues to pass these configs explicitly, so
its experiment parameters and historical output namespace remain unchanged.
The latest validated R1.7 rerun completed with FF 60/60 PASS and AA 60/60
PASS. Generated run evidence is not included in this source-code release;
results are written locally under
`artifacts/experiments/apc_load_sweep/full_factorial_v1/`.

Legacy execution resumes completed runs by default; pass `--no-resume` to
disable that behavior. Standard execution always reserves a new cohort and
does not resume. See the [APC Load guide](src/loraphymac/experiments/apc_load_sweep/README.md)
for explicit Standard and legacy commands and their output layouts.

The older `scalability_relay_bottleneck` R1.7 Load configs are retained for
historical compatibility, but are not the active production R1.7 experiment.
That package remains active for its exploratory Pilot and separate Relay Duty
sensitivity; see its [README](src/loraphymac/experiments/scalability_relay_bottleneck/README.md).
The production launcher and monitor are `tools/run_r17_production.sh` and
`tools/monitor_r17.py`; both operate on the APC Load Sweep namespace.

When experiments are executed, persistent generated results are written
below the repository-local `artifacts/` root. This generated directory is
intentionally excluded from the initial source-only release. The following
layout describes where local runs write their outputs:

```text
artifacts/
├── experiments/single_run_exploration/<run_id>/
├── experiments/apc_closed_loop_verification/<run_id>/
├── experiments/<experiment_name>/<experiment_id>/
├── publications/paper_rapc_multihop_lora/paper_rapc_multihop_lora_v1/
├── publications/paper2_scalability/<dataset_version>/
└── archive/                         # clearly labelled legacy evidence
```

Experiment directories contain run evidence, master data, aggregates, logs,
and experiment-owned diagnostic `tables/` and `figures/`. Publication
directories contain selected or combined manuscript `tables/` and `figures/`. The
historical Paper 1 dataset remains at its original path so its manifest and
fingerprints are not rewritten. Both single-run experiment directories contain their workbooks and topology
plus `figures/` and `tables/` directories. The closed-loop directory also
contains a machine-readable `verification_report.json` PASS marker. Smoke tests use
temporary directories and do not create persistent repository output.

Each experiment and publication has a separate thin notebook. Experiment
notebooks execute a grid and render their own preview artifacts; publication
notebooks only read completed experiments and assemble manuscript outputs.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Verification

Run the complete test suite:

```bash
pytest -q
```

The Minimal APC Load Standard release gate excludes tests requiring frozen
research evidence (`extended_research`) and Unified Standard features outside
this release (`unreleased_standard`). Run it with
`pytest -q -m "not extended_research and not unreleased_standard"`. Excluded
tests are not claimed as passing; see the [Test Suite Guide](docs/TESTING.md)
for markers and commands to run them when their requirements are available.

Validate the Paper 1 production grid without executing simulations:

```bash
python -m loraphymac.experiments.apc_terrain_sweep.run --dry-run
```

Inspect the current Paper 2 draft grid separately:

```bash
python -m loraphymac.experiments.scalability_relay_bottleneck.run --dry-run
```

Validate the completed Paper 1 source dataset without writing output, then
generate its tables, figures, and provenance manifest:

```bash
python -m loraphymac.publications.paper_rapc_multihop_lora.run --check-only
python -m loraphymac.publications.paper_rapc_multihop_lora.run
```

## Simulator API

Use the stable API for new single-run code:

```python
from loraphymac import simulate

result = simulate({"seed": 42, "sim_time": 600, "node_num": 20})
print(result.summary)
```

Two maintained single-run notebooks use the same simulator and shared
diagnostics. Use
[`single_run_exploration.ipynb`](src/loraphymac/experiments/single_run_exploration/notebooks/single_run_exploration.ipynb)
for freely configurable preliminary research, and
[`apc_closed_loop_verification.ipynb`](src/loraphymac/experiments/apc_closed_loop_verification/notebooks/apc_closed_loop_verification.ipynb)
for the contracted Paper 1 closed-loop behavior check. The repository-root
`single_run_experiment.ipynb` is the legacy predecessor.
The maintained replacement for running the root `multi_run_experiments.ipynb`
is [`apc_terrain_sweep_workflow.ipynb`](src/loraphymac/experiments/apc_terrain_sweep/notebooks/apc_terrain_sweep_workflow.ipynb).
The separate R-APC multi-hop LoRa rendering workflow is
[`paper_rapc_multihop_lora_workflow.ipynb`](src/loraphymac/publications/paper_rapc_multihop_lora/notebooks/paper_rapc_multihop_lora_workflow.ipynb).

The legacy `run_single_simulation()` tuple remains available temporarily, but
paper runners and new notebooks should depend on `SimulationResult`. See
[`docs/SIMULATOR_API.md`](docs/SIMULATOR_API.md).

## Repository policy

GitHub stores source code, tests, reusable experiment configurations, and
documentation. Generated workbooks, notebook outputs, caches, plots, and raw
multi-seed results are intentionally ignored. Large research outputs should be
published separately in a versioned research-data archive and linked by DOI.

The simulator records configuration/source fingerprints and finite-horizon
packet fates to support reproducibility. A complete paper run should be
performed only after all tests and smoke simulations pass.

## Source layout

The installable code uses the standard `src` layout:

```text
src/loraphymac/
├── simulator.py    # stable public single-run API
├── engine.py       # backward-compatible facade only
├── simulation/     # runner, builders, output collection/report/export
├── config/         # canonical defaults and centralized validation
├── domain/         # simulation entities and state accounting
├── hardware/       # radio and MCU models
├── phy/            # channel, air-interface, and propagation models
├── mac/            # frame and collision models
├── runtime/        # event-driven TX/RX processes
├── metrics/        # statistics and scientific outputs
├── experiments/    # content-based grids, contracts, and runners
└── publications/   # manuscript-specific tables, figures, notebooks
```

All internal and test imports use the `loraphymac.*` namespace. Repository
root modules and peer top-level packages are no longer part of the source
layout.

The internal engine is decomposed behind `simulator.py` in regression-protected
steps. `simulation/runner.py` owns event registration and execution;
infrastructure, topology, routing, mutable PTx state, and summary assembly are
separate internal modules. Post-horizon table collection, console reporting,
and workbook export are isolated in `output_collection.py`, `reporting.py`, and
`exporting.py`. `engine.py` only re-exports the historical entry points so
existing scripts remain compatible. Canonical defaults and config merging live
in `config/defaults.py`.

Runtime metric recording is separated from read-only scientific table
derivation behind the stable `Statistics` facade. Module ownership, dependency
rules, units, conservation invariants, and the extension checklist are defined
in [`docs/METRICS_ARCHITECTURE.md`](docs/METRICS_ARCHITECTURE.md).

All single-run and multi-seed configurations pass through the same fail-fast
validator before the simulation starts. Validation covers types and ranges,
LoRa PHY combinations, topology/ITM limits, EIRP-compliant power levels,
collision matrices, APC/frame dependencies, queue settings, duty cycle, and
paper-specific experiment profiles. See `docs/CONFIGURATION.md`.

## License and third-party code

LoRaPhyMAC is released under the MIT License. The vendored `pyitm` code has its
own MIT copyright notice and retains the required NTIA disclaimer; see
`THIRD_PARTY_NOTICES.md`.
