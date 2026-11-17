# Changelog

All notable changes to LoRaPhyMAC are documented in this file. Release numbers
follow Semantic Versioning; experiment contracts, datasets, metric schemas, and
publication workflows keep their own independent version identifiers.

## [Unreleased]

### Added

- Separate flexible `single_run_exploration` workflow with a general validated
  profile, dedicated config, thin notebook, CLI, and artifact namespace.
- Contracted `apc_closed_loop_verification_v1` one-run experiment for Paper 1,
  including output-invariant checks and a machine-readable PASS report.

### Changed

- Removed the former combined `experiments/single_run` package; its reusable
  figures and tables now live in `experiments/common/single_run_diagnostics`.
- Removed the redundant `experiments/paper1_apc` compatibility package;
  experiment execution remains in `apc_terrain_sweep` and manuscript outputs
  remain in `publications/paper_rapc_multihop_lora`.
- Removed the redundant `experiments/paper2_scalability_relay_bottleneck`
  compatibility package; the executable pilot remains in
  `experiments/scalability_relay_bottleneck`.

## [1.0.0] - 2026-09-13

### Added

- Stable `loraphymac` package under the `src/` layout and a public simulator API.
- Centralized configuration defaults, validation, and fail-fast scientific checks.
- Reproducible single-run and multi-seed experiment workflows with manifests,
  environment records, source fingerprints, completion gates, and progress output.
- Content-based APC terrain-sweep and scalability/relay-bottleneck experiment packages.
- Separate publication workflows with artifact hashes and provenance manifests.
- Versioned output layout under `artifacts/` for experiments, publications, and
  single-run diagnostics.
- Scientific, characterization, integration, and unit regression tests.

### Changed

- Split simulation setup, event execution, output collection, reporting, and export
  into focused modules behind the stable simulator API.
- Decomposed metric derivation into packet, collision, demodulator, receiver, energy,
  duty-cycle, queue, frame/event, core-summary, and APC/PTX modules while retaining
  the `Statistics` compatibility facade.
- Replaced paper-number-based experiment ownership with content-based experiment
  packages; compatibility import paths remain available for existing notebooks.
- Standardized Paper 1 publication figures and tables, including Times New Roman
  styling and deterministic figure ordering.

### Fixed

- Enforced the configured simulation horizon in energy accounting.
- Rejected unknown or scientifically invalid configuration combinations centrally.
- Prevented publication generation from incomplete multi-seed grids.
- Limited verbose demodulator logs in notebooks while preserving complete exported
  diagnostics.
- Corrected AS923 RX1 data-rate mapping for supported LoRa data rates according to
  LoRaWAN Regional Parameters RP002-1.0.5, including RX1 offset and downlink dwell time.
