# Scalability and relay-bottleneck experiment

This package owns the active exploratory scalability Pilot and the separate
Relay Duty sensitivity. The existing `pilot_v1.json` is preserved unchanged.

**R1.7 Load status:** deprecated. The official R1.7 production experiment is
[`apc_load_sweep`](../apc_load_sweep/README.md), using contract
`apc_load_sweep_full_factorial_v1` and 120 FF/AA runs. The two legacy files
`configs/r1_7_load_ff_v1.json` and `configs/r1_7_load_aa_v1.json` remain in the
repository for historical compatibility, but must not be used for the active
production experiment. Historical source and outputs are retained.

```bash
python -m loraphymac.experiments.scalability_relay_bottleneck.run --dry-run
python -m loraphymac.experiments.scalability_relay_bottleneck.run
```

## Active scalability Pilot

Run the exploratory pilot from the repository root:

```bash
python -m loraphymac.experiments.scalability_relay_bottleneck.run --dry-run
python -m loraphymac.experiments.scalability_relay_bottleneck.run
```

The pilot configuration and output namespace are independently versioned. Its
historical run status should be checked in its own manifest; it is not
automatically the official R1.7 dataset.

## Active Relay Duty sensitivity

Relay Duty is a separate sensitivity experiment and has not been replaced by
APC Load Sweep. Its configurations compare relay duty limits of 10% and 1% at
100 nodes / 30 seconds, for FF and AA with five seeds (20 planned runs total):

```bash
python -m loraphymac.experiments.scalability_relay_bottleneck.run src/loraphymac/experiments/scalability_relay_bottleneck/configs/r1_7_relay_duty_ff_v1.json
python -m loraphymac.experiments.scalability_relay_bottleneck.run src/loraphymac/experiments/scalability_relay_bottleneck/configs/r1_7_relay_duty_aa_v1.json
```

Node duty is measured, not enforced. `channel_occupancy.csv` reports both
summed TX airtime and the interval-union of modeled transmissions per channel;
it is not a receiver sensing measurement or a regulatory duty-cycle compliance
claim. Per-node duty and occupancy tables are compact run-level outputs; raw
event logs remain disabled.

## Historical R1.7 Load compatibility

`configs/r1_7_load_ff_v1.json` and `configs/r1_7_load_aa_v1.json` describe the
superseded 90-run grid (three node counts × three intervals × five seeds × two
modes). They are retained unchanged for provenance/backward compatibility and
are not production instructions. Do not combine their outputs with the active
120-run APC Load Sweep cohort.

Experiment evidence is written below
`artifacts/experiments/scalability_relay_bottleneck/<experiment_id>/`.
Publication-specific transformations live in
`loraphymac.publications.paper2_scalability`.
