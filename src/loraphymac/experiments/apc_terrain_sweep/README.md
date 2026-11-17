# APC terrain sweep experiment

This package owns the publication-independent 4 APC modes x 15 terrain values
x 5 seeds experiment (300 runs). It contains only the executable scientific
contract, versioned experiment configuration, and runner.

Validate the grid:

```bash
python -m loraphymac.experiments.apc_terrain_sweep.run --dry-run
```

Run or resume it:

```bash
python -m loraphymac.experiments.apc_terrain_sweep.run
```

New experiment evidence is written below
`artifacts/experiments/apc_terrain_sweep/<experiment_id>/`. Once the complete
grid passes, the workflow also creates experiment-owned tables, figures, and
`analysis_manifest.json` below that same versioned directory. The notebook
displays the PNG figures as a read-only publication preview.

The historical `production_v1.json` keeps `master_table_mode="full"` and its
original experiment ID/output namespace. For a separate lightweight-output
experiment, use `configs/aggregate_inputs_v1.json`; it has a distinct
experiment ID and output namespace. Its `master/` contains only
`all_runs_core_summary.csv` and `per_run_relay_bottleneck_metrics.csv`, while
analysis-grade per-run evidence remains below `runs/`. Do not use the aggregate
configuration to resume the historical v1 experiment. Completed `runs/` may
be archived only after archive integrity and publication regeneration have been
verified; `aggregate/`, manifests, `tables/`, and `figures/` remain active.

Validate the lightweight configuration without running simulations:

```bash
python -m loraphymac.experiments.apc_terrain_sweep.run \
  src/loraphymac/experiments/apc_terrain_sweep/configs/aggregate_inputs_v1.json \
  --dry-run
```

The separate `loraphymac.publications.paper_rapc_multihop_lora` workflow does
not execute simulations. It selects the terrain-sweep artifacts together with
the contracted closed-loop verification artifacts, assigns manuscript names,
and records multi-source provenance.
