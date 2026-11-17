# Generated-output layout

LoRaPhyMAC uses exactly one persistent generated-output root:

```text
artifacts/
├── experiments/
│   ├── single_run_exploration/<run_id>/
│   │   ├── summary_*.xlsx
│   │   ├── results_*.xlsx
│   │   ├── debug_*.xlsx
│   │   ├── <run_id>_routing_topology.jpg
│   │   ├── figures/
│   │   └── tables/
│   ├── apc_closed_loop_verification/<run_id>/
│   │   ├── verification_report.json
│   │   ├── summary_*.xlsx
│   │   ├── results_*.xlsx
│   │   ├── debug_*.xlsx
│   │   ├── <run_id>_routing_topology.jpg
│   │   ├── figures/
│   │   └── tables/
│   ├── apc_terrain_sweep/<experiment_id>/{runs,master,aggregate,tables,figures,logs}/
│   └── scalability_relay_bottleneck/<experiment_id>/{runs,master,aggregate,tables,figures,logs}/
├── publications/
│   ├── paper_rapc_multihop_lora/<dataset_version>/{publication_manifest.json,tables,figures}/
│   └── paper2_scalability/<dataset_version>/{tables,figures}/
└── archive/
    ├── single_run/       # superseded manual single-run evidence
    └── paper1_apc/       # superseded APC/chapter outputs
```

## Workflow ownership

| Workflow | Maintained notebook | Output namespace |
|---|---|---|
| Flexible single-run exploration | `experiments/single_run_exploration/notebooks/single_run_exploration.ipynb` | `artifacts/experiments/single_run_exploration/` |
| Contracted APC closed-loop verification | `experiments/apc_closed_loop_verification/notebooks/apc_closed_loop_verification.ipynb` | `artifacts/experiments/apc_closed_loop_verification/` |
| APC terrain-sweep experiment | `experiments/apc_terrain_sweep/notebooks/apc_terrain_sweep_workflow.ipynb` | `artifacts/experiments/apc_terrain_sweep/` |
| R-APC multi-hop LoRa publication | `publications/paper_rapc_multihop_lora/notebooks/paper_rapc_multihop_lora_workflow.ipynb` | `artifacts/publications/paper_rapc_multihop_lora/` |
| Scalability/relay experiment | `experiments/scalability_relay_bottleneck/notebooks/scalability_relay_bottleneck_workflow.ipynb` | `artifacts/experiments/scalability_relay_bottleneck/` |
| Paper 2 scalability publication | `publications/paper2_scalability/notebooks/paper2_scalability_workflow.ipynb` | `artifacts/publications/paper2_scalability/` |

The two single-run workflows share the figure and table builders in
`experiments/common/single_run_diagnostics/`, but they do not share
configuration semantics.
Exploration uses the general profile and permits scientific overrides. The APC
closed-loop workflow uses a frozen versioned contract and writes
`verification_report.json` only after its output invariants pass.

The old repository-root `single_run_experiment.ipynb` and
`multi_run_experiments.ipynb` are legacy notebooks. Neither defines a new
output namespace.

Smoke tests must use a temporary directory. Names such as `outputs_latest` or
`outputs_smoke` are prohibited as persistent output namespaces. Everything
below `artifacts/` is generated and therefore excluded from Git.

Experiment data is immutable input to a publication workflow. A publication
configuration explicitly records `source_experiment_dir` or role-labelled
`sources`, plus `output_dir`, so
generating a table or figure cannot overwrite a run, manifest, master table, or
aggregate. The validated R-APC source stays at
`artifacts/experiments/apc_terrain_sweep/apc_terrain_sweep_v1_r01`; its newly
generated tables and figures use the separated publication namespace. The experiment layout-v1
`tables/` and `figures/` directories are owned by experiment postprocessing
and may be empty until that postprocessing runs; publication assemblers read
but do not write those source directories.

Large multi-seed experiments may declare
`master_table_mode="aggregate_inputs"`. In that mode the per-run evidence is
still written normally, but `master/` materializes only the compact core
summary and relay-bottleneck input required for aggregation. Detailed master
tables are reproducible caches and are not duplicated. After an experiment is
complete, verified raw-run archives belong below
`artifacts/archive/<experiment>/<experiment_id>_runs.tar.zst`; the active
experiment keeps its aggregate, provenance manifests, tables, and figures.

`publication_manifest.json` is written last and is the completion marker for
an R-APC publication build. It records both source experiment identities,
recorded scientific source fingerprint, contract/pipeline/API versions,
publication configuration checksum, generator-source checksums, and SHA-256
plus byte size for every generated table and figure. A directory without this
manifest must not be treated as a complete publication artifact set.

For either single run, `summary_*.xlsx` is the compact scientific overview,
`results_*.xlsx` contains analysis-grade tables and histories, and
`debug_*.xlsx` contains raw event/lifecycle evidence. The `tables/` directory
remains flat and is reserved for maintained validation tables; it does not
duplicate workbook sheets.
