# R-APC Multi-Hop LoRa publication

This package assembles manuscript-specific tables and figures from two
validated experiments: the completed 300-run APC terrain sweep and the
contracted one-run APC closed-loop verification. Source roles and the
publication destination are selected by `configs/publication_v1.json`.

The default mapping reads the terrain-sweep artifacts at
`artifacts/experiments/apc_terrain_sweep/apc_terrain_sweep_v1_r01`, combines
them with closed-loop artifacts from
`artifacts/experiments/apc_closed_loop_verification/apc_closed_loop_dh90_s42`,
and writes the assembled files to
`artifacts/publications/paper_rapc_multihop_lora/paper_rapc_multihop_lora_v1`.

Validate the source without writing output:

```bash
python -m loraphymac.publications.paper_rapc_multihop_lora.run --check-only
```

Generate all tables, figures, and the provenance manifest:

```bash
python -m loraphymac.publications.paper_rapc_multihop_lora.run
```

The installed `loraphymac-publish-rapc-multihop-lora` command provides the same CLI. The
workflow requires status `PASS` for both sources, a complete 300-run seed grid,
zero failed terrain runs, a valid closed-loop contract report, and matching
artifact hashes. It writes `publication_manifest.json` only after all expected
outputs exist and their SHA-256 checksums and source lineage are recorded.

The thin `notebooks/paper_rapc_multihop_lora_workflow.ipynb` uses the same
validated workflow and displays the generated PNG figures.
