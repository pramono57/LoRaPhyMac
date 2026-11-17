# Diagnostic APC Terrain Profile v1

This isolated diagnostic contract uses the shared multi-seed pipeline's explicit
`run_matrix`. It does not alter the official `apc_terrain_sweep_v1` contract.

- Profile / experiment ID: `apc_terrain_profile_diagnostic_v1`
- Terrain irregularity: 90 m
- Modes: FF (`fixed/fixed`) and AA (`adaptive/adaptive`)
- Seeds: 30, 36, 42, 48, 56
- Planned runs: 10
- Scheduling: `independent_fifo_v1`, `sha256_seed_node_v1`,
  `unbounded_fifo_v1`, `right_censor_waiting_v1`
- Node RX collision: enabled

The source configuration is
`artifacts/diagnostic_runs_20261111/configs/apc_terrain_profile_diagnostic_v1.json`.
Its output root is `artifacts/diagnostic_runs_20261111/APC_TERRAIN_PROFILE_DIAGNOSTIC/`;
the shared artifact layout places the run cohort in the child directory named
`apc_terrain_profile_diagnostic_v1`.

The production runner must be invoked with `--dry-run` first. A real run is
intended to be started manually from Terminal only after reviewing the dry-run
matrix and provenance.

## Source fingerprints

- Before this change, the latest-runner reference covered 133 source files and
  had digest `bcfb3f26ee1756f0bbf7cb00fdc8b10ef0563adb7bf0507c30d5268eb937fee1`.
- After this change, the working-tree source map covers 134 Python files and
  has digest `4bb5be5440688a2c149739ea012ca1543362d832178c5f02ee8e3ec518f2a466`.
- The separately scoped simulation-core map (config, domain, MAC, metrics, PHY,
  routing, runtime, and simulation packages; 58 Python files) is
  `a93068aae183371796a6d4e1f765b135dadca7c11ab9a21df2a34564d4597eb1` both
  before and after. The pre-change map was reconstructed from the saved 133-file
  latest-runner manifest and compared path-by-path with the current files. No
  files in that scope were edited for this diagnostic support.

The source-map digests use the shared pipeline's path-to-file-SHA256 map and
SHA256 of its sorted-key JSON encoding. The newer digest is orchestration-aware
and must not be represented as the latest-runner digest above.
