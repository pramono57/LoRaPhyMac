# Diagnostic APC Terrain Full Sweep v1

This is a separate 300-run diagnostic contract. The 10-run
`apc_terrain_profile_diagnostic_v1` and official `apc_terrain_sweep_v1`
contracts remain unchanged.

- Experiment ID: `apc_terrain_profile_full_sweep_v1_r01`
- Profile: `apc_terrain_profile_full_sweep_v1`
- Grid: 4 APC modes × 15 official terrain values × 5 official seeds = 300 runs
- Diagnostic settings: node RX collision enabled and independent FIFO
  scheduling/RNG/queue/horizon policies.
- Configuration:
  `artifacts/diagnostic_runs_20261111/configs/apc_terrain_profile_full_sweep_v1.json`
- Output namespace:
  `artifacts/latest_runner_rerun_20261110/APC_TERRAIN_SWEEP/apc_terrain_profile_full_sweep_v1_r01/`
- Contract version: `apc_terrain_profile_full_sweep_contract_v1.0`
- Runner: `multiseed_pipeline_v5`; simulator `1.0.0`; API `simulator_api_v1`
- Configuration SHA-256:
  `22514c18f5951be95f81527b62e2b73aba2d881b7f7deac4e6e8a8db656bf780`
- Explicit run-matrix SHA-256:
  `ee479dab29612ff593e7ae54fb3caec8884d2aa52584d651b5c63d9b61fbbb93`
- Current 135-file source-map fingerprint:
  `4ac886c702af911fd13ec59d87e6f31132279b0fbd87a2c443704d205b9beed1`
- Simulation-core fingerprint (58 files):
  `a93068aae183371796a6d4e1f765b135dadca7c11ab9a21df2a34564d4597eb1`,
  identical to the saved pre-change core map.

The configuration was derived from the previous 10-run diagnostic config and
the official production Terrain Sweep matrix. Only the experiment/profile IDs,
output path, terrain/mode factors and explicit matrix were expanded; the five
diagnostic settings are unchanged. The saved source map has 133 files and
fingerprint `bcfb3f26ee1756f0bbf7cb00fdc8b10ef0563adb7bf0507c30d5268eb937fee1`;
the new source fingerprint is intentionally distinct because the full-sweep
contract was added.

The output namespace must be absent before a real execution. Use the official
terrain CLI with `--dry-run` first; the production run is to be started manually
from macOS Terminal, not by Codex.
