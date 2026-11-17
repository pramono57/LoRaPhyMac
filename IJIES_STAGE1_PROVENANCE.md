# IJIES Stage 1 Validated Baseline Provenance

Freeze date: 22 October 2026

Stage 1 source commit: `6ce0e06d27ce63d477cc83f1310f23a3bfe7bf9d`

Parent software baseline: `530df4d6c2b4253a0adc4fc95d74ceaaa0848275`

This record freezes the validated IJIES Stage 1 instrumentation baseline. It
does not include the later publication-workflow rename, executed notebook
outputs, academic DOCX files, historical simulation outputs, or any Stage 2B
implementation.

## Configuration and evidence hashes

| Artifact | SHA-256 |
|---|---|
| `src/loraphymac/experiments/apc_terrain_sweep/configs/production_v1.json` | `22ce997747080c7ed1b3b572aae508da28110e3c27a36f2227d1ad92ae9ac547` |
| `src/loraphymac/experiments/apc_closed_loop_verification/configs/verification_v1.json` | `415a2e1dfe3e0a0f906127e2b41266a90a5320f70c9d565446c0109a768080bb` |
| `pyproject.toml` in Stage 1 source commit | `58b1562d5584f87a7cb84c93dfba2ea4cb81b1e1eb8d853bd1c879881daf1863` |
| `artifacts/verification/ijies_stage1/regression_before.json` | `f2c4b06b2697d83c65111e4f1e8e4c227c2df1c1ce28fe058950a4cd8a422f00` |
| `artifacts/verification/ijies_stage1/regression_after.json` | `5f143b6291145a63b66b0a58498f269d2ee9563555809ae3e38242dd5595bf7e` |
| `artifacts/verification/ijies_stage1/routing/routing_verification_summary.json` | `4b8b4e8c3fde75465d26469be1227f217397e9007f716e6f31140562a198ca07` |

The `artifacts/` files remain local, generated evidence and are intentionally
not committed. Their hashes above allow the exact evidence used for this freeze
to be verified without moving or rewriting the historical dataset.

## Regression gates

Commands were executed with `MPLBACKEND=Agg` using the project `.venv`.

- Targeted Stage 1 suite: **31 passed**.
- Full repository suite: **837 passed**.
- Stage 1 before/after report: `all_scientific_metrics_exact=true` for FF and
  AA.
- Six historical table hashes: exact for FF and AA.
- Historical 300-run E0 artifacts were not modified.

## Seventeen golden scientific metrics

Values below are the Stage 1 legacy reference recorded in
`regression_after.json`.

| Metric | AA | FF |
|---|---:|---:|
| `apc_apply_success_percent` | 100.0 | 0.0 |
| `apc_total_drop` | 0 | 0 |
| `apc_total_rx` | 92 | 0 |
| `apc_total_tx` | 92 | 0 |
| `avg_e2e_delay_ms` | 276.668 | 278.323 |
| `avg_final_node_ptx` | 10.52 | 17.0 |
| `avg_final_relay_ptx` | 12.0 | 20.0 |
| `collision_decision_count` | 15518 | 17468 |
| `collision_drop_decision_count` | 553 | 572 |
| `collision_loss_rate_percent` | 92.941176 | 77.823129 |
| `energy_per_delivered_packet_J` | 0.581898 | 0.652289 |
| `pdr_percent` | 78.947 | 79.064 |
| `ptx_update_count` | 92 | 0 |
| `total_gateway_energy_J` | 702.810501 | 702.0 |
| `total_network_energy_J` | 2758.197488 | 3096.417675 |
| `total_node_energy_J` | 311.504512 | 432.461482 |
| `total_relay_energy_J` | 1743.882475 | 1961.956194 |

## Six historical table hashes

| Table | AA SHA-256 | FF SHA-256 |
|---|---|---|
| `collision_summary_df` | `6c2833c9448bb5d18b9710e201c97c2eda0752d114745388062e64084b0856f7` | `f11b3ddcf633bdeea02f14f378e9a5890e521e139be0d6aaae0b4f585922732c` |
| `energy_breakdown_df` | `234e2eb8cfd5676a5443d710d0d3932395a4cc843f1adc0262c77cefa41e8d71` | `52ca0aa4be473adb92afb8221afddf28dcf88115e070fc9bd7f889b97045165e` |
| `frame_summary_df` | `29e7ce268352e26b2286098c85725f96139286242b608df01d0e78ce0314fc8e` | `0c2eb6827ad8c1d4f90793d1ae40bbc16b8b2d4c0960dfce80ba20a1a9a02278` |
| `ptx_history_df` | `10e638945cf10a4ceb8629df18bb6aad6a553f5c919f078861e9137c39a9ca87` | `01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b` |
| `routing_table_df` | `634b5215aae8b456737ccd9fcf6a53ab89034393d2af7fd5f490d78e9047db31` | `634b5215aae8b456737ccd9fcf6a53ab89034393d2af7fd5f490d78e9047db31` |
| `selected_next_hop_links_df` | `94603d7407b1c887a1330a6b3c2b8af9e8093f06c1ead20b05fc3246e1c469c7` | `94603d7407b1c887a1330a6b3c2b8af9e8093f06c1ead20b05fc3246e1c469c7` |

## Execution environment

- Python `3.12.3`
- LoRaPhyMAC editable package `0.19.1`
- NumPy `2.5.3`
- pandas `2.3.3`
- SciPy `1.18.1`
- SimPy `4.1.2`
- Matplotlib `3.11.1`
- openpyxl `3.1.5`
- pytest `8.4.2`
- ITM implementation: vendored `loraphymac.pyitm`, not an external PyPI
  dependency.

## Known accounting finding carried into Stage 2

The legacy continuous-RX equation subtracts TX airtime but not transition time.
Stage 1 deliberately preserves that legacy behavior. The Stage 1 report records
continuous-RX/transition overlap of `0.421267000000026` s for AA and
`0.41603100000002624` s for FF. Correcting this is explicitly deferred to the
separately approved Stage 2B energy-accounting work.
