# APC Compare-Model Terrain Sweep

The default V2 experiment is the unified, matched 300-run comparison for:

- `ff`: fixed node/relay transmit power;
- `aa`: the proposed relay-aware APC with adaptive node and relay power;
- `islam_dbpc_itm6`: Islam Algorithm 2 fixed-SF7 power selection using the
  immutable ITM calibration at Delta-h 90 m with a 6 dB margin; and
- `wang_atp_sf7_a2only`: the Wang-inspired, sink-controlled Algorithm 2-only
  fixed-SF7 adaptation (no Algorithm 1, data ACK, ARQ, or power increase).

The grid contains 15 terrain levels (`20..160 m`, step `10 m`) and five seeds
(`30, 36, 42, 48, 56`) for each model. V2 uses 60 nodes, two relays, one
gateway, a 50-byte application payload, 3600 seconds, fixed SF7/BW125/CR4/5,
corrected energy accounting, and receiver-wide NodeRX collision handling for
all four models. Topology, routing inputs, PHY/MAC settings, offered traffic,
and duration are matched within each terrain/seed cell. Routing is recomputed
for each terrain but must match across all models in the same cell.

The parent runner delegates to four standard 75-run child experiments under
`models/`, retaining source fingerprints, immutable manifests, resume, run
validation, and aggregation protections. The historical V1 225-run config is
retained as `configs/production_v1.json`; V2 uses a new experiment ID and never
overwrites V1 artifacts.

## Power-range fairness

All four models use the same SX1276/SX1302 hardware, current tables, antenna
gains, feeder losses, and EIRP limits. These settings give every model the same
effective operational bounds: 5--17 dBm for nodes and 12--20 dBm for relays.
FF, AA, and Wang start at the operational maximum; DBPC uses its immutable
pre-traffic calibration. No hardware bound is changed to favor Wang. AA/DBPC
use their audited 2 dB action sets, while Wang reduces power in 3 dB steps with
hardware-bound clamping. This resolution difference is an explicit algorithmic
characteristic and a potential comparison confounder.

Wang retains the W2 contract: one main-gateway network-wide decision,
300-second non-overlapping windows, 10-second delivery grace, 100% active-link
coverage, physical reports and unicast commands, and causal confirmation.

## Commands

Inspect the exact grid without writing experiment artifacts:

```bash
python -m loraphymac.experiments.apc_compare_model_terrain_sweep.run --dry-run
```

Run the complete grid (resume is enabled by default):

```bash
python -m loraphymac.experiments.apc_compare_model_terrain_sweep.run --fail-fast
```

Rebuild validated tables and Figure 8--10 after all children pass:

```bash
python -m loraphymac.experiments.apc_compare_model_terrain_sweep.postprocessing
```

## Outputs

The default parent directory is
`artifacts/experiments/apc_compare_model_terrain_sweep/apc_compare_model_terrain_sweep_v2_wang_r01/`.
It contains child run evidence, per-run metrics, Student-t 95% confidence
intervals, seed-paired differences, validation/provenance records, and Figures
8--10 as PNG and SVG.

Wang runs additionally export R20 SNR-report lifecycle, R21 sink decisions, and
R22 power-command lifecycle. Derived tables include window coverage, decisions
and no-decision reasons, command states, and exact physical command TX
airtime/energy. The marginal energy of piggybacked SNR-report bytes is not
separately observable from R19 and is reported as unavailable rather than zero.
Wang-only fields are explicitly not applicable for FF/AA/DBPC.

Energy per delivered packet is computed per run before aggregation. It is
undefined (`NaN`) for a zero-delivery run rather than being reported as zero.
Average end-to-end delay is conditional on delivered packets, so Figure 10 must
be interpreted together with the PDR in Figure 8.

Postprocessing fails closed unless all 300 unique cells pass model,
calibration, feedback, corrected-energy, packet/collision conservation, and
cross-model input-equivalence checks. Plotting consumes only validated
simulation CSVs.

