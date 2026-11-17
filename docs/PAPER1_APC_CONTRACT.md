# Paper 1 APC scientific contract

## Identity and status

- Current profile: `apc_terrain_sweep_v1`
- Current contract version: `apc_terrain_sweep_contract_v1.0`
- Historical aliases: `paper1_apc_v1` and `paper1_apc_contract_v1.0`
- Subject: relay-aware adaptive transmit-power control in multi-hop LoRa.
- Publication results generated before the finite-horizon energy and
  receiver-domain corrections are not equivalent and must be rerun.

The executable source of truth is
`src/loraphymac/experiments/apc_terrain_sweep/contract.py`. The former
`src/loraphymac/experiments/paper1_apc/` compatibility package has been
removed. The narrow `src/loraphymac/contracts/paper1_apc.py` import facade
remains temporarily for historical external callers. The older
`paper1_sf7_bw125` profile checks only its PHY subset and is not sufficient for
a Paper 1 publication run.

## Experimental question

The experiment compares energy, delivery, delay, and final transmit power when
power control is enabled independently at end nodes and relays. Terrain
irregularity is the environmental factor; all other scientific parameters are
held constant.

| Mode | Node PTx | Relay PTx |
|---|---|---|
| FF | fixed | fixed |
| AF | adaptive | fixed |
| FA | fixed | adaptive |
| AA | adaptive | adaptive |

The full balanced design contains:

- 4 APC modes;
- 15 ordered terrain-irregularity values, Δh = 20, 30, …, 160 m;
- 5 paired seeds: 30, 36, 42, 48, and 56;
- 300 runs total, with a 95% Student-t confidence interval per cell.

The exact machine-readable grid is
`src/loraphymac/experiments/apc_terrain_sweep/configs/production_v1.json`. Its
fixed parameters are explicit so a later change to simulator defaults cannot
silently change the experiment.

## Frozen model boundary

The contract fixes 100 nodes and 2 relays in a 45 km × 15 km area for 3600 s.
Nodes send a 50-byte application payload every 60 ± 2 s. Node and relay PHY use
SF7/BW125. The LoRaPhyMAC frame adds 12 bytes, so DATA_UP is 62 bytes; APC_REQ
uses 4-byte FOpts, so DATA_DOWN/DATA_DOWN_FWD is 16 bytes.

The EIRP ceilings are 20 dBm for nodes and 26 dBm for relays/gateway. With the
frozen gains and zero feeder loss, initial conducted powers resolve to 17 dBm
for nodes and 20 dBm for relays/gateway. Fixed modes remain at those values.

Energy covers radio energy only (`include_mcu_energy=false`) and is clipped at
the hard simulation horizon. Two energy scopes must be reported:

- network energy = node + relay + gateway;
- constrained-device energy = node + relay, excluding the mains-powered
  gateway.

## APC mechanism

APC decisions use the decoded uplink's effective margin:

`decision margin = RSSI − receiver sensitivity − feasible margin`

The target margin is 6 dB. The controller computes integer 2 dB steps in the
TxPowerIndex domain. Index 0 is the highest EIRP-compliant power; a larger index
means lower conducted power. Commands are carried physically in APC_REQ FOpts:

- `DATA_DOWN` controls an end node and follows its original uplink channel;
- `DATA_DOWN_FWD` controls a relay and uses the relay forwarding channel;
- a target changes PTx only after the physical control frame is decoded;
- node RX1 opens 1 s after uplink for 1 s;
- APC downlink delay is 1 s and retry timeout is 5 s.

FF is the no-APC baseline. AF, FA, and AA isolate node-side, relay-side, and
joint adaptation respectively. All modes keep RX-window and frame settings
identical to prevent protocol configuration from becoming a confounder.

## PHY and interference contract

- ITM area mode with Δh as the only propagation factor varied by the grid.
- Receiver-scoped RF observation for relays and gateway.
- Pairwise spectral capture with the
  `rakibul_islam_2023_table4` directional SIR matrix.
- SF7 co-SF capture threshold: 6 dB.
- End-device downlink collision handling remains a stated model limitation;
  only intended DATA_DOWN is presented to NodeRX.

## Outcomes and evidence

Primary outcomes are delivery by horizon, terminal loss, censoring, end-to-end
delay, node/relay/gateway energy, both energy-per-delivered-packet scopes, final
node/relay PTx, decision/APC-RX margins, and physical APC success.

Every accepted Paper 1 run must satisfy:

- delivered + terminal loss + censored = 100%;
- packet-fate and collision conservation deltas = 0;
- component energy sums to network energy;
- node + relay energy sums to constrained-device energy;
- `apc_total_tx = apc_total_rx + apc_total_drop`;
- `apc_total_rx = ptx_update_count = number of PTx-history rows`;
- `stage1_status = PASS`;
- FF produces no APC frames or PTx updates;
- fixed node/relay entities remain at 17/20 dBm respectively.

Per-run core, packet-fate, and receiver evidence is retained by the shared
pipeline. Paper 1 additionally retains energy breakdown, final PTx
distribution, frame summary, and run metadata for every run. Detailed PTx,
margin, and APC-control histories are retained at Δh = 90 m, the frozen
comparison terrain, to support convergence figures without duplicating large
traces over the full terrain sweep.

## Execution

Validate the complete design without running it:

```bash
python -m loraphymac.experiments.apc_terrain_sweep.run --dry-run
```

Run the full frozen experiment only after the test suite passes. Use a new
`experiment_id` whenever code, contract, or configuration changes; the pipeline
also rejects source-fingerprint changes inside an existing experiment folder.


## Closed-loop behavior verification run

Paper 1 section 1.1 uses a separate one-run experiment rather than the flexible
exploration workflow. Its profile is
`apc_closed_loop_verification_v1`, its contract version is
`apc_closed_loop_contract_v1.0`, and its executable source is
`src/loraphymac/experiments/apc_closed_loop_verification/contract.py`.

The run is the adaptive/adaptive, Δh = 90 m, seed-42 point of the frozen model
boundary. It requires balanced physical APC accounting, non-empty node and
relay PTx update histories, node and relay margin observations, stage-1 PASS,
and zero packet/collision conservation deltas. A successful run writes
`verification_report.json` as a machine-readable completion marker.

Run it with:

```bash
python -m loraphymac.experiments.apc_closed_loop_verification.run
```
