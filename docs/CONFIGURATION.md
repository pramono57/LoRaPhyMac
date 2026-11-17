# Configuration validation

LoRaPhyMAC validates every resolved configuration before topology generation,
output creation, or SimPy execution. The central implementation is
`src/loraphymac/config/validation.py`; both `run_single_simulation()` and the multi-seed
pipeline use this same validator.

## Validation layers

1. **Schema:** unknown keys, strict booleans, integer counts, finite numeric
   values, reproducible NumPy seed range, and non-empty paths/identifiers.
2. **Topology and propagation:** positive dimensions, space outside the node
   placement exclusion zone, ITM distance and antenna-height limits.
3. **LoRa PHY:** supported SF, bandwidth, channel IDs, collision modes, and
   complete directional SIR-matrix coverage for every configured SF pair.
4. **Power and MAC:** hardware/EIRP-compliant PTx values, APC/frame/RX-window
   dependencies, FOpts size, timing, queue capacity, and duty-cycle bounds.
5. **Experiment profile:** constraints owned by a content-based experiment,
   such as the full APC terrain-sweep contract of `apc_terrain_sweep_v1`.
   The historical `paper1_apc_v1` name remains supported for auditing the
   completed dataset. The legacy
   `paper1_sf7_bw125` profile checks only a PHY subset and is not a publication
   contract.

## Usage

Partial user configuration is first merged with `DEFAULT_CONFIG`, then the
resolved mapping is validated:

```python
from loraphymac import resolve_configuration

config = resolve_configuration({"node_num": 200, "seed": 42})
```

Normal callers should use `run_single_simulation(config)`, which performs both
steps automatically. The multi-seed pipeline validates every factor/seed cell
while building the experiment grid, so an invalid paper configuration fails
before a long batch begins.

When a future research question requires a new scientific contract, place its
profile, cross-field rules, factor grid, and runner in a content-based folder
under `src/loraphymac/experiments/`. Register only its contract adapter in
`src/loraphymac/experiments/common/contract_registry.py`. Put manuscript-only
selection, tables, figures, and notebooks under
`src/loraphymac/publications/<publication>/`; keep the simulator core and
shared pipeline free of publication-specific branches.


## Single-run profiles

Preliminary exploration and publication evidence intentionally use different
profiles:

- `general_multisf_multibw` in `single_run_exploration` applies all central
  schema, PHY, power, topology, and cross-field validation while permitting
  scientific parameters to change.
- `apc_closed_loop_verification_v1` freezes one Paper 1 point: 100 nodes,
  3600 s, Δh = 90 m, seed 42, adaptive/adaptive PTx, and relay forwarding
  channel [1]. Scientific drift fails before topology construction.

Changing print, export, run-id, or output-directory fields does not change the
closed-loop scientific scenario. Changing a frozen scientific field requires a
new experiment profile and contract version.
