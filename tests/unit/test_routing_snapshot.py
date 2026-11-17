from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from loraphymac.metrics.instrumentation_metrics import (
    build_topology_parent_by_run_df,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.routing_snapshots import (
    EXPERIMENT_DIR,
    W4_ROUTING_SNAPSHOT_SEEDS,
    load_routing_snapshot_bindings,
)
from loraphymac.routing.snapshot import (
    ROUTING_MANIFEST_FILENAME,
    ROUTING_REFERENCE_DELTA_H,
    ROUTING_SNAPSHOT_FILENAME,
    ROUTING_STRATEGY_FIXED_REFERENCE,
    ROUTING_STRATEGY_TERRAIN_SPECIFIC,
    RoutingSnapshotError,
    file_sha256,
    full_routes,
    load_routing_snapshot_bundle,
    routing_sha256,
    topology_records,
    write_routing_snapshot_bundle,
)
from loraphymac.runtime.islam_dbpc_802 import (
    ISLAM_DBPC_802_POLICY_ID,
    IslamDBPC802Controller,
)
from loraphymac.runtime.node_tx import stable_application_rng_seed
from loraphymac.runtime.wang_power_commands import WangGatewayCommandCoordinator
from loraphymac.runtime.wang_sink_decision import WangActiveTransmitter
from loraphymac.runtime.wang_snr_reporting import (
    reconstruct_physical_uplink_identity,
)
from loraphymac.simulation.builders import (
    build_infrastructure,
    build_routing,
    build_topology,
)
from loraphymac.simulator import resolve_configuration


pytestmark = pytest.mark.unit
SEEDS = (30, 36, 42, 48, 56)
MODELS = (
    "ff",
    "aa",
    "islam_dbpc_802",
    "islam_dbpc_itm6",
    "wang_atp_sf7_a2only",
)
W4_BINDINGS = (
    EXPERIMENT_DIR
    / "configs"
    / "routing_snapshot_bindings_w4_dh90_v1.json"
)
W4_BINDINGS_SHA256 = (
    "c844095d353714bafc2f89112ace5536b88d597f554f6ab4fbea7ca87beb6164"
)


def _config(seed: int, delta_h: int, **overrides):
    value = {
        "experiment_profile": "general_multisf_multibw",
        "seed": seed,
        "terrain_delta_h": delta_h,
        "node_num": 100,
        "relay_num": 2,
        "node_ptx_mode": "fixed",
        "relay_ptx_mode": "fixed",
        "node_sf_list": [7],
        "relay_sf": 7,
        "node_bw": 125000,
        "relay_bw_hz": 125000,
        "node_payload_size": 50,
        "energy_accounting_mode": "corrected",
        "node_rx_collision_enabled": True,
        "frame_default_ack_req": False,
        "application_scheduling_mode": "independent_fifo_v1",
        "application_rng_policy": "sha256_seed_node_v1",
        "application_queue_policy": "unbounded_fifo_v1",
        "application_horizon_policy": "right_censor_waiting_v1",
        "tx_power_model_id": "ff",
        "routing_strategy": ROUTING_STRATEGY_TERRAIN_SPECIFIC,
        "output_dir": None,
    }
    value.update(overrides)
    return resolve_configuration(value)


def _build(config):
    infrastructure = build_infrastructure(config)
    topology = build_topology(config, infrastructure)
    routing = build_routing(config, infrastructure, topology)
    return infrastructure, topology, routing


def _create_bundle(root: Path, seed: int = 30):
    config = _config(seed, ROUTING_REFERENCE_DELTA_H)
    infrastructure, topology, routing = _build(config)
    evidence = build_topology_parent_by_run_df(
        routing,
        {
            "run_id": f"snapshot-s{seed}",
            "seed": seed,
            "node_ptx_mode": "fixed",
            "relay_ptx_mode": "fixed",
            "terrain_delta_h": ROUTING_REFERENCE_DELTA_H,
            "routing_strategy": ROUTING_STRATEGY_TERRAIN_SPECIFIC,
            "route_construction_delta_h": ROUTING_REFERENCE_DELTA_H,
            "physical_terrain_delta_h": ROUTING_REFERENCE_DELTA_H,
            "topology_sha256": routing.topology_sha256,
            "selected_parent_routing_sha256": (
                routing.selected_parent_routing_sha256
            ),
        },
    )
    manifest = write_routing_snapshot_bundle(
        output_dir=root,
        config=config,
        topology=topology,
        routing=routing,
        reference_link_evidence=evidence,
        frequency_mhz=infrastructure.channel_plan.get_freq(
            config["routing_ch_id"]
        ),
        source_provenance={"test": True},
        source_config=config,
    )
    return manifest, file_sha256(manifest), routing.routing_map


def _fixed_config(seed: int, delta_h: int, manifest: Path, manifest_hash: str, **extra):
    return _config(
        seed,
        delta_h,
        routing_strategy=ROUTING_STRATEGY_FIXED_REFERENCE,
        routing_reference_delta_h=ROUTING_REFERENCE_DELTA_H,
        routing_snapshot_manifest=str(manifest),
        routing_snapshot_manifest_sha256=manifest_hash,
        **extra,
    )


def _rewrite_bundle(bundle: Path, mutate) -> str:
    snapshot_path = bundle / ROUTING_SNAPSHOT_FILENAME
    manifest_path = bundle / ROUTING_MANIFEST_FILENAME
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    mutate(snapshot)
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["routing_snapshot"]["sha256"] = file_sha256(snapshot_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return file_sha256(manifest_path)


def test_same_seed_geometry_is_identical_across_delta_h():
    for seed in SEEDS:
        geometries = []
        for delta_h in (20, 90, 160):
            config = _config(seed, delta_h)
            infrastructure = build_infrastructure(config)
            topology = build_topology(config, infrastructure)
            geometries.append(topology_records(topology))
        assert geometries[0] == geometries[1] == geometries[2]


def test_fixed_reference_maps_and_delta90_routes_match_for_all_seeds(tmp_path):
    for seed in SEEDS:
        bundle = tmp_path / f"seed_{seed}"
        manifest, manifest_hash, reference_map = _create_bundle(bundle, seed)
        fixed_maps = []
        for delta_h in (20, 90, 160):
            _, _, routing = _build(
                _fixed_config(seed, delta_h, manifest, manifest_hash)
            )
            fixed_maps.append(routing.routing_map)
            assert routing.route_construction_delta_h == 90
            assert routing.physical_terrain_delta_h == delta_h
        assert fixed_maps[0] == fixed_maps[1] == fixed_maps[2] == reference_map
        _, _, terrain90 = _build(_config(seed, 90))
        assert terrain90.routing_map == reference_map
        assert full_routes(terrain90.routing_map) == full_routes(reference_map)


def test_one_snapshot_is_model_independent_and_never_regenerated(tmp_path):
    manifest, manifest_hash, reference_map = _create_bundle(tmp_path / "bundle")
    for model_id in MODELS:
        _, _, routing = _build(
            _fixed_config(
                30,
                160,
                manifest,
                manifest_hash,
                tx_power_model_id=model_id,
            )
        )
        assert routing.routing_map == reference_map
        assert routing.routing_snapshot_sha256 is not None

    with pytest.raises(RoutingSnapshotError, match="manifest is missing"):
        _build(
            _fixed_config(
                30,
                160,
                tmp_path / "missing.json",
                "0" * 64,
            )
        )


def test_terrain_specific_route_hash_can_change_with_actual_terrain():
    differences = []
    for seed in SEEDS:
        _, _, low = _build(_config(seed, 20))
        _, _, high = _build(_config(seed, 160))
        differences.append(
            low.selected_parent_routing_sha256
            != high.selected_parent_routing_sha256
        )
    assert any(differences)


def test_fixed_route_keeps_parent_but_actual_itm_evidence_changes(tmp_path):
    manifest, manifest_hash, _ = _create_bundle(tmp_path / "bundle")
    _, _, low = _build(_fixed_config(30, 20, manifest, manifest_hash))
    _, _, high = _build(_fixed_config(30, 160, manifest, manifest_hash))
    assert low.routing_map == high.routing_map
    low_pl = pd.to_numeric(low.selected_next_hop_links_df["Pathloss(dB)"])
    high_pl = pd.to_numeric(high.selected_next_hop_links_df["Pathloss(dB)"])
    assert not low_pl.equals(high_pl)


def test_corrupt_snapshot_is_rejected(tmp_path):
    manifest, manifest_hash, _ = _create_bundle(tmp_path / "bundle")
    snapshot = manifest.parent / ROUTING_SNAPSHOT_FILENAME
    snapshot.write_text(snapshot.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RoutingSnapshotError, match="snapshot file hash mismatch"):
        _build(_fixed_config(30, 90, manifest, manifest_hash))


@pytest.mark.parametrize("invalid_parent", ["UNKNOWN", "N1"])
def test_unknown_parent_and_cycle_are_rejected(tmp_path, invalid_parent):
    original = tmp_path / "original"
    manifest, _, _ = _create_bundle(original)
    bundle = tmp_path / f"mutated_{invalid_parent}"
    shutil.copytree(original, bundle)

    def mutate(snapshot):
        row = next(row for row in snapshot["routing_map"] if row["source_id"] == "N1")
        row["parent_id"] = invalid_parent

    manifest_hash = _rewrite_bundle(bundle, mutate)
    expected = "unknown parent" if invalid_parent == "UNKNOWN" else "cycle"
    with pytest.raises(RoutingSnapshotError, match=expected):
        _build(
            _fixed_config(
                30,
                90,
                bundle / ROUTING_MANIFEST_FILENAME,
                manifest_hash,
            )
        )
    assert manifest.is_file()


def test_topology_coordinate_mismatch_is_rejected(tmp_path):
    manifest, manifest_hash, _ = _create_bundle(tmp_path / "bundle")
    config = _fixed_config(30, 90, manifest, manifest_hash)
    infrastructure = build_infrastructure(config)
    topology = build_topology(config, infrastructure)
    topology.nodes[0].loc.x += 1.0
    with pytest.raises(RoutingSnapshotError, match="coordinates"):
        load_routing_snapshot_bundle(
            manifest_path=manifest,
            expected_manifest_sha256=manifest_hash,
            config=config,
            topology=topology,
            frequency_mhz=infrastructure.channel_plan.get_freq(
                config["routing_ch_id"]
            ),
        )


def test_dbpc802_ptx_is_terrain_independent_for_fixed_geometry_and_parent(tmp_path):
    manifest, manifest_hash, _ = _create_bundle(tmp_path / "bundle")
    outcomes = []
    for delta_h in (20, 160):
        routing_config = _fixed_config(
            30, delta_h, manifest, manifest_hash
        )
        _, topology, routing = _build(routing_config)
        policy_config = dict(routing_config)
        policy_config.update(
            tx_power_model_id="islam_dbpc_802",
            tx_power_policy_id=ISLAM_DBPC_802_POLICY_ID,
            tx_power_calibration_margin_db=0,
        )
        selections = IslamDBPC802Controller("islam_dbpc_802").apply(
            config=policy_config,
            nodes=topology.nodes,
            relays=topology.relays,
            routing_map=routing.routing_map,
            node_map=routing.node_map,
        )
        outcomes.append(
            {
                selection.entity_id: selection.selected_ptx_dbm
                for selection in selections
            }
        )
    assert outcomes[0] == outcomes[1]


def test_wang_identity_and_reverse_downlink_use_the_loaded_route(tmp_path):
    manifest, manifest_hash, _ = _create_bundle(tmp_path / "bundle")
    _, topology, routing = _build(
        _fixed_config(30, 160, manifest, manifest_hash)
    )
    node = topology.nodes[0]
    route = full_routes(routing.routing_map)[node.id]
    packet = SimpleNamespace(
        src_id=node.id,
        fcnt=7,
        hop_index=0,
        hop_limit=len(route) - 1,
        next_hop=route[1],
        dst_id=route[-1],
    )
    identity = reconstruct_physical_uplink_identity(
        packet,
        routing_map=routing.routing_map,
        receiver_id=route[1],
    )
    assert identity.route == route
    active = [
        WangActiveTransmitter(
            entity_id=node.id,
            entity_type="node",
            expected_ptx_dbm=node.ptx_max,
        )
    ]
    coordinator = WangGatewayCommandCoordinator(
        env=SimpleNamespace(now=0.0),
        gateway=topology.gateway,
        active_transmitters=active,
        routing_map=routing.routing_map,
        relay_tx={},
        air_interface=SimpleNamespace(),
        statistics=SimpleNamespace(),
    )
    assert coordinator._reverse_route(node.id) == list(reversed(route))


def test_application_rng_is_independent_of_routing_strategy():
    for seed in SEEDS:
        for node_id in ("N1", "N50", "N100"):
            rng_seed = stable_application_rng_seed(seed, node_id)
            fixed_rng = random.Random(rng_seed)
            terrain_rng = random.Random(rng_seed)
            fixed = [fixed_rng.random() for _ in range(3)]
            terrain = [terrain_rng.random() for _ in range(3)]
            assert fixed == terrain


def test_routing_hash_uses_parent_content_not_dictionary_order():
    route = {"N1": "R1", "R1": "GW", "GW": None}
    reordered = {"GW": None, "R1": "GW", "N1": "R1"}
    assert routing_sha256(route) == routing_sha256(reordered)


@pytest.mark.extended_research
def test_w4_seed_registry_binds_exact_verified_manifests():
    bindings = load_routing_snapshot_bindings(
        W4_BINDINGS,
        expected_sha256=W4_BINDINGS_SHA256,
    )
    assert tuple(bindings) == W4_ROUTING_SNAPSHOT_SEEDS
    assert all(binding.manifest_path.is_file() for binding in bindings.values())
    assert all(binding.reference_delta_h == 90 for binding in bindings.values())
    assert all(
        binding.creation_provenance["worktree_dirty"] is True
        for binding in bindings.values()
    )


def test_w4_seed_registry_rejects_unknown_seed_without_fallback():
    with pytest.raises(RoutingSnapshotError, match="configured seeds exactly"):
        load_routing_snapshot_bindings(
            W4_BINDINGS,
            expected_sha256=W4_BINDINGS_SHA256,
            expected_seeds=(*W4_ROUTING_SNAPSHOT_SEEDS, 99),
        )


@pytest.mark.extended_research
def test_w4_seed_registry_rejects_missing_and_bad_manifest(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(RoutingSnapshotError, match="registry is missing"):
        load_routing_snapshot_bindings(
            missing,
            expected_sha256="0" * 64,
        )
    assert not missing.exists()

    raw = json.loads(W4_BINDINGS.read_text(encoding="utf-8"))
    verified = load_routing_snapshot_bindings(
        W4_BINDINGS,
        expected_sha256=W4_BINDINGS_SHA256,
    )
    for seed, binding in verified.items():
        raw["bindings"][str(seed)]["manifest_path"] = str(
            binding.manifest_path
        )
    raw["bindings"]["30"]["manifest_sha256"] = "0" * 64
    corrupted = tmp_path / "bindings.json"
    corrupted.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RoutingSnapshotError, match="manifest hash mismatch"):
        load_routing_snapshot_bindings(
            corrupted,
            expected_sha256=file_sha256(corrupted),
        )
