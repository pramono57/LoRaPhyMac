from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from loraphymac.experiments.apc_compare_model_terrain_sweep.routing_snapshots import (
    FIVE_MODEL_PILOT_CONFIG,
    W4_ROUTING_SNAPSHOT_SEEDS,
    _reference_config,
    load_routing_snapshot_bindings,
)
from loraphymac.experiments.apc_compare_model_terrain_sweep.run import (
    run_experiments,
)
from loraphymac.routing.snapshot import (
    ROUTING_STRATEGY_FIXED_REFERENCE,
    ROUTING_STRATEGY_TERRAIN_SPECIFIC,
    RoutingSnapshotError,
)
from loraphymac.simulation.builders import (
    build_infrastructure,
    build_routing,
    build_topology,
)
from loraphymac.simulator import resolve_configuration


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = (
    REPOSITORY_ROOT
    / "src/loraphymac/experiments/apc_compare_model_terrain_sweep/configs"
)
SNAPSHOT_ROOT = (
    REPOSITORY_ROOT
    / "artifacts/experiments/apc_compare_model_terrain_sweep/routing_snapshots"
)
V2_ROOT = SNAPSHOT_ROOT / "w4_fixed_reference_dh90_v2"
V2_REGISTRY = CONFIG_DIR / "routing_snapshot_bindings_w4_dh90_v2.json"
MAIN_W4_V2 = CONFIG_DIR / "main_w4_fixed_reference_v2.json"
REVIEWER_R1_5_V2 = CONFIG_DIR / "reviewer_r1_5_routing_sensitivity_v2.json"
CHECKPOINT_A_SHA = "e7bb75e367869bfefbac51a980aff18fe1da2fe2"
V2_REGISTRY_SHA256 = (
    "1c63d64d83c1c77daa8aabc4b422c6b62e9423310a13331c41c5f3a2a46dd5b5"
)
# Independently recorded during W3C.3C.2 from the external, intentionally
# untracked v1 registry/profiles before Checkpoint B was created.
AUDITED_EXTERNAL_V1_CONFIG_SHA256 = {
    "routing_snapshot_bindings_w4_dh90_v1.json": (
        "c844095d353714bafc2f89112ace5536b88d597f554f6ab4fbea7ca87beb6164"
    ),
    "main_w4_fixed_reference_v1.json": (
        "91aaef4bdb4f1ccf1efdee2df325304af5d5ad79da8f9115cbb8783e74b77a2a"
    ),
    "reviewer_r1_5_routing_sensitivity_v1.json": (
        "2266d58f2988f446e3caa99e4c69fc3074e19f6f7a1dde5c968a8b3430264dba"
    ),
}
# Independently recorded from the audited v1 manifests and byte-hashed v1
# reference-link CSV files. These constants must not be derived from v2.
AUDITED_V1_SCIENTIFIC_SHA256 = {
    30: {
        "topology": "a54337ac645c027348a8e39024e2d5a29ab24b4d847c60a5a9c1efb6b44f33c9",
        "routing": "eb47f9e9f2435ae0b89b3b6bebad59978088d41d3b10946cb5b15a2d6a2905ef",
        "reference_link_evidence": (
            "614833cab8276a9b80052a8ebcee1fdc926cac081d0cdb320cb0f36af247368d"
        ),
    },
    36: {
        "topology": "74afc73c9193a39e2847c81908d9a97ee724ea8a529d8d2bdafe9fbbb2a17585",
        "routing": "625d5e65dad3410b9be25e74130698f43e8c3a56a7c966623656616bc9a7b234",
        "reference_link_evidence": (
            "3d2da71f8551abc024585104fb03e834b733ddd06ffceb16647e081dacdbf422"
        ),
    },
    42: {
        "topology": "9868f156848212990f38d787a5ca84b03be27b7d72f3a66c9a7a238f66869cf7",
        "routing": "b0eae5ad422d1eb272202803a6a952f09dd1421e18873be59937de13e3edc239",
        "reference_link_evidence": (
            "274a8421467cd868ed7a77db87f827da94906fb5d2e22a5b0e67d5b662d76bd5"
        ),
    },
    48: {
        "topology": "2e912c9ca620eadddc3c6f9791f5d2ed6112e650c8ddaec0ca20c286ec185353",
        "routing": "085c198e6f12216eaa48abfbc39657a0e2e37c8e6cdd003785db7c9ad2902172",
        "reference_link_evidence": (
            "4b30d72e71140d31663f158b7a90b5ba9e85be0aec1e7fc966fc6f785f865ff0"
        ),
    },
    56: {
        "topology": "2c0c4ccbe6685384d585c88f81c07eb604922f969ae7525dc85067b18ab8f938",
        "routing": "137d608233ea36c61c04366b431f3cc95901685b46af6cd2175d303bd8fa3a9f",
        "reference_link_evidence": (
            "df0a0ab470fe538c08c8ac65f62cd140f7932871fd824c6cdd40985fbadf47ce"
        ),
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _routing_ids(routing_map: dict) -> dict[str, str | None]:
    result = {}
    for entity, parent in routing_map.items():
        entity_id = entity.id if hasattr(entity, "id") else str(entity)
        parent_id = (
            None
            if parent is None
            else (parent.id if hasattr(parent, "id") else str(parent))
        )
        result[str(entity_id)] = None if parent_id is None else str(parent_id)
    return result


def _assert_parent_and_full_route_integrity(snapshot: dict) -> None:
    parent_map = {
        row["source_id"]: row["parent_id"] for row in snapshot["routing_map"]
    }
    expected_ids = {"GW", "R1", "R2"} | {
        f"N{index}" for index in range(1, 101)
    }
    assert set(parent_map) == expected_ids
    assert parent_map["GW"] is None
    assert set(snapshot["full_routes"]) == expected_ids
    for source_id, route in snapshot["full_routes"].items():
        assert route[0] == source_id
        assert route[-1] == "GW"
        assert len(route) == len(set(route))
        for transmitter, receiver in zip(route, route[1:]):
            assert parent_map[transmitter] == receiver


@pytest.mark.extended_research
def test_v2_registry_is_complete_hash_pinned_and_clean_source_bound():
    bindings = load_routing_snapshot_bindings(
        V2_REGISTRY,
        expected_sha256=V2_REGISTRY_SHA256,
    )
    assert set(bindings) == set(W4_ROUTING_SNAPSHOT_SEEDS)
    raw = _read_json(V2_REGISTRY)
    assert raw["binding_revision"] == "w4_fixed_reference_dh90_v2"
    assert raw["source_checkpoint_a_sha"] == CHECKPOINT_A_SHA
    assert raw["snapshot_creation_provenance"] == {
        "branch": "",
        "head": CHECKPOINT_A_SHA,
        "status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
        "worktree_dirty": False,
    }
    for seed, binding in bindings.items():
        assert binding.reference_delta_h == 90
        assert raw["bindings"][str(seed)]["source_checkpoint_a_sha"] == (
            CHECKPOINT_A_SHA
        )


def test_v2_registry_fails_closed_for_wrong_hash_and_unknown_seed():
    with pytest.raises(RoutingSnapshotError, match="registry hash mismatch"):
        load_routing_snapshot_bindings(
            V2_REGISTRY,
            expected_sha256="0" * 64,
        )
    with pytest.raises(RoutingSnapshotError, match="configured seeds exactly"):
        load_routing_snapshot_bindings(
            V2_REGISTRY,
            expected_sha256=V2_REGISTRY_SHA256,
            expected_seeds=(*W4_ROUTING_SNAPSHOT_SEEDS, 99),
        )


@pytest.mark.extended_research
def test_v2_registry_fails_closed_for_corrupt_snapshot(tmp_path):
    raw = _read_json(V2_REGISTRY)
    for seed, row in raw["bindings"].items():
        row["manifest_path"] = str(
            (
                CONFIG_DIR
                / "routing_snapshot_bindings_w4_dh90_v2.json"
            ).parent
            / row["manifest_path"]
        )
    copied = tmp_path / "seed_30"
    shutil.copytree(V2_ROOT / "seed_30", copied)
    raw["bindings"]["30"]["manifest_path"] = str(
        copied / "routing_snapshot_manifest.json"
    )
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    with (copied / "routing_snapshot.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(RoutingSnapshotError, match="hash mismatch"):
        load_routing_snapshot_bindings(
            registry,
            expected_sha256=_sha256(registry),
        )


def test_v2_production_inputs_do_not_depend_on_external_v1_files():
    registry = _read_json(V2_REGISTRY)
    main = _read_json(MAIN_W4_V2)
    reviewer = _read_json(REVIEWER_R1_5_V2)
    assert main["routing_snapshot_bindings"] == V2_REGISTRY.name
    assert reviewer["routing_snapshot_bindings"] == V2_REGISTRY.name
    assert all(
        "w4_fixed_reference_dh90_v2" in row["manifest_path"]
        for row in registry["bindings"].values()
    )
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in AUDITED_EXTERNAL_V1_CONFIG_SHA256.values()
    )
    assert _sha256(V2_REGISTRY) not in set(
        AUDITED_EXTERNAL_V1_CONFIG_SHA256.values()
    )


@pytest.mark.extended_research
def test_v2_matches_independently_audited_v1_scientific_hashes():
    registry = _read_json(V2_REGISTRY)
    for seed in W4_ROUTING_SNAPSHOT_SEEDS:
        v2_dir = V2_ROOT / f"seed_{seed}"
        snapshot = _read_json(v2_dir / "routing_snapshot.json")
        manifest = _read_json(v2_dir / "routing_snapshot_manifest.json")
        expected = AUDITED_V1_SCIENTIFIC_SHA256[seed]
        assert snapshot["topology_sha256"] == expected["topology"]
        assert snapshot["selected_parent_routing_sha256"] == expected["routing"]
        assert manifest["topology_sha256"] == expected["topology"]
        assert manifest["selected_parent_routing_sha256"] == expected["routing"]
        assert _sha256(v2_dir / "reference_link_evidence.csv") == expected[
            "reference_link_evidence"
        ]
        binding = registry["bindings"][str(seed)]
        assert binding["topology_sha256"] == expected["topology"]
        assert binding["routing_sha256"] == expected["routing"]
        assert binding["manifest_sha256"] == _sha256(
            v2_dir / "routing_snapshot_manifest.json"
        )
        _assert_parent_and_full_route_integrity(snapshot)


@pytest.mark.extended_research
def test_fixed_routes_are_cross_terrain_invariant_and_match_dh90():
    for seed in W4_ROUTING_SNAPSHOT_SEEDS:
        snapshot = _read_json(V2_ROOT / f"seed_{seed}" / "routing_snapshot.json")
        expected_map = {
            row["source_id"]: row["parent_id"] for row in snapshot["routing_map"]
        }
        manifest = V2_ROOT / f"seed_{seed}" / "routing_snapshot_manifest.json"
        access_pathloss = {}
        relay_pathloss = {}
        for terrain in (20, 90, 160):
            config = dict(_reference_config(FIVE_MODEL_PILOT_CONFIG, seed))
            config.update(
                {
                    "terrain_delta_h": terrain,
                    "routing_strategy": ROUTING_STRATEGY_FIXED_REFERENCE,
                    "routing_reference_delta_h": 90,
                    "routing_snapshot_manifest": str(manifest),
                    "routing_snapshot_manifest_sha256": _sha256(manifest),
                }
            )
            config = resolve_configuration(config)
            infrastructure = build_infrastructure(config)
            topology = build_topology(config, infrastructure)
            routing = build_routing(config, infrastructure, topology)
            assert _routing_ids(routing.routing_map) == expected_map
            assert routing.route_construction_delta_h == 90
            assert routing.physical_terrain_delta_h == terrain
            access_pathloss[terrain] = tuple(routing.link_df["Pathloss(dB)"])
            relay_pathloss[terrain] = tuple(
                routing.relay_chain_links_df["Pathloss(dB)"]
            )

        assert access_pathloss[20] != access_pathloss[160]
        assert relay_pathloss[20] != relay_pathloss[160]

        config = dict(_reference_config(FIVE_MODEL_PILOT_CONFIG, seed))
        config.update(
            {
                "routing_strategy": ROUTING_STRATEGY_TERRAIN_SPECIFIC,
                "routing_snapshot_manifest": None,
                "routing_snapshot_manifest_sha256": None,
            }
        )
        config = resolve_configuration(config)
        infrastructure = build_infrastructure(config)
        topology = build_topology(config, infrastructure)
        routing = build_routing(config, infrastructure, topology)
        assert _routing_ids(routing.routing_map) == expected_map


@pytest.mark.extended_research
def test_production_profiles_are_isolated_and_plan_unique_cells_only():
    main = run_experiments(MAIN_W4_V2, dry_run=True)
    reviewer = run_experiments(REVIEWER_R1_5_V2, dry_run=True)
    assert main["run_count"] == main["unique_cell_count"] == 375
    assert reviewer["run_count"] == reviewer["unique_cell_count"] == 75
    assert main["reporting_namespace"] == "main_w4_fixed_reference"
    assert reviewer["reporting_namespace"] == "reviewer_r1_5_routing_sensitivity"
    assert {row["routing_strategy"] for row in main["runs"]} == {
        "fixed_reference"
    }
    assert {row["routing_strategy"] for row in reviewer["runs"]} == {
        "terrain_specific"
    }
    assert {row["terrain_delta_h"] for row in reviewer["runs"]} == {
        20,
        150,
        160,
    }
    assert all(row["routing_snapshot_manifest_sha256"] for row in main["runs"])
    assert all(
        row["routing_snapshot_manifest_sha256"] is None
        for row in reviewer["runs"]
    )
