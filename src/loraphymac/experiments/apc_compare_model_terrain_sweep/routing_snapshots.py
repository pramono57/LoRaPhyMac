"""Create immutable delta-h=90 routing snapshots without running traffic."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from loraphymac.experiments.common.multiseed_pipeline import (
    _source_fingerprint,
    file_sha256,
)
from loraphymac.metrics.instrumentation_metrics import (
    build_topology_parent_by_run_df,
)
from loraphymac.paths import repository_root
from loraphymac.routing.snapshot import (
    ROUTING_REFERENCE_DELTA_H,
    ROUTING_STRATEGY_TERRAIN_SPECIFIC,
    RoutingSnapshotError,
    file_sha256 as snapshot_file_sha256,
    write_routing_snapshot_bundle,
)
from loraphymac.simulation.builders import (
    build_infrastructure,
    build_routing,
    build_topology,
)
from loraphymac.simulator import resolve_configuration


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = repository_root(EXPERIMENT_DIR)
FIVE_MODEL_PILOT_CONFIG = (
    EXPERIMENT_DIR
    / "configs"
    / "pilot_w3c_five_model_n100_dh90_s30_r01.json"
)
W4_ROUTING_SNAPSHOT_SEEDS = (30, 36, 42, 48, 56)
W4_ROUTING_BINDINGS_SCHEMA_VERSION = "w4_routing_snapshot_bindings_v1"
DEFAULT_SNAPSHOT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
    / "apc_compare_model_terrain_sweep"
    / "routing_snapshots"
    / "w4_fixed_reference_dh90_v1"
)


@dataclass(frozen=True, slots=True)
class RoutingSnapshotBinding:
    """One hash-pinned seed-to-snapshot assignment."""

    seed: int
    manifest_path: Path
    manifest_sha256: str
    snapshot_id: str
    topology_sha256: str
    routing_sha256: str
    reference_delta_h: int
    creation_provenance: Mapping[str, Any]

    def config_overrides(self) -> dict[str, Any]:
        return {
            "routing_snapshot_manifest": str(self.manifest_path),
            "routing_snapshot_manifest_sha256": self.manifest_sha256,
        }


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RoutingSnapshotError(f"cannot read routing binding JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RoutingSnapshotError(f"routing binding JSON {path} must be an object")
    return value


def _binding_file(parent: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RoutingSnapshotError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = parent / path
    path = path.resolve()
    if not path.is_file():
        raise RoutingSnapshotError(f"{label} is missing: {path}")
    return path


def load_routing_snapshot_bindings(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_seeds: Sequence[int] = W4_ROUTING_SNAPSHOT_SEEDS,
) -> dict[int, RoutingSnapshotBinding]:
    """Verify the complete W4 seed registry without generating any routes."""
    registry_path = Path(path).expanduser().resolve()
    if not registry_path.is_file():
        raise RoutingSnapshotError(
            f"routing snapshot binding registry is missing: {registry_path}"
        )
    actual_registry_hash = snapshot_file_sha256(registry_path)
    if actual_registry_hash != str(expected_sha256).lower():
        raise RoutingSnapshotError(
            "routing snapshot binding registry hash mismatch: "
            f"expected {expected_sha256}, got {actual_registry_hash}"
        )
    raw = _read_object(registry_path)
    if raw.get("schema_version") != W4_ROUTING_BINDINGS_SCHEMA_VERSION:
        raise RoutingSnapshotError("unsupported routing snapshot binding schema")
    if raw.get("routing_strategy") != "fixed_reference":
        raise RoutingSnapshotError("routing binding registry must be fixed_reference")
    if int(raw.get("reference_delta_h", -1)) != ROUTING_REFERENCE_DELTA_H:
        raise RoutingSnapshotError("routing binding reference delta_h mismatch")
    creation = raw.get("snapshot_creation_provenance")
    if not isinstance(creation, Mapping):
        raise RoutingSnapshotError("snapshot creation provenance is missing")
    raw_bindings = raw.get("bindings")
    if not isinstance(raw_bindings, Mapping):
        raise RoutingSnapshotError("routing snapshot bindings are missing")
    expected = tuple(int(seed) for seed in expected_seeds)
    if set(raw_bindings) != {str(seed) for seed in expected}:
        raise RoutingSnapshotError(
            "routing snapshot bindings must match the configured seeds exactly"
        )

    result: dict[int, RoutingSnapshotBinding] = {}
    for seed in expected:
        row = raw_bindings[str(seed)]
        if not isinstance(row, Mapping):
            raise RoutingSnapshotError(f"routing binding for seed {seed} is invalid")
        manifest_path = _binding_file(
            registry_path.parent,
            row.get("manifest_path"),
            f"seed {seed} routing manifest",
        )
        manifest_hash = str(row.get("manifest_sha256", "")).lower()
        if snapshot_file_sha256(manifest_path) != manifest_hash:
            raise RoutingSnapshotError(f"seed {seed} routing manifest hash mismatch")
        manifest = _read_object(manifest_path)
        expected_values = {
            "seed": seed,
            "reference_delta_h": ROUTING_REFERENCE_DELTA_H,
            "snapshot_id": row.get("snapshot_id"),
            "topology_sha256": row.get("topology_sha256"),
            "selected_parent_routing_sha256": row.get("routing_sha256"),
        }
        for key, expected_value in expected_values.items():
            if manifest.get(key) != expected_value:
                raise RoutingSnapshotError(
                    f"seed {seed} routing manifest {key} mismatch"
                )
        manifest_creation = manifest.get("source_provenance", {}).get("git")
        if manifest_creation != creation:
            raise RoutingSnapshotError(
                f"seed {seed} snapshot creation provenance mismatch"
            )
        for label in ("routing_snapshot", "reference_link_evidence"):
            file_row = manifest.get(label)
            if not isinstance(file_row, Mapping):
                raise RoutingSnapshotError(
                    f"seed {seed} manifest is missing {label} provenance"
                )
            artifact_path = _binding_file(
                manifest_path.parent,
                file_row.get("path"),
                f"seed {seed} {label}",
            )
            if artifact_path.parent != manifest_path.parent.resolve():
                raise RoutingSnapshotError(
                    f"seed {seed} {label} escapes the snapshot bundle"
                )
            if snapshot_file_sha256(artifact_path) != file_row.get("sha256"):
                raise RoutingSnapshotError(f"seed {seed} {label} hash mismatch")
        result[seed] = RoutingSnapshotBinding(
            seed=seed,
            manifest_path=manifest_path,
            manifest_sha256=manifest_hash,
            snapshot_id=str(row["snapshot_id"]),
            topology_sha256=str(row["topology_sha256"]),
            routing_sha256=str(row["routing_sha256"]),
            reference_delta_h=ROUTING_REFERENCE_DELTA_H,
            creation_provenance=dict(creation),
        )
    return result


def _git_provenance() -> dict[str, object]:
    def output(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = output("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "head": output("rev-parse", "HEAD"),
        "branch": output("branch", "--show-current"),
        "worktree_dirty": bool(status),
        "status_porcelain_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _reference_config(config_path: Path, seed: int) -> dict:
    from loraphymac.experiments.apc_compare_model_terrain_sweep.run import (
        build_model_specs,
    )

    specs = build_model_specs(config_path)
    base = dict(specs["ff"].base_config)
    base.update(
        {
            "experiment_profile": "general_multisf_multibw",
            "seed": int(seed),
            "terrain_delta_h": ROUTING_REFERENCE_DELTA_H,
            "routing_strategy": ROUTING_STRATEGY_TERRAIN_SPECIFIC,
            "routing_reference_delta_h": None,
            "routing_snapshot_manifest": None,
            "routing_snapshot_manifest_sha256": None,
            "output_dir": None,
            "export_summary": False,
            "export_results": False,
            "export_debug": False,
            "export_internal": False,
            "plot_routing": False,
            "print_summary": False,
            "print_detail": False,
            "print_debug": False,
            "print_internal": False,
            "verbose": False,
            "run_id": None,
        }
    )
    return resolve_configuration(base)


def generate_reference_snapshots(
    *,
    config_path: str | Path = FIVE_MODEL_PILOT_CONFIG,
    output_root: str | Path = DEFAULT_SNAPSHOT_ROOT,
    seeds: Sequence[int] = W4_ROUTING_SNAPSHOT_SEEDS,
) -> list[dict[str, object]]:
    """Build route-only reference bundles for the requested deterministic seeds."""
    source_path = Path(config_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    source_provenance = {
        "generator": "w4_fixed_reference_routing_snapshot_v1",
        "source_fingerprint": _source_fingerprint(PROJECT_ROOT),
        "git": _git_provenance(),
        "source_config_path": str(source_path),
        "source_config_sha256": file_sha256(source_path),
    }
    records = []
    for seed in seeds:
        config = _reference_config(source_path, int(seed))
        infrastructure = build_infrastructure(config)
        topology = build_topology(config, infrastructure)
        routing = build_routing(config, infrastructure, topology)
        metadata = {
            "run_id": f"routing_reference_dh90_s{int(seed)}",
            "seed": int(seed),
            "node_ptx_mode": config["node_ptx_mode"],
            "relay_ptx_mode": config["relay_ptx_mode"],
            "terrain_delta_h": ROUTING_REFERENCE_DELTA_H,
            "routing_strategy": ROUTING_STRATEGY_TERRAIN_SPECIFIC,
            "route_construction_delta_h": ROUTING_REFERENCE_DELTA_H,
            "physical_terrain_delta_h": ROUTING_REFERENCE_DELTA_H,
            "routing_snapshot_id": None,
            "routing_snapshot_sha256": None,
            "topology_sha256": routing.topology_sha256,
            "selected_parent_routing_sha256": (
                routing.selected_parent_routing_sha256
            ),
        }
        evidence = build_topology_parent_by_run_df(routing, metadata)
        manifest = write_routing_snapshot_bundle(
            output_dir=output / f"seed_{int(seed)}",
            config=config,
            topology=topology,
            routing=routing,
            reference_link_evidence=evidence,
            frequency_mhz=infrastructure.channel_plan.get_freq(
                config["routing_ch_id"]
            ),
            source_provenance=source_provenance,
            source_config=config,
        )
        records.append(
            {
                "seed": int(seed),
                "manifest": str(manifest),
                "manifest_sha256": snapshot_file_sha256(manifest),
                "topology_sha256": routing.topology_sha256,
                "selected_parent_routing_sha256": (
                    routing.selected_parent_routing_sha256
                ),
            }
        )
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(FIVE_MODEL_PILOT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_SNAPSHOT_ROOT))
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(W4_ROUTING_SNAPSHOT_SEEDS),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    records = generate_reference_snapshots(
        config_path=args.config,
        output_root=args.output_dir,
        seeds=args.seeds,
    )
    print(json.dumps(records, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SNAPSHOT_ROOT",
    "RoutingSnapshotBinding",
    "W4_ROUTING_BINDINGS_SCHEMA_VERSION",
    "W4_ROUTING_SNAPSHOT_SEEDS",
    "generate_reference_snapshots",
    "load_routing_snapshot_bindings",
    "main",
]
