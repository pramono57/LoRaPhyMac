"""Contract-driven persistence for reproducible scientific run outputs.

The multiseed adapter invokes this writer only for an explicit
``persistence_profile``. Omitting that selector preserves legacy persistence.
"""

from __future__ import annotations

import hashlib
import gzip as gzip_module
import importlib.metadata
import io
import json
import os
import platform
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from loraphymac.experiments.common.contract_registry import (
    experiment_contract_version,
    experiment_run_tables,
    validate_run_output_contract,
)
from loraphymac.experiments.common.outputs import ExperimentArtifactPaths
from loraphymac.experiments.common.scientific_output_contract import (
    LEGACY_PERSISTENCE_PROFILE,
    diagnostic_collection_table_keys,
    effective_persistence_profile,
    output_contract_for,
    preflight_output_violations,
    required_table_keys,
    validate_conditional_table_semantics,
)


OUTPUT_SCHEMA_VERSION = "scientific_run_output_v1"
MANIFEST_NAME = "scientific_output_manifest.json"
DIAGNOSTIC_TABLE_KEYS = diagnostic_collection_table_keys()
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TABLE_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_LEGACY_XLSX_SHEETS = {
    "core_summary_df": "01_core_summary",
    "run_metadata_df": "02_run_metadata",
    "frame_summary_df": "03_frame_summary",
    "hop_summary_df": "04_hop_summary",
    "frame_mtype_summary_df": "05_frame_mtype_summary",
    "frame_hop_event_summary_df": "06_frame_hop_event_summary",
    "frame_drop_reason_by_hop_df": "07_frame_drop_reason_by_hop",
    "relay_duty_cycle_df": "08_relay_duty_cycle_summary",
    "tx_queue_waiting_time_summary_df": "09_tx_queue_waiting_summary",
    "demodulator_pool_summary_df": "10_demodulator_pool_summary",
    "collision_summary_df": "11_collision_summary",
    "packet_fate_summary_df": "12_packet_fate_summary",
    "packet_fate_df": "D07_packet_fates",
    "energy_breakdown_df": "R01_energy_breakdown",
    "node_rx_matrix_df": "R02_node_rx_matrix",
    "per_channel_duty_cycle_df": "R03_per_channel_duty_cycle",
    "final_ptx_distribution_df": "R04_final_ptx_distribution",
    "relay_tx_mgc_queue_metrics_df": "R05_relay_tx_mgc_queue",
    "relay_tx_queue_metrics_df": "R06_relay_tx_queue_metrics",
    "tx_queue_traffic_class_metrics_df": "R07_tx_queue_traffic_class",
    "selected_next_hop_links_df": "R08_selected_next_hop_links",
    "relay_chain_links_df": "R09_relay_chain_links",
    "visual_links_df": "R10_visual_links",
    "ptx_history_df": "R11_ptx_history",
    "margin_history_df": "R12_margin_history",
    "apc_control_history_df": "R13_apc_control_history",
    "delivered_packet_df": "R14_delivered_packets",
    "packet_failure_df": "R15_packet_failures",
    "relay_rx_stats_df": "R16_relay_rx_statistics",
    "rx_channel_usage_df": "R17_rx_channel_usage",
    "apc_feedback_lifecycle_df": "R18_apc_feedback_lifecycle",
    "tx_airtime_by_entity_mtype_ptx_channel_df": "R19_tx_airtime_by_entity_mtype_ptx_channel",
    "wang_snr_report_lifecycle_df": "R20_wang_snr_report_lifecycle",
    "wang_power_command_lifecycle_df": "R22_wang_power_command_lifecycle",
    "application_queue_metrics_df": "R23_application_queue_metrics",
    "event_log_df": "D01_event_log",
    "tx_queue_waiting_time_df": "D02_tx_queue_waiting_time",
    "tx_queue_lifecycle_df": "D03_tx_queue_lifecycle",
    "demodulator_event_log_df": "D04_demodulator_events",
    "collision_event_log_df": "D05_collision_events",
    "collision_interference_edges_df": "D06_collision_interference_edges",
}
_FALLBACK_FILENAMES = {
    "core_summary_df": "core_summary.csv",
    "run_metadata_df": "run_metadata.csv",
    "frame_summary_df": "frame_summary.csv",
    "hop_summary_df": "hop_summary.csv",
    "frame_mtype_summary_df": "frame_mtype_summary.csv",
    "frame_hop_event_summary_df": "frame_hop_event_summary.csv",
    "frame_drop_reason_by_hop_df": "frame_drop_reason_by_hop.csv",
    "energy_breakdown_df": "energy_breakdown.csv",
    "final_ptx_distribution_df": "final_ptx_distribution.csv",
    "ptx_history_df": "ptx_history.csv",
    "margin_history_df": "margin_history.csv",
    "apc_control_history_df": "apc_control_history.csv",
    "delivered_packet_df": "delivered_packets.csv",
    "packet_failure_df": "packet_failures.csv",
    "tx_queue_waiting_time_summary_df": "tx_queue_waiting_summary.csv",
    "demodulator_pool_summary_df": "demodulator_pool_summary.csv",
    "collision_summary_df": "collision_summary.csv",
    "packet_fate_df": "packet_fates.csv",
    "packet_fate_summary_df": "packet_fate_summary.csv",
    "relay_duty_cycle_df": "relay_duty_cycle.csv",
    "per_channel_duty_cycle_df": "relay_channel_duty_cycle.csv",
    "node_tx_duty_cycle_df": "node_tx_duty_cycle.csv",
    "channel_occupancy_df": "channel_occupancy.csv",
    "apc_feedback_lifecycle_df": "apc_feedback_lifecycle.csv",
    "wang_snr_report_lifecycle_df": "wang_snr_report_lifecycle.csv",
    "wang_power_command_lifecycle_df": "wang_power_command_lifecycle.csv",
    "tx_airtime_by_entity_mtype_ptx_channel_df": ("tx_airtime_by_entity_mtype_ptx_channel.csv"),
    "event_log_df": "event_log.csv",
    "tx_queue_lifecycle_df": "queue_lifecycle.csv",
    "demodulator_event_log_df": "demodulator_events.csv",
    "collision_event_log_df": "collision_decisions.csv",
    "collision_interference_edges_df": "collision_interference_edges.csv",
}


class ScientificOutputError(ValueError):
    """Raised when a requested scientific output cannot be persisted safely."""


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_json_default,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def collect_source_provenance(
    repository_root: str | Path,
    relevant_paths: Sequence[str | Path],
    *,
    dependency_scope_complete: bool = False,
) -> dict[str, Any]:
    """Fingerprint an explicit source-file scope plus its Git dirty state.

    The caller supplies the complete relevant source/config/test path set; this
    function never scans the repository. Both current file bytes and staged /
    unstaged diffs are represented, so a commit SHA alone is never relied on.
    """

    root = Path(repository_root).resolve()
    normalized: list[str] = []
    for item in relevant_paths:
        relative = Path(item)
        if relative.is_absolute() or ".." in relative.parts:
            raise ScientificOutputError(f"source path must be repository-relative: {item}")
        value = relative.as_posix()
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ScientificOutputError("relevant_paths must not be empty")

    def git(*args: str) -> bytes:
        result = subprocess.run(["git", *args], cwd=root, check=False, capture_output=True)
        if result.returncode:
            message = result.stderr.decode("utf-8", errors="replace").strip()
            raise ScientificOutputError(f"git {' '.join(args[:2])} failed: {message}")
        return result.stdout

    commit = git("rev-parse", "HEAD").decode("ascii").strip()
    status_bytes = git("status", "--porcelain=v1", "-z", "--untracked-files=all", "--", *normalized)
    staged_diff_sha256 = _sha256_bytes(
        git("diff", "--cached", "--binary", "HEAD", "--", *normalized)
    )
    unstaged_diff_sha256 = _sha256_bytes(
        git("diff", "--binary", "--", *normalized)
    )
    status_by_path: dict[str, str] = {}
    status_entries = status_bytes.decode("utf-8", errors="surrogateescape").split("\0")
    cursor = 0
    while cursor < len(status_entries):
        entry = status_entries[cursor]
        cursor += 1
        if not entry:
            continue
        if len(entry) < 4:
            raise ScientificOutputError("could not parse scoped git status output")
        state, relative = entry[:2], entry[3:]
        status_by_path[relative] = state
        if "R" in state or "C" in state:
            if cursor >= len(status_entries) or not status_entries[cursor]:
                raise ScientificOutputError("git rename status omitted its original path")
            original = status_entries[cursor]
            cursor += 1
            status_by_path[original] = "D "

    files: list[dict[str, Any]] = []
    for relative in sorted(normalized):
        path = root / relative
        if path.is_file():
            working_hash: str | None = _sha256_file(path)
        elif path.exists():
            raise ScientificOutputError(f"source path is not a file: {relative}")
        else:
            working_hash = None
        state = status_by_path.get(relative, "  ")
        files.append(
            {
                "path": relative,
                "git_status": state,
                "working_sha256": working_hash,
            }
        )
    changed = [entry for entry in files if entry["git_status"] != "  "]
    payload = {
        "commit_sha": commit,
        "scope_paths": sorted(normalized),
        "files": files,
        "working_tree_status": "dirty" if changed else "clean",
        "changed_files": changed,
        "staged_diff_sha256": staged_diff_sha256,
        "unstaged_diff_sha256": unstaged_diff_sha256,
        "dependency_scope_complete": bool(dependency_scope_complete),
        "runtime_environment": _runtime_environment(),
        "ignored_runtime_files": [],
    }
    return {**payload, "source_fingerprint": _sha256_bytes(_canonical_json(payload).encode())}


def _runtime_environment() -> dict[str, Any]:
    distributions = sorted(
        (item.metadata.get("Name", item.name), item.version)
        for item in importlib.metadata.distributions()
    )
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "distributions": distributions,
    }


def collect_complete_source_provenance(repository_root: str | Path) -> dict[str, Any]:
    """Fingerprint the complete simulator package and declared project runtime.

    Ignored files under the runtime package make completeness unprovable and
    therefore force ``UNSAFE_TO_SKIP``. No repository-wide output artifacts are
    scanned.
    """
    root = Path(repository_root).resolve()

    def git(*args: str) -> list[str]:
        result = subprocess.run(["git", *args], cwd=root, check=False, capture_output=True)
        if result.returncode:
            raise ScientificOutputError(
                "could not establish complete source provenance: "
                + result.stderr.decode("utf-8", errors="replace").strip()
            )
        return [item.decode("utf-8", errors="surrogateescape") for item in result.stdout.split(b"\0") if item]

    source_paths = git(
        "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "src/loraphymac"
    )
    ignored_runtime_files = git(
        "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--", "src/loraphymac"
    )
    non_runtime_ignored = (
        "/__pycache__/",
        "/.ipynb_checkpoints/",
    )
    unknown_ignored = [
        item
        for item in ignored_runtime_files
        if not item.endswith((".pyc", ".DS_Store"))
        and not any(part in f"/{item}/" for part in non_runtime_ignored)
    ]
    project_files = [
        value
        for value in (
            "pyproject.toml",
            "requirements.txt",
            "requirements-dev.txt",
            "uv.lock",
            "poetry.lock",
        )
        if (root / value).is_file()
    ]
    paths = sorted(set(source_paths) | set(project_files))
    if not paths:
        raise ScientificOutputError("complete source provenance found no runtime files")
    provenance = collect_source_provenance(
        root,
        paths,
        dependency_scope_complete=not unknown_ignored,
    )
    provenance["ignored_runtime_files"] = sorted(unknown_ignored)
    # Recompute after adding the completeness evidence.
    payload = {key: value for key, value in provenance.items() if key != "source_fingerprint"}
    provenance["source_fingerprint"] = _sha256_bytes(_canonical_json(payload).encode())
    return provenance


def _validate_source_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "commit_sha",
        "scope_paths",
        "files",
        "working_tree_status",
        "changed_files",
        "staged_diff_sha256",
        "unstaged_diff_sha256",
        "source_fingerprint",
        "dependency_scope_complete",
        "runtime_environment",
        "ignored_runtime_files",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ScientificOutputError("source_provenance missing fields: " + ", ".join(missing))
    payload = {key: value[key] for key in required if key != "source_fingerprint"}
    if value["working_tree_status"] not in {"clean", "dirty"}:
        raise ScientificOutputError("source working_tree_status must be clean or dirty")
    files = value["files"]
    paths = value["scope_paths"]
    if not isinstance(files, list) or not isinstance(paths, list) or not paths:
        raise ScientificOutputError("source scope_paths and files must be non-empty lists")
    file_paths = [item.get("path") for item in files if isinstance(item, Mapping)]
    if sorted(file_paths) != sorted(paths):
        raise ScientificOutputError("source provenance file list does not match scope_paths")
    actual_changed = [item for item in files if item.get("git_status") != "  "]
    if actual_changed != value["changed_files"]:
        raise ScientificOutputError("source changed_files does not match scoped file statuses")
    expected_state = "dirty" if actual_changed else "clean"
    if value["working_tree_status"] != expected_state:
        raise ScientificOutputError("source working tree status contradicts changed_files")
    expected_fingerprint = _sha256_bytes(_canonical_json(payload).encode())
    if value["source_fingerprint"] != expected_fingerprint:
        raise ScientificOutputError("source provenance fingerprint is invalid")
    # Normalize tuples/Path-like JSON values so manifest comparisons are
    # stable after a write/read cycle.
    return json.loads(_canonical_json(dict(value)))


@dataclass(frozen=True, slots=True)
class OutputSelection:
    persistence_profile: str
    table_keys: tuple[str, ...]
    manager_writes: bool


@dataclass(frozen=True, slots=True)
class OutputWriteResult:
    status: str
    run_dir: Path | None
    manifest: Mapping[str, Any] | None
    legacy_passthrough: bool = False


@dataclass(frozen=True, slots=True)
class CompletedOutputCheck:
    safe_to_skip: bool
    reasons: tuple[str, ...]
    manifest: Mapping[str, Any] | None


class ScientificOutputManager:
    """Persists contract-required tables atomically under a run directory."""

    def __init__(
        self,
        experiment_dir: str | Path | ExperimentArtifactPaths,
        *,
        schema_version: str = OUTPUT_SCHEMA_VERSION,
    ) -> None:
        self.experiment_dir = Path(getattr(experiment_dir, "experiment_dir", experiment_dir))
        self.schema_version = schema_version

    def _effective_config(
        self,
        config: Mapping[str, Any],
        collection_profile: str | None,
        persistence_profile: str | None,
    ) -> dict[str, Any]:
        resolved = dict(config)
        configured_collection = str(resolved.get("output_profile", "detailed"))
        collection = collection_profile or configured_collection
        if collection != configured_collection:
            raise ScientificOutputError("collection_profile differs from resolved config")
        configured_persistence = effective_persistence_profile(resolved)
        persistence = persistence_profile or configured_persistence
        if persistence_profile is not None and resolved.get("persistence_profile") not in (
            None,
            persistence_profile,
        ):
            raise ScientificOutputError("persistence_profile differs from resolved config")
        if collection not in {"summary", "detailed"}:
            raise ScientificOutputError("collection_profile must be summary or detailed")
        if persistence != LEGACY_PERSISTENCE_PROFILE:
            resolved["persistence_profile"] = persistence
        return resolved

    def select_tables(
        self,
        config: Mapping[str, Any],
        tables: Mapping[str, pd.DataFrame],
        *,
        persistence_profile: str | None = None,
        additional_table_keys: Sequence[str] = (),
    ) -> OutputSelection:
        """Select complete evidence; legacy selection never drops old tables."""

        persistence = persistence_profile or effective_persistence_profile(config)
        if persistence == LEGACY_PERSISTENCE_PROFILE:
            return OutputSelection(persistence, tuple(sorted(tables)), False)
        contract = output_contract_for(config)
        if contract is None:
            raise ScientificOutputError(
                "explicit persistence requires a registered scientific output contract"
            )
        if persistence not in {"core", "scientific", "diagnostic"}:
            raise ScientificOutputError(f"unsupported persistence_profile: {persistence}")
        violations = preflight_output_violations(config)
        if violations:
            raise ScientificOutputError("; ".join(violations))
        if persistence == "core" and not contract.core_table_keys:
            raise ScientificOutputError(
                f"{contract.contract_id} does not explicitly permit core persistence"
            )
        if persistence == "core":
            required = set(contract.core_table_keys)
        else:
            required = set(required_table_keys(config))
        required.update(additional_table_keys)
        if persistence == "diagnostic":
            required.update(DIAGNOSTIC_TABLE_KEYS)
        missing = sorted(required - set(tables))
        if missing:
            raise ScientificOutputError("required output tables missing: " + ", ".join(missing))
        not_frames = sorted(key for key in required if not isinstance(tables[key], pd.DataFrame))
        if not_frames:
            raise ScientificOutputError(
                "required outputs must be pandas DataFrames: " + ", ".join(not_frames)
            )
        return OutputSelection(persistence, tuple(sorted(required)), True)

    def _table_filename(self, key: str, config: Mapping[str, Any]) -> str:
        if not _TABLE_KEY_RE.fullmatch(key):
            raise ScientificOutputError(f"invalid table key: {key!r}")
        registered = experiment_run_tables(config).get(key)
        if registered:
            return registered
        return _FALLBACK_FILENAMES.get(key, f"{key.removesuffix('_df')}.csv")

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(
                    value,
                    handle,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    default=_json_default,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _atomic_dataframe(path: Path, frame: pd.DataFrame, *, gzip: bool) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            if gzip:
                with temporary.open("wb") as raw:
                    with gzip_module.GzipFile(
                        filename="", mode="wb", fileobj=raw, mtime=0
                    ) as compressed:
                        text = io.TextIOWrapper(
                            compressed, encoding="utf-8", newline=""
                        )
                        try:
                            frame.to_csv(text, index=False)
                            text.flush()
                        finally:
                            text.detach()
            else:
                frame.to_csv(temporary, index=False)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _artifact_record(path: Path, relative_path: str, *, kind: str) -> dict[str, Any]:
        return {
            "path": relative_path,
            "kind": kind,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }

    def write_run(
        self,
        *,
        experiment_id: str,
        run_id: str,
        config: Mapping[str, Any],
        summary: Mapping[str, Any],
        tables: Mapping[str, pd.DataFrame],
        source_provenance: Mapping[str, Any],
        validation_results: Mapping[str, bool],
        validation_tables: Mapping[str, pd.DataFrame] | None = None,
        collection_profile: str | None = None,
        persistence_profile: str | None = None,
        gzip_tables: Sequence[str] = (),
        additional_table_keys: Sequence[str] = (),
        table_filename_overrides: Mapping[str, str] | None = None,
        pipeline_compatibility: Mapping[str, Any] | None = None,
        resolved_config_payload: Mapping[str, Any] | None = None,
    ) -> OutputWriteResult:
        """Write a fresh run; a visible final directory is always complete."""

        if not _RUN_ID_RE.fullmatch(experiment_id) or not _RUN_ID_RE.fullmatch(run_id):
            raise ScientificOutputError("experiment_id and run_id must be path-safe identifiers")
        if self.experiment_dir.name != experiment_id:
            raise ScientificOutputError(
                "experiment_id must match the configured experiment output directory name"
            )
        effective = self._effective_config(config, collection_profile, persistence_profile)
        persistence = persistence_profile or effective_persistence_profile(effective)
        selection = self.select_tables(
            effective,
            tables,
            persistence_profile=persistence,
            additional_table_keys=additional_table_keys,
        )
        gzip_all_enabled = effective.get("gzip_scientific_tables", False)
        if not isinstance(gzip_all_enabled, bool):
            raise ScientificOutputError("gzip_scientific_tables must be a boolean")
        if gzip_all_enabled and persistence not in {"scientific", "diagnostic"}:
            raise ScientificOutputError(
                "gzip_scientific_tables requires managed scientific persistence"
            )
        if not selection.manager_writes:
            if gzip_all_enabled:
                raise ScientificOutputError(
                    "gzip_scientific_tables cannot be used with legacy persistence"
                )
            return OutputWriteResult(
                status="LEGACY_PASSTHROUGH",
                run_dir=None,
                manifest=None,
                legacy_passthrough=True,
            )
        if not validation_results:
            raise ScientificOutputError("validation_results must be supplied before completion")
        failed_validations = sorted(
            key for key, passed in validation_results.items() if passed is not True
        )
        if failed_validations:
            raise ScientificOutputError("run validation failed: " + ", ".join(failed_validations))
        if not output_contract_for(effective):
            raise ScientificOutputError("no scientific output contract registered")
        conditional_errors = validate_conditional_table_semantics(
            effective, validation_tables or tables
        )
        if conditional_errors:
            raise ScientificOutputError("conditional table contract failed: " + "; ".join(conditional_errors))
        try:
            validate_run_output_contract(effective, summary, validation_tables or tables)
        except Exception as exc:
            raise ScientificOutputError(f"experiment output contract failed: {exc}") from exc
        source = _validate_source_provenance(source_provenance)

        config_record = dict(effective)
        config_record["output_profile"] = effective.get("output_profile", "detailed")
        if persistence == LEGACY_PERSISTENCE_PROFILE:
            config_record.pop("persistence_profile", None)
        else:
            config_record["persistence_profile"] = persistence
        config_fingerprint = _sha256_bytes(_canonical_json(config_record).encode())
        contract_version = experiment_contract_version(effective.get("experiment_profile"))
        if contract_version is None:
            raise ScientificOutputError("registered experiment contract has no version")
        expected_run = effective.get("run_id")
        if expected_run is not None and str(expected_run) != run_id:
            raise ScientificOutputError("run_id differs from resolved config")

        tables_root = self.experiment_dir / "runs"
        tables_root.mkdir(parents=True, exist_ok=True)
        final_dir = tables_root / run_id
        if final_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing run output: {final_dir}")
        lock = tables_root / f".{run_id}.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise FileExistsError(f"run output is being written or has stale lock: {lock}") from exc
        os.close(descriptor)
        staging = tables_root / f".{run_id}.partial-{uuid.uuid4().hex}"
        try:
            staging.mkdir()
            tables_dir = staging / "tables"
            tables_dir.mkdir()
            config_path = staging / "resolved_config.json"
            stored_config = dict(resolved_config_payload or config_record)
            self._atomic_json(config_path, stored_config)
            artifacts: dict[str, dict[str, Any]] = {
                "resolved_config.json": self._artifact_record(
                    config_path, "resolved_config.json", kind="resolved_config"
                )
            }
            table_artifacts: dict[str, dict[str, Any]] = {}
            compatibility_artifacts: dict[str, dict[str, Any]] = {}
            used_names: set[str] = set()
            gzip_set = set(gzip_tables)
            if gzip_all_enabled:
                selected_keys = set(selection.table_keys)
                if gzip_set and gzip_set != selected_keys:
                    raise ScientificOutputError(
                        "gzip_tables conflicts with gzip_scientific_tables=true"
                    )
                gzip_set = selected_keys
            unknown_gzip = gzip_set - set(selection.table_keys)
            if unknown_gzip:
                raise ScientificOutputError(
                    "gzip_tables includes unselected keys: " + ", ".join(sorted(unknown_gzip))
                )
            for key in selection.table_keys:
                filename = (table_filename_overrides or {}).get(
                    key, self._table_filename(key, effective)
                )
                if Path(filename).name != filename or filename in {".", ".."}:
                    raise ScientificOutputError(f"invalid table filename override: {filename!r}")
                if key in gzip_set and not filename.endswith(".gz"):
                    filename += ".gz"
                if filename in used_names:
                    raise ScientificOutputError(f"table filename collision: {filename}")
                used_names.add(filename)
                target = tables_dir / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                self._atomic_dataframe(target, tables[key], gzip=key in gzip_set)
                relative = target.relative_to(staging).as_posix()
                table_artifacts[key] = self._artifact_record(target, relative, kind="table")

            allowed_compatibility_files = {"core_summary.csv", "status.json"}
            for filename, value in (pipeline_compatibility or {}).items():
                if filename not in allowed_compatibility_files:
                    raise ScientificOutputError(
                        f"unsupported pipeline compatibility artifact: {filename}"
                    )
                target = staging / filename
                if isinstance(value, pd.DataFrame):
                    self._atomic_dataframe(target, value, gzip=False)
                    kind = "pipeline_compatibility_table"
                elif isinstance(value, Mapping):
                    self._atomic_json(target, value)
                    kind = "pipeline_compatibility_status"
                else:
                    raise ScientificOutputError(
                        f"compatibility artifact {filename} must be a DataFrame or mapping"
                    )
                compatibility_artifacts[filename] = self._artifact_record(
                    target, filename, kind=kind
                )

            for artifact in [
                artifacts["resolved_config.json"],
                *table_artifacts.values(),
                *compatibility_artifacts.values(),
            ]:
                target = staging / artifact["path"]
                if (
                    target.stat().st_size != artifact["size_bytes"]
                    or _sha256_file(target) != artifact["sha256"]
                ):
                    raise ScientificOutputError(f"artifact verification failed: {artifact['path']}")
            manifest: dict[str, Any] = {
                "manifest_schema_version": "scientific_output_manifest_v1",
                "output_schema_version": self.schema_version,
                "status": "COMPLETED",
                "experiment_id": experiment_id,
                "run_id": run_id,
                "experiment_profile": str(effective["experiment_profile"]),
                "experiment_contract_version": contract_version,
                "collection_profile": effective.get("output_profile", "detailed"),
                "persistence_profile": persistence,
                "config_fingerprint_sha256": config_fingerprint,
                "source_provenance": source,
                "required_table_keys": list(selection.table_keys),
                "gzip_table_keys": sorted(gzip_set),
                "additional_table_keys": sorted(set(additional_table_keys)),
                "table_filename_overrides": dict(table_filename_overrides or {}),
                "validation_results": dict(validation_results),
                "artifacts": artifacts,
                "compatibility_artifacts": compatibility_artifacts,
                "tables": table_artifacts,
            }
            manifest["validation_results"] = {
                "experiment_output_contract": True,
                **dict(validation_results),
            }
            manifest["required_artifacts"] = sorted(
                [record["path"] for record in artifacts.values()]
                + [record["path"] for record in compatibility_artifacts.values()]
                + [record["path"] for record in table_artifacts.values()]
            )
            self._atomic_json(staging / MANIFEST_NAME, manifest)
            _fsync_directory(tables_dir)
            _fsync_directory(staging)
            if final_dir.exists():
                raise FileExistsError(f"refusing to overwrite existing run output: {final_dir}")
            os.rename(staging, final_dir)
            _fsync_directory(tables_root)
            return OutputWriteResult("COMPLETED", final_dir, manifest)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        finally:
            if lock.exists():
                lock.unlink()

    def validate_completed(
        self,
        *,
        experiment_id: str,
        run_id: str,
        config: Mapping[str, Any],
        source_provenance: Mapping[str, Any],
        collection_profile: str | None = None,
        persistence_profile: str | None = None,
        additional_table_keys: Sequence[str] = (),
        table_filename_overrides: Mapping[str, str] | None = None,
    ) -> CompletedOutputCheck:
        """Return safe-to-skip only when every provenance and file check passes."""

        if not _RUN_ID_RE.fullmatch(experiment_id) or not _RUN_ID_RE.fullmatch(run_id):
            return CompletedOutputCheck(False, ("unsafe experiment or run identifier",), None)
        reasons: list[str] = []
        run_dir = self.experiment_dir / "runs" / run_id
        if self.experiment_dir.name != experiment_id:
            reasons.append("experiment_id differs from output directory name")
        manifest_path = run_dir / MANIFEST_NAME
        if not manifest_path.is_file():
            return CompletedOutputCheck(False, ("completed manifest missing",), None)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return CompletedOutputCheck(False, (f"manifest unreadable: {exc}",), None)

        try:
            effective = self._effective_config(config, collection_profile, persistence_profile)
        except ScientificOutputError as exc:
            return CompletedOutputCheck(
                False, (f"expected configuration invalid: {exc}",), manifest
            )
        persistence = persistence_profile or effective_persistence_profile(effective)
        selection = (
            self.select_tables(
                effective,
                {},
                persistence_profile=persistence,
                additional_table_keys=additional_table_keys,
            )
            if persistence == LEGACY_PERSISTENCE_PROFILE
            else None
        )
        if selection is not None:
            reasons.append("legacy output has no safe managed skip validation")
        contract = output_contract_for(effective)
        if contract is None:
            reasons.append("no registered scientific output contract")
        violations = preflight_output_violations(effective)
        reasons.extend(f"output contract violation: {item}" for item in violations)
        expected_contract_version = experiment_contract_version(effective.get("experiment_profile"))
        config_record = dict(effective)
        config_record["output_profile"] = effective.get("output_profile", "detailed")
        if persistence == LEGACY_PERSISTENCE_PROFILE:
            config_record.pop("persistence_profile", None)
        else:
            config_record["persistence_profile"] = persistence
        expected_config_hash = _sha256_bytes(_canonical_json(config_record).encode())
        try:
            expected_source = _validate_source_provenance(source_provenance)
            if expected_source.get("dependency_scope_complete") is not True:
                reasons.append("UNSAFE_TO_SKIP: source dependency scope is incomplete")
        except ScientificOutputError as exc:
            expected_source = None
            reasons.append(f"expected source provenance invalid: {exc}")

        expected = {
            "manifest_schema_version": "scientific_output_manifest_v1",
            "output_schema_version": self.schema_version,
            "status": "COMPLETED",
            "experiment_id": experiment_id,
            "run_id": run_id,
            "experiment_profile": str(effective.get("experiment_profile")),
            "experiment_contract_version": expected_contract_version,
            "collection_profile": effective.get("output_profile", "detailed"),
            "persistence_profile": persistence,
            "config_fingerprint_sha256": expected_config_hash,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                reasons.append(f"manifest mismatch: {key}")
        if expected_source is not None and manifest.get("source_provenance") != expected_source:
            reasons.append("manifest mismatch: source_provenance")
        validations = manifest.get("validation_results")
        if (
            not isinstance(validations, dict)
            or not validations
            or not all(value is True for value in validations.values())
        ):
            reasons.append("manifest validation results are not all PASS")

        try:
            if persistence == "core" and contract is not None:
                expected_keys = set(contract.core_table_keys)
            else:
                expected_keys = set(required_table_keys(effective))
            expected_keys.update(additional_table_keys)
            if persistence == "diagnostic":
                expected_keys.update(DIAGNOSTIC_TABLE_KEYS)
            table_records = manifest.get("tables")
            if not isinstance(table_records, dict) or set(table_records) != expected_keys:
                reasons.append("manifest table list differs from contract requirements")
            actual_gzip_keys = {
                key
                for key, record in (table_records or {}).items()
                if str(record.get("path", "")).endswith(".csv.gz")
            }
            manifest_gzip_keys = manifest.get("gzip_table_keys")
            if manifest_gzip_keys is None:
                # Backward compatibility for completed manifests written before
                # the compression inventory was added.
                manifest_gzip_keys = sorted(actual_gzip_keys)
            if set(manifest_gzip_keys) != actual_gzip_keys:
                reasons.append("manifest gzip table list differs from artifact paths")
            if effective.get("gzip_scientific_tables", False) and actual_gzip_keys != expected_keys:
                reasons.append(
                    "gzip_scientific_tables requires every contract table to be compressed"
                )
            artifacts = manifest.get("artifacts")
            if not isinstance(artifacts, dict) or "resolved_config.json" not in artifacts:
                reasons.append("manifest required resolved_config artifact is missing")
            compatibility = manifest.get("compatibility_artifacts", {})
            if not isinstance(compatibility, dict):
                reasons.append("manifest compatibility_artifacts is invalid")
                compatibility = {}
            records = (
                list((artifacts or {}).values())
                + list(compatibility.values())
                + list((table_records or {}).values())
            )
            expected_required_artifacts = sorted(str(record.get("path", "")) for record in records)
            if manifest.get("required_artifacts") != expected_required_artifacts:
                reasons.append("manifest required_artifacts mismatch")
            declared_required = manifest.get("required_table_keys")
            if not isinstance(declared_required, list) or set(declared_required) != expected_keys:
                reasons.append("manifest required_table_keys mismatch")
            if manifest.get("additional_table_keys") != sorted(set(additional_table_keys)):
                reasons.append("manifest additional_table_keys mismatch")
            if manifest.get("table_filename_overrides") != dict(table_filename_overrides or {}):
                reasons.append("manifest table_filename_overrides mismatch")
            if isinstance(artifacts, dict) and "resolved_config.json" in artifacts:
                config_file = run_dir / str(artifacts["resolved_config.json"].get("path", ""))
                if not config_file.is_file():
                    reasons.append("resolved_config file missing")
                else:
                    try:
                        stored_config = json.loads(config_file.read_text(encoding="utf-8"))
                        stored_simulator_config = stored_config.get(
                            "resolved_simulator_config", stored_config
                        )
                        if (
                            _sha256_bytes(_canonical_json(stored_simulator_config).encode())
                            != expected_config_hash
                        ):
                            reasons.append("resolved_config fingerprint mismatch")
                    except (OSError, json.JSONDecodeError, TypeError) as exc:
                        reasons.append(f"resolved_config unreadable: {exc}")
            seen_paths = {"resolved_config.json"}
            expected_paths = {MANIFEST_NAME, "resolved_config.json"}
            for record in records:
                relative = str(record.get("path", ""))
                if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
                    reasons.append("manifest contains unsafe artifact path")
                    continue
                if relative in seen_paths:
                    if relative == "resolved_config.json" and record in (artifacts or {}).values():
                        continue
                    reasons.append(f"duplicate artifact path: {relative}")
                seen_paths.add(relative)
                expected_paths.add(relative)
                path = run_dir / relative
                if path.is_symlink():
                    reasons.append(f"artifact is a symlink: {relative}")
                    continue
                if not path.is_file():
                    reasons.append(f"artifact missing: {relative}")
                    continue
                if path.stat().st_size != record.get("size_bytes"):
                    reasons.append(f"artifact size mismatch: {relative}")
                elif _sha256_file(path) != record.get("sha256"):
                    reasons.append(f"artifact checksum mismatch: {relative}")
            actual_paths = {
                item.relative_to(run_dir).as_posix()
                for item in run_dir.rglob("*")
                if item.is_file()
            }
            if actual_paths != expected_paths:
                reasons.append("run directory contains missing or unregistered files")
        except Exception as exc:
            reasons.append(f"manifest artifact validation failed: {exc}")
        return CompletedOutputCheck(not reasons, tuple(dict.fromkeys(reasons)), manifest)

    def read_table(self, source: str | Path, table_key: str) -> pd.DataFrame:
        """Read a managed table or supported legacy CSV/CSV.GZ/XLSX table."""

        path = Path(source)
        if path.is_dir():
            manifest_path = path / MANIFEST_NAME
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                record = manifest.get("tables", {}).get(table_key)
                if not record:
                    raise ScientificOutputError(f"table not recorded in manifest: {table_key}")
                path = path / record["path"]
            else:
                candidates = [
                    path / "tables" / f"{table_key}.csv",
                    path / "tables" / f"{table_key}.csv.gz",
                    path / "tables" / _FALLBACK_FILENAMES.get(table_key, ""),
                ]
                path = next((candidate for candidate in candidates if candidate.is_file()), path)
                if path.is_dir():
                    workbooks = sorted(path.glob("*.xlsx"))
                    if not workbooks:
                        raise FileNotFoundError(f"no legacy output for {table_key} under {source}")
                    path = workbooks[0]
        suffixes = path.suffixes
        if suffixes[-2:] == [".csv", ".gz"] or path.suffix == ".gz":
            return pd.read_csv(path, compression="gzip")
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix in {".xlsx", ".xls"}:
            sheet = _LEGACY_XLSX_SHEETS.get(table_key)
            if sheet is None:
                raise ScientificOutputError(
                    f"legacy workbook sheet mapping is unavailable for {table_key}; "
                    "pass a CSV export or add a verified sheet mapping"
                )
            return pd.read_excel(path, sheet_name=sheet)
        raise ScientificOutputError(f"unsupported scientific table format: {path.suffix}")


__all__ = [
    "CompletedOutputCheck",
    "DIAGNOSTIC_TABLE_KEYS",
    "MANIFEST_NAME",
    "OUTPUT_SCHEMA_VERSION",
    "OutputSelection",
    "OutputWriteResult",
    "ScientificOutputError",
    "ScientificOutputManager",
    "collect_source_provenance",
]
