"""Immutable publication of analysis outputs for Sweep Standard cohorts.

The first successful analysis keeps the cohort's established ``analysis/``
layout. Later analyses with changed inputs or generator identity are published
under ``analysis/versions/analysis-vN/``. Completed output paths are never
replaced.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping


class AnalysisPublicationError(RuntimeError):
    """Analysis outputs cannot be published without replacing existing data."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _identity(manifest: Mapping[str, Any]) -> str:
    ignored = {
        "analysis_id",
        "analysis_output_root",
        "analysis_revision",
        "status",
        "outputs",
        "tables",
        "figures",
        "artifacts",
        "artifact_counts",
        "created_at",
        "generated_at",
        "revision_error",
    }
    canonical = {key: value for key, value in manifest.items() if key not in ignored}
    data = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _validate_input_hashes(cohort_dir: Path, manifest: Mapping[str, Any]) -> None:
    records = manifest.get("inputs", [])
    if not isinstance(records, list):
        return
    for record in records:
        if not isinstance(record, Mapping):
            continue
        path_value = record.get("path") or record.get("scientific_output_manifest")
        expected = record.get("sha256") or record.get("scientific_output_manifest_sha256")
        if not path_value or not expected:
            continue
        path = Path(str(path_value))
        path = path if path.is_absolute() else cohort_dir / path
        if not path.is_file():
            raise AnalysisPublicationError(f"Analysis input is missing: {path}")
        if record.get("size_bytes") is not None and path.stat().st_size != int(
            record["size_bytes"]
        ):
            raise AnalysisPublicationError(f"Analysis input size mismatch: {path}")
        if _sha256(path) != expected:
            raise AnalysisPublicationError(f"Analysis input SHA-256 mismatch: {path}")


class AnalysisPublisher:
    """Stage analysis products and publish them without clobbering completed data."""

    def __init__(self, cohort_dir: str | Path):
        self.cohort_dir = Path(cohort_dir).resolve()
        self.analysis_dir = self.cohort_dir / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self._lock = self.analysis_dir / ".publication.lock"
        try:
            descriptor = os.open(self._lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise AnalysisPublicationError(
                f"Another analysis publication is active: {self._lock}"
            ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._temporary = tempfile.TemporaryDirectory(
            prefix=".analysis-stage-", dir=self.analysis_dir
        )
        self.stage_dir = Path(self._temporary.name)
        self.tables_dir = self.stage_dir / "tables"
        self.figures_dir = self.stage_dir / "figures"
        self.tables_dir.mkdir()
        self.figures_dir.mkdir()
        self._destination: Path | None = None
        self._manifest_path: Path | None = None
        self._closed = False

    def _manifest_candidates(self) -> list[Path]:
        primary = self.analysis_dir / "analysis_manifest.json"
        versions = self.analysis_dir / "versions"
        return [primary, *sorted(versions.glob("analysis-v*/analysis_manifest.json"))]

    def find_reusable(self, candidate: Mapping[str, Any]) -> dict[str, Any] | None:
        """Return an existing complete result when its provenance is identical."""
        _validate_input_hashes(self.cohort_dir, candidate)
        wanted = _identity(candidate)
        for path in self._manifest_candidates():
            if not path.is_file():
                continue
            try:
                prior = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if prior.get("status") != "COMPLETE" or _identity(prior) != wanted:
                continue
            records = prior.get("outputs", prior.get("artifacts", []))
            if not records:
                raise AnalysisPublicationError(
                    f"Completed analysis manifest has no output hashes: {path}"
                )
            for record in records:
                output = self.cohort_dir / record["path"]
                if not output.is_file() or output.stat().st_size != int(record["size_bytes"]):
                    raise AnalysisPublicationError(
                        f"Completed analysis output is missing or changed: {output}"
                    )
                if _sha256(output) != record["sha256"]:
                    raise AnalysisPublicationError(
                        f"Completed analysis output hash mismatch: {output}"
                    )
            return self._result(path, prior)
        return None

    def _initial_destination_is_empty(self) -> bool:
        manifest_path = self.analysis_dir / "analysis_manifest.json"
        if not manifest_path.is_file():
            return True
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if current.get("status") in {"COMPLETE", "PASS"}:
            return False
        for folder in (self.analysis_dir / "tables", self.analysis_dir / "figures"):
            if folder.is_dir() and any(folder.iterdir()):
                return False
        return True

    def _allocate_destination(self) -> tuple[Path, str]:
        if self._initial_destination_is_empty():
            return self.analysis_dir, "analysis-v1"
        versions = self.analysis_dir / "versions"
        versions.mkdir(exist_ok=True)
        used = []
        for entry in versions.iterdir():
            match = re.fullmatch(r"analysis-v(\d+)", entry.name)
            if match:
                used.append(int(match.group(1)))
        number = max([1, *used]) + 1
        name = f"analysis-v{number}"
        destination = versions / name
        destination.mkdir()  # no-clobber reservation for this revision
        return destination, name

    def publish(
        self,
        manifest: Mapping[str, Any],
        output_files: Iterable[Path],
    ) -> dict[str, Any]:
        """Publish staged outputs and mark the selected manifest COMPLETE last."""
        _validate_input_hashes(self.cohort_dir, manifest)
        files = sorted({Path(path).resolve() for path in output_files})
        if not files:
            raise AnalysisPublicationError("Cannot publish an analysis with no outputs")
        for path in files:
            if not path.is_file() or not path.is_relative_to(self.stage_dir):
                raise AnalysisPublicationError(f"Invalid staged output: {path}")
            if not (path.is_relative_to(self.tables_dir) or path.is_relative_to(self.figures_dir)):
                raise AnalysisPublicationError(f"Output must be in tables/ or figures/: {path}")

        destination, revision = self._allocate_destination()
        self._destination = destination
        self._manifest_path = destination / "analysis_manifest.json"
        relative_map: dict[str, str] = {}
        for source in files:
            area = "tables" if source.is_relative_to(self.tables_dir) else "figures"
            target = destination / area / source.name
            relative_map[source.relative_to(self.stage_dir).as_posix()] = target.relative_to(
                self.cohort_dir
            ).as_posix()

        stage_prefix = self.stage_dir.relative_to(self.cohort_dir).as_posix() + "/"

        def rewrite(value: Any) -> Any:
            if isinstance(value, str):
                if value.startswith(stage_prefix):
                    staged_relative = value[len(stage_prefix) :]
                    return relative_map.get(
                        staged_relative,
                        (destination.relative_to(self.cohort_dir) / staged_relative).as_posix(),
                    )
                return value
            if isinstance(value, list):
                return [rewrite(item) for item in value]
            if isinstance(value, dict):
                return {key: rewrite(item) for key, item in value.items()}
            return value

        pending = rewrite(dict(manifest))
        pending.update({"analysis_revision": revision, "status": "PENDING"})
        if pending.get("source_cohort_id"):
            pending["analysis_id"] = f"{pending['source_cohort_id']}-{revision}"
        try:
            for name in ("tables", "figures"):
                (destination / name).mkdir(exist_ok=True)
            _write_json(self._manifest_path, pending)
            for source in files:
                area = "tables" if source.is_relative_to(self.tables_dir) else "figures"
                target = destination / area / source.name
                try:
                    os.link(source, target)
                except FileExistsError:
                    if _sha256(source) != _sha256(target):
                        raise AnalysisPublicationError(
                            f"Refusing to replace existing analysis output: {target}"
                        )
                if source.stat().st_size != target.stat().st_size or _sha256(source) != _sha256(
                    target
                ):
                    raise AnalysisPublicationError(f"Published output failed validation: {target}")

            _validate_input_hashes(self.cohort_dir, manifest)

            complete = dict(manifest)
            complete.update({"analysis_revision": revision, "status": "COMPLETE"})
            complete["analysis_id"] = (
                f"{complete['source_cohort_id']}-{revision}"
                if complete.get("source_cohort_id")
                else f"{complete.get('analysis_id', 'analysis')}-{revision}"
            )
            complete = rewrite(complete)
            _write_json(self._manifest_path, complete)
            result = self._result(self._manifest_path, complete)
            result["reused"] = False
            return result
        except Exception as exc:
            failed = dict(pending)
            failed["status"] = "FAILED"
            failed["failure"] = f"{type(exc).__name__}: {exc}"
            try:
                _write_json(self._manifest_path, failed)
            except OSError:
                pass
            raise

    def _result(self, manifest_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
        records = manifest.get("outputs", manifest.get("artifacts", []))
        tables = manifest.get("tables") or [
            item["path"] for item in records if str(item.get("path", "")).lower().endswith(".csv")
        ]
        figures = manifest.get("figures") or [
            item["path"]
            for item in records
            if Path(str(item.get("path", ""))).suffix.lower() in {".png", ".pdf", ".svg"}
        ]
        return {
            "status": "COMPLETE",
            "analysis_manifest": str(manifest_path),
            "tables": [str(self.cohort_dir / value) for value in tables],
            "figures": [str(self.cohort_dir / value) for value in figures],
            "analysis_revision": manifest.get("analysis_revision", "analysis-v1"),
            "reused": True,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._temporary.cleanup()
        try:
            self._lock.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> "AnalysisPublisher":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if exc is not None and self._manifest_path is not None:
                current = json.loads(self._manifest_path.read_text(encoding="utf-8"))
                if current.get("status") != "COMPLETE":
                    current["status"] = "FAILED"
                    current["failure"] = f"{exc_type.__name__}: {exc}"
                    _write_json(self._manifest_path, current)
        finally:
            self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["AnalysisPublicationError", "AnalysisPublisher"]
