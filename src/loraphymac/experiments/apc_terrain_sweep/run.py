"""Content-based entry point for the APC mode/terrain sweep."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from loraphymac.experiments.common.cli import build_parser
from loraphymac.experiments.common.multiseed_pipeline import load_experiment_spec, run_experiment
from loraphymac.experiments.common.sweep_standard import reserve_sweep_cohort
from loraphymac.paths import repository_root
from loraphymac.experiments.common.outputs import SWEEP_OUTPUT_LAYOUT_VERSION


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = repository_root(EXPERIMENT_DIR)
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "production_v1.json"
PROTECTED_ARTIFACT_ROOTS = (PROJECT_ROOT / "artifacts",)
DIAGNOSTIC_PROFILES = frozenset(
    {
        "apc_terrain_profile_diagnostic_v1",
        "apc_terrain_profile_full_sweep_v1",
        "apc_terrain_channel_occupancy_pilot_v1",
    }
)


def _validate_output_root(output_root: str | Path | None, profile: str | None) -> Path | None:
    """Require diagnostic outputs to be explicit and outside protected artifacts."""
    if profile in DIAGNOSTIC_PROFILES and output_root is None:
        raise ValueError("Diagnostic Terrain profiles require an explicit --output-root")
    if output_root is None:
        return None
    resolved = Path(output_root).expanduser().resolve()
    if not resolved.is_absolute():
        raise ValueError("--output-root must resolve to an absolute path")
    for protected in PROTECTED_ARTIFACT_ROOTS:
        protected_resolved = protected.expanduser().resolve()
        if resolved == protected_resolved or protected_resolved in resolved.parents:
            raise ValueError(
                f"--output-root resolves inside a protected research artifacts tree: {protected}"
            )
    return resolved


def run_experiments(
    config: str | Path = DEFAULT_CONFIG,
    *,
    resume: bool = True,
    allow_incomplete: bool = False,
    fail_fast: bool = False,
    max_runs: int | None = None,
    dry_run: bool = False,
    show_progress: bool = True,
    build_artifacts: bool = True,
    output_root: str | Path | None = None,
) -> dict:
    """Run or resume the publication-independent APC terrain sweep."""
    spec = load_experiment_spec(config)
    profile = spec.base_config.get("experiment_profile")
    resolved_output_root = _validate_output_root(output_root, profile)
    base_config = dict(spec.base_config)
    base_config["persistence_profile"] = "scientific"
    spec = replace(spec, base_config=base_config)
    if dry_run and resolved_output_root is not None:
        spec = replace(
            spec,
            output_dir=resolved_output_root / "APC_TERRAIN_SWEEP",
            output_layout_version=SWEEP_OUTPUT_LAYOUT_VERSION,
        )
    elif not dry_run:
        spec = reserve_sweep_cohort(
            spec, project_root=PROJECT_ROOT, scenario="terrain_sweep",
            output_root=resolved_output_root,
            design_version=spec.design_version,
        )
    summary = run_experiment(
        spec,
        project_root=PROJECT_ROOT,
        resume=resume,
        allow_incomplete=allow_incomplete,
        fail_fast=fail_fast,
        max_runs=max_runs,
        dry_run=dry_run,
        show_progress=show_progress,
    )
    if (
        build_artifacts
        and not dry_run
        and summary.get("status") in {"PASS", "COMPLETE"}
        and summary.get("seed_grid_complete") is True
    ):
        if profile == "apc_terrain_profile_diagnostic_v1":
            from loraphymac.experiments.apc_terrain_profile_diagnostic.analysis import (
                build_diagnostic_analysis,
            )

            analysis = build_diagnostic_analysis(
                experiment_dir=Path(spec.output_dir) / spec.experiment_id
            )
        else:
            from loraphymac.experiments.apc_terrain_sweep.postprocessing import (
                build_terrain_sweep_artifacts,
            )

            analysis = build_terrain_sweep_artifacts(
                config,
                experiment_dir=Path(spec.output_dir) / spec.experiment_id,
            )

        summary = {
            **summary,
            "analysis_artifacts": analysis,
        }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser(
        description="LoRaPhyMAC: APC mode and terrain-irregularity sweep.",
        default_config=DEFAULT_CONFIG,
    )
    args.add_argument("--output-root", type=Path, default=None)
    args = args.parse_args(argv)
    summary = run_experiments(
        args.config,
        resume=not args.no_resume,
        allow_incomplete=args.allow_incomplete,
        fail_fast=args.fail_fast,
        max_runs=args.max_runs,
        dry_run=args.dry_run,
        show_progress=not args.quiet,
        output_root=args.output_root,
    )
    print(json.dumps(summary, indent=2, default=str))
    failed = summary.get("status_counts", {}).get("FAILED", 0)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
