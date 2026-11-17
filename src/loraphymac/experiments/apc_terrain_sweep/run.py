"""Content-based entry point for the APC mode/terrain sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from loraphymac.experiments.common.cli import build_parser
from loraphymac.experiments.common.multiseed_pipeline import load_experiment_spec, run_experiment
from loraphymac.paths import repository_root


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = repository_root(EXPERIMENT_DIR)
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "production_v1.json"


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
) -> dict:
    """Run or resume the publication-independent APC terrain sweep."""
    spec = load_experiment_spec(config)
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
        and summary.get("status") == "PASS"
        and summary.get("seed_grid_complete") is True
    ):
        from loraphymac.experiments.apc_terrain_sweep.postprocessing import (
            build_terrain_sweep_artifacts,
        )

        summary = {
            **summary,
            "analysis_artifacts": build_terrain_sweep_artifacts(
                config,
                experiment_dir=Path(spec.output_dir) / spec.experiment_id,
            ),
        }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser(
        description="LoRaPhyMAC: APC mode and terrain-irregularity sweep.",
        default_config=DEFAULT_CONFIG,
    ).parse_args(argv)
    summary = run_experiments(
        args.config,
        resume=not args.no_resume,
        allow_incomplete=args.allow_incomplete,
        fail_fast=args.fail_fast,
        max_runs=args.max_runs,
        dry_run=args.dry_run,
        show_progress=not args.quiet,
    )
    print(json.dumps(summary, indent=2, default=str))
    failed = summary.get("status_counts", {}).get("FAILED", 0)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
