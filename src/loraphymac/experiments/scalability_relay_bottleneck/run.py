"""Content-based scalability and relay-bottleneck experiment entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from loraphymac.experiments.common.cli import run_cli
from loraphymac.experiments.common.multiseed_pipeline import load_experiment_spec, run_experiment
from loraphymac.paths import repository_root


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = repository_root(EXPERIMENT_DIR)
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "pilot_v1.json"


def run_experiments(
    config: str | Path = DEFAULT_CONFIG,
    *,
    resume: bool = True,
    allow_incomplete: bool = False,
    fail_fast: bool = False,
    max_runs: int | None = None,
    dry_run: bool = False,
    show_progress: bool = True,
) -> dict:
    """Run or resume the scalability and relay-bottleneck experiment."""
    spec = load_experiment_spec(config)
    return run_experiment(
        spec,
        project_root=PROJECT_ROOT,
        resume=resume,
        allow_incomplete=allow_incomplete,
        fail_fast=fail_fast,
        max_runs=max_runs,
        dry_run=dry_run,
        show_progress=show_progress,
    )


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(
        description="LoRaPhyMAC: scalability and relay-bottleneck experiment.",
        project_root=PROJECT_ROOT,
        default_config=DEFAULT_CONFIG,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
