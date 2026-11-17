"""Shared command-line adapter for content-based experiment runners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from loraphymac.experiments.common.multiseed_pipeline import load_experiment_spec, run_experiment


def build_parser(
    *,
    description: str,
    default_config: str | Path | None = None,
) -> argparse.ArgumentParser:
    """Build the common CLI while allowing an experiment to provide its config."""
    parser = argparse.ArgumentParser(description=description)
    if default_config is None:
        parser.add_argument("config", help="Path to an experiment JSON specification")
    else:
        parser.add_argument(
            "config",
            nargs="?",
            default=str(default_config),
            help="Experiment JSON specification (defaults to the experiment config)",
        )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run progress output")
    return parser


def run_cli(
    *,
    description: str,
    project_root: str | Path,
    default_config: str | Path | None = None,
    argv: Sequence[str] | None = None,
) -> int:
    """Execute one experiment runner through the shared audited pipeline."""
    args = build_parser(
        description=description,
        default_config=default_config,
    ).parse_args(argv)
    spec = load_experiment_spec(args.config)
    summary = run_experiment(
        spec,
        project_root=project_root,
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


__all__ = ["build_parser", "run_cli"]
