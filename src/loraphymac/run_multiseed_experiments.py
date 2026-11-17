from __future__ import annotations

import argparse
import json
from pathlib import Path

from loraphymac.experiments.common.multiseed_pipeline import load_experiment_spec, run_experiment
from loraphymac.paths import repository_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auditable, resumable multi-seed runner for LoRaPhyMAC experiments."
    )
    parser.add_argument("config", help="Path to an experiment JSON specification")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run progress output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    spec = load_experiment_spec(args.config)
    summary = run_experiment(
        spec,
        project_root=repository_root(Path(__file__)),
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
