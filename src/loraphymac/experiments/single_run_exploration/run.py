"""Command-line entry point for an exploratory single run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from loraphymac.experiments.common.single_run_diagnostics import (
    generate_single_run_figures,
    generate_single_run_tables,
)
from loraphymac.experiments.single_run_exploration.workflow import (
    DEFAULT_CONFIG,
    default_exploration_config,
    run_single_exploration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-tables", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = default_exploration_config(
        config_path=args.config,
        output_dir=args.output_dir,
    )
    if args.run_id:
        config["run_id"] = args.run_id
        if args.output_dir is None:
            config.pop("output_dir", None)
    result = run_single_exploration(config)
    if not args.no_figures:
        generate_single_run_figures(result)
    if not args.no_tables:
        generate_single_run_tables(result)
    print(f"Exploration completed: {Path(result.resolved_config['output_dir']).resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
