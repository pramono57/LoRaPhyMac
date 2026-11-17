"""CLI for the contracted APC closed-loop verification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from loraphymac.experiments.apc_closed_loop_verification.workflow import (
    DEFAULT_CONFIG,
    default_apc_closed_loop_config,
    run_apc_closed_loop_verification,
    verification_report_path,
)
from loraphymac.experiments.common.single_run_diagnostics import (
    generate_single_run_figures,
    generate_single_run_tables,
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
    config = default_apc_closed_loop_config(
        config_path=args.config,
        output_dir=args.output_dir,
    )
    if args.run_id:
        config["run_id"] = args.run_id
        if args.output_dir is None:
            config.pop("output_dir", None)
    result = run_apc_closed_loop_verification(config)
    if not args.no_figures:
        generate_single_run_figures(result)
    if not args.no_tables:
        generate_single_run_tables(result)
    print(f"Verification PASS: {verification_report_path(result).resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
