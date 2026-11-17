"""Command-line entry point for the audited R-APC multi-hop LoRa publication."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from loraphymac.publications.paper_rapc_multihop_lora.analysis import DEFAULT_CONFIG
from loraphymac.publications.paper_rapc_multihop_lora.workflow import run_publication
from loraphymac.publications.provenance import PublicationProvenanceError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one completed APC terrain sweep and generate the "
            "R-APC multi-hop LoRa tables, figures, and provenance manifest."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Publication JSON configuration",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate source evidence without writing publication artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_publication(args.config, check_only=args.check_only)
    except (PublicationProvenanceError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
