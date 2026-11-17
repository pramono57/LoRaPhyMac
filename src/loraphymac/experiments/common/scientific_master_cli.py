"""Manually rebuild an opt-in scientific master after its cohort is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from loraphymac.experiments.common.multiseed_pipeline import (
    rebuild_scientific_master_tables,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--project-root", type=Path, default=None)
    args = parser.parse_args(argv)
    tables = rebuild_scientific_master_tables(
        args.experiment_dir,
        project_root=args.project_root,
    )
    print(json.dumps({"status": "FINALIZED", "tables": sorted(tables)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
