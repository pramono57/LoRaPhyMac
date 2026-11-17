"""CLI entry point for one model's APC load-sweep grid."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from loraphymac.experiments.common.cli import run_cli
from loraphymac.paths import repository_root


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = repository_root(EXPERIMENT_DIR)
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "ff_v1.json"


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(
        description="LoRaPhyMAC: paired R-APC and fixed-power scale/load sweep.",
        project_root=PROJECT_ROOT,
        default_config=DEFAULT_CONFIG,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
