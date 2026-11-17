"""CLI entry point for APC feedback robustness runs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from loraphymac.experiments.common.cli import run_cli
from loraphymac.paths import repository_root

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = repository_root(EXPERIMENT_DIR)
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "aa_feedback_robustness_v1.json"


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(
        description="LoRaPhyMAC: controlled additional APC feedback-loss robustness sweep.",
        project_root=PROJECT_ROOT,
        default_config=DEFAULT_CONFIG,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
