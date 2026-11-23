"""CLI entry point for one model's APC load-sweep grid."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

from loraphymac.experiments.common.cli import build_parser, run_cli, run_standard_cli
from loraphymac.paths import repository_root


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = repository_root(EXPERIMENT_DIR)
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "standard_v1.json"
_LEGACY_PROFILES = frozenset(
    {
        "apc_load_sweep_v1",
        "apc_load_sweep_final_v1",
        "apc_load_sweep_full_factorial_v1",
        "apc_load_sweep_profile_acceptance_pilot_v1",
    }
)


def _build_analysis(experiment_dir: Path):
    from loraphymac.experiments.apc_load_sweep.standard_analysis import (
        build_load_sweep_analysis,
    )

    return build_load_sweep_analysis(experiment_dir)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "-h" in arguments or "--help" in arguments:
        return run_standard_cli(
            description="LoRaPhyMAC: paired R-APC and fixed-power scale/load sweep.",
            project_root=PROJECT_ROOT,
            scenario="load_sweep",
            default_config=DEFAULT_CONFIG,
            postprocess=_build_analysis,
            argv=arguments,
        )

    probe = build_parser(
        description="LoRaPhyMAC APC Load CLI profile probe.",
        default_config=DEFAULT_CONFIG,
    )
    probe.add_argument("--profile", choices=("production", "pilot"), default=None)
    probe.add_argument("--output-root", type=Path, default=None)
    parsed, _ = probe.parse_known_args(arguments)

    # Route only explicitly recognized legacy contracts to their historical
    # pipeline. Unknown or invalid profiles stay on the Standard path and fail
    # its contract validation instead of silently downgrading.
    with Path(parsed.config).expanduser().open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    base_config = raw_config.get("base_config") if isinstance(raw_config, dict) else None
    source_profile = (
        base_config.get("experiment_profile")
        if isinstance(base_config, dict)
        else None
    )
    if source_profile in _LEGACY_PROFILES and parsed.profile is None:
        return run_cli(
            description="LoRaPhyMAC: legacy APC load-sweep configuration.",
            project_root=PROJECT_ROOT,
            default_config=DEFAULT_CONFIG,
            argv=arguments,
        )

    return run_standard_cli(
        description="LoRaPhyMAC: paired R-APC and fixed-power scale/load sweep.",
        project_root=PROJECT_ROOT,
        scenario="load_sweep",
        default_config=DEFAULT_CONFIG,
        postprocess=_build_analysis,
        argv=arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
