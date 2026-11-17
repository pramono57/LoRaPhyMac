"""Backward-compatible facade for the relocated simulation event runner.

New code should use :func:`loraphymac.simulate`. These exports remain so
existing notebooks and external scripts do not break during restructuring.
"""

from loraphymac.config.defaults import DEFAULT_CONFIG, build_config
from loraphymac.config.validation import validate_experiment_profile
from loraphymac.simulation.reporting import print_one_run_summary
from loraphymac.simulation.runner import (
    main as _main,
    run_single_simulation,
)


__all__ = [
    "DEFAULT_CONFIG",
    "build_config",
    "print_one_run_summary",
    "run_single_simulation",
    "validate_experiment_profile",
]


if __name__ == "__main__":
    _main()
