"""Shared command-line adapter for content-based experiment runners."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence
from zoneinfo import ZoneInfo

from loraphymac.experiments.common.multiseed_pipeline import load_experiment_spec, run_experiment
from loraphymac.experiments.common.sweep_standard import reserve_sweep_cohort
from loraphymac.experiments.common.outputs import SWEEP_OUTPUT_LAYOUT_VERSION
from loraphymac.experiments.common.sweep_standard import PROJECT_TIMEZONE


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


def run_standard_cli(
    *,
    description: str,
    project_root: str | Path,
    scenario: str,
    default_config: str | Path,
    postprocess: Callable[[Path], object] | None = None,
    argv: Sequence[str] | None = None,
) -> int:
    """Run a future Sweep in a newly reserved, immutable standard cohort."""
    parser = build_parser(description=description, default_config=default_config)
    profiles = ("production", "pilot") if scenario == "load_sweep" else ("production",)
    parser.add_argument("--profile", choices=profiles, default="production")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.profile == "pilot":
        if args.output_root is None:
            raise ValueError(
                "Pilot profile requires --output-root outside repository research artifacts"
            )
        isolated_root = args.output_root.expanduser().resolve()
        protected_root = Path(project_root).resolve() / "artifacts"
        if isolated_root == protected_root or protected_root in isolated_root.parents:
            raise ValueError("Pilot --output-root must be outside repository artifacts")
        args.output_root = isolated_root
    elif args.output_root is not None:
        raise ValueError("--output-root is reserved for the explicitly selected pilot profile")
    spec = load_experiment_spec(args.config)
    spec = _select_sweep_profile(spec, args.profile, default_config)
    base_config = dict(spec.base_config)
    base_config["persistence_profile"] = "scientific"
    spec = replace(spec, base_config=base_config)
    if args.dry_run:
        instant = datetime.now(ZoneInfo(PROJECT_TIMEZONE))
        proposed_id = (
            f"{scenario}_v{int(spec.design_version)}_rNN_"
            f"{instant.strftime('%Y%m%d')}"
        )
        standard_root = (
            (args.output_root if args.profile == "pilot" else Path(project_root).resolve() / "artifacts" / "experiments")
            / f"APC_{scenario.upper()}"
        )
        dry_spec = replace(
            spec,
            experiment_id=proposed_id,
            output_dir=standard_root,
            output_layout_version=SWEEP_OUTPUT_LAYOUT_VERSION,
        )
        summary = run_experiment(
            dry_spec,
            project_root=project_root,
            resume=False,
            allow_incomplete=args.allow_incomplete,
            fail_fast=args.fail_fast,
            max_runs=args.max_runs,
            dry_run=True,
            show_progress=not args.quiet,
        )
        summary["proposed_cohort_id"] = proposed_id
        summary["cohort_reserved"] = False
        summary["output_layout_version"] = SWEEP_OUTPUT_LAYOUT_VERSION
        summary["sweep_profile"] = args.profile
        if summary.get("status") == "COMPLETE" and postprocess is not None:
            summary["analysis_artifacts"] = postprocess(Path(summary["experiment_dir"]))
            from loraphymac.experiments.common.multiseed_pipeline import atomic_write_json

            atomic_write_json(
                Path(summary["experiment_dir"]) / "experiment_status.json", summary
            )
    else:
        spec = reserve_sweep_cohort(
            spec, project_root=project_root, scenario=scenario,
            output_root=args.output_root,
            design_version=spec.design_version,
        )
        summary = run_experiment(
            spec,
            project_root=project_root,
            resume=False,
            allow_incomplete=args.allow_incomplete,
            fail_fast=args.fail_fast,
            max_runs=args.max_runs,
            dry_run=False,
            show_progress=not args.quiet,
        )
    analysis_failed = False

    if (
        not args.dry_run
        and summary.get("status") == "COMPLETE"
        and postprocess is not None
    ):
        from loraphymac.experiments.common.multiseed_pipeline import (
            atomic_write_json,
        )

        experiment_dir = Path(summary["experiment_dir"]).resolve()

        try:
            result = postprocess(experiment_dir)

            if not isinstance(result, dict) or result.get("status") != "COMPLETE":
                raise ValueError("Postprocessor did not report COMPLETE")

            analysis_path = Path(result["analysis_manifest"]).resolve()

            if experiment_dir not in analysis_path.parents:
                raise ValueError("Analysis manifest is outside the cohort")

            analysis_manifest = json.loads(
                analysis_path.read_text(encoding="utf-8")
            )

            if analysis_manifest.get("status") != "COMPLETE":
                raise ValueError("Analysis manifest is not COMPLETE")

            summary["analysis_status"] = "COMPLETE"
            summary["analysis_artifacts"] = result

        except Exception as exc:
            analysis_failed = True
            summary["analysis_status"] = "FAILED"
            summary["analysis_error_type"] = type(exc).__name__
            summary["analysis_error"] = str(exc)

        atomic_write_json(
            experiment_dir / "experiment_status.json",
            summary,
        )

    print(json.dumps(summary, indent=2, default=str))

    return 1 if (
        analysis_failed
        or summary.get("status") in {"FAILED", "INCOMPLETE", "PARTIAL"}
    ) else 0


def _select_sweep_profile(spec, selected: str, default_config: str | Path):
    """Apply an explicit standard-runner profile while rejecting silent downgrade."""
    from dataclasses import replace

    from loraphymac.experiments.apc_load_sweep.contract import (
        APC_LOAD_SWEEP_PILOT_PROFILE,
        APC_LOAD_SWEEP_STANDARD_PROFILE,
    )

    base = dict(spec.base_config)
    annotations = dict(spec.run_annotations)
    source_profile = base.get("experiment_profile")
    if selected == "production":
        if source_profile == APC_LOAD_SWEEP_PILOT_PROFILE:
            raise ValueError(
                "Pilot configuration requires explicit --profile pilot; refusing production mode"
            )
        if source_profile != APC_LOAD_SWEEP_STANDARD_PROFILE:
            raise ValueError(
                f"Production load profile requires {APC_LOAD_SWEEP_STANDARD_PROFILE}"
            )
        annotations["sweep_profile"] = "production"
        return replace(spec, run_annotations=annotations)

    if selected != "pilot":
        raise ValueError(f"Unsupported sweep profile: {selected}")
    if source_profile not in {APC_LOAD_SWEEP_STANDARD_PROFILE, APC_LOAD_SWEEP_PILOT_PROFILE}:
        raise ValueError("Pilot profile can only be selected from an APC load configuration")

    template = load_experiment_spec(default_config)
    candidate_base = dict(base)
    candidate_base["experiment_profile"] = APC_LOAD_SWEEP_STANDARD_PROFILE
    candidate_base["sim_time"] = template.base_config["sim_time"]
    template_base = dict(template.base_config)
    if candidate_base != template_base:
        raise ValueError(
            "Pilot profile accepts only production scientific parameters; "
            "unexpected base_config values were supplied"
        )
    production_factors = {key: list(values) for key, values in template.factors.items()}
    pilot_factors = {
        "node_num": [50],
        "node_period": [90],
        "node_ptx_mode": ["adaptive", "fixed"],
    }
    supplied_factors = {key: list(values) for key, values in spec.factors.items()}
    supplied_key = tuple(sorted((key, tuple(values)) for key, values in supplied_factors.items()))
    if supplied_key not in {
        tuple(sorted((key, tuple(values)) for key, values in production_factors.items())),
        tuple(sorted((key, tuple(values)) for key, values in pilot_factors.items())),
    }:
        raise ValueError("Pilot profile rejects unexpected parameter-grid values")
    if list(spec.seeds) not in [list(template.seeds), [42]]:
        raise ValueError("Pilot profile accepts only production seeds or pilot seed [42]")
    if spec.run_matrix is not None or spec.seed_config_overrides:
        raise ValueError("Pilot profile does not accept explicit run matrices or seed overrides")
    if spec.confidence_level != 0.95 or spec.retain_debug_tables:
        raise ValueError("Pilot profile requires 95% analysis confidence and standard output retention")
    base["experiment_profile"] = APC_LOAD_SWEEP_PILOT_PROFILE
    base["sim_time"] = 300
    annotations["sweep_profile"] = "pilot"
    annotations["evidence_class"] = "non-production integration evidence"
    return replace(
        spec,
        base_config=base,
        factors=pilot_factors,
        seeds=[42],
        run_annotations=annotations,
    )


__all__ = ["build_parser", "run_cli", "run_standard_cli"]
