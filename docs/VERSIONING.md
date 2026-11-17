# Versioning policy

The current LoRaPhyMAC software release is **V1.0**. Python package and
machine-readable metadata use the semantic version **`1.0.0`**.

The release version has one source of truth:

```text
src/loraphymac/_version.py
└── __version__ = SIMULATOR_VERSION = "1.0.0"
```

`pyproject.toml` and `CITATION.cff` must carry the same value. The automated
version test rejects a mismatch.

## Independent version namespaces

The following identifiers are deliberately not reset when the software release
changes:

| Identifier | Meaning | Example |
|---|---|---|
| Software release | Complete LoRaPhyMAC package release | `1.0.0` / V1.0 |
| Simulator API | Compatibility of the public Python calling interface | `simulator_api_v1` |
| Experiment contract | Frozen factors, seeds, required outputs, and validation rules | `apc_terrain_sweep_contract_v1.0` |
| Dataset/experiment ID | Identity of immutable run evidence | `paper1_apc_v1` |
| Output layout | Directory and table-storage schema | `paper_artifacts_v1` |
| Metric/model schema | Meaning of a scientific model or table | `queue_metrics_version=2.0` |
| Publication workflow | Rules used to derive manuscript artifacts | `paper_rapc_multihop_lora_publication_v2_multi_experiment` |

Changing these identifiers merely to match the software release would destroy
compatibility information and could misrepresent historical research evidence.
They change only when their own contract changes.

## Provenance behavior

New single runs record `simulator_version` in run metadata and core output. New
multi-seed experiments record it in their experiment manifest, resolved-config
evidence, dry-run summary, and final status. Publication manifests record both
the generator release and the source experiment release when the latter is
available.

Historical experiment manifests without `simulator_version` remain readable.
Their source fingerprint, simulator API version, pipeline version, output-layout
version, and experiment-contract version continue to identify their actual
provenance. Historical artifacts are never rewritten just to display V1.0.

## Generated packaging metadata

Directories such as `src/loraphymac.egg-info/`, `build/`, and `dist/` are
generated locally and ignored by Git. They are not version sources and must not
be edited manually. Reinstalling the project regenerates them from
`pyproject.toml`:

```bash
python -m pip install -e '.[dev]'
```

The local checkout directory name is also not part of the package or simulator
version. Renaming a working directory is a separate filesystem migration and
is not required for a V1.0 release.

## Release checklist

1. Update `CHANGELOG.md` and `src/loraphymac/_version.py`.
2. Synchronize `pyproject.toml` and `CITATION.cff`.
3. Remove generated `build/`, `dist/`, and `src/loraphymac.egg-info/`
   directories before building release artifacts.
4. Run `python -m ruff check --no-cache src tests`, `git diff --check`, the
   version-consistency test, and the complete test suite.
5. Build source and wheel distributions from the clean tree, inspect their
   contents, and test an isolated wheel installation plus both console CLIs.
6. Run a smoke simulation and verify `simulator_version` in run metadata.
7. Use a new experiment ID when a source fingerprint or scientific contract
   differs from existing run evidence.
8. Wait for the GitHub Actions quality and package jobs to pass.
9. Tag the immutable release as `v1.0.0` only after every gate passes.
