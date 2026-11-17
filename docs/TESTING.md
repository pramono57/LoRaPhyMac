# Test Suite Guide

LoRaPhyMAC keeps the public release checks separate from extended validation
that depends on local research evidence. No test or scientific assertion has
been removed.

## Public CI

GitHub Actions runs the public test suite on Python 3.10, 3.11, and 3.12:

```bash
pytest -q -m "not extended_research"
```

This runs the tests that can be collected from a clean public checkout without
frozen calibration bundles, routing snapshots, production output bundles, or
other local research evidence. Characterization tests remain included. The
CI workflow also runs Ruff, builds the source and wheel distributions, installs
the wheel, and verifies the package version.

## Extended Research Validation

Run all tests marked `extended_research` when the corresponding research
artifacts are available:

```bash
pytest -q -m extended_research
```

These tests verify frozen DBPC calibration inputs and hashes, W4 routing
snapshot manifests and evidence, terrain diagnostic configurations, and a
validated local production bundle. The artifacts are not part of the initial
public source checkout. Their paths and provenance are recorded by the
experiment configurations and research records; use the original validated
research workspace or its authorized evidence archive. Do not substitute
synthetic data for frozen production calibration or routing evidence.

The full local suite can be run with:

```bash
pytest -q
```

Without the research artifacts, the extended tests are expected to fail or
skip when they reach those missing inputs. Public CI instead excludes only
these exact nodes by marker; tests in the same files that do not depend on the
artifacts remain in the public suite.

## Scope and integrity policy

The public suite checks the release behavior that is self-contained in the
source distribution. Extended Research Validation checks claims tied to frozen
experiment inputs and production evidence. Both scopes remain in the
repository so they can be reviewed and run independently.

No test has been deleted, no assertion or golden metric has been weakened, and
no frozen experiment configuration, simulator source, calibration, snapshot,
or provenance record has been changed as part of this separation.
