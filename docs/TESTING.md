# Test Suite Guide

LoRaPhyMAC separates software regression, synthetic integration, and research-data
acceptance. A passing result applies only to the test boundary named by the
command; excluded tests are not reported as passed.

## Minimal APC Load Standard release CI

GitHub Actions runs the release test gate on Python 3.10, 3.11, and 3.12:

```bash
pytest -q -m "not extended_research and not unreleased_standard"
```

Pytest reports the number of passed, failed, and deselected tests in its final
summary. The gate covers software included in the minimal APC Load Standard
release, including software regression and the synthetic integration tests
within that release boundary. It does not claim that excluded tests passed.
The workflow also runs Ruff and builds and inspects the source and wheel
distributions.

## Test categories and markers

- **Software regression** checks deterministic code behavior, configuration
  contracts, grid construction, validation, and aggregation without relying on
  external research evidence.
- **Synthetic integration** exercises multiple components using temporary
  cohorts and synthetic outputs. It validates software plumbing only; it does
  not establish scientific equivalence to a real experiment.
- **Research-data acceptance** checks results or configuration against frozen
  scientific inputs, manifests, routing snapshots, production bundles, or
  other evidence. Such checks require the original, hash-verified evidence.

`extended_research` means a test requires frozen research evidence that is not
part of the public source checkout. Run these tests only when the corresponding
original evidence is available and its hashes have been verified:

```bash
pytest -q -m extended_research
```

Do not replace DBPC calibration or W4 routing evidence with synthetic files and
claim equivalent research validation.

`unreleased_standard` means a test exercises Unified Sweep Standard features
that are outside the current minimal APC Load Standard release. These tests
remain in the repository but are excluded from the minimal release gate. They
have **not** passed the APC Load release gate and are not part of its claims.
Their current dependencies include Compare figure-stem support, a Terrain
analysis helper, and Feedback Standard analysis source that are not included in
this release boundary. When the corresponding feature source is available, run:

```bash
pytest -q -m unreleased_standard
```

For the complete Unified Standard and research-data acceptance scope, provide
all required feature source and original evidence, then run:

```bash
pytest -q -m "unreleased_standard or extended_research"
```

The Compare Standard grid and Compare analysis acceptance tests are marked
`extended_research` because they also require the original DBPC/W4 references.
They must be run with both their feature source and the verified research data
available.

## Full local test suite

The unfiltered local suite is:

```bash
pytest -q
```

It includes both markers and therefore requires the corresponding features and
research evidence. A failure caused by a missing feature source or evidence is
not a pass and must be reported as such.

## Scope and integrity policy

The minimal public gate makes claims only about APC Load Standard and the
released shared software it exercises. Tests outside that boundary remain
available for their appropriate source and evidence environment. No test or
scientific assertion is removed or weakened by this boundary; excluded tests
must be identified by marker and reported as not run by the minimal gate.
