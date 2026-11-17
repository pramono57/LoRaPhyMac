from __future__ import annotations

import pytest

from loraphymac.runtime.wang_atp_a2only import (
    WANG_ATP_SF7_A2ONLY_MODEL_ID,
    WangA2PowerReductionController,
)


pytestmark = pytest.mark.unit


def decide(entity_type="node", current=17, samples=()):
    return WangA2PowerReductionController.decide(
        entity_type=entity_type,
        current_ptx_dbm=current,
        snr_samples_db=samples,
    )


def test_model_identity_and_initial_power_use_operational_maximum():
    assert WANG_ATP_SF7_A2ONLY_MODEL_ID == "wang_atp_sf7_a2only"
    assert WangA2PowerReductionController.initial_ptx_dbm("node") == 17
    assert WangA2PowerReductionController.initial_ptx_dbm("relay") == 20


@pytest.mark.parametrize(
    ("samples", "expected_margin", "expected_steps", "expected_ptx"),
    [
        ([-4.5], 3.0, 1, 14),
        ([-7.5], 0.0, 0, 17),
        ([-10.0], -2.5, 0, 17),
    ],
)
def test_positive_zero_and_negative_margin(samples, expected_margin, expected_steps, expected_ptx):
    result = decide(samples=samples)

    assert result.margin_db == pytest.approx(expected_margin)
    assert result.reduction_steps == expected_steps
    assert result.target_ptx_dbm == expected_ptx


def test_mean_mad_and_all_reduction_steps_are_applied_in_one_decision():
    result = decide(samples=[1.0, 3.0])

    assert result.snr_avg_db == pytest.approx(2.0)
    assert result.snr_mad_db == pytest.approx(1.0)
    assert result.margin_db == pytest.approx(8.5)
    assert result.reduction_steps == 2
    assert result.target_ptx_dbm == 11


@pytest.mark.parametrize(
    ("entity_type", "current", "expected_minimum"),
    [("node", 8, 5), ("relay", 14, 12)],
)
def test_large_margin_clamps_at_entity_hardware_minimum(
    entity_type, current, expected_minimum
):
    result = decide(entity_type=entity_type, current=current, samples=[100.0])

    assert result.target_ptx_dbm == expected_minimum


@pytest.mark.parametrize("samples", [[], [None], [None, None]])
def test_missing_samples_are_an_explicit_no_op(samples):
    result = decide(current=14, samples=samples)

    assert result.snr_sample_count == 0
    assert result.target_ptx_dbm == 14
    assert result.reduction_steps == 0
    assert result.reason == "missing_snr_samples"


@pytest.mark.parametrize(
    ("entity_type", "current", "samples"),
    [
        ("node", 17, [-20.0]),
        ("node", 11, [-4.5]),
        ("relay", 20, [10.0]),
        ("relay", 12, [10.0]),
    ],
)
def test_decision_never_increases_power(entity_type, current, samples):
    result = decide(entity_type=entity_type, current=current, samples=samples)

    assert result.target_ptx_dbm <= current
