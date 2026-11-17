from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from loraphymac.metrics.apc_metrics import (
    PTX_HISTORY_COLUMNS,
    build_apc_control_history_df,
    build_final_ptx_distribution_df,
    build_margin_history_df,
    build_ptx_history_df,
)
from loraphymac.metrics.statistics import Statistics


def _infer_entity_type(entity_id):
    if str(entity_id).startswith("R"):
        return "relay"
    if str(entity_id).startswith("GW"):
        return "gateway"
    return "node"


@pytest.mark.parametrize(
    "builder",
    [
        build_ptx_history_df,
        build_apc_control_history_df,
        build_margin_history_df,
    ],
)
def test_history_builders_preserve_record_values_and_empty_legacy_schema(builder):
    records = [{"run_id": "apc-8i", "time": 1.5, "entity_id": "N1"}]

    result = builder(records)
    empty = builder([])

    assert result.to_dict("records") == records
    assert result.columns.tolist() == ["run_id", "time", "entity_id"]
    assert empty.empty
    if builder is build_ptx_history_df:
        assert empty.columns.tolist() == list(PTX_HISTORY_COLUMNS)
    else:
        assert empty.columns.tolist() == []


def test_final_ptx_distribution_keeps_dbm_index_and_explicit_margin_domains():
    node = SimpleNamespace(
        id="N1",
        ptx_mode="adaptive",
        default_ptx=10.0,
        apc_tx_power_index=3,
        last_margin=99.0,
        last_rssi=-99.0,
        last_snr=-9.0,
        last_decision_margin=7.0,
        last_decision_rssi=-110.0,
        last_decision_snr=-5.0,
        last_apc_rx_margin=5.0,
        last_apc_rx_rssi=-112.0,
        last_apc_rx_snr=-7.0,
        last_feedback_from="GW",
        last_feedback_time=12.5,
    )
    relay_with_legacy_metrics = SimpleNamespace(
        id="R1",
        ptx_mode="fixed",
        default_ptx=12.0,
        apc_tx_power_index=4,
        last_margin=8.0,
        last_rssi=-108.0,
        last_snr=-4.0,
    )

    result = build_final_ptx_distribution_df(
        [node, relay_with_legacy_metrics],
        {"run_id": "apc-8i"},
        _infer_entity_type,
    )

    assert result.columns.tolist() == [
        "run_id",
        "entity_id",
        "entity_type",
        "ptx_mode",
        "final_ptx",
        "final_tx_power_index",
        "last_margin",
        "last_rssi",
        "last_snr",
        "last_decision_margin",
        "last_decision_rssi",
        "last_decision_snr",
        "last_apc_rx_margin",
        "last_apc_rx_rssi",
        "last_apc_rx_snr",
        "last_feedback_from",
        "last_feedback_time",
    ]
    explicit = result.iloc[0]
    assert explicit["entity_type"] == "node"
    assert explicit["final_ptx"] == 10.0
    assert explicit["final_tx_power_index"] == 3
    assert explicit["last_margin"] == 7.0
    assert explicit["last_rssi"] == -110.0
    assert explicit["last_snr"] == -5.0
    assert explicit["last_apc_rx_margin"] == 5.0

    legacy = result.iloc[1]
    assert legacy["entity_type"] == "relay"
    assert legacy["last_margin"] == 8.0
    assert legacy["last_rssi"] == -108.0
    assert legacy["last_snr"] == -4.0
    assert pd.isna(legacy["last_decision_margin"])
    assert pd.isna(legacy["last_apc_rx_margin"])


def test_statistics_recorders_preserve_apc_history_domain_contracts():
    statistics = Statistics()
    statistics.set_run_metadata(run_id="apc-history-8i")
    statistics.record_ptx_update(
        entity_id="N1",
        entity_type="node",
        old_ptx=17.0,
        new_ptx=10.0,
        old_tx_power_index=0,
        new_tx_power_index=3,
        margin=7.0,
        rssi=-110.0,
        snr=-5.0,
    )
    statistics.record_apc_control(
        time=2.0,
        event="APC_REQ_RX",
        sender_id="GW",
        sender_type="gateway",
        receiver_id="N1",
        receiver_type="node",
        controller_id="GW",
        controller_type="gateway",
        target_id="N1",
        target_type="node",
        control_direction="downlink",
        control_scope="node",
        old_ptx=17.0,
        new_ptx=10.0,
        new_tx_power_index=3,
    )
    statistics.record_margin_sample(
        entity_id="N1",
        entity_type="node",
        margin="7.5",
        rssi="invalid",
        snr="-5.5",
    )
    statistics.record_margin_sample(
        entity_id="N2",
        entity_type="node",
        margin=None,
    )

    ptx = statistics.get_ptx_history_df().iloc[0]
    control = statistics.get_apc_control_history_df().iloc[0]
    margin = statistics.get_margin_history_df().iloc[0]

    assert ptx["run_id"] == "apc-history-8i"
    assert ptx["tx_power_index"] == 3
    assert ptx["margin"] == 7.0
    assert ptx["decision_margin"] == 7.0
    assert pd.isna(ptx["apc_rx_margin"])
    assert control["tx_power_index"] == 3
    assert margin["margin"] == 7.5
    assert pd.isna(margin["rssi"])
    assert margin["snr"] == -5.5
    assert len(statistics.get_margin_history_df()) == 1


def test_final_ptx_distribution_empty_input_preserves_empty_legacy_schema():
    result = build_final_ptx_distribution_df([], {}, _infer_entity_type)

    assert result.empty
    assert result.columns.tolist() == []
