from __future__ import annotations

import pandas as pd
import pytest

from loraphymac.mac.frame_defs import CID, MType
from loraphymac.metrics.frame_metrics import (
    FRAME_EVENT_NUMERIC_COLUMNS,
    FRAME_EVENT_REQUIRED_COLUMNS,
    build_frame_drop_reason_by_hop_df,
    build_frame_hop_event_summary_df,
    build_frame_mtype_summary_df,
    build_frame_summary_df,
    normalize_frame_event_df,
)


def _event(event, mtype, hop_count=1, hop_index=0, **overrides):
    values = {
        "event": event,
        "mtype": int(mtype),
        "hop_count": hop_count,
        "hop_index": hop_index,
        "fctrl": 0,
        "fopts_len": 0,
        "reason": None,
    }
    values.update(overrides)
    return values


def _summary_events():
    return normalize_frame_event_df(
        pd.DataFrame(
            [
                _event("TX", MType.DATA_UP, hop_count=1),
                _event("TX", MType.DATA_UP, hop_count=2),
                _event("RX", MType.DATA_UP, hop_count=1),
                _event("DROP", MType.DATA_UP, hop_count=2, reason="collision"),
                _event("TX", MType.DATA_UP_FWD, hop_count=2, hop_index=1),
                _event("RX", MType.DATA_UP_FWD, hop_count=2, hop_index=1),
                _event(
                    "TX",
                    MType.DATA_DOWN,
                    fctrl=0x80,
                    fopts_len=1,
                    control_cid=CID.APC_REQ,
                ),
                _event(
                    "RX",
                    MType.DATA_DOWN,
                    fctrl=0x80,
                    fopts_len=1,
                    control_cid=CID.APC_REQ,
                ),
                _event(
                    "TX",
                    MType.DATA_DOWN_FWD,
                    hop_count=2,
                    fctrl=0x80,
                    fopts_len=1,
                    control_cid=CID.APC_REQ,
                ),
                _event(
                    "DROP",
                    MType.DATA_DOWN_FWD,
                    hop_count=2,
                    reason="collision",
                    fctrl=0x80,
                    fopts_len=1,
                    control_cid=CID.APC_REQ,
                ),
                _event("TX", MType.ACK),
                _event("RX", MType.ACK),
            ]
        )
    )


def test_normalizer_adds_stable_columns_and_coerces_numeric_values_without_mutation():
    source = pd.DataFrame([{"event": "TX", "mtype": "1", "hop_count": "2"}])

    result = normalize_frame_event_df(source)

    assert set(FRAME_EVENT_REQUIRED_COLUMNS).issubset(result.columns)
    assert all(
        pd.api.types.is_numeric_dtype(result[column]) for column in FRAME_EVENT_NUMERIC_COLUMNS
    )
    assert result.loc[0, "mtype"] == 1
    assert result.loc[0, "hop_count"] == 2
    assert source.columns.tolist() == ["event", "mtype", "hop_count"]


def test_frame_summary_preserves_event_apc_delivery_hop_and_header_metrics():
    result = build_frame_summary_df(
        _summary_events(),
        {
            "run_id": "frame-8g",
            "apc_downlink_mode": "physical_all",
            "node_ptx_mode": "adaptive",
            "relay_ptx_mode": "fixed",
            "frame_enabled": True,
        },
        ptx_update_count=1,
        delivered_count=1,
    ).iloc[0]

    assert result["run_id"] == "frame-8g"
    assert result["total_event"] == 12
    assert result["total_tx_event"] == 6
    assert result["total_rx_event"] == 4
    assert result["total_drop_event"] == 2
    assert result["total_original_data_up_tx"] == 2
    assert result["total_data_up"] == 4
    assert result["total_data_up_fwd"] == 2
    assert result["total_data_down"] == 2
    assert result["total_data_down_fwd"] == 2
    assert result["total_ack"] == 2
    assert result["apc_total_tx"] == 2
    assert result["apc_total_rx"] == 1
    assert result["apc_total_drop"] == 1
    assert result["apc_apply_success_percent"] == 50.0
    assert result["apc_down_node_success_percent"] == 100.0
    assert result["apc_down_relay_success_percent"] == 0.0
    assert result["total_delivered_to_gateway"] == 1
    assert result["frame_pdr_percent"] == 50.0
    assert result["single_hop_original_tx"] == 1
    assert result["multi_hop_original_tx"] == 1
    assert result["avg_hop_count"] == 1.5
    assert result["max_hop_count"] == 2
    assert result["fctrl_nonzero_count"] == 4
    assert result["fopts_nonzero_count"] == 4
    assert result["apc_flag_count"] == 4
    assert result["ptx_update_count"] == 1
    assert result["stage1_status"] == "PASS"


@pytest.mark.parametrize(
    ("records", "metadata", "ptx_update_count", "expected"),
    [
        (
            [_event("TX", MType.DATA_UP)],
            {"apc_downlink_mode": "legacy"},
            0,
            "CHECK_LEGACY_APC_MODE",
        ),
        (
            [_event("TX", MType.DATA_UP)],
            {"node_ptx_mode": "adaptive", "frame_enabled": False},
            0,
            "CHECK_FRAME_DISABLED",
        ),
        (
            [_event("TX", MType.DATA_UP)],
            {"node_ptx_mode": "adaptive", "frame_enabled": True},
            0,
            "CHECK_ADAPTIVE_WITHOUT_APC_DOWNLINK_TX",
        ),
        (
            [
                _event("TX", MType.DATA_UP),
                _event(
                    "TX",
                    MType.DATA_DOWN,
                    fctrl=0x80,
                    fopts_len=1,
                    control_cid=CID.APC_REQ,
                ),
            ],
            {},
            1,
            "CHECK_PTX_UPDATE_WITHOUT_APC_RX",
        ),
        (
            [
                _event("TX", MType.DATA_UP),
                _event(
                    "TX",
                    MType.DATA_DOWN,
                    fctrl=0x80,
                    fopts_len=1,
                    control_cid=CID.APC_REQ,
                ),
            ],
            {},
            0,
            "CHECK_APC_TX_WITHOUT_APC_RX",
        ),
    ],
)
def test_frame_summary_preserves_stage1_validation_priority(
    records,
    metadata,
    ptx_update_count,
    expected,
):
    frame_events = normalize_frame_event_df(pd.DataFrame(records))

    result = build_frame_summary_df(
        frame_events,
        metadata,
        ptx_update_count,
        delivered_count=0,
    )

    assert result.loc[0, "stage1_status"] == expected


def test_mtype_summary_preserves_names_unknown_values_and_sort_order():
    frame_events = normalize_frame_event_df(
        pd.DataFrame(
            [
                _event("TX", MType.DATA_UP),
                _event("RX", MType.DATA_UP),
                _event("TX", MType.HELLO),
                _event("DROP", 99),
            ]
        )
    )

    result = build_frame_mtype_summary_df(frame_events)

    assert result.to_dict("records") == [
        {"mtype": 0, "mtype_name": "HELLO", "event": "TX", "count": 1},
        {"mtype": 1, "mtype_name": "DATA_UP", "event": "RX", "count": 1},
        {"mtype": 1, "mtype_name": "DATA_UP", "event": "TX", "count": 1},
        {"mtype": 99, "mtype_name": "UNKNOWN", "event": "DROP", "count": 1},
    ]


def test_frame_summary_separates_rapc_and_wang_downlink_by_wire_identity():
    records = []
    for mtype, event in (
        (MType.DATA_DOWN, "TX"),
        (MType.DATA_DOWN, "RX"),
        (MType.DATA_DOWN_FWD, "TX"),
        (MType.DATA_DOWN_FWD, "DROP"),
    ):
        records.append(
            _event(
                event,
                mtype,
                fctrl=0x80,
                fopts_len=4,
                control_cid=CID.APC_REQ,
            )
        )
    for mtype, event in (
        (MType.DATA_DOWN, "TX"),
        (MType.DATA_DOWN, "RX"),
        (MType.DATA_DOWN_FWD, "TX"),
        (MType.DATA_DOWN_FWD, "DROP"),
    ):
        records.append(
            _event(
                event,
                mtype,
                fctrl=11,
                fopts_len=11,
                control_cid=CID.WANG_POWER_COMMAND,
            )
        )
    # Unknown downlink control is physical traffic but is not credited to
    # either a successful R-APC or Wang command class.
    records.append(
        _event(
            "TX",
            MType.DATA_DOWN,
            fctrl=0x80,
            fopts_len=1,
            control_cid=0x7F,
        )
    )

    result = build_frame_summary_df(
        normalize_frame_event_df(pd.DataFrame(records)),
        {"frame_enabled": True},
        ptx_update_count=1,
        delivered_count=0,
    ).iloc[0]

    assert result["total_data_down_tx"] == 3
    assert result["total_data_down_rx"] == 2
    assert result["total_data_down_fwd_tx"] == 2
    assert result["total_data_down_fwd_drop"] == 2
    assert result["apc_total_tx"] == 2
    assert result["apc_total_rx"] == 1
    assert result["apc_total_drop"] == 1
    assert result["apc_down_node_success_percent"] == 100.0
    assert result["apc_down_relay_success_percent"] == 0.0


def test_hop_and_drop_summaries_preserve_reason_classification_and_order():
    frame_events = normalize_frame_event_df(
        pd.DataFrame(
            [
                _event("TX", MType.DATA_UP, hop_count=1),
                _event("DROP", MType.DATA_UP, hop_count=1, reason="collision"),
                _event("DROP", MType.DATA_UP, hop_count=1, reason="collision"),
                _event(
                    "DROP",
                    MType.DATA_UP_FWD,
                    hop_count=2,
                    hop_index=1,
                    reason="half_duplex_tx_busy",
                ),
                _event(
                    "DROP",
                    MType.DATA_UP_FWD,
                    hop_count=2,
                    hop_index=1,
                    reason="link_not_feasible",
                ),
                _event(
                    "DROP",
                    MType.DATA_UP_FWD,
                    hop_count=2,
                    hop_index=1,
                    reason="demod_busy",
                ),
            ]
        )
    )

    hop = build_frame_hop_event_summary_df(frame_events)
    drops = build_frame_drop_reason_by_hop_df(frame_events)

    assert hop["hop_count"].tolist() == [1, 2]
    assert hop.loc[0, "original_data_up_tx"] == 1
    assert hop.loc[0, "collision_drop"] == 2
    assert hop.loc[1, "half_duplex_drop"] == 1
    assert hop.loc[1, "not_feasible_drop"] == 1
    assert hop.loc[1, "demod_drop"] == 1
    assert drops.iloc[0].to_dict() == {
        "hop_count": 1,
        "reason": "collision",
        "drop_count": 2,
    }
    assert drops["hop_count"].tolist() == [1, 2, 2, 2]


def test_empty_frame_outputs_keep_legacy_schemas():
    empty = pd.DataFrame()

    assert build_frame_summary_df(empty, {}, 0, 0).empty
    assert build_frame_hop_event_summary_df(empty).empty
    assert build_frame_mtype_summary_df(empty).columns.tolist() == [
        "mtype",
        "mtype_name",
        "event",
        "count",
    ]
    assert build_frame_drop_reason_by_hop_df(empty).columns.tolist() == [
        "hop_count",
        "reason",
        "drop_count",
    ]
