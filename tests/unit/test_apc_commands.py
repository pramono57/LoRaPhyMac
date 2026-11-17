import pytest

from loraphymac.mac.frame_defs import CID
from loraphymac.runtime.apc_commands import (
    APC_STATUS_APPLIED,
    APC_STATUS_BUSY,
    APC_STATUS_CH_OK,
    APC_STATUS_DR_OK,
    APC_STATUS_NBTRANS_OK,
    APC_STATUS_PTX_OK,
    APC_STATUS_UNSUPPORTED,
    decode_apc_ans,
    decode_apc_req,
    decode_dr_txpower,
    encode_apc_ans,
    encode_apc_req,
    encode_dr_txpower,
    is_apc_ans,
    is_apc_req,
    make_apc_status,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("dr", "index", "expected"),
    [(0, 0, 0x00), (1, 2, 0x12), (7, 15, 0x7F), (15, 15, 0xFF)],
)
def test_dr_txpower_layout_and_round_trip(dr, index, expected):
    assert encode_dr_txpower(dr, index) == expected
    assert decode_dr_txpower(expected) == {"dr": dr, "tx_power_index": index}


@pytest.mark.parametrize(
    ("dr", "index"),
    [(-1, 0), (16, 0), (0, -1), (0, 16)],
)
def test_dr_txpower_rejects_out_of_range_fields(dr, index):
    with pytest.raises(ValueError):
        encode_dr_txpower(dr, index)


@pytest.mark.parametrize(
    ("dr", "index", "mask", "nb_trans", "expected"),
    [
        (0, 0, 0x00, 0, bytes([0x01, 0x00, 0x00, 0x00])),
        (5, 3, 0xA5, 1, bytes([0x01, 0x53, 0xA5, 0x01])),
        (15, 15, 0xFF, 15, bytes([0x01, 0xFF, 0xFF, 0x0F])),
    ],
)
def test_apc_request_exact_wire_bytes_and_round_trip(
    dr, index, mask, nb_trans, expected
):
    encoded = encode_apc_req(dr, index, mask, nb_trans)
    assert encoded == expected
    assert len(encoded) == 4
    assert decode_apc_req(encoded) == {
        "cid": int(CID.APC_REQ),
        "cid_name": "APC_REQ",
        "dr": dr,
        "tx_power_index": index,
        "ch_mask": mask,
        "nb_trans": nb_trans,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dr": 0, "tx_power_index": 0, "ch_mask": -1},
        {"dr": 0, "tx_power_index": 0, "ch_mask": 256},
        {"dr": 0, "tx_power_index": 0, "nb_trans": -1},
        {"dr": 0, "tx_power_index": 0, "nb_trans": 16},
    ],
)
def test_apc_request_rejects_out_of_range_fields(kwargs):
    with pytest.raises(ValueError):
        encode_apc_req(**kwargs)


@pytest.mark.parametrize("data", [None, b"", b"\x01", b"\x01\x00\xff"])
def test_apc_request_rejects_missing_or_short_payload(data):
    with pytest.raises(ValueError):
        decode_apc_req(data)


def test_apc_request_rejects_unknown_cid():
    with pytest.raises(ValueError, match="Invalid APC_REQ CID"):
        decode_apc_req(b"\x7f\x00\x00\x00")


@pytest.mark.parametrize("data", [object(), "abcd", [999, 0, 0, 0]])
def test_apc_request_rejects_invalid_data_types(data):
    with pytest.raises((TypeError, ValueError)):
        decode_apc_req(data)


def test_status_bitmap_and_answer_wire_layout():
    status = make_apc_status()
    expected_status = (
        APC_STATUS_DR_OK
        | APC_STATUS_PTX_OK
        | APC_STATUS_CH_OK
        | APC_STATUS_NBTRANS_OK
        | APC_STATUS_APPLIED
    )
    assert status == expected_status == 0x1F
    encoded = encode_apc_ans(status)
    assert encoded == bytes([int(CID.APC_ANS), 0x1F])
    assert len(encoded) == 2
    decoded = decode_apc_ans(encoded)
    assert decoded["status"] == 0x1F
    assert all(decoded[key] for key in ("dr_ok", "ptx_ok", "ch_ok", "nbtrans_ok", "applied"))
    assert decoded["unsupported"] is False
    assert decoded["busy"] is False


def test_status_unsupported_and_busy_bits():
    status = make_apc_status(
        dr_ok=False,
        ptx_ok=False,
        ch_ok=False,
        nbtrans_ok=False,
        applied=False,
        unsupported=True,
        busy=True,
    )
    assert status == APC_STATUS_UNSUPPORTED | APC_STATUS_BUSY == 0x60
    decoded = decode_apc_ans(bytes([int(CID.APC_ANS), status]))
    assert decoded["unsupported"] is True
    assert decoded["busy"] is True
    assert decoded["applied"] is False


@pytest.mark.parametrize("data", [None, b"", b"\x02"])
def test_apc_answer_rejects_missing_or_short_payload(data):
    with pytest.raises(ValueError):
        decode_apc_ans(data)


def test_apc_answer_rejects_unknown_cid():
    with pytest.raises(ValueError, match="Invalid APC_ANS CID"):
        decode_apc_ans(b"\x01\x00")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-1, b"\x02\xff"), (0, b"\x02\x00"), (255, b"\x02\xff"), (256, b"\x02\x00")],
)
def test_answer_status_is_masked_to_one_byte(value, expected):
    assert encode_apc_ans(value) == expected


@pytest.mark.parametrize(
    ("data", "req", "ans"),
    [
        (b"\x01", True, False),
        (b"\x02", False, True),
        (b"", False, False),
        (None, False, False),
        ("bad", False, False),
    ],
)
def test_command_identity_helpers(data, req, ans):
    assert is_apc_req(data) is req
    assert is_apc_ans(data) is ans

