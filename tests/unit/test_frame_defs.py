import pytest

from loraphymac.mac.frame_defs import (
    CID,
    FRAME_FIXED_OVERHEAD_BYTES,
    FPort,
    MType,
    cid_name,
    decode_fctrl,
    decode_hop_ctrl,
    decode_mhdr,
    encode_fctrl,
    encode_hop_ctrl,
    encode_mhdr,
    frame_size_bytes,
    mtype_name,
)
from tests.helpers.reference_formulas import reference_frame_size


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("member", "value"),
    [
        (MType.DATA_UP, 0x01),
        (MType.DATA_UP_FWD, 0x02),
        (MType.DATA_DOWN, 0x03),
        (MType.DATA_DOWN_FWD, 0x04),
    ],
)
def test_required_mtypes(member, value):
    assert int(member) == value


def test_frame_enum_metadata_values():
    assert int(CID.APC_REQ) == 1
    assert int(CID.APC_ANS) == 2
    assert int(FPort.MAC_CONTROL) == 0
    assert int(FPort.SENSOR_DATA) == 1


@pytest.mark.parametrize("payload", [0, 1, 50])
@pytest.mark.parametrize("fopts_len", [0, 4, 15])
def test_frame_size_uses_independent_wire_layout(payload, fopts_len):
    expected = reference_frame_size(payload, fopts_len)
    assert FRAME_FIXED_OVERHEAD_BYTES == 12
    assert frame_size_bytes(payload, fopts_len) == expected


@pytest.mark.parametrize(
    ("payload", "fopts_len"),
    [(-1, 0), (0, -1), (0, 16)],
)
def test_frame_size_rejects_invalid_lengths(payload, fopts_len):
    with pytest.raises(ValueError):
        frame_size_bytes(payload, fopts_len)


@pytest.mark.parametrize(
    ("mtype", "rfu", "version"),
    [(0, 0, 0), (1, 0, 0), (4, 3, 2), (7, 7, 3)],
)
def test_mhdr_encode_decode_round_trip(mtype, rfu, version):
    encoded = encode_mhdr(mtype=mtype, rfu=rfu, version=version)
    assert encoded == (mtype << 5) | (rfu << 2) | version
    assert decode_mhdr(encoded) == {
        "mtype": mtype,
        "rfu": rfu,
        "version": version,
    }


def test_mhdr_masks_out_of_width_fields():
    assert decode_mhdr(encode_mhdr(15, 15, 7)) == {
        "mtype": 7,
        "rfu": 7,
        "version": 3,
    }


@pytest.mark.parametrize(
    ("hop_index", "hop_limit"),
    [(0, 0), (0, 15), (3, 12), (15, 15)],
)
def test_hop_ctrl_round_trip(hop_index, hop_limit):
    expected = (hop_index << 4) | hop_limit
    assert encode_hop_ctrl(hop_index, hop_limit) == expected
    assert decode_hop_ctrl(expected) == {
        "hop_index": hop_index,
        "hop_limit": hop_limit,
    }


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, 0x00),
        ({"apc": True}, 0x80),
        ({"ack_req": True}, 0x40),
        ({"bcast": True}, 0x20),
        ({"pending": True}, 0x10),
        ({"fopts_len": 15}, 0x0F),
        (
            {
                "apc": True,
                "ack_req": True,
                "bcast": True,
                "pending": True,
                "fopts_len": 4,
            },
            0xF4,
        ),
    ],
)
def test_fctrl_individual_bits_and_layout(kwargs, expected):
    encoded = encode_fctrl(**kwargs)
    assert encoded == expected
    decoded = decode_fctrl(encoded)
    assert decoded["apc"] == bool(expected & 0x80)
    assert decoded["ack_req"] == bool(expected & 0x40)
    assert decoded["bcast"] == bool(expected & 0x20)
    assert decoded["pending"] == bool(expected & 0x10)
    assert decoded["fopts_len"] == expected & 0x0F


@pytest.mark.parametrize("length", [-1, 16])
def test_fctrl_rejects_invalid_fopts_length(length):
    with pytest.raises(ValueError):
        encode_fctrl(fopts_len=length)


def test_name_helpers_cover_known_and_unknown_values():
    assert mtype_name(MType.DATA_DOWN) == "DATA_DOWN"
    assert cid_name(CID.APC_REQ) == "APC_REQ"
    assert mtype_name(99) == "UNKNOWN"
    assert cid_name(None) == "UNKNOWN"

