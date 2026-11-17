"""Characterization tests record current behavior, not an ideal specification."""

from types import SimpleNamespace

import pytest

from loraphymac.hardware.hardware import SX1276, SX1302, apply_hw_ptx_bounds, validate_ptx
from loraphymac.mac.frame_defs import decode_mhdr, encode_mhdr, frame_size_bytes
from loraphymac.phy.channel_plan import AS923_2
from loraphymac.runtime.apc_commands import decode_apc_ans, decode_apc_req
from loraphymac.runtime.apc_power import dbm_to_tx_power_index
from tests.helpers.packet_factory import make_packet


pytestmark = pytest.mark.characterization


def test_packet_copy_currently_shares_nested_apc_info_reference():
    source = make_packet(apc_info={"nested": {"index": 1}})
    copied = source.copy()
    assert copied.apc_info is source.apc_info
    copied.apc_info["nested"]["index"] = 2
    assert source.apc_info["nested"]["index"] == 2


def test_packet_at_last_hop_currently_uses_itself_as_next_hop():
    packet = make_packet(hop_index=1, current_tx="R1", next_hop="GW")
    packet.advance_hop()
    assert packet.current_tx == "GW"
    assert packet.next_hop == "GW"


def test_fopts_limit_is_strict_but_numeric_strings_are_accepted():
    assert frame_size_bytes("50", "15") == 77
    with pytest.raises(ValueError):
        frame_size_bytes(0, 16)
    with pytest.raises(ValueError):
        make_packet(fopts=bytes(16))


def test_dbm_mapping_currently_clips_out_of_range_values():
    entity = SimpleNamespace(id="N1", hw=SX1276())
    apply_hw_ptx_bounds(entity)
    assert dbm_to_tx_power_index(entity, 1000) == 0
    assert dbm_to_tx_power_index(entity, -1000) == 7


def test_validate_ptx_currently_truncates_float_toward_zero():
    entity = SimpleNamespace(id="N1", hw=SX1276())
    assert validate_ptx(entity, 19.9) == 19


def test_hardware_bandwidth_attributes_currently_use_different_units_and_names():
    assert SX1276.supported_bw == [125, 250, 500]
    assert SX1302.supported_bw_hz == [125_000, 250_000, 500_000]
    assert not hasattr(SX1276, "supported_bw_hz")
    assert not hasattr(SX1302, "supported_bw")


def test_unknown_channel_mode_currently_falls_back_to_loraphy_rules():
    plan = AS923_2("not-a-real-mode")
    assert plan.mode == "not-a-real-mode"
    assert plan.dr_table is None
    assert plan.duty_cycle == 1.0


def test_duplicate_channel_id_currently_returns_first_match():
    plan = AS923_2()
    plan.channels = [{"id": 1, "freq": 900.0}, {"id": 1, "freq": 901.0}]
    assert plan.get_freq(1) == 900.0


def test_decoders_currently_ignore_trailing_command_bytes():
    req = decode_apc_req(b"\x01\x12\xff\x01\xaa")
    ans = decode_apc_ans(b"\x02\x1f\xaa")
    assert req["dr"] == 1
    assert req["tx_power_index"] == 2
    assert ans["status"] == 0x1F


def test_mhdr_encoder_currently_masks_out_of_range_fields():
    assert decode_mhdr(encode_mhdr(mtype=9, rfu=9, version=5)) == {
        "mtype": 1,
        "rfu": 1,
        "version": 1,
    }
