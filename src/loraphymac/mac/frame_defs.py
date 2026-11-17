# ============================================================
# mac/frame_defs.py
# Physical DATA_DOWN / DATA_DOWN_FWD + 
# Relay APC RX cleanup, APC TX index, and timing validation
#
# ============================================================
from enum import IntEnum


class MType(IntEnum):
    HELLO = 0x00
    DATA_UP = 0x01        # Uplink Node -> Relay/GW
    DATA_UP_FWD = 0x02    # Uplink Relay -> Relay/GW
    DATA_DOWN = 0x03      # Downlink GW/Relay -> Node
    DATA_DOWN_FWD = 0x04  # Downlink GW/Relay -> Relay
    ACK = 0x05
    RFU_6 = 0x06
    RFU_7 = 0x07


class CID(IntEnum):
    NO_OP = 0x00
    APC_REQ = 0x01
    APC_ANS = 0x02
    LINK_REPORT = 0x03
    BCAST_SCOPE = 0x04
    ACK_STAT = 0x05
    ERR_CODE = 0x06
    ROUTE_HINT = 0x07
    WANG_SNR_REPORT = 0x08
    WANG_POWER_COMMAND = 0x09


class FPort(IntEnum):
    MAC_CONTROL = 0x00
    SENSOR_DATA = 0x01
    ROUTING_DEBUG = 0x02
    NETWORK_MGMT = 0x03
    DUTY_CYCLE_REPORT = 0x04



# ============================================================
# LoRaPhyMAC frame size
# ============================================================

# Fixed LoRaPhyMAC frame fields:
# MHDR    : 1 byte
# SrcID   : 1 byte
# DstID   : 1 byte
# NextHop : 1 byte
# HopCtrl : 1 byte
# FCtrl   : 1 byte
# FCnt    : 1 bytes
# FPort   : 1 byte
# MIC     : 4 bytes
#
# Total fixed overhead = 12 bytes.
FRAME_FIXED_OVERHEAD_BYTES = 12


def frame_size_bytes(
    app_payload_size: int = 0,
    fopts_len: int = 0,
) -> int:
    """
    Return total LoRaPhyMAC frame size used as LoRa PHY payload.

    Parameters
    ----------
    app_payload_size : int
        Application payload size in bytes.

    fopts_len : int
        FOpts length in bytes.

    Returns
    -------
    int
        Total LoRaPhyMAC frame size in bytes.

    Examples
    --------
    DATA_UP:
        12 fixed + 50 application + 0 FOpts = 62 bytes

    DATA_DOWN APC:
        12 fixed + 0 application + 4 FOpts = 16 bytes
    """
    app_payload_size = int(app_payload_size)
    fopts_len = int(fopts_len)

    if app_payload_size < 0:
        raise ValueError(
            "app_payload_size must be >= 0"
        )

    if fopts_len < 0 or fopts_len > 15:
        raise ValueError(
            "fopts_len must be in range 0..15"
        )

    return (
        FRAME_FIXED_OVERHEAD_BYTES
        + app_payload_size
        + fopts_len
    )


def encode_mhdr(mtype: int, version: int = 0, rfu: int = 0) -> int:
    return (
        ((int(mtype) & 0x07) << 5)
        | ((int(rfu) & 0x07) << 2)
        | (int(version) & 0x03)
    )


def decode_mhdr(mhdr: int) -> dict:
    return {
        "mtype": (int(mhdr) >> 5) & 0x07,
        "rfu": (int(mhdr) >> 2) & 0x07,
        "version": int(mhdr) & 0x03,
    }


def encode_hop_ctrl(hop_index: int, hop_limit: int) -> int:
    return ((int(hop_index) & 0x0F) << 4) | (int(hop_limit) & 0x0F)


def decode_hop_ctrl(hop_ctrl: int) -> dict:
    return {
        "hop_index": (int(hop_ctrl) >> 4) & 0x0F,
        "hop_limit": int(hop_ctrl) & 0x0F,
    }


def encode_fctrl(
    apc: bool = False,
    ack_req: bool = False,
    bcast: bool = False,
    pending: bool = False,
    fopts_len: int = 0,
) -> int:
    fopts_len = int(fopts_len)

    if fopts_len < 0 or fopts_len > 15:
        raise ValueError("fopts_len must be in range 0..15")

    return (
        (int(apc) << 7)
        | (int(ack_req) << 6)
        | (int(bcast) << 5)
        | (int(pending) << 4)
        | (fopts_len & 0x0F)
    )


def decode_fctrl(fctrl: int) -> dict:
    return {
        "apc": bool((int(fctrl) >> 7) & 0x01),
        "ack_req": bool((int(fctrl) >> 6) & 0x01),
        "bcast": bool((int(fctrl) >> 5) & 0x01),
        "pending": bool((int(fctrl) >> 4) & 0x01),
        "fopts_len": int(fctrl) & 0x0F,
    }


def mtype_name(mtype: int) -> str:
    try:
        return MType(int(mtype)).name
    except Exception:
        return "UNKNOWN"


def cid_name(cid: int) -> str:
    try:
        return CID(int(cid)).name
    except Exception:
        return "UNKNOWN"
