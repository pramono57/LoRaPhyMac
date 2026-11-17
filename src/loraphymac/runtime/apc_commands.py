# ============================================================
# runtime/apc_commands.py
# Physical DATA_DOWN / DATA_DOWN_FWD + 
# Relay APC RX cleanup, APC TX index, and timing validation
#
# LH-APC Command Encoder / Decoder
# Compatible with:
# - DATA_DOWN       : GW/Relay -> Node
# - DATA_DOWN_FWD   : GW/Relay -> Relay
# ============================================================

from loraphymac.mac.frame_defs import CID


# ============================================================
# 1. APC_REQ FIELD ENCODING
# ============================================================
def encode_dr_txpower(dr: int, tx_power_index: int) -> int:
    """
    Encode DR and TxPowerIndex into one byte.

    Format:
        bit 7..4 : DR
        bit 3..0 : TxPowerIndex

    Notes:
        - DR follows LoRaPhyMAC / LoRaWAN-style data rate index.
        - TxPowerIndex follows APC power mapping:
          index 0 = highest PTx, larger index = lower PTx.
    """
    dr = int(dr)
    tx_power_index = int(tx_power_index)

    if dr < 0 or dr > 15:
        raise ValueError(f"dr must be in range 0..15, got {dr}")

    if tx_power_index < 0 or tx_power_index > 15:
        raise ValueError(
            f"tx_power_index must be in range 0..15, got {tx_power_index}"
        )

    return ((dr & 0x0F) << 4) | (tx_power_index & 0x0F)


def decode_dr_txpower(value: int) -> dict:
    """
    Decode one byte into DR and TxPowerIndex.
    """
    value = int(value) & 0xFF

    return {
        "dr": (value >> 4) & 0x0F,
        "tx_power_index": value & 0x0F,
    }


# ============================================================
# 2. APC_REQ
# ============================================================
def encode_apc_req(
    dr: int,
    tx_power_index: int,
    ch_mask: int = 0xFF,
    nb_trans: int = 0x01,
) -> bytes:
    """
    Encode APC_REQ command.

    Format, 4 bytes:
        Byte 0 : CID.APC_REQ
        Byte 1 : DR_TXPower
                 bit 7..4 = DR
                 bit 3..0 = TxPowerIndex
        Byte 2 : ChMask
        Byte 3 : NbTrans

    Returns:
        bytes
    """
    ch_mask = int(ch_mask)
    nb_trans = int(nb_trans)

    if ch_mask < 0 or ch_mask > 0xFF:
        raise ValueError(f"ch_mask must be in range 0..255, got {ch_mask}")

    if nb_trans < 0 or nb_trans > 15:
        raise ValueError(f"nb_trans must be in range 0..15, got {nb_trans}")

    return bytes([
        int(CID.APC_REQ),
        encode_dr_txpower(dr, tx_power_index),
        ch_mask & 0xFF,
        nb_trans & 0x0F,
    ])


def decode_apc_req(data: bytes) -> dict:
    """
    Decode APC_REQ command.

    Expected format, 4 bytes:
        Byte 0 : CID.APC_REQ
        Byte 1 : DR_TXPower
        Byte 2 : ChMask
        Byte 3 : NbTrans
    """
    if data is None:
        raise ValueError("APC_REQ data is None")

    data = bytes(data)

    if len(data) < 4:
        raise ValueError(f"APC_REQ requires 4 bytes, got {len(data)}")

    cid = int(data[0])

    if cid != int(CID.APC_REQ):
        raise ValueError(
            f"Invalid APC_REQ CID: expected {int(CID.APC_REQ)}, got {cid}"
        )

    dr_tx = decode_dr_txpower(data[1])

    return {
        "cid": cid,
        "cid_name": CID(cid).name,
        "dr": dr_tx["dr"],
        "tx_power_index": dr_tx["tx_power_index"],
        "ch_mask": int(data[2]) & 0xFF,
        "nb_trans": int(data[3]) & 0x0F,
    }


# ============================================================
# 3. APC_ANS STATUS BIT
# ============================================================
APC_STATUS_DR_OK = 1 << 0
APC_STATUS_PTX_OK = 1 << 1
APC_STATUS_CH_OK = 1 << 2
APC_STATUS_NBTRANS_OK = 1 << 3
APC_STATUS_APPLIED = 1 << 4
APC_STATUS_UNSUPPORTED = 1 << 5
APC_STATUS_BUSY = 1 << 6
APC_STATUS_RFU = 1 << 7


def make_apc_status(
    dr_ok: bool = True,
    ptx_ok: bool = True,
    ch_ok: bool = True,
    nbtrans_ok: bool = True,
    applied: bool = True,
    unsupported: bool = False,
    busy: bool = False,
) -> int:
    """
    Build APC_ANS status byte.
    """
    status = 0

    if dr_ok:
        status |= APC_STATUS_DR_OK

    if ptx_ok:
        status |= APC_STATUS_PTX_OK

    if ch_ok:
        status |= APC_STATUS_CH_OK

    if nbtrans_ok:
        status |= APC_STATUS_NBTRANS_OK

    if applied:
        status |= APC_STATUS_APPLIED

    if unsupported:
        status |= APC_STATUS_UNSUPPORTED

    if busy:
        status |= APC_STATUS_BUSY

    return status & 0xFF


# ============================================================
# 4. APC_ANS
# ============================================================
def encode_apc_ans(status: int) -> bytes:
    """
    Encode APC_ANS command.

    Format, 2 bytes:
        Byte 0 : CID.APC_ANS
        Byte 1 : Status bitmap
    """
    return bytes([
        int(CID.APC_ANS),
        int(status) & 0xFF,
    ])


def decode_apc_ans(data: bytes) -> dict:
    """
    Decode APC_ANS command.

    Expected format, 2 bytes:
        Byte 0 : CID.APC_ANS
        Byte 1 : Status bitmap
    """
    if data is None:
        raise ValueError("APC_ANS data is None")

    data = bytes(data)

    if len(data) < 2:
        raise ValueError(f"APC_ANS requires 2 bytes, got {len(data)}")

    cid = int(data[0])

    if cid != int(CID.APC_ANS):
        raise ValueError(
            f"Invalid APC_ANS CID: expected {int(CID.APC_ANS)}, got {cid}"
        )

    status = int(data[1]) & 0xFF

    return {
        "cid": cid,
        "cid_name": CID(cid).name,
        "status": status,
        "dr_ok": bool(status & APC_STATUS_DR_OK),
        "ptx_ok": bool(status & APC_STATUS_PTX_OK),
        "ch_ok": bool(status & APC_STATUS_CH_OK),
        "nbtrans_ok": bool(status & APC_STATUS_NBTRANS_OK),
        "applied": bool(status & APC_STATUS_APPLIED),
        "unsupported": bool(status & APC_STATUS_UNSUPPORTED),
        "busy": bool(status & APC_STATUS_BUSY),
    }


# ============================================================
# 5. HELPER
# ============================================================
def is_apc_req(data: bytes) -> bool:
    """
    Return True if data starts with CID.APC_REQ.
    """
    try:
        return len(data) > 0 and int(data[0]) == int(CID.APC_REQ)
    except Exception:
        return False


def is_apc_ans(data: bytes) -> bool:
    """
    Return True if data starts with CID.APC_ANS.
    """
    try:
        return len(data) > 0 and int(data[0]) == int(CID.APC_ANS)
    except Exception:
        return False
